#!/usr/bin/env python3
"""Native structured-auto tool-routing acceptance fixture.

This harness exercises the product's ordinary OpenAI tool-call response contract.
It adds only ``tool_routing='structured'`` to each existing agent request and does
not reinterpret model response content or streaming events.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


SYSTEM_INSTRUCTIONS = (
    'You are a tool-using assistant. Use declared functions when external data is '
    'needed, follow their argument schemas, and wait for tool results before answering. '
    'When no external data is needed, answer directly. Only the final answer should '
    'be a JSON object without Markdown.'
)

FULL_SCENARIOS = (
    'smoke', 'streamed_read', 'two_step', 'three_step',
    'stale_path_recovery', 'changed_followup',
)


def load_agent_harness():
    harness_path = Path(__file__).with_name('test_hip_oga_agent_api.py')
    spec = importlib.util.spec_from_file_location('hip_oga_agent_api_for_structured_auto', harness_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import agent harness: {harness_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_auto_harness(agent_module):
    """Create a subclass that changes only the outgoing request body."""

    class StructuredAutoHarness(agent_module.AgentHarness):
        def _send_stdlib(self, encoded_body, stream):
            original_body_utf8 = encoded_body.decode('utf-8')
            body = json.loads(original_body_utf8)
            original_body = json.loads(json.dumps(body, ensure_ascii=False))
            body['tool_routing'] = 'structured'
            if body.get('messages') and body['messages'][0].get('role') == 'system':
                body['messages'][0]['content'] = SYSTEM_INSTRUCTIONS
            actual_body = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

            record = self.records[-1]
            record['request_before_routing'] = original_body
            record['request_before_routing_body_utf8'] = original_body_utf8
            record['request'] = body
            record['request_body_utf8'] = actual_body.decode('utf-8')
            if record.get('history_before') and record['history_before'][0].get('role') == 'system':
                record['history_before'][0]['content'] = SYSTEM_INSTRUCTIONS

            status, headers, raw_response = super()._send_stdlib(actual_body, stream)
            record['response_bytes_utf8'] = raw_response.decode('utf-8', 'replace')
            return status, headers, raw_response

        def run_no_call_probe(self, name, prompt, answer_pattern):
            start = len(self.records)
            scenario = {
                'name': name,
                'status': 'FAIL',
                'probe_kind': 'no_call_negative_control',
                'prompt': prompt,
                'expected_tool_calls': 0,
                'tool_calls': [],
                'history': [],
                'outcomes': [],
                'failures': [],
            }
            self.scenarios.append(scenario)
            history = [
                {'role': 'system', 'content': SYSTEM_INSTRUCTIONS},
                {'role': 'user', 'content': prompt},
            ]
            scenario['history'].append({'before_turn': 1, 'messages': json.loads(json.dumps(history))})
            try:
                message, finish, calls, _usage = self._http_turn(name, 1, history, stream=False)
                scenario['http_status'] = self.records[-1].get('http_status')
                scenario['finish_reason'] = finish
                scenario['tool_calls'] = json.loads(json.dumps(calls))
                content = message.get('content')
                if scenario['http_status'] != 200:
                    raise agent_module.AgentLoopFailure(
                        f'no-call control expected HTTP 200, got {scenario["http_status"]!r}')
                if finish != 'stop':
                    raise agent_module.AgentLoopFailure(
                        f'no-call control expected finish_reason stop, got {finish!r}')
                if calls:
                    raise agent_module.AgentLoopFailure(
                        f'no-call control expected zero tool calls, observed {len(calls)}')
                if not isinstance(content, str) or not content.strip():
                    raise agent_module.AgentLoopFailure('no-call answer must be nonempty text')
                if answer_pattern.search(content) is None:
                    raise agent_module.AgentLoopFailure(
                        f'no-call answer did not match {answer_pattern.pattern!r}: {content!r}')
                scenario['actual_answer'] = content
                scenario['history'].append({'after_turn': 1, 'messages': json.loads(json.dumps(history + [message]))})
                scenario['status'] = 'PASS'
            except Exception as error:
                message_text = f'{name}: {type(error).__name__}: {error}'
                scenario['failure'] = message_text
                scenario['failures'].append(message_text)
                self.failures.append(message_text)
            finally:
                scenario['request_records'] = self.records[start:]
            return scenario

        def run_auto_suite(self):
            scenario_names = ['smoke'] if self.args.phase == 'smoke' else list(FULL_SCENARIOS)
            for name in scenario_names:
                result = self.run_scenario(name, streamed=(name == 'streamed_read'))
                # The request record is authoritative; keep the result history in
                # sync with the system message that was actually sent on the wire.
                for snapshot in result.get('history', []):
                    messages = snapshot.get('messages', [])
                    if messages and messages[0].get('role') == 'system':
                        messages[0]['content'] = SYSTEM_INSTRUCTIONS

            self.run_no_call_probe(
                'greeting_no_call', 'Hello! Please greet me briefly.',
                re.compile(r'\b(?:hello|hi)\b', re.IGNORECASE))
            self.run_no_call_probe(
                'week_no_call', 'How many days are there in a week? Answer briefly.',
                re.compile(r'\b(?:seven|7)\b', re.IGNORECASE))

            passed = not self.failures and all(item.get('status') == 'PASS' for item in self.scenarios)
            full_qualification = passed and self.args.phase == 'full'
            return {
                'status': 'PASS' if passed else 'FAIL',
                'model': self.args.model,
                'base_url': self.args.base_url,
                'phase': self.args.phase,
                'cache': self.args.cache,
                'client': 'stdlib',
                'tool_policy': 'auto',
                'tool_routing': 'structured',
                'nativeAutoQualification': bool(full_qualification),
                'qualification_scope': (
                    'full six-scenario plus two no-call suite' if full_qualification
                    else 'not qualified by this run'),
                'source_sha256': {
                    'path': str(Path(__file__).resolve()),
                    'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                },
                'limits': {
                    'max_model_turns_per_scenario': agent_module.MAX_TURNS,
                    'max_tool_executions_per_scenario': agent_module.MAX_TOOL_CALLS,
                    'http_timeout_seconds': agent_module.HTTP_TIMEOUT_SECONDS,
                    'max_tokens': agent_module.MAX_TOKENS,
                },
                'records': self.records,
                'scenarios': self.scenarios,
                'failures': self.failures,
            }

    return StructuredAutoHarness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', choices=('on', 'off'), required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    args = parser.parse_args(argv)
    args.client = 'stdlib'
    args.tool_policy = 'auto'
    args.scenario = None

    try:
        agent_module = load_agent_harness()
        result = make_auto_harness(agent_module)(args).run_auto_suite()
    except Exception as error:
        result = {
            'status': 'FAIL',
            'model': args.model,
            'base_url': args.base_url,
            'phase': args.phase,
            'cache': args.cache,
            'client': 'stdlib',
            'tool_policy': 'auto',
            'tool_routing': 'structured',
            'nativeAutoQualification': False,
            'source_sha256': {
                'path': str(Path(__file__).resolve()),
                'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            'records': [],
            'scenarios': [],
            'failures': [f'{type(error).__name__}: {error}'],
        }

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / 'result.json'
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    passed = sum(item.get('status') == 'PASS' for item in result.get('scenarios', []))
    total = len(result.get('scenarios', []))
    print(f"{result['status']}: structured auto {passed}/{total}; "
          f"nativeAutoQualification={result.get('nativeAutoQualification', False)}; {output_file}")
    for failure in result.get('failures', []):
        print(f'FAIL: {failure}')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
