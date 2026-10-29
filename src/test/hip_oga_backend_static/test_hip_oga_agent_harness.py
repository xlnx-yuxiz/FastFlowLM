#!/usr/bin/env python3
"""Offline contract tests for the bounded HIP OGA agent harness."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


FIXTURE = Path(__file__).with_name('test_hip_oga_agent_api.py')
SPEC = importlib.util.spec_from_file_location('hip_oga_agent_fixture', FIXTURE)
AGENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AGENT
SPEC.loader.exec_module(AGENT)


def _usage():
    return {
        'prompt_tokens': 48,
        'completion_tokens': 3,
        'total_tokens': 51,
        'prompt_tokens_details': {'cached_tokens': 0},
    }


def _completion(message, finish_reason):
    """A hand-written OpenAI chat.completion HTTP receipt."""
    return json.dumps({
        'id': 'chatcmpl-offline-1',
        'object': 'chat.completion',
        'created': 1,
        'model': 'offline-fixture-model',
        'choices': [{
            'index': 0,
            'message': message,
            'finish_reason': finish_reason,
        }],
        'usage': _usage(),
    }, separators=(',', ':')).encode('utf-8')


def _tool_receipt(call_id, name='read_record', arguments='{"path":"single.json"}'):
    return _completion({
        'role': 'assistant',
        'content': None,
        'tool_calls': [{
            'id': call_id,
            'type': 'function',
            'function': {'name': name, 'arguments': arguments},
        }],
    }, 'tool_calls')


def _harness(cache='off'):
    return AGENT.AgentHarness(SimpleNamespace(
        base_url='http://127.0.0.1:18769/v1',
        model='offline-fixture-model',
        client='stdlib',
        cache=cache,
        phase='smoke',
    ))


class SseReceiptTests(unittest.TestCase):
    def test_choice_events_after_finish_fail_before_any_tool_execution(self):
        partial = {'choices': [{'index': 0, 'delta': {'tool_calls': [{
            'index': 0, 'id': 'late-call', 'type': 'function',
            'function': {'name': 'read_record', 'arguments': '{"path":'},
        }]}, 'finish_reason': None}]}
        finish = {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}], 'usage': _usage()}
        late = {'choices': [{'index': 0, 'delta': {'tool_calls': [{
            'index': 0, 'function': {'arguments': '"single.json"}'},
        }]}, 'finish_reason': None}]}
        for events in ([partial, finish, late], [partial, late, finish, finish]):
            raw = (''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n').encode()
            harness = _harness()
            with patch.object(harness, '_send_stdlib', return_value=(200, [], raw)):
                result = harness.run_scenario('streamed_read', streamed=True)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['executed_calls'], [])
            self.assertEqual(result['turn_count'], 1)

    def test_fragmented_tool_call_is_assembled_from_http_sse_receipt(self):
        raw = b'\n'.join([
            b'data: {"id":"chunk-a","object":"chat.completion.chunk","model":"offline-fixture-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call-91","type":"function","function":{"name":"read_","arguments":"{\\"pa"}}]},"finish_reason":null}]}',
            b'data: {"id":"chunk-a","object":"chat.completion.chunk","model":"offline-fixture-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"record","arguments":"th\\":\\"single.json\\"}"}}]},"finish_reason":null}]}',
            b'data: {"id":"chunk-a","object":"chat.completion.chunk","model":"offline-fixture-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            b'data: {"id":"chunk-a","object":"chat.completion.chunk","model":"offline-fixture-model","choices":[],"usage":{"prompt_tokens":48,"completion_tokens":1,"total_tokens":49,"prompt_tokens_details":{"cached_tokens":0}}}',
            b'data: [DONE]',
        ])
        harness = _harness()

        with patch.object(harness, '_send_stdlib', return_value=(200, [], raw)):
            message, finish, calls, usage = harness._http_turn(
                'manual_sse', 1, [{'role': 'user', 'content': 'Read single.json.'}], True)

        self.assertEqual(finish, 'tool_calls')
        self.assertIsNone(message['content'])
        self.assertEqual(calls, [{
            'id': 'call-91',
            'type': 'function',
            'function': {'name': 'read_record', 'arguments': '{"path":"single.json"}'},
        }])
        self.assertEqual(usage['prompt_tokens_details']['cached_tokens'], 0)
        self.assertIn('data: [DONE]', harness.records[0]['raw_response'])

    def test_invalid_and_incomplete_sse_receipts_fail_before_tool_execution(self):
        complete_tool_event = {
            'choices': [{
                'index': 0,
                'delta': {'tool_calls': [{
                    'index': 0,
                    'id': 'partial',
                    'type': 'function',
                    'function': {'name': 'read_record', 'arguments': '{"path":"single.json"}'},
                }]},
                'finish_reason': 'tool_calls',
            }],
        }
        receipts = (
            ('invalid JSON event', b'data: {not-json}\ndata: [DONE]\n'),
            ('missing DONE', b'data: ' + json.dumps(complete_tool_event).encode('utf-8') + b'\n'),
        )
        for reason, raw in receipts:
            with self.subTest(reason=reason):
                harness = _harness()
                with patch.object(harness, '_send_stdlib', return_value=(200, [], raw)):
                    result = harness.run_scenario('streamed_read', streamed=True)
                self.assertEqual(result['status'], 'FAIL')
                self.assertEqual(result['executed_calls'], [])
                self.assertEqual(result['turn_count'], 1)
                self.assertTrue(result['request_records'][0].get('transport_or_protocol_error'))


class ToolExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'single.json').write_text('{"answer":"owned"}', encoding='utf-8')
        self.outside = self.root.parent / f'{self.root.name}-outside.json'
        self.outside.write_text('{"answer":"outside"}', encoding='utf-8')
        self.addCleanup(self.outside.unlink, missing_ok=True)
        self.executor = AGENT.BoundedToolExecutor(self.root, {'cedar': 'single.json'}, {'single.json'})

    def _call(self, call_id, name, arguments):
        return {
            'id': call_id,
            'type': 'function',
            'function': {'name': name, 'arguments': arguments},
        }

    def test_allowlisted_read_and_path_confinement(self):
        message, trace = self.executor.execute(
            self._call('read-1', 'read_record', '{"path":"single.json"}'))
        self.assertEqual(message['tool_call_id'], 'read-1')
        self.assertEqual(json.loads(message['content']), {'answer': 'owned'})
        self.assertEqual(trace['status'], 'ok')

        message, trace = self.executor.execute(
            self._call('read-2', 'read_record', json.dumps({'path': f'../{self.outside.name}'})))
        self.assertEqual(trace['status'], 'error')
        self.assertEqual(json.loads(message['content']), {'error': 'record not found'})

        message, trace = self.executor.execute(
            self._call('tool-3', 'run_shell', '{"command":"echo unsafe"}'))
        self.assertEqual(trace['status'], 'error')
        self.assertIn('unknown tool', json.loads(message['content'])['error'])

    def test_lookup_and_integer_addition_execute_declared_tools(self):
        message, trace = self.executor.execute(
            self._call('lookup-1', 'lookup_record', '{"name":"cedar"}'))
        self.assertEqual(json.loads(message['content']), {'path': 'single.json'})
        self.assertEqual(trace['status'], 'ok')

        message, trace = self.executor.execute(
            self._call('sum-1', 'add_numbers', '{"a":-5,"b":12}'))
        self.assertEqual(json.loads(message['content']), {'result': 7})
        self.assertEqual(trace['status'], 'ok')

    def test_malformed_json_and_wrong_argument_shapes_are_rejected_safely(self):
        with self.assertRaises(AGENT.AgentLoopFailure):
            self.executor.execute(self._call('bad-json', 'read_record', '{"path":'))
        self.assertEqual(self.executor.executed_count, 0)

        message, trace = self.executor.execute(
            self._call('wrong-shape', 'read_record', '{"filename":"single.json"}'))
        self.assertEqual(trace['status'], 'error')
        self.assertEqual(
            json.loads(message['content']),
            {'error': 'read_record expects exactly one string field named path'},
        )

        message, trace = self.executor.execute(
            self._call('wrong-type', 'add_numbers', '{"a":true,"b":4}'))
        self.assertEqual(trace['status'], 'error')
        self.assertEqual(
            json.loads(message['content']),
            {'error': 'add_numbers expects exactly two integer fields named a and b'},
        )

    def test_reused_call_ids_and_execution_bound_are_rejected(self):
        bounded = AGENT.BoundedToolExecutor(self.root, {}, {'single.json'}, max_calls=1)
        call = self._call('repeat-me', 'read_record', '{"path":"single.json"}')
        bounded.execute(call)
        with self.assertRaisesRegex(AGENT.AgentLoopFailure, 'limit exceeded'):
            bounded.execute(call)
        self.assertEqual(bounded.executed_count, 1)

        ids = AGENT.BoundedToolExecutor(self.root, {}, {'single.json'}, max_calls=2)
        ids.execute(call)
        with self.assertRaisesRegex(AGENT.AgentLoopFailure, 'reused tool-call ID'):
            ids.execute(call)
        self.assertEqual(ids.executed_count, 1)


class AgentLoopReceiptTests(unittest.TestCase):
    def test_cancelled_tool_stream_is_not_executed_or_replayed_into_retry(self):
        harness = _harness()
        harness.args.tool_policy = 'tool_only'
        replies = iter([_tool_receipt('read-1'),
            _tool_receipt('finish-2', 'finish_task', '{"answer":{"label":"hidden-label","code":1007}}')])
        sent = []
        def respond(encoded, stream):
            sent.append(json.loads(encoded))
            return 200, [], next(replies)
        with patch.object(AGENT, '_random_label', return_value='hidden-label'), \
             patch.object(AGENT.secrets, 'randbelow', return_value=7), \
             patch.object(harness, '_cancel_tool_stream', create=True,
                          return_value={'status': 'PASS', 'cancelled': True, 'partial_id': 'cancelled-partial'}), \
             patch.object(harness, '_send_stdlib', side_effect=respond):
            result = harness.run_scenario('cancel_then_read')
        self.assertEqual(result['status'], 'PASS', result.get('failure'))
        self.assertEqual(result['cancellation']['cancelled'], True)
        self.assertNotIn('cancelled-partial', json.dumps(sent))
        self.assertEqual(len(result['executed_calls']), 1)

    def test_required_policy_selects_tools_and_finishes_using_hidden_results(self):
        harness = _harness()
        harness.args.tool_policy = 'tool_only'
        replies = iter([
            _tool_receipt('read-1'),
            _tool_receipt('finish-2', 'finish_task', '{"answer":{"label":"hidden-label","code":1007}}'),
        ])
        sent = []
        def respond(encoded, stream):
            sent.append(json.loads(encoded))
            return 200, [], next(replies)
        with patch.object(AGENT, '_random_label', return_value='hidden-label'), \
             patch.object(AGENT.secrets, 'randbelow', return_value=7), \
             patch.object(harness, '_send_stdlib', side_effect=respond):
            result = harness.run_scenario('smoke')
        self.assertEqual(result['status'], 'PASS', result.get('failure'))
        self.assertEqual(result['outcomes'][0]['actual_answer'], {'label': 'hidden-label', 'code': 1007})
        self.assertEqual(len(result['executed_calls']), 1)
        self.assertEqual(sent[0]['tool_choice'], 'required')
        self.assertIn('finish_task', [tool['function']['name'] for tool in sent[0]['tools']])
        self.assertNotIn('hidden-label', json.dumps(sent[0]))
        self.assertEqual(sent[1]['messages'][-1]['tool_call_id'], 'read-1')
        self.assertEqual(result['history'][-1]['messages'][-1]['tool_call_id'], 'finish-2')

    def test_required_policy_rejects_an_early_finish_even_with_a_correct_guess(self):
        harness = _harness()
        harness.args.tool_policy = 'tool_only'
        reply = _tool_receipt('finish-1', 'finish_task', '{"answer":{"label":"hidden-label","code":1007}}')
        with patch.object(AGENT, '_random_label', return_value='hidden-label'), \
             patch.object(AGENT.secrets, 'randbelow', return_value=7), \
             patch.object(harness, '_send_stdlib', return_value=(200, [], reply)):
            result = harness.run_scenario('smoke')
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('expected 1 tool calls, observed 0', result['failure'])

    def test_wrong_final_answer_is_reported_as_a_harness_failure(self):
        harness = _harness()
        wrong_final = _completion({'role': 'assistant', 'content': '{}'}, 'stop')
        with patch.object(harness, '_send_stdlib', return_value=(200, [], wrong_final)):
            result = harness.run_scenario('smoke')
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('answer keys', result['failure'])
        self.assertEqual(result['turn_count'], 1)

    def test_repeating_tool_receipts_stop_at_the_scenario_turn_bound(self):
        harness = _harness()
        count = 0

        def repeat_tool(_encoded_body, _stream):
            nonlocal count
            count += 1
            return 200, [], _tool_receipt(f'repeated-{count}')

        with patch.object(harness, '_send_stdlib', side_effect=repeat_tool):
            result = harness.run_scenario('smoke')
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn(f'exceeded {AGENT.MAX_TURNS} model turns', result['failure'])
        self.assertEqual(result['turn_count'], AGENT.MAX_TURNS)
        self.assertEqual(count, AGENT.MAX_TURNS)
        self.assertEqual(len(result['executed_calls']), AGENT.MAX_TURNS)

    def test_hidden_expected_label_is_absent_from_the_outbound_request(self):
        harness = _harness()
        sent_bodies = []

        def wrong_answer(encoded_body, _stream):
            sent_bodies.append(json.loads(encoded_body))
            return 200, [], _completion({'role': 'assistant', 'content': '{}'}, 'stop')

        with patch.object(harness, '_send_stdlib', side_effect=wrong_answer):
            result = harness.run_scenario('smoke')

        hidden_label = result['outcomes'][0]['expected_answer']['label']
        self.assertTrue(hidden_label)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(len(sent_bodies), 1)
        serialized_request = json.dumps(sent_bodies[0], ensure_ascii=False)
        self.assertNotIn(hidden_label, serialized_request)
        self.assertNotIn('const', serialized_request)
        self.assertNotIn('enum', serialized_request)


if __name__ == '__main__':
    unittest.main()
