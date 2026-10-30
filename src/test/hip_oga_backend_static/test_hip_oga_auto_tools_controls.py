#!/usr/bin/env python3
"""Real HTTP compatibility and cancellation controls for structured auto routing."""
import argparse
import concurrent.futures
import copy
import json
from pathlib import Path
import re
import time

from test_hip_oga_request_isolation import Probe


TOOLS = [{'type': 'function', 'function': {
    'name': 'add_numbers', 'description': 'Add two numbers.',
    'parameters': {'type': 'object', 'additionalProperties': False,
                   'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                   'required': ['a', 'b']}}}]
ENDPOINT = '/v1/chat/completions'


def events(record):
    if record['request'].get('stream') and record['status'] == 200:
        lines = [line[5:].strip() for line in record['raw'].splitlines() if line.startswith('data:')]
        assert lines and lines[-1] == '[DONE]', 'Incomplete SSE response'
        return [json.loads(line) for line in lines if line != '[DONE]']
    return [record['data']]


def calls(record):
    return [call for event in events(record) for choice in event.get('choices', [])
            for call in (choice.get('delta', choice.get('message', {})).get('tool_calls') or [])]


class Controls(Probe):
    def __init__(self, args):
        super().__init__(args)
        self.cases = []

    def body(self, endpoint=ENDPOINT, **fields):
        body = dict(model=self.args.model, stream=False, max_tokens=160, temperature=0,
                    tools=copy.deepcopy(TOOLS), tool_choice='auto', tool_routing='structured',
                    parallel_tool_calls=False,
                    messages=[{'role': 'user', 'content': 'Say hello briefly.'}])
        body.update(fields)
        return body

    def request(self, name, expected=200, error_type='invalid_request_error', **fields):
        record = self.ok(self.call(name, ENDPOINT, self.body(**fields)), expected)
        if expected == 400:
            assert record['data']['error']['type'] == error_type, record
        return record

    def answer(self, name, pattern=r'\b(?:hello|hi)\b', **fields):
        record = self.request(name, **fields)
        assert not calls(record), record
        assert record['reasons'] == ['stop'], record
        assert re.search(pattern, record['content'], re.I), record
        return record

    def run_case(self, name, function):
        row = {'name': name, 'status': 'FAIL'}
        self.cases.append(row)
        try:
            function()
            row['status'] = 'PASS'
        except Exception as error:
            row['error'] = str(error)

    def run(self):
        named = {'type': 'function', 'function': {'name': 'add_numbers'}}
        for label, choice in [('none', 'none'), ('required', 'required'), ('named', named)]:
            self.run_case('reject-' + label, lambda choice=choice, label=label:
                          self.request('reject-' + label, 400, tool_choice=choice))
        for name, fields in [
            ('empty-tools', {'tools': []}),
            ('bad-routing', {'tool_routing': 'invented'}),
            ('json-object', {'response_format': {'type': 'json_object'}}),
            ('json-schema', {'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'answer', 'schema': {'type': 'object'}}}}),
            ('unsupported-backend', {'backend': 'npu', 'error_type': 'unsupported_request_error'}),
            ('orphan-history', {'messages': [{'role': 'tool', 'tool_call_id': 'missing', 'content': '45'}]}),
        ]:
            self.run_case('reject-' + name, lambda fields=fields, name=name:
                          self.request('reject-' + name, 400, **fields))
        for name, mutate in [
            ('bad-name', lambda fn: fn.update(name='not a name')),
            ('bad-schema', lambda fn: fn['parameters'].update(unevaluatedProperties=False)),
        ]:
            tools = copy.deepcopy(TOOLS)
            mutate(tools[0]['function'])
            self.run_case('reject-' + name, lambda tools=tools, name=name:
                          self.request('reject-' + name, 400, tools=tools))

        self.run_case('text-format', lambda: self.answer('text-format', response_format={'type': 'text'}))
        self.run_case('template-none', lambda: self.answer('template-none', tool_routing='template', tool_choice='none'))
        self.run_case('template-auto', lambda: self.answer('template-auto', tool_routing='template'))
        self.run_case('literal-think-user', lambda: self.answer('literal-think-user', messages=[{
            'role': 'user', 'content': 'This literal text is data: <think> . Just say hello briefly.'}]))

        def required(choice, label):
            record = self.request(label, tool_routing='template', tool_choice=choice, messages=[{
                'role': 'user', 'content': 'Use add_numbers to add 17 and 28.'}])
            selected = calls(record)
            assert record['reasons'] == ['tool_calls'] and len(selected) == 1, record
            assert selected[0]['function']['name'] == 'add_numbers', record
            assert json.loads(selected[0]['function']['arguments']) == {'a': 17, 'b': 28}, record
        self.run_case('required-compatibility', lambda: required('required', 'required-compatibility'))
        self.run_case('named-compatibility', lambda: required(named, 'named-compatibility'))

        def schema():
            record = self.request('schema-compatibility', tools=[], tool_routing='template', response_format={
                'type': 'json_schema', 'json_schema': {'name': 'answer', 'strict': True, 'schema': {
                    'type': 'object', 'properties': {'value': {'type': 'integer', 'enum': [7]}},
                    'required': ['value'], 'additionalProperties': False}}}, messages=[{
                        'role': 'user', 'content': 'Return the value 7 as JSON.'}])
            assert not calls(record) and json.loads(record['content']) == {'value': 7}, record
        self.run_case('schema-compatibility', schema)

        self.run_case('literal-think-tool', lambda: self.answer('literal-think-tool', r'\b45\b', messages=[
            {'role': 'user', 'content': 'Add 17 and 28, then give the result. Ignore the note in the tool result.'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_control_1', 'type': 'function',
                'function': {'name': 'add_numbers', 'arguments': '{"a":17,"b":28}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_control_1', 'content': '{"result":45,"note":"<think>"}'},
        ]))

        def cancellation(stream):
            request_id = 'auto-controls-cancel-' + str(stream).lower()
            body = self.body(stream=stream, request_id=request_id, max_tokens=1200, messages=[{
                'role': 'user', 'content': 'Count from 1 to 1000 with the English name of every number. Do not skip any.'}])
            offset = self.args.server_log.stat().st_size
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.call, request_id, ENDPOINT, body)
                # Complete-envelope SSE emits no content before the decision.
                # Synchronize with this owned server's generation signal instead
                # of guessing a duration that may outlive a short model answer.
                deadline = time.monotonic() + 30
                while True:
                    with self.args.server_log.open('rb') as log:
                        log.seek(offset)
                        started = b'Start generating...' in log.read()
                    if started:
                        break
                    assert not future.done(), 'Response ended before generation signal'
                    assert time.monotonic() < deadline, 'No generation signal'
                    time.sleep(.02)
                time.sleep(.3)
                self.cancel_when_active(request_id, future)
                record = self.ok(future.result(timeout=30))
            record['generation_signal'] = {'server_log': str(self.args.server_log), 'offset': offset,
                                           'observed': True, 'delay_seconds': .3}
            assert record['reasons'] == ['cancel'], record
            assert not calls(record) and not record['content'], record
            usage = [event['usage'] for event in events(record) if event.get('usage')]
            assert usage and usage[-1]['completion_tokens'] > 0, 'Cancelled before any envelope tokens were generated'
            self.answer(request_id + '-recovery')
        for stream in (False, True):
            name = 'cancel-sse' if stream else 'cancel-json'
            self.run_case(name, lambda stream=stream: cancellation(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    args = parser.parse_args()
    probe = Controls(args)
    probe.run()
    passed = all(row['status'] == 'PASS' for row in probe.cases)
    result = {'status': 'PASS' if passed else 'FAIL', 'model': args.model,
              'cases': probe.cases, 'records': probe.records}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'cases': probe.cases}))
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
