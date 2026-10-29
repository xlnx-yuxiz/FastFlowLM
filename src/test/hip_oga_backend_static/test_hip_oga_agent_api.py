#!/usr/bin/env python3
"""Bounded client-managed tool-loop qualification for OpenAI chat completions.

The server is externally owned. This fixture sends ordinary HTTP requests,
executes only three fixture-backed tools in a temporary directory, and writes
all requests, responses, histories, tool traces, outcomes, and failures to
``<output>/result.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import secrets
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


MAX_TURNS = 6
MAX_TOOL_CALLS = 6
HTTP_TIMEOUT_SECONDS = 120
MAX_TOKENS = 256


class AgentLoopFailure(RuntimeError):
    """A response or tool trace violated the agent-loop contract."""


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _json_object(text, where):
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise AgentLoopFailure(f'{where}: invalid JSON: {error}') from error
    if not isinstance(value, dict):
        raise AgentLoopFailure(f'{where}: expected a JSON object')
    return value


def assemble_sse(raw_response):
    """Assemble one complete OpenAI SSE completion or reject it atomically.

    No partially received or malformed tool call is returned to the executor.
    Tool-call fragments are joined by index, while repeated IDs and types must
    remain stable across chunks.
    """
    events = []
    done = False
    for line_number, line in enumerate(raw_response.splitlines(), 1):
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        if done:
            raise AgentLoopFailure(f'SSE line {line_number}: data arrived after [DONE]')
        if payload == '[DONE]':
            done = True
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            raise AgentLoopFailure(f'SSE line {line_number}: malformed event JSON: {error}') from error
        if not isinstance(event, dict):
            raise AgentLoopFailure(f'SSE line {line_number}: event must be a JSON object')
        events.append(event)

    if not done or not events:
        raise AgentLoopFailure('SSE response did not contain events followed by [DONE]')

    contents = []
    calls = {}
    finishes = []
    usage = None
    for event_number, event in enumerate(events, 1):
        if event.get('error'):
            raise AgentLoopFailure(f'SSE event {event_number}: server returned {event["error"]!r}')
        event_usage = event.get('usage')
        if isinstance(event_usage, dict):
            usage = event_usage
        choices = event.get('choices')
        if choices is None:
            continue
        if not isinstance(choices, list):
            raise AgentLoopFailure(f'SSE event {event_number}: choices must be an array')
        for choice in choices:
            if finishes:
                raise AgentLoopFailure(f'SSE event {event_number}: choice arrived after finish_reason')
            if not isinstance(choice, dict) or choice.get('index') != 0:
                raise AgentLoopFailure(f'SSE event {event_number}: expected one choice at index 0')
            delta = choice.get('delta') or {}
            if not isinstance(delta, dict):
                raise AgentLoopFailure(f'SSE event {event_number}: delta must be an object')
            piece = delta.get('content')
            if piece is not None:
                if not isinstance(piece, str):
                    raise AgentLoopFailure(f'SSE event {event_number}: content delta must be a string')
                contents.append(piece)
            for part in delta.get('tool_calls') or []:
                if not isinstance(part, dict):
                    raise AgentLoopFailure(f'SSE event {event_number}: tool-call delta must be an object')
                index = part.get('index')
                if type(index) is not int or index < 0:
                    raise AgentLoopFailure(f'SSE event {event_number}: tool-call delta has invalid index')
                call = calls.setdefault(index, {
                    'index': index, 'id': '', 'type': '',
                    'function': {'name': '', 'arguments': ''},
                })
                for key in ('id', 'type'):
                    value = part.get(key)
                    if value:
                        if not isinstance(value, str):
                            raise AgentLoopFailure(f'SSE event {event_number}: tool-call {key} must be a string')
                        if call[key] and call[key] != value:
                            raise AgentLoopFailure(f'SSE event {event_number}: tool-call {key} changed within stream')
                        call[key] = value
                function = part.get('function') or {}
                if not isinstance(function, dict):
                    raise AgentLoopFailure(f'SSE event {event_number}: tool-call function must be an object')
                for key in ('name', 'arguments'):
                    value = function.get(key)
                    if value is not None:
                        if not isinstance(value, str):
                            raise AgentLoopFailure(f'SSE event {event_number}: function {key} must be a string')
                        call['function'][key] += value
            finish = choice.get('finish_reason')
            if finish is not None:
                finishes.append(finish)

    if not finishes:
        raise AgentLoopFailure('SSE response did not include a finish_reason')
    if len(set(finishes)) != 1:
        raise AgentLoopFailure(f'SSE response had inconsistent finish reasons: {finishes!r}')
    ordered_calls = [calls[index] for index in sorted(calls)]
    ids = [call['id'] for call in ordered_calls]
    if any(not call['id'] or call['type'] != 'function' or not call['function']['name']
           for call in ordered_calls):
        raise AgentLoopFailure('SSE response ended with an incomplete tool call')
    if len(ids) != len(set(ids)):
        raise AgentLoopFailure('SSE response reused a tool-call ID')
    for call in ordered_calls:
        _json_object(call['function']['arguments'], 'streamed tool arguments')

    content = ''.join(contents)
    message = {'role': 'assistant', 'content': content if content or not ordered_calls else None}
    if ordered_calls:
        message['tool_calls'] = [
            {key: value for key, value in call.items() if key != 'index'}
            for call in ordered_calls
        ]
    return {
        'events': events,
        'message': message,
        'finish_reason': finishes[-1],
        'usage': usage,
        'sse_done': True,
    }


def tool_schemas():
    """Return broad, non-leaking JSON argument schemas for the fixed tools."""
    return [
        {'type': 'function', 'function': {
            'name': 'lookup_record',
            'description': 'Resolve a logical record name to a file path. Do not pass a .json filename here; known file paths go directly to read_record.',
            'parameters': {'type': 'object', 'required': ['name'], 'additionalProperties': False,
                           'properties': {'name': {'type': 'string'}}},
        }},
        {'type': 'function', 'function': {
            'name': 'read_record',
            'description': 'Read a known .json file path directly. Use this when the user provides a filename or lookup_record returns a path.',
            'parameters': {'type': 'object', 'required': ['path'], 'additionalProperties': False,
                           'properties': {'path': {'type': 'string'}}},
        }},
        {'type': 'function', 'function': {
            'name': 'add_numbers',
            'description': 'Add two integers.',
            'parameters': {'type': 'object', 'required': ['a', 'b'], 'additionalProperties': False,
                           'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}},
        }},
    ]


class BoundedToolExecutor:
    """Execute only declared tools against fixture files owned by this run."""

    def __init__(self, fixture_root, lookup, allowed_paths, max_calls=MAX_TOOL_CALLS):
        self.fixture_root = Path(fixture_root).resolve()
        self.lookup = dict(lookup)
        self.allowed_paths = set(allowed_paths)
        self.max_calls = max_calls
        self.executed_count = 0
        self.call_ids = set()
        self.trace = []

    def execute(self, call):
        if self.executed_count >= self.max_calls:
            raise AgentLoopFailure(f'tool execution limit exceeded ({self.max_calls})')
        if not isinstance(call, dict):
            raise AgentLoopFailure('assistant tool call must be an object')
        call_id = call.get('id')
        if not isinstance(call_id, str) or not call_id:
            raise AgentLoopFailure('assistant tool call is missing an ID')
        if call_id in self.call_ids:
            raise AgentLoopFailure(f'assistant reused tool-call ID {call_id!r}')
        if call.get('type') != 'function':
            raise AgentLoopFailure(f'assistant tool call has unsupported type {call.get("type")!r}')
        function = call.get('function')
        if not isinstance(function, dict) or not isinstance(function.get('name'), str):
            raise AgentLoopFailure('assistant tool call is missing its function name')
        name = function['name']
        arguments_raw = function.get('arguments')
        arguments = _json_object(arguments_raw, f'{name} arguments')
        trace = {
            'id': call_id,
            'type': call['type'],
            'name': name,
            'arguments_raw': arguments_raw,
            'arguments': _json_copy(arguments),
        }
        self.call_ids.add(call_id)
        self.executed_count += 1
        try:
            value = self._execute(name, arguments)
            trace['status'] = 'ok'
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
            value = {'error': str(error)}
            trace['status'] = 'error'
        output = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        trace['output'] = output
        self.trace.append(trace)
        return {'role': 'tool', 'tool_call_id': call_id, 'content': output}, trace

    def _execute(self, name, arguments):
        if name == 'lookup_record':
            if set(arguments) != {'name'} or not isinstance(arguments['name'], str):
                raise ValueError('lookup_record expects exactly one string field named name')
            record_name = arguments['name']
            if record_name not in self.lookup:
                raise ValueError('record name not found')
            return {'path': self.lookup[record_name]}
        if name == 'read_record':
            if set(arguments) != {'path'} or not isinstance(arguments['path'], str):
                raise ValueError('read_record expects exactly one string field named path')
            relative = arguments['path']
            if relative not in self.allowed_paths:
                raise ValueError('record not found')
            path = (self.fixture_root / relative).resolve()
            if path.parent != self.fixture_root:
                raise ValueError('record not found')
            with path.open('r', encoding='utf-8') as fixture:
                value = json.load(fixture)
            if not isinstance(value, dict):
                raise ValueError('record must contain a JSON object')
            return value
        if name == 'add_numbers':
            if (set(arguments) != {'a', 'b'} or type(arguments.get('a')) is not int or
                    type(arguments.get('b')) is not int):
                raise ValueError('add_numbers expects exactly two integer fields named a and b')
            return {'result': arguments['a'] + arguments['b']}
        raise ValueError(f'unknown tool {name!r}')


def _random_label():
    return secrets.token_hex(6)


def _write_fixture(root, filename, value):
    path = Path(root) / filename
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    path.write_bytes(raw)
    return {'path': filename, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def _scenario_fixture(name, root):
    """Create owned hidden fixture values and return prompts plus expectations."""
    fixture_files = []
    lookup = {}

    def add_file(filename, value):
        fixture_files.append(_write_fixture(root, filename, value))
        return value

    if name in ('smoke', 'streamed_read', 'cancel_then_read'):
        label = _random_label()
        value = add_file('single.json', {'label': label, 'code': secrets.randbelow(8000) + 1000})
        return lookup, fixture_files, [{
            'prompt': 'Read single.json with read_record and return a JSON object with exactly its label and code. Use the provided tool calls to obtain the data. Only the final answer after all tool results should be JSON without Markdown.',
            'expected_calls': [{'name': 'read_record', 'arguments': {'path': 'single.json'}, 'status': 'ok'}],
            'expected_answer': value,
        }]

    if name == 'two_step':
        label = _random_label()
        value = add_file('two_step.json', {'label': label, 'code': secrets.randbelow(8000) + 1000})
        lookup['willow'] = 'two_step.json'
        return lookup, fixture_files, [{
            'prompt': "Look up the record named 'willow', read the returned path, and return JSON with exactly its label and code. Use the provided tool calls to obtain the data. Only the final answer after all tool results should be JSON without Markdown.",
            'expected_calls': [
                {'name': 'lookup_record', 'arguments': {'name': 'willow'}, 'status': 'ok'},
                {'name': 'read_record', 'arguments': {'path': 'two_step.json'}, 'status': 'ok'},
            ],
            'expected_answer': value,
        }]

    if name == 'three_step':
        label = _random_label()
        a, b = secrets.randbelow(80) + 10, secrets.randbelow(80) + 10
        value = add_file('three_step.json', {'label': label, 'a': a, 'b': b})
        lookup['elm'] = 'three_step.json'
        return lookup, fixture_files, [{
            'prompt': "Look up the record named 'elm', read its path, add its a and b values with add_numbers, then return JSON with exactly the label and sum. Use the provided tool calls to obtain the data. Only the final answer after all tool results should be JSON without Markdown.",
            'expected_calls': [
                {'name': 'lookup_record', 'arguments': {'name': 'elm'}, 'status': 'ok'},
                {'name': 'read_record', 'arguments': {'path': 'three_step.json'}, 'status': 'ok'},
                {'name': 'add_numbers', 'arguments': {'a': a, 'b': b}, 'status': 'ok'},
            ],
            'expected_answer': {'label': label, 'sum': a + b},
        }]

    if name == 'stale_path_recovery':
        label = _random_label()
        value = add_file('current.json', {'label': label, 'code': secrets.randbelow(8000) + 1000})
        lookup['spruce'] = 'current.json'
        return lookup, fixture_files, [{
            'prompt': "Try read_record on retired.json. If it is unavailable, look up the record named 'spruce', read its current path, and return JSON with exactly the current label and code. Use the provided tool calls to obtain the data. Only the final answer after all tool results should be JSON without Markdown.",
            'expected_calls': [
                {'name': 'read_record', 'arguments': {'path': 'retired.json'}, 'status': 'error'},
                {'name': 'lookup_record', 'arguments': {'name': 'spruce'}, 'status': 'ok'},
                {'name': 'read_record', 'arguments': {'path': 'current.json'}, 'status': 'ok'},
            ],
            'expected_answer': value,
        }]

    if name == 'changed_followup':
        first = {'label': _random_label()}
        second = {'label': _random_label()}
        while second == first:
            second = {'label': _random_label()}
        fixture_files.append(_write_fixture(root, 'mutable.json', first))
        return lookup, fixture_files, [
            {
                'prompt': 'Read mutable.json with read_record and return a JSON object with exactly its label. Use the provided tool calls to obtain the data. Only the final answer after all tool results should be JSON without Markdown.',
                'expected_calls': [{'name': 'read_record', 'arguments': {'path': 'mutable.json'}, 'status': 'ok'}],
                'expected_answer': first,
                'replace_after_round': second,
            },
            {
                'prompt': 'Read mutable.json again now with read_record and return a JSON object with exactly its current label. Use the provided tool calls to obtain the data. Only the final answer after all tool results should be JSON without Markdown.',
                'expected_calls': [{'name': 'read_record', 'arguments': {'path': 'mutable.json'}, 'status': 'ok'}],
                'expected_answer': second,
            },
        ]

    raise AgentLoopFailure(f'unknown scenario {name!r}')


def _extract_nonstream(record):
    try:
        data = json.loads(record['raw_response'])
        record['response_json'] = data
        choice = data['choices'][0]
        message = choice['message']
        finish = choice.get('finish_reason')
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise AgentLoopFailure(f'malformed chat completion: {error}') from error
    if not isinstance(message, dict) or message.get('role') != 'assistant':
        raise AgentLoopFailure('chat completion did not contain an assistant message')
    calls = message.get('tool_calls') or []
    if not isinstance(calls, list):
        raise AgentLoopFailure('assistant tool_calls must be an array')
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
            raise AgentLoopFailure('assistant returned a malformed tool call')
        _json_object(call['function'].get('arguments'), 'tool arguments')
    record.update(message=message, finish_reason=finish,
                  content=message.get('content') or '', tool_calls=calls,
                  usage=data.get('usage'))
    return message, finish, calls


class AgentHarness:
    def __init__(self, args):
        self.args = args
        self.url = urlsplit(args.base_url)
        if self.url.scheme not in ('http', 'https') or not self.url.hostname:
            raise ValueError('--base-url must be an http(s) URL with a hostname')
        if self.url.query or self.url.fragment:
            raise ValueError('--base-url must not contain a query or fragment')
        api_path = self.url.path.rstrip('/')
        if api_path.endswith('/v1'):
            self.api_base = f'{self.url.scheme}://{self.url.netloc}{api_path}'
        else:
            self.api_base = f'{self.url.scheme}://{self.url.netloc}{api_path}/v1'
        self.endpoint = urlsplit(self.api_base).path.rstrip('/') + '/chat/completions'
        self.connection_type = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        self.openai_client = None
        if args.client == 'openai':
            from openai import OpenAI
            self.openai_client = OpenAI(
                base_url=self.api_base,
                api_key='local-test',
                max_retries=0,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
        self.records = []
        self.failures = []
        self.scenarios = []

    def _http_turn(self, scenario, turn, messages, stream):
        tools = tool_schemas()
        tool_only = getattr(self.args, 'tool_policy', 'auto') == 'tool_only'
        if tool_only:
            tools.append({'type': 'function', 'function': {
                'name': 'finish_task',
                'description': 'Deliver the final answer after the requested work and all necessary tool calls are complete.',
                'parameters': {'type': 'object', 'properties': {'answer': {
                    'type': 'object', 'properties': {'label': {'type': 'string'},
                        'code': {'type': 'integer'}, 'sum': {'type': 'integer'}},
                    'required': ['label'], 'additionalProperties': False}},
                    'required': ['answer'], 'additionalProperties': False}}})
        body = {
            'model': self.args.model,
            'messages': _json_copy(messages),
            'tools': tools,
            'tool_choice': 'required' if tool_only else 'auto',
            'parallel_tool_calls': False,
            'max_tokens': MAX_TOKENS,
            'temperature': 0,
            'stream': stream,
        }
        if stream:
            body['stream_options'] = {'include_usage': True}
        encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        record = {
            'scenario': scenario,
            'turn': turn,
            'endpoint': self.endpoint,
            'client': self.args.client,
            'stream': stream,
            'request': body,
            'request_body_utf8': encoded.decode('utf-8'),
            'history_before': _json_copy(messages),
            'expected_http_status': 200,
        }
        self.records.append(record)
        started = time.monotonic()
        try:
            if self.args.client == 'openai':
                status, headers, raw_response = self._send_openai(body, stream)
            else:
                status, headers, raw_response = self._send_stdlib(encoded, stream)
            record['http_status'] = status
            record['response_headers'] = headers
            record['raw_response'] = raw_response.decode('utf-8', 'replace')
            if status != 200:
                raise AgentLoopFailure(f'HTTP {status}: {record["raw_response"][:1000]}')
            if stream:
                parsed = assemble_sse(record['raw_response'])
                record.update(sse_events=parsed['events'], sse_done=parsed['sse_done'],
                              message=parsed['message'], finish_reason=parsed['finish_reason'],
                              content=parsed['message'].get('content') or '',
                              tool_calls=parsed['message'].get('tool_calls') or [], usage=parsed['usage'])
                return parsed['message'], parsed['finish_reason'], parsed['message'].get('tool_calls') or [], parsed['usage']
            message, finish, calls = _extract_nonstream(record)
            return message, finish, calls, record.get('usage')
        except Exception as error:
            record['transport_or_protocol_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            record['elapsed_seconds'] = round(time.monotonic() - started, 3)

    def _send_openai(self, body, stream):
        """Exercise SDK response parsing and retain the original response bytes."""
        if self.openai_client is None:
            raise AgentLoopFailure('OpenAI client was not initialized')
        endpoint = self.openai_client.chat.completions
        if stream:
            with endpoint.with_streaming_response.create(**body) as response:
                raw_body = response.read()
                if raw_body is None:
                    raw_body = response.http_response.content
                # read() caches the wire bytes; parse() then consumes the SDK's
                # typed stream over that same body. This does not qualify live
                # cancellation, because this path intentionally buffers.
                chunks = list(response.parse())
                self.records[-1]['sdk_parsed_types'] = sorted({type(chunk).__name__ for chunk in chunks})
                self.records[-1]['sdk_parsed_chunks'] = [chunk.model_dump(mode='json', exclude_unset=True) for chunk in chunks]
                if not chunks:
                    raise AgentLoopFailure('SDK parsed no streaming chunks')
                return response.status_code, list(response.headers.items()), raw_body
        response = endpoint.with_raw_response.create(**body)
        raw_body = response.http_response.content
        parsed = response.parse()
        self.records[-1]['sdk_parsed_type'] = type(parsed).__name__
        self.records[-1]['sdk_parsed_response'] = parsed.model_dump(mode='json', exclude_unset=True)
        return response.status_code, list(response.headers.items()), raw_body

    def _send_stdlib(self, encoded_body, stream):
        """Transport seam: return (status, headers, raw body bytes.

        A future SDK-backed client can replace this method while preserving the
        scenario loop, HTTP record contract, and response/oracle processing.
        """
        conn = self.connection_type(self.url.hostname, self.url.port, timeout=HTTP_TIMEOUT_SECONDS)
        try:
            conn.request('POST', self.endpoint, body=encoded_body, headers={
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream' if stream else 'application/json',
                'Connection': 'close',
            })
            response = conn.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            conn.close()

    def _cancel_tool_stream(self, messages):
        """Cancel a live SDK tool stream before execution, then consume its end."""
        if self.openai_client is None:
            raise AgentLoopFailure('Live agent cancellation requires --client openai')
        request_id = 'agent-cancel-' + secrets.token_hex(8)
        body = {'model': self.args.model, 'messages': _json_copy(messages),
                'tools': tool_schemas(), 'stream': True, 'temperature': 0,
                'max_tokens': MAX_TOKENS, 'parallel_tool_calls': False,
                'tool_choice': {'type': 'function', 'function': {'name': 'read_record'}},
                'stream_options': {'include_usage': True}}
        record = {'status': 'FAIL', 'request': {**body, 'request_id': request_id},
                  'sdk_chunks': [], 'cancelled': False, 'executed_calls': []}
        self.scenarios[-1]['cancellation'] = record
        finish_reasons = []
        with self.openai_client.chat.completions.with_streaming_response.create(
                **body, extra_body={'request_id': request_id}) as response:
            record['http_status'] = response.status_code
            for chunk in response.parse():
                event = chunk.model_dump(mode='json', exclude_unset=True)
                record['sdk_chunks'].append(event)
                for choice in chunk.choices:
                    if choice.finish_reason:
                        finish_reasons.append(choice.finish_reason)
                    if choice.delta.tool_calls and not record['cancelled'] and not finish_reasons:
                        conn = self.connection_type(self.url.hostname, self.url.port, timeout=30)
                        base_path = self.url.path.rstrip('/')
                        if base_path.endswith('/v1'): base_path = base_path[:-3]
                        try:
                            conn.request('POST', base_path+'/api/cancel',
                                body=json.dumps({'request_id': request_id}),
                                headers={'Content-Type': 'application/json', 'Connection': 'close'})
                            cancel_response = conn.getresponse()
                            raw = cancel_response.read().decode('utf-8')
                            record['cancel_response'] = {'http_status': cancel_response.status, 'raw_response': raw}
                            record['cancelled'] = cancel_response.status == 200 and json.loads(raw).get('cancelled') is True
                        finally:
                            conn.close()
        record['finish_reasons'] = finish_reasons
        if not record['cancelled'] or 'cancel' not in finish_reasons:
            raise AgentLoopFailure('Live tool stream was not cancelled before completion')
        record['status'] = 'PASS'
        return record

    @staticmethod
    def _assert_answer(actual_content, expected, scenario_name, round_index):
        actual = _json_object(actual_content, f'{scenario_name} round {round_index} answer')
        if set(actual) != set(expected):
            raise AgentLoopFailure(
                f'round {round_index}: answer keys {sorted(actual)!r} != {sorted(expected)!r}')
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                raise AgentLoopFailure(
                    f'round {round_index}: answer field {key!r} expected {expected_value!r}, got {actual_value!r}')
        return actual

    @staticmethod
    def _check_round_trace(round_spec, calls, scenario_name, round_index):
        actual = calls
        expected = round_spec['expected_calls']
        if len(actual) != len(expected):
            raise AgentLoopFailure(
                f'round {round_index}: expected {len(expected)} tool calls, observed {len(actual)}')
        for index, (got, want) in enumerate(zip(actual, expected), 1):
            if got['name'] != want['name'] or got['arguments'] != want['arguments'] or got['status'] != want['status']:
                raise AgentLoopFailure(
                    f'round {round_index} call {index}: expected {want!r}, got '
                    f"{{'name': {got['name']!r}, 'arguments': {got['arguments']!r}, 'status': {got['status']!r}}}")

    def _assert_cache(self, usage, scenario_name, turn):
        if not isinstance(usage, dict):
            raise AgentLoopFailure(f'{scenario_name} turn {turn}: response omitted usage needed for cache checks')
        details = usage.get('prompt_tokens_details')
        cached = details.get('cached_tokens') if isinstance(details, dict) else None
        prompt_tokens = usage.get('prompt_tokens')
        if type(cached) is not int or type(prompt_tokens) is not int:
            raise AgentLoopFailure(
                f'{scenario_name} turn {turn}: usage omitted integer cached/prompt token counts')
        if self.args.cache == 'off' and cached != 0:
            raise AgentLoopFailure(f'{scenario_name} turn {turn}: cache=off but cached_tokens={cached}')
        if self.args.cache == 'on' and turn > 1 and not (0 < cached < prompt_tokens):
            raise AgentLoopFailure(
                f'{scenario_name} turn {turn}: expected prefix-cache hit, got {cached}/{prompt_tokens}')

    def run_scenario(self, name, streamed=False):
        result = {
            'name': name,
            'streamed': streamed,
            'status': 'FAIL',
            'executed_calls': [],
            'history': [],
            'outcomes': [],
            'failures': [],
            'fixture_manifest': [],
        }
        self.scenarios.append(result)
        scenario_start = len(self.records)
        turn_count = 0
        try:
            with TemporaryDirectory(prefix='hip-oga-agent-') as temporary:
                fixture_root = Path(temporary)
                lookup, manifest, rounds = _scenario_fixture(name, fixture_root)
                result['fixture_manifest'] = manifest
                allowed_paths = {row['path'] for row in manifest}
                executor = BoundedToolExecutor(fixture_root, lookup, allowed_paths)
                result['lookup_fixture'] = dict(lookup)
                history = [{'role': 'system', 'content': (
                    'You are a tool-using assistant. Actually invoke the provided functions to obtain data; '
                    'do not describe an invocation as ordinary text or a Markdown code block. '
                    'Write each invocation as <tool_call>{"name": "function_name", "arguments": {}}</tool_call> '
                    'with the appropriate function name and arguments. The <tools> tags describe available '
                    'functions and are not invocation tags. Wait for tool results before answering. '
                    'Only your final answer should be a JSON object without Markdown.') }]
                tool_only = getattr(self.args, 'tool_policy', 'auto') == 'tool_only'
                if tool_only:
                    history[0]['content'] = (
                        'You are a tool-using assistant. Use the provided functions to do the requested work. '
                        'Read tool results before choosing your next action. Once the task is complete, '
                        'call finish_task with the answer object containing exactly the fields requested by the user. '
                        'Do not call finish_task until you have obtained the requested data from the tools.')

                for round_index, round_spec in enumerate(rounds, 1):
                    history.append({'role': 'user', 'content': round_spec['prompt']})
                    if name == 'cancel_then_read' and round_index == 1:
                        result['cancellation'] = self._cancel_tool_stream(history)
                    round_trace_start = len(executor.trace)
                    round_turn_start = turn_count
                    round_outcome = {
                        'round': round_index,
                        'prompt': round_spec['prompt'],
                        'expected_answer': _json_copy(round_spec['expected_answer']),
                        'expected_calls': _json_copy(round_spec['expected_calls']),
                        'turns': [],
                    }
                    result['outcomes'].append(round_outcome)

                    while True:
                        if turn_count >= MAX_TURNS:
                            raise AgentLoopFailure(f'scenario exceeded {MAX_TURNS} model turns')
                        turn_count += 1
                        result['history'].append({'before_turn': turn_count, 'messages': _json_copy(history)})
                        message, finish, calls, usage = self._http_turn(name, turn_count, history, streamed)
                        self._assert_cache(usage, name, turn_count)
                        round_outcome['turns'].append({
                            'turn': turn_count,
                            'finish_reason': finish,
                            'tool_calls': _json_copy(calls),
                            'usage': _json_copy(usage),
                        })
                        history.append(_json_copy(message))

                        terminal_answer = None
                        if calls:
                            if finish != 'tool_calls':
                                raise AgentLoopFailure(f'turn {turn_count}: tool calls ended with {finish!r}')
                            if len(calls) != 1:
                                raise AgentLoopFailure(f'turn {turn_count}: expected at most one call, got {len(calls)}')
                            call = calls[0]
                            if tool_only and call['function']['name'] == 'finish_task':
                                call_id = call.get('id')
                                if not isinstance(call_id, str) or not call_id or call_id in executor.call_ids:
                                    raise AgentLoopFailure('terminal call has a missing or reused ID')
                                executor.call_ids.add(call_id)
                                arguments = _json_object(call['function']['arguments'], 'finish_task arguments')
                                if set(arguments) != {'answer'} or not isinstance(arguments['answer'], dict):
                                    raise AgentLoopFailure('finish_task requires exactly one answer object')
                                terminal_answer = json.dumps(arguments['answer'])
                                history.append({'role': 'tool', 'tool_call_id': call_id,
                                                'content': '{"delivered":true}'})
                                result.setdefault('terminal_calls', []).append(_json_copy(call))
                            else:
                                tool_message, trace = executor.execute(call)
                                result['executed_calls'].append(_json_copy(trace))
                                history.append(tool_message)
                                result['history'].append({'after_turn': turn_count, 'messages': _json_copy(history)})
                                continue

                        if terminal_answer is None and (finish != 'stop' or tool_only):
                            raise AgentLoopFailure(
                                f'turn {turn_count}: final answer finish_reason must be stop, got {finish!r}')
                        if terminal_answer is None and message.get('tool_calls'):
                            raise AgentLoopFailure(f'turn {turn_count}: final answer unexpectedly has tool calls')
                        actual_answer = self._assert_answer(
                            terminal_answer if terminal_answer is not None else message.get('content') or '',
                            round_spec['expected_answer'], name, round_index)
                        round_outcome['actual_answer'] = actual_answer
                        round_outcome['finish_reason'] = finish
                        self._check_round_trace(
                            round_spec, executor.trace[round_trace_start:], name, round_index)
                        result['history'].append({'after_turn': turn_count, 'messages': _json_copy(history)})
                        break

                    if turn_count - round_turn_start > MAX_TURNS:
                        raise AgentLoopFailure(f'round {round_index} exceeded the turn bound')
                    if 'replace_after_round' in round_spec:
                        updated = round_spec['replace_after_round']
                        replacement = _write_fixture(fixture_root, 'mutable.json', updated)
                        result['fixture_manifest'].append({**replacement, 'revision': 2})
                        result['outcomes'][-1]['fixture_replaced_after_answer'] = _json_copy(updated)

                if turn_count > MAX_TURNS:
                    raise AgentLoopFailure(f'scenario exceeded {MAX_TURNS} model turns')
                if executor.executed_count > MAX_TOOL_CALLS:
                    raise AgentLoopFailure(f'scenario exceeded {MAX_TOOL_CALLS} tool executions')
                result['executed_calls'] = _json_copy(executor.trace)
                result['status'] = 'PASS'
        except Exception as error:
            message = f'{name}: {type(error).__name__}: {error}'
            result['failure'] = message
            result['failures'].append(message)
            result['executed_calls'] = result.get('executed_calls', [])
            self.failures.append(message)
        finally:
            result['turn_count'] = turn_count
            result['request_records'] = [
                record for record in self.records[scenario_start:] if record['scenario'] == name
            ]
        return result

    def run(self):
        names = ['smoke'] if self.args.phase == 'smoke' else [
            'smoke', 'streamed_read', 'two_step', 'three_step', 'stale_path_recovery', 'changed_followup',
            'cancel_then_read'
        ]
        if getattr(self.args, 'scenario', None):
            names = self.args.scenario
        for name in names:
            self.run_scenario(name, streamed=(name == 'streamed_read'))
        return {
            'status': 'PASS' if not self.failures else 'FAIL',
            'model': self.args.model,
            'base_url': self.args.base_url,
            'phase': self.args.phase,
            'cache': self.args.cache,
            'client': self.args.client,
            'tool_policy': getattr(self.args, 'tool_policy', 'auto'),
            'scenario_filter': getattr(self.args, 'scenario', None),
            'limits': {'max_model_turns_per_scenario': MAX_TURNS,
                       'max_tool_executions_per_scenario': MAX_TOOL_CALLS,
                       'http_timeout_seconds': HTTP_TIMEOUT_SECONDS,
                       'max_tokens': MAX_TOKENS},
            'records': self.records,
            'scenarios': self.scenarios,
            'failures': self.failures,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--cache', choices=('on', 'off'), required=True)
    parser.add_argument('--client', choices=('stdlib', 'openai'), default='stdlib')
    parser.add_argument('--tool-policy', choices=('auto', 'tool_only'), default='auto')
    parser.add_argument('--scenario', action='append', choices=(
        'smoke', 'streamed_read', 'two_step', 'three_step', 'stale_path_recovery',
        'changed_followup', 'cancel_then_read'))
    args = parser.parse_args(argv)
    try:
        harness = AgentHarness(args)
        result = harness.run()
    except Exception as error:
        result = {
            'status': 'FAIL', 'model': args.model, 'base_url': args.base_url,
            'phase': args.phase, 'cache': args.cache, 'client': args.client,
            'records': [], 'scenarios': [],
            'failures': [f'{type(error).__name__}: {error}'],
        }
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / 'result.json'
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{result["status"]}: {output_file}')
    for failure in result.get('failures', []):
        print(f'FAIL: {failure}')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
