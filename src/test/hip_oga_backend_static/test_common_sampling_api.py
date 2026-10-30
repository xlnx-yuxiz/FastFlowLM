#!/usr/bin/env python3
"""Bounded common-sampler HTTP contract checks against an already running server.

The caller owns server startup. The selected startup sampler is recorded here;
the outer fixture runner should verify that value against the server log.
"""
import argparse
import http.client
import json
from pathlib import Path
import shlex
import sys
import time
from urllib.parse import urlsplit


MAX_TOKENS = 32
TIMEOUT_SECONDS = 90
PROBE_MESSAGE = [{'role': 'user', 'content': 'Name a few colors in one short sentence.'}]
STRING_SCHEMA = {
    'type': 'object',
    'required': ['label'],
    'additionalProperties': False,
    'properties': {'label': {'type': 'string', 'minLength': 3, 'maxLength': 3}},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True,
                        help='Base server URL, for example http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--startup-sampler', choices=('backend', 'common'), required=True,
                        help='Sampler used to start the server; outer runner verifies server logs')
    parser.add_argument('--tool-routing', choices=('template', 'unsupported'), default='template',
                        help='Tool-call route exercised by the bounded required-tool case')
    parser.add_argument('--output', type=Path, required=True,
                        help='Directory for common-sampling-api-result.json')
    args = parser.parse_args()

    url = urlsplit(args.base_url)
    if url.scheme not in ('http', 'https') or not url.hostname or url.query or url.fragment:
        parser.error('--base-url must be an HTTP(S) URL without a query or fragment')
    base_path = url.path.rstrip('/')
    endpoint = (base_path + '/chat/completions' if base_path.endswith('/v1')
                else base_path + '/v1/chat/completions')
    connection_type = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
    args.output.mkdir(parents=True, exist_ok=True)
    records, failures = [], []

    def check(condition, message):
        if not condition:
            failures.append(message)

    def request(name, fields=None, *, prompt=None, messages=None, stream=False, expected_status=200):
        body = {
            'model': args.model,
            'messages': messages if messages is not None else [
                {'role': 'user', 'content': prompt or 'Reply briefly.'}],
            'max_tokens': MAX_TOKENS,
            'stream': stream,
        }
        body.update(fields or {})
        encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        option_values = body.get('options')
        explicit_sampler = body.get('sampler')
        if explicit_sampler is None and isinstance(option_values, dict):
            explicit_sampler = option_values.get('sampler')
        record = {
            'name': name,
            'endpoint': endpoint,
            'request': body,
            'request_body_utf8': encoded.decode('utf-8'),
            'expected_http_status': expected_status,
            'expected_sampler': (args.startup_sampler if explicit_sampler is None
                                 else explicit_sampler),
            'http_status': None,
            'raw_response': None,
        }
        records.append(record)
        connection = connection_type(url.hostname, url.port, timeout=TIMEOUT_SECONDS)
        started = time.monotonic()
        try:
            connection.request('POST', endpoint, body=encoded, headers={
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream' if stream else 'application/json',
                'Connection': 'close',
            })
            response = connection.getresponse()
            raw = response.read().decode('utf-8', 'replace')
            record.update(http_status=response.status, response_headers=response.getheaders(),
                          raw_response=raw)
        except Exception as error:
            record['transport_error'] = f'{type(error).__name__}: {error}'
            failures.append(f'{name}: {record["transport_error"]}')
            return record
        finally:
            connection.close()
            record['elapsed_seconds'] = round(time.monotonic() - started, 3)

        check(record['http_status'] == expected_status,
              f'{name}: HTTP {record["http_status"]}, expected {expected_status}')
        if record['http_status'] != expected_status:
            return record
        if expected_status == 400:
            try:
                data = json.loads(record['raw_response'])
                record['response_json'] = data
                error = data.get('error') if isinstance(data, dict) else None
                record['error'] = error
                check(isinstance(error, dict) and error.get('type') == 'invalid_request_error',
                      f'{name}: expected JSON invalid_request_error, got {error!r}')
            except (json.JSONDecodeError, AttributeError, TypeError) as error:
                failures.append(f'{name}: malformed HTTP 400 error payload: {error}')
            return record

        if stream:
            events, chunks, finish_reasons, done = [], [], [], False
            for line in record['raw_response'].splitlines():
                if not line.startswith('data:'):
                    continue
                value = line[5:].strip()
                if value == '[DONE]':
                    done = True
                    continue
                try:
                    event = json.loads(value)
                except json.JSONDecodeError as error:
                    failures.append(f'{name}: malformed SSE JSON: {error}')
                    continue
                events.append(event)
                if not isinstance(event, dict) or 'error' in event:
                    failures.append(f'{name}: SSE event is malformed or contains an error')
                    continue
                for choice in event.get('choices', []):
                    if not isinstance(choice, dict):
                        failures.append(f'{name}: malformed SSE choice')
                        continue
                    delta = choice.get('delta') or {}
                    if not isinstance(delta, dict):
                        failures.append(f'{name}: malformed SSE delta')
                        continue
                    part = delta.get('content')
                    if part is not None:
                        if isinstance(part, str):
                            chunks.append(part)
                        else:
                            failures.append(f'{name}: SSE content is not a string')
                    if choice.get('finish_reason'):
                        finish_reasons.append(choice['finish_reason'])
            record.update(sse_events=events, sse_done=done, content=''.join(chunks),
                          finish_reason=finish_reasons[-1] if finish_reasons else None)
            check(done and bool(events), f'{name}: missing SSE event or [DONE]')
            check(bool(record['content'].strip()), f'{name}: empty streamed assistant content')
            return record

        try:
            data = json.loads(record['raw_response'])
            record['response_json'] = data
            check(isinstance(data, dict) and 'error' not in data,
                  f'{name}: malformed JSON completion or response error')
            choices = data.get('choices') if isinstance(data, dict) else None
            check(isinstance(choices, list) and bool(choices), f'{name}: completion has no choices')
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                return record
            choice = choices[0]
            message = choice.get('message')
            check(isinstance(message, dict), f'{name}: completion has no assistant message')
            if not isinstance(message, dict):
                return record
            record['message'] = message
            record['content'] = message.get('content') or ''
            record['tool_calls'] = message.get('tool_calls') or []
            record['finish_reason'] = choice.get('finish_reason')
            check(isinstance(record['content'], str), f'{name}: assistant content is not a string')
            if not record['tool_calls']:
                check(bool(record['content'].strip()), f'{name}: empty assistant content')
        except (json.JSONDecodeError, TypeError, AttributeError) as error:
            failures.append(f'{name}: malformed JSON completion: {error}')
        return record

    def assert_equal_content(left, right, name):
        check(left.get('http_status') == 200 and right.get('http_status') == 200,
              f'{name}: cannot compare unsuccessful requests')
        check(left.get('content') == right.get('content'),
              f'{name}: deterministic content mismatch: {left.get("content")!r} != '
              f'{right.get("content")!r}')

    def assert_bounded_json_string(record, name):
        try:
            value = json.loads(record.get('content', ''))
        except (json.JSONDecodeError, TypeError) as error:
            failures.append(f'{name}: invalid constrained JSON: {error}')
            return
        check(isinstance(value, dict) and set(value) == {'label'},
              f'{name}: expected exactly the label property, got {value!r}')
        if isinstance(value, dict) and isinstance(value.get('label'), str):
            check(len(value['label']) == 3,
                  f'{name}: expected label length 3, got {len(value["label"])}')
            record['validated_json'] = value
        else:
            failures.append(f'{name}: label is not a string')

    try:
        seeded_fields = {'temperature': 0.9, 'top_k': 20, 'top_p': 0.9, 'seed': 0}

        # Both request locations select the common sampler. Identical seeded
        # probes make a successful placement test observable and reproducible.
        top_common = request('sampler_top_level_common',
                             {**seeded_fields, 'sampler': 'common'}, messages=PROBE_MESSAGE)
        options_common = request('sampler_under_options_common',
                                 {'options': {**seeded_fields, 'sampler': 'common'}},
                                 messages=PROBE_MESSAGE)
        assert_equal_content(top_common, options_common, 'sampler_top_level_vs_options_common')

        # Omitted sampler must resolve to the server's startup provider even
        # after a request explicitly selects the opposite provider.
        inherited_first = request('startup_sampler_seed0_first', seeded_fields,
                                  messages=PROBE_MESSAGE)
        alternate = 'common' if args.startup_sampler == 'backend' else 'backend'
        request('intervening_opposite_sampler',
                {**seeded_fields, 'seed': 19, 'sampler': alternate},
                prompt='Give a brief greeting.')
        inherited_replay = request('startup_sampler_seed0_replay', seeded_fields,
                                   messages=PROBE_MESSAGE)
        assert_equal_content(inherited_first, inherited_replay, 'startup_sampler_seed0_replay')

        # An explicit backend selection is request-local. The following
        # sampler-omitted request must return to the recorded startup default.
        explicit_backend = request('explicit_backend_seed0',
                                   {**seeded_fields, 'sampler': 'backend'}, messages=PROBE_MESSAGE)
        omitted_after_backend = request('omitted_after_backend_seed0', seeded_fields,
                                        messages=PROBE_MESSAGE)
        assert_equal_content(inherited_first, omitted_after_backend,
                             'omitted_sampler_restored_after_backend_override')
        omitted_after_backend['backend_override_record'] = explicit_backend['name']

        # Invalid request values must fail cleanly; valid requests still work
        # immediately after each rejection.
        request('invalid_sampler_string', {'sampler': 'bogus'}, expected_status=400)
        request('recovery_after_invalid_sampler_string',
                {**seeded_fields, 'sampler': 'common'}, messages=PROBE_MESSAGE)
        request('invalid_sampler_type_in_options', {'options': {'sampler': 7}},
                expected_status=400)
        request('recovery_after_invalid_sampler_type',
                {**seeded_fields, 'sampler': 'common'}, messages=PROBE_MESSAGE)

        # Greedy common-sampler output must have identical stream framing
        # semantics and decoded text for otherwise identical requests.
        greedy = {'sampler': 'common', 'temperature': 0}
        greedy_json = request('common_temperature_zero_nonstream', greedy,
                              prompt='Reply with one short color name.', stream=False)
        greedy_stream = request('common_temperature_zero_stream', greedy,
                                prompt='Reply with one short color name.', stream=True)
        assert_equal_content(greedy_json, greedy_stream, 'common_temperature_zero_stream_parity')

        schema = {'type': 'json_schema', 'json_schema': {
            'name': 'bounded_label', 'strict': True, 'schema': STRING_SCHEMA}}
        constrained = request('common_string_min_max_length_schema',
                              {'sampler': 'common', 'temperature': 0, 'response_format': schema},
                              prompt='Return JSON with label exactly equal to cat.')
        assert_bounded_json_string(constrained, 'common_string_min_max_length_schema')
        if constrained.get('validated_json', {}).get('label') != 'cat':
            failures.append('common_string_min_max_length_schema: expected constrained label "cat"')

        tool = {'type': 'function', 'function': {
            'name': 'capture_label',
            'description': 'Capture the requested short label.',
            'strict': True,
            'parameters': STRING_SCHEMA,
        }}
        tool_prompt = 'Call capture_label exactly once with label cat.'
        if args.tool_routing == 'unsupported':
            request('common_required_template_tool_rejected',
                    {'sampler':'common','temperature':0,'tools':[tool],
                     'tool_choice':'required','parallel_tool_calls':False},
                    prompt=tool_prompt,expected_status=400)
            request('common_structured_tool_rejected',
                    {'sampler':'common','temperature':0,'tools':[tool],
                     'tool_choice':'auto','tool_routing':'structured','parallel_tool_calls':False},
                    prompt=tool_prompt,expected_status=400)
            recovery=request('common_recovery_after_unsupported_tools',
                             {**seeded_fields,'sampler':'common'},messages=PROBE_MESSAGE)
            assert_equal_content(top_common,recovery,'recovery_after_unsupported_tools')
        else:
            tool_result = request(
                'common_single_tool_string_min_max_length', {'sampler':'common','temperature':0,'tools':[tool],
                    'tool_choice':'required','parallel_tool_calls':False}, prompt=tool_prompt)
            calls = tool_result.get('tool_calls', [])
            check(tool_result.get('finish_reason') == 'tool_calls',
                  'common_single_tool_string_min_max_length: expected tool_calls finish reason')
            check(isinstance(calls, list) and len(calls) == 1,
                  f'common_single_tool_string_min_max_length: expected one tool call, got {calls!r}')
            if isinstance(calls, list) and len(calls) == 1:
                call = calls[0]
                check(call.get('type') == 'function' and bool(call.get('id')),
                      'common_single_tool_string_min_max_length: missing function type or call ID')
                function = call.get('function') or {}
                check(function.get('name') == 'capture_label',
                      f'common_single_tool_string_min_max_length: wrong function {function!r}')
                try:
                    arguments = json.loads(function.get('arguments', ''))
                    tool_result['validated_tool_arguments'] = arguments
                    check(isinstance(arguments, dict) and arguments.get('label') == 'cat',
                          f'common_single_tool_string_min_max_length: unexpected arguments {arguments!r}')
                    if isinstance(arguments, dict) and isinstance(arguments.get('label'), str):
                        check(3 <= len(arguments['label']) <= 3,
                              'common_single_tool_string_min_max_length: label violates min/max length')
                except (json.JSONDecodeError, TypeError) as error:
                    failures.append(f'common_single_tool_string_min_max_length: invalid tool arguments: {error}')

    except Exception as error:
        failures.append(f'{type(error).__name__}: {error}')

    command = shlex.join([sys.executable, *sys.argv])
    result = {
        'status': 'PASS' if not failures else 'FAIL',
        'model': args.model,
        'base_url': args.base_url,
        'startup_sampler': args.startup_sampler,
        'tool_routing': args.tool_routing,
        'max_tokens': MAX_TOKENS,
        'timeout_seconds': TIMEOUT_SECONDS,
        'exact_command': command,
        'records': records,
        'case_names': [record['name'] for record in records],
        'failures': failures,
    }
    output_path = args.output / 'common-sampling-api-result.json'
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('COMMAND: ' + command)
    print(json.dumps({'status': result['status'], 'requests': len(records),
                      'result_file': str(output_path), 'failures': failures}, ensure_ascii=False))
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
