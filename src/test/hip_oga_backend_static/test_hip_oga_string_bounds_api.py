#!/usr/bin/env python3
"""Exact decoded string bounds over real HTTP, with a local fixture echo tool."""
import argparse
import base64
import hashlib
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from test_hip_oga_agent_api import assemble_sse


MAX_TOKENS = 256
TIMEOUT_SECONDS = 120
ROUTES = ('response_format', 'named', 'required', 'structured_auto')
PAYLOADS = (('backslash', 'C:\\tmp'), ('mixed_unicode', 'A\n"中文😀"'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError('Non-JSON numeric constant: ' + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def string_schema(target):
    return {'type': 'object', 'additionalProperties': False, 'required': ['values'],
            'properties': {'values': {'type': 'array', 'minItems': 1, 'maxItems': 1,
                'items': {'type': 'string', 'minLength': len(target), 'maxLength': len(target)}}}}


def copy_prompt(route, target):
    action = ('Return only a JSON object with the property values containing this one string.'
              if route == 'response_format' else
              'Call capture_string exactly once with values containing this one string. '
              'This is a new capture request; do not answer with ordinary text.')
    return (action + '\nCopy the exact decoded value of the following JSON string literal. '
            'Decode the literal once; preserve every character, including whitespace and punctuation. '
            'Do not trim, translate, or change escaping into literal backslash letters. '
            f'The decoded value has {len(target)} Unicode code points.\n'
            + json.dumps(target, ensure_ascii=False))


def request_body(args, route, target, stream, history):
    schema = string_schema(target)
    body = {'model': args.model, 'messages': history, 'stream': stream,
            'temperature': 0, 'max_tokens': MAX_TOKENS, 'parallel_tool_calls': False}
    if route == 'response_format':
        body.update(tools=[], tool_choice='none', response_format={'type': 'json_schema',
                    'json_schema': {'name': 'bounded_strings', 'strict': True, 'schema': schema}})
    else:
        body['tools'] = [{'type': 'function', 'function': {'name': 'capture_string',
            'description': 'Fixture echo tool: capture the supplied strings exactly; no external action.',
            'strict': True, 'parameters': schema}}]
        body['tool_choice'] = ({'type': 'function', 'function': {'name': 'capture_string'}}
                               if route == 'named' else 'auto' if route == 'structured_auto' else 'required')
        if route == 'structured_auto':
            body['tool_routing'] = 'structured'
    if stream:
        body['stream_options'] = {'include_usage': True}
    return body


def parse_response(record):
    raw = record['response_body_utf8']
    if record['stream']:
        record['events'] = []
        for line in raw.splitlines():
            if line.startswith('data:') and line[5:].strip() != '[DONE]':
                event = strict_json(line[5:])
                record['events'].append(event)
                require(isinstance(event, dict) and 'error' not in event, 'SSE error or invalid event')
        parsed = assemble_sse(raw)
        record['sse_done'] = parsed['sse_done']
        return parsed['message'], parsed['finish_reason'], parsed['usage']
    response = strict_json(raw)
    record['response'] = response
    require(isinstance(response, dict) and 'error' not in response, 'Invalid completion or server error')
    require(response.get('object') == 'chat.completion', 'Unexpected completion object')
    choices = response.get('choices')
    require(isinstance(choices, list) and len(choices) == 1 and choices[0].get('index') == 0,
            'Expected exactly one completion choice at index zero')
    return choices[0]['message'], choices[0]['finish_reason'], response.get('usage')


def validate(record, target, cache):
    message, finish, usage = parse_response(record)
    record.update(message=message, finish_reason=finish, usage=usage)
    require(isinstance(message, dict) and message.get('role') == 'assistant', 'Missing assistant message')
    calls = message.get('tool_calls') or []
    if record['route'] == 'response_format':
        require(not calls and finish == 'stop', 'Response format did not finish as ordinary content')
        require(isinstance(message.get('content'), str), 'Missing string response content')
        value = strict_json(message['content'])
    else:
        require(finish == 'tool_calls' and isinstance(calls, list) and len(calls) == 1,
                'Expected exactly one completed tool call')
        require(message.get('content') in (None, ''), 'Tool route also returned ordinary content')
        call = calls[0]
        require(call.get('type') == 'function' and isinstance(call.get('id'), str) and call['id'],
                'Tool call has no function type or nonempty ID')
        function = call['function']
        require(function.get('name') == 'capture_string', 'Wrong tool name')
        require(isinstance(function.get('arguments'), str), 'Tool arguments are not a JSON string')
        value = strict_json(function['arguments'])
    require(isinstance(value, dict) and set(value) == {'values'}, 'Wrong decoded object properties')
    values = value['values']
    require(isinstance(values, list) and len(values) == 1 and isinstance(values[0], str),
            'Expected one string in values')
    actual = values[0]
    actual.encode('utf-8', 'strict')
    record.update(decoded_value=actual, decoded_codepoints=len(actual))
    require(actual == target and len(actual) == len(target),
            f'Exact decoded value/length mismatch: expected {target!r}/{len(target)}, got {actual!r}/{len(actual)}')
    require(isinstance(usage, dict), 'Missing actual usage receipt')
    prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens')
    require(type(prompt) is int and prompt > 0 and type(completion) is int and completion > 0,
            'Missing positive prompt/completion token counts')
    require(type(cached) is int and 0 <= cached <= prompt, 'Invalid cached token count')
    if cache == 'off':
        require(cached == 0, 'Cache-off lane reused cached tokens')
    elif record['cache_hit_expected']:
        require(0 < cached < prompt, 'Cache-on continuation did not reuse its actual prompt prefix')
    return message, value


def run_case(args, url, route, label, target, stream, history, cache_hit_expected):
    body = request_body(args, route, target, stream, history)
    encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    record = {'name': f'{route}-{label}-' + ('sse' if stream else 'json'), 'status': 'FAIL',
              'route': route, 'stream': stream, 'expected_value': target, 'expected_codepoints': len(target),
              'cache_hit_expected': cache_hit_expected, 'request': body, 'request_body_utf8': encoded.decode('utf-8')}
    connection = (http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection)(
        url.hostname, url.port, timeout=TIMEOUT_SECONDS)
    path = url.path.rstrip('/')
    endpoint = path + ('/chat/completions' if path.endswith('/v1') else '/v1/chat/completions')
    record['endpoint'] = endpoint
    started = time.monotonic()
    result = None
    try:
        connection.request('POST', endpoint, body=encoded, headers={'Content-Type': 'application/json',
            'Accept': 'text/event-stream' if stream else 'application/json', 'Connection': 'close'})
        response = connection.getresponse()
        record.update(http_status=response.status, response_headers=response.getheaders())
        try:
            raw = response.read()
        except http.client.IncompleteRead as error:
            record['partial_response_base64'] = base64.b64encode(error.partial).decode('ascii')
            raise
        record['response_body_base64'] = base64.b64encode(raw).decode('ascii')
        record['response_body_utf8'] = raw.decode('utf-8', 'strict')
        require(response.status == 200, f'Expected HTTP 200, received {response.status}')
        expected_type = 'text/event-stream' if stream else 'application/json'
        require(response.getheader('Content-Type', '').split(';')[0].strip() == expected_type,
                'Unexpected response Content-Type')
        result = validate(record, target, args.cache)
        record['status'] = 'PASS'
    except Exception as error:
        record['failure'] = f'{type(error).__name__}: {error}'
    finally:
        connection.close()
        record['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return record, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', choices=('on', 'off'), required=True)
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    require(url.scheme in ('http', 'https') and url.hostname and not url.query and not url.fragment,
            'Expected an HTTP(S) base URL without query or fragment')
    records = []
    for route in ROUTES:
        payloads = PAYLOADS + (('empty', ''),) if route == 'response_format' else PAYLOADS
        for label, target in payloads:
            prompt = copy_prompt(route, target)
            history = [{'role': 'system', 'content': 'Copy the exact supplied string using the requested schema.'},
                       {'role': 'user', 'content': prompt}]
            prior = None
            for stream in (False, True):
                record, parsed = run_case(args, url, route, label, target, stream, history,
                                          args.cache == 'on' and stream and prior is not None)
                records.append(record)
                if parsed is not None and not stream:
                    message, value = parsed
                    history = history + [message]
                    if route != 'response_format':
                        # Local fixture echo handler: return exactly the received arguments.
                        echo = {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'],
                                'content': json.dumps(value, ensure_ascii=False, separators=(',', ':'))}
                        record['fixture_echo_result'] = echo
                        history = history + [echo]
                    history = history + [{'role': 'user', 'content': prompt}]
                    prior = record
    passed = len(records) == 18 and all(row['status'] == 'PASS' for row in records)
    result = {'status': 'PASS' if passed else 'FAIL', 'model': args.model, 'cache': args.cache,
              'base_url': args.base_url, 'expected_cases': 18, 'records': records,
              'limits': {'max_tokens': MAX_TOKENS, 'http_timeout_seconds': TIMEOUT_SECONDS},
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in (Path(__file__).name, 'test_hip_oga_agent_api.py')}}
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / 'result.json'
    # Escaped artifact encoding also preserves invalid surrogate data on failed cases.
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    print(f"{result['status']}: string bounds {sum(row['status'] == 'PASS' for row in records)}/18; {output}")
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
