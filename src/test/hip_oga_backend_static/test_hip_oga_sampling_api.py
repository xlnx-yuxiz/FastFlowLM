#!/usr/bin/env python3
"""Bounded HTTP regression against an already running HIP OGA test server.

This executes real model requests. Start the isolated server on the assigned test host first.
The harness retains requests, responses and expected settings for log readback.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def parse_success_response(endpoint, streaming, raw):
    """Return generated text and cached-token count for each real API framing."""
    if endpoint in ('/v1/chat/completions', '/v1/completions'):
        is_chat = endpoint == '/v1/chat/completions'

        def read_event(value, content):
            event = json.loads(value)
            if not isinstance(event, dict) or 'error' in event:
                raise ValueError('error or malformed SSE event')
            choices = event.get('choices')
            if not isinstance(choices, list) or not choices:
                raise ValueError('SSE event has no choices')
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError('SSE choice is malformed')
                if is_chat:
                    delta = choice.get('delta', {})
                    if not isinstance(delta, dict):
                        raise ValueError('SSE delta is malformed')
                    part = delta.get('content')
                else:
                    part = choice.get('text')
                if part is not None:
                    if not isinstance(part, str):
                        raise ValueError('SSE content is not a string')
                    content.append(part)

        if streaming:
            chunks, done = [], False
            for line in raw.splitlines():
                if not line:
                    continue
                if not line.startswith('data: ') or done:
                    raise ValueError('malformed or trailing SSE data')
                value = line[6:]
                if value == '[DONE]':
                    done = True
                else:
                    read_event(value, chunks)
            if not done:
                raise ValueError('SSE stream did not finish')
            content = ''.join(chunks)
            cached = 0
        else:
            data = json.loads(raw)
            if not isinstance(data, dict) or 'error' in data:
                raise ValueError('error or malformed JSON response')
            choices = data.get('choices')
            if not isinstance(choices, list) or not choices:
                raise ValueError('JSON response has no choices')
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ValueError('JSON choice is malformed')
            if is_chat:
                message = choice.get('message')
                if not isinstance(message, dict):
                    raise ValueError('JSON response has no assistant message')
                value = message.get('content')
            else:
                value = choice.get('text')
            if not isinstance(value, str):
                raise ValueError('JSON response content is not a string')
            content = value
            cached = (data.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens', 0)
                      if isinstance(data.get('usage'), dict) else 0)
    elif endpoint in ('/api/chat', '/api/generate'):
        if streaming:
            events = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict) or 'error' in event:
                    raise ValueError('error or malformed NDJSON event')
                if endpoint == '/api/chat':
                    if not isinstance(event.get('message'), dict):
                        raise ValueError('NDJSON event has no message')
                    part = event['message'].get('content', '')
                else:
                    part = event.get('response')
                    if not isinstance(part, str):
                        raise ValueError('NDJSON response content is not a string')
                if not isinstance(part, str):
                    raise ValueError('NDJSON content is not a string')
                event['_response_part'] = part
                events.append(event)
            if not events or events[-1].get('done') is not True:
                raise ValueError('NDJSON stream did not finish')
            content = ''.join(event['_response_part'] for event in events)
            cached = 0
        else:
            data = json.loads(raw)
            if not isinstance(data, dict) or 'error' in data or data.get('done') is not True:
                raise ValueError('error or unfinished JSON response')
            if endpoint == '/api/chat':
                message = data.get('message')
                if not isinstance(message, dict) or not isinstance(message.get('content'), str):
                    raise ValueError('JSON response has no assistant message')
                content = message['content']
            else:
                content = data.get('response')
                if not isinstance(content, str):
                    raise ValueError('JSON response content is not a string')
            cached = 0
    else:
        raise ValueError(f'unsupported response endpoint: {endpoint}')
    if not content.strip():
        raise ValueError('empty generated content')
    return content, cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', choices=['on', 'off'], required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))['search']
    defaults = {k: config[k] for k in ('do_sample', 'temperature', 'top_k', 'top_p', 'repetition_penalty')}
    defaults['random_seed'] = config.get('random_seed', -1)
    records, failures = [], []

    def expected(**values):
        result = dict(defaults)
        result.update(values)
        return result

    def request(name, fields, settings=None, status=200, endpoint='/v1/chat/completions', messages=None,
                runtime=False):
        if endpoint == '/v1/completions':
            body = {'model': args.model, 'prompt': 'Name a color.', 'max_tokens': 16, 'stream': False}
        elif endpoint == '/api/generate':
            body = {'model': args.model, 'prompt': 'Name a color.', 'stream': False, 'options': {'num_predict': 16}}
        elif endpoint == '/api/chat':
            body = {'model': args.model, 'messages': messages or [{'role': 'user', 'content': 'Name a few colors in one sentence.'}],
                    'stream': False, 'options': {'num_predict': 16}}
        else:
            body = {'model': args.model,
                    'messages': messages or [{'role': 'user', 'content': 'Name a few colors in one sentence.'}],
                    'max_tokens': 16, 'stream': False}
        body.update(copy.deepcopy(fields))
        if endpoint in ('/api/chat', '/api/generate'):
            body.setdefault('options', {}).setdefault('num_predict', 16)
        payload = json.dumps(body).encode()
        started = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(args.base_url + endpoint, data=payload,
                    headers={'Content-Type': 'application/json'}), timeout=90) as response:
                actual, raw = response.status, response.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            actual, raw = e.code, e.read().decode('utf-8')
        record = {'name': name, 'endpoint': endpoint, 'request': body, 'http_status': actual,
                  'expected_http_status': status, 'elapsed_seconds': round(time.monotonic()-started, 3),
                  'raw_response': raw, 'expected_settings': settings,
                  'expected_runtime_settings': None}
        if runtime and settings is not None:
            record['expected_runtime_settings'] = {
                'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'repetition_penalty': 1.0,
                'do_sample': settings['do_sample'], 'random_seed': settings['random_seed']}
        records.append(record)
        if actual != status:
            failures.append(f'{name}: HTTP {actual}, expected {status}')
            return record
        if status == 400:
            try:
                data = json.loads(raw)
                error = data.get('error') if isinstance(data, dict) else None
                if endpoint == '/api/chat':
                    valid_error = isinstance(error, str) and bool(error.strip()) or isinstance(error, dict) and bool(error)
                else:
                    valid_error = isinstance(error, dict) and error.get('code') == 400
                if not valid_error:
                    failures.append(f'{name}: missing endpoint error payload')
                record['error'] = error
            except (ValueError, TypeError) as error:
                failures.append(f'{name}: invalid error response: {error}')
            return record
        try:
            record['content'], record['cached_tokens'] = parse_success_response(endpoint, body['stream'], raw)
        except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            failures.append(f'{name}: invalid successful response: {error}')
        return record

    try:
        request('min_p_only', {'min_p': 0.25}, expected(min_p=0.25), runtime=True)
        if args.full:
            for endpoint, label in (('/v1/chat/completions', 'chat'), ('/v1/completions', 'completion'),
                                    ('/api/chat', 'api_chat'), ('/api/generate', 'api_generate')):
                for streaming in (False, True):
                    name = f'min_p_endpoint_{label}_stream' if streaming else f'min_p_endpoint_{label}_json'
                    fields = {'stream': streaming, 'min_p': 0.2}
                    if endpoint in ('/api/chat', '/api/generate'):
                        fields = {'stream': streaming, 'options': {'min_p': 0.2}}
                    request(name, fields, expected(min_p=0.2), endpoint=endpoint, runtime=True)
            invalid = [('temperature_negative', 'temperature', -1), ('temperature_bool', 'temperature', True),
                       ('top_k_fraction', 'top_k', 1.5), ('top_k_large', 'top_k', 2147483647),
                       ('top_p_above_one', 'top_p', 1.1), ('seed_fraction', 'seed', 1.5),
                       ('seed_large', 'seed', 2147483648), ('repetition_zero', 'repetition_penalty', 0),
                       ('min_p_negative', 'min_p', -0.1), ('min_p_above_one', 'min_p', 1.1),
                       ('min_p_bool', 'min_p', True), ('min_p_string', 'min_p', '0.25'),
                       ('min_p_nan', 'min_p', float('nan')), ('min_p_infinity', 'min_p', float('inf')),
                       ('frequency_bool', 'frequency_penalty', True), ('frequency_string', 'frequency_penalty', '0.5'),
                       ('frequency_nan', 'frequency_penalty', float('nan')),
                       ('frequency_infinity', 'frequency_penalty', float('inf')),
                       ('presence_bool', 'presence_penalty', True), ('presence_string', 'presence_penalty', '-0.5'),
                       ('presence_nan', 'presence_penalty', float('nan')),
                       ('presence_infinity', 'presence_penalty', float('inf')), ('do_sample_integer', 'do_sample', 1)]
            for name, key, value in invalid:
                request(f'invalid_{name}', {key: value}, status=400, endpoint='/v1/chat/completions')
            # Also reject invalid sampler values through the Ollama option map.
            for name, key, value in (('min_p_outside', 'min_p', 1.1),
                                     ('presence_type', 'presence_penalty', 'bad'),
                                     ('frequency_nonfinite', 'frequency_penalty', float('nan'))):
                request(f'invalid_api_chat_{name}', {'options': {key: value}}, status=400, endpoint='/api/chat')
        request('settings_a', {'temperature': 0.5, 'top_k': 3, 'top_p': 0.5, 'repetition_penalty': 1.25, 'seed': 42},
                expected(temperature=0.5, top_k=3, top_p=0.5, repetition_penalty=1.25, random_seed=42, do_sample=True))
        request('defaults_after_a', {}, expected())
        request('neutral_options', {'min_p': 0, 'presence_penalty': 0, 'frequency_penalty': 0}, expected())
        request('top_level_precedence', {'options': {'temperature': 1.5, 'top_k': 3, 'top_p': 0.5, 'min_p': 0.25},
                'temperature': 0, 'min_p': 0}, expected(temperature=1.0, top_k=3, top_p=0.5, do_sample=False))

        request('positive_presence_penalty', {'presence_penalty': 0.4}, expected(presence_penalty=0.4), runtime=True)
        request('positive_frequency_penalty', {'frequency_penalty': 0.3}, expected(frequency_penalty=0.3), runtime=True)
        request('negative_presence_penalty', {'presence_penalty': -0.4}, expected(presence_penalty=-0.4), runtime=True)
        request('negative_frequency_penalty', {'frequency_penalty': -0.3}, expected(frequency_penalty=-0.3), runtime=True)

        messages = [{'role': 'user', 'content': 'Remember the word pebble. Reply briefly.'}]
        first = request('greedy_first', {'temperature': 0}, expected(temperature=1.0, do_sample=False), messages=messages)
        history = copy.deepcopy(messages) + [{'role': 'assistant', 'content': first.get('content', '')},
                                             {'role': 'user', 'content': 'Which word did I ask you to remember?'}]
        follow = request('greedy_follow', {'temperature': 0}, expected(temperature=1.0, do_sample=False), messages=history)
        if args.cache == 'on' and follow.get('cached_tokens', 0) <= 0:
            failures.append('greedy_follow: experimental cache did not reuse the first prompt')
        if args.cache == 'off' and follow.get('cached_tokens', 0) != 0:
            failures.append('greedy_follow: disabled cache reported cached tokens')

        seeded = {'temperature': 1.25, 'top_k': 20, 'top_p': 0.875, 'seed': 123}
        want_seeded = expected(temperature=1.25, top_k=20, top_p=0.875, random_seed=123, do_sample=True)
        one = request('seeded_first', seeded, want_seeded)
        two = request('seeded_repeat', seeded, want_seeded)
        if one.get('content') != two.get('content'):
            failures.append('seeded_repeat: identical seed/request did not repeat')
        if args.cache == 'on' and two.get('cached_tokens', 0) != 0:
            failures.append('seeded_repeat: RNG reset requires a fresh generator')
        seeded_history = [{'role': 'user', 'content': 'Name a few colors in one sentence.'},
                          {'role': 'assistant', 'content': two.get('content', '')},
                          {'role': 'user', 'content': 'Name one more color.'}]
        seeded_follow = request('seeded_follow', seeded, want_seeded, messages=seeded_history)
        if seeded_follow.get('cached_tokens', 0) != 0:
            failures.append('seeded_follow: history continuation must reset the seeded RNG')

        pure_top_p = {'top_k': 0, 'top_p': 0.75, 'seed': 123}
        want_pure_top_p = expected(top_k=0, top_p=0.75, random_seed=123, do_sample=True)
        pure_first = request('pure_top_p_seeded_first', pure_top_p, want_pure_top_p, runtime=True)
        pure_repeat = request('pure_top_p_seeded_repeat', pure_top_p, want_pure_top_p, runtime=True)
        if pure_first.get('content') != pure_repeat.get('content'):
            failures.append('pure_top_p_seeded_repeat: identical seed/request did not repeat')
        for record in (pure_first, pure_repeat):
            if record.get('cached_tokens', 0) != 0:
                failures.append(f'{record["name"]}: seeded pure-top-p request reused cached tokens')

        extended_greedy = {'temperature': 0, 'min_p': 0.2, 'presence_penalty': 0.15, 'frequency_penalty': -0.2}
        want_extended_greedy = expected(temperature=1.0, do_sample=False, min_p=0.2,
                                        presence_penalty=0.15, frequency_penalty=-0.2)
        extended_messages = [{'role': 'user', 'content': 'Remember the word quartz. Reply briefly.'}]
        extended_first = request('extended_greedy_first', extended_greedy, want_extended_greedy,
                                 messages=extended_messages, runtime=True)
        extended_history = copy.deepcopy(extended_messages) + [
            {'role': 'assistant', 'content': extended_first.get('content', '')},
            {'role': 'user', 'content': 'Which word did I ask you to remember?'}]
        extended_follow = request('extended_greedy_follow', extended_greedy, want_extended_greedy,
                                  messages=extended_history, runtime=True)
        if args.cache == 'on' and extended_follow.get('cached_tokens', 0) <= 0:
            failures.append('extended_greedy_follow: experimental cache did not reuse the first prompt')
        if args.cache == 'off' and extended_follow.get('cached_tokens', 0) != 0:
            failures.append('extended_greedy_follow: disabled cache reported cached tokens')

        extended_seeded = {'top_k': 0, 'top_p': 0.75, 'seed': 321,
                           'presence_penalty': 0.2, 'frequency_penalty': -0.1}
        want_extended_seeded = expected(top_k=0, top_p=0.75, random_seed=321, do_sample=True,
                                        presence_penalty=0.2, frequency_penalty=-0.1)
        extended_json = request('extended_seeded_nonstream', extended_seeded, want_extended_seeded, runtime=True)
        extended_stream = request('extended_seeded_stream', {**extended_seeded, 'stream': True},
                                  want_extended_seeded, runtime=True)
        if extended_json.get('content') != extended_stream.get('content'):
            failures.append('extended_seeded_stream: streaming text differs for the same seeded options')

        schema = {'type': 'json_schema', 'json_schema': {'name': 'color_choice', 'strict': True,
                  'schema': {'type': 'object', 'properties': {'color': {'type': 'string',
                           'enum': ['red', 'blue', 'green']}}, 'required': ['color'], 'additionalProperties': False}}}
        schema_options = {'min_p': 0.1, 'top_k': 0, 'top_p': 0.8,
                          'presence_penalty': -0.1, 'frequency_penalty': -0.1}
        schema_settings = expected(top_k=0, top_p=0.8, min_p=0.1,
                                   presence_penalty=-0.1, frequency_penalty=-0.1, do_sample=True)
        schema_record = request('json_schema_color_enum', {'response_format': schema, **schema_options},
                                schema_settings, runtime=True,
                                messages=[{'role': 'user', 'content': 'Return one color as JSON.'}])
        try:
            parsed = json.loads(schema_record.get('content', ''))
            if not isinstance(parsed, dict) or set(parsed) != {'color'} or parsed['color'] not in ('red', 'blue', 'green'):
                raise ValueError('JSON does not match the color enum schema')
            schema_record['validated_json'] = parsed
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            failures.append(f'json_schema_color_enum: invalid JSON output: {error}')

        request('defaults_after_extensions', {}, expected())
        plain = request('plain_greedy', {'temperature': 0, 'max_tokens': 8}, expected(temperature=1.0, do_sample=False))
        stream = request('stream_greedy', {'temperature': 0, 'max_tokens': 8, 'stream': True}, expected(temperature=1.0, do_sample=False))
        if plain.get('content') != stream.get('content'):
            failures.append('stream_greedy: streaming text differs from the nonstream greedy request')
        request('defaults_after_seed', {}, expected())
    except Exception as e:
        failures.append(f'{type(e).__name__}: {e}')
    result = {'status': 'PASS' if not failures else 'FAIL', 'model': args.model, 'cache': args.cache,
              'full': args.full, 'records': records, 'failures': failures,
              'case_names': [record['name'] for record in records]}
    (args.output / 'api-result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'requests': len(records), 'failures': failures}))
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
