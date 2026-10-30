#!/usr/bin/env python3
"""Real chat consumer parity, SSE/usage and unsupported-tool recovery checks."""
import argparse
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


def run(args):
    url = urlsplit(args.base_url)
    records = []
    result = dict(status='FAIL', model=args.model, cache=args.cache, records=records)
    messages = [{'role': 'user', 'content': 'Say hello to 世界. Give a short greeting.'}]

    def call(name, stream=False, expected_status=200, **extra):
        body = dict(model=args.model, stream=stream, max_tokens=24,
                    temperature=0, messages=messages)
        body.update(extra)
        record = dict(name=name, request=body)
        records.append(record)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=120)
        start = time.monotonic()
        try:
            connection.request('POST', '/v1/chat/completions', json.dumps(body, ensure_ascii=False).encode('utf-8'),
                               {'Content-Type': 'application/json', 'Connection': 'close'})
            response = connection.getresponse()
            record.update(status=response.status, raw=response.read().decode('utf-8'))
        finally:
            connection.close()
            record['seconds'] = round(time.monotonic() - start, 3)
        assert record['status'] == expected_status, record
        if expected_status != 200:
            data = json.loads(record['raw'])
            assert data['error']['code'] == expected_status, record
            assert data['error']['type'] == 'invalid_request_error', record
            return record
        if stream:
            events = []
            done = False
            for line in record['raw'].splitlines():
                if not line:
                    continue
                assert line.startswith('data: ') and not done, record
                text = line[6:]
                if text == '[DONE]':
                    done = True
                else:
                    events.append(json.loads(text))
            assert done and events, record
            content, reasoning, finishes, usages = [], [], [], []
            ids = set()
            for event in events:
                assert 'error' not in event and event['object'] == 'chat.completion.chunk', record
                assert event['model'] == args.model and len(event['choices']) == 1, record
                ids.add(event['id'])
                choice = event['choices'][0]
                assert choice['index'] == 0 and isinstance(choice['delta'], dict), record
                delta = choice['delta']
                assert not delta.get('tool_calls'), record
                for field, target in [('content', content), ('reasoning_content', reasoning)]:
                    value = delta.get(field)
                    assert value is None or isinstance(value, str), record
                    if value: target.append(value)
                if choice.get('finish_reason'): finishes.append(choice['finish_reason'])
                if event.get('usage'): usages.append(event['usage'])
            assert len(ids) == 1 and next(iter(ids)) and len(finishes) == 1 and len(usages) == 1, record
            assert events[-1]['choices'][0]['finish_reason'] == finishes[0], record
            content, reasoning, finish, usage = ''.join(content), ''.join(reasoning), finishes[0], usages[0]
        else:
            data = json.loads(record['raw'])
            assert 'error' not in data and data['object'] == 'chat.completion', record
            assert data['model'] == args.model and len(data['choices']) == 1, record
            choice = data['choices'][0]
            message = choice['message']
            assert message['role'] == 'assistant' and not message.get('tool_calls'), record
            content, reasoning = message.get('content') or '', message.get('reasoning_content') or ''
            finish, usage = choice['finish_reason'], data['usage']
        prompt, generated = usage['prompt_tokens'], usage['completion_tokens']
        assert 0 < prompt < 1024 and 0 < generated <= body['max_tokens'], record
        assert usage['total_tokens'] == prompt + generated, record
        assert usage['prompt_tokens_details']['cached_tokens'] == 0, record
        assert finish in ('stop', 'length'), record
        record.update(content=content, reasoning=reasoning, finish=finish, usage=usage)
        return record

    def same(actual, expected):
        for key in ('content', 'reasoning', 'finish'):
            assert actual[key] == expected[key], (key, actual, expected)
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            assert actual['usage'][key] == expected['usage'][key], (key, actual, expected)

    try:
        reference = call('unicode-json')
        assert reference['content'], reference
        same(call('unicode-stream', stream=True), reference)
        same(call('after-stream-json'), reference)
        call('bounded-stream', stream=True, max_tokens=1)
        tools = [{'type': 'function', 'function': {'name': 'lookup', 'description': 'Look up a city',
                 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]
        same(call('tools-none-json', tools=tools, tool_choice='none'), reference)
        call('invalid-tool-choice', expected_status=400, tools=tools, tool_choice='invalid')
        same(call('after-rejection-json'), reference)
        result['status'] = 'PASS'
    except Exception as error:
        result['failure'] = repr(error)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-url', 'model', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--cache', choices=('off',), default='off')
    raise SystemExit(run(parser.parse_args()))
