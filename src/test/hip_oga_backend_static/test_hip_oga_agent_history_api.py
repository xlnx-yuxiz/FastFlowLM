#!/usr/bin/env python3
"""HTTP contract checks for OpenAI assistant tool-call history admission."""
import argparse
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


ECHO_TOKEN = 'TOOL_HISTORY_RESULT_ECHO_7Q4'
ALPHA_TOKEN = 'TOOL_HISTORY_RESULT_ALPHA_9M2'
BETA_TOKEN = 'TOOL_HISTORY_RESULT_BETA_5K8'


def tool_call(call_id, name='lookup', arguments='{"value":"ok"}'):
    return {
        'id': call_id,
        'type': 'function',
        'function': {'name': name, 'arguments': arguments},
    }


def assistant_calls(calls, content_marker='omitted'):
    message = {'role': 'assistant', 'tool_calls': calls}
    if content_marker == 'null':
        message['content'] = None
    return message


def resolved_echo_history(content_marker):
    return [
        {'role': 'user', 'content': 'Call the lookup function and report its returned value.'},
        assistant_calls([tool_call('call_echo', arguments='{"key":"record"}')],
                        content_marker),
        {'role': 'tool', 'tool_call_id': 'call_echo', 'content': ECHO_TOKEN},
        {'role': 'user', 'content': 'Echo the exact text returned by the tool.'},
    ]


def negative_histories():
    call = tool_call('call_known')
    return {
        'orphan_role_tool': [
            {'role': 'user', 'content': 'Continue.'},
            {'role': 'tool', 'tool_call_id': 'call_orphan', 'content': 'orphan result'},
        ],
        'unknown_tool_call_id': [
            {'role': 'user', 'content': 'Continue.'},
            assistant_calls([call]),
            {'role': 'tool', 'tool_call_id': 'call_unknown', 'content': 'unknown result'},
        ],
        'missing_tool_call_id': [
            {'role': 'user', 'content': 'Continue.'},
            assistant_calls([call]),
            {'role': 'tool', 'content': 'result without call ID'},
        ],
        'duplicate_tool_result': [
            {'role': 'user', 'content': 'Continue.'},
            assistant_calls([call]),
            {'role': 'tool', 'tool_call_id': 'call_known', 'content': 'first result'},
            {'role': 'tool', 'tool_call_id': 'call_known', 'content': 'duplicate result'},
        ],
        'unresolved_call_before_next_user': [
            {'role': 'user', 'content': 'Continue.'},
            assistant_calls([call]),
            {'role': 'user', 'content': 'A new user turn arrived before the tool result.'},
        ],
        'unresolved_call_at_end': [
            {'role': 'user', 'content': 'Continue.'},
            assistant_calls([call]),
        ],
        'duplicate_call_ids_in_batch': [
            {'role': 'user', 'content': 'Continue.'},
            assistant_calls([
                tool_call('call_duplicate', arguments='{"value":"first"}'),
                tool_call('call_duplicate', arguments='{"value":"second"}'),
            ]),
            {'role': 'tool', 'tool_call_id': 'call_duplicate', 'content': 'ambiguous result'},
        ],
        'malformed_arguments_json': [
            {'role': 'user', 'content': 'Continue.'},
            assistant_calls([tool_call('call_bad_json', arguments='{"value":}')]),
            {'role': 'tool', 'tool_call_id': 'call_bad_json', 'content': 'result'},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--cache', choices=('on', 'off'))
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()

    url = urlsplit(args.base_url)
    if url.scheme not in ('http', 'https') or not url.hostname:
        parser.error('--base-url must be an http(s) URL')
    endpoint = url.path.rstrip('/') + '/v1/chat/completions'
    connection_type = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
    records, failures = [], []

    def check(record, passed, message):
        result = {'passed': bool(passed), 'message': message}
        record.setdefault('checks', []).append(result)
        if not passed:
            failure = {'kind': 'assertion', 'request': record.get('name'), 'message': message}
            failures.append(failure)

    def request(name, messages, *, stream=False, expected_status=200, valid_control=False):
        body = {
            'model': args.model,
            'messages': messages,
            'temperature': 0,
            'max_tokens': 64,
            'stream': stream,
        }
        if valid_control:
            # Closed tool history is legal after tools are no longer offered.
            body['tool_choice'] = 'none'
        encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        record = {
            'name': name,
            'endpoint': endpoint,
            'stream': stream,
            'request': body,
            'request_body_utf8': encoded.decode('utf-8'),
            'expected_http_status': expected_status,
            'checks': [],
        }
        records.append(record)

        conn = connection_type(url.hostname, url.port, timeout=args.timeout)
        started = time.monotonic()
        try:
            conn.request('POST', endpoint, body=encoded, headers={
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream' if stream else 'application/json',
                'Connection': 'close',
            })
            response = conn.getresponse()
            raw = response.read().decode('utf-8', 'replace')
            record['http_status'] = response.status
            record['response_headers'] = response.getheaders()
            record['raw_response'] = raw
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
            record['transport_error'] = failure
            failures.append({'kind': 'transport', 'request': name, 'message': failure})
            return record
        finally:
            conn.close()
            record['elapsed_seconds'] = round(time.monotonic() - started, 3)

        check(record, record['http_status'] == expected_status,
              f'HTTP {record["http_status"]}, expected {expected_status}')
        raw = record['raw_response']
        headers = {str(k).lower(): str(v) for k, v in record.get('response_headers', [])}

        if expected_status == 400:
            content_type = headers.get('content-type', '').lower()
            check(record, 'text/event-stream' not in content_type,
                  f'error Content-Type must not be event-stream, got {content_type!r}')
            check(record, not any(line.startswith('data:') for line in raw.splitlines()),
                  'error response must not contain SSE data lines')
            try:
                data = json.loads(raw)
                record['response_json'] = data
                error = data.get('error', {}) if isinstance(data, dict) else {}
                check(record, isinstance(error, dict) and
                      error.get('type') == 'invalid_request_error',
                      f'expected JSON invalid_request_error, got {error!r}')
            except (json.JSONDecodeError, AttributeError) as error:
                check(record, False, f'expected JSON error response: {error}')
            return record

        if stream:
            events, done, content, finishes = [], False, [], []
            for line in raw.splitlines():
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    continue
                try:
                    event = json.loads(data)
                    events.append(event)
                    for choice in event.get('choices', []):
                        delta = choice.get('delta') or {}
                        if delta.get('content'):
                            content.append(delta['content'])
                        if choice.get('finish_reason'):
                            finishes.append(choice['finish_reason'])
                except (json.JSONDecodeError, AttributeError) as error:
                    check(record, False, f'malformed SSE event: {error}')
            record.update(sse_events=events, sse_done=done,
                          content=''.join(content),
                          finish_reason=finishes[-1] if finishes else None)
            check(record, done and bool(events), 'expected SSE events followed by [DONE]')
        else:
            try:
                data = json.loads(raw)
                record['response_json'] = data
                choice = data['choices'][0]
                message = choice['message']
                record['message'] = message
                record['content'] = message.get('content') or ''
                record['finish_reason'] = choice.get('finish_reason')
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                check(record, False, f'malformed chat completion response: {error}')
                return record

        if valid_control:
            check(record, bool(record.get('content', '').strip()),
                  'valid history control must return nonempty assistant content')
            check(record, record.get('finish_reason') == 'stop',
                  f'valid history control must finish with stop, got {record.get("finish_reason")!r}')
        return record

    histories = negative_histories()
    if args.phase == 'smoke':
        selected = {
            name: histories[name]
            for name in ('orphan_role_tool', 'unknown_tool_call_id', 'malformed_arguments_json')
        }
    else:
        selected = histories

    for name, messages in selected.items():
        for stream in (False, True):
            request(name + ('_stream' if stream else '_json'), messages,
                    stream=stream, expected_status=400)

    if args.phase == 'full':
        for marker in ('null', 'omitted'):
            # The fixed tool-return token is in history and intentionally absent
            # from the final user question.
            history = resolved_echo_history(marker)
            question = history[-1]['content']
            request('valid_resolved_call_content_' + marker, history,
                    valid_control=True)
            records[-1]['control_token'] = ECHO_TOKEN
            check(records[-1], records[-1].get('content', '').strip() == ECHO_TOKEN,
                  'resolved tool result must reach the assistant answer')
            check(records[-1], ECHO_TOKEN not in question,
                  'control token must be absent from the final user question')

        reversed_history = [
            {'role': 'user', 'content': 'Acknowledge the completed lookups.'},
            assistant_calls([
                tool_call('call_alpha', arguments='{"value":"' + ALPHA_TOKEN + '"}'),
                tool_call('call_beta', arguments='{"value":"' + BETA_TOKEN + '"}'),
            ]),
            {'role': 'tool', 'tool_call_id': 'call_beta', 'content': BETA_TOKEN},
            {'role': 'tool', 'tool_call_id': 'call_alpha', 'content': ALPHA_TOKEN},
            {'role': 'user', 'content': 'Briefly acknowledge the completed lookups.'},
        ]
        request('valid_two_calls_results_reversed_by_id', reversed_history,
                valid_control=True)
        records[-1]['control_tokens'] = [ALPHA_TOKEN, BETA_TOKEN]
        check(records[-1], all(token not in reversed_history[-1]['content']
                               for token in (ALPHA_TOKEN, BETA_TOKEN)),
              'control tokens must be absent from the final user question')
    else:
        history = resolved_echo_history('omitted')
        request('valid_resolved_call_content_omitted', history, valid_control=True)
        records[-1]['control_token'] = ECHO_TOKEN
        check(records[-1], records[-1].get('content', '').strip() == ECHO_TOKEN,
              'resolved tool result must reach the assistant answer')
        check(records[-1], ECHO_TOKEN not in history[-1]['content'],
              'control token must be absent from the final user question')

    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        'status': 'PASS' if not failures else 'FAIL',
        'phase': args.phase,
        'model': args.model,
        'cache': args.cache,
        'request_count': len(records),
        'failure_count': len(failures),
        'failures': failures,
        'records': records,
    }
    (args.output / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in
                      ('status', 'phase', 'model', 'request_count', 'failures')},
                     ensure_ascii=False))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
