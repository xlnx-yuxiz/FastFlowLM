#!/usr/bin/env python3
"""Real HTTP isolation/cancellation regression; run against a caller-owned test server.

Catches missing admission, cancellation ID replacement, token reset, incomplete
endpoint coverage, and leaked state after cancelled or rejected requests.
"""
import argparse
import concurrent.futures
import http.client
import json
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit


class Probe:
    def __init__(self, args):
        self.args = args
        self.url = urlsplit(args.base_url)
        self.records = []
        self.mutex = threading.Lock()

    def body(self, endpoint='/v1/chat/completions', **fields):
        body = dict(model=self.args.model, stream=False, max_tokens=16,
                    temperature=0, options={'num_predict': 16})
        if endpoint in ('/api/generate', '/v1/completions'):
            body['prompt'] = 'The first five prime numbers are'
        else:
            body['messages'] = [{'role': 'user', 'content': 'List five prime numbers.'}]
        body.update(fields)
        return body

    def call(self, name, endpoint, body=None, first=None, disconnect=False):
        conn = http.client.HTTPConnection(self.url.hostname, self.url.port, timeout=90)
        record = dict(name=name, endpoint=endpoint, request=body)
        started = time.monotonic()
        try:
            conn.request('POST' if body is not None else 'GET', endpoint,
                         json.dumps(body) if body is not None else None,
                         {'Content-Type': 'application/json', 'Connection': 'close'})
            response = conn.getresponse()
            record.update(status=response.status, headers=dict(response.getheaders()))
            chunks, content, reasons = [], [], []
            streaming = bool(body and body.get('stream') and response.status == 200)
            if streaming:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    text = line.decode('utf-8').strip()
                    chunks.append(text)
                    if text.startswith('data: '):
                        text = text[6:]
                    if not text or text == '[DONE]':
                        continue
                    data = json.loads(text)
                    piece = data.get('response') or data.get('message', {}).get('content', '')
                    reason = data.get('done_reason')
                    for choice in data.get('choices', []):
                        piece += choice.get('delta', {}).get('content') or choice.get('text') or ''
                        reason = choice.get('finish_reason') or reason
                    if reason:
                        reasons.append(reason)
                    if piece:
                        content.append(piece)
                        if first:
                            first.set()
                        if disconnect:
                            # Force TCP close while the real model is still generating.
                            response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                            break
                record.update(raw='\n'.join(chunks), content=''.join(content), reasons=reasons)
            else:
                raw = response.read().decode('utf-8')
                record['raw'] = raw
                data = json.loads(raw)
                record['data'] = data
                if isinstance(data, dict):
                    record['content'] = data.get('response') or (data.get('message', {}).get('content', '') if isinstance(data.get('message', {}), dict) else '')
                    for choice in data.get('choices', []):
                        record['content'] += choice.get('message', {}).get('content') or choice.get('text') or ''
                    record['reasons'] = [c['finish_reason'] for c in data.get('choices', []) if c.get('finish_reason')]
                    if data.get('done_reason'):
                        record['reasons'].append(data['done_reason'])
        except Exception as exc:
            record['exception'] = repr(exc)
        finally:
            conn.close()
            record['seconds'] = round(time.monotonic() - started, 3)
            with self.mutex:
                self.records.append(record)
        return record

    def ok(self, record, status=200):
        assert record.get('status') == status and 'exception' not in record, record
        if status == 200:
            assert 'error' not in record.get('data', {}), record
        return record

    def busy(self, record):
        self.ok(record, 503)
        assert record['data']['error']['type'] == 'engine_busy', record
        assert record['headers'].get('Retry-After') == '1', record
        assert record['seconds'] < 3, record

    def long_body(self, endpoint, request_id):
        body = self.body(endpoint, stream=True, max_tokens=600,
                         options={'num_predict': 600}, request_id=request_id)
        prompt = 'Count from 1 to 1000. Put every number on its own line and write its English name. Do not skip any numbers. Begin now.'
        if 'prompt' in body:
            body['prompt'] = prompt
        else:
            body['messages'] = [{'role': 'user', 'content': prompt}]
        return body

    def healthy(self, name, expected=None):
        deadline = time.monotonic() + 15
        while True:
            record = self.call(name, '/v1/chat/completions', self.body())
            if record.get('status') != 503 or time.monotonic() >= deadline:
                break
            time.sleep(.05)
        self.ok(record)
        assert record.get('content'), record
        if expected is not None:
            assert record['content'] == expected, (expected, record)
        return record['content']

    def active(self, pool, endpoint, request_id):
        first = threading.Event()
        future = pool.submit(self.call, request_id, endpoint,
                             self.long_body(endpoint, request_id), first)
        assert first.wait(30), 'No generated stream content: ' + request_id
        assert not future.done(), 'Stream ended before overlap: ' + request_id
        return future

    def cancel(self, request_id):
        record = self.ok(self.call('cancel-' + request_id, '/api/cancel', {'request_id': request_id}))
        assert record['data']['cancelled'] is True, record
        assert record['seconds'] < 3, record

    def cancel_when_active(self, request_id, future):
        # Poll the cancellation registry, which does not take engine admission.
        # Polling /api/ps could briefly own the gate and reject our test request.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            record = self.ok(self.call('cancel-' + request_id, '/api/cancel', {'request_id': request_id}))
            if record['data']['cancelled']:
                assert record['seconds'] < 3, record
                return
            assert not future.done(), future.result() if future.done() else 'Request ended'
            time.sleep(.02)
        raise AssertionError('Request was never registered: ' + request_id)

    def cancelled(self, future):
        record = self.ok(future.result(timeout=20))
        assert 'cancel' in record.get('reasons', []), record

    def run(self):
        endpoint = '/v1/chat/completions'
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            if self.args.case == 'cancel':
                future = self.active(pool, endpoint, 'baseline-cancel')
                self.cancel('baseline-cancel')
                self.cancelled(future)
                return
            if self.args.case == 'overlap':
                future = self.active(pool, endpoint, 'baseline-overlap')
                self.busy(self.call('overlap-B', endpoint, self.body()))
                self.cancel('baseline-overlap')
                self.cancelled(future)
                return

            for invalid_id in ('', 17):
                self.ok(self.call('invalid-request-id', endpoint, self.body(request_id=invalid_id)), 400)
            reference = self.healthy('serial-reference')
            future = self.active(pool, endpoint, 'owner')
            duplicate = self.call('duplicate-owner', endpoint, self.body(request_id='owner'))
            self.ok(duplicate, 409)
            for path in ('/api/generate', '/api/chat', '/v1/completions', endpoint):
                for streaming in (False, True):
                    self.busy(self.call('busy-' + path + str(streaming), path,
                                        self.body(path, stream=streaming, model=self.args.other_model)))
            self.busy(self.call('busy-backend-switch', endpoint, self.body(backend='npu')))
            self.busy(self.call('busy-ps', '/api/ps'))
            burst = [pool.submit(self.call, 'burst-' + str(i), endpoint, self.body()) for i in range(4)]
            for item in burst:
                self.busy(item.result(timeout=5))
            for path in ('/api/version', '/v1/models', '/api/npu/status'):
                record = self.ok(self.call('responsive-' + path, path))
                assert record['seconds'] < 3, record
            unknown = self.ok(self.call('cancel-unknown', '/api/cancel', {'request_id': 'missing'}))
            assert unknown['data']['cancelled'] is False, unknown
            assert not future.done(), 'Owner ended before isolation assertions'
            self.cancel('owner')
            self.cancelled(future)
            self.healthy('after-cancel', reference)

            # Both response formats and streaming modes must use the same token.
            for path in ('/api/generate', '/api/chat', '/v1/completions', endpoint):
                request_id = 'cancel-' + path
                future = self.active(pool, path, request_id)
                self.cancel(request_id)
                self.cancelled(future)
                self.healthy('after-' + request_id, reference)
                body = self.long_body(path, request_id + '-nonstream')
                body['stream'] = False
                future = pool.submit(self.call, request_id + '-nonstream', path, body)
                self.cancel_when_active(request_id + '-nonstream', future)
                self.cancelled(future)
                self.healthy('after-nonstream-' + path, reference)

            # Switching models includes a 500 ms load boundary before prefill. Cancel
            # after admission, while no generated token has been observed.
            early = self.long_body(endpoint, 'cancel-during-load')
            early['model'] = self.args.other_model
            first = threading.Event()
            future = pool.submit(self.call, 'cancel-during-load', endpoint, early, first)
            self.cancel_when_active('cancel-during-load', future)
            assert not first.is_set(), 'First token preceded model-load cancellation'
            self.cancelled(future)
            self.healthy('after-load-cancel', reference)

            self.ok(self.call('disconnect', endpoint, self.long_body(endpoint, 'disconnect'), disconnect=True))
            self.healthy('after-disconnect', reference)
            # Replacement must be safe after owner release; switch back and compare.
            self.ok(self.call('switch-other', endpoint, self.body(model=self.args.other_model)))
            self.healthy('switch-back', reference)
            self.ok(self.call('invalid-sampling', endpoint, self.body(min_p=1.3)), 400)
            self.healthy('after-error', reference)
            self.ok(self.call('generate-invalid-sampling', '/api/generate',
                              self.body('/api/generate', min_p=1.3)), 400)
            self.healthy('after-generate-error', reference)
            self.ok(self.call('failed-model-switch', endpoint,
                              self.body(model='t02-missing-model:invalid')), 500)
            self.healthy('after-failed-switch', reference)
            self.ok(self.call('reuse-completed-id', endpoint, self.body(request_id='owner')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--other-model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=['cancel', 'overlap', 'full'], default='full')
    args = parser.parse_args()
    probe = Probe(args)
    result = {'case': args.case, 'status': 'PASS'}
    try:
        probe.run()
    except Exception as exc:
        result.update(status='FAIL', error=str(exc))
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'requests.json').write_text(json.dumps(probe.records, ensure_ascii=False, indent=2), encoding='utf-8')
        result['requests'] = len(probe.records)
        (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(result['status'] != 'PASS')


if __name__ == '__main__':
    main()
