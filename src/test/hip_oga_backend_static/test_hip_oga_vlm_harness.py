#!/usr/bin/env python3
"""Black-box HTTP contract checks for the VLM qualification harness."""
import base64
import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
OUTPUT = tempfile.TemporaryDirectory(prefix='vlm-harness-guards-')
GUARDS = Path(OUTPUT.name)
HARNESS = ROOT / 'test_hip_oga_vlm_api.py'
MODEL = 'fake-gemma'

spec = importlib.util.spec_from_file_location('vlm_probe_draft', HARNESS)
vlm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlm)
RED_BLUE = vlm.make_split_png((255, 0, 0), (0, 0, 255))
BLUE_RED = vlm.make_split_png((0, 0, 255), (255, 0, 0))
ASSET_HASHES = {
    'left-red-right-blue': hashlib.sha256(RED_BLUE).hexdigest(),
    'left-blue-right-red': hashlib.sha256(BLUE_RED).hexdigest(),
}
USAGE = {
    'prompt_tokens': 16,
    'completion_tokens': 1,
    'total_tokens': 17,
    'prompt_tokens_details': {'cached_tokens': 0},
}


def json_bytes(value):
    return json.dumps(value, separators=(',', ':')).encode('utf-8')


def completion(content, finish='stop', usage=None):
    return {
        'id': 'chatcmpl-fake',
        'object': 'chat.completion',
        'model': MODEL,
        'choices': [{'index': 0,
                     'message': {'role': 'assistant', 'content': content,
                                 'reasoning_content': ''},
                     'finish_reason': finish}],
        'usage': usage or dict(USAGE),
    }


def chunk(content='', finish=None, usage=None):
    choice = {'index': 0, 'delta': {'content': content}, 'finish_reason': finish}
    result = {'id': 'chatcmpl-fake', 'object': 'chat.completion.chunk',
              'model': MODEL, 'choices': [choice]}
    if usage is not None:
        result['usage'] = usage
    return b'data: ' + json_bytes(result) + b'\n\n'


class FakeState:
    def __init__(self, scenario, cache_mode):
        self.scenario = scenario
        self.cache_mode = cache_mode
        self.lock = threading.Lock()
        self.active = {}
        self.audit = []
        self.after_cancel = False
        self.after_corrupt = False
        self.last_color = None
        self.red_color_request_count = 0
        self.red_history_request_count = 0

    def add_audit(self, item):
        with self.lock:
            self.audit.append(item)

    def route_cancel(self, request_id):
        with self.lock:
            event = self.active.get(request_id)
            if event is not None:
                event.set()
                return True
            return False

    def cached_tokens_for(self, body, image_hash):
        if self.cache_mode != 'on' or self.scenario == 'zero-cache':
            return 0
        messages = body.get('messages', [])
        if (image_hash == ASSET_HASHES['left-red-right-blue'] and len(messages) == 3 and
                messages[-1].get('content') ==
                'What color is the RIGHT half of the original image? Answer with one color word.'):
            self.red_history_request_count += 1
            return 4 if self.red_history_request_count == 1 else 0
        if (self.scenario == 'unexpected-positive-repeat' and
                image_hash == ASSET_HASHES['left-red-right-blue'] and len(messages) == 1 and
                messages[0].get('content') == [
                    {'type': 'text', 'text': vlm.COLOR_QUESTION},
                    {'type': 'image_url', 'image_url': {
                        'url': 'data:image/png;base64,' + base64.b64encode(RED_BLUE).decode('ascii')}}]):
            self.red_color_request_count += 1
            return 1 if self.red_color_request_count == 3 else 0
        return 0


class FakeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, payload, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _image_hash(self, body):
        for message in body.get('messages', []):
            content = message.get('content')
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get('type') != 'image_url':
                    continue
                url = part.get('image_url', {}).get('url', '')
                if not url.startswith('data:image/png;base64,'):
                    continue
                raw = base64.b64decode(url.split(',', 1)[1])
                return hashlib.sha256(raw).hexdigest(), raw
        return None, None

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        body = json.loads(self.rfile.read(length))
        image_hash, image_bytes = self._image_hash(body)
        state = self.server.state
        audit = {'path': self.path, 'stream': bool(body.get('stream')),
                 'image_sha256': image_hash,
                 'request_id': body.get('request_id')}
        state.add_audit(audit)

        if self.path == '/api/cancel':
            accepted = state.route_cancel(body.get('request_id'))
            audit['cancelled'] = accepted
            self._send(200, json_bytes({'cancelled': accepted}))
            return

        if self.path != '/v1/chat/completions':
            self._send(404, json_bytes({'error': 'unexpected route'}))
            return

        if image_hash is not None and image_bytes == b'\x89PNG\r\n\x1a\nnot-a-valid-png-payload':
            state.after_corrupt = True
            if state.scenario == 'full-http-200-corrupt':
                self._send(200, json_bytes({'error': {
                    'type': 'invalid_request_error', 'code': 400,
                    'message': 'corrupt image rejected by fake server'}}))
            else:
                self._send(400, json_bytes({'error': {
                    'type': 'invalid_request_error', 'code': 400,
                    'message': 'corrupt image'}}))
            return

        if body.get('stream'):
            if state.scenario == 'malformed-sse' and image_hash == ASSET_HASHES['left-red-right-blue'] and not body.get('request_id'):
                self._send(200, b'event: message\ndata: {bad json}\n\n', 'text/event-stream')
                return
            if state.scenario == 'late-sse-data' and image_hash == ASSET_HASHES['left-red-right-blue'] and not body.get('request_id'):
                valid = chunk('red') + chunk('', 'stop', dict(USAGE)) + b'data: [DONE]\n\n'
                late = chunk('blue')
                payload = valid + late
                self._send(200, payload, 'text/event-stream')
                return
            if body.get('request_id'):
                request_id = body['request_id']
                cancelled = threading.Event()
                with state.lock:
                    state.active[request_id] = cancelled
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    self.wfile.write(chunk('red '))
                    self.wfile.flush()
                    cancelled.wait(5)
                    cancel_usage = dict(USAGE)
                    self.wfile.write(chunk('', 'cancel', cancel_usage))
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                finally:
                    with state.lock:
                        state.active.pop(request_id, None)
                        state.after_cancel = True
                return
            self._send(200, chunk('red') + chunk('', 'stop', dict(USAGE)) + b'data: [DONE]\n\n',
                       'text/event-stream')
            return

        if image_hash == ASSET_HASHES['left-blue-right-red']:
            answer = 'blue'
        elif image_hash == ASSET_HASHES['left-red-right-blue']:
            answer = 'blue' if state.scenario == 'wrong-image-answer' else 'red'
        else:
            messages = body.get('messages', [])
            answer = 'READY' if any(m.get('content') == 'Say only the word READY.' for m in messages) else 'red'
        history = body.get('messages', [])
        if len(history) > 1:
            if state.scenario == 'drop-history-image':
                answer = history[-2].get('content', 'red')
            elif 'RIGHT' in history[-1].get('content', ''):
                answer = 'red' if answer == 'blue' else 'blue'
        elif state.after_cancel and state.scenario == 'stale-cancel-image' and image_hash:
            answer = 'red'
        elif state.after_corrupt and state.scenario == 'stale-error-image' and image_hash:
            answer = state.last_color
        if image_hash:
            state.last_color = answer
        usage = json.loads(json.dumps(USAGE))
        usage['prompt_tokens_details']['cached_tokens'] = state.cached_tokens_for(body, image_hash)
        if state.scenario in ('boolean-cache', 'float-cache') and not body.get('stream') and not image_hash:
            usage['prompt_tokens_details']['cached_tokens'] = True if state.scenario == 'boolean-cache' else 0.5
        self._send(200, json_bytes(completion(answer, usage=usage)))


class FakeHttpServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, scenario, cache_mode='on'):
        super().__init__(('127.0.0.1', 0), FakeHandler)
        self.state = FakeState(scenario, cache_mode)


class HarnessGuard(unittest.TestCase):
    def run_harness(self, name, scenario, phase='smoke', cache='on'):
        case_dir = GUARDS / name
        case_dir.mkdir(parents=True, exist_ok=True)
        server = FakeHttpServer(scenario, cache_mode=cache)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            command = [sys.executable, str(HARNESS), '--base-url',
                       'http://127.0.0.1:%d' % server.server_address[1],
                       '--model', MODEL, '--output', str(case_dir / 'output'),
                       '--cache', cache, '--phase', phase]
            if scenario == 'full-http-200-corrupt':
                command += ['--invalid-image-http-status', '400']
            proc = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=35)
            (case_dir / 'stdout.log').write_text(proc.stdout[:4000], encoding='utf-8')
            (case_dir / 'stderr.log').write_text(proc.stderr[:8000], encoding='utf-8')
            (case_dir / 'server-audit.json').write_text(
                json.dumps(server.state.audit, indent=2), encoding='utf-8')
            result_path = case_dir / 'output' / 'result.json'
            self.assertTrue(result_path.is_file(), 'harness did not retain result.json')
            result = json.loads(result_path.read_text(encoding='utf-8'))
            return proc, result, server.state.audit
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


    def test_history_image_loss_must_fail(self):
        proc, result, _ = self.run_harness('drop-history-image', 'drop-history-image', phase='full')
        self.assertEqual(proc.returncode, 1, 'lost historical image passed')
        self.assertEqual(result['cases'][-1]['name'], 'image-red-history-followup')

    def test_declared_history_cache_hit_rejects_zero_cached_tokens(self):
        proc, result, _ = self.run_harness('zero-cache-declared-hit', 'zero-cache', phase='full')
        self.assertEqual(proc.returncode, 1, 'zero cached tokens passed the declared history hit')
        self.assertEqual(result['cases'][-1]['name'], 'image-red-history-followup')
        self.assertIn('expected cache hit', result['cases'][-1]['failure'])

    def test_identical_red_repeat_rejects_unexpected_positive_cache(self):
        proc, result, _ = self.run_harness('positive-identical-repeat', 'unexpected-positive-repeat', phase='full')
        self.assertEqual(proc.returncode, 1, 'unexpected cached tokens passed on the identical prompt repeat')
        self.assertEqual(result['cases'][-1]['name'], 'image-red-identical-repeat')
        self.assertIn('expected cache miss', result['cases'][-1]['failure'])

    def test_cache_off_requires_misses_even_for_declared_hit(self):
        proc, result, _ = self.run_harness('cache-off', 'success', phase='full', cache='off')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['case_count'], 14)

    def test_cached_token_count_rejects_boolean_and_float(self):
        for scenario in ('boolean-cache', 'float-cache'):
            with self.subTest(scenario=scenario):
                proc, result, _ = self.run_harness(scenario, scenario)
                self.assertEqual(proc.returncode, 1)
                self.assertEqual(result['cases'][-1]['name'], 'text-baseline')
                self.assertIn('cached_tokens must be an integer', result['cases'][-1]['failure'])

    def test_cancel_stale_image_must_fail(self):
        proc, result, _ = self.run_harness('stale-cancel-image', 'stale-cancel-image', phase='full')
        self.assertEqual(proc.returncode, 1, 'stale cancelled image passed')
        self.assertEqual(result['cases'][-1]['name'], 'image-recovery-after-cancel')

    def test_error_stale_image_must_fail(self):
        proc, result, _ = self.run_harness('stale-error-image', 'stale-error-image', phase='full')
        self.assertEqual(proc.returncode, 1, 'stale pre-error image passed')
        self.assertEqual(result['cases'][-1]['name'], 'image-recovery-after-corrupt')

    def test_smoke_success_text_colors_and_repeat(self):
        proc, result, audit = self.run_harness('smoke-success', 'success')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['case_count'], 4)
        self.assertEqual([case['status'] for case in result['cases']], ['PASS'] * 4)
        self.assertEqual([case['name'] for case in result['cases']],
                         ['text-baseline', 'image-red', 'image-blue', 'image-red-repeat'])
        self.assertEqual(len(audit), 4)

    def test_wrong_image_answer_fails(self):
        proc, result, audit = self.run_harness('wrong-image-answer', 'wrong-image-answer')
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['case_count'], 2)
        self.assertEqual(result['cases'][0]['status'], 'PASS')
        self.assertEqual(result['cases'][1]['status'], 'FAIL')
        self.assertIn('expected whole color word', result['cases'][1]['failure'])
        self.assertEqual(len(audit), 2)

    def test_full_cancel_recovery_and_http_400_pass(self):
        proc, result, audit = self.run_harness('full-cancel-recovery', 'success', phase='full')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['case_count'], 14)
        self.assertTrue(all(case['status'] == 'PASS' for case in result['cases']))
        self.assertEqual([case['name'] for case in result['cases']], [
            'text-baseline', 'image-red', 'image-blue', 'image-red-repeat',
            'image-red-identical-repeat', 'image-red-history-followup', 'image-red-history-repeat',
            'image-blue-history-followup', 'image-red-sse-parity', 'active-image-stream-cancel',
            'image-recovery-after-cancel', 'invalid-image-corrupt-png',
            'image-recovery-after-corrupt', 'text-recovery-after-corrupt'])
        self.assertEqual(result['case_names_expected'], [case['name'] for case in result['cases']])
        self.assertEqual(result['case_count_expected'], 14)
        records = {record['name']: record for record in result['records']}
        self.assertEqual({name: record['expected_cache'] for name, record in records.items()
                          if 'expected_cache' in record}, {
            'text-baseline': 'miss', 'image-red': 'miss', 'image-blue': 'miss',
            'image-red-repeat': 'miss', 'image-red-identical-repeat': 'miss',
            'image-red-history-followup': 'hit', 'image-red-history-repeat': 'miss',
            'image-blue-history-followup': 'miss', 'image-red-sse-parity': 'miss',
            'image-red-cancel-stream': 'miss', 'image-recovery-after-cancel': 'miss',
            'image-recovery-after-corrupt': 'miss', 'text-recovery-after-corrupt': 'miss'})
        self.assertEqual(records['image-red-history-followup']['request']['messages'],
                         records['image-red-history-repeat']['request']['messages'])
        self.assertEqual(records['image-red-history-followup']['usage']['prompt_tokens_details']['cached_tokens'], 4)
        self.assertEqual(records['image-red-history-repeat']['usage']['prompt_tokens_details']['cached_tokens'], 0)
        cancel_case = next(case for case in result['cases'] if case['name'] == 'active-image-stream-cancel')
        self.assertTrue(cancel_case['accepted_cancel'])
        self.assertTrue(cancel_case['observed_content_before_cancel'])
        self.assertGreaterEqual(cancel_case['cancel_poll_count'], 1)
        self.assertEqual(result['records'][-1]['name'], 'text-recovery-after-corrupt')
        self.assertTrue(any(item.get('cancelled') for item in audit if item['path'] == '/api/cancel'))

    def test_full_http_200_corrupt_response_remains_failure(self):
        proc, result, audit = self.run_harness('full-http-200-corrupt', 'full-http-200-corrupt', phase='full')
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(result['knownDefects'])
        self.assertEqual(result['case_count'], 14)
        corrupt = next(case for case in result['cases'] if case['name'] == 'invalid-image-corrupt-png')
        self.assertEqual(corrupt['status'], 'KNOWN_DEFECT')
        self.assertEqual(result['cases'][-1]['status'], 'PASS')
        self.assertTrue(any(item['image_sha256'] is not None and item['path'] == '/v1/chat/completions'
                            for item in audit))

    def test_malformed_sse_framing_fails(self):
        proc, result, audit = self.run_harness('malformed-sse', 'malformed-sse', phase='full')
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['cases'][-1]['name'], 'image-red-sse-parity')
        self.assertEqual(result['cases'][-1]['status'], 'FAIL')
        self.assertIn('unexpected SSE line', result['cases'][-1]['failure'])
        self.assertFalse(any(item.get('request_id') for item in audit))

    def test_sse_data_after_done_fails(self):
        proc, result, audit = self.run_harness('late-sse-data', 'late-sse-data', phase='full')
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['cases'][-1]['name'], 'image-red-sse-parity')
        self.assertEqual(result['cases'][-1]['status'], 'FAIL')
        self.assertIn('SSE data followed [DONE]', result['cases'][-1]['failure'])
        self.assertTrue(audit)


if __name__ == '__main__':
    GUARDS.mkdir(parents=True, exist_ok=True)
    unittest.main(verbosity=2)
