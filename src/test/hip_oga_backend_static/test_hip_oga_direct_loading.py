#!/usr/bin/env python3
"""Three real HTTP requests: direct directory, OGA load failure, recovery.

The owning controller supplies qualified good/bad directories and server logs.
Use --failure-http-status 200 only to characterize the pre-fix loading revision;
the corrected API must use the default 500.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


def run(args):
    url = urlsplit(args.base_url)
    records = []
    result = dict(status='FAIL', records=records,
                  expectedFailureHttpStatus=args.failure_http_status,
                  testSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def call(name, model):
        body = dict(model=model, backend='hip', stream=False, temperature=0,
                    max_tokens=16, messages=[{'role': 'user', 'content': 'List five prime numbers.'}])
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=90)
        record = dict(name=name, request=body)
        records.append(record)
        started = time.monotonic()
        try:
            conn.request('POST', '/v1/chat/completions', json.dumps(body),
                         {'Content-Type': 'application/json', 'Connection': 'close'})
            response = conn.getresponse()
            record['status'] = response.status
            record['data'] = json.loads(response.read().decode('utf-8'))
        finally:
            record['seconds'] = round(time.monotonic() - started, 3)
            conn.close()
        return record

    def content(record):
        assert record['status'] == 200 and 'error' not in record['data'], record
        text = record['data']['choices'][0]['message']['content']
        assert text and record['data']['usage']['completion_tokens'] > 0, record
        return text

    try:
        baseline = content(call('direct-directory-baseline', args.good_directory))
        failed = call('oga-factory-load-failure', args.bad_directory)
        assert failed['status'] == args.failure_http_status, failed
        assert failed['data'].get('error'), failed
        assert 'Failed to load' in json.dumps(failed['data']['error']), failed
        assert content(call('direct-directory-recovery', args.good_directory)) == baseline, records
        if args.failure_http_status == 200:
            result['knownDefect'] = 'In-load failure returns HTTP 200 with an error body; T04 API repair pending.'
        result['status'] = 'PASS'
    except Exception as error:
        result['failure'] = repr(error)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-url', 'good-directory', 'bad-directory', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--failure-http-status', type=int, choices=(200, 500), default=500)
    raise SystemExit(run(parser.parse_args()))
