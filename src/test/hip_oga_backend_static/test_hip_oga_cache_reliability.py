#!/usr/bin/env python3
"""Real HTTP cache qualification; execute only against a caller-owned test server.

Catches stale KV after history edits, context exhaustion, wrong hit accounting,
and cache-dependent greedy output. The on lane replays exact off-lane requests.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


class Run:
    def __init__(self, args):
        self.args = args
        self.records = []

    def call(self, name, messages, cache='miss', status=200, **fields):
        body = dict(model=self.args.model, messages=copy.deepcopy(messages),
                    temperature=0, max_tokens=8, stream=False)
        body.update(fields)
        return self.request(name, body, cache, status)

    def request(self, name, body, cache, status):
        record = dict(name=name, request=body, expected_cache=cache,
                      expected_status=status)
        self.records.append(record)
        start = time.monotonic()
        req = urllib.request.Request(self.args.base_url + '/v1/chat/completions',
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                record.update(status=response.status, raw=response.read().decode())
        except urllib.error.HTTPError as error:
            record.update(status=error.code, raw=error.read().decode())
        finally:
            record['seconds'] = round(time.monotonic() - start, 3)
        assert record['status'] == status, record
        if status != 200:
            assert 'error' in json.loads(record['raw']), record
            return record
        if body['stream']:
            data = []
            for line in record['raw'].splitlines():
                if line.startswith('data: ') and line[6:] != '[DONE]':
                    data.append(json.loads(line[6:]))
            assert '[DONE]' in record['raw'], record
            content = ''.join(c.get('delta', {}).get('content', '') or ''
                              for chunk in data for c in chunk.get('choices', []))
            reasons = [c['finish_reason'] for chunk in data
                       for c in chunk.get('choices', []) if c.get('finish_reason')]
            usages = [chunk['usage'] for chunk in data if chunk.get('usage')]
            assert usages and reasons, record
            usage, reason = usages[-1], reasons[-1]
        else:
            data = json.loads(record['raw'])
            assert 'error' not in data, record
            choice = data['choices'][0]
            content, reason = choice['message']['content'], choice['finish_reason']
            usage = data['usage']
        cached = usage['prompt_tokens_details']['cached_tokens']
        prompt, generated = usage['prompt_tokens'], usage['completion_tokens']
        record.update(content=content, reason=reason, prompt_tokens=prompt,
                      completion_tokens=generated, cached_tokens=cached)
        assert 0 <= cached <= prompt < self.args.context, record
        assert 0 < generated <= min(body['max_tokens'], self.args.context-prompt), record
        assert usage['total_tokens'] == prompt + generated, record
        should_hit = self.args.cache == 'on' and cache == 'hit'
        assert (cached > 0) == should_hit, record
        return record

    def scenarios(self):
        messages = []
        for turn in range(24):
            messages.append({'role':'user', 'content':f'Key {turn}: blue. Reply OK.'})
            result = self.call(f'grow-{turn:02d}', messages,
                               'miss' if turn == 0 else 'hit', stream=turn % 5 == 4)
            messages.append({'role':'assistant', 'content':result['content']})
        # Edits to content already present in the previous checkpoint must miss.
        user_edit = copy.deepcopy(messages)
        user_edit[0]['content'] = 'Key 0: green. Reply OK.'
        user_edit.append({'role':'user', 'content':'What is key zero?'})
        answer = self.call('edited-user', user_edit)
        user_edit += [{'role':'assistant','content':answer['content']},
                      {'role':'user','content':'Reply OK.'}]
        answer = self.call('after-user-edit', user_edit, 'hit')
        assistant_edit = copy.deepcopy(user_edit)
        assistant_edit[1]['content'] = 'Different earlier answer.'
        assistant_edit += [{'role':'assistant','content':answer['content']},
                           {'role':'user','content':'Reply OK.'}]
        self.call('edited-assistant', assistant_edit)
        short = messages[:9]
        answer = self.call('shortened-history', short)
        short += [{'role':'assistant','content':answer['content']},
                  {'role':'user','content':'What is key zero?'}]
        self.call('after-shortening', short, 'hit')
        self.call('repeat-prompt', short)
        new = [{'role':'user','content':'Write the word ocean.'}]
        answer = self.call('unrelated-conversation', new)
        new += [{'role':'assistant','content':answer['content']},
                {'role':'user','content':'Write it again.'}]
        answer = self.call('changed-sampling', new, repetition_penalty=1.1)
        new += [{'role':'assistant','content':answer['content']},
                {'role':'user','content':'Write it again.'}]
        self.call('same-sampling', new, 'hit', repetition_penalty=1.1)
        # Measure template overhead once; then verify exact prompt-token counts.
        def boundary(n):
            return [{'role':'user','content':'Repeat x:' + ' x' * n}]
        base = self.call('boundary-calibration', boundary(8), max_tokens=1)
        count = 8 + self.args.context - 1 - base['prompt_tokens']
        near = self.call('boundary-1023', boundary(count), max_tokens=8)
        assert near['prompt_tokens'] == self.args.context - 1, near
        assert near['completion_tokens'] == 1 and near['reason'] == 'length', near
        self.call('boundary-1024', boundary(count+1), status=400)
        self.call('boundary-1025', boundary(count+2), status=400, stream=True)
        recovery = [{'role':'user','content':'Write the word ocean.'}]
        answer = self.call('after-overflow', recovery)
        recovery += [{'role':'assistant','content':answer['content']},
                     {'role':'user','content':'Write it again.'}]
        self.call('after-overflow-follow', recovery, 'hit')

    def replay(self, path):
        reference = json.loads(path.read_text(encoding='utf-8-sig'))
        assert reference['status'] == 'PASS', 'Off lane must pass before replay'
        for expected in reference['records']:
            actual = self.request(expected['name'], expected['request'],
                                  expected['expected_cache'], expected['expected_status'])
            if actual['status'] == 200:
                for key in ('content','reason','prompt_tokens','completion_tokens'):
                    assert actual[key] == expected[key], (key, expected, actual)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--cache', choices=['off','on'], required=True)
    p.add_argument('--context', type=int, default=1024)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run, failure = Run(args), None
    try:
        run.replay(args.reference) if args.reference else run.scenarios()
    except Exception as error:
        failure = repr(error)
    result = dict(status='FAIL' if failure else 'PASS', model=args.model,
                  cache=args.cache, context=args.context,
                  records=run.records, failure=failure)
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'status':result['status'],'requests':len(run.records),
                      'failure':failure},ensure_ascii=False))
    return int(failure is not None)


if __name__ == '__main__':
    raise SystemExit(main())
