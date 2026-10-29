#!/usr/bin/env python3
"""Reconcile every real prefill diagnostic with HTTP usage and paired output."""
import argparse
import json
from pathlib import Path
import re
import shlex


def check_lane_identity(name, result, receipt):
    match=re.fullmatch(r'(gemma|qwen)-(oga|flm|compare)-(off|on)',name)
    assert match, f'Unexpected lane name: {name}'
    model,codec,cache=match.groups()
    tag={'gemma':'hip-gemma3-text-strict:4b',
         'qwen':'hip-qwen25-coder-strict:14b'}[model]
    assert result['model']==tag and result['cache']==cache, 'Result model/cache differs from lane'
    assert result['context']==1024, 'Result context differs from qualification contract'
    assert receipt['lane']==name and receipt['codec']==codec and receipt['cache']==cache, 'Receipt labels differ from lane'
    arguments=shlex.split(receipt['arguments'])
    assert arguments[:2]==['serve',tag], 'Server model invocation differs from lane'

    def option(arguments, flag):
        assert arguments.count(flag)==1, f'Expected one {flag} in invocation'
        index=arguments.index(flag)
        assert index+1<len(arguments), f'Missing value for {flag}'
        return arguments[index+1]

    assert option(arguments,'--backend')=='hip', 'Server backend differs'
    assert option(arguments,'--ctx-len')=='1024', 'Server context differs'
    assert option(arguments,'--hip-oga-text-codec')==codec, 'Server codec differs from lane'
    test_arguments=receipt['testArguments']
    assert option(test_arguments,'--model')==tag, 'Harness model differs from lane'
    assert option(test_arguments,'--cache')==cache, 'Harness cache mode differs from lane'
    assert option(test_arguments,'--base-url')==f"http://127.0.0.1:{option(arguments,'--port')}", 'Harness/server endpoints differ'
    assert all(r['request']['model']==tag for r in result['records']), 'Request model differs from lane'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    lanes, failures, binaries={},[],set()
    pattern=re.compile(r'HIP OGA prefix cache attempted=(\d) hit=(\d) fallback=(\S+) cached_tokens=(\d+) newly_prefilled_tokens=(\d+) checkpoint_tokens=(\d+)')
    for path in sorted(args.root.glob('*/result.json')):
        d=json.loads(path.read_text(encoding='utf-8-sig'))
        if 'records' not in d or 'cache' not in d: continue
        name=path.parent.name; lanes[name]=d
        try:
            assert d['status']=='PASS', d.get('failure')
            receipt=json.loads((path.parent/'receipt.json').read_text(encoding='utf-8-sig'))
            assert receipt['exitCode']==0, receipt
            check_lane_identity(name,d,receipt)
            binaries.add(receipt['binarySha256'])
            expected_names={f'grow-{i:02d}' for i in range(24)} | {
                'edited-user','after-user-edit','edited-assistant','shortened-history',
                'after-shortening','repeat-prompt','unrelated-conversation','changed-sampling',
                'same-sampling','boundary-calibration','boundary-1023','boundary-1024',
                'boundary-1025','after-overflow','after-overflow-follow'}
            assert len(d['records'])==39 and {r['name'] for r in d['records']}==expected_names
            logs='\n'.join((path.parent/f).read_text(encoding='utf-8-sig',errors='replace') for f in ('stdout.log','stderr.log'))
            observations=[dict(zip(('attempted','hit','reason','cached','new','checkpoint'),m.groups())) for m in pattern.finditer(logs)]
            requests=[r for r in d['records'] if r['status']==200]
            assert len(observations)==len(requests), (len(observations),len(requests))
            for r,o in zip(requests,observations):
                assert int(o['cached'])==r['cached_tokens'], (r['name'],o,r)
                assert int(o['cached'])+int(o['new'])==r['prompt_tokens'], (r['name'],o,r)
                assert int(o['checkpoint'])==r['prompt_tokens'], (r['name'],o,r)
                assert (o['hit']=='1')==(d['cache']=='on' and r['expected_cache']=='hit'), (r['name'],o)
                if d['cache']=='off':
                    assert o['reason']=='disabled', (r['name'],o)
                elif r['name']=='changed-sampling':
                    assert o['reason']=='signature_changed', (r['name'],o)
                r['diagnostic']=o
            (path.parent/'accounting.json').write_text(json.dumps(observations,indent=2))
        except Exception as error:
            failures.append(f'{name}: {error!r}')
    models=('gemma',) if args.smoke else ('gemma','qwen')
    codecs=('oga',) if args.smoke else ('oga','flm','compare')
    for model in models:
        for codec in codecs:
            pair=[lanes.get(f'{model}-{codec}-{cache}') for cache in ('off','on')]
            if not all(pair): failures.append(f'Missing pair {model}/{codec}'); continue
            left,right=pair
            if len(left['records'])!=len(right['records']): failures.append(f'Incomplete pair {model}/{codec}')
            for a,b in zip(left['records'],right['records']):
                keys=['name','request','status']+(['content','reason','prompt_tokens','completion_tokens'] if a.get('status')==200 else [])
                for key in keys:
                    if a.get(key)!=b.get(key): failures.append(f'{model}/{codec}/{a["name"]}: {key} differs')
    if len(binaries)!=1: failures.append(f'Expected one binary identity, found {sorted(binaries)}')
    result=dict(status='FAIL' if failures else 'PASS',lanes=len(lanes),binarySha256=sorted(binaries),
                requests=sum(len(d['records']) for d in lanes.values()),failures=failures,
                hits=sum(r.get('cached_tokens',0)>0 for d in lanes.values() for r in d['records']))
    (args.root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result)); return int(bool(failures))


if __name__=='__main__':
    raise SystemExit(main())
