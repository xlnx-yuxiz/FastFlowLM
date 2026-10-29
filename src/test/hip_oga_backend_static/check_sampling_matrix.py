#!/usr/bin/env python3
"""Check real HTTP responses against OGA diagnostic readback and cache lanes."""
import argparse
import json
import math
from pathlib import Path
import re


BASE_CASE_NAMES = (
    'min_p_only', 'settings_a', 'defaults_after_a', 'neutral_options', 'top_level_precedence',
    'positive_presence_penalty', 'positive_frequency_penalty', 'negative_presence_penalty',
    'negative_frequency_penalty', 'greedy_first', 'greedy_follow', 'seeded_first', 'seeded_repeat',
    'seeded_follow', 'pure_top_p_seeded_first', 'pure_top_p_seeded_repeat', 'extended_greedy_first',
    'extended_greedy_follow', 'extended_seeded_nonstream', 'extended_seeded_stream',
    'json_schema_color_enum', 'defaults_after_extensions', 'plain_greedy', 'stream_greedy',
    'defaults_after_seed',
)
FULL_CASE_NAMES = (
    'min_p_endpoint_chat_json', 'min_p_endpoint_chat_stream',
    'min_p_endpoint_completion_json', 'min_p_endpoint_completion_stream',
    'min_p_endpoint_api_chat_json', 'min_p_endpoint_api_chat_stream',
    'min_p_endpoint_api_generate_json', 'min_p_endpoint_api_generate_stream',
)
INVALID_CASE_NAMES = (
    'invalid_temperature_negative', 'invalid_temperature_bool', 'invalid_top_k_fraction',
    'invalid_top_k_large', 'invalid_top_p_above_one', 'invalid_seed_fraction', 'invalid_seed_large',
    'invalid_repetition_zero', 'invalid_min_p_negative', 'invalid_min_p_above_one', 'invalid_min_p_bool',
    'invalid_min_p_string', 'invalid_min_p_nan', 'invalid_min_p_infinity', 'invalid_frequency_bool',
    'invalid_frequency_string', 'invalid_frequency_nan', 'invalid_frequency_infinity', 'invalid_presence_bool',
    'invalid_presence_string', 'invalid_presence_nan', 'invalid_presence_infinity', 'invalid_do_sample_integer',
    'invalid_api_chat_min_p_outside', 'invalid_api_chat_presence_type', 'invalid_api_chat_frequency_nonfinite',
)
CACHE_PARITY_CASE_NAMES = (
    'top_level_precedence', 'settings_a', 'greedy_first', 'greedy_follow', 'plain_greedy', 'stream_greedy',
    'seeded_first', 'seeded_repeat', 'seeded_follow', 'pure_top_p_seeded_first', 'pure_top_p_seeded_repeat',
    'extended_greedy_first', 'extended_greedy_follow', 'extended_seeded_nonstream', 'extended_seeded_stream',
)


def read_diagnostics(lane):
    found = {'effective': [], 'runtime': []}
    ordered = {}
    for stream in ('stdout.log', 'stderr.log'):
        text = (lane / stream).read_text(encoding='utf-8-sig', errors='replace')
        entries = []
        pattern = r'HIP OGA sampling (effective|runtime)=(\{[^\r\n]*\})'
        for match in re.finditer(pattern, text):
            kind = match.group(1)
            value = json.loads(match.group(2))
            found[kind].append(value)
            entries.append((kind, value))
        ordered[stream] = entries
    return found, ordered


def matches(value, want):
    if isinstance(want, bool):
        return value == want
    return isinstance(value, (int, float)) and math.isfinite(value) and math.isclose(
        value, want, rel_tol=1e-6, abs_tol=1e-7)


def diagnostics_follow_requests(records, entries):
    position = 0
    for receipt in records:
        if receipt['expected_settings'] is None:
            continue
        options = [('effective', 'runtime'), ('runtime', 'effective')] if (
            receipt.get('expected_runtime_settings') is not None) else [('effective',)]
        for option in options:
            if tuple(kind for kind, _ in entries[position:position + len(option)]) == option:
                position += len(option)
                break
        else:
            return False
    return position == len(entries)


def readback_failures(records, effective, runtime, lane_name='lane'):
    failures = []
    expected = [receipt for receipt in records if receipt['expected_settings'] is not None]
    if len(effective) != len(expected):
        failures.append(f'{lane_name}: {len(effective)} OGA effective readbacks, expected {len(expected)}')
    for receipt, observed in zip(expected, effective):
        for key, want in receipt['expected_settings'].items():
            if not matches(observed.get(key), want):
                failures.append(f'{lane_name}/{receipt["name"]}: OGA {key}={observed.get(key)}, expected {want}')
    expected_runtime = [receipt for receipt in records if receipt.get('expected_runtime_settings') is not None]
    if len(runtime) != len(expected_runtime):
        failures.append(f'{lane_name}: {len(runtime)} OGA runtime readbacks, expected {len(expected_runtime)}')
    for receipt, observed in zip(expected_runtime, runtime):
        for key, want in receipt['expected_runtime_settings'].items():
            if not matches(observed.get(key), want):
                failures.append(f'{lane_name}/{receipt["name"]}: OGA runtime {key}={observed.get(key)}, expected {want}')
    return failures


def cache_content_failures(left_records, right_records, label):
    left = {record['name']: record.get('content') for record in left_records}
    right = {record['name']: record.get('content') for record in right_records}
    failures = []
    for case in CACHE_PARITY_CASE_NAMES:
        if case in left and case in right and left[case] != right[case]:
            failures.append(f'{label}/{case}: cache on/off content differs')
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    failures, lanes = [], {}
    for result_path in sorted(args.root.glob('*/api-result.json')):
        lane = result_path.parent
        data = json.loads(result_path.read_text(encoding='utf-8-sig'))
        lanes[lane.name] = data
        if data['status'] != 'PASS':
            failures.append(f'{lane.name}: HTTP harness: {data["failures"]}')
        diagnostics, ordered = read_diagnostics(lane)
        effective, runtime = diagnostics['effective'], diagnostics['runtime']
        failures.extend(readback_failures(data['records'], effective, runtime, lane.name))
        expected = [r for r in data['records'] if r['expected_settings'] is not None]
        for receipt, observed in zip(expected, effective):
            if receipt['name'] == 'greedy_follow':
                if observed.get('generator_reused') != (data['cache'] == 'on'):
                    failures.append(f'{lane.name}: greedy generator reuse flag disagrees with cache lane')
            if receipt['name'] == 'extended_greedy_follow':
                if observed.get('generator_reused') != (data['cache'] == 'on'):
                    failures.append(f'{lane.name}: extended greedy generator reuse flag disagrees with cache lane')
            if 'seeded_' in receipt['name'] and observed.get('generator_reused') is not False:
                failures.append(f'{lane.name}: seeded RNG was not reset with a fresh generator')
        for entries in ordered.values():
            if len(entries) == len(effective) + len(runtime) and entries:
                if not diagnostics_follow_requests(data['records'], entries):
                    failures.append(f'{lane.name}: effective/runtime diagnostics do not align with request order')
        case_names = [r['name'] for r in data['records']]
        if len(case_names) != len(set(case_names)):
            failures.append(f'{lane.name}: duplicate sampling case names')
        missing = set(BASE_CASE_NAMES) - set(case_names)
        if missing:
            failures.append(f'{lane.name}: missing base cases {sorted(missing)}')
        if data.get('full'):
            missing = (set(FULL_CASE_NAMES) | set(INVALID_CASE_NAMES)) - set(case_names)
            if missing:
                failures.append(f'{lane.name}: missing full cases {sorted(missing)}')
        for receipt in data['records']:
            if receipt['name'] == 'json_schema_color_enum':
                parsed = receipt.get('validated_json')
                if parsed not in ({'color': 'red'}, {'color': 'blue'}, {'color': 'green'}):
                    failures.append(f'{lane.name}: color-enum output was not validated JSON')
        data['_diagnostic_counts'] = {'effective': len(effective), 'runtime': len(runtime)}
        (lane / 'readback.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')

    for model in ('gemma', 'qwen'):
        for codec in ('oga', 'flm', 'compare'):
            names = [f'{model}-{codec}-{cache}' for cache in ('off', 'on')]
            if not all(name in lanes for name in names):
                failures.append(f'missing cache pair: {names}')
                continue
            left_records = lanes[names[0]]['records']
            right_records = lanes[names[1]]['records']
            failures.extend(cache_content_failures(left_records, right_records, f'{model}/{codec}'))
            for pair_lane in names:
                records = {r['name']: r for r in lanes[pair_lane]['records']}
                for first, repeat in (('seeded_first', 'seeded_repeat'),
                                      ('pure_top_p_seeded_first', 'pure_top_p_seeded_repeat')):
                    if records.get(first, {}).get('content') != records.get(repeat, {}).get('content'):
                        failures.append(f'{pair_lane}/{repeat}: seeded first/repeat content differs')
                    for item in (first, repeat):
                        if records.get(item, {}).get('cached_tokens', 0) != 0:
                            failures.append(f'{pair_lane}/{item}: seeded request reported cached tokens')
                if records.get('extended_seeded_nonstream', {}).get('content') != records.get(
                        'extended_seeded_stream', {}).get('content'):
                    failures.append(f'{pair_lane}: extended seeded stream/nonstream content differs')
    report = {'status': 'PASS' if not failures else 'FAIL', 'lanes': len(lanes),
              'requests': sum(len(v['records']) for v in lanes.values()),
              'case_counts': {name: len(value['records']) for name, value in lanes.items()},
              'readbacks': {name: value.get('_diagnostic_counts') for name, value in lanes.items()},
              'failures': failures}
    (args.root / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
