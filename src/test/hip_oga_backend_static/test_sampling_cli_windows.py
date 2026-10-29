#!/usr/bin/env python3
"""Drive a real Windows console to check persistent HIP OGA /set commands."""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--model', default='hip-gemma3-text-strict:4b',
                   help='Model alias or path (default: %(default)s)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sampler', choices=('backend', 'common'), default='backend')
    args = p.parse_args()
    for name, value in (('--binary', args.binary), ('--model', args.model)):
        if any(char in value for char in ('"', '\r', '\n')):
            p.error(f'{name} must not contain quotes or newlines')
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / 'stdout.log'
    error_log = args.output / 'stderr.log'
    failures = []
    commands = []
    child = None
    kernel = c.WinDLL('kernel32', use_last_error=True)

    class Key(c.Structure):
        _fields_ = [('down', w.BOOL), ('repeat', w.WORD), ('vk', w.WORD),
                    ('scan', w.WORD), ('char', w.WCHAR), ('control', w.DWORD)]
    class Event(c.Union):
        _fields_ = [('key', Key), ('padding', c.c_byte * 16)]
    class Record(c.Structure):
        _fields_ = [('type', w.WORD), ('event', Event)]
    kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, c.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
    kernel.CreateFileW.restype = w.HANDLE
    kernel.WriteConsoleInputW.argtypes = [w.HANDLE, c.POINTER(Record), w.DWORD, c.POINTER(w.DWORD)]
    kernel.CloseHandle.argtypes = [w.HANDLE]

    def text():
        return log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''

    def wait_for(predicate, limit=90):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if predicate(): return
            if child.poll() is not None: raise RuntimeError(f'CLI exited early: {child.returncode}')
            time.sleep(0.1)
        raise TimeoutError('CLI did not reach the expected state')

    try:
        with log.open('wb') as out, error_log.open('wb') as err:
            # Explicit CONIN$ prevents Python's redirected startup handles from
            # leaving the child with a pipe where ReadConsoleInput expects a console.
            child = subprocess.Popen('cmd.exe /d /s /c ""' + args.binary +
                                     '" run "' + args.model + '" --backend hip --sampler ' + args.sampler +
                                     ' --ctx-len 512 --hip-oga-text-codec oga < CONIN$"',
                                     stdout=out, stderr=err, creationflags=subprocess.CREATE_NEW_CONSOLE)
            wait_for(lambda: 'Type /? for help' in text())
            kernel.FreeConsole()
            if not kernel.AttachConsole(child.pid): raise c.WinError(c.get_last_error())
            handle = kernel.CreateFileW('CONIN$', 0xC0000000, 3, None, 3, 0, None)
            if handle == w.HANDLE(-1).value: raise c.WinError(c.get_last_error())

            def command(value):
                commands.append(value)
                for char in value + '\r':
                    record = Record(); record.type = 1
                    record.event.key = Key(True, 1, 13 if char == '\r' else 0, 0, char, 0)
                    count = w.DWORD()
                    if not kernel.WriteConsoleInputW(handle, c.byref(record), 1, c.byref(count)):
                        raise c.WinError(c.get_last_error())
                    time.sleep(0.015)
                time.sleep(0.2)

            rejected_commands = ['/set temp 0.5', '/set topk 3', '/set topp 0.5', '/set rep-pen 1.25',
                                 '/set seed 123', '/set gen-lim 8', '/set minp 0.25', '/set seed 1.5',
                                 '/set topk 1.5', '/set temp 1junk']
            if args.sampler == 'common':
                rejected_commands.append('/set sampler future')
            for value in rejected_commands:
                command(value)
            expected_rejections = 4 if args.sampler == 'common' else 3
            wait_for(lambda: text().count('Setting rejected:') == expected_rejections, 15)
            command('Name a color.')
            wait_for(lambda: text().count('HIP OGA sampling effective=') == 1)
            command('/status')
            wait_for(lambda: text().count('  Statistics:') == 1)
            command('/set temp 0')
            if args.sampler == 'common':
                command('/set sampler backend')
            command('Name another color.')
            wait_for(lambda: text().count('HIP OGA sampling effective=') == 2)
            command('/status')
            wait_for(lambda: text().count('  Statistics:') == 2)
            command('/bye')
            child.wait(timeout=15)
            kernel.CloseHandle(handle)
            kernel.FreeConsole()
        all_logs = text() + error_log.read_text(encoding='utf-8', errors='replace')
        readbacks = [json.loads(m.group(1)) for m in re.finditer(r'HIP OGA sampling effective=(\{[^\r\n]*\})', all_logs)]
        if len(readbacks) != 2: failures.append(f'expected two generations, got {len(readbacks)}')
        expected = [dict(temperature=0.5, top_k=3, top_p=0.5, repetition_penalty=1.25, random_seed=123,
                         do_sample=True, min_p=0.25, sampler=args.sampler),
                    dict(temperature=1, top_k=3, top_p=0.5, repetition_penalty=1.25, random_seed=123,
                         do_sample=False, min_p=0.25, sampler='backend')]
        for actual, want in zip(readbacks, expected):
            if any(actual.get(k) != v for k, v in want.items()): failures.append(f'{actual} != {want}')
        runtime_readbacks = [json.loads(m.group(1)) for m in re.finditer(
            r'HIP OGA sampling runtime=(\{[^\r\n]*\})', all_logs)]
        if len(runtime_readbacks) != 2:
            failures.append(f'expected two OGA runtime readbacks, got {len(runtime_readbacks)}')
        expected_runtime = [dict(temperature=1, top_k=0, top_p=1, repetition_penalty=1,
                                 random_seed=123, do_sample=args.sampler == 'backend'),
                            dict(temperature=1, top_k=0, top_p=1, repetition_penalty=1,
                                 random_seed=123, do_sample=False)]
        for actual, want in zip(runtime_readbacks, expected_runtime):
            if any(actual.get(k) != v for k, v in want.items()):
                failures.append(f'OGA runtime {actual} != neutral settings {want}')
        if child.returncode != 0: failures.append(f'CLI exit={child.returncode}')
    except Exception as e:
        failures.append(f'{type(e).__name__}: {e}')
    finally:
        if child and child.poll() is None:
            subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
            child.wait()
        result = {'status': 'FAIL' if failures else 'PASS', 'sampler': args.sampler,
                  'commands': commands, 'failures': failures}
        (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
