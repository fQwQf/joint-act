"""Safety boundary tests use only stdlib; no model allocation."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'memory_safe_queue', Path(__file__).parents[1] / 'scripts/memory_safe_queue.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_reject_unbounded_and_swap(tmp_path):
    (tmp_path / 'memory.max').write_text('max')
    (tmp_path / 'memory.swap.max').write_text('0')
    with pytest.raises(RuntimeError):
        guard.validate_bound(tmp_path)
    (tmp_path / 'memory.max').write_text('96000000000')
    assert guard.validate_bound(tmp_path) == 96 * guard.GB
    (tmp_path / 'memory.swap.max').write_text('1')
    with pytest.raises(RuntimeError):
        guard.validate_bound(tmp_path)


def test_admission_reserves_host_headroom():
    guard.check_headroom(10 * guard.GB, 0, 96 * guard.GB)
    with pytest.raises(RuntimeError):
        guard.check_headroom(20 * guard.GB, 0, 96 * guard.GB)
    with pytest.raises(RuntimeError):
        guard.check_headroom(20 * guard.GB, 50 * guard.GB, 96 * guard.GB)


def test_queue_does_not_accept_shell_string(tmp_path):
    path = tmp_path / 'queue.json'
    path.write_text(json.dumps(['python train.py; second-command']))
    with pytest.raises(ValueError):
        guard.load_queue(path)
    path.write_text(json.dumps([['python', 'train.py']]))
    assert guard.load_queue(path) == [['python', 'train.py']]


def test_fail_fast_and_child_cleanup(tmp_path, monkeypatch):
    import argparse
    import sys

    (tmp_path / 'memory.max').write_text('96000000000')
    (tmp_path / 'memory.swap.max').write_text('0')
    (tmp_path / 'memory.current').write_text('0')
    (tmp_path / 'memory.events').write_text('oom 0')
    monkeypatch.setattr(guard, 'current_group', lambda: tmp_path)
    monkeypatch.setattr(guard, 'host_used', lambda: 0)
    marker = tmp_path / 'should-not-exist'
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps([[sys.executable, '-c', 'raise SystemExit(7)'],
                                 [sys.executable, '-c', f'open({str(marker)!r}, "w").close()']]))
    assert guard.run(argparse.Namespace(queue=queue, log=tmp_path / 'log.jsonl')) == 7
    assert not marker.exists()


def test_watchdog_stops_child(tmp_path, monkeypatch):
    import argparse
    import sys

    for name, value in {'memory.max': '96000000000', 'memory.swap.max': '0',
                        'memory.current': '0', 'memory.events': 'oom 0'}.items():
        (tmp_path / name).write_text(value)
    monkeypatch.setattr(guard, 'current_group', lambda: tmp_path)
    readings = iter([0, 111 * guard.GB])
    monkeypatch.setattr(guard, 'host_used', lambda: next(readings))
    processes = []
    real_popen = guard.subprocess.Popen

    def capture(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(guard.subprocess, 'Popen', capture)
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps([[sys.executable, '-c', 'import time; time.sleep(30)']]))
    with pytest.raises(RuntimeError, match='110 GB'):
        guard.run(argparse.Namespace(queue=queue, log=tmp_path / 'log.jsonl'))
    assert len(processes) == 1 and processes[0].poll() is not None
