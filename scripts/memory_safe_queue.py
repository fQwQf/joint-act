#!/usr/bin/env python3
"""Run a serial experiment queue inside an externally bounded cgroup v2.

Standard library only: all checks happen before loading torch or model weights.
The cgroup must contain the whole project, including evaluators and data jobs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GB = 1_000_000_000


def host_used(path=Path('/proc/meminfo')):
    fields = {k: int(v.split()[0]) * 1024
              for k, v in (line.split(':', 1) for line in path.read_text().splitlines())}
    return fields['MemTotal'] - fields['MemAvailable']


def current_group():
    relative = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
    if '..' in Path(relative).parts:
        raise RuntimeError('Unsupported cgroup namespace; use a host cgroup v2 scope.')
    group = Path('/sys/fs/cgroup') / relative.lstrip('/')
    if not (group / 'memory.current').exists():
        raise RuntimeError('Cannot resolve cgroup v2 memory controller.')
    return group


def validate_bound(group):
    raw = (group / 'memory.max').read_text().strip()
    if raw == 'max' or not 0 < int(raw) <= 96 * GB:
        raise RuntimeError('Require a dedicated cgroup with memory.max <= 96000000000.')
    if (group / 'memory.swap.max').read_text().strip() != '0':
        raise RuntimeError('Require memory.swap.max=0 for this experiment queue.')
    return int(raw)


def check_headroom(used, current, maximum):
    # Do not subtract memory.current: it includes reclaimable cache that the
    # MemAvailable-based host estimate may already exclude. Double-counting
    # resident project memory is preferable to underestimating outside usage.
    if used + maximum > 110 * GB:
        raise RuntimeError('Insufficient host headroom: outside usage + cgroup cap exceeds 110 GB. '
                           'Lower the cgroup cap or wait for other jobs to finish.')


def load_queue(path):
    queue = json.loads(path.read_text())
    if not isinstance(queue, list) or not queue:
        raise ValueError('Queue must be a nonempty JSON list of argv lists.')
    for command in queue:
        if (not isinstance(command, list) or not command
                or any(not isinstance(arg, str) or not arg for arg in command)):
            raise ValueError('Each command must be a nonempty list of nonempty strings.')
    return queue


def terminate(process):
    # Clean up descendants even if their direct parent has already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run(args):
    queue = load_queue(args.queue)
    group = current_group()
    maximum = validate_bound(group)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    # Shared across invocations by this Unix user; prevents accidental second queues.
    lock_path = Path(f'/tmp/jointact-memory-queue-{os.getuid()}.lock')
    with lock_path.open('a') as lock, args.log.open('a', buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def sample(event, index):
            used = host_used()
            current = int((group / 'memory.current').read_text())
            peak_file = group / 'memory.peak'
            record = dict(time=time.time(), event=event, command_index=index,
                          host_used_bytes=used, cgroup_current_bytes=current,
                          cgroup_limit_bytes=maximum,
                          cgroup_peak_bytes=int(peak_file.read_text()) if peak_file.exists() else None,
                          memory_events=(group / 'memory.events').read_text())
            log.write(json.dumps(record) + '\n')
            return used, current

        for index, command in enumerate(queue):
            used, current = sample('admission', index)
            check_headroom(used, current, maximum)
            process = subprocess.Popen(command, start_new_session=True)
            try:
                while process.poll() is None:
                    used, _ = sample('running', index)
                    if used >= 110 * GB:
                        raise RuntimeError('Host usage reached 110 GB; terminating this job.')
                    time.sleep(1)
                sample('finished', index)
                if process.returncode:
                    return process.returncode if process.returncode > 0 else 128 - process.returncode
            finally:
                terminate(process)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, StopIteration) as error:
        print(f'Refusing/halting queue: {error}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
