#!/usr/bin/env python3
"""P340 collection/policy runner. No motion unless --enable-motion is supplied."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
from urllib.request import urlopen

import numpy as np

from jointact.robots.p340 import Controller, EpisodeWriter, HTTPPolicy, Limits, SDKProcess, build_dataset, summarize_trials


class Camera:
    """Keep only the most recently received RGB frame; never buffer an episode."""
    def __init__(self, index):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError('Cannot open camera')
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest, self.error = None, None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.stop.is_set():
            ok, bgr = self.capture.read()
            received = time.monotonic()
            if not ok:
                self.error = RuntimeError('Camera read failed')
                return
            rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            with self.lock:
                self.latest = (rgb, received)

    def read(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.error:
                raise self.error
            with self.lock:
                if self.latest is not None:
                    return self.latest[0].copy(), self.latest[1]
            time.sleep(0.01)
        raise TimeoutError('No camera frame received')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        # A blocked backend read must be released to unblock the daemon.
        self.capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    summary = sub.add_parser('summarize')
    summary.add_argument('--recordings', required=True)
    build = sub.add_parser('dataset')
    build.add_argument('--recordings', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--seed', type=int, default=42)
    for mode in ('collect', 'policy'):
        run = sub.add_parser(mode)
        run.add_argument('--port', required=True)
        run.add_argument('--camera', type=int, default=0)
        run.add_argument('--limits', required=True)
        run.add_argument('--output', required=True)
        run.add_argument('--episode-id', required=True)
        run.add_argument('--group-id', required=True, help='Same collection batch/layout family -> same split')
        run.add_argument('--instruction', required=True)
        run.add_argument('--max-steps', type=int, default=100)
        run.add_argument('--enable-motion', action='store_true')
        if mode == 'policy':
            run.add_argument('--server', required=True, help='Server base URL, preferably localhost via SSH forwarding')
    args = parser.parse_args()
    if args.mode == 'summarize':
        print(json.dumps(summarize_trials(args.recordings), indent=2))
        return
    if args.mode == 'dataset':
        print(json.dumps(build_dataset(args.recordings, args.output, args.seed), indent=2))
        return
    if not args.enable_motion:
        parser.error('No motion enabled. Calibrate workspace/gripper and inspect the setup before --enable-motion.')
    if args.max_steps < 1:
        parser.error('max-steps must be positive')
    limits = Limits(**json.loads(Path(args.limits).read_text()))
    policy, identity = None, None
    if args.mode == 'policy':
        with urlopen(args.server.rstrip('/') + '/health', timeout=5) as response:
            health = json.load(response)
        from jointact.data.schema import P340_ACTION_SEMANTICS
        if health['action_semantics'] != P340_ACTION_SEMANTICS or not health.get('policy_identity'):
            raise ValueError('Need a P340 policy with recorded checkpoint identity')
        identity = health['policy_identity']
        policy = HTTPPolicy(args.server.rstrip('/') + '/predict', identity=identity)
    writer = EpisodeWriter(args.output, args.episode_id, args.instruction, args.group_id,
                           limits.calibration_id, 'evaluation' if policy else 'demonstration')
    sdk = camera = None
    events_path = Path(args.output) / f'{args.episode_id}.events.jsonl'
    try:
        with events_path.open('x', buffering=1) as events:
            writer.file.attrs['limits_json'] = json.dumps(asdict(limits), allow_nan=False)
            writer.file.attrs['camera_index'] = args.camera
            writer.file.attrs['policy_identity_json'] = json.dumps(identity)
            writer.file.flush()
            def log(**values):
                events.write(json.dumps(dict(monotonic=time.monotonic(), **values), allow_nan=False) + '\n')
            log(event='start', limits=asdict(limits), policy_identity=identity,
                instruction=args.instruction, camera_index=args.camera, execution_horizon=1,
                group_id=args.group_id, max_steps=args.max_steps,
                timestamp_semantics='host_receipt_not_exposure', wall_time=time.time())
            try:
                camera = Camera(args.camera)
                sdk = SDKProcess(args.port, timeout=limits.motion_timeout_s)
                controller = Controller(sdk, limits)
                for step in range(args.max_steps):
                    action = None
                    if not policy:
                        text = input('dx dy dz (mm), gripper open01; q=finish: ').strip()
                        if text == 'q':
                            break
                        action = np.asarray([float(x) for x in text.split()])
                    state_start = time.monotonic()
                    state = controller.state()
                    state_end = time.monotonic()
                    image, frame_time = camera.read()
                    observation_time = min(state_start, frame_time)
                    predict_start = time.monotonic()
                    if policy:
                        action = policy.predict(image, state, args.instruction)
                    predict_end = time.monotonic()
                    log(event='requested_action', step=step, state=state.tolist(), action=action.tolist(),
                        frame_received=frame_time, state_read_start=state_start, state_read_end=state_end,
                        predict_start=predict_start, predict_end=predict_end)
                    result = controller.execute(state, action, observation_time)
                    timing = dict(frame_received=frame_time, state_read_start=state_start, state_read_end=state_end,
                                  predict_start=predict_start, predict_end=predict_end, **result)
                    writer.append(image, state, action, timing)
                    log(event='completed_action', step=step, **timing)
                label = input('Task succeeded? Enter yes or no: ').strip().lower()
                if label not in {'yes', 'no'}:
                    raise ValueError('Explicit yes/no outcome required; partial recording retained')
                writer.finish(label == 'yes')
                log(event='finished', success=label == 'yes', completed_steps=writer.length)
            except BaseException as error:
                log(event='aborted', error=repr(error), completed_steps=writer.length)
                raise
    finally:
        writer.close()
        if sdk:
            sdk.close()
        if camera:
            camera.close()


if __name__ == '__main__':
    main()
