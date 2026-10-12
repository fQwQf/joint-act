"""P340 fixed-orientation adapter. XYZ in base-frame mm; gripper 1=open.

Controller coordinates are reported values, not independently measured pose.
No automatic homing, torque release or claims of a hardware emergency stop.
"""
import base64
from dataclasses import dataclass
import io
import json
import multiprocessing as mp
from pathlib import Path
import time
from urllib.request import Request, urlopen

import h5py
import numpy as np
from PIL import Image

from jointact.data.schema import P340_ACTION_SEMANTICS, SCHEMA_VERSION, episode_split, write_manifest
from jointact.utils import sha256


@dataclass
class Limits:
    xyz_min: list
    xyz_max: list
    calibration_id: str
    gripper_closed: int
    gripper_open: int
    initial_gripper_open01: float
    max_delta_mm: float = 5.0
    speed_mm_s: int = 10
    gripper_speed: int = 100
    observation_max_age_s: float = 2.0
    motion_timeout_s: float = 10.0
    gripper_settle_s: float = 1.0

    def __post_init__(self):
        low, high = np.asarray(self.xyz_min), np.asarray(self.xyz_max)
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or (low >= high).any():
            raise ValueError('Calibrated XYZ bounds required')
        if not self.calibration_id or self.calibration_id == 'UNCONFIGURED':
            raise ValueError('Calibrate the workspace and gripper before connecting')
        scalars = [self.max_delta_mm, self.observation_max_age_s, self.motion_timeout_s,
                   self.gripper_settle_s, self.initial_gripper_open01, self.speed_mm_s,
                   self.gripper_speed, self.gripper_open, self.gripper_closed]
        if not np.isfinite(scalars).all():
            raise ValueError('Nonfinite limits')
        if not (0 < self.max_delta_mm <= 10 and 0 < self.speed_mm_s <= 20
                and 0 <= self.initial_gripper_open01 <= 1 and self.observation_max_age_s > 0
                and self.motion_timeout_s > 0 and self.gripper_settle_s > 0
                and 0 < self.gripper_speed <= 1500
                and 0 <= self.gripper_closed <= 100 and 0 <= self.gripper_open <= 100
                and self.gripper_closed != self.gripper_open):
            raise ValueError('Invalid calibrated limits')

    def target(self, state, action, observation_time, now=None):
        state, action = np.asarray(state, dtype=float), np.asarray(action, dtype=float)
        if state.shape != (4,) or action.shape != (4,) or not np.isfinite([state, action]).all():
            raise ValueError('Expected finite four-dimensional state/action')
        age = (time.monotonic() if now is None else now) - observation_time
        if not np.isfinite(age) or not 0 <= age <= self.observation_max_age_s:
            raise ValueError('Stale observation')
        target = state[:3] + action[:3]
        if (np.linalg.norm(action[:3]) > self.max_delta_mm or not 0 <= action[3] <= 1
                or (state[:3] < self.xyz_min).any() or (state[:3] > self.xyz_max).any()
                or (target < self.xyz_min).any() or (target > self.xyz_max).any()):
            raise ValueError('Action exceeds calibrated workspace/step/gripper limits')
        return target.tolist(), round(self.gripper_closed + action[3] * (self.gripper_open - self.gripper_closed))


def _sdk_worker(connection, port):
    try:
        from pymycobot.ultraArmP340 import ultraArmP340
        sdk = ultraArmP340(port, 115200)
        connection.send(('ok', None))
        while True:
            method, args = connection.recv()
            if method not in {'get_coords_info', 'set_mode', 'set_coords', 'set_gripper_state', 'is_moving_end'}:
                raise ValueError('Unsupported SDK operation')
            connection.send(('ok', getattr(sdk, method)(*args)))
    except BaseException as error:
        try:
            connection.send(('error', repr(error)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class SDKProcess:
    """Bound SDK blocking in a child process; timeout latches a fault.

    Killing the SDK process does NOT cancel motion already queued in firmware.
    """
    def __init__(self, port, timeout=10):
        self.timeout, self.fault = timeout, False
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_sdk_worker, args=(child, port), daemon=True)
        self.process.start()
        child.close()
        self._receive()

    def _receive(self):
        try:
            if not self.connection.poll(self.timeout):
                raise TimeoutError('SDK timed out; inspect robot and use physical stop if needed')
            status, value = self.connection.recv()
            if status != 'ok':
                raise RuntimeError(value)
            return value
        except BaseException:
            self.fault = True
            self.close()
            raise

    def call(self, method, *args):
        if self.fault:
            raise RuntimeError('SDK fault latched; reconnect only after inspecting robot')
        self.connection.send((method, args))
        return self._receive()

    def close(self):
        self.fault = True
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
        self.connection.close()


class Controller:
    def __init__(self, sdk, limits):
        self.sdk, self.limits = sdk, limits
        self.gripper = limits.initial_gripper_open01
        self.fault = False

    def state(self):
        xyz = self.sdk.call('get_coords_info')
        if xyz is None or len(xyz) < 3 or not np.isfinite(xyz[:3]).all():
            raise RuntimeError('Missing controller coordinates')
        return np.asarray([*xyz[:3], self.gripper], dtype=np.float32)

    def execute(self, state, action, observation_time):
        if self.fault:
            raise RuntimeError('Controller fault latched')
        target, grip = self.limits.target(state, action, observation_time)
        start = time.monotonic()
        try:
            # Reassert absolute mode; never inherit an unknown relative-mode setting.
            self.sdk.call('set_mode', 0)
            # Mode RPC can itself block; recheck freshness immediately before motion.
            self.limits.target(state, action, observation_time)
            self.sdk.call('set_coords', target, self.limits.speed_mm_s)
            deadline = time.monotonic() + self.limits.motion_timeout_s
            while self.sdk.call('is_moving_end') != 1:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Motion completion not confirmed')
                time.sleep(0.05)
            # Avoid accumulating a gripper command at every Cartesian step.
            if not np.isclose(action[3], self.gripper):
                self.sdk.call('set_gripper_state', grip, self.limits.gripper_speed)
                time.sleep(self.limits.gripper_settle_s)
                self.gripper = float(action[3])
            return dict(command_start=start, completion_time=time.monotonic(), target_xyz_mm=target,
                        gripper_command=grip, completion_source='controller_flag+gripper_settle')
        except BaseException:
            self.fault = True
            raise


class HTTPPolicy:
    def __init__(self, url, timeout=10, identity=None):
        self.url, self.timeout, self.identity = url, timeout, identity

    def predict(self, image, state, instruction):
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format='PNG')
        payload = dict(images=[base64.b64encode(stream.getvalue()).decode()],
                       proprio=state.tolist(), instruction=instruction)
        request = Request(self.url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if self.identity is not None and payload.get('policy_identity') != self.identity:
            raise ValueError('Policy identity changed during trial')
        result = payload['predictions'][0]
        actions = np.asarray(result['actions'], dtype=float)
        if (result['action_semantics'] != P340_ACTION_SEMANTICS or actions.ndim != 2
                or actions.shape[1] != 4 or len(actions) == 0 or not np.isfinite(actions).all()):
            raise ValueError('Policy must be trained for P340 four-dimensional mm actions')
        return actions[0]  # Reobserve after every executed action, same for every baseline.


class EpisodeWriter:
    """Append one frame at a time; interrupted attempts remain .partial.h5."""
    def __init__(self, directory, episode_id, instruction, group_id, calibration_id, mode):
        if not episode_id or Path(episode_id).name != episode_id or episode_id in {'.', '..'}:
            raise ValueError('Unsafe episode ID')
        if not instruction.strip() or not group_id or mode not in {'demonstration', 'evaluation'}:
            raise ValueError('Instruction, split group and recording mode required')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.final = directory / f'{episode_id}.h5'
        if self.final.exists():
            raise FileExistsError(self.final)
        self.path = directory / f'{episode_id}.partial.h5'
        self.file = h5py.File(self.path, 'x')
        self.file.attrs.update(episode_id=episode_id, instruction=instruction, group_id=group_id,
                               calibration_id=calibration_id, mode=mode,
                               action_semantics=P340_ACTION_SEMANTICS,
                               proprio_semantics='controller_xyz_mm+last_commanded_gripper_open01',
                               timestamp_semantics='host_monotonic_receipt_not_camera_exposure')
        self.length = 0

    def append(self, image, state, action, timing):
        image, state, action = np.asarray(image), np.asarray(state), np.asarray(action)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('RGB uint8 frame required')
        if state.shape != (4,) or action.shape != (4,) or not np.isfinite([state, action]).all():
            raise ValueError('Invalid state/action')
        values = dict(images=image[None], proprio=state.astype('float32'), actions=action.astype('float32'))
        if not self.length:
            for name, value in values.items():
                self.file.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                         chunks=(1, *value.shape), dtype=value.dtype, compression='lzf')
            self.file.create_dataset('timing_json', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        encoded = json.dumps(timing, allow_nan=False)
        for name, value in values.items():
            dataset = self.file[name]
            if dataset.shape[1:] != value.shape:
                raise ValueError('Frame shape changed during episode')
            dataset.resize(self.length + 1, axis=0)
            dataset[self.length] = value
        self.file['timing_json'].resize(self.length + 1, axis=0)
        self.file['timing_json'][self.length] = encoded
        self.length += 1
        self.file.flush()

    def finish(self, success):
        if self.length == 0 or type(success) is not bool:
            raise ValueError('Nonempty episode and explicit success label required')
        self.file.attrs['success'] = success
        self.file.close()
        self.path.rename(self.final)
        return self.final

    def close(self):
        if self.file.id.valid:
            self.file.close()


def build_dataset(recordings, output, seed=42):
    """Copy accepted demonstrations without decoding images or splitting frames."""
    import shutil
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh immutable dataset directory')
    files = sorted(Path(recordings).glob('*.h5'))
    rows = []
    for path in files:
        if path.name.endswith('.partial.h5'):
            continue
        with h5py.File(path, 'r') as file:
            if file.attrs['mode'] != 'demonstration' or not file.attrs['success']:
                continue
            if file.attrs['action_semantics'] != P340_ACTION_SEMANTICS:
                raise ValueError('Mixed action semantics')
            episode_id = str(file.attrs['episode_id'])
            target = output / 'episodes' / f'{episode_id}.h5'
            if Path(episode_id).name != episode_id or target.exists():
                raise ValueError('Unsafe/duplicate episode ID')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            rows.append(dict(schema_version=SCHEMA_VERSION, episode_id=episode_id,
                             path=str(target.relative_to(output)), length=len(file['actions']), action_dim=4,
                             proprio_dim=4, num_images=1, cameras=['fixed_rgb'],
                             instruction=str(file.attrs['instruction']),
                             split=episode_split(str(file.attrs['group_id']), seed),
                             action_semantics=P340_ACTION_SEMANTICS,
                             source=dict(group_id=str(file.attrs['group_id']),
                                         calibration_id=str(file.attrs['calibration_id']),
                                         recording_sha256=sha256(path)), sha256=sha256(target)))
    write_manifest(output, rows)
    return rows


def summarize_trials(directory):
    """Include aborted/unfinished attempts; never pool different model identities."""
    groups = {}
    for path in sorted(Path(directory).glob('*.events.jsonl')):
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not events or events[0].get('event') != 'start':
            raise ValueError(f'Missing trial header: {path.name}')
        header = events[0]
        key = json.dumps(dict(policy_identity=header['policy_identity'], instruction=header['instruction'],
                              limits=header['limits'], execution_horizon=header['execution_horizon'],
                              camera_index=header['camera_index'], max_steps=header['max_steps']), sort_keys=True)
        group = groups.setdefault(key, dict(protocol=json.loads(key), trials=[], latency_ms=[]))
        terminal = events[-1]
        status = terminal['event'] if terminal['event'] in {'finished', 'aborted'} else 'unfinished'
        group['trials'].append(dict(file=path.name, status=status,
                                   success=bool(terminal.get('success')) if status == 'finished' else False))
        for event in events:
            if event['event'] == 'completed_action':
                group['latency_ms'].append(dict(
                    inference_roundtrip=1000 * (event['predict_end'] - event['predict_start']),
                    frame_receipt_to_command=1000 * (event['command_start'] - event['frame_received']),
                    command_to_completion=1000 * (event['completion_time'] - event['command_start'])))
    results = []
    for group in groups.values():
        counts = {s: sum(t['status'] == s for t in group['trials']) for s in ('finished', 'aborted', 'unfinished')}
        successes = sum(t['success'] for t in group['trials'])
        latency = {}
        if group['latency_ms'] and group['protocol']['policy_identity'] is not None:
            for metric in group['latency_ms'][0]:
                values = [v[metric] for v in group['latency_ms']]
                latency[metric] = dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)))
        results.append(dict(protocol=group['protocol'], trials=group['trials'], counts=counts,
                            successes=successes, attempts=len(group['trials']),
                            success_rate=None if counts['unfinished'] else successes / len(group['trials']),
                            latency_ms=latency, latency_scope='completed_actions_including_first_no_warmup_removed'))
    return results
