import json
import time

import h5py
import numpy as np
import pytest

from jointact.config import Config
from jointact.data.dataset import EpisodeDataset
from jointact.data.prepare import prepare_artifacts, validate_dataset
from jointact.data.schema import P340_ACTION_SEMANTICS, episode_split
from jointact.inference import PolicyRuntime
from jointact.robots.p340 import Controller, EpisodeWriter, Limits, build_dataset
from jointact.training import train


def limits(**overrides):
    values = dict(xyz_min=[100, -50, 20], xyz_max=[250, 50, 100], calibration_id='test-only',
                  gripper_closed=80, gripper_open=20, initial_gripper_open01=1,
                  gripper_settle_s=0.001)
    return Limits(**dict(values, **overrides))


def test_units_gripper_polarity_and_rejection():
    calibrated = limits()
    target, grip = calibrated.target([150, 0, 50, 1], [2, -1, 0, 0], 10, now=10.1)
    assert target == [152, -1, 50] and grip == 80
    for action in ([6, 0, 0, 1], [0, 0, 0, 2], [float('nan'), 0, 0, 0]):
        with pytest.raises(ValueError):
            calibrated.target([150, 0, 50, 1], action, 10, now=10.1)
    with pytest.raises(ValueError, match='Stale'):
        calibrated.target([150, 0, 50, 1], [0, 0, 0, 1], 10, now=20)
    with pytest.raises(ValueError):
        calibrated.target([249, 0, 50, 1], [2, 0, 0, 1], 10, now=10.1)
    with pytest.raises(ValueError):
        limits(calibration_id='UNCONFIGURED')


def test_controller_no_queue_accumulation_and_fault_latch():
    class SDK:
        def __init__(self):
            self.calls, self.fail = [], False

        def call(self, method, *args):
            self.calls.append((method, args))
            if self.fail:
                raise TimeoutError('fake timeout')
            return 1 if method == 'is_moving_end' else None

    sdk = SDK()
    controller = Controller(sdk, limits())
    controller.execute([150, 0, 50, 1], [1, 0, 0, 0], time.monotonic())
    assert [name for name, _ in sdk.calls] == ['set_mode', 'set_coords', 'is_moving_end', 'set_gripper_state']
    assert controller.gripper == 0
    sdk.fail = True
    with pytest.raises(TimeoutError):
        controller.execute([151, 0, 50, 0], [1, 0, 0, 0], time.monotonic())
    count = len(sdk.calls)
    with pytest.raises(RuntimeError, match='latched'):
        controller.execute([151, 0, 50, 0], [1, 0, 0, 0], time.monotonic())
    assert len(sdk.calls) == count


def record(directory, name, group, mode='demonstration', success=True):
    writer = EpisodeWriter(directory, name, 'place block', group, 'test-only', mode)
    for step in range(8):
        writer.append(np.full((16, 16, 3), step, np.uint8), [150 + step, 0, 50, 1],
                      [step % 3 - 1, 0, 0, step % 2], {'frame_received': step, 'completion_time': step + .5})
    writer.finish(success)


def test_stream_split_and_four_dimensional_training(tmp_path):
    raw = tmp_path / 'raw'
    groups = {}
    for i in range(1000):
        group = f'batch-{i}'
        groups.setdefault(episode_split(group), group)
        if len(groups) == 3:
            break
    for split, group in groups.items():
        record(raw, split, group)
    record(raw, 'same-group', groups['train'])
    record(raw, 'eval-excluded', groups['train'], mode='evaluation')
    record(raw, 'failed-excluded', groups['train'], success=False)
    interrupted = EpisodeWriter(raw, 'interrupted', 'place block', groups['train'], 'test-only', 'demonstration')
    interrupted.close()
    root = tmp_path / 'data'
    rows = build_dataset(raw, root)
    assert len(rows) == 4
    assert next(r for r in rows if r['episode_id'] == 'same-group')['split'] == 'train'
    assert validate_dataset(root)['signature'] == [4, 4, 1, P340_ACTION_SEMANTICS]
    with h5py.File(root / rows[0]['path']) as f:
        assert f['images'].chunks[0] == 1
        assert 'completion_time' in json.loads(f['timing_json'][0])
    artifacts = root / 'artifacts'
    meta = prepare_artifacts(root, artifacts, horizon=2, num_modes=2, max_samples=32, iterations=2)
    assert meta['action_normalizer']['mask'] == [True, True, True, False]
    cfg = Config.from_dict(dict(
        model=dict(action_dim=4, proprio_dim=4, num_images=1, horizon=2, num_modes=2,
                   hidden_dim=16, head_dim=16, image_size=16),
        data=dict(root=str(root), artifacts=str(artifacts), image_aug=False, crop_scale=1),
        train=dict(output=str(tmp_path / 'run'), device='cpu', precision='fp32', batch_size=2,
                   max_steps=2, warmup_steps=0, eval_every=2, save_every=2, eval_batches=2, min_free_disk_gb=0)))
    result = train(cfg)
    runtime = PolicyRuntime(result['checkpoint'])
    dataset = EpisodeDataset(root, 2, 'test')
    try:
        prediction = runtime.predict(dataset[0])
    finally:
        dataset.close()
    assert prediction['action_semantics'] == P340_ACTION_SEMANTICS
    assert np.asarray(prediction['actions']).shape == (2, 4)
    assert all(0 <= a[3] <= 1 for a in prediction['actions'])


def test_http_rejects_libero_actions(monkeypatch):
    import io
    from jointact.robots import p340
    response = dict(predictions=[dict(action_semantics='wrong', actions=[[0, 0, 0, 1]])])
    monkeypatch.setattr(p340, 'urlopen', lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    policy = p340.HTTPPolicy('http://localhost/predict')
    with pytest.raises(ValueError, match='P340'):
        policy.predict(np.zeros((8, 8, 3), np.uint8), np.zeros(4), 'place')
    response['predictions'][0]['action_semantics'] = P340_ACTION_SEMANTICS
    np.testing.assert_array_equal(policy.predict(np.zeros((8, 8, 3), np.uint8), np.zeros(4), 'place'), [0, 0, 0, 1])
    policy.identity = {'weights_sha256': 'expected'}
    with pytest.raises(ValueError, match='identity changed'):
        policy.predict(np.zeros((8, 8, 3), np.uint8), np.zeros(4), 'place')


def test_trial_summary_keeps_aborted_and_unfinished(tmp_path):
    from jointact.robots.p340 import summarize_trials
    header = dict(event='start', policy_identity={'weights_sha256': 'test'}, instruction='place',
                  limits={}, execution_horizon=1, camera_index=0, max_steps=10)
    for name, tail in [('ok', [{'event': 'finished', 'success': True}]),
                       ('abort', [{'event': 'aborted'}]), ('pending', [])]:
        (tmp_path / f'{name}.events.jsonl').write_text('\n'.join(json.dumps(x) for x in [header, *tail]))
    result = summarize_trials(tmp_path)[0]
    assert result['attempts'] == 3 and result['successes'] == 1
    assert result['counts'] == dict(finished=1, aborted=1, unfinished=1)
    assert result['success_rate'] is None


def test_slow_mode_rpc_cannot_send_stale_motion(monkeypatch):
    from jointact.robots import p340
    clock = [0.0]
    monkeypatch.setattr(p340.time, 'monotonic', lambda: clock[0])

    class SlowSDK:
        def __init__(self):
            self.calls = []

        def call(self, method, *args):
            self.calls.append(method)
            clock[0] = 3.0

    sdk = SlowSDK()
    controller = Controller(sdk, limits())
    with pytest.raises(ValueError, match='Stale'):
        controller.execute([150, 0, 50, 1], [1, 0, 0, 1], 0.0)
    assert sdk.calls == ['set_mode'] and controller.fault
