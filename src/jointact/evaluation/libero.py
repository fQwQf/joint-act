"""LIBERO evaluation independent of TensorFlow and the upstream training package."""

import json
from pathlib import Path
import time

import numpy as np

from jointact.data.schema import ACTION_SEMANTICS
from jointact.inference import ChunkController
from jointact.utils import check_disk, json_digest, seed_everything, sha256, source_fingerprint
from jointact.evaluation.records import TrialLedger, evaluation_lock, trial_seed

MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520, libero_90=400)


def load_initial_states(path):
    """Read trusted benchmark assets, which contain pickled NumPy arrays.

    LIBERO's torch.load call predates PyTorch 2.6's weights_only default. This
    opt-out is confined to the explicitly installed benchmark's initial states;
    policy checkpoint loading continues to use weights_only=True.
    """
    import torch
    values = torch.load(path, map_location="cpu", weights_only=False)
    states = np.asarray(values)
    if states.ndim != 2 or not len(states) or not np.isfinite(states).all():
        raise ValueError("LIBERO initial states must be a finite nonempty matrix")
    return states


def quat_to_axisangle(quaternion):
    q = np.asarray(quaternion, np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or norm < 1e-12:
        raise ValueError("Expected nonzero xyzw quaternion")
    q = q / norm
    if q[3] < 0:
        q = -q
    sine = np.linalg.norm(q[:3])
    return np.zeros(3, np.float32) if sine < 1e-8 else (q[:3] / sine * 2 * np.arctan2(sine, q[3])).astype(np.float32)


def observation_from_libero(observation, instruction, cameras=2):
    images = [observation["agentview_image"][::-1, ::-1].copy()]
    if cameras == 2:
        images.append(observation["robot0_eye_in_hand_image"][::-1, ::-1].copy())
    elif cameras != 1:
        raise ValueError("LIBERO evaluator supports one or two cameras")
    state = np.concatenate([observation["robot0_eef_pos"], quat_to_axisangle(observation["robot0_eef_quat"]),
                            observation["robot0_gripper_qpos"]]).astype(np.float32)
    return dict(images=images, proprio=state, instruction=instruction)


def action_to_libero(action):
    action = np.asarray(action, np.float32).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Expected a finite 7D action")
    # Canonical gripper 1=open,0=close; LIBERO -1=open,+1=close.
    action[-1] = -1.0 if action[-1] >= 0.5 else 1.0
    return action


def run_episode(env, initial_state, runtime, instruction, max_steps, settle_steps=10, execute_horizon=None,
                collect_frames=True):
    if max_steps < 1 or settle_steps < 0:
        raise ValueError("max_steps must be positive and settle_steps nonnegative")
    controller = ChunkController(runtime, execute_horizon)
    controller.reset()
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(settle_steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    frames, times, decision_times = [], [], []
    success = False
    for step in range(max_steps):
        observation = observation_from_libero(obs, instruction, runtime.config.model.num_images)
        if collect_frames:
            frames.append(observation["images"][0])
        replanned = not controller.queue
        started = time.perf_counter()
        action = controller.action(observation)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        if replanned:
            decision_times.append(elapsed)
        obs, reward, done, info = env.step(action_to_libero(action).tolist())
        if done:
            success = True
            break
    return dict(success=success, steps=step + 1, control_mean_ms=float(np.mean(times) * 1000),
                control_p95_ms=float(np.percentile(times, 95) * 1000),
                decisions=len(decision_times), decision_mean_ms=float(np.mean(decision_times) * 1000),
                decision_p95_ms=float(np.percentile(decision_times, 95) * 1000)), frames


def evaluate_libero(runtime, output, suite="libero_spatial", trials=50, seed=42, task_ids=None,
                    execute_horizon=None, video=False, max_steps=None, min_free_gb=2.0, resume=False,
                    max_new_trials=None):
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:
        raise ImportError("Install LIBERO and robosuite using docs/libero.md") from error
    if runtime.metadata["action_semantics"] != ACTION_SEMANTICS:
        raise ValueError("LIBERO requires canonical delta-pose/open01 action semantics")
    if trials < 1 or suite not in MAX_STEPS or (max_steps is not None and max_steps < 1):
        raise ValueError("Invalid trial count or suite")
    if max_new_trials is not None and max_new_trials < 1:
        raise ValueError("max_new_trials must be positive")
    output = Path(output)
    check_disk(output, min_free_gb)
    output.mkdir(parents=True, exist_ok=True)
    suite_instance = benchmark.get_benchmark_dict()[suite]()
    ids = list(range(suite_instance.n_tasks)) if task_ids is None else task_ids
    if not ids or any(i < 0 or i >= suite_instance.n_tasks for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid or duplicate task IDs")
    executed = runtime.config.model.horizon if execute_horizon is None else execute_horizon
    if not 1 <= executed <= runtime.config.model.horizon:
        raise ValueError("Invalid executed horizon")
    import importlib.metadata
    config = dict(protocol_version="jointact-libero-v2", suite=suite, trials_per_task=trials, task_ids=ids, seed=seed,
                  max_steps=max_steps or MAX_STEPS[suite], settle_steps=10,
                  execute_horizon=executed, policy_config=runtime.config.to_dict(),
                  policy_identity=runtime.identity, precision=runtime.precision, device=str(runtime.device), video=video,
                  source_sha256=source_fingerprint()["sha256"],
                  environment={name: importlib.metadata.version(name) for name in ("mujoco", "robosuite", "torch", "numpy")},
                  training_data_sha256=runtime.metadata["manifest_sha256"],
                  observation=dict(num_images=runtime.config.model.num_images, proprio_dim=runtime.config.model.proprio_dim,
                                   image_size=runtime.config.model.image_size, crop_scale=runtime.config.data.crop_scale),
                  initial_state_assets={}, task_assets={})
    tasks = {}
    for task_id in ids:
        task = suite_instance.get_task(task_id)
        state_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        states = load_initial_states(state_path)
        config["initial_state_assets"][str(task_id)] = dict(file=task.init_states_file, sha256=sha256(state_path))
        if trials > len(states):
            raise ValueError(f"Requested {trials} trials but task {task_id} has only {len(states)} unique initial states")
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        config["task_assets"][str(task_id)] = dict(file=task.bddl_file, sha256=sha256(bddl), instruction=task.language)
        tasks[task_id] = (task, states, bddl)
    with evaluation_lock(output):
        ledger = TrialLedger(output, config, resume)
        added = 0
        for task_id, (task, states, bddl) in tasks.items():
            pending = [trial for trial in range(trials) if (task_id, trial) not in ledger.rows]
            if not pending:
                continue
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            try:
                for trial in pending:
                    check_disk(output, min_free_gb)
                    individual_seed = trial_seed(seed, task_id, trial)
                    seed_everything(individual_seed)
                    env.seed(individual_seed)
                    result, frames = run_episode(env, states[trial], runtime, task.language, config["max_steps"],
                                                  execute_horizon=executed, collect_frames=video)
                    row = dict(task_id=task_id, task=task.language, trial=trial, seed=seed,
                               trial_seed=individual_seed, protocol_sha256=json_digest(config), **result)
                    if video:
                        import imageio.v2 as imageio
                        path = output / f"task-{task_id:03d}-trial-{trial:03d}.mp4"
                        imageio.mimsave(path, frames, fps=20)
                        row["video"] = path.name
                    ledger.commit(row)
                    print(json.dumps(row), flush=True)
                    added += 1
                    if max_new_trials is not None and added >= max_new_trials:
                        return ledger.publish()
            finally:
                env.close()
        return ledger.publish()
