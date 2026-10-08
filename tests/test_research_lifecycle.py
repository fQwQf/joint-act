import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jointact.config import Config
from jointact.data.prepare import load_artifacts
from jointact.evaluation.records import TrialLedger, compare_libero, evaluation_lock, trial_seed
from jointact.inference import PolicyRuntime, benchmark_dataset
from jointact.models import JointActionPolicy
from jointact.study import prepare_study, run_study
from jointact.training import train
from jointact.utils import atomic_json, check_gpu, json_digest


def protocol(weight="weight-a"):
    return dict(protocol_version="jointact-libero-v2", suite="libero_spatial", trials_per_task=2,
                task_ids=[0], seed=42, max_steps=220, settle_steps=10, execute_horizon=3,
                observation=dict(num_images=2, proprio_dim=8, image_size=16, crop_scale=.9),
                initial_state_assets={"0": {"sha256": "states"}}, task_assets={"0": {"sha256": "bddl"}},
                precision="fp32", policy_identity={"weights_sha256": weight}, policy_config={},
                training_data_sha256="data", source_sha256="source")


def trial(config, index, success):
    return dict(task_id=0, trial=index, trial_seed=trial_seed(config["seed"], 0, index),
                protocol_sha256=json_digest(config), steps=12, success=success)


def test_resume_recovers_committed_trials_without_trusting_summary(tmp_path):
    config = protocol()
    with evaluation_lock(tmp_path):
        ledger = TrialLedger(tmp_path, config)
        ledger.commit(trial(config, 0, True))
        atomic_json(tmp_path / "summary.json", {"completed": True, "success_rate": 1})
        (tmp_path / "trials" / ".interrupted.tmp").write_text('{"partial":')
        resumed = TrialLedger(tmp_path, config, resume=True)
        assert not resumed.publish()["completed"]
        assert (0, 0) in resumed.rows and (0, 1) not in resumed.rows
        resumed.commit(trial(config, 1, False))
        assert resumed.publish()["completed"] and resumed.publish()["success_rate"] == .5
        with pytest.raises(ValueError, match="different weights"):
            TrialLedger(tmp_path, protocol("changed-weights"), resume=True)
        with pytest.raises(ValueError, match="already recorded"):
            resumed.commit(trial(config, 0, True))
        with pytest.raises(RuntimeError, match="Another evaluator"):
            with evaluation_lock(tmp_path):
                pass


def test_comparison_rejects_incomplete_or_unmatched_evidence(tmp_path):
    runs = {}
    for label, outcomes in (("a", [True, False]), ("b", [False, False])):
        directory = tmp_path / label
        config = protocol(label)
        with evaluation_lock(directory):
            ledger = TrialLedger(directory, config)
            ledger.commit(trial(config, 0, outcomes[0]))
            runs[label] = directory
            if label == "b":
                with pytest.raises(ValueError, match="incomplete"):
                    compare_libero(runs, tmp_path / "comparison.json")
            ledger.commit(trial(config, 1, outcomes[1]))
    result = compare_libero(runs, tmp_path / "comparison.json")
    assert result["pairs"][0]["paired_success_delta_a_minus_b"] == .5
    assert result["pairs"][0]["a_only_successes"] == 1
    cfg = protocol("b")
    cfg["execute_horizon"] = 1
    atomic_json(runs["b"] / "config.json", cfg)
    for index in range(2):
        atomic_json(runs["b"] / "trials" / f"task-000-trial-{index:03d}.json", trial(cfg, index, False))
    with pytest.raises(ValueError, match="unmatched evaluation controls"):
        compare_libero(runs, tmp_path / "comparison.json")


def test_prototype_ablation_freezes_and_bypasses_residual(prepared):
    _, artifacts, config = prepared
    _, prototypes = load_artifacts(artifacts)
    config.model.residual_enabled = False
    model = JointActionPolicy(config.model, prototypes)
    assert all(not p.requires_grad for p in model.head.residual.parameters())
    features = torch.randn(2, model.backbone.hidden_dim, requires_grad=True)
    output = model.head(features)
    torch.testing.assert_close(output["actions"], model.head.prototypes[output["mode"]], rtol=0, atol=0)
    output["logits"].sum().backward()
    assert features.grad is not None and model.head.mode_embedding.weight.grad is None
    trained = train(config, stop_after=1)
    loaded = PolicyRuntime(trained["checkpoint"])
    assert not loaded.model.head.residual_enabled


def test_training_state_checksum_and_dataset_benchmark(prepared, tmp_path):
    root, _, config = prepared
    result = train(config, stop_after=1)
    runtime = PolicyRuntime(result["checkpoint"])
    report = benchmark_dataset(runtime, tmp_path / "benchmark.json", root, observations=2, warmup=0,
                               repeats=2, execute_horizon=1)
    assert len(report["duration_ms"]) == 2 and report["policy_identity"] == runtime.identity
    assert report["executed_actions_per_second"] == 1000 / report["mean_ms"]
    assert report["peak_allocated_mib"] is None
    with open(Path(result["checkpoint"]) / "training.pt", "ab") as stream:
        stream.write(b"damage")
    config.train.resume = result["checkpoint"]
    with pytest.raises(ValueError, match="training-state checksum"):
        train(config)


def test_low_memory_foreign_gpu_process_is_still_occupied(monkeypatch):
    from jointact import utils
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setattr(utils, "gpu_inventory", lambda: [dict(index=6, uuid="GPU-x", used_mib=399, utilization=0)])
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=f"GPU-x, {os.getpid()+1}\n"))
    with pytest.raises(RuntimeError, match="compute processes"):
        check_gpu("cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="hidden"):
        check_gpu("cuda:0")


def test_controlled_study_runs_and_resumes(prepared, tmp_path):
    _, _, config = prepared
    base = tmp_path / "base.yaml"
    config.train.max_steps = 2
    config.save(base)
    result = prepare_study(base, tmp_path / "study", seeds=[3], variants=["joint", "regression"])
    with open(result["plan"]) as stream:
        plan = json.load(stream)
    configs = [Config.load(tmp_path / "study" / run["config"]) for run in plan["runs"]]
    assert configs[0].data == configs[1].data
    lhs, rhs = copy.deepcopy(configs[0].model), copy.deepcopy(configs[1].model)
    rhs.head_type = lhs.head_type
    assert lhs == rhs
    report = run_study(result["plan"])
    assert len(report["completed"]) == 2
    for cfg in configs:
        output = Path(cfg.train.output)
        with open(output / "test.json") as stream:
            assert json.load(stream)["evaluated_action_elements"] > 0
        assert (output / "bundle" / "weights.pt").is_file()
    before = (Path(configs[0].train.output) / "metrics.jsonl").read_bytes()
    run_study(result["plan"], max_runs=1)
    assert (Path(configs[0].train.output) / "metrics.jsonl").read_bytes() == before


def test_diagnostic_oracle_does_not_replace_deployed_actions(prepared):
    root, artifacts, config = prepared
    from jointact.data.dataset import EpisodeDataset, ObservationCollator
    from jointact.training import evaluate
    from torch.utils.data import DataLoader
    _, prototypes = load_artifacts(artifacts)
    model = JointActionPolicy(config.model, prototypes)
    data = EpisodeDataset(root, config.model.horizon, "val", artifacts)
    loader = DataLoader(data, batch_size=4, collate_fn=ObservationCollator(config.model))
    report = evaluate(model, loader, torch.device("cpu"))
    diagnostics = report["diagnostics"]
    assert diagnostics["oracle_is_deployable"] is False
    assert diagnostics["residual_abs_mean"] == 0
    assert np.isclose(diagnostics["quantization_l1"], diagnostics["oracle_mode_action_l1"])
    assert np.isclose(diagnostics["predicted_prototype_l1"], report["action_l1"])
    assert sum(diagnostics["predicted_mode_counts"]) == len(data)
    data.close()
