"""Mechanism, mask, gradient and continuation checks for the two auxiliary objectives."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jointact.checkpoint import read_weights
from jointact.config import Config
from jointact.data.dataset import EpisodeDataset, ObservationCollator
from jointact.data.prepare import load_artifacts
from jointact.models import JointActionPolicy
from jointact.study import prepare_study
from jointact.training import train, validation_indices
from jointact.utils import sha256


def controlled(config):
    prototypes = np.broadcast_to(np.array([0., .04, .5, 2.])[:, None, None], (4, 3, 7)).copy()
    model = JointActionPolicy(config.model, prototypes)
    batch = dict(actions=torch.zeros(4, 3, 7), valid=torch.tensor([[True, True, False]] * 4))
    batch['actions'][:, -1] = 10000  # Padding must not change costs or reachability.
    logits = torch.full((4, 4), -2.)
    logits[torch.arange(4), torch.tensor([1, 2, 3, 0])] = 2.
    output = dict(logits=logits.requires_grad_(), target_mode=torch.zeros(4, dtype=torch.long),
                  actions=torch.zeros(4, 3, 7, requires_grad=True),
                  selected_actions=torch.full((4, 3, 7), .1, requires_grad=True))
    return model, batch, output


def test_alignment_gates_wrong_near_reachable_modes_and_padding(prepared):
    model, batch, output = controlled(prepared[2])
    loss, metrics = model.loss(output, batch, alignment_weight=1.)
    assert metrics['alignment_coverage'] == .25
    torch.testing.assert_close(metrics['alignment_loss'], torch.tensor(.025))
    loss.backward()
    grad = output['selected_actions'].grad
    assert grad[0, :2].abs().sum() > 0
    assert torch.count_nonzero(grad[1:]) == 0 and torch.count_nonzero(grad[:, -1]) == 0
    # A broad neighborhood still cannot override the bounded-residual reachability gate.
    _, metrics = model.loss(output, batch, alignment_weight=1., alignment_margin=10.)
    assert metrics['alignment_coverage'] == .5
    batch['valid'].zero_()
    loss, metrics = model.loss(output, batch, alignment_weight=1., action_cost_weight=1.)
    assert torch.isfinite(loss) and metrics['alignment_loss'] == 0


def test_cost_supplies_geometry_gradient_to_classifier_only(prepared):
    model, batch, output = controlled(prepared[2])
    output['logits'] = torch.zeros(4, 4, requires_grad=True)
    base, _ = model.loss(output, batch)
    augmented, _ = model.loss(output, batch, action_cost_weight=1.)
    cost = augmented - base
    grad, = torch.autograd.grad(cost, output['logits'])
    assert (grad[:, 0] < 0).all() and (grad[:, 3] > 0).all()
    torch.testing.assert_close(grad.sum(-1), torch.zeros(4), atol=1e-7, rtol=0)
    assert not model.head.prototypes.requires_grad


def test_alignment_encodes_observation_once_and_preserves_inference(prepared):
    root, artifacts, cfg = prepared
    _, prototypes = load_artifacts(artifacts)
    model = JointActionPolicy(cfg.model, prototypes).eval()
    data = EpisodeDataset(root, 3, 'train', artifacts)
    batch = ObservationCollator(cfg.model)([data[0], data[8]])
    calls = []
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    output = model(batch, supervised=True, align_predictions=True)
    assert len(calls) == 1
    hook.remove()
    torch.testing.assert_close(output['selected_actions'], model(batch)['actions'], atol=0, rtol=0)
    loss, _ = model.loss(output, batch, alignment_weight=.5, action_cost_weight=1.)
    loss.backward()
    assert model.head.classifier.weight.grad.abs().sum() > 0
    data.close()


def test_augmented_exact_resume_and_explicit_objective_fork(prepared, tmp_path):
    _, _, cfg = prepared
    cfg.train.alignment_weight, cfg.train.action_cost_weight = .5, 1.
    cfg.train.eval_sampling = 'uniform'
    full = train(cfg)
    split = copy.deepcopy(cfg)
    split.train.output = str(tmp_path / 'split')
    first = train(split, stop_after=2)
    parent_hash = sha256(Path(first['checkpoint']) / 'weights.pt')
    split.train.resume = first['checkpoint']
    continued = train(split)
    _, expected = read_weights(full['checkpoint'])
    _, actual = read_weights(continued['checkpoint'])
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    split.train.action_cost_weight = 2.
    with pytest.raises(ValueError, match='action_cost_weight'):
        train(split)
    split.train.resume = None
    split.train.fork_from = first['checkpoint']
    split.train.output = str(tmp_path / 'fork')
    fork = train(split, stop_after=3)
    assert fork['step'] == 3
    lineage = json.loads((Path(split.train.output) / 'lineage.json').read_text())
    assert lineage['parent_step'] == 2
    assert lineage['objective_changes']['action_cost_weight'] == {'before': 1., 'after': 2.}
    assert sha256(Path(first['checkpoint']) / 'weights.pt') == parent_hash
    split.train.resume = fork['checkpoint']
    assert train(split)['step'] == 4


def test_study_factorial_controls_and_validation_coverage(prepared, tmp_path):
    _, _, cfg = prepared
    path = tmp_path / 'base.yaml'
    cfg.save(path)
    study = tmp_path / 'study'
    prepare_study(path, study, seeds=[42], variants=['joint', 'aligned', 'cost', 'aligned_cost', 'regression'])
    expected = {'joint': (0., 0.), 'aligned': (.5, 0.), 'cost': (0., 1.), 'aligned_cost': (.5, 1.), 'regression': (0., 0.)}
    for name, weights in expected.items():
        item = Config.load(study / 'configs' / f'{name}-seed42.yaml')
        assert (item.train.alignment_weight, item.train.action_cost_weight) == weights
        assert item.train.eval_sampling == 'uniform'
    indices = validation_indices(4380, 100, 'uniform')
    assert len(set(indices)) == 100 and indices[0] < 44 and indices[-1] > 4336
    assert validation_indices(9, 100, 'uniform') == list(range(9))
    assert validation_indices(4380, 100, 'prefix') == list(range(100))


def test_alignment_config_rejects_incompatible_or_nonfinite():
    for raw in [dict(model=dict(head_type='regression'), train=dict(action_cost_weight=1)),
                dict(model=dict(residual_enabled=False), train=dict(alignment_weight=1)),
                dict(train=dict(alignment_margin=float('nan')))]:
        with pytest.raises(ValueError):
            Config.from_dict(raw)


def test_conditional_error_aggregation_excludes_empty_microbatches(prepared, monkeypatch):
    _, _, cfg = prepared
    cfg.train.alignment_weight = .5
    original = JointActionPolicy.loss
    calls = []
    def controlled_metrics(self, *args, **kwargs):
        loss, metrics = original(self, *args, **kwargs)
        calls.append(1)
        metrics['_alignment_error_sum'] = torch.tensor(10. if len(calls) == 1 else 0.)
        metrics['_alignment_element_count'] = torch.tensor(2. if len(calls) == 1 else 0.)
        return loss, metrics
    monkeypatch.setattr(JointActionPolicy, 'loss', controlled_metrics)
    train(cfg, stop_after=1)
    rows = [json.loads(line) for line in (Path(cfg.train.output) / 'metrics.jsonl').read_text().splitlines()]
    row = next(row for row in rows if row['kind'] == 'train')
    assert row['alignment_eligible_l1'] == 5.
    assert '_alignment_error_sum' not in row
