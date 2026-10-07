import pytest
import torch

from jointact.config import Config
from jointact.data.prepare import prepare_artifacts
from jointact.fixture import create_fixture


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "data"
    create_fixture(root, episodes=8, length=9, image_size=16)
    artifacts = root / "artifacts"
    prepare_artifacts(root, artifacts, horizon=3, num_modes=4, max_samples=100, iterations=4)
    config = Config.from_dict(dict(model=dict(hidden_dim=16, head_dim=32, horizon=3, num_modes=4,
                                              image_size=16, gradient_checkpointing=False),
                                  data=dict(root=str(root), artifacts=str(artifacts), image_aug=False),
                                  train=dict(output=str(tmp_path / "run"), device="cpu", precision="fp32",
                                             max_steps=4, warmup_steps=1, learning_rate=0.003,
                                             batch_size=5, grad_accumulation=2, log_every=1, eval_every=4,
                                             save_every=4, eval_batches=2, min_free_disk_gb=0)))
    return root, artifacts, config
