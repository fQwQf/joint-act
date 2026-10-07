"""Deterministic bounded-memory k-means over complete joint action chunks."""

import numpy as np


def squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum((values * values).sum(1, keepdims=True) + (centers * centers).sum(1)[None] - 2 * values @ centers.T, 0)


def fit_codebook(chunks: np.ndarray, num_modes: int, iterations: int = 40, seed: int = 42) -> tuple[np.ndarray, dict]:
    if chunks.ndim != 3 or len(chunks) < num_modes or not np.isfinite(chunks).all():
        raise ValueError("Need at least num_modes finite training chunks of shape [N,H,D]")
    if iterations < 1 or num_modes < 1:
        raise ValueError("iterations and num_modes must be positive")
    shape = chunks.shape[1:]
    x = chunks.reshape(len(chunks), -1).astype(np.float32)
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))].copy()]
    nearest = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, num_modes):
        index = rng.choice(len(x), p=nearest / nearest.sum()) if nearest.sum() > 1e-12 else rng.integers(len(x))
        centers.append(x[index].copy())
        nearest = np.minimum(nearest, ((x - centers[-1]) ** 2).sum(1))
    centers = np.stack(centers)
    for iteration in range(iterations):
        sums = np.zeros_like(centers)
        counts = np.zeros(num_modes, np.int64)
        error = 0.0
        for start in range(0, len(x), 1024):
            batch = x[start:start + 1024]
            distances = squared_distances(batch, centers)
            labels = distances.argmin(1)
            error += float(distances[np.arange(len(batch)), labels].sum())
            np.add.at(sums, labels, batch)
            np.add.at(counts, labels, 1)
        new = sums / np.maximum(counts[:, None], 1)
        for empty in np.flatnonzero(counts == 0):
            new[empty] = x[rng.integers(len(x))]
        shift = float(np.max(np.abs(new - centers)))
        centers = new.astype(np.float32)
        if shift < 1e-5:
            break
    return centers.reshape(num_modes, *shape), dict(iterations=iteration + 1, train_quantization_mse=error / x.size,
                                                   counts=counts.tolist(), seed=seed, num_fit_chunks=len(chunks))


def nearest_modes(actions, valid, prototypes):
    """Torch assignment ignores padded tail timesteps."""
    delta = actions[:, None].float() - prototypes[None].float()
    distance = (delta.square() * valid[:, None, :, None]).sum((-1, -2))
    return distance.argmin(-1)
