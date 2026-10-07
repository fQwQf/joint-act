"""Quantile normalization with an explicit mask and inverse transform."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Normalizer:
    low: np.ndarray
    high: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.low = np.asarray(self.low, np.float32)
        self.high = np.asarray(self.high, np.float32)
        self.mask = np.asarray(self.mask, bool)
        if self.low.ndim != 1 or self.low.shape != self.high.shape or self.mask.shape != self.low.shape:
            raise ValueError("Normalizer dimensions do not match")
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all() or (self.high < self.low).any():
            raise ValueError("Invalid normalization bounds")

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, np.float32)
        scale = np.maximum(self.high - self.low, 1e-6)
        output = 2 * (values - self.low) / scale - 1
        # Constant dimensions map to zero; inverse returns their physical constant.
        output = np.where(self.high - self.low < 1e-6, 0, output)
        return np.where(self.mask, np.clip(output, -1, 1), values).astype(np.float32)

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, np.float32)
        output = (values + 1) * 0.5 * (self.high - self.low) + self.low
        return np.where(self.mask, output, values).astype(np.float32)

    def to_dict(self) -> dict:
        return dict(low=self.low.tolist(), high=self.high.tolist(), mask=self.mask.tolist())

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray | None = None) -> "Normalizer":
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Statistics need finite nonempty [N,D] training values")
        return cls(np.quantile(values, 0.01, axis=0), np.quantile(values, 0.99, axis=0),
                   np.ones(values.shape[-1], bool) if mask is None else mask)
