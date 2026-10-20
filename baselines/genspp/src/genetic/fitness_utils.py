from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def safe_fitness(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0.0 else 0.0


def normalized_fitness_weights(values: Iterable, *, min_positive: int = 1) -> list[float] | None:
    weights = np.asarray([safe_fitness(value) for value in values], dtype=np.float64)
    if weights.size == 0:
        return None
    if np.count_nonzero(weights > 0.0) < int(min_positive):
        return None
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return None
    probabilities = weights / total
    if not np.isfinite(probabilities).all():
        return None
    return probabilities.tolist()
