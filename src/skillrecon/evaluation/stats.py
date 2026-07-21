"""Statistical helpers for experiment reporting."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def bootstrap_resample_ci(
    items: Sequence[T],
    statistic: Callable[[list[T]], float],
    *,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for an arbitrary statistic."""
    if not items:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(len(items))] for _ in range(len(items))]
        samples.append(statistic(sample))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = samples[int(alpha * (n_boot - 1))]
    upper = samples[int((1.0 - alpha) * (n_boot - 1))]
    return (lower, upper)


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean."""
    return bootstrap_resample_ci(
        list(values),
        lambda sample: sum(sample) / len(sample),
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
    )


def mcnemar_exact(
    outcomes_a: Sequence[bool],
    outcomes_b: Sequence[bool],
) -> float:
    """Exact McNemar p-value using the binomial tail on discordant pairs."""
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("McNemar inputs must have the same length")
    b01 = 0
    b10 = 0
    for a, b in zip(outcomes_a, outcomes_b, strict=True):
        if a and not b:
            b10 += 1
        elif b and not a:
            b01 += 1
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def holm_correction(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    """Apply Holm-Bonferroni correction and return rejection decisions."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    decisions: dict[str, bool] = {name: False for name in p_values}
    m = len(ordered)
    for index, (name, p_value) in enumerate(ordered, start=1):
        threshold = alpha / (m - index + 1)
        if p_value <= threshold:
            decisions[name] = True
            continue
        break
    return decisions
