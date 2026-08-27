"""Skill gap: where the learner is vs what the target role needs.

The gap is importance-weighted on purpose. An unweighted gap sends people off
to learn a nice-to-have before a must-have, because the raw distance happens
to be bigger. Weighting by how much the market cares fixes that, and it is
also what makes the "you are 34% ready" number mean anything.
"""

from __future__ import annotations

import numpy as np

from ..catalog import Catalog
from ..schemas import GapItem


def gap_vector(m: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, target - m) * weight


def readiness(m: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    """1.0 = every weighted requirement met. Clamped so overshooting one skill
    cannot compensate for missing another."""
    total = float((np.minimum(target, target) * weight).sum())
    if total <= 0:
        return 1.0
    have = float((np.minimum(m, target) * weight).sum())
    return round(have / total, 4)


def gap_items(m: np.ndarray, target: np.ndarray, weight: np.ndarray, cat: Catalog) -> list[GapItem]:
    g = gap_vector(m, target, weight)
    items = []
    for i, skill in enumerate(cat.skills):
        if g[i] <= 1e-6:
            continue
        items.append(
            GapItem(
                skill_id=skill.id,
                skill_name=skill.name,
                domain=skill.domain,
                current=round(float(m[i]), 3),
                target=round(float(target[i]), 3),
                weight=round(float(weight[i]), 3),
                gap=round(float(g[i]), 4),
            )
        )
    items.sort(key=lambda x: x.gap, reverse=True)
    return items


def strengths(m: np.ndarray, target: np.ndarray, weight: np.ndarray, cat: Catalog, n: int = 6):
    """Skills the learner already meets. Worth surfacing - a roadmap that only
    shows what you lack is demoralising, and it also tells them we read their
    history properly."""
    out = []
    for i, skill in enumerate(cat.skills):
        if weight[i] > 0 and target[i] > 0 and m[i] >= target[i] * 0.9:
            out.append((float(m[i] * weight[i]), skill.id, skill.name))
    out.sort(reverse=True)
    return [{"skill_id": sid, "skill_name": name} for _, sid, name in out[:n]]
