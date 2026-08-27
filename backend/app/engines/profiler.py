"""Turns a learner profile into a mastery vector m in [0,1]^S.

Four evidence sources, in increasing order of how much we trust them:

  self-report   -> shrunk hard. People are bad at this in both directions.
  course history-> decent, weighted by how much the course actually teaches.
  parent credit -> if you can do CNNs you know something about neural nets.
  diagnostic    -> the only thing we actually measured, so it dominates.

There is no time decay yet (see README, Known gaps) - the profile only stores
course ids, not verified completion dates.
"""

from __future__ import annotations

import numpy as np

from ..catalog import Catalog
from ..schemas import COMPLETION_CREDIT, LearnerProfile

SELF_REPORT_SHRINK = 0.6      # a confident 5/5 becomes 0.6, not 1.0
PARENT_CREDIT = 0.55
QUIZ_WEIGHT = 0.7             # how much a measured result overrides the priors


def mastery(profile: LearnerProfile, cat: Catalog) -> np.ndarray:
    m = np.zeros(cat.n_skills, dtype=np.float32)

    # 1. self-reported
    for claim in profile.known_skills:
        i = cat.index.get(claim.skill_id)
        if i is None:
            continue
        m[i] = max(m[i], (claim.self_rating / 5.0) * SELF_REPORT_SHRINK)

    # 2. completed courses
    for cid in profile.completed_courses:
        if cid not in cat.course_by_id:
            continue
        m = np.maximum(m, cat.teaches_vec(cid) * COMPLETION_CREDIT)

    # 3. parent credit - one pass is enough for a two-level taxonomy
    for skill in cat.skills:
        if not skill.parent:
            continue
        pi, ci = cat.index.get(skill.parent), cat.index[skill.id]
        if pi is None:
            continue
        m[pi] = max(m[pi], m[ci] * PARENT_CREDIT)

    # 4. diagnostic results override the priors
    for sid, score in profile.quiz_results.items():
        i = cat.index.get(sid)
        if i is None:
            continue
        m[i] = (1 - QUIZ_WEIGHT) * m[i] + QUIZ_WEIGHT * float(np.clip(score, 0.0, 1.0))

    return np.clip(m, 0.0, 1.0)


def probe_skills(
    profile: LearnerProfile,
    target: np.ndarray,
    weight: np.ndarray,
    cat: Catalog,
    n: int = 6,
) -> list[str]:
    """Pick which skills the diagnostic should ask about.

    Information gain, roughly: ask about skills that matter for the goal AND
    where our estimate is least trustworthy. A skill we have measured is
    worthless to ask about; a skill the learner merely claimed is the most
    valuable question we can ask.
    """
    m = mastery(profile, cat)
    claimed = {c.skill_id for c in profile.known_skills}

    scored: list[tuple[float, str]] = []
    for skill in cat.skills:
        i = cat.index[skill.id]
        if weight[i] <= 0:
            continue                      # not relevant to the target role
        if skill.id in profile.quiz_results:
            continue                      # already measured

        # uncertainty peaks at m=0.5 and is higher for self-reported claims
        uncertainty = 1.0 - abs(m[i] - 0.5) * 2.0
        if skill.id in claimed:
            uncertainty = min(1.0, uncertainty + 0.35)

        relevance = weight[i] * max(target[i] - m[i], 0.15)
        scored.append((float(relevance * (0.4 + uncertainty)), skill.id))

    scored.sort(reverse=True)
    return [sid for _, sid in scored[:n]]
