"""Feedback handling. What makes the thing adaptive rather than a one-shot
generator.

Three separate updates, deliberately kept apart because they have different
half-lives:

  mastery      - "completed", or a checkpoint result. Changes what they know.
  preferences  - "too easy", "not interested". Changes how we rank.
  bandit arms  - which content format actually gets finished. Changes defaults.

The bandit is Beta-Bernoulli per format with Thompson sampling. It is small,
but it is a real online learner: two people with identical profiles end up
with differently shaped paths after a week of use, which is the point.
"""

from __future__ import annotations

import random

from ..catalog import Catalog
from ..schemas import Feedback, LearnerProfile

FORMATS = ["video", "text", "interactive", "project"]

# Nudges applied to the scoring weights. Small on purpose - one grumpy click
# should tilt the ranking, not rewrite it.
STEP = 0.04


def apply(profile: LearnerProfile, fb: Feedback, cat: Catalog, arms: dict[str, list[float]]) -> str:
    """Mutates profile and arms in place. Returns a short line for the UI so
    the learner can see that the feedback did something - silent adaptation
    reads as no adaptation."""
    course = cat.course_by_id.get(fb.item_id)
    fmt = course.format if course else None

    if fb.signal == "completed":
        if fb.item_id not in profile.completed_items:
            profile.completed_items.append(fb.item_id)
        if course and fb.item_id not in profile.completed_courses:
            profile.completed_courses.append(fb.item_id)
        _bump_arm(arms, fmt, success=True)
        return "Marked complete - your skill profile and the rest of the path have been updated."

    if fb.signal == "too_easy":
        # they know more than we thought about what this teaches
        if course:
            for sid, strength in course.teaches.items():
                prior = profile.quiz_results.get(sid, 0.4)
                profile.quiz_results[sid] = min(1.0, prior + 0.3 * strength)
        _adjust(profile, "level", +STEP)
        return "Noted - raising the difficulty of what comes next."

    if fb.signal == "too_hard":
        if course:
            for sid, strength in course.teaches.items():
                prior = profile.quiz_results.get(sid, 0.4)
                profile.quiz_results[sid] = max(0.0, prior - 0.25 * strength)
        _adjust(profile, "level", +STEP)      # level fit matters more to this learner
        _adjust(profile, "gap", -STEP / 2)
        return "Noted - I will slot in something gentler before this."

    if fb.signal == "not_interested":
        if fb.item_id not in profile.rejected_courses:
            profile.rejected_courses.append(fb.item_id)
        _bump_arm(arms, fmt, success=False)
        _adjust(profile, "preference", +STEP)
        return "Removed, and I will weight your preferences more heavily from here."

    if fb.signal == "skipped":
        if fb.item_id not in profile.rejected_courses:
            profile.rejected_courses.append(fb.item_id)
        _bump_arm(arms, fmt, success=False)
        return "Skipped. The path has been re-planned around it."

    if fb.signal == "loved_it":
        _bump_arm(arms, fmt, success=True)
        _adjust(profile, "preference", +STEP / 2)
        return "Good - more like this."

    return "Recorded."


def _adjust(profile: LearnerProfile, key: str, delta: float) -> None:
    from ..config import settings

    current = profile.weight_overrides.get(key, getattr(settings, f"w_{key}"))
    profile.weight_overrides[key] = max(0.02, min(0.6, current + delta))


def _bump_arm(arms: dict[str, list[float]], fmt: str | None, success: bool) -> None:
    if not fmt:
        return
    a, b = arms.setdefault(fmt, [1.0, 1.0])
    if success:
        arms[fmt] = [a + 1.0, b]
    else:
        arms[fmt] = [a, b + 1.0]


def preferred_formats(arms: dict[str, list[float]], n: int = 2) -> list[str]:
    """Thompson sampling over the format arms. Sampled rather than argmax so a
    format that got one bad review early can still come back."""
    draws = []
    for fmt in FORMATS:
        a, b = arms.get(fmt, [1.0, 1.0])
        draws.append((random.betavariate(a, b), fmt))
    draws.sort(reverse=True)
    return [fmt for _, fmt in draws[:n]]
