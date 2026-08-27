"""Turns reason codes into sentences.

The rule the whole design hangs on: the model receives reason codes and writes
prose about them. It does not receive the catalogue and it does not get to
decide anything. If a reason is not in the codes it cannot end up in the
explanation, which is why we can claim a zero hallucination rate on the "why
this course" panel and actually back it up in the notebook.

Every function here has a templated fallback so the app still explains itself
with the API turned off. The fallback is stiffer to read, not less true.
"""

from __future__ import annotations

import json

import numpy as np

from .. import llm
from ..catalog import Catalog
from ..schemas import LearnerProfile, LearningPath, Recommendation

SYSTEM = """You write short explanations for a learning path recommender.

You will be given a JSON object of reason codes produced by the recommender's
scoring function, plus a small amount of learner context. Write 2-3 sentences,
second person, plain English, no marketing voice, no bullet points.

Hard rules:
- Every claim must trace to a reason code you were given. If it is not in the
  codes, you do not know it and you must not say it.
- Do not invent course names, statistics, durations or outcomes.
- Do not hedge with "may" or "might" about things the codes state directly.
- If the codes are thin, write a shorter explanation rather than padding it.

Reason code meanings:
  GAP_COVERAGE   this course teaches skills the learner is missing for the goal
  PREREQ_FOR     it is required before another course already in the path
  NEEDS_FIRST    it assumes skills the learner does not yet have
  LEVEL_FIT      how well the difficulty matches current ability (0-1)
  TIME_FIT       hours, and weeks at the learner's stated pace
  PREFERENCE_FIT matches a stated format or budget preference
"""


def explain_recommendation(rec: Recommendation, profile: LearnerProfile, cat: Catalog) -> str:
    payload = {
        "course_title": rec.course.title,
        "provider": rec.course.provider,
        "goal": profile.goal_text,
        "weekly_hours": profile.weekly_hours,
        "reason_codes": [r.model_dump() for r in rec.reasons],
        "score_components": rec.components,
    }
    out = llm.text(json.dumps(payload, sort_keys=True), SYSTEM, effort="low", max_tokens=400)
    return out or _template_recommendation(rec, cat)


def _template_recommendation(rec: Recommendation, cat: Catalog) -> str:
    bits = []
    covers = [cat.name(s) for s in rec.covers[:3]]
    if covers:
        bits.append(
            f"Covers {len(rec.covers)} skill gap(s) for your goal, most importantly "
            + ", ".join(covers)
            + "."
        )
    for r in rec.reasons:
        if r.type == "PREREQ_FOR":
            bits.append(f"It is a prerequisite for {r.detail.get('course_title')} later in your path.")
        elif r.type == "TIME_FIT":
            bits.append(f"{r.detail['hours']:g} hours, about {r.detail['weeks']:g} weeks at your pace.")
        elif r.type == "NEEDS_FIRST":
            names = ", ".join(cat.name(s) for s in r.detail.get("skills", [])[:2])
            if names:
                bits.append(f"It assumes {names}, which is why it sits where it does.")
    return " ".join(bits) or "Selected because it moves you toward your stated goal."


def explain_rejection(
    course_id: str,
    path: LearningPath,
    profile: LearnerProfile,
    m: np.ndarray,
    cat: Catalog,
) -> str:
    """The "why not X?" answer.

    This is the thing a prompt-only system cannot do honestly, because the
    answer lives in the prerequisite graph and the budget, not in the text of
    the course description.
    """
    course = cat.course_by_id.get(course_id)
    if course is None:
        return "That course is not in our catalogue, so it was never a candidate."

    in_path = any(i.id == course_id for ms in path.milestones for i in ms.items)
    if in_path:
        return f"{course.title} is in your path already."

    m_end = _mastery_after(path, m, cat)
    unlock = _unlock_point(course, path, m, cat)

    # Report the blocker that is actually operative. If the path unlocks the
    # course later, the interesting missing skills are the ones you lack today.
    # If it never unlocks it, they are the ones the path still does not teach -
    # naming skills the path already covers is the classic wrong answer here.
    reference = m if unlock else m_end
    unmet = []
    for sid, need in course.requires.items():
        i = cat.index.get(sid)
        if i is not None and reference[i] < need - 0.05:
            unmet.append(
                {
                    "skill": cat.name(sid),
                    "needed": round(need, 2),
                    "current": round(float(reference[i]), 2),
                    "taught_by": [
                        cat.course_by_id[c].title
                        for c in cat.teachers.get(sid, [])[:1]
                    ],
                }
            )
    weeks = course.hours / max(profile.weekly_hours, 1.0)
    payload = {
        "question": f"why is {course.title} not in the path",
        "course_title": course.title,
        "unmet_prerequisites": unmet,
        "becomes_reachable_after": (
            {"milestone": unlock.title, "week": unlock.end_week} if unlock else None
        ),
        "hours": course.hours,
        "weeks_at_current_pace": round(weeks, 1),
        "horizon_weeks": profile.horizon_weeks,
        "time_unconstrained": profile.time_unconstrained,
        "budget_exceeded": not profile.time_unconstrained and weeks > profile.horizon_weeks * 0.6,
        "cost": course.cost,
        "learner_budget": profile.budget,
        "path_total_hours": path.total_hours,
    }
    system = SYSTEM + (
        "\nHere you are explaining why a course was NOT included. Be concrete about "
        "the blocking reason and, if there is one, name the point in the path where "
        "it would become reachable."
    )
    out = llm.text(json.dumps(payload, sort_keys=True), system, effort="low", max_tokens=400)
    if out:
        return out

    if unmet:
        names = ", ".join(u["skill"] for u in unmet[:2])
        if unlock:
            return (
                f"{course.title} assumes {names}, which you do not have yet. Your path covers "
                f"that in {unlock.title}, so it becomes a sensible next step around week "
                f"{unlock.end_week}."
            )
        if profile.time_unconstrained:
            return (
                f"{course.title} needs {names}, which the currently sourced route does not "
                f"teach yet. It needs an additional prerequisite resource first."
            )
        return (
            f"{course.title} needs {names}, which this path does not reach inside your "
            f"{profile.horizon_weeks}-week window. Extend the window or raise your weekly "
            f"hours and it comes into range."
        )
    if not profile.time_unconstrained and weeks > profile.horizon_weeks * 0.6:
        return (
            f"{course.title} is {course.hours:g} hours, roughly {weeks:.0f} weeks at "
            f"{profile.weekly_hours:g}h/week. It would eat most of your {profile.horizon_weeks}-week "
            f"window on its own, so shorter courses covering the same gaps won out."
        )
    return (
        f"{course.title} ranked below the courses that made the cut - it overlaps with skills "
        f"already covered elsewhere in your path."
    )


def _mastery_after(path: LearningPath, m: np.ndarray, cat: Catalog) -> np.ndarray:
    out = m.copy()
    for ms in path.milestones:
        for item in ms.items:
            if item.kind == "course":
                out = np.maximum(out, cat.teaches_vec(item.id) * 0.85)
    return out


def _unlock_point(course, path: LearningPath, m: np.ndarray, cat: Catalog):
    """Walk the path accumulating mastery and return the first milestone after
    which this course's prerequisites are all satisfied.

    This is the bit that makes "why not X" a real answer rather than a polite
    deflection - it comes out of the prerequisite graph and the path we already
    generated, so it is checkable.
    """
    current = m.copy()
    for ms in path.milestones:
        for item in ms.items:
            if item.kind == "course":
                current = np.maximum(current, cat.teaches_vec(item.id) * 0.85)
        if all(
            current[cat.index[sid]] >= need - 0.05
            for sid, need in course.requires.items()
            if sid in cat.index
        ):
            return ms
    return None


def write_project_brief(title: str, skills: list[str], cat: Catalog, level: str) -> str:
    names = [cat.name(s) for s in skills]
    system = (
        "You write short capstone project briefs for a learning path. You are given the "
        "exact list of skills the learner has just finished. Write 2-3 sentences describing "
        "one concrete project that exercises those skills and nothing else. State a "
        "deliverable. Do not mention tools or topics outside the given skills. No preamble."
    )
    out = llm.text(
        json.dumps({"phase": title, "skills": names, "level": level}, sort_keys=True),
        system,
        effort="low",
        max_tokens=300,
    )
    if out:
        return out
    return project_brief_template(title, skills, cat)


def project_brief_template(title: str, skills: list[str], cat: Catalog) -> str:
    """Fast, grounded project text used while initially building a path."""
    names = [cat.name(s) for s in skills]
    return (
        f"Build and ship something small that uses {', '.join(names[:3])}. "
        f"Deliverable: a repo with a README explaining your approach and what you would do "
        f"differently with more time."
    )


def summarise_path(path: LearningPath, profile: LearnerProfile) -> str:
    payload = {
        "role": path.role_title,
        "milestones": [
            {"title": m.title, "weeks": [m.start_week, m.end_week],
             "courses": [i.title for i in m.items if i.kind == "course"]}
            for m in path.milestones
        ],
        "total_hours": path.total_hours,
        "total_weeks": path.total_weeks,
        "readiness_before": path.readiness_before,
        "readiness_after": path.readiness_after,
        "coverage": path.coverage,
        "not_covered": [i for i in path.dropped],
        "weekly_hours": profile.weekly_hours,
        "time_unconstrained": profile.time_unconstrained,
    }
    system = (
        "Summarise a generated learning path in 3-4 sentences for the learner who asked "
        "for it. Second person, direct, no marketing language. Mention the shape of the "
        "path, the time commitment, and be honest about anything it does not cover. "
        "Only use the numbers you are given."
    )
    out = llm.text(json.dumps(payload, sort_keys=True), system, effort="low", max_tokens=400)
    if out:
        return out
    return summarise_path_template(path, profile)


def summarise_path_template(path: LearningPath, profile: LearnerProfile) -> str:
    """Deterministic summary for the initial route; rich prose stays on-demand."""
    phases = len(path.milestones)
    weeks = path.total_weeks
    return (
        f"{phases} phase{'' if phases == 1 else 's'}, {path.total_hours:g} hours, about "
        f"{weeks} week{'' if weeks == 1 else 's'} at an estimated {profile.weekly_hours:g}h/week pace. "
        f"It moves your readiness for {path.role_title} from "
        f"{path.readiness_before:.0%} to {path.readiness_after:.0%}."
    )
