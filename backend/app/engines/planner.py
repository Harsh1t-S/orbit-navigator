"""Path generation. This is the part that is not a language model.

Three stages:

  1. Selection - which courses. Covering a weighted skill gap under an hours
     budget is budgeted maximum coverage. The objective is submodular, so the
     greedy "best marginal coverage per hour" rule has the standard 1-1/e
     guarantee. That is good enough and it runs in milliseconds.

  2. Ordering - what order. Build the induced prerequisite DAG over the chosen
     courses and topologically sort it, breaking ties by difficulty then by how
     urgent the skill is. Because the order comes out of the DAG, a path can
     never put a course before its prerequisite. Prerequisite violations are
     zero by construction, not by luck, and the eval notebook checks it.

  3. Shaping - milestones, projects, assessments. Bucket the ordered list by
     the learner's weekly hours, then close each phase with a project and a
     checkpoint over exactly the skills that phase taught.

Deliberately no LLM in here. The planner has to be deterministic or we cannot
test it, and "the model chose the order" is not an explanation anyone can
audit.
"""

from __future__ import annotations

import datetime as dt
import logging

import networkx as nx
import numpy as np

from ..catalog import Catalog
from ..config import settings
from ..schemas import (
    COMPLETION_CREDIT,
    GapItem,
    LearnerProfile,
    LearningPath,
    Milestone,
    PathItem,
    ReasonCode,
    Recommendation,
)
from . import gap as gap_engine
from .profiler import mastery

log = logging.getLogger(__name__)

WEEKS_PER_MILESTONE = 4


def build_path(
    profile: LearnerProfile,
    cat: Catalog,
    retriever,
    role_id: str | None = None,
) -> LearningPath:
    role = cat.role_by_id.get(role_id or "") or cat.resolve_role(profile.target_role) \
        or cat.resolve_role(profile.goal_text)
    if role is None:
        role = cat.roles[0]
        notes = [f"Could not match a role to the stated goal, defaulted to {role.title}."]
    else:
        notes = []

    target, weight = cat.role_target(role)
    m0 = mastery(profile, cat)
    gap0 = gap_engine.gap_vector(m0, target, weight)
    gap_before = gap_engine.gap_items(m0, target, weight, cat)

    # Projects and checkpoints are real hours the learner has to spend, so the
    # course selection only gets to spend part of the window. 0.78 lines up
    # with the 25%-of-phase project plus one hour per checkpoint below.
    if profile.time_unconstrained:
        available = float("inf")
        budget_hours = float("inf")
    else:
        available = max(profile.weekly_hours, 1.0) * max(profile.horizon_weeks, 1)
        budget_hours = available * 0.78

    chosen, dropped, spent = _select(profile, cat, retriever, m0, gap0, budget_hours)

    # Reserve room for project work in the normal greedy pass.  That reserve is
    # deliberately conservative for a multi-course route, but it must not hide
    # a single starter bundle that *does* fit once its real project/checkpoint
    # hours are included.  Without this escape hatch a 124-hour course in a
    # 156-hour window is discarded at the 78% selection cap even though its
    # fully shaped phase is exactly 156 hours.
    if not chosen:
        chosen = _single_bundle_fallback(profile, cat, retriever, m0, gap0, available)
        if chosen:
            spent = sum(rec.course.hours for rec in chosen)
            remaining = gap0.copy()
            for rec in chosen:
                remaining = np.maximum(0.0, remaining - cat.teaches_vec(rec.course.id) * CREDIT)
            dropped = _dropped_skills(remaining, cat)
    ordered = _order(chosen, cat, gap0)
    milestones = _bucket(ordered, cat, profile, gap0)

    # what the learner ends up with if they finish everything
    m1 = m0.copy()
    for cid in [r.course.id for r in ordered]:
        m1 = np.maximum(m1, cat.teaches_vec(cid) * CREDIT)
    m1 = np.clip(m1, 0.0, 1.0)

    gap_after = gap_engine.gap_vector(m1, target, weight)
    closed = float(gap0.sum() - gap_after.sum())
    coverage = closed / float(gap0.sum()) if gap0.sum() > 0 else 1.0

    total_hours = sum(i.hours for ms in milestones for i in ms.items)

    # Be honest when the stated window does not fit the goal. Quietly handing
    # someone a 51-week plan they asked to fit in 20 is worse than telling them.
    if dropped and coverage < 0.9 and profile.time_unconstrained:
        notes.append(
            "This is an open-ended route. The remaining requirements were not covered by "
            "the current set of sourced resources, rather than being cut to fit a deadline."
        )
    elif dropped and coverage < 0.9:
        remaining_hours = _estimate_remaining_hours(dropped, cat)
        extra_weeks = round(remaining_hours / max(profile.weekly_hours, 1.0))
        notes.append(
            f"This plan is what {profile.weekly_hours:g}h/week for {profile.horizon_weeks} weeks "
            f"actually buys you. Full coverage of the role needs roughly {extra_weeks} more weeks "
            f"at that pace - raise your hours or extend the window and I will re-plan."
        )

    return LearningPath(
        learner_id=profile.learner_id,
        target_role=role.id,
        role_title=role.title,
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        milestones=milestones,
        total_hours=round(total_hours, 1),
        total_weeks=milestones[-1].end_week if milestones else 0,
        readiness_before=gap_engine.readiness(m0, target, weight),
        readiness_after=gap_engine.readiness(m1, target, weight),
        gap_before=gap_before[:20],
        coverage=round(coverage, 4),
        dropped=dropped,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 1. selection
# ---------------------------------------------------------------------------

CREDIT = COMPLETION_CREDIT


def _select(
    profile: LearnerProfile,
    cat: Catalog,
    retriever,
    m: np.ndarray,
    gap: np.ndarray,
    budget_hours: float,
) -> tuple[list[Recommendation], list[str], float]:
    """Greedy budgeted maximum coverage.

    Value per course is marginal gap covered per hour, nudged by source
    specificity and preference terms when two resources cover the same skills.

    Prerequisites are resolved as part of the pick and charged to the same
    budget. Doing it as a post-pass (which is what we tried first) quietly blew
    the hour budget by 90% - a course that needs three prerequisites is not a
    cheap course, and the selection has to know that when it compares options.
    """
    remaining = gap.copy()
    floor = float(gap.sum()) * settings.gap_tolerance

    chosen: list[Recommendation] = []
    have = m.copy()
    spent = 0.0
    query = profile.goal_text or ""

    while (
        float(remaining.sum()) > floor
        and spent < budget_hours
        and len(chosen) < settings.max_path_courses
    ):
        ranked = retriever.rank(
            profile, m, remaining, query,
            selected=[r.course.id for r in chosen],
            limit=25,
        )
        chosen_ids = {r.course.id for r in chosen}

        picked = None
        picked_prereqs: list[Recommendation] = []
        best_value = 0.0

        for rec in ranked:
            if rec.course.id in chosen_ids:
                continue

            prereqs = _prereq_courses(rec.course, cat, have, chosen_ids, profile)
            if prereqs is None:
                continue  # cannot satisfy a prerequisite from the catalogue

            bundle_hours = rec.course.hours + sum(p.course.hours for p in prereqs)
            if spent + bundle_hours > budget_hours:
                continue

            gained = np.zeros_like(remaining)
            for c in [rec.course.id] + [p.course.id for p in prereqs]:
                gained = np.maximum(gained, cat.teaches_vec(c) * CREDIT)
            marginal = float(np.minimum(gained, remaining).sum())
            if marginal <= 1e-3:
                continue

            value = (marginal / bundle_hours) * (0.75 + 0.25 * rec.score)
            if value > best_value:
                picked, picked_prereqs, best_value = rec, prereqs, value

        if picked is None:
            break

        for item in picked_prereqs + [picked]:
            chosen.append(item)
            spent += item.course.hours
            teaches = cat.teaches_vec(item.course.id) * CREDIT
            remaining = np.maximum(0.0, remaining - teaches)
            have = np.maximum(have, teaches)

    dropped = _dropped_skills(remaining, cat)
    return chosen, dropped, spent


def _dropped_skills(remaining: np.ndarray, cat: Catalog) -> list[str]:
    return [
        cat.skills[i].id
        for i in np.argsort(-remaining)[:8]
        if remaining[i] > 0.05
    ]


def _single_bundle_fallback(
    profile: LearnerProfile,
    cat: Catalog,
    retriever,
    m: np.ndarray,
    gap: np.ndarray,
    available_hours: float,
) -> list[Recommendation]:
    """Return the best startable course bundle that fits its *real* shaped
    time cost, but only when the regular reserved-budget pass found nothing.

    The fixed 78% reserve is right for a rich route with several projects. A
    single long course, however, only needs one project and checkpoint, so use
    the existing milestone shaper to test exact fit instead of widening the
    regular budget and accidentally overbooking multi-course plans.
    """
    ranked = retriever.rank(profile, m, gap, profile.goal_text or "", limit=25)
    best: list[Recommendation] = []
    best_value = 0.0

    for rec in ranked:
        prereqs = _prereq_courses(rec.course, cat, m, set(), profile)
        if prereqs is None:
            continue

        bundle = prereqs + [rec]
        # `_prereq_courses` normally prevents duplicates. Keep this defensive
        # because a bundle is about to be shaped into visible checklist items.
        unique = {item.course.id: item for item in bundle}
        bundle = list(unique.values())
        ordered = _order(bundle, cat, gap)
        shaped = _bucket(ordered, cat, profile, gap)
        total = sum(item.hours for milestone in shaped for item in milestone.items)
        if not shaped or total > available_hours + 1e-6:
            continue

        gained = np.zeros_like(gap)
        for item in ordered:
            gained = np.maximum(gained, cat.teaches_vec(item.course.id) * CREDIT)
        marginal = float(np.minimum(gained, gap).sum())
        value = marginal / max(total, 1.0) * (0.75 + 0.25 * rec.score)
        if value > best_value:
            best, best_value = bundle, value

    if best:
        log.info(
            "planner used exact-fit starter fallback: %d resource(s) in %.1f available hours",
            len(best), available_hours,
        )
    return best


def _prereq_courses(
    course,
    cat: Catalog,
    have: np.ndarray,
    chosen_ids: set[str],
    profile: LearnerProfile,
    depth: int = 0,
) -> list[Recommendation] | None:
    """Cheapest set of courses that unlocks `course`.

    Cheapest, not strongest. Picking the highest-teaches course for a missing
    prerequisite is how you end up recommending an 80-hour specialisation to
    unlock a 10-hour course. Depth-capped, otherwise prerequisite chasing walks
    back to arithmetic.
    """
    if depth > 2:
        return None

    out: list[Recommendation] = []
    covered = have.copy()

    for sid, need in sorted(course.requires.items(), key=lambda kv: -kv[1]):
        i = cat.index.get(sid)
        if i is None or covered[i] >= need - 0.05:
            continue

        candidates = [
            cat.course_by_id[cid]
            for cid in cat.teachers.get(sid, [])
            if cid not in chosen_ids
            and cid not in profile.completed_courses
            # A course the learner rejected cannot come back through the side
            # door as somebody else's prerequisite. If there is no alternative
            # teacher we drop the dependent course instead.
            and cid not in profile.rejected_courses
            and cat.course_by_id[cid].teaches.get(sid, 0.0) * CREDIT >= need - 0.05
        ]
        if not candidates:
            return None

        # hours per unit of the skill we actually need
        candidates.sort(key=lambda c: c.hours / max(c.teaches.get(sid, 0.01), 0.01))
        pick = candidates[0]

        # A prerequisite has prerequisites of its own. Skipping this is how the
        # ML specialisation ended up in a path with no linear algebra in front
        # of it - the violation was invisible because the missing course was
        # absent rather than out of order.
        nested = _prereq_courses(
            pick, cat, covered, chosen_ids | {r.course.id for r in out}, profile, depth + 1
        )
        if nested is None:
            return None
        for n in nested:
            out.append(n)
            covered = np.maximum(covered, cat.teaches_vec(n.course.id) * CREDIT)

        out.append(
            Recommendation(
                course=pick,
                score=0.0,
                components={},
                reasons=[
                    ReasonCode(
                        type="PREREQ_FOR",
                        detail={
                            "course_id": course.id,
                            "course_title": course.title,
                            "skill": sid,
                            "skill_name": cat.name(sid),
                        },
                    )
                ],
                covers=[sid],
            )
        )
        covered = np.maximum(covered, cat.teaches_vec(pick.id) * CREDIT)

    return out


def _estimate_remaining_hours(dropped: list[str], cat: Catalog) -> float:
    """Rough hours to close the skills we could not fit. Cheapest teacher per
    skill, deduplicated - a lower bound, and we say so in the wording."""
    seen: set[str] = set()
    total = 0.0
    for sid in dropped:
        teachers = cat.teachers.get(sid, [])
        if not teachers:
            continue
        pick = min(teachers, key=lambda cid: cat.course_by_id[cid].hours)
        if pick in seen:
            continue
        seen.add(pick)
        total += cat.course_by_id[pick].hours
    return total


# ---------------------------------------------------------------------------
# 2. ordering
# ---------------------------------------------------------------------------

def _order(chosen: list[Recommendation], cat: Catalog, gap: np.ndarray) -> list[Recommendation]:
    by_id = {r.course.id: r for r in chosen}
    g = nx.DiGraph()
    for cid in by_id:
        g.add_node(cid)

    for target_id, rec in by_id.items():
        for sid, need in rec.course.requires.items():
            for source_id, other in by_id.items():
                if source_id == target_id:
                    continue
                if other.course.teaches.get(sid, 0.0) >= need * 0.7:
                    g.add_edge(source_id, target_id, skill=sid)

    if not nx.is_directed_acyclic_graph(g):
        # Real catalogues do contain mutual prerequisites (two courses that each
        # claim to need a bit of the other). Break the weakest edge in each
        # cycle rather than giving up.
        while True:
            try:
                cycle = nx.find_cycle(g, orientation="original")
            except nx.NetworkXNoCycle:
                break
            weakest = min(
                cycle,
                key=lambda e: by_id[e[1]].course.requires.get(g.edges[e[0], e[1]]["skill"], 1.0),
            )
            log.info("breaking prerequisite cycle at %s -> %s", weakest[0], weakest[1])
            g.remove_edge(weakest[0], weakest[1])

    def sort_key(cid: str) -> tuple:
        rec = by_id[cid]
        urgency = float(np.minimum(cat.teaches_vec(cid), gap).sum())
        return (rec.course.level_value, -urgency, rec.course.title)

    order = list(nx.lexicographical_topological_sort(g, key=sort_key))
    return [by_id[cid] for cid in order]


# ---------------------------------------------------------------------------
# 3. shaping
# ---------------------------------------------------------------------------

def _bucket(
    ordered: list[Recommendation],
    cat: Catalog,
    profile: LearnerProfile,
    gap: np.ndarray,
) -> list[Milestone]:
    if not ordered:
        return []

    weekly = max(profile.weekly_hours, 1.0)
    hours_per_milestone = weekly * WEEKS_PER_MILESTONE

    milestones: list[Milestone] = []
    bucket: list[Recommendation] = []
    bucket_hours = 0.0
    start_date = dt.date.today()
    order_counter = 0
    week_cursor = 0

    def flush() -> None:
        nonlocal bucket, bucket_hours, order_counter, week_cursor
        if not bucket:
            return

        idx = len(milestones)
        items: list[PathItem] = []
        skills: list[str] = []

        for rec in bucket:
            order_counter += 1
            unmet = [
                r for r in rec.reasons if r.type in ("NEEDS_FIRST", "PREREQ_FOR")
            ]
            items.append(
                PathItem(
                    kind="course",
                    id=rec.course.id,
                    title=rec.course.title,
                    hours=rec.course.hours,
                    order=order_counter,
                    course=rec.course,
                    skills=list(rec.course.teaches),
                    reasons=rec.reasons,
                    depends_on=[
                        d for r in unmet for d in [r.detail.get("course_id")] if d
                    ],
                    status="available" if idx == 0 else "locked",
                    description=rec.course.description,
                )
            )
            skills.extend(rec.course.teaches)

        top_skills = _dominant_skills(bucket, cat, gap)
        title = _milestone_title(idx, top_skills, cat)
        # Four phases all called "Core machine learning" is useless in a
        # sidebar. Qualify repeats with the phase's own headline skill.
        if any(m.title.split(":")[0] == title for m in milestones):
            qualifier = cat.name(top_skills[0]) if top_skills else f"part {idx + 1}"
            title = f"{title}: {qualifier}"

        order_counter += 1
        items.append(
            PathItem(
                kind="project",
                id=f"proj_m{idx + 1}",
                title=f"Project: {title}",
                hours=round(bucket_hours * 0.25, 1),
                order=order_counter,
                skills=top_skills,
                status="locked",
                description="",  # filled in by explainer.write_project_brief
            )
        )
        order_counter += 1
        items.append(
            PathItem(
                kind="assessment",
                id=f"assess_m{idx + 1}",
                title=f"Checkpoint: {title}",
                hours=1.0,
                order=order_counter,
                skills=top_skills,
                status="locked",
                description="Short check on the skills from this phase. Results feed straight "
                            "back into your profile, so a weak area re-plans the rest of the path.",
            )
        )

        total = sum(i.hours for i in items)
        weeks = max(1, round(total / weekly))
        ms = Milestone(
            index=idx,
            title=title,
            items=items,
            hours=round(total, 1),
            start_week=week_cursor + 1,
            end_week=week_cursor + weeks,
            target_date=start_date + dt.timedelta(weeks=week_cursor + weeks),
            skills_unlocked=sorted(set(skills)),
        )
        milestones.append(ms)
        week_cursor += weeks
        bucket, bucket_hours = [], 0.0

    for rec in ordered:
        bucket.append(rec)
        bucket_hours += rec.course.hours
        if bucket_hours >= hours_per_milestone:
            flush()
    flush()

    return milestones


def _dominant_skills(bucket: list[Recommendation], cat: Catalog, gap: np.ndarray, n: int = 4) -> list[str]:
    agg = np.zeros(cat.n_skills, dtype=np.float32)
    for rec in bucket:
        agg = np.maximum(agg, cat.teaches_vec(rec.course.id))
    weighted = agg * (gap + 0.1)
    return [cat.skills[i].id for i in np.argsort(-weighted)[:n] if agg[i] > 0.1]


def _milestone_title(idx: int, skills: list[str], cat: Catalog) -> str:
    if not skills:
        return f"Phase {idx + 1}"
    domains = {cat.skill_by_id[s].domain for s in skills if s in cat.skill_by_id}
    pretty = {
        "programming": "Programming foundations",
        "mathematics": "Maths for ML",
        "data": "Working with data",
        "machine_learning": "Core machine learning",
        "deep_learning": "Deep learning",
        "nlp": "Language and LLMs",
        "computer_vision": "Computer vision",
        "mlops": "Shipping models",
        "cloud": "Cloud",
        "web": "APIs and interfaces",
        "professional": "Communication and ethics",
    }
    label = next((pretty[d] for d in ("machine_learning", "deep_learning", "nlp", "mlops",
                                      "data", "mathematics", "cloud", "web", "programming",
                                      "computer_vision", "professional") if d in domains), None)
    return label or cat.name(skills[0])
