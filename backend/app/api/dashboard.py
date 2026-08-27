from collections import defaultdict

import numpy as np
from fastapi import APIRouter

from .. import store
from ..engines import gap as gap_engine
from ..engines.profiler import mastery
from .deps import goal_session, learner_path

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard/{learner_id}")
def dashboard(learner_id: str):
    active = goal_session(learner_id)
    profile = store.get_profile(learner_id)
    return build(active, profile, learner_path(learner_id, active, profile))


def build(active, profile, path):
    """Readiness, radar, progress and next actions for one learner.

    Takes the goal session, profile and path rather than fetching them so the
    workspace snapshot can assemble everything from a single goal build instead
    of paying for one per panel.
    """
    cat = active.catalog
    role = active.role

    target, weight = cat.role_target(role)
    m = mastery(profile, cat)

    done = set(profile.completed_courses) | set(profile.completed_items)

    # --- radar: one axis per domain, averaged over the skills the role cares
    #     about. Per-skill would be 27 axes and unreadable.
    current: dict[str, list[float]] = defaultdict(list)
    wanted: dict[str, list[float]] = defaultdict(list)
    for i, skill in enumerate(cat.skills):
        if weight[i] <= 0:
            continue
        current[skill.domain].append(float(m[i]))
        wanted[skill.domain].append(float(target[i]))

    radar = [
        {
            "domain": d,
            "current": round(float(np.mean(current[d])), 3),
            "target": round(float(np.mean(wanted[d])), 3),
        }
        for d in sorted(current)
    ]

    # --- progress
    #
    # Completed courses get dropped from the regenerated path - you do not
    # re-take something you have finished. If we only counted what is currently
    # scheduled, finishing a course would remove it from both sides of the
    # fraction and the bar would sit at 0% forever. So work already done counts
    # even when it is no longer on the route.
    total_items = done_items = 0
    total_hours = done_hours = 0.0
    next_actions = []
    scheduled: set[str] = set()

    if path:
        for ms in path.milestones:
            for item in ms.items:
                scheduled.add(item.id)
                total_items += 1
                total_hours += item.hours
                if item.id in done or item.status == "done":
                    done_items += 1
                    done_hours += item.hours
                elif len(next_actions) < 3:
                    next_actions.append(
                        {
                            "id": item.id,
                            "kind": item.kind,
                            "title": item.title,
                            "hours": item.hours,
                            "milestone": ms.title,
                            "why": next(
                                (
                                    ", ".join(r.detail.get("names", []))
                                    for r in item.reasons
                                    if r.type == "GAP_COVERAGE"
                                ),
                                "",
                            ),
                        }
                    )

    for cid in done:
        if cid in scheduled or cid not in cat.course_by_id:
            continue
        hours = cat.course_by_id[cid].hours
        total_items += 1
        done_items += 1
        total_hours += hours
        done_hours += hours

    gaps = gap_engine.gap_items(m, target, weight, cat)

    return {
        "role": {"id": role.id, "title": role.title},
        "readiness": gap_engine.readiness(m, target, weight),
        "readiness_projected": path.readiness_after if path else None,
        "radar": radar,
        "top_gaps": [g.model_dump() for g in gaps[:8]],
        "strengths": gap_engine.strengths(m, target, weight, cat),
        "progress": {
            "items_done": done_items,
            "items_total": total_items,
            "hours_done": round(done_hours, 1),
            "hours_total": round(total_hours, 1),
            "percent": round(done_hours / total_hours * 100, 1) if total_hours else 0.0,
        },
        "milestones": [
            {
                "title": ms.title,
                "start_week": ms.start_week,
                "end_week": ms.end_week,
                "target_date": ms.target_date,
            "done": bool(ms.items) and all(i.id in done or i.status == "done" for i in ms.items),
            }
            for ms in (path.milestones if path else [])
        ],
        "next_actions": next_actions,
        "timeline": store.events(profile.learner_id)[-20:],
    }
