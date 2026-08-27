import copy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import store
from ..engines import adapter, explainer, planner
from ..engines.profiler import mastery
from ..schemas import Feedback, LearningPath
from .deps import apply_completed_status, build_path, goal_session, learner_path
from .workspace import snapshot

router = APIRouter(prefix="/api", tags=["path"])


class GenerateRequest(BaseModel):
    role_id: str | None = None


@router.post("/path/{learner_id}/generate")
def generate(learner_id: str, req: GenerateRequest):
    """Plan the route and return the whole workspace with it.

    The response carries profile, catalogue and dashboard alongside the route
    so the browser opens a workspace on one round trip instead of six.
    """
    profile = store.get_profile(learner_id)
    if not profile.goal_text:
        raise HTTPException(400, "Tell me the goal first - POST /api/intake.")

    active = goal_session(learner_id)

    # There is exactly one derived role in a goal session.  `role_id` remains
    # accepted for older clients but cannot redirect a learner to a fixed list.
    path = build_path(active, profile)

    # A 200 response with an empty path sends the client into a workspace with
    # nothing to work on. This can still happen when every real resource is
    # longer than the learner's available window, so make the constraint
    # explicit instead of presenting an unfinished-looking blank route.
    if not path.milestones and any(item.gap > 0.05 for item in path.gap_before):
        # Say which of the two things actually happened. This used to blame the
        # learner's time window whenever they had one, including for a resource
        # set that was fully gated - nine short resources, none of them longer
        # than three hours, and the advice was "add weekly hours".
        window = (
            None if profile.time_unconstrained
            else max(profile.weekly_hours, 1.0) * max(profile.horizon_weeks, 1)
        )
        shortest = min((c.hours for c in active.catalog.courses), default=0.0)
        if window is not None and shortest > window:
            detail = (
                f"The shortest resource I found for this goal is {shortest:g} hours, and you "
                f"have {window:g}. Add weekly hours or extend the window and I will re-plan."
            )
        else:
            detail = (
                "I found the goal and its resources, but could not assemble a route from "
                "them. Build it again and I will search for a wider set."
            )
        raise HTTPException(422, detail)

    store.save_path(path)
    store.log_event(
        learner_id,
        "path_generated",
        {"readiness": path.readiness_before, "coverage": path.coverage},
    )
    return snapshot(
        learner_id,
        active=active,
        profile=profile,
        path=path,
        summary=explainer.summarise_path_template(path, profile),
    )


@router.get("/path/{learner_id}", response_model=LearningPath)
def get_path(learner_id: str) -> LearningPath:
    path = store.get_path(learner_id)
    if path is None:
        raise HTTPException(404, "No path generated yet.")
    return path


@router.get("/path/{learner_id}/explain/{course_id}")
def explain(learner_id: str, course_id: str):
    active = goal_session(learner_id)
    cat = active.catalog
    profile = store.get_profile(learner_id)
    path = learner_path(learner_id, active, profile)
    if path is None:
        raise HTTPException(404, "No path generated yet.")

    item = next(
        (i for ms in path.milestones for i in ms.items if i.id == course_id), None
    )
    if item is None or item.course is None:
        raise HTTPException(404, "That item is not in the path.")

    from ..schemas import Recommendation

    covers = next(
        (r.detail.get("skills", []) for r in item.reasons if r.type == "GAP_COVERAGE"), []
    )
    rec = Recommendation(
        course=item.course, score=0.0, components={}, reasons=item.reasons, covers=covers
    )
    return {
        "course_id": course_id,
        "explanation": explainer.explain_recommendation(rec, profile, cat),
        "reason_codes": [r.model_dump() for r in item.reasons],
    }


@router.get("/path/{learner_id}/why-not/{course_id}")
def why_not(learner_id: str, course_id: str):
    """The counterfactual. Answers "why isn't X on my path" from the
    prerequisite graph and the time budget, not from vibes."""
    active = goal_session(learner_id)
    cat = active.catalog
    profile = store.get_profile(learner_id)
    path = learner_path(learner_id, active, profile)
    if path is None:
        raise HTTPException(404, "No path generated yet.")

    m = mastery(profile, cat)
    return {
        "course_id": course_id,
        "explanation": explainer.explain_rejection(course_id, path, profile, m, cat),
    }


@router.post("/path/{learner_id}/feedback")
def feedback(learner_id: str, fb: Feedback):
    active = goal_session(learner_id)
    cat = active.catalog
    profile = store.get_profile(learner_id)
    arms = store.arms(learner_id)

    message = adapter.apply(profile, fb, cat, arms)
    store.save_profile(profile)
    store.log_event(learner_id, "feedback", {"signal": fb.signal, "item": fb.item_id})

    # A course completion changes mastery and must re-plan the remaining
    # course sequence. Projects/checkpoints are checklist evidence only, so
    # preserve the current route and its stable item ids when they are ticked.
    if fb.signal == "completed" and fb.item_id not in cat.course_by_id:
        path = store.get_path(learner_id) or build_path(active, profile)
    else:
        path = build_path(active, profile)

    apply_completed_status(path, profile)
    store.save_path(path)

    return {
        "message": message,
        "preferred_formats": adapter.preferred_formats(arms),
        **snapshot(learner_id, active=active, profile=profile, path=path),
    }


class SimulateRequest(BaseModel):
    weekly_hours: float | None = None
    horizon_weeks: int | None = None
    budget: str | None = None
    time_unconstrained: bool | None = None


@router.post("/path/{learner_id}/simulate", response_model=LearningPath)
def simulate(learner_id: str, req: SimulateRequest) -> LearningPath:
    """What-if. Drives the "drag your weekly hours and watch the timeline
    move" control - runs on a copy so nothing is persisted."""
    active = goal_session(learner_id)
    cat = active.catalog
    profile = copy.deepcopy(store.get_profile(learner_id))
    updates = req.model_dump(exclude_none=True)
    for field, value in updates.items():
        setattr(profile, field, value)
    if "horizon_weeks" in updates and "time_unconstrained" not in updates:
        profile.time_unconstrained = False
    return planner.build_path(profile, cat, active.retriever)


@router.get("/recommendations/{learner_id}")
def recommendations(learner_id: str, q: str = "", limit: int = 10):
    """Standalone recommendations, outside the path. Used by the "explore"
    panel and by the chat when someone asks for an alternative."""
    active = goal_session(learner_id)
    cat = active.catalog
    profile = store.get_profile(learner_id)
    role = active.role

    from ..engines import gap as gap_engine

    target, weight = cat.role_target(role)
    m = mastery(profile, cat)
    gap = gap_engine.gap_vector(m, target, weight)

    recs = active.retriever.rank(profile, m, gap, q or profile.goal_text, limit=limit)
    return [
        {
            "course": r.course.model_dump(),
            "score": r.score,
            "components": r.components,
            "covers": [cat.name(s) for s in r.covers],
            "reason_codes": [rc.model_dump() for rc in r.reasons],
        }
        for r in recs
    ]
