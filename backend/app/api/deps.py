"""Dependencies shared by routes that need a learner's dynamic graph."""

from fastapi import HTTPException

from .. import session, store
from ..engines import explainer, intake, planner
from ..schemas import LearnerProfile, LearningPath


def goal_session(learner_id: str) -> session.GoalSession:
    """Get the graph/resources for this learner or return an honest API error.

    A graph cannot be inferred safely without either an available model or a
    matching disk cache.  Returning 503 here is much clearer than a seed-data
    fallback that looks like a personalised recommendation.
    """
    profile = store.get_profile(learner_id)
    try:
        active = session.ensure_for(learner_id, profile.goal_text, profile.budget)
    except session.GoalUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # The graph exists now, so the learner's own words for what they already
    # know can finally be resolved onto it. This belongs here rather than in
    # each endpoint: when only the route endpoint did it, the first route was
    # planned against a profile that had not yet had the learner's stated
    # experience applied, and it showed as "readiness 0%" for someone who had
    # just said they do the job at work.
    claimed_before = len(profile.known_skills)
    intake.snap(profile, active.spec)
    if len(profile.known_skills) != claimed_before or profile.target_role != active.role.id:
        profile.target_role = active.role.id
        store.save_profile(profile)
    return active


def fill_project_briefs(path: LearningPath, cat) -> None:
    for ms in path.milestones:
        for item in ms.items:
            if item.kind == "project" and not item.description:
                item.description = explainer.project_brief_template(ms.title, item.skills, cat)


def apply_completed_status(path: LearningPath, profile: LearnerProfile) -> None:
    """Carry non-course checklist completions through a saved route."""
    completed = set(profile.completed_courses) | set(profile.completed_items)
    for milestone in path.milestones:
        for item in milestone.items:
            if item.id in completed:
                item.status = "done"


def build_path(active: session.GoalSession, profile: LearnerProfile) -> LearningPath:
    """Plan a route and finish it the way every caller needs it."""
    path = planner.build_path(profile, active.catalog, active.retriever)
    fill_project_briefs(path, active.catalog)
    apply_completed_status(path, profile)
    return path


def learner_path(
    learner_id: str, active: session.GoalSession, profile: LearnerProfile
) -> LearningPath | None:
    """The learner's current route, replanned when this process never stored it.

    Same serverless problem as ``store.adopt``: the instance that generated the
    route is not necessarily the instance answering the next request, and a
    dashboard that reports zero progress because of which machine picked up the
    call is a bug, not an honest empty state.

    Replanning is safe because ``planner.build_path`` is a pure function of
    (profile, catalog) and always runs against the *current* profile - a goal
    change clears the stored path, so a rebuild can never resurrect a route
    that belonged to the old goal.
    """
    stored = store.get_path(learner_id)
    if stored is not None:
        return stored
    if not profile.goal_text:
        return None

    path = build_path(active, profile)
    if not path.milestones:
        return None
    store.save_path(path)
    return path
