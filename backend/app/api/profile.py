from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import session, store
from ..engines import diagnostic, intake
from ..engines.profiler import mastery
from ..schemas import LearnerProfile, SkillClaim
from .deps import goal_session

router = APIRouter(prefix="/api", tags=["profile"])

# Diagnostics are regenerated per request but the question text is cached in
# llm.py, so this is only here to keep the correct answers server-side.
_pending: dict[str, list[diagnostic.QuizItem]] = {}


class IntakeRequest(BaseModel):
    learner_id: str
    message: str


class IntakeResponse(BaseModel):
    profile: LearnerProfile
    follow_up_question: str | None = None
    confidence: float
    ready: bool


@router.post("/intake", response_model=IntakeResponse)
def intake_turn(req: IntakeRequest) -> IntakeResponse:
    profile = store.get_profile(req.learner_id)

    history = [f"{h['role']}: {h['content']}" for h in store.history(req.learner_id)]
    previous_goal = profile.goal_text
    previous_budget = profile.budget
    planning_before = _planning_signature(profile)
    draft = intake.extract(req.message, history, existing_goal=previous_goal)
    profile = intake.merge(profile, draft)

    # A graph, its resource ids, and measured answers belong to one goal.  Do
    # not accidentally carry any of them into a different learner intention.
    if profile.goal_text != previous_goal:
        session.drop(req.learner_id)
        store.clear_path(req.learner_id)
        profile.target_role = None
        profile.known_skills.clear()
        profile.completed_courses.clear()
        profile.rejected_courses.clear()
        profile.quiz_results.clear()
    elif _planning_signature(profile) != planning_before:
        # Intake is another way to change budget, time, or stated experience;
        # do not leave an old route looking current after one of those changed.
        store.clear_path(req.learner_id)
        if profile.budget != previous_budget:
            session.drop(req.learner_id)

    store.save_profile(profile)

    store.append_history(req.learner_id, "user", req.message)
    if draft.follow_up_question:
        store.append_history(req.learner_id, "assistant", draft.follow_up_question)

    ready = bool(profile.goal_text) and draft.confidence >= 0.7
    return IntakeResponse(
        profile=profile,
        follow_up_question=draft.follow_up_question,
        confidence=draft.confidence,
        ready=ready,
    )


@router.get("/profile/{learner_id}", response_model=LearnerProfile)
def get_profile(learner_id: str) -> LearnerProfile:
    return store.get_profile(learner_id)


class ProfilePatch(BaseModel):
    goal_text: str | None = None
    target_role: str | None = None
    weekly_hours: float | None = None
    horizon_weeks: int | None = None
    time_unconstrained: bool | None = None
    budget: str | None = None
    format_prefs: list[str] | None = None
    known_skills: list[SkillClaim] | None = None
    completed_courses: list[str] | None = None


def _planning_signature(profile: LearnerProfile) -> tuple:
    """Fields whose changes make an already stored route stale."""
    return (
        profile.weekly_hours,
        profile.horizon_weeks,
        profile.time_unconstrained,
        profile.budget,
        tuple(profile.format_prefs),
        tuple(profile.claimed_skills),
        tuple((claim.skill_id, claim.self_rating) for claim in profile.known_skills),
        tuple(profile.completed_courses),
    )


@router.patch("/profile/{learner_id}", response_model=LearnerProfile)
def patch_profile(learner_id: str, patch: ProfilePatch) -> LearnerProfile:
    """The chat is the headline interface but nobody wants to type
    'actually make it 8 hours a week' - the dashboard edits land here."""
    profile = store.get_profile(learner_id)
    old_goal, old_budget = profile.goal_text, profile.budget
    planning_before = _planning_signature(profile)
    updates = patch.model_dump(exclude_none=True)
    for field, value in updates.items():
        if field == "target_role":
            continue
        setattr(profile, field, value)
    if "horizon_weeks" in updates and "time_unconstrained" not in updates:
        profile.time_unconstrained = False

    # `target_role` is retained as a wire-format compatibility field, but it
    # is derived from the goal graph and must never select a hidden static role.
    goal_changed = profile.goal_text != old_goal
    if goal_changed:
        profile.target_role = None
        profile.known_skills.clear()
        profile.completed_courses.clear()
        profile.rejected_courses.clear()
        profile.quiz_results.clear()

    if goal_changed or profile.budget != old_budget:
        session.drop(learner_id)

    if goal_changed or _planning_signature(profile) != planning_before:
        store.clear_path(learner_id)

    store.save_profile(profile)
    return profile


@router.get("/profile/{learner_id}/diagnostic")
def get_diagnostic(learner_id: str):
    active = goal_session(learner_id)   # also snaps stated skills onto the graph
    cat = active.catalog
    profile = store.get_profile(learner_id)
    role = active.role

    target, weight = cat.role_target(role)
    items = diagnostic.build(profile, target, weight, cat)
    _pending[learner_id] = items

    return {
        "role": role.title,
        "questions": [
            {
                "skill_id": i.skill_id,
                "skill_name": i.skill_name,
                "question": i.question,
                "options": i.options,
            }
            for i in items
        ],
    }


class DiagnosticAnswers(BaseModel):
    answers: dict[str, int]   # skill_id -> chosen option index


@router.post("/profile/{learner_id}/diagnostic")
def submit_diagnostic(learner_id: str, payload: DiagnosticAnswers):
    items = _pending.get(learner_id)
    if not items:
        raise HTTPException(400, "No diagnostic in progress - request one first.")

    cat = goal_session(learner_id).catalog
    profile = store.get_profile(learner_id)
    before = mastery(profile, cat)

    # grade() already returns the observed mastery per skill, and the profiler
    # blends quiz_results against the priors. Nudging the same answers a second
    # time here was double-counting them.
    scored = diagnostic.grade(items, payload.answers)
    profile.quiz_results.update(scored)
    store.save_profile(profile)

    after = mastery(profile, cat)
    moved = [
        {
            "skill_id": s.id,
            "skill_name": s.name,
            "before": round(float(before[i]), 2),
            "after": round(float(after[i]), 2),
        }
        for i, s in enumerate(cat.skills)
        if abs(float(after[i] - before[i])) > 0.05
    ]
    store.log_event(learner_id, "diagnostic", {"skills_moved": len(moved)})

    return {"scored": scored, "changed": moved, "profile": profile}
