"""One request, one snapshot of everything the workspace renders.

The UI used to open a workspace with six calls - generate, profile, three
catalogue reads, dashboard - and every one of them independently asked for the
learner's goal session. On a serverless deployment that is six chances to land
on an instance that has never heard of this learner, and up to six goal graph
derivations for a single click. Both of the failure reports ("no goal to work
from yet", "the pages feel slow") come out of that fan-out.

Assembling the snapshot here means one goal build per interaction and one
instance answering it.
"""

from fastapi import APIRouter

from .. import store
from ..engines import explainer
from ..schemas import LearnerProfile, LearningPath
from . import dashboard as dashboard_api
from .deps import goal_session, learner_path

router = APIRouter(prefix="/api", tags=["workspace"])


def catalog_payload(active) -> dict:
    cat = active.catalog
    return {
        "courses": [c.model_dump() for c in cat.courses],
        "roles": [
            {"id": r.id, "title": r.title, "blurb": r.blurb, "skills": len(r.requirements)}
            for r in cat.roles
        ],
        "skills": [s.model_dump() for s in cat.skills],
    }


def snapshot(
    learner_id: str,
    *,
    active=None,
    profile: LearnerProfile | None = None,
    path: LearningPath | None = None,
    summary: str | None = None,
) -> dict:
    """Everything the workspace needs, built off a single goal session.

    Callers that have already done part of the work (generate has just planned
    a route, feedback has just replanned one) pass it in rather than making the
    snapshot redo it.
    """
    active = active or goal_session(learner_id)
    profile = profile or store.get_profile(learner_id)

    if path is None:
        path = learner_path(learner_id, active, profile)
    if summary is None:
        summary = explainer.summarise_path_template(path, profile) if path else ""

    return {
        "profile": profile,
        "path": path,
        "summary": summary,
        "catalog": catalog_payload(active),
        "dashboard": dashboard_api.build(active, profile, path),
    }


@router.get("/workspace/{learner_id}")
def workspace(learner_id: str):
    return snapshot(learner_id)


@router.get("/catalog")
def catalog(learner_id: str | None = None):
    """The whole goal-specific catalogue in one read.

    Replaces three separate /api/catalog/* calls that each rebuilt the session.
    """
    if not learner_id:
        return {"courses": [], "roles": [], "skills": []}
    return catalog_payload(goal_session(learner_id))
