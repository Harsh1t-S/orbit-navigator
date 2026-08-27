"""The deployed failure: a request answered by a process that never saw the learner.

Vercel spreads one learner's requests over several instances, each with its own
memory and its own /tmp. Before this, the second request of a session could land
on an instance where ``store`` was empty and the session died with "No goal to
work from yet" - reproduced live against the deployment, on the first click of a
mission profile.

Here a cold instance is simulated by clearing the store between requests while
the browser keeps its copy of the profile, exactly as the real client does.
"""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app import session, store
from app.main import app
from app.schemas import LearnerProfile
from tests.conftest import GOAL_TEXT

LEARNER = "cold-instance-learner"


def _header(profile: dict) -> dict[str, str]:
    """Encode a profile the way the browser does."""
    import json

    raw = json.dumps(profile).encode("utf-8")
    return {"X-Orbit-Profile": base64.b64encode(raw).decode("ascii")}


def _go_cold() -> None:
    """Forget everything, as a request landing on another instance would find."""
    for mapping in (store._profiles, store._paths, store._arms, store._history, store._events):
        mapping.clear()


def test_generate_survives_a_cold_instance(install_session):
    with TestClient(app) as client:
        intake = client.post(
            "/api/intake",
            json={
                "learner_id": LEARNER,
                "message": GOAL_TEXT + " I know basic Python, can study 8 hours a week for 20 weeks.",
            },
        )
        profile = intake.json()["profile"]
        install_session(LEARNER, profile["goal_text"], profile["budget"])

        _go_cold()

        generated = client.post(
            f"/api/path/{LEARNER}/generate", json={}, headers=_header(profile)
        )

    assert generated.status_code == 200, generated.json()
    assert generated.json()["path"]["milestones"]


def test_workspace_survives_a_cold_instance(install_session):
    with TestClient(app) as client:
        intake = client.post(
            "/api/intake",
            json={
                "learner_id": LEARNER,
                "message": GOAL_TEXT + " I know basic Python, can study 8 hours a week for 20 weeks.",
            },
        )
        profile = intake.json()["profile"]
        install_session(LEARNER, profile["goal_text"], profile["budget"])
        client.post(f"/api/path/{LEARNER}/generate", json={})

        _go_cold()
        install_session(LEARNER, profile["goal_text"], profile["budget"])

        # No stored path on this instance either: the route is replanned from
        # the adopted profile rather than reported as missing progress.
        snapshot = client.get(f"/api/workspace/{LEARNER}", headers=_header(profile))

    assert snapshot.status_code == 200, snapshot.json()
    body = snapshot.json()
    assert body["path"]["milestones"]
    assert body["catalog"]["courses"] and body["catalog"]["skills"]
    assert body["dashboard"]["progress"]["items_total"] > 0


def test_without_the_header_a_cold_instance_still_fails_honestly(install_session):
    """The header is the fix; nothing else silently invents a goal."""
    with TestClient(app) as client:
        client.post(
            "/api/intake",
            json={"learner_id": LEARNER, "message": GOAL_TEXT + " 8 hours a week for 20 weeks."},
        )
        install_session(LEARNER)
        _go_cold()
        session.drop(LEARNER)
        response = client.post(f"/api/path/{LEARNER}/generate", json={})

    assert response.status_code == 400
    assert "goal" in response.json()["detail"].lower()


def test_adopt_never_moves_state_backwards():
    _go_cold()
    ahead = LearnerProfile(learner_id="rev-learner", goal_text="learn to sail", rev=4)
    store.adopt(ahead)

    stale = LearnerProfile(learner_id="rev-learner", goal_text="learn to ski", rev=2)
    assert store.adopt(stale) is False
    assert store.get_profile("rev-learner").goal_text == "learn to sail"

    newer = LearnerProfile(learner_id="rev-learner", goal_text="learn to ski", rev=5)
    assert store.adopt(newer) is True
    assert store.get_profile("rev-learner").goal_text == "learn to ski"


def test_save_profile_advances_the_revision():
    _go_cold()
    profile = store.get_profile("rev-bump-learner")
    before = profile.rev
    store.save_profile(profile)
    assert store.get_profile("rev-bump-learner").rev == before + 1


def test_a_malformed_header_is_ignored_not_fatal():
    with TestClient(app) as client:
        response = client.get("/api/health", headers={"X-Orbit-Profile": "not-base64-at-all"})
    assert response.status_code == 200
