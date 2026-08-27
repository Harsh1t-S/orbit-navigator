"""HTTP flow over an in-memory dynamic goal session."""

from fastapi.testclient import TestClient

from app import session
from app.main import app
from tests.conftest import GOAL_TEXT


LEARNER = "dynamic-api-learner"


def test_health_describes_per_goal_catalogue():
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["catalog_mode"] == "per_goal"
    assert body["courses"] == 0


def test_no_key_or_cache_is_an_honest_error():
    learner = "no-key-learner"
    session.drop(learner)
    with TestClient(app) as client:
        client.post("/api/intake", json={"learner_id": learner, "message": "I want to become a UX researcher"})
        response = client.post(f"/api/path/{learner}/generate", json={})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_full_dynamic_flow(install_session):
    with TestClient(app) as client:
        intake = client.post(
            "/api/intake",
            json={
                "learner_id": LEARNER,
                "message": GOAL_TEXT + " I know basic Python, can study 8 hours a week for 20 weeks.",
            },
        )
        assert intake.status_code == 200
        profile = intake.json()["profile"]
        install_session(LEARNER, profile["goal_text"], profile["budget"])

        generated = client.post(f"/api/path/{LEARNER}/generate", json={})
        assert generated.status_code == 200
        path = generated.json()["path"]
        assert path["role_title"] == "Research Data Analyst"
        assert path["milestones"]
        assert path["readiness_after"] > path["readiness_before"]

        # The session is now also visible through learner-scoped catalogue APIs.
        courses = client.get("/api/catalog/courses", params={"learner_id": LEARNER, "limit": 20}).json()
        skills = client.get("/api/catalog/skills", params={"learner_id": LEARNER}).json()
        assert courses and skills
        assert client.get("/api/catalog/courses").json() == []

        course_id = next(i["id"] for m in path["milestones"] for i in m["items"] if i["kind"] == "course")
        explain = client.get(f"/api/path/{LEARNER}/explain/{course_id}")
        assert explain.status_code == 200
        assert explain.json()["reason_codes"]

        scheduled = {i["id"] for m in path["milestones"] for i in m["items"]}
        alternate = next(course["id"] for course in courses if course["id"] not in scheduled)
        assert client.get(f"/api/path/{LEARNER}/why-not/{alternate}").status_code == 200

        dashboard_before = client.get(f"/api/dashboard/{LEARNER}").json()
        # Projects and checkpoints are first-class route checklist entries,
        # rather than fake catalogue courses. Their checked state survives
        # without re-planning away the item the learner just completed.
        checkpoint_id = next(
            i["id"]
            for milestone in path["milestones"]
            for i in milestone["items"]
            if i["kind"] == "assessment"
        )
        checked = client.post(
            f"/api/path/{LEARNER}/feedback",
            json={"learner_id": LEARNER, "item_id": checkpoint_id, "signal": "completed"},
        )
        assert checked.status_code == 200
        assert any(
            i["id"] == checkpoint_id and i["status"] == "done"
            for milestone in checked.json()["path"]["milestones"]
            for i in milestone["items"]
        )

        feedback = client.post(
            f"/api/path/{LEARNER}/feedback",
            json={"learner_id": LEARNER, "item_id": course_id, "signal": "completed"},
        )
        assert feedback.status_code == 200
        dashboard_after = client.get(f"/api/dashboard/{LEARNER}").json()
        assert dashboard_after["readiness"] >= dashboard_before["readiness"]

        recommendations = client.get(f"/api/recommendations/{LEARNER}", params={"limit": 3}).json()
        assert recommendations
        assert all(row["reason_codes"] for row in recommendations)

        with client.stream("POST", "/api/chat", json={"learner_id": LEARNER, "message": "What should I do next?"}) as stream:
            body = "".join(stream.iter_text())
        assert stream.status_code == 200
        assert "[DONE]" in body


def test_goal_change_invalidates_old_session_and_path(install_session):
    learner = "goal-change-learner"
    with TestClient(app) as client:
        intake = client.post("/api/intake", json={"learner_id": learner, "message": GOAL_TEXT})
        install_session(learner, intake.json()["profile"]["goal_text"])
        assert client.post(f"/api/path/{learner}/generate", json={}).status_code == 200

        patched = client.patch(f"/api/profile/{learner}", json={"goal_text": "Learn sound design"})
        assert patched.status_code == 200
        assert session.get(learner) is None
        assert client.get(f"/api/path/{learner}").status_code == 404
