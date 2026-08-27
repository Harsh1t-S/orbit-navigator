"""The per-learner graph/session boundary without a live model call."""

import pytest

from app import session


def test_ensure_for_builds_once_and_reuses_matching_goal(goal_spec, resources, monkeypatch):
    calls = {"goal": 0, "resources": 0}

    def derive(goal_text, *, refresh=False):
        calls["goal"] += 1
        return goal_spec

    def find(spec, skills, budget, *, refresh=False):
        calls["resources"] += 1
        return resources

    monkeypatch.setattr(session.goal, "derive", derive)
    monkeypatch.setattr(session.discovery, "find", find)

    first = session.ensure_for("learner", "Learn research data analysis", "free")
    second = session.ensure_for("learner", "  learn  research data analysis  ", "free")

    assert first is second
    assert first.catalog.roles[0].title == goal_spec.goal_title
    assert calls == {"goal": 1, "resources": 1}


def test_session_refuses_to_make_up_missing_goal_or_resources(goal_spec, monkeypatch):
    monkeypatch.setattr(session.goal, "derive", lambda *args, **kwargs: None)
    with pytest.raises(session.GoalUnavailable, match="GEMINI_API_KEY"):
        session.build_for("missing-graph", "Learn sound design")

    monkeypatch.setattr(session.goal, "derive", lambda *args, **kwargs: goal_spec)
    monkeypatch.setattr(session.discovery, "find", lambda *args, **kwargs: [])
    monkeypatch.setattr(session.discovery, "available", lambda: True)
    monkeypatch.setattr(session.llm, "available", lambda: True)
    with pytest.raises(session.GoalUnavailable, match="found no resources"):
        session.build_for("missing-resources", "Learn sound design")


def test_session_names_the_missing_live_search_key(goal_spec, monkeypatch):
    monkeypatch.setattr(session.goal, "derive", lambda *args, **kwargs: goal_spec)
    monkeypatch.setattr(session.discovery, "find", lambda *args, **kwargs: [])
    monkeypatch.setattr(session.discovery, "available", lambda: False)
    with pytest.raises(session.GoalUnavailable, match="TAVILY_API_KEY"):
        session.build_for("missing-search", "Learn sound design")
