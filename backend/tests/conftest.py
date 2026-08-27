"""Deterministic, discovered-resource-shaped fixtures.

These are test doubles for the graph and resources returned by the model/web
search boundary.  They are deliberately kept out of the application data
directory: production never falls back to them.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("LPR_OFFLINE", "true")
os.environ.setdefault("LPR_USE_EMBEDDINGS", "false")

from app.catalog import Catalog, build
from app.engines.goal import GoalSpec, SkillTarget
from app.engines.retrieval import Retriever
from app.schemas import Course
from app.session import GoalSession


GOAL_TEXT = "Become a research data analyst who can turn messy data into clear decisions."


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Keep API tests from changing the application's persisted demo state."""
    from app import store

    monkeypatch.setattr(store, "_FILE", tmp_path / "store.json")
    for mapping in (store._profiles, store._paths, store._arms, store._history, store._events):
        mapping.clear()
    yield
    for mapping in (store._profiles, store._paths, store._arms, store._history, store._events):
        mapping.clear()


def _course(
    course_id: str,
    title: str,
    hours: float,
    teaches: dict[str, float],
    *,
    level: str = "beginner",
    requires: dict[str, float] | None = None,
    hours_stated: bool = True,
) -> Course:
    return Course(
        id=course_id,
        title=title,
        provider="Example Learning",
        url=f"https://example.test/{course_id}",
        description=f"A resource covering {', '.join(teaches)}.",
        level=level,
        hours=hours,
        hours_stated=hours_stated,
        cost="free",
        format="interactive",
        teaches=teaches,
        requires=requires or {},
    )


@pytest.fixture
def goal_spec() -> GoalSpec:
    return GoalSpec(
        goal_title="Research Data Analyst",
        domain="data",
        summary="Turn messy data into defensible decisions.",
        skills=[
            SkillTarget(id="foundation.python", name="Python fundamentals", domain="foundations", level=.7, weight=.9),
            SkillTarget(id="foundation.git", name="Version control", domain="foundations", level=.4, weight=.35),
            SkillTarget(id="data.cleaning", name="Data cleaning", domain="data", level=.75, weight=.85, requires=["foundation.python"]),
            SkillTarget(id="data.analysis", name="Exploratory analysis", domain="data", level=.75, weight=.95, requires=["data.cleaning"]),
            SkillTarget(id="communication.reporting", name="Decision reporting", domain="communication", level=.6, weight=.65, requires=["data.analysis"]),
        ],
    )


@pytest.fixture
def resources() -> list[Course]:
    return [
        _course("python_intro", "Python foundations", 8, {"foundation.python": .95}),
        _course("python_alt", "Python practice", 10, {"foundation.python": .9}),
        _course("git_intro", "Version control basics", 4, {"foundation.git": .8}),
        _course("clean_data", "Cleaning real data", 8, {"data.cleaning": .95}),
        _course("clean_data_alt", "Data quality drills", 9, {"data.cleaning": .9}),
        _course("analyse_data", "Exploratory data analysis", 10, {"data.analysis": .95}, level="intermediate"),
        _course("report_findings", "Decision-ready reports", 6, {"communication.reporting": .85}, level="intermediate", hours_stated=False),
    ]


@pytest.fixture
def catalog(goal_spec: GoalSpec, resources: list[Course]) -> Catalog:
    return build(goal_spec, resources)


@pytest.fixture
def retriever(catalog: Catalog) -> Retriever:
    return Retriever(catalog)


@pytest.fixture
def install_session(goal_spec: GoalSpec, resources: list[Course], monkeypatch):
    """Install a dynamic session without calling a model or the network."""
    from app import session

    session._sessions.clear()

    def install(learner_id: str, goal_text: str = GOAL_TEXT, budget: str = "freemium") -> GoalSession:
        cat = build(goal_spec, resources)
        active = GoalSession(
            spec=goal_spec,
            catalog=cat,
            retriever=Retriever(cat),
            goal_key=session._goal_key(goal_text),
            budget=budget,
        )
        session._sessions[learner_id] = active
        return active

    yield install
    session._sessions.clear()
