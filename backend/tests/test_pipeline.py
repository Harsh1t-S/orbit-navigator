"""Invariants for a graph assembled from a derived goal and discovered rows."""

import numpy as np
import pytest

from app.catalog import build
from app.engines import gap as gap_engine
from app.engines import planner
from app.engines.goal import GoalSpec, SkillTarget
from app.engines.profiler import mastery
from app.engines.retrieval import Retriever
from app.schemas import Course, LearnerProfile, SkillClaim


def make_profile(catalog, **changes) -> LearnerProfile:
    base = dict(
        learner_id="t",
        goal_text="Become a research data analyst who can turn messy data into clear decisions.",
        target_role=catalog.roles[0].id,
        weekly_hours=8,
        horizon_weeks=20,
        known_skills=[SkillClaim(skill_id="foundation.python", self_rating=4)],
    )
    base.update(changes)
    return LearnerProfile(**base)


def test_catalog_is_assembled_from_one_goal_graph(catalog, goal_spec):
    assert catalog.roles[0].title == goal_spec.goal_title
    assert set(catalog.index) == {s.id for s in goal_spec.skills}
    assert not set(catalog.role_by_id) - {"research_data_analyst"}


def test_courses_inherit_skill_graph_prerequisites(catalog):
    analysis = catalog.course_by_id["analyse_data"]
    cleaning = catalog.course_by_id["clean_data"]
    assert analysis.requires["data.cleaning"] > 0
    assert cleaning.requires["foundation.python"] > 0


def test_broad_resource_keeps_a_startable_primary_skill_path():
    """A side topic must not make an otherwise useful starter resource wait
    on the advanced skill it eventually discusses."""
    spec = GoalSpec(
        goal_title="Data engineering",
        domain="data",
        skills=[
            SkillTarget(id="foundation.python", name="Python", domain="foundation", level=.75, weight=.8),
            SkillTarget(id="data.sql", name="Advanced SQL", domain="data", level=.85, weight=.95,
                        requires=["foundation.python"]),
            SkillTarget(id="data.etl", name="ETL pipelines", domain="data", level=.75, weight=.95,
                        requires=["data.sql"]),
        ],
    )

    def resource(course_id, *, teaches, hours, requires=None):
        return Course(
            id=course_id,
            title=course_id.replace("_", " ").title(),
            provider="Example Learning",
            url=f"https://example.test/{course_id}",
            description="A structured learning resource.",
            level="beginner",
            hours=hours,
            hours_stated=True,
            cost="free",
            format="text",
            teaches=teaches,
            requires=requires or {},
        )

    # This starter guide mainly teaches Python. Its short ETL section should
    # not force SQL before its first lesson, otherwise both resources lock each
    # other out and the planner returns an empty route.
    starter = resource(
        "python_with_etl",
        teaches={"foundation.python": .6, "data.etl": .5},
        hours=5,
        requires={"data.sql": .59},
    )
    roadmap = resource(
        "sql_roadmap",
        teaches={"data.sql": .8, "data.etl": .7},
        hours=80,
    )
    dynamic_catalog = build(spec, [starter, roadmap])

    assert dynamic_catalog.course_by_id["python_with_etl"].requires == {}
    # The graph projects 0.75 * 0.7 = .52, but the only resource that teaches
    # Python tops out at .6 * COMPLETION_CREDIT = .51. Requirements are capped
    # at what the catalogue can actually deliver, so this stays satisfiable.
    assert dynamic_catalog.course_by_id["sql_roadmap"].requires == {"foundation.python": .51}

    profile = LearnerProfile(
        learner_id="t",
        goal_text="Become a data engineer",
        target_role="data_engineering",
        weekly_hours=6,
        horizon_weeks=26,
    )
    path = planner.build_path(profile, dynamic_catalog, Retriever(dynamic_catalog))
    course_ids = [item.id for milestone in path.milestones for item in milestone.items if item.kind == "course"]

    assert course_ids == ["python_with_etl", "sql_roadmap"]


def test_exact_fit_starter_is_not_lost_to_the_course_reserve():
    spec = GoalSpec(
        goal_title="Data quality specialist",
        domain="data",
        skills=[
            SkillTarget(id="data.quality", name="Data quality", domain="data", level=.9, weight=1.0),
        ],
    )
    starter = Course(
        id="long_starter",
        title="Long starter course",
        provider="Example Learning",
        url="https://example.test/long-starter",
        description="A complete, time-boxed starter course.",
        level="beginner",
        hours=124,
        hours_stated=True,
        cost="free",
        format="text",
        teaches={"data.quality": .95},
        requires={},
    )
    dynamic_catalog = build(spec, [starter])
    profile = LearnerProfile(
        learner_id="t",
        goal_text="Become a data quality specialist",
        target_role="data_quality_specialist",
        weekly_hours=6,
        horizon_weeks=26,
    )

    path = planner.build_path(profile, dynamic_catalog, Retriever(dynamic_catalog))
    course_ids = [item.id for milestone in path.milestones for item in milestone.items if item.kind == "course"]

    assert course_ids == ["long_starter"]
    assert path.coverage > 0
    assert path.total_hours <= 6 * 26


def test_open_ended_profile_does_not_inherit_the_default_12_week_ceiling():
    spec = GoalSpec(
        goal_title="AI engineering foundations",
        domain="ai",
        skills=[SkillTarget(id="foundation.python", name="Python", domain="foundation", level=.8, weight=1.0)],
    )
    starter = Course(
        id="python_starter",
        title="Python starter",
        provider="Example Learning",
        url="https://example.test/python-starter",
        description="A complete beginner foundation.",
        level="beginner",
        hours=60,
        cost="free",
        format="interactive",
        teaches={"foundation.python": .95},
    )
    dynamic_catalog = build(spec, [starter])
    retriever = Retriever(dynamic_catalog)

    finite = planner.build_path(
        LearnerProfile(learner_id="finite", goal_text=spec.goal_title), dynamic_catalog, retriever
    )
    flexible = planner.build_path(
        LearnerProfile(learner_id="flex", goal_text=spec.goal_title, time_unconstrained=True),
        dynamic_catalog,
        retriever,
    )

    assert finite.milestones == []
    assert any(item.id == "python_starter" for phase in flexible.milestones for item in phase.items)


def test_every_required_skill_is_teachable(catalog):
    missing = [sid for sid in catalog.roles[0].requirements if not catalog.teachers.get(sid)]
    assert not missing


def test_no_prerequisite_violations(catalog, retriever):
    profile = make_profile(catalog, known_skills=[])
    path = planner.build_path(profile, catalog, retriever)
    have = mastery(profile, catalog)
    for milestone in path.milestones:
        for item in sorted(milestone.items, key=lambda i: i.order):
            if item.kind != "course":
                continue
            for sid, need in catalog.course_by_id[item.id].requires.items():
                assert have[catalog.index[sid]] >= need - .05, item.title
            have = np.maximum(have, catalog.teaches_vec(item.id) * .85)


@pytest.mark.parametrize("weekly,weeks", [(4, 12), (8, 20), (16, 26)])
def test_path_respects_time_budget(catalog, retriever, weekly, weeks):
    profile = make_profile(catalog, weekly_hours=weekly, horizon_weeks=weeks)
    path = planner.build_path(profile, catalog, retriever)
    assert path.total_hours <= weekly * weeks * 1.05


def test_more_time_never_reduces_projected_readiness(catalog, retriever):
    small = planner.build_path(make_profile(catalog, weekly_hours=4, horizon_weeks=12), catalog, retriever)
    large = planner.build_path(make_profile(catalog, weekly_hours=12, horizon_weeks=24), catalog, retriever)
    assert large.readiness_after >= small.readiness_after


def test_completed_and_rejected_resources_do_not_return(catalog, retriever):
    profile = make_profile(catalog, completed_courses=["python_intro"])
    first_path = planner.build_path(profile, catalog, retriever)
    scheduled = {i.id for m in first_path.milestones for i in m.items}
    assert "python_intro" not in scheduled

    first_course = next(i.id for m in first_path.milestones for i in m.items if i.kind == "course")
    profile.rejected_courses.append(first_course)
    replanned = planner.build_path(profile, catalog, retriever)
    assert first_course not in {i.id for m in replanned.milestones for i in m.items}


def test_diagnostic_moves_mastery(catalog):
    profile = make_profile(catalog)
    before = mastery(profile, catalog)[catalog.index["foundation.python"]]
    profile.quiz_results["foundation.python"] = 0.0
    after = mastery(profile, catalog)[catalog.index["foundation.python"]]
    assert after < before


def test_path_items_have_projects_checkpoints_and_reason_codes(catalog, retriever):
    profile = make_profile(catalog)
    path = planner.build_path(profile, catalog, retriever)
    for milestone in path.milestones:
        kinds = [item.kind for item in milestone.items]
        assert kinds[-2:] == ["project", "assessment"]

    role = catalog.roles[0]
    target, weight = catalog.role_target(role)
    m = mastery(profile, catalog)
    gap = gap_engine.gap_vector(m, target, weight)
    for rec in retriever.rank(profile, m, gap, profile.goal_text, limit=5):
        assert rec.reasons
        assert set(rec.components) >= {"gap_coverage", "level_fit", "quality"}


def test_a_prerequisite_nobody_teaches_does_not_lock_the_route_out():
    """The failure mode that produced a three-hour route for a ten-skill goal.

    Live search found nothing that teaches Python properly, so every resource
    that depended on it was permanently unschedulable and the planner returned
    one starter course with eight skills dropped. An uncoverable skill is a
    coverage gap - the gap report says so - not a lock on the rest of the route.
    """
    spec = GoalSpec(
        goal_title="Machine learning engineer",
        domain="ml",
        skills=[
            SkillTarget(id="python.core", name="Python", domain="prog", level=.8, weight=.9),
            SkillTarget(id="ml.fundamentals", name="ML fundamentals", domain="ml", level=.8, weight=.95,
                        requires=["python.core"]),
            SkillTarget(id="ops.deployment", name="Deployment", domain="ops", level=.7, weight=.8,
                        requires=["ml.fundamentals"]),
        ],
    )

    def resource(course_id, *, teaches, hours, requires=None, level="intermediate"):
        return Course(
            id=course_id, title=course_id.replace("_", " ").title(), provider="Example Learning",
            url=f"https://example.test/{course_id}", description="A structured learning resource.",
            level=level, hours=hours, hours_stated=True, cost="free", format="text",
            teaches=teaches, requires=requires or {},
        )

    # Nothing here teaches python.core at all.
    resources = [
        resource("ml_crash_course", teaches={"ml.fundamentals": .9}, hours=15, level="beginner"),
        resource("deploy_course", teaches={"ops.deployment": .8}, hours=12),
    ]
    cat = build(spec, resources)

    for course in cat.courses:
        assert "python.core" not in course.requires, (
            f"{course.id} still waits on a skill no resource teaches"
        )

    profile = make_profile(cat, known_skills=[])
    path = planner.build_path(profile, cat, Retriever(cat))
    scheduled = {item.id for ms in path.milestones for item in ms.items}
    assert {"ml_crash_course", "deploy_course"} <= scheduled
    # The uncovered foundation is still reported honestly.
    assert "python.core" in path.dropped


def test_a_fully_gated_resource_set_still_has_somewhere_to_start():
    """Live search returned nine resources for an LLM-retrieval goal and every
    one of them was gated behind another, so the route came back empty and the
    error blamed the learner's time window.

    The skill graph is acyclic; the projection of it onto resources is not, so
    the catalogue has to guarantee an entry point itself.
    """
    spec = GoalSpec(
        goal_title="Build LLM applications with retrieval",
        domain="ai",
        skills=[
            SkillTarget(id="python.fundamentals", name="Python", domain="prog", level=.7, weight=.85),
            SkillTarget(id="nlp.text", name="Text processing", domain="nlp", level=.65, weight=.8,
                        requires=["python.fundamentals"]),
            SkillTarget(id="embeddings.vector", name="Embeddings", domain="ai", level=.75, weight=.9,
                        requires=["nlp.text"]),
            SkillTarget(id="retrieval.hybrid", name="Hybrid search", domain="ai", level=.75, weight=.9,
                        requires=["embeddings.vector"]),
        ],
    )

    def resource(course_id, *, teaches, hours):
        return Course(
            id=course_id, title=course_id.replace("_", " ").title(), provider="Example Learning",
            url=f"https://example.test/{course_id}", description="A structured learning resource.",
            level="intermediate", hours=hours, hours_stated=True, cost="free", format="text",
            teaches=teaches, requires={},
        )

    # Every resource teaches something downstream, so every one of them inherits
    # a gate - including the only teacher of the foundation.
    resources = [
        resource("embeddings_intro", teaches={"nlp.text": .8, "embeddings.vector": .7}, hours=2),
        resource("vector_db", teaches={"embeddings.vector": .8}, hours=3),
        resource("hybrid_search", teaches={"retrieval.hybrid": .7, "python.fundamentals": .7}, hours=3),
    ]
    cat = build(spec, resources)

    assert any(not course.requires for course in cat.courses), (
        "nothing in the catalogue can be taken first"
    )

    profile = make_profile(cat, known_skills=[], weekly_hours=10, horizon_weeks=12)
    path = planner.build_path(profile, cat, Retriever(cat))
    assert path.milestones, "a fully gated catalogue produced an empty route"
