"""Domain-blind intake tests.

The extractor must retain what the learner actually said; matching those words
to a graph is tested separately once a goal-specific graph exists.
"""

import pytest

from app.engines import intake
from app.schemas import LearnerProfile


def extract(text: str, goal_spec) -> LearnerProfile:
    draft = intake._fallback(text, [])
    profile = intake.merge(LearnerProfile(learner_id="t"), draft)
    return intake.snap(profile, goal_spec)


@pytest.mark.parametrize("text,weeks", [
    ("job ready in five months", 20),
    ("job ready in 5 months", 20),
    ("over six months", 24),
    ("in 10 weeks", 10),
])
def test_written_out_durations(text, weeks):
    assert intake._fallback(f"I want to become a data analyst {text}", []).horizon_weeks == weeks


@pytest.mark.parametrize("text,hours", [
    ("8 hours a week", 8),
    ("about six hours each week", 6),
    ("10 hrs per week", 10),
])
def test_weekly_hours(text, hours):
    assert intake._fallback(f"I want to learn analysis, {text}", []).weekly_hours == hours


def test_goal_is_kept_verbatim_instead_of_resolved_to_a_fixed_role():
    text = "I want to build a Rust game engine"
    assert intake._fallback(text, []).goal_text == text


def test_follow_up_does_not_replace_an_existing_goal():
    draft = intake._fallback("six hours a week", [], existing_goal="Become a UX researcher")
    assert draft.goal_text == "Become a UX researcher"
    assert draft.weekly_hours == 6


def test_aspiration_is_not_read_as_experience(goal_spec):
    profile = extract("I know basic Python and want to become a research data analyst", goal_spec)
    claimed = {c.skill_id for c in profile.known_skills}
    assert "foundation.python" in claimed
    assert "data.analysis" not in claimed


def test_level_words_change_the_self_rating(goal_spec):
    weak = extract("I know basic Python, I want to analyse data", goal_spec)
    strong = extract("I am strong in Python, I want to analyse data", goal_spec)
    rating = lambda p: next(c.self_rating for c in p.known_skills if c.skill_id == "foundation.python")
    assert rating(weak) < rating(strong)


def test_budget_and_format_signals(goal_spec):
    profile = extract(
        "I want to analyse data, no budget for paid courses, I learn by building projects", goal_spec
    )
    assert profile.budget == "free"
    assert "project" in profile.format_prefs


def test_vague_brief_asks_a_follow_up():
    draft = intake._fallback("I want to get better at tech stuff", [])
    assert draft.confidence < 0.7
    assert draft.follow_up_question


def test_open_ended_beginner_brief_is_ready_without_inventing_a_window():
    text = (
        "I want to become an AI engineer in MNCs. I am a complete beginner "
        "and I have no time constraints."
    )
    draft = intake._fallback(text, [])
    profile = intake.merge(LearnerProfile(learner_id="t"), draft)

    assert draft.confidence >= 0.7
    assert draft.follow_up_question is None
    assert profile.time_unconstrained is True
    assert profile.claimed_skills == []


def test_no_experience_follow_up_keeps_goal_and_ignores_assistant_time_words():
    goal = "I want to become an AI engineer in a multinational company"
    draft = intake._fallback(
        "no experience",
        ["user: " + goal, "assistant: How many hours a week can you study?"],
        existing_goal=goal,
    )

    assert draft.goal_text == goal
    assert draft.horizon_weeks is None
    assert draft.known_skills == []


def test_stated_experience_snaps_onto_multi_word_skill_names():
    """"write SQL at work" has to reach the SQL skill.

    Before this, only a phrase that was a substring of a skill name matched, so
    a learner who does the job daily was planned for as a complete beginner and
    the profile card reported "stated skills: 0".
    """
    from app.engines.goal import GoalSpec, SkillTarget
    from app.engines.intake import snap
    from app.schemas import LearnerProfile

    spec = GoalSpec(
        goal_title="Data engineering",
        domain="data",
        skills=[
            SkillTarget(id="py.core", name="Data-Centric Python Programming", domain="prog", level=.7, weight=.9),
            SkillTarget(id="sql.advanced", name="Advanced SQL & Query Optimization", domain="data", level=.8, weight=.95),
            SkillTarget(id="infra.docker", name="Docker & Basic Infrastructure", domain="infra", level=.5, weight=.5),
        ],
    )
    profile = LearnerProfile(learner_id="snap-learner", claimed_skills=["write SQL at work", "basic Python"])

    snap(profile, spec)

    claimed = {claim.skill_id: claim.self_rating for claim in profile.known_skills}
    assert claimed["sql.advanced"] == 4       # "at work" is not beginner
    assert claimed["py.core"] == 2            # "basic" is
    assert "infra.docker" not in claimed      # nothing was claimed about it


def test_filler_words_alone_never_claim_a_skill():
    from app.engines.goal import GoalSpec, SkillTarget
    from app.engines.intake import snap
    from app.schemas import LearnerProfile

    spec = GoalSpec(
        goal_title="Data engineering",
        domain="data",
        skills=[SkillTarget(id="git.core", name="Git & Collaborative Workflows", domain="tools", level=.4, weight=.4)],
    )
    profile = LearnerProfile(learner_id="filler-learner", claimed_skills=["I work at a job", "some experience"])

    snap(profile, spec)

    assert profile.known_skills == []
