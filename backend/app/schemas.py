"""Wire format for the whole app. The frontend only ever sees these shapes."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

Level = Literal["beginner", "intermediate", "advanced"]
Cost = Literal["free", "freemium", "paid"]
Fmt = Literal["video", "text", "interactive", "project"]

# Courses carry a label, the maths wants a number.
LEVEL_VALUE: dict[str, float] = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.85}

# Finishing a course is not mastering everything in it. The profiler, the
# planner and the catalogue all have to agree on this number or a resource can
# teach a prerequisite by one measure and fail to satisfy it by another.
COMPLETION_CREDIT = 0.85


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------

class Skill(BaseModel):
    id: str
    name: str
    domain: str
    parent: Optional[str] = None


class Course(BaseModel):
    id: str
    title: str
    provider: str
    url: str
    description: str
    level: Level
    hours: float
    # False when the source stated no duration and the figure is a model
    # estimate. The UI labels those - an estimate shown as a fact is the same
    # failure as an invented rating.
    hours_stated: bool = True
    cost: Cost = "freemium"
    format: Fmt = "video"
    teaches: dict[str, float] = Field(default_factory=dict)
    requires: dict[str, float] = Field(default_factory=dict)

    @property
    def level_value(self) -> float:
        return LEVEL_VALUE[self.level]

    def search_text(self) -> str:
        return f"{self.title} {self.provider} {self.description} {' '.join(self.teaches)}"


class Role(BaseModel):
    id: str
    title: str
    blurb: str = ""
    aliases: list[str] = Field(default_factory=list)
    requirements: dict[str, dict[str, float]] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# learner
# --------------------------------------------------------------------------

class SkillClaim(BaseModel):
    skill_id: str
    # 1-5 the way a human would rate themselves; profiler converts to 0-1 and
    # deliberately shrinks it, self-assessment is not evidence.
    self_rating: int = Field(ge=1, le=5)


class LearnerProfile(BaseModel):
    learner_id: str
    # Bumped by the store on every save. The browser echoes back the value it
    # last received, which is how a server process that has never seen this
    # learner tells "the client is ahead of me" (adopt it) apart from "the
    # client is stale" (ignore it). See store.adopt.
    rev: int = 0
    goal_text: str = ""
    target_role: Optional[str] = None
    # The learner's own words are retained until a goal-specific graph exists.
    # `known_skills` is the resolved, machine-readable form used by the
    # planner; `claimed_skills` is deliberately not discarded when a graph is
    # rebuilt for the same goal.
    claimed_skills: list[str] = Field(default_factory=list)
    known_skills: list[SkillClaim] = Field(default_factory=list)
    completed_courses: list[str] = Field(default_factory=list)
    # Courses influence mastery and leave the regenerated course plan. Project
    # and checkpoint ids stay here so the route checklist can preserve their
    # completed state without pretending they are catalogue courses.
    completed_items: list[str] = Field(default_factory=list)
    # `weekly_hours` remains the pace used to lay out milestones.  An explicit
    # open-ended goal must not silently inherit the legacy 12-week selection
    # ceiling, though, so keep that fact separately from the numeric defaults.
    weekly_hours: float = 6.0
    horizon_weeks: int = 12
    time_unconstrained: bool = False
    format_prefs: list[Fmt] = Field(default_factory=list)
    budget: Cost = "freemium"
    motivation: Optional[str] = None
    # filled by the diagnostic, skill_id -> observed correctness
    quiz_results: dict[str, float] = Field(default_factory=dict)
    # things the learner has explicitly rejected; retrieval hard-filters these
    rejected_courses: list[str] = Field(default_factory=list)
    # learned online from feedback, see engines/adapter.py
    weight_overrides: dict[str, float] = Field(default_factory=dict)


class ProfileDraft(BaseModel):
    """What the intake agent extracts from free text. Everything optional -
    the agent fills what it can and asks about the rest."""
    goal_text: Optional[str] = None
    target_role: Optional[str] = None
    known_skills: list[str] = Field(default_factory=list)
    completed_courses: list[str] = Field(default_factory=list)
    weekly_hours: Optional[float] = None
    horizon_weeks: Optional[int] = None
    time_unconstrained: Optional[bool] = None
    format_prefs: list[str] = Field(default_factory=list)
    budget: Optional[str] = None
    motivation: Optional[str] = None
    confidence: float = 0.0
    follow_up_question: Optional[str] = None


# --------------------------------------------------------------------------
# gap + recommendations
# --------------------------------------------------------------------------

class GapItem(BaseModel):
    skill_id: str
    skill_name: str
    domain: str
    current: float
    target: float
    weight: float
    gap: float  # weighted, this is what the planner optimises against


class ReasonCode(BaseModel):
    """Emitted by the scorer while it scores. The LLM turns these into prose
    but never invents them - that is the whole point of the two-tier setup."""
    type: str
    detail: dict = Field(default_factory=dict)
    contribution: float = 0.0


class Recommendation(BaseModel):
    course: Course
    score: float
    components: dict[str, float]
    reasons: list[ReasonCode]
    covers: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# path
# --------------------------------------------------------------------------

class PathItem(BaseModel):
    kind: Literal["course", "project", "assessment"]
    id: str
    title: str
    hours: float
    order: int
    course: Optional[Course] = None
    skills: list[str] = Field(default_factory=list)
    reasons: list[ReasonCode] = Field(default_factory=list)
    prerequisite_of: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["locked", "available", "in_progress", "done"] = "locked"
    description: str = ""


class Milestone(BaseModel):
    index: int
    title: str
    summary: str = ""
    items: list[PathItem]
    hours: float
    start_week: int
    end_week: int
    target_date: Optional[date] = None
    skills_unlocked: list[str] = Field(default_factory=list)


class LearningPath(BaseModel):
    learner_id: str
    target_role: str
    role_title: str
    generated_at: str
    milestones: list[Milestone]
    total_hours: float
    total_weeks: int
    readiness_before: float
    readiness_after: float
    gap_before: list[GapItem]
    coverage: float          # fraction of weighted gap this path closes
    dropped: list[str] = Field(default_factory=list)   # gaps we could not cover
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# feedback / progress
# --------------------------------------------------------------------------

class Feedback(BaseModel):
    learner_id: str
    item_id: str
    signal: Literal["too_easy", "too_hard", "not_interested", "loved_it", "completed", "skipped"]
    note: Optional[str] = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    learner_id: str
    message: str
    history: list[ChatTurn] = Field(default_factory=list)
