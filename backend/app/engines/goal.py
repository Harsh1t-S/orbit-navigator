"""Derives a skill graph for whatever goal the learner actually typed.

This replaces the hand-authored roles.json + taxonomy.json. Those capped the
whole system at six roles and sixty-four skills: ask for "UX researcher" or
"Rust game development" and the old build silently fell back to ML Engineer,
which is not a personalised recommender, it is a lookup table.

Here the model does the one thing a fixed list cannot - decompose an arbitrary
goal into the skills it actually needs, how much of each, how much each one
matters, and which ones have to come before which. Everything downstream
(gap analysis, set cover, topological sort, reason codes) is unchanged and
still fully deterministic. The AI builds the graph; the algorithms guarantee
the route through it.

Derived graphs are cached to disk by normalised goal text, so the second person
to ask for "become a data analyst" pays nothing and the demo is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from .. import llm
from ..config import CACHE_DIR

log = logging.getLogger(__name__)

GOAL_CACHE = CACHE_DIR / "goals"
GOAL_CACHE.mkdir(parents=True, exist_ok=True)
MAX_SKILLS = 12


class SkillTarget(BaseModel):
    id: str                       # slug, stable within this goal
    name: str
    domain: str
    level: float = Field(ge=0.0, le=1.0)   # mastery a competent practitioner has
    weight: float = Field(ge=0.0, le=1.0)  # how much this goal actually depends on it
    why: str = ""                          # one line, surfaced in the UI
    requires: list[str] = Field(default_factory=list)  # ids of prerequisite skills


class GoalSpec(BaseModel):
    goal_title: str
    domain: str
    summary: str = ""
    skills: list[SkillTarget] = Field(default_factory=list)
    # things the model judged out of scope; shown so the learner can correct us
    excluded: list[str] = Field(default_factory=list)


SYSTEM = """You decompose a learning goal into the skill graph behind it.

You will be given whatever the learner typed. It can be any domain at all - a job
title, a project they want to build, a technology, a vague ambition. Do not force
it into a familiar shape and do not substitute a neighbouring goal you know more
about. Decompose the goal in front of you.

Return the skills someone genuinely needs to reach that goal, and for each:

  id       lowercase dotted slug, stable and readable, e.g. "rust.ownership",
           "research.interviewing", "audio.mixing". Group with a leading domain
           segment so related skills sort together.
  name     how a practitioner would say it
  domain   the coarse grouping the leading segment refers to
  level    0-1, the mastery a competent practitioner in this goal actually has.
           Not everything is 0.9. Supporting skills sit around 0.4-0.6.
  weight   0-1, how much the goal genuinely depends on it. This is the most
           important number you produce: it decides what gets learned first when
           there is not enough time for everything. Reserve values above 0.9 for
           skills without which the goal is simply not achievable.
  why      one short clause explaining why this goal needs it. Concrete.
  requires ids of skills from this same list that must come first. Only real
           dependencies - "you cannot do X before Y" - not a suggested order.
           Most skills need none. Never create a cycle.

Rules:
- 8 to 12 skills. Keep related micro-skills together so the first live route
  stays useful without turning discovery into a slow syllabus crawl.
- Include the unglamorous prerequisites people skip. If the goal needs maths,
  version control, or statistics, say so.
- Include non-technical skills when the goal genuinely needs them.
- If the goal is too vague to decompose, still return your best reading and put
  what you had to assume into `excluded`.
- `excluded` is also for adjacent things you deliberately left out, so the
  learner can push back on your scoping.
"""


# A goal title arrives in the learner's own phrasing - "become a machine
# learning engineer", "I want to learn to restore watches". That is the right
# thing to keep in the graph, but dropped into a sentence it produces "of what
# a become a machine learning engineer role expects". Strip the intent verb for
# display only; the derivation itself never sees this.
_TITLE_PREFIX = re.compile(
    r"^(?:i\s+want\s+to\s+|i\s+would\s+like\s+to\s+|i'd\s+like\s+to\s+|"
    r"learn(?:ing)?\s+(?:how\s+to\s+|to\s+)?|become\s+(?:an?\s+)?|be\s+(?:an?\s+)?|"
    r"get\s+into\s+|move\s+into\s+|transition\s+(?:in)?to\s+|switch\s+to\s+|"
    r"how\s+to\s+)+",
    re.IGNORECASE,
)


def display_title(title: str) -> str:
    """The goal as a capability, for anywhere it is shown to the learner."""
    collapsed = " ".join(title.split()).strip(" .")
    cleaned = _TITLE_PREFIX.sub("", collapsed).strip(" .")
    if not cleaned:
        cleaned = collapsed
    return cleaned[0].upper() + cleaned[1:] if cleaned else "Your goal"


def _key(goal_text: str) -> str:
    normalised = re.sub(r"\s+", " ", goal_text.strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:20]


def _cache_path(goal_text: str) -> Path:
    return GOAL_CACHE / f"{_key(goal_text)}.json"


def cached(goal_text: str) -> GoalSpec | None:
    path = _cache_path(goal_text)
    if path.exists():
        try:
            return GoalSpec(**json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # a corrupt cache entry is not worth crashing on
            log.warning("dropping unreadable goal cache %s (%s)", path.name, exc)
            path.unlink(missing_ok=True)
    return None


def derive(goal_text: str, *, refresh: bool = False) -> GoalSpec | None:
    """Goal text in, skill graph out. None when there is no model available and
    nothing cached - callers surface that rather than inventing a graph."""
    if not refresh:
        hit = cached(goal_text)
        if hit is not None:
            return hit

    spec = llm.parse(
        GoalSpec,
        f"Learning goal, exactly as the learner wrote it:\n\n{goal_text}",
        SYSTEM,
        effort="low",
        max_tokens=1800,
    )
    if spec is None:
        return None

    spec = _sanitise(spec)
    _cache_path(goal_text).write_text(spec.model_dump_json(indent=1), encoding="utf-8")
    return spec


def _sanitise(spec: GoalSpec) -> GoalSpec:
    """Make the graph safe for the planner regardless of what came back.

    The planner assumes a DAG over known ids. A dangling `requires` or a cycle
    would surface as a confusing route rather than an error, so both are fixed
    here, at the boundary, where it is cheap.
    """
    if len(spec.skills) > MAX_SKILLS:
        log.info("trimming derived skill graph from %d to %d skills", len(spec.skills), MAX_SKILLS)
        spec.skills = spec.skills[:MAX_SKILLS]

    known = {s.id for s in spec.skills}

    for skill in spec.skills:
        skill.requires = [r for r in skill.requires if r in known and r != skill.id]
        skill.level = min(1.0, max(0.0, skill.level))
        skill.weight = min(1.0, max(0.0, skill.weight))

    _break_cycles(spec)
    return spec


def _break_cycles(spec: GoalSpec) -> None:
    by_id = {s.id: s for s in spec.skills}
    colour: dict[str, int] = {}   # 0 = visiting, 1 = done

    def visit(sid: str) -> None:
        colour[sid] = 0
        for req in list(by_id[sid].requires):
            if colour.get(req) == 0:
                # back edge: the model claimed a mutual dependency
                log.info("breaking prerequisite cycle %s -> %s", req, sid)
                by_id[sid].requires.remove(req)
            elif req not in colour:
                visit(req)
        colour[sid] = 1

    for skill in spec.skills:
        if skill.id not in colour:
            visit(skill.id)
