"""The skill/resource graph the engines run on.

Nothing in here is loaded from a committed catalogue any more. A `Catalog` is
built per goal, at request time, from:

  * a GoalSpec  - the skill decomposition the model derived for this learner's
                  actual goal (engines/goal.py)
  * resources   - what web search found for the gaps in it (engines/discovery.py)

The class interface is unchanged, which is the point: gap analysis, retrieval,
the set-cover planner, the topological sort and the reason codes all take a
`Catalog` and have no idea whether it came from a file or from the model. The
graph became dynamic; the algorithms over it did not have to change.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .engines import goal
from .engines.goal import GoalSpec
from .schemas import COMPLETION_CREDIT, Course, Role, Skill

log = logging.getLogger(__name__)


class Catalog:
    def __init__(self, skills: list[Skill], courses: list[Course], roles: list[Role]):
        self.skills = skills
        self.courses = courses
        self.roles = roles

        self.skill_by_id = {s.id: s for s in skills}
        self.index = {s.id: i for i, s in enumerate(skills)}
        self.course_by_id = {c.id: c for c in courses}
        self.role_by_id = {r.id: r for r in roles}

        self.n_skills = len(skills)

        # course id -> teaches/requires as dense vectors, built once
        self._teaches = {c.id: self.vector(c.teaches) for c in courses}
        self._requires = {c.id: self.vector(c.requires) for c in courses}

        # reverse index: skill -> courses that teach it, best teacher first
        self.teachers: dict[str, list[str]] = {}
        for c in courses:
            for sid in c.teaches:
                self.teachers.setdefault(sid, []).append(c.id)
        for sid, ids in self.teachers.items():
            ids.sort(key=lambda cid: self.course_by_id[cid].teaches.get(sid, 0), reverse=True)

    # -- vector helpers ----------------------------------------------------

    def vector(self, mapping: dict[str, float]) -> np.ndarray:
        v = np.zeros(self.n_skills, dtype=np.float32)
        for sid, val in mapping.items():
            i = self.index.get(sid)
            if i is None:
                continue  # skill not in this goal's graph; ignore quietly
            v[i] = val
        return v

    def to_mapping(self, vec: np.ndarray, threshold: float = 1e-6) -> dict[str, float]:
        return {
            self.skills[i].id: float(vec[i])
            for i in range(self.n_skills)
            if vec[i] > threshold
        }

    def teaches_vec(self, course_id: str) -> np.ndarray:
        return self._teaches[course_id]

    def requires_vec(self, course_id: str) -> np.ndarray:
        return self._requires[course_id]

    def name(self, skill_id: str) -> str:
        s = self.skill_by_id.get(skill_id)
        return s.name if s else skill_id

    # -- goal lookup -------------------------------------------------------

    def resolve_role(self, text: str | None) -> Role | None:
        """A dynamic catalogue holds exactly one goal, so this is trivial - it
        exists because the engines still ask. Kept rather than removed so the
        planner does not need a special case."""
        return self.roles[0] if self.roles else None

    def role_target(self, role: Role) -> tuple[np.ndarray, np.ndarray]:
        """Returns (target level vector, importance weight vector)."""
        target = np.zeros(self.n_skills, dtype=np.float32)
        weight = np.zeros(self.n_skills, dtype=np.float32)
        for sid, req in role.requirements.items():
            i = self.index.get(sid)
            if i is None:
                continue
            target[i] = req.get("level", 0.0)
            weight[i] = req.get("weight", 1.0)
        return target, weight


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def goal_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:48] or "goal"


def _cap_requirements_to_what_is_teachable(courses: list[Course]) -> list[Course]:
    """Never demand a prerequisite level nothing in this catalogue can reach.

    The requirement above is projected from the skill graph, which describes
    the goal, not the resources that happened to be found for it. When search
    returns nothing that teaches a foundation properly - a Python foundation,
    say - every resource depending on it becomes permanently unschedulable and
    the route collapses: a ten-skill ML goal came out as a single three-hour
    course with eight skills dropped.

    That is a coverage gap, and the gap report is where it belongs. Turning it
    into an unsatisfiable lock says nothing true and costs the learner the rest
    of the route, so each requirement is capped at what the strongest available
    teacher can actually deliver.
    """
    achievable: dict[str, float] = {}
    for course in courses:
        for sid, strength in course.teaches.items():
            achievable[sid] = max(achievable.get(sid, 0.0), strength * COMPLETION_CREDIT)

    capped: list[Course] = []
    for course in courses:
        requires = {
            sid: round(min(need, achievable.get(sid, 0.0)), 2)
            for sid, need in course.requires.items()
        }
        requires = {sid: need for sid, need in requires.items() if need > 0.05}
        capped.append(
            course if requires == course.requires else course.model_copy(update={"requires": requires})
        )
    return capped


def _skill_depth(prereq_of: dict[str, list[str]]) -> dict[str, int]:
    """How far each skill sits from a foundation in the goal's own graph."""
    depth: dict[str, int] = {}

    def resolve(sid: str, seen: frozenset[str] = frozenset()) -> int:
        if sid in depth:
            return depth[sid]
        if sid in seen:      # goal.py breaks cycles, but never trust that here
            return 0
        parents = prereq_of.get(sid, [])
        value = 1 + max((resolve(p, seen | {sid}) for p in parents), default=-1)
        depth[sid] = value
        return value

    for sid in prereq_of:
        resolve(sid)
    return depth


def _ensure_a_startable_resource(
    courses: list[Course], prereq_of: dict[str, list[str]]
) -> list[Course]:
    """Guarantee an entry point, or none of the route is reachable.

    Requirements are projected onto resources from the skill graph. That graph
    is acyclic; the projection onto resources is not. A resource can be the
    only teacher of a foundation while inheriting a gate from the topic it
    teaches most strongly, and then every resource waits on another one. Live
    search hit exactly that for "build LLM applications with retrieval": nine
    resources, all of them gated, an empty route, and an error blaming the
    learner's time window for a problem that had nothing to do with it.

    The skill graph decides who goes first - the resource whose strongest skill
    sits closest to a foundation starts the route, and its projected gate is
    dropped. Everything else still waits its turn.
    """
    if not courses or any(not course.requires for course in courses):
        return courses

    depth = _skill_depth(prereq_of)

    def entry_rank(course: Course) -> tuple[int, int, float, str]:
        strongest = max(course.teaches.values())
        primary = [sid for sid, level in course.teaches.items() if level >= strongest * 0.85]
        return (
            min((depth.get(sid, 0) for sid in primary), default=0),
            len(course.requires),
            course.hours,
            course.id,          # deterministic tie-break
        )

    entry = min(courses, key=entry_rank)
    log.info(
        "catalog: no startable resource, opening the route with %r", entry.title
    )
    return [
        course.model_copy(update={"requires": {}}) if course.id == entry.id else course
        for course in courses
    ]


def build(spec: GoalSpec, resources: list[Course]) -> Catalog:
    """Assemble a Catalog for one goal.

    The one piece of real work here is translating the skill-level prerequisite
    graph onto the resources. The model tells us "you need linear algebra before
    backpropagation" as a relation between *skills*; the planner orders
    *resources*.

    A broad resource may touch both a foundation and an advanced topic. Making
    the whole resource inherit prerequisites from every secondary topic is
    misleading: a beginner Python guide that briefly discusses ETL would appear
    to require advanced SQL before its first lesson. Worse, several broad
    resources can become mutually unstartable. We therefore carry prerequisites
    only from a resource's primary teaching promise (the skills it teaches most
    strongly). The skill graph remains the source of truth and the planner still
    orders genuinely dependent resources, while a learner always has a viable
    starting point when one exists.
    """
    skills = [
        Skill(id=s.id, name=s.name, domain=s.domain)
        for s in spec.skills
    ]
    known = {s.id for s in spec.skills}
    prereq_of = {s.id: [r for r in s.requires if r in known] for s in spec.skills}
    level_of = {s.id: s.level for s in spec.skills}

    enriched: list[Course] = []
    for course in resources:
        teaches = {sid: level for sid, level in course.teaches.items() if sid in known}
        if not teaches:
            continue

        # A discovery result's course-level prerequisites are an LLM reading of
        # a broad source page. Keep explicit requirements, except when one only
        # belongs to a side topic and would turn the whole resource into an
        # impossible prerequisite bundle. The canonical graph still provides
        # the requirements for the course's main teaching promise.
        strongest = max(teaches.values())
        primary = {
            sid for sid, level in teaches.items()
            if level >= strongest * 0.85
        }
        primary_requires = {
            prereq
            for sid in primary
            for prereq in prereq_of.get(sid, [])
            if prereq not in teaches
        }
        secondary_requires = {
            prereq
            for sid in teaches
            if sid not in primary
            for prereq in prereq_of.get(sid, [])
            if prereq not in teaches
        }
        requires = {
            sid: level
            for sid, level in course.requires.items()
            if sid in known
            and sid not in teaches
            and (sid in primary_requires or sid not in secondary_requires)
        }
        for prereq in primary_requires:
            # Need enough of the prerequisite to follow along, not mastery.
            needed = round(min(0.7, level_of.get(prereq, 0.5) * 0.7), 2)
            requires[prereq] = max(requires.get(prereq, 0.0), needed)

        enriched.append(course.model_copy(update={"teaches": teaches, "requires": requires}))

    enriched = _cap_requirements_to_what_is_teachable(enriched)
    enriched = _ensure_a_startable_resource(enriched, prereq_of)

    role = Role(
        id=goal_slug(spec.goal_title),
        title=goal.display_title(spec.goal_title),
        blurb=spec.summary,
        requirements={s.id: {"level": s.level, "weight": s.weight} for s in spec.skills},
    )

    log.info(
        "catalog: goal=%r skills=%d resources=%d (from %d discovered)",
        spec.goal_title, len(skills), len(enriched), len(resources),
    )
    return Catalog(skills=skills, courses=enriched, roles=[role])
