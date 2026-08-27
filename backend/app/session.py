"""Per-learner goal graph, resources and retriever.

There is no global catalogue any more, so there is no global anything. Each
learner has a goal, and each goal gets its own derived skill graph, its own
discovered resources and its own retrieval index over them.

Building one is the expensive step in the whole system - a model call to derive
the graph, then several searched calls to find resources. Both layers cache to
disk underneath, so the first learner to ask for "become a data analyst" pays
for it and everyone after that does not.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from time import perf_counter

from . import llm
from .catalog import Catalog, build
from .engines import discovery, goal
from .engines.goal import GoalSpec
from .engines.retrieval import Retriever
from .schemas import Course, Role

log = logging.getLogger(__name__)

_sessions: dict[str, "GoalSession"] = {}
_lock = threading.Lock()
# Catalog endpoints are fetched in parallel by the browser.  A separate build
# lock keeps a server restart from turning those three reads into three costly
# graph/resource discovery calls for the same learner.
_build_lock = threading.Lock()


@dataclass
class GoalSession:
    spec: GoalSpec
    catalog: Catalog
    retriever: Retriever
    goal_key: str
    budget: str

    @property
    def role(self) -> Role:
        return self.catalog.roles[0]

    @property
    def resources(self) -> list[Course]:
        return self.catalog.courses


class GoalUnavailable(RuntimeError):
    """Raised when no graph can be produced - no model, nothing cached.

    Deliberately loud. The alternative is falling back to a hardcoded
    catalogue, which is the thing this design exists to avoid: a route built
    from data nobody can trace is worse than an honest failure.
    """


def get(learner_id: str) -> GoalSession | None:
    return _sessions.get(learner_id)


def _goal_key(goal_text: str) -> str:
    """A process-local identity, not the disk-cache key.

    The disk cache intentionally hides its hashing details in goal.py.  The
    session only needs to know whether the profile has changed since it was
    built, so a normalised string is clearer and avoids reaching into a private
    helper.
    """
    return " ".join(goal_text.split()).casefold()


def build_for(
    learner_id: str,
    goal_text: str,
    budget: str = "freemium",
    *,
    refresh: bool = False,
) -> GoalSession:
    """Derive the graph for this goal and discover resources for it."""
    if not goal_text.strip():
        raise GoalUnavailable("No goal to work from yet.")

    started = perf_counter()
    log.info("goal build started")
    spec = goal.derive(goal_text, refresh=refresh)
    if spec is None:
        raise GoalUnavailable(
            "Could not derive a skill graph for that goal. This build needs an "
            "GEMINI_API_KEY to decompose a goal it has not seen before - see "
            "backend/.env.example."
        )

    log.info("goal graph ready in %.1fs with %d skills", perf_counter() - started, len(spec.skills))
    resources = discovery.find(spec, spec.skills, budget=budget, refresh=refresh)
    if not resources:
        if not discovery.available():
            raise GoalUnavailable(
                "Could not search for live resources. An uncached goal needs a "
                "TAVILY_API_KEY - see backend/.env.example."
            )
        if not llm.available():
            raise GoalUnavailable(
                "Could not select resources from the live search results. An uncached goal "
                "needs a GEMINI_API_KEY - see backend/.env.example."
            )
        raise GoalUnavailable(
            f"Derived the skill graph for {spec.goal_title!r} but found no "
            "resources for it. Nothing is invented to fill the gap."
        )

    log.info("resource discovery ready in %.1fs with %d resources", perf_counter() - started, len(resources))
    catalog = build(spec, resources)
    session = GoalSession(
        spec=spec,
        catalog=catalog,
        retriever=Retriever(catalog),
        goal_key=_goal_key(goal_text),
        budget=budget,
    )

    with _lock:
        _sessions[learner_id] = session
    log.info("goal build completed in %.1fs", perf_counter() - started)
    return session


def ensure_for(learner_id: str, goal_text: str, budget: str = "freemium") -> GoalSession:
    """Return the current session, rebuilding only when its input changed.

    Every request after the first one comes through here.  That gives a server
    restart the same behaviour as a first request (disk caches are reused), but
    avoids rebuilding an expensive graph/retrieval index for every dashboard
    refresh.
    """
    current = get(learner_id)
    if (
        current is not None
        and current.goal_key == _goal_key(goal_text)
        and current.budget == budget
    ):
        return current

    with _build_lock:
        # Re-check after waiting: another concurrent endpoint may have built
        # the exact same session while this request was queued.
        current = get(learner_id)
        if (
            current is not None
            and current.goal_key == _goal_key(goal_text)
            and current.budget == budget
        ):
            return current
        return build_for(learner_id, goal_text, budget)


def require(learner_id: str) -> GoalSession:
    session = get(learner_id)
    if session is None:
        raise GoalUnavailable("No goal has been set for this learner yet.")
    return session


def drop(learner_id: str) -> None:
    with _lock:
        _sessions.pop(learner_id, None)


def extend(learner_id: str, extra_skills: list[str], budget: str = "freemium") -> GoalSession:
    """Search for more resources covering skills the current set does not.

    Called when the planner reports uncovered gaps: rather than shrugging, we
    go back out and look specifically for what is missing.
    """
    session = require(learner_id)
    wanted = [s for s in session.spec.skills if s.id in set(extra_skills)]
    if not wanted:
        return session

    found = discovery.find(session.spec, wanted, budget=budget)
    fresh = [c for c in found if c.id not in session.catalog.course_by_id]
    if not fresh:
        return session

    log.info("extend: %d additional resources for %d uncovered skills", len(fresh), len(wanted))
    catalog = build(session.spec, session.resources + fresh)
    session = GoalSession(
        spec=session.spec,
        catalog=catalog,
        retriever=Retriever(catalog),
        goal_key=session.goal_key,
        budget=budget,
    )
    with _lock:
        _sessions[learner_id] = session
    return session
