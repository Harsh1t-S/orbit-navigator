"""Process-local state with a JSON file behind it.

Not a database. It is enough for a prototype and a demo, it survives a restart,
and swapping it for Postgres is a matter of replacing four functions. See
README, Known gaps.
"""

from __future__ import annotations

import json
import threading

from .config import CACHE_DIR
from .schemas import LearnerProfile, LearningPath

_FILE = CACHE_DIR / "store.json"
_lock = threading.Lock()

_profiles: dict[str, LearnerProfile] = {}
_paths: dict[str, LearningPath] = {}
_arms: dict[str, dict[str, list[float]]] = {}
_history: dict[str, list[dict]] = {}
_events: dict[str, list[dict]] = {}


def _load() -> None:
    if not _FILE.exists():
        return
    try:
        raw = json.loads(_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return  # corrupt cache is not worth crashing over, start clean
    _profiles.update({k: LearnerProfile(**v) for k, v in raw.get("profiles", {}).items()})
    _paths.update({k: LearningPath(**v) for k, v in raw.get("paths", {}).items()})
    _arms.update(raw.get("arms", {}))
    _history.update(raw.get("history", {}))
    _events.update(raw.get("events", {}))


def _flush() -> None:
    payload = {
        "profiles": {k: v.model_dump(mode="json") for k, v in _profiles.items()},
        "paths": {k: v.model_dump(mode="json") for k, v in _paths.items()},
        "arms": _arms,
        "history": _history,
        "events": _events,
    }
    _FILE.write_text(json.dumps(payload), encoding="utf-8")


def get_profile(learner_id: str) -> LearnerProfile:
    with _lock:
        if learner_id not in _profiles:
            _profiles[learner_id] = LearnerProfile(learner_id=learner_id)
            _flush()
        return _profiles[learner_id]


def save_profile(profile: LearnerProfile) -> None:
    with _lock:
        profile.rev += 1
        _profiles[profile.learner_id] = profile
        _flush()


def adopt(profile: LearnerProfile) -> bool:
    """Take a profile handed back by the browser when this process is behind it.

    A serverless deployment spreads one learner's requests over several
    instances, and each of them has its own memory and its own /tmp. A profile
    written by the instance that handled intake simply does not exist on the
    instance that handles the next call, which ends the session with "No goal
    to work from yet" seconds after the learner set a goal.

    The browser holds the authoritative copy - this app has always said session
    state lives in the browser - so it sends it back on every request and an
    instance that is behind adopts it. ``rev`` is what stops that from moving
    state backwards: a client older than what this process already holds is
    ignored.
    """
    with _lock:
        known = _profiles.get(profile.learner_id)
        if known is not None and known.rev >= profile.rev:
            return False
        _profiles[profile.learner_id] = profile
        _flush()
        return True


def get_path(learner_id: str) -> LearningPath | None:
    return _paths.get(learner_id)


def save_path(path: LearningPath) -> None:
    with _lock:
        _paths[path.learner_id] = path
        _flush()


def clear_path(learner_id: str) -> None:
    """Remove a route that belongs to an old goal or a changed constraint set."""
    with _lock:
        if learner_id in _paths:
            del _paths[learner_id]
            _flush()


def arms(learner_id: str) -> dict[str, list[float]]:
    return _arms.setdefault(learner_id, {})


def history(learner_id: str) -> list[dict]:
    return _history.setdefault(learner_id, [])


def append_history(learner_id: str, role: str, content: str) -> None:
    with _lock:
        _history.setdefault(learner_id, []).append({"role": role, "content": content})
        _flush()


def log_event(learner_id: str, kind: str, detail: dict) -> None:
    """Progress timeline. The dashboard reads this to draw readiness over
    time - without it every chart is a single point."""
    with _lock:
        _events.setdefault(learner_id, []).append({"kind": kind, **detail})
        _flush()


def events(learner_id: str) -> list[dict]:
    return _events.get(learner_id, [])


_load()
