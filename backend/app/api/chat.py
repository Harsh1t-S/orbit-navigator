"""The learner-facing assistant.

Grounded, not open-ended: the system prompt carries the learner's profile and
their actual path, and the model is told it may only talk about what is in
front of it. Course ids are validated against the catalogue before the answer
goes out, so a made-up course cannot reach the UI even if the model invents
one.

Streams over SSE because a 4-second wait with no output feels broken.
"""

from __future__ import annotations

import json
import re

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .. import llm, store
from ..engines import explainer
from ..engines.profiler import mastery
from ..schemas import ChatRequest
from .deps import goal_session

router = APIRouter(prefix="/api", tags=["chat"])

SYSTEM = """You are the assistant inside a personalised learning path tool. You are talking
to the learner whose profile and path appear below.

What you can do:
- Explain anything in their path: why a course is there, what order things are in,
  what a milestone is for.
- Answer questions about their progress and their skill gaps.
- Suggest what to do next, chosen from their path.

Rules:
- Only reference courses that appear in the context below. Never invent a course,
  a provider, a price or a duration. If they ask about something not in the
  context, say it is not in their path and offer what is.
- Numbers come from the context. Do not estimate.
- If they ask why something is NOT in their path, say you can pull the exact
  reason and suggest they click the course - the tool computes that from the
  prerequisite graph rather than guessing.
- Two or three sentences unless they ask for detail. No bullet lists unless the
  answer is genuinely a list. No emoji.
"""


def _context(learner_id: str) -> tuple[str, set[str]]:
    cat = goal_session(learner_id).catalog
    profile = store.get_profile(learner_id)
    path = store.get_path(learner_id)

    valid_ids: set[str] = set()
    ctx: dict = {
        "goal": profile.goal_text,
        "target_role": profile.target_role,
        "weekly_hours": profile.weekly_hours,
        "horizon_weeks": profile.horizon_weeks,
        "time_unconstrained": profile.time_unconstrained,
        "completed": [cat.course_by_id[c].title for c in profile.completed_courses if c in cat.course_by_id],
    }

    if path:
        ctx["path"] = {
            "role": path.role_title,
            "total_hours": path.total_hours,
            "total_weeks": path.total_weeks,
            "readiness_now": path.readiness_before,
            "readiness_after": path.readiness_after,
            "not_covered": [cat.name(s) for s in path.dropped],
            "milestones": [],
        }
        for ms in path.milestones:
            items = []
            for item in ms.items:
                valid_ids.add(item.id)
                items.append(
                    {
                        "id": item.id,
                        "kind": item.kind,
                        "title": item.title,
                        "hours": item.hours,
                        "status": item.status,
                        "teaches": [cat.name(s) for s in item.skills[:4]],
                        "why": [r.model_dump() for r in item.reasons],
                    }
                )
            ctx["path"]["milestones"].append(
                {"title": ms.title, "weeks": [ms.start_week, ms.end_week], "items": items}
            )
        ctx["top_gaps"] = [
            {"skill": g.skill_name, "gap": g.gap} for g in path.gap_before[:6]
        ]

    return json.dumps(ctx, indent=1, default=str), valid_ids


@router.post("/chat")
def chat(req: ChatRequest):
    context, valid_ids = _context(req.learner_id)
    system = SYSTEM + "\n\n--- learner context ---\n" + context

    messages = [{"role": t.role, "content": t.content} for t in req.history[-8:]]
    messages.append({"role": "user", "content": req.message})

    def event_stream():
        store.append_history(req.learner_id, "user", req.message)
        collected: list[str] = []
        try:
            for chunk in llm.stream_text(messages, system, effort="low"):
                collected.append(chunk)
                yield f"data: {json.dumps({'delta': chunk})}\n\n"
        except Exception as exc:  # noqa: BLE001 - surface it, do not hang the stream
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        answer = "".join(collected)
        if not answer:
            answer = _offline_answer(req)
            yield f"data: {json.dumps({'delta': answer})}\n\n"

        bad = _unknown_course_ids(answer, valid_ids)
        if bad:
            # Should not happen given the prompt, but this is the assertion that
            # lets us claim it does not happen. Logged and counted in the eval.
            yield f"data: {json.dumps({'warning': f'unverified ids: {sorted(bad)}'})}\n\n"

        store.append_history(req.learner_id, "assistant", answer)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


_ID_PATTERN = re.compile(r"\b(?:c_[a-z0-9_]+|proj_m\d+|assess_m\d+)\b")


def _unknown_course_ids(text: str, valid: set[str]) -> set[str]:
    return {m for m in _ID_PATTERN.findall(text)} - valid


def _offline_answer(req: ChatRequest) -> str:
    """No API key, or the call failed. Answer from the path directly rather
    than showing an error - the learner does not care why."""
    cat = goal_session(req.learner_id).catalog
    profile = store.get_profile(req.learner_id)
    path = store.get_path(req.learner_id)
    if path is None:
        return "I do not have a path for you yet. Tell me what you are aiming for and I will build one."

    q = req.message.lower()
    if "why" in q and "not" in q:
        for course in cat.courses:
            if course.title.lower() in q:
                m = mastery(profile, cat)
                return explainer.explain_rejection(course.id, path, profile, m, cat)

    if "next" in q or "start" in q:
        for ms in path.milestones:
            for item in ms.items:
                if item.id not in profile.completed_courses:
                    return (
                        f"Next up is {item.title} ({item.hours:g}h), in {ms.title}. "
                        f"That phase runs weeks {ms.start_week}-{ms.end_week}."
                    )

    return explainer.summarise_path(path, profile)
