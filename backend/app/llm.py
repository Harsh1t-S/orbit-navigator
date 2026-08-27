"""Small Gemini adapter used by the graph, intake, and explanation layers.

The rest of the application talks in Pydantic models or plain text. Keeping
the provider-specific SDK calls here makes those deterministic layers agnostic
to which model is in use. A request is routed through a small, configured
model pool rather than making one Gemini model a single point of failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from time import monotonic
from typing import Callable, Iterator, Type, TypeVar

from pydantic import BaseModel

from .config import CACHE_DIR, settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

_client = None
_client_failed = False

# Caches are deliberately shared by all usable models in a lane: a valid,
# schema-checked answer costs nothing when the next request lands on another
# fallback model. Changing either pool invalidates the key safely.
CACHE_VERSION = "gemini-fallback-v2"
_model_cooldowns: dict[str, float] = {}
_disabled_models: set[str] = set()


def _get_client():
    """Create the Gemini client lazily, without ever logging credentials."""
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    if not settings.gemini_api_key:
        log.warning("No Gemini credentials found - running in offline mode")
        _client_failed = True
        return None
    try:
        from google import genai

        _client = genai.Client(api_key=settings.gemini_api_key)
    except Exception as exc:  # missing package or invalid local SDK setup
        # Do not interpolate the exception: some SDKs include request metadata
        # and this boundary must never risk writing credentials to a log.
        log.warning("Gemini client unavailable (%s) - running in offline mode", type(exc).__name__)
        _client_failed = True
    return _client


def _config(
    *,
    model: str,
    system: str,
    max_output_tokens: int,
    schema: dict | None = None,
    thinking_level: str | None = None,
):
    """Build an SDK configuration only after the optional dependency exists."""
    from google.genai import types

    kwargs: dict = {
        "system_instruction": system,
        "max_output_tokens": max_output_tokens,
        # We never provide callable tools. Explicitly disabling the SDK's
        # automatic function-calling wrapper avoids an irrelevant warning on
        # every direct generate_content request.
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if schema is not None:
        kwargs.update(
            response_mime_type="application/json",
            response_json_schema=schema,
        )
    if thinking_level and model.startswith("gemini-3"):
        # Gemini 2.5 is kept as a compatible fallback, but it does not share
        # the Gemini 3 thinking-level surface. Let it use its default instead
        # of turning a healthy fallback into an unsupported-config error.
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, thinking_level)
        )
    return types.GenerateContentConfig(**kwargs)


def _unique(models: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(model.strip() for model in models if model and model.strip()))


def _models_for(lane: str) -> tuple[str, ...]:
    """Return a stable model pool for quality or routine work.

    Goal construction favours the more capable models. Intake, resource
    labelling, explanations, and chat favour the fast models first. The
    initial model is selected deterministically per request below, so a cold
    Vercel function does not repeatedly load every new goal onto one model.
    """
    configured = tuple(
        model.strip()
        for model in (settings.graph_models if lane == "graph" else settings.fast_models).split(",")
    )
    if lane == "graph":
        return _unique((settings.model, *configured))
    return _unique((*configured, settings.model))


def _pool_signature(lane: str) -> str:
    return ",".join(_models_for(lane))


def configured_models() -> dict[str, list[str]]:
    """Expose provider-safe pool metadata for the health endpoint."""
    return {"graph": list(_models_for("graph")), "fast": list(_models_for("fast"))}


def _model_order(lane: str, affinity: str) -> tuple[str, ...]:
    """Choose a model order with a lightweight per-process circuit breaker."""
    candidates = [model for model in _models_for(lane) if model not in _disabled_models]
    if not candidates:
        return ()

    now = monotonic()
    ready = [model for model in candidates if _model_cooldowns.get(model, 0) <= now]
    if not ready:
        # All models recently failed. Try only the one whose cooldown ends
        # first instead of making a burst of likely-wasted requests.
        return (min(candidates, key=lambda model: _model_cooldowns.get(model, 0)),)

    digest = hashlib.sha256(f"{lane}\x00{affinity}".encode("utf-8")).digest()
    start = int.from_bytes(digest[:4], "big") % len(ready)
    return tuple(ready[start:] + ready[:start])


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if value is not None:
            match = re.search(r"\b(401|403|404|429|5\d\d)\b", str(value))
            if match:
                return int(match.group(1))
    match = re.search(r"\b(401|403|404|429|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def _failure_kind(exc: Exception) -> tuple[str, int | None]:
    """Classify provider failures without logging their raw message."""
    code = _status_code(exc)
    message = str(exc).lower()
    auth_markers = (
        "invalid api key", "api key not valid", "authentication", "unauthenticated",
        "unauthorized", "credential",
    )
    if code == 401 or any(marker in message for marker in auth_markers):
        return "credentials", code
    if code in (404, 405) or "model not found" in message or "unsupported model" in message:
        return "unsupported", code
    # A 403 can be a model/tier restriction. It must not poison every other
    # model or turn a working credential into an app-wide offline mode.
    if code == 403 or "permission denied" in message:
        return "unsupported", code
    if code == 429 or any(marker in message for marker in ("resource exhausted", "quota", "rate limit")):
        return "quota", code
    if (code is not None and code >= 500) or any(
        marker in message for marker in ("unavailable", "high demand", "temporarily")
    ):
        return "temporary", code
    return "response", code


def _record_failure(model: str, exc: Exception, where: str) -> bool:
    """Record a model failure and return whether another model may be tried."""
    global _client_failed
    kind, code = _failure_kind(exc)
    detail = f"HTTP {code}" if code is not None else type(exc).__name__

    if kind == "credentials":
        log.warning("%s: Gemini credentials were rejected (%s)", where, detail)
        _client_failed = True
        return False
    if kind == "unsupported":
        _disabled_models.add(model)
        log.warning("%s: %s is unavailable for this account (%s); trying another model", where, model, detail)
        return True
    if kind == "quota":
        _model_cooldowns[model] = monotonic() + 15 * 60
        log.warning("%s: %s is rate-limited (%s); trying another model", where, model, detail)
        return True
    if kind == "temporary":
        _model_cooldowns[model] = monotonic() + 60
        log.warning("%s: %s is busy (%s); trying another model", where, model, detail)
        return True

    # Invalid JSON/empty answers are usually model-specific on a single turn.
    # Do not disable the model, but do let a different one repair the response.
    log.warning("%s: %s returned an unusable response (%s); trying another model", where, model, detail)
    return True


def _with_fallback(
    lane: str,
    affinity: str,
    where: str,
    request: Callable[[object, str], R],
) -> R | None:
    """Run a bounded request across the configured model pool, without waits."""
    if not available():
        return None

    client = _get_client()
    if client is None:
        return None
    for attempt, model in enumerate(_model_order(lane, affinity), start=1):
        try:
            result = request(client, model)
        except Exception as exc:  # provider, schema, and empty-response errors
            if not _record_failure(model, exc, where):
                return None
            continue
        log.info("%s completed with %s%s", where, model, " after fallback" if attempt > 1 else "")
        return result

    log.warning("%s: every configured Gemini model was unavailable", where)
    return None


def available() -> bool:
    return not settings.offline and _get_client() is not None


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]


def _cache_get(key: str):
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("dropping unreadable model cache %s (%s)", path.name, type(exc).__name__)
        path.unlink(missing_ok=True)
        return None


def _cache_put(key: str, value) -> None:
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")


def _response_text(response, *, strip: bool = True) -> str:
    """The SDK exposes `.text`; safely handle empty/non-text responses too.

    ``strip`` must be off for streamed fragments. A chunk boundary regularly
    falls on a space, so stripping every chunk deleted the word gaps and the
    assistant answered with "The firstcourse in your path".
    """
    try:
        text = str(response.text or "")
    except Exception as exc:  # e.g. blocked response with no text part
        log.info("Gemini returned no text (%s)", type(exc).__name__)
        return ""
    return text.strip() if strip else text


def _nonempty_response(response) -> str:
    text = _response_text(response)
    if not text:
        raise ValueError("empty model response")
    return text


def parse(
    schema: Type[T],
    prompt: str,
    system: str,
    effort: str = "low",
    max_tokens: int = 1600,
    *,
    lane: str = "graph",
) -> T | None:
    """Return a validated structured Gemini response, or ``None`` honestly."""
    key = _key(
        "parse", CACHE_VERSION, lane, _pool_signature(lane), schema.__name__, system, prompt,
        effort, str(max_tokens),
    )
    cached = _cache_get(key)
    if cached is not None:
        try:
            return schema.model_validate(cached)
        except Exception as exc:
            log.warning("dropping invalid %s cache (%s)", schema.__name__, type(exc).__name__)
            (CACHE_DIR / f"{key}.json").unlink(missing_ok=True)

    def request(client, model: str) -> T:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_config(
                model=model,
                system=system,
                max_output_tokens=max_tokens,
                schema=schema.model_json_schema(),
                thinking_level="LOW" if effort == "high" else "MINIMAL",
            ),
        )
        return schema.model_validate_json(_nonempty_response(response))

    out = _with_fallback(lane, key, "parse()", request)
    if out is None:
        return None
    _cache_put(key, out.model_dump(mode="json"))
    return out


def text(
    prompt: str,
    system: str,
    effort: str = "medium",
    max_tokens: int = 1500,
    *,
    lane: str = "fast",
) -> str | None:
    key = _key("text", CACHE_VERSION, lane, _pool_signature(lane), system, prompt, effort, str(max_tokens))
    cached = _cache_get(key)
    if cached is not None:
        return str(cached)

    def request(client, model: str) -> str:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_config(
                model=model,
                system=system,
                max_output_tokens=max_tokens,
                thinking_level="MINIMAL",
            ),
        )
        return _nonempty_response(response)

    out = _with_fallback(lane, key, "text()", request)
    if not out:
        return None
    _cache_put(key, out)
    return out


def _contents(messages: list[dict]) -> list[dict]:
    """Translate the app's history roles to the Gemini ``model`` role."""
    contents: list[dict] = []
    for message in messages:
        text = str(message.get("content", "")).strip()
        if not text:
            continue
        role = "model" if message.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents


def stream_text(
    messages: list[dict],
    system: str,
    effort: str = "medium",
    max_tokens: int = 4000,
    *,
    lane: str = "fast",
) -> Iterator[str]:
    """Stream chat text, falling back only before a model emits a token."""
    del effort  # retained in the provider-neutral call surface
    if not available():
        return

    client = _get_client()
    if client is None:
        return
    affinity = _key("stream", CACHE_VERSION, lane, _pool_signature(lane), system, json.dumps(messages, sort_keys=True))
    for attempt, model in enumerate(_model_order(lane, affinity), start=1):
        emitted = False
        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=_contents(messages),
                config=_config(
                    model=model,
                    system=system,
                    max_output_tokens=max_tokens,
                    thinking_level="MINIMAL",
                ),
            )
            for chunk in stream:
                chunk_text = _response_text(chunk, strip=False)
                if chunk_text:
                    emitted = True
                    yield chunk_text
            if emitted:
                log.info("stream_text() completed with %s%s", model, " after fallback" if attempt > 1 else "")
                return
            raise ValueError("empty streamed model response")
        except Exception as exc:
            if emitted:
                # Switching providers after sending partial text would splice
                # together two answers. The caller will preserve the honest
                # partial answer instead of pretending it completed.
                log.warning("stream_text(): %s ended after partial output", model)
                return
            if not _record_failure(model, exc, "stream_text()"):
                return

    log.warning("stream_text(): every configured Gemini model was unavailable")


def collect(
    system: str,
    prompt: str,
    tool_name: str,
    tool_schema: dict,
    *,
    effort: str = "medium",
    web_search: bool = False,
    max_rounds: int = 1,
    max_tokens: int = 1400,
    lane: str = "fast",
) -> dict | None:
    """Return structured JSON for callers that already supplied their evidence.

    Gemini is deliberately not asked to search here. Tavily does live search in
    ``engines.discovery``; this adapter only turns supplied source snippets
    into a schema-validated selection. Legacy arguments remain so callers have
    a narrow provider-neutral surface.
    """
    del tool_name, web_search, max_rounds
    schema_json = json.dumps(tool_schema, sort_keys=True)
    key = _key(
        "collect", CACHE_VERSION, lane, _pool_signature(lane), system, prompt, schema_json,
        effort, str(max_tokens),
    )
    cached = _cache_get(key)
    if isinstance(cached, dict):
        return cached

    def request(client, model: str) -> dict:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_config(
                model=model,
                system=system,
                max_output_tokens=max_tokens,
                schema=tool_schema,
                thinking_level="MINIMAL",
            ),
        )
        out = json.loads(_nonempty_response(response))
        if not isinstance(out, dict):
            raise ValueError("structured response was not a JSON object")
        return out

    out = _with_fallback(lane, key, "collect()", request)
    if out is None:
        return None
    _cache_put(key, out)
    return out
