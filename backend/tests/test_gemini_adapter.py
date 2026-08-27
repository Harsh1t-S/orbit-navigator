"""Provider-boundary tests using fakes: no network or real keys required."""

from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from app import llm


class _Answer(BaseModel):
    answer: str


class _Models:
    def __init__(self):
        self.generate_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(text='{"answer":"grounded"}')

    def generate_content_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return iter([SimpleNamespace(text="first "), SimpleNamespace(text="second")])


def _install_fake_client(monkeypatch):
    models = _Models()
    client = SimpleNamespace(models=models)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    return models


def test_parse_uses_json_schema_and_cache(tmp_path, monkeypatch):
    models = _install_fake_client(monkeypatch)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)

    first = llm.parse(_Answer, "prompt", "system")
    second = llm.parse(_Answer, "prompt", "system")

    assert first == _Answer(answer="grounded")
    assert second == first
    assert len(models.generate_calls) == 1
    config = models.generate_calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["type"] == "object"


def test_stream_maps_assistant_to_gemini_model_role(monkeypatch):
    models = _install_fake_client(monkeypatch)

    chunks = list(
        llm.stream_text(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            "system",
        )
    )

    # Fragments are yielded exactly as the model produced them - the space
    # after "first" is the gap between two words, not padding to trim.
    assert chunks == ["first ", "second"]
    assert "".join(chunks) == "first second"
    assert models.stream_calls[0]["contents"] == [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"text": "hi"}]},
    ]


def test_parse_uses_another_model_when_the_first_is_busy(tmp_path, monkeypatch):
    models = _install_fake_client(monkeypatch)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "_model_cooldowns", {})
    monkeypatch.setattr(llm, "_disabled_models", set())
    monkeypatch.setattr(llm, "_model_order", lambda *_: ("busy-model", "fallback-model"))

    def busy_then_success(**kwargs):
        models.generate_calls.append(kwargs)
        if len(models.generate_calls) == 1:
            raise RuntimeError("503 UNAVAILABLE: high demand")
        return SimpleNamespace(text='{"answer":"grounded"}')

    monkeypatch.setattr(models, "generate_content", busy_then_success)

    assert llm.parse(_Answer, "new prompt", "system") == _Answer(answer="grounded")
    assert [call["model"] for call in models.generate_calls] == ["busy-model", "fallback-model"]
    assert "busy-model" in llm._model_cooldowns


def test_model_specific_403_does_not_disable_all_gemini_requests(monkeypatch):
    models = _install_fake_client(monkeypatch)
    monkeypatch.setattr(llm, "_client_failed", False)
    monkeypatch.setattr(llm, "_model_cooldowns", {})
    monkeypatch.setattr(llm, "_disabled_models", set())
    monkeypatch.setattr(llm, "_models_for", lambda _: ("restricted-model", "working-model"))

    assert llm._record_failure("restricted-model", RuntimeError("403 permission denied"), "test")
    assert not llm._client_failed
    assert llm._model_order("graph", "any request") == ("working-model",)


def test_streamed_chunks_keep_the_spaces_between_them():
    """A chunk boundary on a space must not eat the space.

    Stripping each streamed fragment produced "The firstcourse in your path"
    in the assistant panel.
    """
    from app import llm

    class _Chunk:
        def __init__(self, text):
            self.text = text

    joined = "".join(
        llm._response_text(_Chunk(part), strip=False)
        for part in ["The first", " course", " in your path", " covers Docker", " & Azure."]
    )
    assert joined == "The first course in your path covers Docker & Azure."
    # A complete response is still tidied.
    assert llm._response_text(_Chunk("  answer  ")) == "answer"
