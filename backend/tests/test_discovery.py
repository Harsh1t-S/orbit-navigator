"""Tavily/Gemini discovery stays grounded without calling either service."""

from __future__ import annotations

from app.engines import discovery


def _row(url: str, skill: str) -> dict:
    return {
        "title": "Grounded Python course",
        "provider": "Example Academy",
        "url": url,
        "description": "A Python course.",
        "level": "beginner",
        "hours": 8,
        "hours_stated": True,
        "cost": "free",
        "format": "interactive",
        "teaches": [{"skill": skill, "strength": 0.9}],
        "requires": [],
    }


def test_discovery_rejects_url_not_returned_by_tavily(goal_spec):
    row = _row("https://made-up.example/python", goal_spec.skills[0].id)
    assert discovery._to_course(
        row,
        {skill.id for skill in goal_spec.skills},
        {"https://search-result.example/python"},
    ) is None


def test_discovery_uses_search_evidence_and_cache(goal_spec, tmp_path, monkeypatch):
    class FakeTavily:
        calls: list[dict] = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "results": [
                    {
                        "title": "Python tutorial",
                        "url": "https://search-result.example/python",
                        "content": "An 8 hour Python tutorial.",
                    }
                ]
            }

    fake = FakeTavily()
    cache = tmp_path / "resources"
    cache.mkdir()
    monkeypatch.setattr(discovery, "RESOURCE_CACHE", cache)
    monkeypatch.setattr(discovery, "available", lambda: True)
    monkeypatch.setattr(discovery, "_get_client", lambda: fake)
    monkeypatch.setattr(discovery.llm, "available", lambda: True)

    def collect(system, prompt, *args, **kwargs):
        assert "Tavily search results" in prompt
        assert "https://search-result.example/python" in prompt
        return {
            "resources": [
                _row("https://search-result.example/python", goal_spec.skills[0].id),
                _row("https://invented.example/python", goal_spec.skills[0].id),
            ]
        }

    monkeypatch.setattr(discovery.llm, "collect", collect)

    first = discovery.find(goal_spec, goal_spec.skills[:1], budget="free")
    second = discovery.find(goal_spec, goal_spec.skills[:1], budget="free")

    assert [course.url for course in first] == ["https://search-result.example/python"]
    assert [course.url for course in second] == ["https://search-result.example/python"]
    assert len(fake.calls) == 1
    assert fake.calls[0]["search_depth"] == "basic"
    assert goal_spec.goal_title in fake.calls[0]["query"]
