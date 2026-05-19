"""Tests for deterministic F.2 state projection."""

from __future__ import annotations

import time

import pytest
from lsm.core.facts import (
    CAT_ASSISTANT_NOTES,
    CAT_USER_DISLIKES,
    CAT_USER_MEDIA,
    CAT_USER_OPEN_QUESTIONS,
    CAT_USER_PEOPLE,
    CAT_USER_PREFERENCES,
    CAT_USER_TASKS,
    CAT_USER_TRAITS,
    Fact,
)
from lsm.core.identity import IdentityProfile
from lsm.core.projection import (
    DeterministicProjection,
    LLMPolishedProjection,
    ProjectionContext,
)


def _fact(category: str, value: str, *, subject: str | None = None, age_s: float = 0.0) -> Fact:
    now = time.time()
    ts = now - age_s
    return Fact(
        owner="user",
        category=category,
        subject=subject,
        value=value,
        created_at=ts,
        updated_at=ts,
    )


@pytest.mark.asyncio
async def test_projection_empty_without_identity() -> None:
    p = DeterministicProjection()
    txt = await p.render(ProjectionContext(facts=[]))
    assert txt == ""


@pytest.mark.asyncio
async def test_projection_identity_only() -> None:
    p = DeterministicProjection()
    ident = IdentityProfile(name="Elsum", description="Assistant.")
    txt = await p.render(ProjectionContext(facts=[], identity=ident))
    assert txt == "identity/role: Elsum - Assistant."


@pytest.mark.asyncio
async def test_projection_renders_core_slots() -> None:
    p = DeterministicProjection()
    ident = IdentityProfile(name="Elsum", description="Assistant.")
    facts = [
        _fact(CAT_USER_PEOPLE, "reishi"),
        _fact(CAT_USER_PREFERENCES, "yakiniku", subject="restaurants"),
        _fact(CAT_USER_PREFERENCES, "korean bbq", subject="restaurants"),
        _fact(CAT_USER_DISLIKES, "crowded places"),
        _fact(CAT_USER_MEDIA, "action RPG", subject="Crimson Desert"),
        _fact(CAT_USER_TASKS, "remind me to ship F.2"),
        _fact(CAT_USER_OPEN_QUESTIONS, "what is the right verifier threshold?"),
        _fact(CAT_ASSISTANT_NOTES, "keep answers concise"),
    ]
    txt = await p.render(
        ProjectionContext(facts=facts, identity=ident, rolling_notes="small thought")
    )
    assert "identity/role: Elsum - Assistant." in txt
    assert "people/names: " in txt
    assert "preferences/likes: restaurants: " in txt
    assert "games/media: " in txt
    assert "open questions: " in txt
    assert "assistant notes: " in txt
    assert "notes: small thought" in txt


@pytest.mark.asyncio
async def test_projection_recency_marks_recent_facts() -> None:
    p = DeterministicProjection(recency_window_s=60.0)
    facts = [
        _fact(CAT_USER_PREFERENCES, "yakiniku", subject="restaurants", age_s=1.0),
        _fact(CAT_USER_PREFERENCES, "korean bbq", subject="restaurants", age_s=300.0),
    ]
    txt = await p.render(ProjectionContext(facts=facts))
    assert "*yakiniku" in txt
    assert "korean bbq" in txt


@pytest.mark.asyncio
async def test_projection_correction_slots_float_earlier_via_categories() -> None:
    """Reorder is driven by touched_categories, not memory-text matching."""
    p = DeterministicProjection()
    facts = [
        _fact(CAT_USER_PREFERENCES, "tea", subject="drinks"),
        _fact(CAT_USER_MEDIA, "shoegaze", subject="Slowdive"),
    ]
    ctx = ProjectionContext(
        facts=facts,
        correction_just_happened=True,
        touched_categories=frozenset({CAT_USER_MEDIA}),
    )
    txt = await p.render(ctx)
    assert txt.index("games/media:") < txt.index("preferences/likes:")


@pytest.mark.asyncio
async def test_projection_correction_reorder_noop_when_no_touched_categories() -> None:
    """Without touched_categories, default slot order is preserved."""
    p = DeterministicProjection()
    facts = [
        _fact(CAT_USER_PREFERENCES, "tea", subject="drinks"),
        _fact(CAT_USER_MEDIA, "shoegaze", subject="Slowdive"),
    ]
    ctx = ProjectionContext(
        facts=facts,
        correction_just_happened=True,
        touched_categories=frozenset(),
    )
    txt = await p.render(ctx)
    assert txt.index("preferences/likes:") < txt.index("games/media:")


@pytest.mark.asyncio
async def test_projection_surprise_trims_traits_and_notes() -> None:
    p = DeterministicProjection(surprise_trim_threshold=1.5)
    facts = [
        _fact(CAT_USER_TRAITS, "curious"),
        _fact(CAT_USER_PREFERENCES, "tea"),
    ]
    txt = await p.render(
        ProjectionContext(
            facts=facts,
            rolling_notes="this should be removed",
            recent_surprise=2.0,
        )
    )
    assert "traits/personality:" not in txt
    assert "notes:" not in txt
    assert "preferences/likes:" in txt


class _StubPolishBackend:
    """Minimal LLMBackend stand-in for polish-guard tests."""

    name = "stub"
    model = "stub"

    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, messages, options):  # type: ignore[no-untyped-def]
        return self._response

    async def stream(self, messages, options):  # type: ignore[no-untyped-def]
        yield self._response

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_polish_guard_accepts_when_only_asterisks_stripped() -> None:
    """A polish that drops `*` recency markers should be accepted."""
    facts = [
        _fact(CAT_USER_PEOPLE, "reishi", age_s=1.0),
        _fact(CAT_USER_PREFERENCES, "tea", subject="drinks", age_s=1.0),
    ]
    base = DeterministicProjection()
    body = await base.render(ProjectionContext(facts=facts))
    assert "*" in body  # recency marker present
    polished_text = body.replace("*", "")
    backend = _StubPolishBackend(polished_text)
    polish = LLMPolishedProjection(base=base, backend=backend, enabled=True)
    out = await polish.render(ProjectionContext(facts=facts))
    assert out == polished_text  # polish accepted, not falling back


@pytest.mark.asyncio
async def test_polish_guard_rejects_content_change() -> None:
    """A polish that changes an actual value should be rejected (fallback)."""
    facts = [_fact(CAT_USER_PEOPLE, "reishi", age_s=1.0)]
    base = DeterministicProjection()
    body = await base.render(ProjectionContext(facts=facts))
    bad = body.replace("reishi", "someone else")
    backend = _StubPolishBackend(bad)
    polish = LLMPolishedProjection(base=base, backend=backend, enabled=True)
    out = await polish.render(ProjectionContext(facts=facts))
    assert out == body  # fell back to deterministic


@pytest.mark.asyncio
async def test_projection_budget_trims_from_bottom() -> None:
    p = DeterministicProjection()
    facts = [
        _fact(CAT_USER_PEOPLE, "reishi"),
        _fact(CAT_USER_MEDIA, "action RPG", subject="Crimson Desert"),
        _fact(CAT_USER_PREFERENCES, "yakiniku", subject="restaurants"),
        _fact(CAT_ASSISTANT_NOTES, "note-" + ("x" * 120)),
        _fact(CAT_USER_OPEN_QUESTIONS, "question-" + ("y" * 120)),
    ]
    txt = await p.render(ProjectionContext(facts=facts, rolling_notes="z" * 120, max_chars=140))
    assert "people/names:" in txt
    assert "games/media:" in txt
    # Lowest-priority lines should be dropped first.
    assert "assistant notes:" not in txt or "open questions:" not in txt
