"""Tests for structured FactStore and extraction."""

from __future__ import annotations

from pathlib import Path

from lsm.core.facts import (
    CAT_ASSISTANT_IDENTITY,
    CAT_USER_DISLIKES,
    CAT_USER_MEDIA,
    CAT_USER_OPEN_QUESTIONS,
    CAT_USER_PEOPLE,
    CAT_USER_PREFERENCES,
    CAT_USER_TASKS,
    CAT_USER_TRAITS,
    Fact,
    FactStore,
    apply_extracted_facts,
    extract_facts_from_user_turn,
    split_into_segments,
)
from lsm.core.identity import IdentityProfile


def test_fact_store_roundtrip(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_PREFERENCES,
            subject="restaurants",
            value="yakiniku",
        )
    )
    store2 = FactStore(tmp_path)
    prefs = store2.query(owner="user", category=CAT_USER_PREFERENCES)
    assert len(prefs) == 1
    assert prefs[0].value == "yakiniku"


def test_replace_by_subject_updates_value(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    store.replace_by_subject(
        owner="user",
        category=CAT_USER_MEDIA,
        subject="Crimson Desert",
        new_value="strategy game",
    )
    store.replace_by_subject(
        owner="user",
        category=CAT_USER_MEDIA,
        subject="Crimson Desert",
        new_value="action RPG",
    )
    media = store.query(owner="user", category=CAT_USER_MEDIA)
    assert len(media) == 1
    assert media[0].value == "action RPG"


def test_additive_preferences_accumulate(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    store.add_preference_value(subject="restaurants", value="yakiniku")
    store.add_preference_value(subject="restaurants", value="Korean BBQ")
    prefs = store.query(owner="user", category=CAT_USER_PREFERENCES)
    values = {p.value.lower() for p in prefs}
    assert "yakiniku" in values
    assert "korean bbq" in values


def test_extract_trait_not_task() -> None:
    facts = extract_facts_from_user_turn("I am always expanding my knowledge base and skillset.")
    categories = {f.category for f in facts}
    assert CAT_USER_TRAITS in categories
    assert CAT_USER_TASKS not in categories


def test_apply_extracted_media_correction(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    apply_extracted_facts(
        store,
        "actually Crimson Desert is not a strategy game, it is an action RPG",
    )
    media = store.query(owner="user", category=CAT_USER_MEDIA)
    assert any("action" in f.value.lower() for f in media)


def test_sync_assistant_identity(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    profile = IdentityProfile(name="Elsum", description="Helper.")
    f = store.sync_assistant_identity(profile)
    assert f.owner == "assistant"
    assert "Elsum" in f.value


# --------------------------------------------------------------------------- #
# F.1.1 hardening cases
# --------------------------------------------------------------------------- #


def test_arpg_correction_overrides_strategy(tmp_path: Path) -> None:
    """Crimson Desert: strategy first, then ARPG correction wins, no leftover strategy."""
    store = FactStore(tmp_path)
    apply_extracted_facts(store, "i like the game Crimson Desert, it is a strategy game")
    apply_extracted_facts(
        store,
        "actually Crimson Desert is not a strategy game, it is an action RPG",
    )
    media = store.query(owner="user", category=CAT_USER_MEDIA)
    cd = [f for f in media if (f.subject or "").lower() == "crimson desert"]
    assert len(cd) == 1, media
    assert "action" in cd[0].value.lower()
    assert "strategy" not in cd[0].value.lower()


def test_restaurants_without_word_restaurants(tmp_path: Path) -> None:
    """`the answer to three is yakiniku and korean bbq` → two restaurant prefs."""
    store = FactStore(tmp_path)
    apply_extracted_facts(
        store,
        "actually the answer to three is yakiniku and korean bbq",
    )
    prefs = store.query(owner="user", category=CAT_USER_PREFERENCES)
    restaurant_vals = [f.value.lower() for f in prefs if f.subject == "restaurants"]
    assert any("yakiniku" in v for v in restaurant_vals)
    assert any("korean bbq" in v for v in restaurant_vals)


def test_identity_quiz_does_not_extract_traits(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    apply_extracted_facts(store, "how do you know i am reishi?")
    traits = store.query(owner="user", category=CAT_USER_TRAITS)
    assert traits == []


def test_multi_answer_correction_three_segments(tmp_path: Path) -> None:
    """A 3-part correction with mixed numbering: media + restaurants + no traits."""
    store = FactStore(tmp_path)
    msg = (
        "actually, here are corrections. "
        "the answer to one is Crimson Desert is an action RPG, not a strategy game. "
        "the answer to two is i like several flavors of ice cream. "
        "the answer to three is yakiniku and korean bbq."
    )
    apply_extracted_facts(store, msg)

    media = store.query(owner="user", category=CAT_USER_MEDIA)
    cd = [f for f in media if (f.subject or "").lower() == "crimson desert"]
    assert len(cd) == 1, media
    assert "action" in cd[0].value.lower()

    prefs = store.query(owner="user", category=CAT_USER_PREFERENCES)
    restaurant_vals = [f.value.lower() for f in prefs if f.subject == "restaurants"]
    assert any("yakiniku" in v for v in restaurant_vals)
    assert any("korean bbq" in v for v in restaurant_vals)

    traits = store.query(owner="user", category=CAT_USER_TRAITS)
    assert traits == []


def test_long_preference_value_rejected(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    blob = (
        "i like several flavors of icecream including custard, chocolate, vanilla, "
        "strawberry, pistachio, and mint chocolate chip on alternating tuesdays"
    )
    apply_extracted_facts(store, blob)
    prefs = store.query(owner="user", category=CAT_USER_PREFERENCES)
    for p in prefs:
        assert len(p.value) <= 80, f"got long pref: {p.value!r}"


def test_correction_does_not_introduce_traits(tmp_path: Path) -> None:
    """Trade-off documented in plan: correction turns never emit user.traits."""
    store = FactStore(tmp_path)
    apply_extracted_facts(store, "actually i am a vegetarian")
    traits = store.query(owner="user", category=CAT_USER_TRAITS)
    assert traits == []


def test_split_into_segments_numbered_dot() -> None:
    parts = split_into_segments(
        "actually here are corrections.\n"
        "1. Crimson Desert is an action RPG.\n"
        "2. yakiniku and korean bbq.\n"
        "3. i like ice cream."
    )
    assert len(parts) >= 2
    assert any("crimson desert" in p.lower() for p in parts)
    assert any("yakiniku" in p.lower() for p in parts)


def test_split_into_segments_parens_and_bullets() -> None:
    parts = split_into_segments(
        "fixes:\n(1) action RPG\n(2) yakiniku and korean bbq\n(3) custard ice cream"
    )
    assert len(parts) >= 2

    parts2 = split_into_segments(
        "corrections:\n- action RPG\n- yakiniku and korean bbq\n- custard ice cream"
    )
    assert len(parts2) >= 2


def test_split_into_segments_fallback_single() -> None:
    parts = split_into_segments("just a plain sentence")
    assert parts == ["just a plain sentence"]


def test_extracted_facts_carry_source_segment() -> None:
    facts = extract_facts_from_user_turn(
        "actually Crimson Desert is not a strategy game, it is an action RPG"
    )
    media = [f for f in facts if f.category == CAT_USER_MEDIA]
    assert media
    assert media[0].source_segment is not None
    assert "crimson desert" in (media[0].source_segment or "").lower()


def test_delete_where_filters(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    store.upsert(Fact(owner="user", category=CAT_USER_TRAITS, value="reishi"))
    store.upsert(Fact(owner="user", category=CAT_USER_TRAITS, value="ctually reishi"))
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_PREFERENCES,
            subject="general",
            value="paragraph blob preference",
        )
    )
    n = store.delete_where(category=CAT_USER_TRAITS)
    assert n == 2
    assert store.query(owner="user", category=CAT_USER_TRAITS) == []
    prefs = store.query(owner="user", category=CAT_USER_PREFERENCES)
    assert len(prefs) == 1


def test_clear_all_preserves_assistant_identity(tmp_path: Path) -> None:
    store = FactStore(tmp_path)
    store.sync_assistant_identity(IdentityProfile(name="Elsum", description="Helper."))
    store.upsert(Fact(owner="user", category=CAT_USER_PEOPLE, value="reishi"))
    store.upsert(Fact(owner="user", category=CAT_USER_DISLIKES, value="spam"))
    n = store.clear_all(preserve_assistant_identity=True)
    assert n == 2
    remaining = store.all_facts()
    assert len(remaining) == 1
    assert remaining[0].category == CAT_ASSISTANT_IDENTITY


def test_low_quality_trait_filtered() -> None:
    """Bare 'reishi' and 'ctually reishi' do not become traits on a plain message."""
    facts = extract_facts_from_user_turn("i am reishi")
    traits = [f for f in facts if f.category == CAT_USER_TRAITS]
    assert traits == [], traits


# --------------------------------------------------------------------------- #
# Open-question gate (F.2 review fixes)
# --------------------------------------------------------------------------- #


def test_plain_question_is_not_open_question() -> None:
    """Routine '?' messages should not pollute user.open_questions."""
    facts = extract_facts_from_user_turn("what's the weather like?")
    open_qs = [f for f in facts if f.category == CAT_USER_OPEN_QUESTIONS]
    assert open_qs == [], open_qs


def test_identity_quiz_is_not_open_question() -> None:
    """Identity-quiz turns must not emit open_questions facts."""
    facts = extract_facts_from_user_turn("how do you know i am reishi?")
    open_qs = [f for f in facts if f.category == CAT_USER_OPEN_QUESTIONS]
    assert open_qs == [], open_qs


def test_i_wonder_emits_open_question_with_subject() -> None:
    """Positive cue still produces an open-question fact with a topic chip."""
    facts = extract_facts_from_user_turn("i wonder about the future of small local models")
    open_qs = [f for f in facts if f.category == CAT_USER_OPEN_QUESTIONS]
    assert len(open_qs) == 1
    assert open_qs[0].subject is not None
    assert "future" in open_qs[0].subject.lower()
    assert "i wonder" in open_qs[0].value.lower()


def test_open_question_tag_emits_without_subject() -> None:
    """Explicit (open question) marker is accepted even without cue-phrase subject."""
    facts = extract_facts_from_user_turn("should we try lance? (open question)")
    open_qs = [f for f in facts if f.category == CAT_USER_OPEN_QUESTIONS]
    assert len(open_qs) == 1
