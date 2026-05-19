"""fact_correction_v1 — scripted fact-fidelity scenarios (no live LLM).

Each scenario replays user/assistant turn pairs through fact extraction and
the summarizer updater with a capturing backend. Assertions check FactStore
and optional ``s_t.text`` substrings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lsm.core.embeddings import HashEmbedder
from lsm.core.facts import (
    CAT_USER_MEDIA,
    CAT_USER_PREFERENCES,
    CAT_USER_TASKS,
    CAT_USER_TRAITS,
    FactStore,
    apply_extracted_facts,
)
from lsm.core.identity import IdentityProfile
from lsm.core.projection import DeterministicProjection
from lsm.core.state import FactsBackedUpdater, LLMSummarizerUpdater, State

TASK_NAME = "fact_correction_v1"


@dataclass(slots=True)
class ScenarioTurn:
    user: str
    assistant: str = "acknowledged."


@dataclass(slots=True)
class ScenarioExpect:
    fact_count_min: int = 0
    preference_values_contain: list[str] = field(default_factory=list)
    media_subject_value: tuple[str, str] | None = None
    trait_present: bool = False
    trait_absent: bool = False
    task_absent: bool = False
    st_contains: list[str] = field(default_factory=list)
    st_excludes: list[str] = field(default_factory=list)
    st_equals: str | None = None
    preference_subject_count: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class FactScenario:
    scenario_id: str
    turns: list[ScenarioTurn]
    expect: ScenarioExpect
    summarizer_reply: str = "summary placeholder"


SCENARIOS: list[FactScenario] = [
    FactScenario(
        scenario_id="additive_restaurants",
        turns=[
            ScenarioTurn("i like yakiniku restaurants"),
            ScenarioTurn(
                "actually i prefer korean bbq",
                assistant="noted.",
            ),
            ScenarioTurn(
                'no, please make it "korean bbq and yakiniku restaurants."',
                assistant="got it.",
            ),
        ],
        expect=ScenarioExpect(
            fact_count_min=2,
            preference_values_contain=["yakiniku", "korean bbq"],
            st_contains=["yakiniku", "korean bbq"],
        ),
        summarizer_reply=(
            "identity/role: Elsum - assistant\n"
            "preferences/likes: korean bbq restaurants\n"
            "people/names: Reishi\n"
        ),
    ),
    FactScenario(
        scenario_id="genre_correction",
        turns=[
            ScenarioTurn("i like the game Crimson Desert, it is a strategy game"),
            ScenarioTurn(
                "actually Crimson Desert is not a strategy game, it is an action RPG",
                assistant="understood.",
            ),
        ],
        expect=ScenarioExpect(
            media_subject_value=("Crimson Desert", "action RPG"),
        ),
        summarizer_reply=(
            "identity/role: Elsum - assistant\n" "games/media: Crimson Desert (strategy game)\n"
        ),
    ),
    FactScenario(
        scenario_id="identity_slots",
        turns=[
            ScenarioTurn("your name is Elsum. my name is reishi."),
        ],
        expect=ScenarioExpect(
            st_contains=["identity/role: Elsum"],
            st_excludes=["people/names: Elsum"],
            st_equals="identity/role: Elsum - Test assistant.\npeople/names: *reishi",
        ),
        summarizer_reply=("identity/role: Reishi\npeople/names: Elsum, Reishi\n"),
    ),
    FactScenario(
        scenario_id="trait_not_task",
        turns=[
            ScenarioTurn(
                "I am always expanding my knowledge base and skillset, constantly "
                "pushing the boundaries of what we believe."
            ),
        ],
        expect=ScenarioExpect(
            trait_present=True,
            task_absent=True,
        ),
        summarizer_reply=(
            "identity/role: Elsum\n" "ongoing tasks: expanding knowledge base and skillset\n"
        ),
    ),
    FactScenario(
        scenario_id="identity_quiz_clarification",
        turns=[
            ScenarioTurn(
                "how do you know i am reishi? i am actually reishi.",
                assistant="i can ask a memorable question to confirm.",
            ),
            ScenarioTurn(
                "actually, here are corrections. "
                "the answer to one is Crimson Desert is an action RPG, "
                "not a strategy game. "
                "the answer to two is i like several flavors of ice cream. "
                "the answer to three is yakiniku and korean bbq.",
                assistant="thanks, recording corrections.",
            ),
        ],
        expect=ScenarioExpect(
            media_subject_value=("Crimson Desert", "action RPG"),
            preference_values_contain=["yakiniku", "korean bbq"],
            trait_absent=True,
            preference_subject_count={"restaurants": 2},
            st_excludes=["strategy game"],
        ),
        summarizer_reply=(
            "identity/role: Elsum - assistant\n"
            "people/names: Reishi\n"
            "games/media: Crimson Desert (action RPG)\n"
            "preferences/likes: yakiniku, korean bbq\n"
        ),
    ),
]


class _CapturingBackend:
    name = "capturing"
    model = "capturing"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last_body: str = ""

    async def generate(self, messages, options):  # type: ignore[no-untyped-def]
        for m in messages:
            if getattr(m, "role", None) == "user":
                self.last_body = m.content
        return self._reply

    async def stream(self, messages, options):  # type: ignore[no-untyped-def]
        yield await self.generate(messages, options)

    async def close(self) -> None:
        return None


async def run_scenario(
    scenario: FactScenario,
    tmp_home: Path,
    *,
    reply: str | None = None,
    updater_mode: Literal["facts_backed", "llm_summarizer"] = "facts_backed",
) -> tuple[FactStore, State]:
    """Replay turns; return final fact store and state."""
    store = FactStore(tmp_home)
    identity = IdentityProfile(name="Elsum", description="Test assistant.")
    store.sync_assistant_identity(identity)
    backend = _CapturingBackend(reply or scenario.summarizer_reply)
    if updater_mode == "llm_summarizer":
        updater = LLMSummarizerUpdater(
            backend,
            HashEmbedder(dim=32),
            identity=identity,
            fact_store=store,
        )
    else:
        updater = FactsBackedUpdater(
            HashEmbedder(dim=32),
            DeterministicProjection(),
            identity=identity,
            fact_store=store,
        )
    state = State()
    for turn in scenario.turns:
        apply_extracted_facts(store, turn.user)
        state = await updater.update(state, turn.user, turn.assistant)
    return store, state


def assert_scenario(
    scenario: FactScenario,
    store: FactStore,
    state: State,
    *,
    enforce_st_equals: bool = True,
) -> list[str]:
    """Return list of failure messages (empty if all pass)."""
    failures: list[str] = []
    exp = scenario.expect
    facts = store.all_facts()
    if len(facts) < exp.fact_count_min:
        failures.append(f"expected >={exp.fact_count_min} facts, got {len(facts)}")
    if exp.preference_values_contain:
        prefs = store.query(owner="user", category=CAT_USER_PREFERENCES)
        blob = " ".join(f.value.lower() for f in prefs)
        for needle in exp.preference_values_contain:
            if needle.lower() not in blob:
                failures.append(f"missing preference value {needle!r} in {blob!r}")
    if exp.media_subject_value is not None:
        subj, val = exp.media_subject_value
        media = store.query(owner="user", category=CAT_USER_MEDIA)
        found = any(
            (f.subject or "").lower() == subj.lower() and val.lower() in f.value.lower()
            for f in media
        )
        if not found:
            failures.append(f"media fact {subj!r} -> {val!r} not found")
    if exp.trait_present:
        traits = store.query(owner="user", category=CAT_USER_TRAITS)
        if not traits:
            failures.append("expected user.traits fact")
    if exp.trait_absent:
        traits = store.query(owner="user", category=CAT_USER_TRAITS)
        if traits:
            failures.append(f"expected no user.traits, got {[t.value for t in traits]}")
    if exp.preference_subject_count:
        prefs = store.query(owner="user", category=CAT_USER_PREFERENCES)
        for subj, expected in exp.preference_subject_count.items():
            actual = sum(1 for f in prefs if (f.subject or "") == subj)
            if actual < expected:
                failures.append(
                    f"expected >={expected} preferences for subject {subj!r}, got {actual}"
                )
    if exp.task_absent:
        tasks = store.query(owner="user", category=CAT_USER_TASKS)
        if tasks:
            failures.append(f"expected no user.tasks, got {tasks}")
    text = state.text.lower()
    for needle in exp.st_contains:
        if needle.lower() not in text:
            failures.append(f"s_t missing {needle!r}")
    for needle in exp.st_excludes:
        if needle.lower() in text:
            failures.append(f"s_t should exclude {needle!r}")
    if (
        enforce_st_equals
        and exp.st_equals is not None
        and state.text.strip() != exp.st_equals.strip()
    ):
        failures.append(f"expected exact s_t.text {exp.st_equals!r}, got {state.text!r}")
    return failures


async def run_all_scenarios(
    tmp_home: Path,
    on_scenario: Callable[[str, list[str]], None] | None = None,
) -> dict[str, list[str]]:
    """Run every scenario; return map of id -> failure messages."""
    results: dict[str, list[str]] = {}
    for sc in SCENARIOS:
        sub = tmp_home / sc.scenario_id
        sub.mkdir(parents=True, exist_ok=True)
        store, state = await run_scenario(sc, sub)
        failures = assert_scenario(sc, store, state)
        results[sc.scenario_id] = failures
        if on_scenario is not None:
            on_scenario(sc.scenario_id, failures)
    return results
