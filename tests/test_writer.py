"""Tests for the gated memory writer and memory_write_precision_v1 benchmark."""

from __future__ import annotations

from pathlib import Path

from lsm.core.writer import (
    AlwaysWriter,
    LabeledCandidate,
    LearnedWriter,
    LearnedWriterConfig,
    NeverWriter,
    RuleBasedWriter,
    RuleBasedWriterConfig,
    WriteCandidate,
    train_learned_writer,
)


def _cand(text: str, role: str = "user", novelty: float | None = None) -> WriteCandidate:
    return WriteCandidate(text=text, role=role, novelty=novelty)


def test_always_writer_keeps_non_empty_and_drops_empty() -> None:
    w = AlwaysWriter()
    assert w.decide(_cand("my name is Kalen")).write is True
    assert w.decide(_cand("")).write is False
    assert w.decide(_cand("   ")).write is False


def test_never_writer_never_writes() -> None:
    w = NeverWriter()
    assert w.decide(_cand("my name is Kalen")).write is False
    assert w.decide(_cand("ok")).write is False


def test_rules_writer_accepts_durable_facts() -> None:
    w = RuleBasedWriter()
    for text in [
        "my name is Kalen",
        "i prefer green tea over coffee",
        "remember that the deploy target is lunar",
        "i work at archive as a software engineer",
        "i live in lumen",
    ]:
        d = w.decide(_cand(text))
        assert d.write is True, f"expected write for {text!r}: {d.reason}"
        assert d.score > 0


def test_rules_writer_rejects_filler_and_short() -> None:
    w = RuleBasedWriter()
    for text in [
        "ok",
        "thanks",
        "hi",
        "hmm",
        "sure sure",
        "got it",
        "yeah",
    ]:
        d = w.decide(_cand(text))
        assert d.write is False, f"expected skip for {text!r}: {d.reason}"


def test_rules_writer_rejects_pure_questions() -> None:
    w = RuleBasedWriter()
    d = w.decide(_cand("what time is it today?"))
    assert d.write is False
    assert d.reason == "question_only"


def test_rules_writer_novelty_threshold() -> None:
    cfg = RuleBasedWriterConfig(min_novelty=0.4)
    w = RuleBasedWriter(cfg)
    low = w.decide(_cand("my name is Kalen", novelty=0.1))
    assert low.write is False
    assert low.reason == "low_novelty"
    high = w.decide(_cand("my name is Kalen", novelty=0.9))
    assert high.write is True


def test_rules_writer_assistant_requires_durable_hit() -> None:
    w = RuleBasedWriter()
    generic = w.decide(_cand("sure, that's a great point to discuss", role="assistant"))
    assert generic.write is False
    hit = w.decide(_cand("remember that the server ip is 10.0.0.5", role="assistant"))
    assert hit.write is True


def test_learned_writer_falls_back_when_untrained() -> None:
    w = LearnedWriter()
    d = w.decide(_cand("my favourite colour is green"))
    assert d.write is True
    assert d.reason.startswith("fallback:")


def test_learned_writer_can_learn_mixture() -> None:
    examples: list[LabeledCandidate] = []
    signal = [
        "my name is Kalen",
        "i prefer green tea",
        "our family car is a blue wagon",
        "i live in lumen",
        "remember the server ip is 10.0.0.5",
        "my dog's name is azuki",
        "call me gee",
        "i work at archive as a software engineer",
    ]
    filler = [
        "ok",
        "thanks",
        "cool cool",
        "yeah sure",
        "got it",
        "interesting",
        "hi there",
        "hmm",
    ]
    examples += [LabeledCandidate(_cand(t), 1) for t in signal]
    examples += [LabeledCandidate(_cand(t), 0) for t in filler]

    model = train_learned_writer(examples, epochs=600, lr=0.5)
    writer = LearnedWriter(LearnedWriterConfig(threshold=0.5), model=model)

    for t in signal:
        assert writer.decide(_cand(t)).write is True, t
    for t in filler:
        assert writer.decide(_cand(t)).write is False, t


def test_learned_writer_save_load_roundtrip(tmp_path: Path) -> None:
    examples = [
        LabeledCandidate(_cand("my name is Kalen"), 1),
        LabeledCandidate(_cand("ok"), 0),
        LabeledCandidate(_cand("i prefer tea"), 1),
        LabeledCandidate(_cand("thanks"), 0),
    ]
    model = train_learned_writer(examples, epochs=200, lr=0.5)
    w1 = LearnedWriter(LearnedWriterConfig(threshold=0.5), model=model)
    p = tmp_path / "model.json"
    w1.save(p)

    w2 = LearnedWriter()
    w2.load(p)
    assert w2.model is not None
    assert set(w2.model.weights.keys()) == set(model.weights.keys())
    assert w2.decide(_cand("my name is Kalen")).write is True


def test_memory_write_precision_v1_rules_and_learned_beat_always() -> None:
    from lsm.eval.tasks.memory_write_precision_v1 import (
        WritePrecisionConfig,
        run,
    )

    result = run(WritePrecisionConfig(n_signal=30, n_filler=30, seed=1))
    by_name = {r["writer"]: r for r in result["results"]}

    always = by_name["always"]
    rules = by_name["rules"]
    learned = by_name["learned"]
    never = by_name["never"]

    # baselines
    assert always["recall"] == 1.0
    assert always["false_positive_rate"] == 1.0
    assert never["recall"] == 0.0
    assert never["false_positive_rate"] == 0.0

    # rules must strictly beat always on F1 by cutting false positives.
    assert rules["f1"] > always["f1"]
    assert rules["false_positive_rate"] < always["false_positive_rate"]

    # learned must not regress against always.
    assert learned["f1"] >= always["f1"]
