"""Build unified verifier datasets from telemetry + mined labels."""

from __future__ import annotations

import uuid
from pathlib import Path

from lsm.core.facts import Fact
from lsm.core.identity import IdentityProfile
from lsm.core.memory import MemoryEntry
from lsm.core.verifier import RuleBasedVerifier, Verdict, VerifierContext, Violation
from lsm.core.verifier_features import feature_dict
from lsm.core.verifier_log import VerifierContextRecord, VerifierLog
from lsm.verifier_data.context_text import render_context_text
from lsm.verifier_data.correction_miner import mine_correction_examples
from lsm.verifier_data.perturb import perturb_from_events
from lsm.verifier_data.schema import VerifierDatasetRow, load_rows, write_rows


def export_verifier_dataset(
    home: Path,
    *,
    out_path: Path,
    gold_path: Path | None = None,
    include_mined: bool = True,
    include_perturb: bool = True,
    limit: int = 2000,
) -> list[VerifierDatasetRow]:
    """Export unified JSONL for verifier training/eval."""
    rows: list[VerifierDatasetRow] = []
    telemetry = _load_telemetry_rows(home, limit=limit)
    rows.extend(telemetry)
    if include_mined:
        rows.extend(_mined_rows(home, limit=max(50, limit // 4)))
    if include_perturb:
        rows.extend(_perturbed_rows(telemetry, limit=max(100, limit // 2)))
    if gold_path is not None and gold_path.exists():
        rows.extend(load_rows(gold_path))
    deduped = _dedupe(rows)
    write_rows(out_path, deduped)
    return deduped


def _load_telemetry_rows(home: Path, *, limit: int) -> list[VerifierDatasetRow]:
    path = home / "verifier_log.sqlite"
    if not path.exists():
        return []
    vlog = VerifierLog(path)
    try:
        events = vlog.recent(limit=max(1, limit))
        out: list[VerifierDatasetRow] = []
        for ev in events:
            ctx_record = vlog.context_for(ev.id)
            ctx = context_from_snapshot(ctx_record)
            verdict = _verdict_from_event(ev)
            out.append(
                VerifierDatasetRow(
                    event_id=ev.id,
                    source=ev.source,
                    candidate=ev.candidate,
                    ctx_text=render_context_text(ctx_record),
                    ctx_features=feature_dict(
                        ev.candidate, ctx, source=ev.source, rule_verdict=verdict
                    ),
                    rule_energy=float(ev.energy),
                    rule_decision=ev.decision,  # type: ignore[arg-type]
                    rule_violations=[str(v.get("kind", "")) for v in ev.violations],
                    label_source="rule_distill",
                    label=ev.decision,  # type: ignore[arg-type]
                    reprompt_used=ev.reprompt_used,
                    meta=_meta_with_context(ctx_record),
                )
            )
        return out
    finally:
        vlog.close()


def _mined_rows(home: Path, *, limit: int) -> list[VerifierDatasetRow]:
    mined = mine_correction_examples(home, limit=limit)
    rule = RuleBasedVerifier()
    out: list[VerifierDatasetRow] = []
    for ex in mined:
        ctx = context_from_snapshot(ex.context)
        verdict = rule.score(ex.candidate, ctx)
        out.append(
            VerifierDatasetRow(
                event_id=ex.event_id,
                source=ex.source,
                candidate=ex.candidate,
                ctx_text=render_context_text(ex.context),
                ctx_features=feature_dict(
                    ex.candidate,
                    ctx,
                    source=ex.source,
                    rule_verdict=verdict,
                ),
                rule_energy=float(verdict.energy),
                rule_decision=verdict.decision,
                rule_violations=[v.kind for v in verdict.violations],
                label_source=ex.label_source,
                label="reject",
                reprompt_used=False,
                meta={**ex.meta, **_meta_with_context(ex.context)},
            )
        )
    return out


def _perturbed_rows(telemetry: list[VerifierDatasetRow], *, limit: int) -> list[VerifierDatasetRow]:
    # We only need minimal event/context wrappers for perturb generator.
    from lsm.core.verifier_log import VerifierEvent

    wrapped: list[tuple[VerifierEvent, VerifierContextRecord | None]] = []
    for row in telemetry:
        wrapped.append(
            (
                VerifierEvent(
                    id=row.event_id,
                    ts=0.0,
                    source=row.source,
                    candidate=row.candidate,
                    energy=row.rule_energy,
                    decision=row.rule_decision,
                    violations=tuple({"kind": kind} for kind in row.rule_violations),
                    reprompt_used=row.reprompt_used,
                ),
                None,
            )
        )
    synthetic = perturb_from_events(wrapped, limit=limit)
    rule = RuleBasedVerifier()
    out: list[VerifierDatasetRow] = []
    for ex in synthetic:
        ctx = context_from_snapshot(ex.context)
        verdict = rule.score(ex.candidate, ctx)
        out.append(
            VerifierDatasetRow(
                event_id=ex.event_id,
                source=ex.source,
                candidate=ex.candidate,
                ctx_text=render_context_text(ex.context),
                ctx_features=feature_dict(
                    ex.candidate, ctx, source=ex.source, rule_verdict=verdict
                ),
                rule_energy=float(verdict.energy),
                rule_decision=verdict.decision,
                rule_violations=[v.kind for v in verdict.violations],
                label_source=ex.label_source,
                label="reject",
                reprompt_used=False,
                meta={**ex.meta, **_meta_with_context(ex.context)},
            )
        )
    return out


def context_from_snapshot(record: VerifierContextRecord | None) -> VerifierContext:
    facts = _SnapshotFactStore.from_record(record)
    identity: IdentityProfile | None = None
    if record is not None and record.identity_snapshot is not None:
        identity = IdentityProfile(
            name=str(record.identity_snapshot.get("name", "")),
            description=str(record.identity_snapshot.get("description", "")),
        )
    corrections: tuple[MemoryEntry, ...] = ()
    turns: tuple[str, ...] = ()
    if record is not None:
        corrections = tuple(
            MemoryEntry(
                id=str(c.get("id", uuid.uuid4().hex)),
                text=str(c.get("text", "")),
                source=str(c.get("source", "conversation")),
                tags=[str(t) for t in c.get("tags", [])],
            )
            for c in record.recent_corrections
        )
        turns = tuple(record.recent_user_turns)
    return VerifierContext(
        facts=facts,  # type: ignore[arg-type]
        identity=identity,
        recent_corrections=corrections,
        recent_user_turns=turns,
    )


def _verdict_from_event(event: object) -> Verdict:
    violations = tuple(
        Violation(
            kind=v.get("kind", "list_merge"),
            detail=v.get("detail", ""),
            weight=float(v.get("weight", 0.0)),
            evidence=v.get("evidence", ""),
        )
        for v in getattr(event, "violations", ())
    )
    return Verdict(
        energy=float(getattr(event, "energy", 0.0)),
        decision=str(getattr(event, "decision", "accept")),  # type: ignore[arg-type]
        violations=violations,
        fix_hints=tuple(v.fix_hint() for v in violations),
    )


def _dedupe(rows: list[VerifierDatasetRow]) -> list[VerifierDatasetRow]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[VerifierDatasetRow] = []
    for row in rows:
        key = (
            row.candidate.strip().lower(),
            row.ctx_text.strip().lower(),
            row.label_source,
            row.label,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _meta_with_context(ctx: VerifierContextRecord | None) -> dict[str, object]:
    if ctx is None:
        return {"ctx_hash": ""}
    return {
        "ctx_hash": ctx.ctx_hash,
        "facts_snapshot": list(ctx.facts_snapshot),
        "identity_snapshot": ctx.identity_snapshot or {},
        "recent_corrections": list(ctx.recent_corrections),
        "recent_user_turns": list(ctx.recent_user_turns),
    }


class _SnapshotFactStore:
    """Tiny adapter exposing ``all_facts()`` for :class:`VerifierContext`."""

    def __init__(self, facts: list[Fact]) -> None:
        self._facts = facts

    def all_facts(self) -> list[Fact]:
        return list(self._facts)

    @classmethod
    def from_record(cls, record: VerifierContextRecord | None) -> _SnapshotFactStore:
        facts: list[Fact] = []
        if record is not None:
            for row in record.facts_snapshot:
                facts.append(
                    Fact(
                        id=str(row.get("id", uuid.uuid4().hex)),
                        owner=str(row.get("owner", "user")),  # type: ignore[arg-type]
                        category=str(row.get("category", "user.preferences")),
                        subject=str(row.get("subject", "")) or None,
                        value=str(row.get("value", "")),
                    )
                )
        return cls(facts)
