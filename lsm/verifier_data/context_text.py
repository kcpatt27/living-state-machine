"""Render compact textual verifier context for encoder training."""

from __future__ import annotations

from lsm.core.verifier_log import VerifierContextRecord


def render_context_text(ctx: VerifierContextRecord | None) -> str:
    """Build a compact text block consumed by encoder-based verifier heads."""
    if ctx is None:
        return ""
    parts: list[str] = []
    if ctx.identity_snapshot:
        name = str(ctx.identity_snapshot.get("name", "")).strip()
        desc = str(ctx.identity_snapshot.get("description", "")).strip()
        if name or desc:
            parts.append(f"identity: {name} - {desc}".strip())
    if ctx.facts_snapshot:
        parts.append("facts:")
        for fact in ctx.facts_snapshot[:24]:
            subj = str(fact.get("subject", "")).strip()
            if subj:
                parts.append(
                    f"- [{fact.get('category', '')}] {subj}: {fact.get('value', '')}".strip()
                )
            else:
                parts.append(f"- [{fact.get('category', '')}] {fact.get('value', '')}".strip())
    if ctx.recent_corrections:
        parts.append("recent_corrections:")
        for corr in ctx.recent_corrections[:8]:
            parts.append(f"- {str(corr.get('text', '')).strip()}")
    if ctx.recent_user_turns:
        parts.append("recent_user_turns:")
        for turn in ctx.recent_user_turns[:8]:
            parts.append(f"- {turn}")
    return "\n".join(parts)
