"""Structured facts with owner + category (Phase F.1, hardened in F.1.1).

Ground-truth layer beside LanceDB raw turns. Facts are extracted from user
messages via conservative heuristics and injected into the summarizer prompt.

F.1.1 hardening:

- Segment-aware extraction for multi-answer correction turns
  (``the answer to one is ...``, ``1.``, ``2)``, ``(3)``, bullets).
- Turn classifiers: ``_text_is_identity_quiz``, ``_text_has_paired_terms``.
- All surface-form canonicalization (genres, restaurants, paired chips)
  routed through :class:`~lsm.core.normalize.TermNormalizer`. No
  hardcoded alias tables in the extractor.
- Trait / name guards on correction and identity-quiz turns.
- ``MAX_FACT_VALUE_LEN`` cap on preference values.
- Optional ``source_segment`` on each :class:`Fact` for downstream verifier
  consumption.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lsm.core.fact_categories import FactCategoryRegistry

from lsm.core.identity import IdentityProfile
from lsm.core.normalize import NullTermNormalizer, TermNormalizer, clean_term

Owner = Literal["user", "assistant", "shared"]
FactSource = Literal["user", "inferred"]

CAT_ASSISTANT_IDENTITY = "assistant.identity"
CAT_USER_PREFERENCES = "user.preferences"
CAT_USER_DISLIKES = "user.dislikes"
CAT_USER_PEOPLE = "user.people"
CAT_USER_MEDIA = "user.media"
CAT_USER_TASKS = "user.tasks"
CAT_USER_TRAITS = "user.traits"
CAT_USER_OPEN_QUESTIONS = "user.open_questions"
CAT_ASSISTANT_NOTES = "assistant.notes"

USER_CATEGORIES = frozenset(
    {
        CAT_USER_PREFERENCES,
        CAT_USER_DISLIKES,
        CAT_USER_PEOPLE,
        CAT_USER_MEDIA,
        CAT_USER_TASKS,
        CAT_USER_TRAITS,
        CAT_USER_OPEN_QUESTIONS,
    }
)

# Hard cap on preference / dislike / trait values (anti-paragraph-blob).
MAX_FACT_VALUE_LEN = 80


# --------------------------------------------------------------------------- #
# Surface-form patterns
# --------------------------------------------------------------------------- #

_LIKE = re.compile(
    r"\b(?:like|love|enjoy|prefer)\s+([^.\n?!]{2,120})",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(
    r"\b(?:i(?:'m| am)?|my|mine|we(?:'re| are)?|our|us)\b",
    re.IGNORECASE,
)
_YOU_BEFORE_LIKE = re.compile(
    r"\b(?:you|your|yours)\s+"
    r"(?:(?:would|could|should|might|may)\s+)?"
    r"(?:like|love|enjoy|prefer)\b",
    re.IGNORECASE,
)
_FUNCTION_WORD_STARTS = frozenset({"for", "to", "of", "about", "with"})
_PREF_VALUE_STOP = frozenset(
    {
        "for",
        "to",
        "of",
        "about",
        "with",
        "from",
        "the",
        "me",
        "you",
        "your",
        "my",
        "our",
        "and",
        "or",
        "but",
        "that",
        "this",
        "make",
        "have",
        "been",
        "being",
        "would",
        "could",
        "should",
        "will",
        "can",
        "did",
        "does",
        "do",
        "are",
        "was",
        "were",
        "is",
        "am",
        "be",
    }
)
_DISLIKE = re.compile(
    r"\b(?:i\s+)?(?:dislike|hate|avoid)\s+([^.\n?!]{2,120})",
    re.IGNORECASE,
)
_NAME = re.compile(
    r"\b(?:my\s+name\s+is|call\s+me|i(?:'m| am)\s+called)\s+([A-Za-z][A-Za-z\s\-']{0,40})",
    re.IGNORECASE,
)
_TRAIT = re.compile(
    r"\b(?:i(?:'m| am)\s+(?:a|an|the)?\s*)([a-z][a-z\s\-']{2,60})",
    re.IGNORECASE,
)
_TASK = re.compile(
    r"\b(?:help\s+me|remind\s+me|i\s+need\s+(?:you\s+)?to|can\s+you\s+help)\s+([^.\n?!]{3,120})",
    re.IGNORECASE,
)
_OPEN_QUESTION = re.compile(
    r"\b(?:i\s+wonder|what\s+do\s+you\s+think\s+about|"
    r"can\s+you\s+remind\s+me\s+to\s+think\s+about)\b",
    re.IGNORECASE,
)

# Media: positive "<Title> is/was (a|an) <genre>" + anaphoric "it is (a|an) X"
# and "actually (a|an) X" — anaphoric forms pair with the most recent
# capitalized title in the segment. Negation handled separately.
_MEDIA_TITLE_GENRE = re.compile(
    r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\s\-']{1,40}?)\s+"
    r"(?:is|was)\s+(?:a|an)\s+([A-Za-z][A-Za-z0-9\s\-']{2,40})",
)
_MEDIA_IT_IS = re.compile(
    r"\bit(?:'?s|\s+is|\s+was)\s+(?:actually\s+)?(?:a|an)\s+" r"([A-Za-z][A-Za-z0-9\s\-']{2,40})",
    re.IGNORECASE,
)
_MEDIA_ACTUALLY_GENRE = re.compile(
    r"\bactually\s+(?:a|an)\s+([A-Za-z][A-Za-z0-9\s\-']{2,40})",
    re.IGNORECASE,
)
_MEDIA_NEGATED = re.compile(
    r"\bnot\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z0-9\s\-']{2,40})",
    re.IGNORECASE,
)
# Capitalized title: one or more capitalized words joined by spaces.
# Intentionally narrow — must NOT span verbs / lowercase words, otherwise
# a title like "Crimson Desert" gets swallowed into a phrase "Crimson
# Desert is not a strategy game".
_TITLE_NP = re.compile(r"\b([A-Z][A-Za-z0-9\-']+(?:\s+[A-Z][A-Za-z0-9\-']+){0,5})\b")

_QUOTED = re.compile(r'["\']([^"\']{2,80})["\']')

_CORRECTION_PATTERNS = re.compile(
    r"\b("
    r"actually"
    r"|not\s+(?:a|an|the)?\s*"
    r"|i\s+meant"
    r"|correction"
    r"|to\s+clarify"
    r"|that'?s\s+wrong"
    r"|isn'?t"
    r"|please\s+make\s+it"
    r"|no,\s*please"
    r")\b",
    re.IGNORECASE,
)
_ADDITIVE_PATTERNS = re.compile(
    r"\b(" r"both\b" r"|as\s+well\s+as" r"|also\s+like" r"|and\s+\w+\s+and\s+" r")\b",
    re.IGNORECASE,
)
_IDENTITY_QUIZ_PATTERNS = re.compile(
    r"\b("
    r"how\s+do\s+you\s+know"
    r"|confirm\s+(?:your|my)\s+identity"
    r"|questions\s+you\s+can\s+ask"
    r"|am\s+i\s+actually"
    r"|who\s+am\s+i"
    r"|prove\s+(?:your|my)\s+identity"
    r")\b",
    re.IGNORECASE,
)

# Multi-answer segmenters, tried in order. The leading-bullet patterns
# work only at line starts.
_SEG_ANSWER_TO = re.compile(
    r"\bthe\s+answer\s+to\s+(?:one|two|three|four|five|first|second|third|fourth|fifth|\d+)"
    r"\s+(?:is|=)\s*",
    re.IGNORECASE,
)
_SEG_ANSWER_COLON = re.compile(r"\banswer\s+to\s+\d+\s*[:\-]\s*", re.IGNORECASE)
_SEG_LINE_LEAD = re.compile(r"(?m)^\s*(?:\(\d+\)|\d+[.)]|[-*\u2022])\s+")


# --------------------------------------------------------------------------- #
# Public predicates
# --------------------------------------------------------------------------- #


def _text_is_correction(text: str) -> bool:
    if not text:
        return False
    return _CORRECTION_PATTERNS.search(text) is not None


def _text_is_additive(text: str) -> bool:
    if not text:
        return False
    return bool(
        _ADDITIVE_PATTERNS.search(text)
        or re.search(r"\b\w+\s+and\s+\w+\s+restaurants?", text, re.IGNORECASE)
    )


def _text_is_identity_quiz(text: str) -> bool:
    if not text:
        return False
    return _IDENTITY_QUIZ_PATTERNS.search(text) is not None


def _text_has_paired_terms(text: str, terms: list[str]) -> bool:
    """At least two distinct ``terms`` co-occur with an ``and``-style connector.

    ``terms`` is the *raw alias surface forms* the caller cares about
    (e.g. ``["yakiniku", "korean bbq", "kbbq"]``). All comparisons are
    case-insensitive on cleaned forms.
    """
    if not text or len(terms) < 2:
        return False
    low = text.lower()
    hits = {t for t in terms if t.lower() in low}
    if len(hits) < 2:
        return False
    # Some connector between two of the hits.
    return bool(re.search(r"\b(?:and|both|as\s+well\s+as|,)\b", low))


# --------------------------------------------------------------------------- #
# Segment parsing
# --------------------------------------------------------------------------- #


def split_into_segments(text: str) -> list[str]:
    """Split a multi-answer correction turn into per-answer segments.

    Strategy: try each segmenter in order; first one with two or more
    matches wins. Fallback: return ``[text]`` unchanged so existing
    single-sentence callers keep working.
    """
    if not text or not text.strip():
        return []

    for pattern in (_SEG_ANSWER_TO, _SEG_ANSWER_COLON):
        parts = pattern.split(text)
        chunks = [p.strip() for p in parts if p and p.strip()]
        if len(chunks) >= 2:
            return chunks

    line_leads = list(_SEG_LINE_LEAD.finditer(text))
    if len(line_leads) >= 2:
        out: list[str] = []
        for i, m in enumerate(line_leads):
            start = m.end()
            end = line_leads[i + 1].start() if i + 1 < len(line_leads) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                out.append(chunk)
        if len(out) >= 2:
            return out

    return [text.strip()]


# --------------------------------------------------------------------------- #
# Fact model + store
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Fact:
    """A single owned, categorized fact."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    owner: Owner = "user"
    category: str = CAT_USER_PREFERENCES
    subject: str | None = None
    value: str = ""
    source: FactSource = "user"
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # F.1.1: raw segment text the fact was extracted from (None for
    # programmatic / sync paths). Consumed by the future Verifier.
    source_segment: str | None = None

    def display_line(self) -> str:
        subj = f"{self.subject}: " if self.subject else ""
        return f"- [{self.category}] {subj}{self.value}"


class FactStore:
    """Filesystem-backed store at ``$LSM_HOME/facts.json``."""

    FILENAME = "facts.json"

    def __init__(
        self,
        home: Path,
        *,
        category_registry: FactCategoryRegistry | None = None,
    ) -> None:
        from lsm.core.fact_categories import FactCategoryRegistry as _Registry

        self.home = home
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / self.FILENAME
        self._category_registry = category_registry or _Registry(home)
        self._facts: dict[str, Fact] = {}
        self.load()

    def load(self) -> FactStore:
        if not self.path.exists():
            self._facts = {}
            self.save()
            return self
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._facts = {}
        for row in data.get("facts", []):
            f = Fact(
                id=str(row["id"]),
                owner=row["owner"],
                category=str(row["category"]),
                subject=row.get("subject"),
                value=str(row["value"]),
                source=row.get("source", "user"),
                tags=list(row.get("tags") or []),
                created_at=float(row.get("created_at", 0.0)),
                updated_at=float(row.get("updated_at", 0.0)),
                source_segment=row.get("source_segment"),
            )
            self._facts[f.id] = f
        return self

    def save(self) -> None:
        payload = {"facts": [asdict(f) for f in self._facts.values()]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Windows: os.replace transiently fails with PermissionError when
        # AV / indexer briefly holds the target. Tiny retry loop is cheap
        # and avoids spurious test flakes.
        for attempt in range(5):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def upsert(self, fact: Fact) -> Fact:
        if not self._category_registry.is_allowed(fact.category):
            raise ValueError(
                f"unknown category {fact.category!r}; register with "
                f"'lsm fact category add <slug>'"
            )
        now = time.time()
        if fact.id in self._facts:
            existing = self._facts[fact.id]
            fact.created_at = existing.created_at
        else:
            fact.created_at = now
        fact.updated_at = now
        self._facts[fact.id] = fact
        self.save()
        return fact

    def query(
        self,
        *,
        owner: str | None = None,
        category: str | None = None,
    ) -> list[Fact]:
        out: list[Fact] = []
        for f in self._facts.values():
            if owner is not None and f.owner != owner:
                continue
            if category is not None and f.category != category:
                continue
            out.append(f)
        out.sort(key=lambda x: x.updated_at, reverse=True)
        return out

    def replace_by_subject(
        self,
        *,
        owner: Owner,
        category: str,
        subject: str,
        new_value: str,
        tags: list[str] | None = None,
        source_segment: str | None = None,
    ) -> Fact:
        for f in self._facts.values():
            if (
                f.owner == owner
                and f.category == category
                and (f.subject or "").lower() == subject.lower()
            ):
                f.value = new_value
                if tags:
                    f.tags = list(set(f.tags) | set(tags))
                if source_segment is not None:
                    f.source_segment = source_segment
                return self.upsert(f)
        fact = Fact(
            owner=owner,
            category=category,
            subject=subject,
            value=new_value,
            tags=list(tags or []),
            source_segment=source_segment,
        )
        return self.upsert(fact)

    def add_preference_value(
        self,
        *,
        subject: str | None,
        value: str,
        tags: list[str] | None = None,
        source_segment: str | None = None,
    ) -> Fact:
        """Additive upsert: add a preference without removing siblings."""
        val_norm = value.strip().lower()
        for f in self.query(owner="user", category=CAT_USER_PREFERENCES):
            if f.subject == subject and val_norm in f.value.lower():
                return f
            if f.subject == subject and f.value.strip().lower() == val_norm:
                return f
        fact = Fact(
            owner="user",
            category=CAT_USER_PREFERENCES,
            subject=subject,
            value=value.strip(),
            tags=list(tags or []),
            source_segment=source_segment,
        )
        return self.upsert(fact)

    def delete(self, fact_id: str) -> bool:
        if fact_id not in self._facts:
            return False
        del self._facts[fact_id]
        self.save()
        return True

    def delete_where(
        self,
        *,
        owner: str | None = None,
        category: str | None = None,
        subject: str | None = None,
        value_contains: str | None = None,
    ) -> int:
        """Bulk delete by predicate. Returns number of rows removed.

        All provided filters are ANDed. With no filters, this is a no-op
        and returns 0 — callers must explicitly pass ``--all`` semantics
        upstream if they want to wipe everything.
        """
        if owner is None and category is None and subject is None and value_contains is None:
            return 0
        needle = value_contains.lower() if value_contains else None
        victims = [
            fid
            for fid, f in self._facts.items()
            if (owner is None or f.owner == owner)
            and (category is None or f.category == category)
            and (subject is None or (f.subject or "").lower() == subject.lower())
            and (needle is None or needle in f.value.lower())
        ]
        for fid in victims:
            del self._facts[fid]
        if victims:
            self.save()
        return len(victims)

    def clear_all(self, *, preserve_assistant_identity: bool = True) -> int:
        """Drop every fact (optionally keeping ``assistant.identity``)."""
        victims = [
            fid
            for fid, f in self._facts.items()
            if not (preserve_assistant_identity and f.category == CAT_ASSISTANT_IDENTITY)
        ]
        for fid in victims:
            del self._facts[fid]
        if victims:
            self.save()
        return len(victims)

    def all_facts(self) -> list[Fact]:
        return sorted(self._facts.values(), key=lambda f: f.updated_at, reverse=True)

    def sync_assistant_identity(self, profile: IdentityProfile) -> Fact:
        return self.replace_by_subject(
            owner="assistant",
            category=CAT_ASSISTANT_IDENTITY,
            subject="self",
            new_value=f"{profile.name} - {profile.description}",
        )


# --------------------------------------------------------------------------- #
# Extraction helpers
# --------------------------------------------------------------------------- #


def _new_fact(
    category: str,
    value: str,
    *,
    subject: str | None = None,
    tags: list[str] | None = None,
    source_segment: str | None = None,
) -> Fact:
    return Fact(
        owner="user",
        category=category,
        subject=subject,
        value=value.strip(),
        tags=list(tags or []),
        source_segment=source_segment,
    )


def _is_low_quality_trait(s: str) -> bool:
    """Reject obviously-broken trait captures like 'ctually Kalen'."""
    if not s:
        return True
    s = s.strip()
    if len(s) <= 2:
        return True
    if re.match(r"^(actually|ctually|ust|nd|or|so|Kalen)\b", s, re.IGNORECASE):
        return True
    # Single-token traits of bare length are almost always false positives
    # ("Kalen", "tired", "cool"). Real traits tend to be multi-word.
    return " " not in s and len(s) <= 12


def _truncate_value(value: str, *, limit: int = MAX_FACT_VALUE_LEN) -> str | None:
    """Apply length policy. Returns trimmed string, or ``None`` to reject."""
    v = value.strip()
    if not v:
        return None
    if len(v) > limit:
        # Hard reject paragraph-blob captures; truncation hides the bug.
        return None
    return v


def _like_match_is_first_person(segment: str, match: re.Match[str]) -> bool:
    """Reject second-person ``you like …``; require a first-person anchor."""
    before = segment[: match.start()]
    tail = before[-80:] if len(before) > 80 else before
    if _YOU_BEFORE_LIKE.search(tail):
        return False
    window = before[-40:] + segment[match.start() : min(len(segment), match.end() + 24)]
    if _FIRST_PERSON.search(window):
        return True
    prefix = before.strip()
    if not prefix or re.search(r"[.?!]\s*$", prefix):
        return _FIRST_PERSON.search(segment[match.start() : match.end() + 24]) is not None
    return False


def _reject_function_word_preference(value: str) -> bool:
    """Drop captures like ``for me to make to you`` (no concrete noun)."""
    words = re.findall(r"[a-z']{2,}", value.strip().lower())
    if not words:
        return True
    if words[0] not in _FUNCTION_WORD_STARTS:
        return False
    concrete = [w for w in words if len(w) >= 4 and w not in _PREF_VALUE_STOP]
    return len(concrete) == 0


def _derive_question_subject(text: str) -> str | None:
    """Best-effort topic chip for open-question facts."""
    if not text:
        return None
    for pat in (
        r"what\s+do\s+you\s+think\s+about\s+([^.?!,\n]{2,50})",
        r"i\s+wonder\s+(?:about\s+)?([^.?!,\n]{2,50})",
        r"remind\s+me\s+to\s+think\s+about\s+([^.?!,\n]{2,50})",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        subj = clean_term(m.group(1))
        if subj:
            return subj[:40]
    return None


def _extract_paired_preferences(
    segment: str,
    *,
    scope: str,
    subject: str,
    aliases: list[str],
    normalizer: TermNormalizer,
    tags: list[str],
) -> list[Fact]:
    """Emit one preference fact per paired-term hit in ``segment``.

    ``aliases`` is the list of raw surface forms the caller knows about
    (e.g. ``["yakiniku", "korean bbq", "kbbq"]``). Surface forms not in
    ``aliases`` will not be paired here — that is intentional; pairing
    is for known multi-word terms whose split is ambiguous under naive
    tokenization. Single-term fallback runs elsewhere.
    """
    if not _text_has_paired_terms(segment, aliases):
        return []
    low = segment.lower()
    hits: list[str] = []
    for raw in aliases:
        if raw.lower() in low and raw.lower() not in [h.lower() for h in hits]:
            hits.append(raw)
    out: list[Fact] = []
    for raw in hits:
        canonical = normalizer.canonicalize(scope, raw)
        if not canonical:
            continue
        out.append(
            _new_fact(
                CAT_USER_PREFERENCES,
                canonical,
                subject=subject,
                tags=[*tags, "additive"],
                source_segment=segment,
            )
        )
    return out


def _extract_media_in_segment(
    segment: str,
    *,
    normalizer: TermNormalizer,
    tags: list[str],
    is_correction: bool,
) -> list[Fact]:
    """Extract media (title -> canonical genre) facts from a single segment.

    Strategy:

    1. Collect ``title -> raw_genre`` candidates from positive patterns.
    2. Collect "negated" genres (``not a strategy game``) and exclude
       them from the candidates set for any title in the segment.
    3. If a correction cue puts a positive genre *after* a negation, that
       positive wins (handled implicitly: the negation drops its term,
       the positive carries through).
    4. Canonicalize the surviving genre per title through the normalizer.
    """
    if not segment:
        return []

    negated_genres: set[str] = set()
    for m in _MEDIA_NEGATED.finditer(segment):
        g = clean_term(m.group(1))
        if g:
            negated_genres.add(g)

    candidates: list[tuple[str, str]] = []
    for m in _MEDIA_TITLE_GENRE.finditer(segment):
        title = m.group(1).strip()
        genre = m.group(2).strip()
        if title and genre and clean_term(genre) not in negated_genres:
            candidates.append((title, genre))

    # Anaphoric resolution: "Crimson Desert ... actually an ARPG" or
    # "... it is an action RPG" — the title appears earlier in the segment,
    # the genre is announced via "actually a/an X" or "it is a/an X". Pair
    # each anaphoric genre with the nearest capitalized title that *precedes*
    # the genre's position in the segment.
    anaphoric_hits: list[tuple[int, str]] = []
    for m in _MEDIA_ACTUALLY_GENRE.finditer(segment):
        anaphoric_hits.append((m.start(), m.group(1).strip()))
    for m in _MEDIA_IT_IS.finditer(segment):
        anaphoric_hits.append((m.start(), m.group(1).strip()))
    title_positions = [(tm.start(), tm.group(1).strip()) for tm in _TITLE_NP.finditer(segment)]
    for pos, genre in anaphoric_hits:
        if not genre or clean_term(genre) in negated_genres:
            continue
        cleaned_g = clean_term(genre)
        prior_titles = [t for (tp, t) in title_positions if tp < pos and clean_term(t) != cleaned_g]
        if prior_titles:
            candidates.append((prior_titles[-1], genre))

    # Dedupe by title: prefer the LAST candidate per title (correction
    # cues typically come last, e.g. "X is a strategy, actually an RPG").
    by_title: dict[str, str] = {}
    for title, genre in candidates:
        by_title[title] = genre

    out: list[Fact] = []
    for title, raw_genre in by_title.items():
        canonical = normalizer.canonicalize("user.media:genre", raw_genre)
        if not canonical:
            continue
        out.append(
            _new_fact(
                CAT_USER_MEDIA,
                canonical,
                subject=title,
                tags=[*tags, "correction"] if is_correction else tags,
                source_segment=segment,
            )
        )
    return out


# Surface forms the F.1.1 plan calls out by name. The normalizer collapses
# them to canonicals; the extractor only needs to know *they exist as
# surface tokens* so paired-term detection can fire on them.
_RESTAURANT_ALIASES = ["yakiniku", "korean bbq", "kbbq"]


# --------------------------------------------------------------------------- #
# Top-level extraction
# --------------------------------------------------------------------------- #


def extract_facts_from_user_turn(
    text: str,
    *,
    normalizer: TermNormalizer | None = None,
) -> list[Fact]:
    """Conservative rule-based extraction from a user message.

    F.1.1 routes all canonicalization through the supplied ``normalizer``.
    With ``normalizer=None`` (default), :class:`NullTermNormalizer` is
    used so behavior degrades gracefully to lowercase / whitespace
    canonicalization.
    """
    if not text or not text.strip():
        return []
    norm: TermNormalizer = normalizer if normalizer is not None else NullTermNormalizer()

    is_correction = _text_is_correction(text)
    is_quiz = _text_is_identity_quiz(text)
    is_additive = _text_is_additive(text)

    base_tags: list[str] = []
    if is_correction:
        base_tags.append("correction")
    if is_additive:
        base_tags.append("additive")

    segments = split_into_segments(text) if is_correction else [text]

    facts: list[Fact] = []

    # --- Name (skip on quiz turns; allow on plain `my name is` always) ---
    if not is_quiz:
        for m in _NAME.finditer(text):
            name = m.group(1).strip()
            if name:
                facts.append(_new_fact(CAT_USER_PEOPLE, name, subject="self", tags=base_tags))

    # --- Tasks (whole-text; tasks rarely care about segments) ---
    if not is_quiz:
        for m in _TASK.finditer(text):
            facts.append(_new_fact(CAT_USER_TASKS, m.group(1).strip(), tags=base_tags))

    # --- Open questions ---
    # Only triggered by explicit cue phrases ("I wonder", "what do you think
    # about", "can you remind me to think about") or the literal "(open
    # question)" tag. We deliberately do NOT treat every '?' as an open
    # question -- routine user questions ("what's the weather like?") and
    # identity quizzes ("how do you know i am Kalen?") would otherwise
    # pollute the store. Note: "remind me to think about ..." is also
    # captured as a user.task above; that dual capture is intentional --
    # tasks track action, open questions track reflection.
    if not is_quiz:
        cue_match = _OPEN_QUESTION.search(text)
        open_tag = "(open question)" in text.lower()
        if cue_match or open_tag:
            subject = _derive_question_subject(text)
            if cue_match:
                cue_start = cue_match.start()
                tail = text[cue_start:].split("\n", 1)[0]
                raw_value = tail.strip().rstrip(".?!")
            else:
                raw_value = text.strip()
            val = _truncate_value(raw_value)
            if val is not None and (subject is not None or open_tag):
                facts.append(
                    _new_fact(
                        CAT_USER_OPEN_QUESTIONS,
                        val,
                        subject=subject,
                        tags=base_tags,
                    )
                )

    # --- Traits (skip entirely on correction or quiz turns) ---
    if not is_correction and not is_quiz:
        for m in _TRAIT.finditer(text):
            trait = m.group(1).strip()
            if not trait or _is_low_quality_trait(trait):
                continue
            if _TASK.search(trait):
                continue
            facts.append(_new_fact(CAT_USER_TRAITS, trait, tags=base_tags))

    # --- Dislikes (whole-text; conservative) ---
    for m in _DISLIKE.finditer(text):
        val = _truncate_value(m.group(1))
        if val:
            facts.append(_new_fact(CAT_USER_DISLIKES, val, tags=base_tags))

    # --- Media (per-segment for correction turns; whole-text otherwise) ---
    for seg in segments:
        facts.extend(
            _extract_media_in_segment(
                seg,
                normalizer=norm,
                tags=base_tags,
                is_correction=is_correction,
            )
        )

    # --- Paired-term preferences (restaurants today, generic infrastructure) ---
    segments_with_paired_restaurants: set[str] = set()
    for seg in segments:
        paired = _extract_paired_preferences(
            seg,
            scope="user.preferences:restaurants",
            subject="restaurants",
            aliases=_RESTAURANT_ALIASES,
            normalizer=norm,
            tags=base_tags,
        )
        facts.extend(paired)
        if len(paired) >= 2:
            segments_with_paired_restaurants.add(seg)

    # --- Single-term preferences (per-segment; skip when paired path won) ---
    for seg in segments:
        if seg in segments_with_paired_restaurants:
            continue
        for m in _LIKE.finditer(seg):
            if not _like_match_is_first_person(seg, m):
                continue
            val = _truncate_value(m.group(1))
            if val is None or _reject_function_word_preference(val):
                continue
            if is_correction:
                scope = "user.preferences:general"
                subj = None
            else:
                subj = "restaurants" if "restaurant" in val.lower() else None
                scope = (
                    "user.preferences:restaurants"
                    if subj == "restaurants"
                    else "user.preferences:general"
                )
            canonical = norm.canonicalize(scope, val)
            if not canonical:
                continue
            facts.append(
                _new_fact(
                    CAT_USER_PREFERENCES,
                    canonical,
                    subject=subj,
                    tags=base_tags,
                    source_segment=seg,
                )
            )

    return facts


def apply_extracted_facts(
    store: FactStore,
    text: str,
    *,
    normalizer: TermNormalizer | None = None,
) -> list[Fact]:
    """Extract and merge facts into the store with F.1.1 merge semantics.

    Rules:

    - ``user.media`` + subject: always ``replace_by_subject``.
    - ``user.preferences`` with the ``additive`` tag: always
      ``add_preference_value`` (even on correction turns — paired
      restaurants are the canonical case).
    - ``user.preferences`` on correction turn (non-additive): only if the
      value passes the length gate, ``replace_by_subject`` on
      ``subject or "general"``.
    - ``user.preferences`` non-correction: existing ``upsert``.
    - ``user.traits`` on correction or identity-quiz turn: SKIP entirely.
    """
    extracted = extract_facts_from_user_turn(text, normalizer=normalizer)
    stored: list[Fact] = []
    is_correction = _text_is_correction(text)
    is_quiz = _text_is_identity_quiz(text)

    for fact in extracted:
        if fact.category == CAT_USER_TRAITS and (is_correction or is_quiz):
            continue

        if fact.category == CAT_USER_MEDIA and fact.subject:
            stored.append(
                store.replace_by_subject(
                    owner="user",
                    category=CAT_USER_MEDIA,
                    subject=fact.subject,
                    new_value=fact.value,
                    tags=fact.tags,
                    source_segment=fact.source_segment,
                )
            )
            continue

        if fact.category == CAT_USER_PREFERENCES and "additive" in fact.tags:
            stored.append(
                store.add_preference_value(
                    subject=fact.subject,
                    value=fact.value,
                    tags=fact.tags,
                    source_segment=fact.source_segment,
                )
            )
            continue

        if fact.category == CAT_USER_PREFERENCES and is_correction:
            val = _truncate_value(fact.value)
            if val is None:
                continue
            subj = fact.subject or "general"
            stored.append(
                store.replace_by_subject(
                    owner="user",
                    category=CAT_USER_PREFERENCES,
                    subject=subj,
                    new_value=val,
                    tags=fact.tags,
                    source_segment=fact.source_segment,
                )
            )
            continue

        if fact.category in (CAT_USER_PREFERENCES, CAT_USER_DISLIKES, CAT_USER_TRAITS):
            val = _truncate_value(fact.value)
            if val is None:
                continue
            fact.value = val

        stored.append(store.upsert(fact))
    return stored


def format_structured_facts_block(facts: list[Fact]) -> str:
    """Prompt block for the summarizer."""
    if not facts:
        return ""
    lines = ["Structured facts (authoritative for user-owned slots):"]
    for f in facts:
        lines.append(f.display_line())
    return "\n".join(lines)
