"""Defences against prompt injection hidden in retrieved documents.

Retrieved text is untrusted: anyone can open an issue, and docs can be
edited. Three layers keep it from steering the model:

1. sanitize_hits (here): strips invisible Unicode and replaces any paragraph
   that reads like an instruction to the assistant before the LLM sees it.
2. The answer prompt wraps sources in <source> tags and tells the model to
   treat them as evidence, never as instructions (prompts.ANSWER_SYSTEM).
3. The grounding check in graph.py rejects answers without valid citations.
"""

from __future__ import annotations

import re
from dataclasses import replace

from contributor_copilot.bm25 import Hit

# Zero-width characters, bidi controls and Unicode tag characters can hide
# text from a human reviewer while the model still reads it.
HIDDEN_CHARS_RE = re.compile("[​-‏‪-‮⁠-⁤﻿\U000e0000-\U000e007f]")

INJECTION_PATTERNS = (
    r"\b(ignore|disregard|forget|override)\b.{0,40}?\b(previous|prior|above|earlier|preceding|all|any|system)\b"
    r".{0,20}?\b(instructions?|prompts?|rules|directions|messages)\b",
    r"\byou are now\b",
    r"\bnew instructions\s*:",
    r"\b(reveal|print|show|repeat|output)\b.{0,30}?\bsystem prompt\b",
)
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)
PARAGRAPH_SPLIT_RE = re.compile(r"(\n[ \t]*\n)")
REDACTED = "[removed by guardrail: text that looked like an instruction to the assistant]"


def looks_like_injection(text: str) -> bool:
    normalized = " ".join(HIDDEN_CHARS_RE.sub("", text).split())
    return INJECTION_RE.search(normalized) is not None


def sanitize(text: str) -> tuple[str, bool]:
    """Remove hidden characters and redact suspicious paragraphs.

    Matching runs per paragraph with whitespace collapsed, so an instruction
    wrapped across lines is still caught. Returns (clean_text, flagged).
    """
    text = HIDDEN_CHARS_RE.sub("", text)
    parts = PARAGRAPH_SPLIT_RE.split(text)  # keeps the blank-line separators
    flagged = False
    for index in range(0, len(parts), 2):
        if looks_like_injection(parts[index]):
            parts[index] = REDACTED
            flagged = True
    return "".join(parts), flagged


def sanitize_hits(hits: list[Hit]) -> tuple[list[Hit], list[str]]:
    """Sanitize each hit's text. Returns (clean_hits, locations_that_were_flagged)."""
    clean, flagged = [], []
    for hit in hits:
        text, was_flagged = sanitize(hit.chunk.text)
        if was_flagged:
            flagged.append(hit.chunk.location)
        if text != hit.chunk.text:
            hit = Hit(replace(hit.chunk, text=text), hit.score)
        clean.append(hit)
    return clean, flagged
