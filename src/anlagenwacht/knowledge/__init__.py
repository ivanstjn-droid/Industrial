"""Tiny keyword search over the knowledge base (SOP / troubleshooting notes).

In a pilot this is replaced by the customer's own manuals, SOPs and shift logs.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

_DIR = Path(__file__).parent


def _sections() -> list[tuple[str, str]]:
    out = []
    for path in sorted(_DIR.glob("*.md")):
        for block in path.read_text(encoding="utf-8").split("\n## ")[1:]:
            title, _, body = block.partition("\n")
            out.append((f"{path.stem}: {title.strip()}", body.strip()))
    return out


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zäöüß0-9]+", text.lower()) if len(w) > 3]


def search(query: str | list[tuple[str, float]], k: int = 3) -> list[dict]:
    """Rank knowledge-base sections against a query.

    ``query`` is either plain text or ``[(text, weight), ...]`` (e.g. tag
    descriptions weighted by contribution). Rare words count more (IDF);
    prefix matching lets "temperatures" match "temperature".
    """
    parts = [(query, 1.0)] if isinstance(query, str) else query
    sections = _sections()
    docs = [set(_words(t + " " + b)) for t, b in sections]
    n = len(docs)

    def matches(w: str, words: set[str]) -> bool:
        return any(x.startswith(w[:6]) for x in words)

    scored = []
    for (title, body), words in zip(sections, docs):
        score = 0.0
        for text, weight in parts:
            for w in set(_words(text)):
                if matches(w, words):
                    df = sum(1 for d in docs if matches(w, d))
                    score += weight * math.log(1 + n / df)
        if score > 0:
            scored.append((score, title, body))
    scored.sort(key=lambda s: -s[0])
    return [{"title": t, "text": b} for _, t, b in scored[:k]]
