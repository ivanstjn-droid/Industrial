"""Tiny keyword search over the knowledge base (SOP / troubleshooting notes).

In a pilot this is replaced by the customer's own manuals, SOPs and shift logs.
"""

from __future__ import annotations

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


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zäöüß0-9]+", text.lower()) if len(w) > 3}


def search(query: str, k: int = 3) -> list[dict]:
    q = _words(query)
    scored = []
    for title, body in _sections():
        words = _words(title + " " + body)
        # prefix match so "temperatures" also hits "temperature"
        hits = sum(1 for w in q if any(x.startswith(w[:6]) for x in words))
        if hits:
            scored.append((hits, title, body))
    scored.sort(key=lambda s: -s[0])
    return [{"title": t, "text": b} for _, t, b in scored[:k]]
