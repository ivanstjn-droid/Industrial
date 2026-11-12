"""Small download-and-cache helper for public datasets."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path


def data_dir() -> Path:
    path = Path(os.environ.get("ANLAGENWACHT_DATA", "data/raw"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(url: str, dest: Path) -> Path | None:
    """Download ``url`` to ``dest`` unless cached. Returns None on HTTP 404."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    tmp.replace(dest)
    return dest
