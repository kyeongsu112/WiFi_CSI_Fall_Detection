from __future__ import annotations

import re
from pathlib import Path
import csv

# Matches: WiFall/ID{n}/{action}/{filename}.csv
# Groups:  (subject, action)
_ZIP_ENTRY_RE = re.compile(r"WiFall/(ID\d+)/(\w+)/[^/]+\.csv$")


def parse_zip_entry(name: str) -> tuple[str, str] | None:
    """Parse a WiFall zip entry path into (subject, action_normalized).

    Action is lowercased so 'Jump' and 'jump' both become 'jump'.
    Returns None for directory entries, non-CSV files, or unexpected paths.

    Example:
        >>> parse_zip_entry("WiFall/ID0/fall/2023-01-01.csv")
        ('ID0', 'fall')
        >>> parse_zip_entry("WiFall/ID3/Jump/xxx.csv")
        ('ID3', 'jump')
        >>> parse_zip_entry("WiFall/ID0/fall/")
        None
    """
    m = _ZIP_ENTRY_RE.match(name)
    if m is None:
        return None
    return m.group(1), m.group(2).lower()


def binary_label(raw_label: str) -> str:
    return "fall" if raw_label.lower() == "fall" else "non_fall"


def write_manifest(rows: list[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
