#!/usr/bin/env python3
"""Validate repository-relative Markdown links without using the network."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def markdown_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> int:
    missing: list[str] = []
    for document in markdown_paths():
        for raw_target in LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_part = unquote(target.split("#", 1)[0])
            candidate = (document.parent / path_part).resolve()
            if not candidate.exists():
                relative = document.relative_to(ROOT)
                missing.append(f"{relative}: {target}")
    if missing:
        print("missing local Markdown links:")
        for item in missing:
            print(f"- {item}")
        return 1
    print("local Markdown links passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
