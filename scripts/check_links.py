#!/usr/bin/env python3
"""Check that relative links in markdown files resolve on disk.

Usage:
    python scripts/check_links.py [files...]

With no arguments, checks every tracked .md file. Skips http(s) links,
mailto:, and anchors-only links. Resolves relative targets against each
file's own directory. Exits 1 if any target is missing.
"""

import os
import re
import subprocess
import sys

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)")


def tracked_markdown_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    )
    return [f for f in out.stdout.splitlines() if f.strip()]


def check_file(path: str) -> list[str]:
    broken = []
    base = os.path.dirname(path) or "."
    text = open(path, encoding="utf-8").read()
    for target in LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = target.split("#")[0]
        if not target:
            continue
        resolved = os.path.normpath(os.path.join(base, target))
        if not os.path.exists(resolved):
            broken.append(f"{path}: broken link -> {target}")
    return broken


def main() -> int:
    files = sys.argv[1:] or tracked_markdown_files()
    broken = []
    for path in files:
        if path.endswith(".md") and os.path.exists(path):
            broken.extend(check_file(path))
    for line in broken:
        print(line)
    if broken:
        print(f"{len(broken)} broken link(s)")
        return 1
    print(f"links ok ({len(files)} file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
