"""Relative links and images in the Markdown documentation resolve to files in the repository."""

from __future__ import annotations

import re

import pytest
from conftest import REPOSITORY_ROOT

DOCUMENTS = [
    "README.md",
    "data/README.md",
    "results/README.md",
    "assets/paper_figures/README.md",
    "docs/implementation_notes.md",
]
LINK = re.compile(r"\]\(([^)\s]+)\)|src=\"([^\"]+)\"")


@pytest.mark.parametrize("document", DOCUMENTS)
def test_relative_links_resolve(document):
    path = REPOSITORY_ROOT / document
    targets = [a or b for a, b in LINK.findall(path.read_text(encoding="utf-8"))]
    local = [t for t in targets if not t.startswith(("http://", "https://", "#", "mailto:"))]
    for target in local:
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        assert resolved.exists(), f"{document}: broken link {target}"
    if document == "README.md":
        assert sum(target.startswith("assets/paper_figures/fig") for target in local) == 5


PATH_SPAN = re.compile(r"`((?:qcv|scripts|configs|data|docs|results|assets|tests)/[^`]+)`")


@pytest.mark.parametrize("document", ["README.md", "docs/implementation_notes.md"])
def test_paths_and_symbols_named_in_code_spans_exist(document):
    """``path`` and ``path::symbol`` spans point at existing files and definitions."""
    text = (REPOSITORY_ROOT / document).read_text(encoding="utf-8")
    checked = 0
    for span in set(PATH_SPAN.findall(text)):
        path, _, symbol = span.partition("::")
        if any(character in path for character in "*{<$ "):
            continue
        target = REPOSITORY_ROOT / path
        assert target.exists(), f"{document} names a missing path: {path}"
        checked += 1
        if symbol:
            name = symbol.split(".")[-1]
            source = target.read_text(encoding="utf-8")
            assert re.search(rf"def {re.escape(name)}\(", source), f"{document}: {span} not found"
    assert checked >= 12
