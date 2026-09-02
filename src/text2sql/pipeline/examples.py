"""Example/scenario store for the agent's ``lookup_example`` tool.

Loads a scenarios.md file, indexes it by ``##`` headings, and searches those headings.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ExampleStore:
    """Loads and indexes a scenarios.md file for the ``lookup_example`` agent tool.

    Each ``##`` heading delimits a section of domain-specific guidance.

    Usage::

        store = ExampleStore("scenarios.md")
        results = store.search("net revenue", top_k=3)
        for r in results:
            print(r)
    """

    def __init__(self, scenarios_file: str | Path | None = None) -> None:
        """Initialize the example store.

        Args:
            scenarios_file: Path to a scenarios.md file. If None, store is empty.
        """
        self._sections: dict[str, str] = {}
        self._headings: list[str] = []

        if scenarios_file:
            path = Path(scenarios_file)
            if path.exists():
                self._load(path)
            else:
                logger.warning("Scenarios file not found: %s", path)

    @property
    def headings(self) -> list[str]:
        """Every section heading, in file order."""
        return list(self._headings)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Search for sections matching the query.

        Matching is a case-insensitive substring or word overlap over the headings.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            The matching sections, best first, each with its heading.
        """
        if not self._sections:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[float, str]] = []
        for heading in self._headings:
            heading_lower = heading.lower()
            heading_words = set(heading_lower.split())

            if query_lower in heading_lower or heading_lower in query_lower:
                score = 1.0
            elif query_words & heading_words:
                overlap = len(query_words & heading_words)
                score = overlap / max(len(query_words), len(heading_words))
            else:
                score = 0.0

            if score > 0:
                scored.append((score, heading))

        scored.sort(key=lambda x: -x[0])

        results = []
        for score, heading in scored[:top_k]:
            content = self._sections[heading]
            results.append(f"## {heading}\n{content}")

        return results

    def _load(self, path: Path) -> None:
        """Parse a markdown file into sections indexed by ``##`` headings.

        Args:
            path: The scenarios file to read.
        """
        content = path.read_text(encoding="utf-8")
        sections: dict[str, str] = {}
        current_heading = ""
        current_lines: list[str] = []

        for line in content.splitlines():
            heading_match = re.match(r"^##\s+(.+)$", line)
            if heading_match:
                if current_heading:
                    sections[current_heading] = "\n".join(
                        current_lines).strip()
                current_heading = heading_match.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_heading:
            sections[current_heading] = "\n".join(current_lines).strip()

        self._sections = sections
        self._headings = list(sections.keys())
        logger.info("Loaded %d scenario sections from %s", len(sections), path)
