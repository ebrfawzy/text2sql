"""SQL generation using LLM with Jinja2 prompt templates."""

from __future__ import annotations

import logging
import random
import re
from collections.abc import AsyncIterator
from typing import Any

from text2sql.db import DatabaseConnection
from text2sql.llm import LLMClient, extract_sql
from text2sql.pipeline.events import TokenDelta
from text2sql.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

# The agent's ``retrieval``-mode schema: a single ``"Tables: a, b, c"`` line.
_TABLE_LIST = re.compile(r"Tables:\s*")


def randomize_schema_order(schema_text: str) -> str:
    """Randomize the order of field lines within each table block.

    Shared by both generation paths, since repeated runs of one prompt would otherwise be
    byte-identical.

    Args:
        schema_text: The rendered schema, or the agent's one-line table list, whose names
            are shuffled instead so that mode keeps the diversity lever.

    Returns:
        The schema with its field lines reordered.
    """
    if prefix := _TABLE_LIST.match(schema_text):
        head, _, joins = schema_text[prefix.end():].partition("\n")
        names = [n.strip() for n in head.split(",")]
        random.shuffle(names)
        return prefix.group(0) + ", ".join(names) + (f"\n{joins}" if joins else "")

    blocks = schema_text.split("\n\n")
    randomized: list[str] = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) > 2:
            header = [lines[0]]
            field_lines = lines[1:]
            random.shuffle(field_lines)
            randomized.append("\n".join(header + field_lines))
        else:
            randomized.append(block)
    return "\n\n".join(randomized)


# Reasoning styles for candidate diversity (CHASE-SQL, Pourreza et al. 2024); the
# instruction each name stands for is inlined in the two generation templates.
STRATEGIES = ("direct", "decompose", "query_plan")


def strategy_for(mode: str, index: int) -> str:
    """Pick the reasoning strategy for one candidate.

    Args:
        mode: A strategy name, or ``diverse`` to cycle through all of them.
        index: Candidate index.

    Returns:
        The strategy name.
    """
    names = STRATEGIES if mode == "diverse" else (mode,)
    return names[index % len(names)]


class SQLGenerator:
    """Generates candidate SQL queries for a natural language question.

    Diversity comes from varying the sampling seed and the order of the schema fields in
    the prompt.
    """

    # Kept as a method for call-site readability; the implementation is shared.
    _randomize_schema_order = staticmethod(randomize_schema_order)

    def __init__(self, llm: LLMClient, db: DatabaseConnection, prompt_manager: PromptManager) -> None:
        self.llm = llm
        self.db = db
        self.prompt_manager = prompt_manager

    async def generate(
        self,
        question: str,
        schema_text: str,
        *,
        num_candidates: int = 3,
        context: dict[str, Any] | None = None,
        strategy_mode: str = "direct",
    ) -> AsyncIterator[TokenDelta | list[str]]:
        """Generate candidates with live token streaming.

        Args:
            question: The user's natural language question.
            schema_text: Schema rendered into the prompt.
            num_candidates: How many candidates to attempt.
            context: Extra template arguments (knowledge, examples).
            strategy_mode: A strategy name, or ``diverse``.

        Yields:
            A :class:`TokenDelta` per token, then the ``list[str]`` of SQL candidates.

        Raises:
            Exception: The last generation failure, when it lost every candidate - a silent
                empty list is indistinguishable from a model that declined.
        """
        candidates: list[str] = []
        failure: Exception | None = None
        self.conversations: list[list[dict[str, Any]]] = []

        for i in range(num_candidates):
            schema_variant = self._randomize_schema_order(schema_text) if i > 0 else schema_text
            prompt = self.prompt_manager.render(
                "generate_sql",
                schema=schema_variant,
                question=question,
                **(context or {}),
                dialect=self.db.dialect_name,
                strategy=strategy_for(strategy_mode, i),
            )
            messages = [{"role": "user", "content": prompt}]

            try:
                full_text: list[str] = []
                async for text, is_thinking in self.llm.stream_chat_messages(  # type: ignore[misc]
                    messages, seed=i,
                ):
                    yield TokenDelta(text=text, is_thinking=is_thinking)
                    if not is_thinking:
                        full_text.append(text)

                raw = "".join(full_text)
                self.conversations.append([*messages, {"role": "assistant", "content": raw}])
                sql = extract_sql(raw)
                if sql:
                    candidates.append(sql)
                    logger.info("Candidate %d/%d generated (%d chars)", i + 1, num_candidates, len(sql))
                else:
                    # No exception, no SQL: the reply was empty, prose, or unparseable. Log the
                    # reply itself, else the loss is undiagnosable.
                    logger.warning(
                        "Candidate %d/%d yielded no SQL from a %d-char reply: %r",
                        i + 1, num_candidates, len(raw), raw[:300])
            except Exception as e:
                logger.warning("Candidate %d generation failed: %s", i + 1, e)
                failure = e

        if not candidates and failure is not None:
            raise failure

        logger.info("Generated %d/%d candidates", len(candidates), num_candidates)
        yield candidates
