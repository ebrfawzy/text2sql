"""Jinja2 prompt template manager with versioning and overrides.

Loads ``templates/{version}/{name}.j2``, which a custom template directory or a
``TEXT2SQL_PROMPT_{NAME}_PATH`` env var can override per prompt.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class _OverrideLoader(BaseLoader):
    """Jinja2 loader that checks per-template overrides before the bundled templates.

    An override comes from the ``overrides`` mapping or from
    ``TEXT2SQL_PROMPT_{NAME}_PATH``, e.g. ``TEXT2SQL_PROMPT_GENERATE_SQL_PATH``.
    """

    def __init__(self, fallback: FileSystemLoader, overrides: dict[str, Path]) -> None:
        self._fallback = fallback
        self._overrides = overrides  # {template_name: path}

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str | None, Any]:
        """Load one template's source.

        Args:
            environment: The Jinja2 environment.
            template: Template filename.

        Returns:
            ``(source, path, uptodate)``, from the first override that exists, else from the
            bundled templates.
        """
        base_name = template.removesuffix(".j2")
        env_key = f"TEXT2SQL_PROMPT_{base_name.upper()}_PATH"

        if base_name in self._overrides:
            path = self._overrides[base_name]
            if path.exists():
                source = path.read_text(encoding="utf-8")
                return source, str(path), lambda: path.stat().st_mtime

        env_path = os.environ.get(env_key)
        if env_path:
            path = Path(env_path)
            if path.exists():
                logger.info("Using override template for %s: %s",
                            template, path)
                source = path.read_text(encoding="utf-8")
                return source, str(path), lambda: path.stat().st_mtime

        return self._fallback.get_source(environment, template)


class PromptManager:
    """Manages Jinja2 prompt templates with versioning and caching.

    Usage::

        pm = PromptManager()
        prompt = pm.render("generate_sql", schema="...", question="...")

    With custom template dir::

        pm = PromptManager(template_dir="/my/templates", version="v2")

    With per-template override::

        pm = PromptManager(overrides={"generate_sql": Path("/custom/generate_sql.j2")})
    """

    # Default template directory: bundled with the package.
    _BUNDLED_DIR = Path(__file__).parent / "templates"

    def __init__(
        self,
        template_dir: str | Path | None = None,
        version: str = "v1",
        overrides: dict[str, Path] | None = None,
    ) -> None:
        """Initialize the prompt manager.

        Args:
            template_dir: Root directory containing versioned template subdirs.
                          Defaults to the bundled templates/ directory.
            version: Template version to load (subdirectory name, e.g. "v1").
            overrides: ``{template base name: override path}``.
        """
        self.version = version
        self._overrides = overrides or {}

        root = Path(template_dir) if template_dir else self._BUNDLED_DIR
        versioned_dir = root / version

        if not versioned_dir.exists():
            logger.warning(
                "Template directory %s does not exist, falling back to bundled templates",
                versioned_dir,
            )
            versioned_dir = self._BUNDLED_DIR / version

        fs_loader = FileSystemLoader(str(versioned_dir))
        loader = _OverrideLoader(fs_loader, self._overrides)

        self._env = Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
            autoescape=False,  # these are prompts, not HTML
        )
        logger.info(
            "PromptManager initialized: dir=%s, version=%s, overrides=%d",
            versioned_dir,
            version,
            len(self._overrides),
        )

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render one template.

        Args:
            template_name: Template name, with or without the ``.j2`` suffix.
            **kwargs: Template arguments.

        Returns:
            The rendered prompt, stripped, so a loop's trailing newlines stay out of the
            blocks templates interpolate into each other.
        """
        if not template_name.endswith(".j2"):
            template_name = f"{template_name}.j2"

        rendered = self._env.get_template(template_name).render(**kwargs).strip()
        logger.debug("Rendered template %s (%d chars):\n%s",
                     template_name, len(rendered), rendered)
        return rendered
