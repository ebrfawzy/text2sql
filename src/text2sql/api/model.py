"""Pydantic request models for the Text2SQL API endpoints.

Each body's config half is generated from ``text2sql.config.Settings``: exactly the fields
its YAML sections scope in, made optional so an absent key keeps the server's configured
value, forwarded straight into ``Text2SQL.build(**overrides)``. Only fields with no
``Settings`` counterpart are declared below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from text2sql.config import Settings, section_fields


def _optional(field: FieldInfo) -> tuple[Any, Any]:
    """Make a ``Settings`` field optional.

    Args:
        field: The pydantic field.

    Returns:
        ``(annotation, FieldInfo)`` with ``ge``/``le`` kept on the inner type.
    """
    inner: Any = Annotated[tuple([field.annotation, *field.metadata])] if field.metadata else field.annotation
    return (inner | None, Field(default=None, description=field.description))


def _config_model(endpoint: str) -> Any:
    """Build the config half of one endpoint's request body.

    Args:
        endpoint: ``ask``, ``profile`` or ``benchmark``.

    Returns:
        A model of every ``Settings`` field that endpoint accepts, all optional.
    """
    fields: dict[str, Any] = {n: _optional(Settings.model_fields[n]) for n in section_fields(endpoint)}
    return create_model(f"{endpoint}_config", **fields)


if TYPE_CHECKING:  # mypy cannot follow create_model(); these are plain models at runtime
    _AskConfig = BaseModel
    _ProfileConfig = BaseModel
    _BenchmarkConfig = BaseModel
else:
    _AskConfig, _ProfileConfig, _BenchmarkConfig = map(_config_model, ("ask", "profile", "benchmark"))


class AskRequest(_AskConfig):
    """Request body for the /ask endpoint."""

    question: str = Field(..., description="Natural language question about the database.")
    config: str | None = Field(None, description="Path to YAML config file.")


class ProfileRequest(_ProfileConfig):
    """Request body for the /profile and /cache endpoints."""

    db_uri: str = Field(..., description="SQLAlchemy database URI to profile.")
    config: str | None = Field(None, description="Path to YAML config file.")


class BenchmarkRequest(_BenchmarkConfig):
    """Request body for the /benchmark endpoint."""

    config: str | None = Field(None, description="Path to YAML config file.")
    max_examples: int | None = Field(None, ge=1, description="Max examples to run.")


class SchemaRequest(BaseModel):
    """Request body for the /schema endpoint (profiling-selection discovery)."""

    db_uri: str = Field(..., description="SQLAlchemy database URI to introspect.")


class CacheDeleteRequest(BaseModel):
    """Request body for the /cache/delete endpoint."""

    db_uri: str = Field(..., description="SQLAlchemy database URI whose cache to edit.")
    table: str = Field(..., description="Table to remove (or whose columns to remove).")
    columns: list[str] | None = Field(
        None, description="Columns to remove; None removes the whole table.")


# Per-endpoint form fields with no ``Settings`` counterpart, for the UI's config form;
# ``config`` is excluded, since the YAML path has its own control.
REQUEST_ONLY: dict[str, dict[str, FieldInfo]] = {
    "benchmark": {n: f for n, f in BenchmarkRequest.model_fields.items()
                  if n not in Settings.model_fields and n != "config"},
}
