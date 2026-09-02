"""Config-form schema for the Text2SQL web UI, derived from ``Settings``.

Every descriptor is read off the pydantic field itself: control and options from the
annotation, ``min``/``max`` from the ``ge``/``le`` metadata, tooltip from the description,
default from the live settings. The group label and the endpoints a field applies to come
from ``SECTION_SCOPE``, so a new setting appears in the UI as soon as it is added to
:class:`~text2sql.config.Settings` and its YAML section.

Fields with no renderable control (``profile_selection``'s nested dict) are skipped;
they stay valid in the request body, they just have no form input.
"""

from __future__ import annotations

from enum import Enum
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from annotated_types import Ge, Le
from pydantic.fields import FieldInfo

from text2sql.api.model import REQUEST_ONLY
from text2sql.config import FIELD_DEPENDS, SECTION_SCOPE, Settings, section_items

# Label tokens rendered upper-case instead of title-case ("db_uri" -> "DB URI").
_ACRONYMS = {"db", "uri", "kb", "llm", "ms", "id", "k", "sql", "s3", "aws", "jsonl"}

# A closed int range this small reads better as a dropdown than a number input.
_MAX_SELECT_RANGE = 10


def _base_type(annotation: Any) -> Any:
    """Strip ``| None``, in either spelling, from an annotation.

    Args:
        annotation: The field annotation.

    Returns:
        The single remaining type, or None when the union holds several.
    """
    if get_origin(annotation) in (Union, UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return args[0] if len(args) == 1 else None
    return annotation


def _label(name: str) -> str:
    """Title-case an identifier for display.

    Args:
        name: The field name.

    Returns:
        The label, with known acronyms upper-cased.
    """
    return " ".join(w.upper() if w in _ACRONYMS else w.title() for w in name.split("_"))


def _control(field: FieldInfo) -> dict[str, Any]:
    """Choose the UI control for one field.

    Args:
        field: The pydantic field.

    Returns:
        The control with its options and bounds, or ``{}`` when the type has no form input.
    """
    annotation = _base_type(field.annotation)
    low: Any = next((m.ge for m in field.metadata if isinstance(m, Ge)), None)
    high: Any = next((m.le for m in field.metadata if isinstance(m, Le)), None)

    if annotation is bool:
        return {"control": "toggle"}
    if get_origin(annotation) is list and get_origin(inner := get_args(annotation)[0]) is Literal:
        return {"control": "multi", "options": list(get_args(inner))}
    if get_origin(annotation) is Literal:
        return {"control": "select", "options": list(get_args(annotation))}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"control": "select", "options": [m.value for m in annotation]}
    if annotation is int and low is not None and high is not None and high - low < _MAX_SELECT_RANGE:
        return {"control": "select", "options": list(range(low, high + 1))}
    if annotation in (int, float):
        return {"control": "number", "min": low, "max": high, "step": 1 if annotation is int else 0.01}
    return {"control": "text"} if annotation is str else {}


def _field_schema(name: str, field: FieldInfo, label: str, default: Any) -> dict[str, Any] | None:
    """Describe one form field.

    Args:
        name: Field name.
        field: The pydantic field.
        label: Display label.
        default: The effective default.

    Returns:
        The descriptor, or None when the field has no renderable control.
    """
    if not (control := _control(field)):
        return None
    depends = FIELD_DEPENDS.get(name)
    return {
        "name": name,
        "label": label,
        "default": default.value if isinstance(default, Enum) else default,
        "help": field.description,
        **control,
        **({"depends_on": {"field": depends[0], "values": list(depends[1])}} if depends else {}),
    }


def build_config_schema(settings: Settings) -> list[dict[str, Any]]:
    """Build the grouped config-form descriptor.

    Args:
        settings: The effective settings, supplying each field's default.

    Returns:
        One group per section; request-only fields keep their request-model default.
    """
    groups: list[dict[str, Any]] = []
    for section, (label, endpoints) in SECTION_SCOPE.items():
        rows = [(n, Settings.model_fields[n], _label(key), getattr(settings, n))
                for key, n in section_items(section)]
        rows += [(n, f, _label(n), f.default) for n, f in REQUEST_ONLY.get(section, {}).items()]
        if fields := [s | {"endpoints": list(endpoints)} for r in rows if (s := _field_schema(*r))]:
            groups.append({"key": section, "label": label, "fields": fields})
    return groups
