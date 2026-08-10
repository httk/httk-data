"""Validate values offline against OPTIMADE property definitions with JSON Schema.

httk-core models each OPTIMADE property as a self-describing definition whose
:meth:`~httk.core.PropertyDefinition.as_optimade` document *is* a JSON Schema
(plus ``x-optimade-*`` annotations that validators ignore). This module uses
that document directly to validate concrete values with ``jsonschema``'s Draft
2020-12 dialect.

The definitions are self-contained: they carry no ``$ref``, so validation never
needs to resolve or fetch anything. The document's ``$schema`` key points at the
OPTIMADE property-definition *meta*-schema URI (which describes definitions, not
values); it is removed before building the validator so ``jsonschema`` never
attempts to resolve it, and the dialect is pinned explicitly to
``jsonschema.Draft202012Validator``. Validation is therefore fully offline.
The local RFC 3339 ``date-time`` checker below supplies format validation
without network access or additional dependencies.
"""

import datetime
import re
from collections.abc import Mapping
from typing import Any

import jsonschema
import jsonschema.exceptions
from httk.core import EntryTypeDefinition, PropertyDefinition

_RFC3339_DATETIME = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})$"
)
_FORMAT_CHECKER = jsonschema.FormatChecker()


@_FORMAT_CHECKER.checks("date-time")
def _is_rfc3339_datetime(value: Any) -> bool:
    """Accept RFC 3339 date-times, not ISO date-only or naive values.

    The accepted grammar is ``YYYY-MM-DD[Tt]HH:MM:SS[.fraction](Z|z|+HH:MM|-HH:MM)``.
    ``fromisoformat`` validates the calendar, clock, and offset ranges after the
    explicit grammar check; JSON Schema applies ``format`` only to strings, so
    non-strings pass here and are rejected by the property's ``type`` keyword.
    """
    if not isinstance(value, str):
        return True
    if not _RFC3339_DATETIME.fullmatch(value):
        return False
    normalized = value.replace("t", "T", 1)
    if normalized.endswith(("Z", "z")):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


class PropertyValidationError(ValueError):
    """Report that a value did not conform to its OPTIMADE property definition.

    Carries the offending property ``name`` and a human-readable ``message``. For
    single-value failures the message wraps the underlying ``jsonschema`` error
    message, and that ``jsonschema.exceptions.ValidationError`` is preserved
    as the chained ``__cause__``.

    :param name: The name of the invalid property.
    :param message: The validation failure message.
    """

    def __init__(self, name: str, message: str) -> None:
        self.name = name
        self.message = message
        super().__init__(f"Property {name!r}: {message}")


def _validator_schema(definition: PropertyDefinition) -> dict[str, Any]:
    """The definition's OPTIMADE document as a self-contained JSON Schema.

    Drops the ``$schema`` meta-schema reference so ``jsonschema`` treats the
    document as a plain schema for the *value* (pinned to Draft 2020-12 by the
    caller) and never tries to resolve the meta-schema URI.
    """
    schema = definition.as_optimade()
    schema.pop("$schema", None)
    # Nullable enum properties use ``type: [<kind>, "null"]`` while the enum lists
    # only the non-null vocabulary. JSON Schema applies both constraints, so explicitly
    # include null in the validator copy instead of rejecting a value the OPTIMADE
    # definition declares nullable.
    if definition.nullable and isinstance(schema.get("enum"), list) and None not in schema["enum"]:
        schema["enum"] = [*schema["enum"], None]
    return schema


def validate_property(definition: PropertyDefinition, value: Any) -> None:
    """Validate a single ``value`` against ``definition``'s JSON-Schema payload.

    Builds a ``jsonschema.Draft202012Validator`` directly from the
    definition's document (with the ``$schema`` meta-schema reference removed)
    and validates ``value`` against it using the local format checker. Returns
    ``None`` on success; raises :class:`PropertyValidationError` on failure,
    chaining the underlying ``jsonschema.exceptions.ValidationError`` as the
    cause. No network access or registry lookup ever happens.

    :param definition: The self-contained OPTIMADE property definition.
    :param value: The value to validate.
    :return: None.
    :raises PropertyValidationError: If ``value`` violates ``definition``.
    """
    validator = jsonschema.Draft202012Validator(_validator_schema(definition), format_checker=_FORMAT_CHECKER)
    try:
        validator.validate(value)
    except jsonschema.exceptions.ValidationError as exc:
        raise PropertyValidationError(definition.name, exc.message) from exc


def validate_record(entry_type: EntryTypeDefinition, record: Mapping[str, Any]) -> None:
    """Validate every property present in ``record`` against ``entry_type``.

    Each key in ``record`` must be described by ``entry_type``; unknown property
    names are rejected with a :class:`PropertyValidationError` naming them and the
    entry type. ``id`` and ``type`` must both be present. Properties described by
    the definition but absent from ``record`` are simply not checked (serving a
    subset of the described properties is normal). The value of every property
    that *is* present is validated via :func:`validate_property`. Returns ``None``
    on success.

    :param entry_type: The entry definition describing allowed properties.
    :param record: The record mapping to validate.
    :return: None.
    :raises PropertyValidationError: If a property is unknown, ``id`` or ``type``
        is missing, or a value violates its property definition.
    """
    properties = entry_type.properties

    unknown = sorted(name for name in record if name not in properties)
    if unknown:
        plural = "y" if len(unknown) == 1 else "ies"
        raise PropertyValidationError(
            ", ".join(unknown),
            f"unknown propert{plural} for entry type {entry_type.name!r}: {', '.join(unknown)}.",
        )

    for required in ("id", "type"):
        if required not in record:
            raise PropertyValidationError(
                required,
                f"required property {required!r} is missing from the {entry_type.name!r} record.",
            )

    for name, value in record.items():
        validate_property(properties[name], value)
