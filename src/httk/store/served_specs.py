"""Derive OPTIMADE property names and types from the storage schema IR.

This module deliberately depends only on the neutral schema representation and
httk-core definitions. SQL and Mongo layers can therefore share the same
property derivation without importing one another's backend.
"""

from collections.abc import Mapping
from typing import Any, Final, cast

from httk.core import PropertyDefinition

from httk.store.db.schema import FieldSpec, TableSchema

__all__ = ["definition_fulltype", "served_specs"]

_SCALAR_FULLTYPES: Final[dict[str, str]] = {
    "str": "string",
    "int": "integer",
    "bool": "boolean",
    "float": "float",
}
_CODEC_FULLTYPES: Final[dict[str, str]] = {
    "float": "float",
    "fraction": "float",
    "fracscalar": "float",
    "surdscalar": "float",
    "datetime": "timestamp",
}


def _fulltype_of(spec: FieldSpec) -> str | None:
    """Return the served OPTIMADE fulltype for one schema field, if any."""
    if spec.role == "scalar":
        return _SCALAR_FULLTYPES.get(spec.columns[0].kind)
    if spec.role == "encoded":
        assert spec.codec_name is not None
        return _CODEC_FULLTYPES.get(spec.codec_name)
    if spec.role == "fixed_array":
        return "list of list of float"
    if spec.role == "child":
        if spec.shape is not None:
            return "list of list of float"
        if spec.target is not None:
            return None
        if spec.codec_name is not None:
            element = _CODEC_FULLTYPES.get(spec.codec_name)
        else:
            element = _SCALAR_FULLTYPES.get(spec.child.element_columns[0].kind) if spec.child is not None else None
        return None if element is None else f"list of {element}"
    return None


def served_specs(schema: TableSchema, prefix: str) -> list[tuple[str, FieldSpec, str]]:
    """Return the served ``(name, field spec, fulltype)`` triples.

    :param schema: The resolved schema whose fields are inspected.
    :param prefix: The registered prefix used for served property names.
    :return: One triple for every schema field with an OPTIMADE value type.
    """
    served: list[tuple[str, FieldSpec, str]] = []
    for spec in schema.fields:
        fulltype = _fulltype_of(spec)
        if fulltype is not None:
            served.append((f"{prefix}custom_{spec.field}", spec, fulltype))
    return served


def _fulltype_from_doc(doc: Mapping[str, Any]) -> str:
    optimade_type = doc["x-optimade-type"]
    if optimade_type == "list":
        return "list of " + _fulltype_from_doc(doc["items"])
    if optimade_type == "dictionary":
        return "dict"
    return cast(str, optimade_type)


def definition_fulltype(definition: PropertyDefinition) -> str:
    """Return the OPTIMADE fulltype declared by a property definition.

    :param definition: The property definition to inspect.
    :return: Its scalar, list, or dictionary fulltype string.
    """
    return _fulltype_from_doc(definition.as_optimade())
