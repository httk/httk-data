"""Derive portable query fields and operations from OPTIMADE definitions."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from httk.core import EntryTypeDefinition, PropertyDefinition

__all__ = ["PortableQueryCapabilities", "portable_query_capabilities", "portable_query_fields"]

_SCALAR_TYPES = frozenset({"boolean", "string", "integer", "float", "timestamp"})
_STRUCTURED_TYPES = frozenset({"list", "dictionary", "object"})
_ALL_MANDATORY = "all mandatory"
_EQUALITY_ONLY = "equality only"


@dataclass(frozen=True, slots=True)
class PortableQueryCapabilities:
    """The query operations guaranteed by one property definition.

    ``query-support`` expresses a cross-provider guarantee, not a particular
    server's implementation detail.  ``all optional`` is deliberately
    fail-closed here: it gives a portable client no operation it can rely on.
    A server may offer more, but that is not represented by the definition.
    """

    query_support: str | None
    operations: frozenset[str]

    def supports(self, operation: str) -> bool:
        """Whether ``operation`` is guaranteed by this definition."""
        return operation in self.operations


def _names(argument: Iterable[str], *, parameter: str, known: set[str]) -> set[str]:
    """Validate a caller-supplied field-name iterable and return its members."""
    names: set[str] = set()
    for name in argument:
        if not isinstance(name, str):
            raise ValueError(f"{parameter} names must be strings, got {type(name).__name__}")
        if name in names:
            raise ValueError(f"duplicate {parameter} field name: {name!r}")
        if name not in known:
            raise ValueError(f"unknown {parameter} field name: {name!r}")
        names.add(name)
    return names


def _query_support(definition: PropertyDefinition) -> str | None:
    """Return the normalized declared query-support level, if valid."""
    requirements = definition.requirements
    if "query-support" not in requirements:
        return None
    support = requirements["query-support"]
    return support.casefold() if isinstance(support, str) else None


def _is_flat_list(definition: PropertyDefinition) -> bool:
    """Whether an OPTIMADE list definition has a non-structured item definition."""
    items = definition.as_optimade().get("items")
    if not isinstance(items, Mapping):
        return False
    item_type = items.get("x-optimade-type")
    return not (isinstance(item_type, str) and item_type.casefold() in _STRUCTURED_TYPES)


def _is_portable_type(definition: PropertyDefinition) -> bool:
    """Whether a property type belongs to the portable query profile."""
    type_name = definition.optimade_type.casefold()
    return type_name in _SCALAR_TYPES or (type_name == "list" and _is_flat_list(definition))


def portable_query_capabilities(definition: PropertyDefinition) -> PortableQueryCapabilities:
    """Derive the portable operation subset for ``definition``.

    The operation names are ``"equality"``, ``"ordering"``,
    ``"stringmatching"``, and ``"set"``.  They intentionally describe the
    query-language operation families rather than storage implementation.
    ``IS [NOT] KNOWN`` is part of the equality family because it is the NULL
    spelling of equality/inequality in the OPTIMADE filter language.
    """
    support = _query_support(definition)
    if support == _EQUALITY_ONLY:
        return PortableQueryCapabilities(support, frozenset({"equality"}))
    if support != _ALL_MANDATORY or not _is_portable_type(definition):
        return PortableQueryCapabilities(support, frozenset())

    kind = definition.optimade_type.casefold()
    if kind == "list":
        return PortableQueryCapabilities(support, frozenset({"set"}))
    operations = {"equality"}
    if kind != "boolean":
        operations.add("ordering")
    if kind == "string":
        operations.add("stringmatching")
    return PortableQueryCapabilities(support, frozenset(operations))


def portable_query_fields(
    entry_type: EntryTypeDefinition,
    *,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the ordered portable query fields described by ``entry_type``.

    By default, this selects scalar fields and flat lists with at least one
    operation guaranteed by their definition. ``include`` is an explicit
    binding override for named existing properties; it is appended after the
    derived fields in entry definition order, but does not manufacture query
    capabilities absent from that definition. ``exclude`` always wins. Both
    arguments reject unknown or duplicate names so binding mistakes cannot
    silently broaden a profile.
    """
    properties = entry_type.properties
    property_names = tuple(properties)
    known = set(property_names)
    included = _names(include, parameter="include", known=known)
    excluded = _names(exclude, parameter="exclude", known=known)

    selected = [
        name
        for name in property_names
        if name not in excluded and portable_query_capabilities(properties[name]).operations
    ]
    selected.extend(
        name for name in property_names if name in included and name not in excluded and name not in selected
    )
    return tuple(selected)
