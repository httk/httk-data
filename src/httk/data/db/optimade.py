"""OPTIMADE-filter querying over stored dataclasses: sugar tying the SQL layer to the translator.

:func:`optimade_filter_searcher` builds a :class:`~httk.data.db.searcher.SqlSearcher`
over a storable class directly from an OPTIMADE filter string, deriving the
recognized property names and their types from the class's resolved
:class:`~httk.data.db.schema.TableSchema` — the same derivation
:class:`~httk.data.db.entry_provider.StoreEntryProvider` uses to serve the
class (via the shared :func:`~httk.data.db.entry_provider.served_specs` /
:func:`~httk.data.db.entry_provider.auto_definition` helpers), so a filter
that works against the served API works against the store.

Filter property names are the served names: ``{prefix}{field}`` (default
``_httk_<field>``) for every servable stored field. Related storable classes
declared via ``related_classes`` can be filtered relationally —
``references.id HAS "references-3"`` and depth-1 related-property filters like
``references._httk_doi CONTAINS "10.1"`` — over the class's reference or
child-of-storable fields.
"""

from collections.abc import Mapping
from typing import Any, Callable, cast

from httk.core import EntryTypeDefinition, FilterAst, PropertyDefinition

from httk.data.db.entry_provider import served_specs
from httk.data.db.schema import resolve_schema
from httk.data.db.searcher import SqlColumn, SqlExpression, SqlSearcher, SqlVariable
from httk.data.db.store import SqlStore
from httk.data.optimade_query import (
    FilterTranslationError,
    filter_searcher,
    known_unknown_handler,
    set_handler,
    simple_property_handlers,
)
from httk.data.query import Searcher, SearchExpression, SearchVariable

__all__ = [
    "optimade_filter_searcher",
]


def _fulltype_from_doc(doc: Mapping[str, Any]) -> str:
    optimade_type = doc["x-optimade-type"]
    if optimade_type == "list":
        return "list of " + _fulltype_from_doc(doc["items"])
    if optimade_type == "dictionary":
        return "dict"
    return cast(str, optimade_type)


def _definition_fulltype(definition: PropertyDefinition) -> str:
    """The ``fulltype`` string a property definition describes."""
    return _fulltype_from_doc(definition.as_optimade())


def _related_sid(related_type: str, value: Any) -> int:
    """Parse a default-minted ``"<related_type>-<sid>"`` id back to its sid.

    Any value not of that exact form (a foreign id format, a different entry
    type, a non-string) parses to ``-1``, a sentinel sid that never exists —
    such ids simply match nothing, they are not an error.
    """
    if isinstance(value, str) and value.startswith(related_type + "-"):
        tail = value[len(related_type) + 1 :]
        if tail.isdigit():
            return int(tail)
    return -1


def _related_id_has_handlers(related_type: str, field: str) -> Mapping[str, Callable[..., Any]]:
    """The ``'<related_type>.id'`` HAS handler over a reference or child-of-storable field."""

    def has_handler(
        entry: str, ops: Any, values: Any, search_variable: SearchVariable, has_type: str, inv: bool
    ) -> tuple[SearchExpression, bool]:
        # A SqlVariable satisfies the set-handler contract here: both its
        # reference fields (via SqlReference's set operations over the foreign
        # key) and its child-of-storable fields (a SqlColumn over the child
        # element column) accept sid values.
        sids = [_related_sid(related_type, value) for value in values]
        return set_handler(field, ops, sids, inv, has_type, search_variable)

    return {'HAS': has_handler}


def _own_id_handlers(related_type: str) -> Mapping[str, Callable[..., Any]]:
    """Handlers serving the related class's own ``id`` property in a nested sub-search."""

    def comparison(entry: str, op: str, value: Any, search_variable: SqlVariable) -> SqlExpression:
        if op not in ('=', '!='):
            raise FilterTranslationError("Ordering comparisons on relationship ids not implemented.", "not-implemented")
        sid_column = search_variable.sid
        assert isinstance(sid_column, SqlColumn)
        sid = _related_sid(related_type, value)
        return sid_column == sid if op == '=' else sid_column != sid

    return {'comparison': comparison, 'unknown': known_unknown_handler}


def optimade_filter_searcher(
    store: SqlStore,
    cls: type,
    filter_string: str | FilterAst,
    *,
    prefix: str = "_httk_",
    definition: EntryTypeDefinition | None = None,
    extra_handlers: Mapping[str, Mapping[str, Callable[..., Any]]] | None = None,
    related_classes: Mapping[str, type] | None = None,
) -> Searcher:
    """Build a searcher over the stored rows of ``cls`` from an OPTIMADE filter.

    ``filter_string`` is an OPTIMADE filter string or an already-parsed
    :py:type:`~httk.core.FilterAst`. The returned searcher outputs the matching
    stored instances (``item[0][0]`` per match).

    **Property names.** Every servable stored field of ``cls`` (per
    :func:`~httk.data.db.entry_provider.served_specs`) is filterable as
    ``{prefix}{field}``, with its schema-derived fulltype driving constant
    conversion and handler dispatch (rational fields compare on their
    documented-approximate float query column). Unknown names carrying
    ``prefix`` raise :class:`~httk.data.optimade_query.FilterTranslationError`
    (``"unrecognized-property"``); other unknown names match nothing, per the
    OPTIMADE specification. Unprefixed property names (beyond ``id``/``type``)
    are recognized only when a ``definition`` describes them — and even then
    they translate only if ``extra_handlers`` supplies their handlers, since
    the store knows no column for them. The OPTIMADE core ``id`` and ``type``
    properties are recognized but **not supported** without ``extra_handlers``
    entries (a store row has no served id: ids are minted at serving time), so
    filtering on them raises ``"not-implemented"``.

    **Relationships.** ``related_classes`` maps relationship-type names to
    storable classes; each must be the target of exactly one reference or
    child-of-storable (``list[Target]``) field of ``cls`` (anything else
    raises :class:`ValueError`). For each such ``(rtype, rcls)``:

    - ``<rtype>.id HAS "<rtype>-<sid>"`` filters over the field's foreign-key
      (or child-element) column against default-minted
      ``"<entry type>-<sid>"`` ids, as
      :class:`~httk.data.db.entry_provider.StoreEntryProvider` mints them;
      ids of any other format match nothing. Custom ``id_of`` minting is out
      of scope here — with a custom minting scheme, supply your own
      ``'<rtype>.id'`` entry via ``extra_handlers``.
    - Depth-1 related-property filters (``<rtype>._httk_doi CONTAINS "10.1"``,
      ``<rtype>.id != "..."``, ``<rtype>._httk_year IS KNOWN``, ...) resolve by
      a two-phase semi-join: a nested ``optimade_filter_searcher`` over
      ``rcls`` (with the same ``prefix``; no further relationship nesting)
      collects the matching related sids, which are then matched as
      ``<rtype>.id HAS ANY ...``. Each dotted filter node resolves
      independently — see :func:`~httk.data.optimade_query.translate_filter_ast`.

    ``extra_handlers`` entries are merged over the derived handler table last
    (so they can also override derived handlers); extra property names that are
    otherwise unknown are recognized with fulltype ``"unknown"`` (filter
    constants pass through unconverted).

    Raises:
        httk.data.optimade_query.FilterTranslationError: When the filter cannot be translated.
        httk.core.ParserSyntaxError: When a filter string does not parse.
        ValueError: When a ``related_classes`` entry does not match exactly one
            reference or child-of-storable field of ``cls``.
    """
    schema = resolve_schema(cls)
    served = served_specs(schema, prefix)
    property_fulltypes: dict[str, str] = {"id": "string", "type": "string"}
    property_fulltypes.update({name: fulltype for name, _spec, fulltype in served})
    columns = {name: spec.field for name, spec, _fulltype in served}
    if definition is not None:
        for name, prop in definition.properties.items():
            if name in ("id", "type"):
                continue
            property_fulltypes[name] = _definition_fulltype(prop)

    handlers = simple_property_handlers(cls.__name__, columns, property_fulltypes)
    # The default id/type handlers of simple_property_handlers query the
    # serving-layer '__id' column and constant entry-type name; neither exists
    # on a store row, so drop them (see the docstring: id/type filtering needs
    # extra_handlers).
    del handlers["id"]
    del handlers["type"]

    relationship_targets: tuple[str, ...] = ()
    resolver = None
    if related_classes:
        related = dict(related_classes)
        for related_type, related_cls in related.items():
            matching = [
                spec for spec in schema.fields if spec.target is related_cls and spec.role in ("reference", "child")
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"related_classes entry {related_type!r} ({related_cls.__name__}) matches "
                    f"{'no' if not matching else str(len(matching))} reference or child-of-storable "
                    f"field{'' if len(matching) == 1 else 's'} of {cls.__name__}; exactly one is required"
                )
            handlers[f"{related_type}.id"] = _related_id_has_handlers(related_type, matching[0].field)
        relationship_targets = tuple(related)

        def resolve_related(related_type: str, sub_ast: FilterAst) -> tuple[str, ...]:
            nested = optimade_filter_searcher(
                store,
                related[related_type],
                sub_ast,
                prefix=prefix,
                extra_handlers={"id": _own_id_handlers(related_type)},
            )
            assert isinstance(nested, SqlSearcher)
            sid_column = nested._variables[0].sid
            assert isinstance(sid_column, SqlColumn)
            nested.output(sid_column, "sid")
            # Output 0 is the matched instance (declared by filter_searcher), output 1 the sid.
            return tuple(f"{related_type}-{int(values[1])}" for values, _names in nested)

        resolver = resolve_related

    if extra_handlers:
        for name in extra_handlers:
            if "." not in name:
                property_fulltypes.setdefault(name, "unknown")
        handlers.update(extra_handlers)

    return filter_searcher(
        store,
        cls,
        filter_string,
        entry_type=cls.__name__,
        property_fulltypes=property_fulltypes,
        handlers=handlers,
        recognized_prefixes=(prefix,),
        relationship_targets=relationship_targets,
        related_property_resolver=resolver,
    )
