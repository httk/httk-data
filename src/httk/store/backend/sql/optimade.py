"""Serve OPTIMADE filters over stored dataclasses through the SQL query layer.

:func:`optimade_filter_searcher` builds a :class:`~httk.store.backend.sql.searcher.SqlSearcher`
over a storable class directly from an OPTIMADE filter string, deriving the
recognized property names and their types from the class's resolved
:class:`~httk.store.backend.sql.schema.TableSchema` — the same derivation
:class:`~httk.store.backend.sql.entry_provider.StoreEntryProvider` uses to serve the
class (via the shared :func:`~httk.store.backend.sql.entry_provider.served_specs` /
:func:`~httk.store.backend.sql.entry_provider.auto_definition` helpers), so a filter
that works against the served API works against the store.

Filter property names are the served names: ``{prefix}{field}`` (default
``_httk_<field>``) for every servable stored field. Related storable classes
declared via ``related_classes`` can be filtered relationally —
``references.id HAS "references-3"`` and depth-1 related-property filters like
``references._httk_doi CONTAINS "10.1"`` — over the class's reference or
child-of-storable fields.

"""

from collections.abc import Callable, Mapping
from typing import Any

from httk.core import EntryTypeDefinition
from httk.core.optimade import FilterAst

from httk.store.backend.sql.schema import resolve_schema
from httk.store.backend.sql.searcher import SqlColumn, SqlExpression, SqlSearcher, SqlVariable
from httk.store.backend.sql.store import SqlStore
from httk.store.query import Searcher, SearchExpression, SearchVariable
from httk.store.query.optimade_filters import (
    FilterTranslationError,
    filter_searcher,
    known_unknown_handler,
    set_handler,
    simple_property_handlers,
)
from httk.store.served_specs import definition_fulltype, served_specs

__all__ = [
    "optimade_filter_searcher",
]


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
        entry: str, ops: Any, values: Any, search_variable: SearchVariable, has_type: str
    ) -> SearchExpression:
        # A SqlVariable satisfies the set-handler contract here: both its
        # reference fields (via SqlReference's set operations over the foreign
        # key) and its child-of-storable fields (a SqlColumn over the child
        # element column) accept sid values.
        sids = [_related_sid(related_type, value) for value in values]
        return set_handler(field, ops, sids, has_type, search_variable)

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
    :py:type:`~httk.core.optimade.FilterAst`. The returned searcher outputs the matching
    stored instances (``item[0][0]`` per match).

    Property names: every servable stored field of ``cls`` (per
    :func:`~httk.store.backend.sql.entry_provider.served_specs`) is filterable as
    ``{prefix}{field}``, with its schema-derived fulltype driving constant
    conversion and handler dispatch (rational fields compare on their
    documented-approximate float query column). Unknown names carrying
    ``prefix`` raise :class:`~httk.store.query.optimade_filters.FilterTranslationError`
    (``"unrecognized-property"``); other unknown names match nothing, per the
    OPTIMADE specification. Unprefixed property names (beyond ``id``/``type``)
    are recognized only when a ``definition`` describes them — and even then
    they translate only if ``extra_handlers`` supplies their handlers, since
    the store knows no column for them. The OPTIMADE core ``id`` and ``type``
    properties are recognized but **not supported** without ``extra_handlers``
    entries (a store row has no served id: ids are minted at serving time), so
    filtering on them raises ``"not-implemented"``.

    Relationships: ``related_classes`` maps relationship-type names to
    storable classes; each must be the target of exactly one reference or
    child-of-storable (``list[Target]``) field of ``cls`` (anything else
    raises :class:`ValueError`). For each such ``(rtype, rcls)``:

    - ``<rtype>.id HAS "<rtype>-<sid>"`` filters over the field's foreign-key
      (or child-element) column against default-minted
      ``"<entry type>-<sid>"`` ids, as
      :class:`~httk.store.backend.sql.entry_provider.StoreEntryProvider` mints them;
      ids of any other format match nothing. Custom ``id_of`` minting is out
      of scope here — with a custom minting scheme, supply your own
      ``'<rtype>.id'`` entry via ``extra_handlers``.
    - Depth-1 related-property filters (``<rtype>._httk_doi CONTAINS "10.1"``,
      ``<rtype>.id != "..."``, ``<rtype>._httk_year IS KNOWN``, ...) resolve by
      a two-phase semi-join: a nested ``optimade_filter_searcher`` over
      ``rcls`` (with the same ``prefix``; no further relationship nesting)
      collects the matching related sids, which are then matched as
      ``<rtype>.id HAS ANY ...``. Each dotted filter node resolves
      independently — see :func:`~httk.store.query.optimade_filters.translate_filter_ast`.

    ``extra_handlers`` entries are merged over the derived handler table last
    (so they can also override derived handlers); extra property names that are
    otherwise unknown are recognized with fulltype ``"unknown"`` (filter
    constants pass through unconverted).

    :param store: The SQL store containing the rows to query.
    :param cls: The storable class whose rows are searched.
    :param filter_string: The OPTIMADE filter string or parsed filter tree.
    :param prefix: The registered prefix used for served property names.
    :param definition: An optional entry definition supplying additional property types.
    :param extra_handlers: Optional handlers for ids, types, or additional properties.
    :param related_classes: Related entry types and their storable classes.
    :return: A searcher yielding the matching stored instances.
    :raises ~httk.store.query.optimade_filters.FilterTranslationError: When the filter cannot be translated.
    :raises ~httk.core.optimade.ParserSyntaxError: When a filter string does not parse.
    :raises ValueError: When a ``related_classes`` entry does not match exactly one
        reference or child-of-storable field of ``cls``.
    """
    schema = resolve_schema(cls)
    served = served_specs(schema, prefix)
    property_fulltypes = {"id": "string", "type": "string"}
    property_fulltypes.update({name: fulltype for name, _spec, fulltype in served})
    property_keys = {name: spec.field for name, spec, _fulltype in served}
    if store.store_timestamps:
        property_fulltypes[f"{prefix}store_timestamp"] = "integer"
        property_keys[f"{prefix}store_timestamp"] = "store_timestamp"
    # The store-managed lineage id is unconditional: every parent row carries a
    # ``logical_id`` regardless of the timestamp option, and it needs no unit
    # scaling (a plain integer resolved through ``SqlVariable.logical_id``).
    property_fulltypes[f"{prefix}logical_id"] = "integer"
    property_keys[f"{prefix}logical_id"] = "logical_id"
    if definition is not None:
        for name, prop in definition.properties.items():
            if name in ("id", "type"):
                continue
            property_fulltypes[name] = definition_fulltype(prop)

    handlers = simple_property_handlers(cls.__name__, property_keys, property_fulltypes)
    # The default id/type handlers of simple_property_handlers query the
    # serving-layer '__id' column and constant entry-type name; neither exists
    # on an ordinary store row, so filtering needs explicit extra_handlers.
    del handlers["id"]
    del handlers["type"]
    entry_type = cls.__name__

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
            nested._outputs.clear()
            nested.output(sid_column, "sid")
            return tuple(f"{related_type}-{int(values[0])}" for values, _names in nested)

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
        entry_type=entry_type,
        property_fulltypes=property_fulltypes,
        handlers=handlers,
        recognized_prefixes=(prefix,),
        relationship_targets=relationship_targets,
        related_property_resolver=resolver,
    )
