"""Serve OPTIMADE filters over stored dataclasses through MongoDB.

The property and filter translation layers are backend-neutral. This module
only supplies Mongo-specific field handlers for stored properties and for
relationship ids, plus the semi-join that resolves dotted relationship
filters through a nested Mongo searcher.
"""

from collections.abc import Callable, Mapping
from typing import Any

from httk.core import EntryTypeDefinition
from httk.core.optimade import FilterAst

from httk.store.backend.mongo.searcher import MongoExpression, MongoField, MongoReference, MongoSearcher, MongoVariable
from httk.store.backend.mongo.store import MongoStore
from httk.store.backend.schema import resolve_schema
from httk.store.query import Searcher
from httk.store.query.optimade_filters import (
    FilterTranslationError,
    filter_searcher,
    known_unknown_handler,
    simple_property_handlers,
)
from httk.store.served_specs import definition_fulltype, served_specs

__all__ = ["optimade_filter_searcher"]


def _related_sid(related_type: str, value: Any) -> int:
    """Parse a default-minted ``'<related_type>-<sid>'`` id into its SID."""
    if isinstance(value, str) and value.startswith(related_type + "-"):
        tail = value[len(related_type) + 1 :]
        if tail.isdigit():
            return int(tail)
    return -1


def _related_id_has_handlers(related_type: str, field: str, role: str) -> Mapping[str, Callable[..., Any]]:
    """Build the ``'<related_type>.id'`` handler over a reference or child SID field."""

    def has_handler(
        entry: str, ops: Any, values: Any, search_variable: MongoVariable, has_type: str
    ) -> MongoExpression:
        sids = tuple(_related_sid(related_type, value) for value in values)
        relation = getattr(search_variable, field)
        query_field = relation._field if isinstance(relation, MongoReference) else relation
        if not isinstance(query_field, MongoField):
            raise FilterTranslationError("Relationship id field is not queryable.", "internal")
        if role == "reference":

            def member(sid: int) -> MongoExpression:
                # Preserve SQL's outer-join complement: a negated relationship
                # predicate includes rows whose nullable reference is absent.
                return query_field.is_in(sid) & ~(query_field == None)

            if has_type == "HAS_ALL":
                expression = member(sids[0])
                for sid in sids[1:]:
                    expression = expression & member(sid)
                return expression
            if has_type == "HAS_ANY":
                expression = member(sids[0])
                for sid in sids[1:]:
                    expression = expression | member(sid)
                return expression
            if has_type == "HAS_ONLY":
                # A reference is a set of zero or one elements: the empty
                # set satisfies HAS ONLY vacuously, as SqlReference.has_only
                # does. Mongo's nullable membership expression includes the
                # absent/null reference alongside the supplied SIDs.
                return query_field.is_in(None, *sids)
        elif role == "child":
            if has_type == "HAS_ALL":
                expression = query_field.has_any(sids[0])
                for sid in sids[1:]:
                    expression = expression & query_field.has_any(sid)
                return expression
            if has_type == "HAS_ANY":
                return query_field.has_any(*sids)
            if has_type == "HAS_ONLY":
                return query_field.has_only(*sids)
        raise FilterTranslationError("Unexpected set operator type: " + str(has_type), "internal")

    return {"HAS": has_handler}


def _own_id_handlers(related_type: str) -> Mapping[str, Callable[..., Any]]:
    """Build handlers for a related class's own ``id`` in a nested search."""

    def comparison(entry: str, op: str, value: Any, search_variable: MongoVariable) -> MongoExpression:
        if op not in ("=", "!="):
            raise FilterTranslationError("Ordering comparisons on relationship ids not implemented.", "not-implemented")
        sid = _related_sid(related_type, value)
        field = search_variable.sid
        return field == sid if op == "=" else field != sid

    return {"comparison": comparison, "unknown": known_unknown_handler}


def optimade_filter_searcher(
    store: MongoStore,
    cls: type,
    filter_string: str | FilterAst,
    *,
    prefix: str = "_httk_",
    definition: EntryTypeDefinition | None = None,
    extra_handlers: Mapping[str, Mapping[str, Callable[..., Any]]] | None = None,
    related_classes: Mapping[str, type] | None = None,
) -> Searcher:
    """Build a Mongo searcher over ``cls`` from an OPTIMADE filter.

    :param store: The Mongo store containing the rows to query.
    :param cls: The storable class whose rows are searched.
    :param filter_string: An OPTIMADE filter string or parsed filter tree.
    :param prefix: The registered prefix used for served property names.
    :param definition: An optional definition supplying additional property types.
    :param extra_handlers: Optional handlers for ids, types, or extra properties.
    :param related_classes: Related entry types and their storable classes.
    :return: A Mongo searcher yielding matching stored instances.
    :raises ~httk.store.query.optimade_filters.FilterTranslationError: If the filter cannot be translated.
    :raises ValueError: If a related class does not match exactly one relationship field.
    """
    schema = resolve_schema(cls)
    served = served_specs(schema, prefix)
    property_fulltypes = {"id": "string", "type": "string"}
    property_fulltypes.update({name: fulltype for name, _spec, fulltype in served})
    property_keys = {name: spec.field for name, spec, _fulltype in served}
    if definition is not None:
        for name, prop in definition.properties.items():
            if name not in ("id", "type"):
                property_fulltypes[name] = definition_fulltype(prop)

    handlers = simple_property_handlers(cls.__name__, property_keys, property_fulltypes)
    del handlers["id"]
    del handlers["type"]
    entry_type = cls.__name__

    relationship_targets: tuple[str, ...] = ()
    resolver: Callable[[str, FilterAst], tuple[str, ...]] | None = None
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
            handlers[f"{related_type}.id"] = _related_id_has_handlers(related_type, matching[0].field, matching[0].role)
        relationship_targets = tuple(related)

        def resolve_related(related_type: str, sub_ast: FilterAst) -> tuple[str, ...]:
            nested = optimade_filter_searcher(
                store,
                related[related_type],
                sub_ast,
                prefix=prefix,
                extra_handlers={"id": _own_id_handlers(related_type)},
            )
            assert isinstance(nested, MongoSearcher)
            result = nested.results(sid=nested._root_sid_field())
            return tuple(f"{related_type}-{int(sid)}" for sid in result.scalars("sid"))

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
