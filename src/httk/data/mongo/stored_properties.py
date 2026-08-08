"""Stored-property plans and exact query contexts for MongoDB.

The context builds the frozen neutral AST in :mod:`httk.data.mongo.evaluator`.
MongoDB supplies a conservative candidate stream and the existing verified
iterator evaluates that AST over hydrated backing records.
"""

import dataclasses
import datetime
import decimal
import fractions
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

from httk.core import (
    EntryTypeDefinition,
    FracVector,
    PropertyDefinition,
    known_definition_prefixes,
    load_entry_type_definition,
)
from httk.core.optimade import FilterAst, parse_optimade_filter
from httk.core.register import entry_family_info
from httk.core.storage import QueryLiteralError, StoredPropertyProjection, content_id, stored_property_projections

from httk.data.db.schema import FieldSpec, SchemaError, resolve_schema
from httk.data.query import SearchResult
from httk.data.query.optimade_filters import FilterTranslationError, HandlerTable, translate_filter_ast

from .evaluator import MongoPredicate, MongoScope, MongoValue, canonical_predicate, evaluate
from .searcher import MongoField, MongoSearcher, MongoVariable

__all__ = [
    "MongoStoredPropertyCandidateStream",
    "MongoStoredPropertyConfigurationError",
    "MongoStoredPropertyPlan",
    "stored_property_mongo_plan",
]

_CORE_PROPERTIES = frozenset(("id", "type"))
_RFC3339_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})\Z", re.IGNORECASE
)


class MongoStoredPropertyConfigurationError(ValueError):
    """A Mongo entry family cannot realize its stored-property declaration."""


class _MongoQueryContext:
    """Mongo implementation of httk-core's neutral ``QueryContext`` protocol."""

    def __init__(self, backing: type) -> None:
        self._next_scope = 0
        self._root = self._new_scope(resolve_schema(backing))

    def _new_scope(
        self,
        schema: Any,
        parent: MongoScope | None = None,
        relationship: FieldSpec | None = None,
        *,
        scalar_child: bool = False,
    ) -> MongoScope:
        result = MongoScope(self._next_scope, schema, parent, relationship, scalar_child=scalar_child, context=self)
        self._next_scope += 1
        return result

    def field(self, name: str) -> MongoValue:
        return self._field(self._root, name)

    def scope(self, name: str) -> MongoScope:
        return self._scope(self._root, name)

    def constant(self, value: object) -> MongoValue:
        return MongoValue("constant", literal=value)

    def null(self) -> MongoValue:
        return MongoValue("null")

    def always_true(self) -> MongoPredicate:
        return MongoPredicate("constant", (True,))

    def always_false(self) -> MongoPredicate:
        return MongoPredicate("constant", (False,))

    def compare(self, left: MongoValue, operator: str, right: MongoValue) -> MongoPredicate:
        left, right = self._coerce_literals(_value(left), _value(right))
        if operator == "=":
            return self.equal(left, right)
        if operator == "!=":
            return MongoPredicate("compare", (left, operator, right))
        if operator not in {"<", "<=", ">", ">=", "CONTAINS", "STARTS", "ENDS"}:
            raise MongoStoredPropertyConfigurationError(f"unsupported stored-property comparison operator {operator!r}")
        return MongoPredicate("compare", (left, operator, right))

    def equal(self, left: MongoValue, right: MongoValue) -> MongoPredicate:
        left, right = self._coerce_literals(_value(left), _value(right))
        return MongoPredicate("compare", (left, "=", right))

    def exact_equal(self, left: MongoValue, right: MongoValue) -> MongoPredicate:
        left, right = self._coerce_literals(_value(left), _value(right))
        return MongoPredicate("compare", (left, "=", right))

    def is_null(self, value: MongoValue) -> MongoPredicate:
        return MongoPredicate("is_null", (_value(value),))

    def exists(self, scope: MongoScope, predicate: MongoPredicate) -> MongoPredicate:
        return MongoPredicate("exists", (_scope(scope), _predicate(predicate)))

    def filtered(self, scope: MongoScope, predicate: MongoPredicate) -> MongoScope:
        target = _scope(scope)
        return replace(target, filter_predicate=_predicate(predicate))

    def count(self, scope: MongoScope) -> MongoValue:
        return MongoValue("count", scope=_scope(scope))

    def distinct_count(self, scope: MongoScope, value: MongoValue) -> MongoValue:
        target, selected = _scope(scope), _value(value)
        if selected.scope is not target:
            raise MongoStoredPropertyConfigurationError("distinct_count value must belong to its scope")
        return MongoValue("distinct_count", scope=target, value=selected)

    def scaled_exact_equal(
        self, left: MongoValue, left_factor: MongoValue, right: MongoValue, right_factor: MongoValue
    ) -> MongoPredicate:
        return MongoPredicate("scaled", tuple(_value(item) for item in (left, left_factor, right, right_factor)))

    def and_(self, *predicates: MongoPredicate) -> MongoPredicate:
        result = self.always_true()
        for predicate in predicates:
            result &= _predicate(predicate)
        return result

    def or_(self, *predicates: MongoPredicate) -> MongoPredicate:
        result = self.always_false()
        for predicate in predicates:
            result |= _predicate(predicate)
        return result

    def not_(self, predicate: MongoPredicate) -> MongoPredicate:
        return ~_predicate(predicate)

    def when_known(self, known: MongoPredicate, predicate: MongoPredicate) -> MongoPredicate:
        return MongoPredicate("when_known", (_predicate(known), _predicate(predicate)))

    def _field(self, scope: MongoScope, name: str) -> MongoValue:
        if name.startswith("__content_id__"):
            return MongoValue("field", scope=scope, field=name)
        if scope.scalar_child:
            if scope.relationship is None or name not in {"value", scope.relationship.field}:
                raise MongoStoredPropertyConfigurationError("scalar child scopes use field('value')")
            return MongoValue("field", scope=scope, field=name, spec=scope.relationship)
        if name.endswith("_present"):
            try:
                child_spec = scope.schema.field(name.removesuffix("_present"))
            except SchemaError:
                child_spec = None
            if child_spec is not None and child_spec.role == "child" and child_spec.optional:
                return MongoValue("present", scope=scope, field=child_spec.field)
        try:
            spec = scope.schema.field(name)
        except SchemaError as error:
            raise MongoStoredPropertyConfigurationError(str(error)) from error
        if spec.role not in {"scalar", "encoded"}:
            raise MongoStoredPropertyConfigurationError(
                f"{scope.schema.cls.__name__}.{name} is not a scalar query field"
            )
        return MongoValue("field", scope=scope, field=name, spec=spec)

    def _scope(self, parent: MongoScope, name: str) -> MongoScope:
        try:
            spec = parent.schema.field(name)
        except SchemaError as error:
            raise MongoStoredPropertyConfigurationError(str(error)) from error
        if spec.role == "reference":
            assert spec.target is not None
            return self._new_scope(resolve_schema(spec.target), parent, spec)
        if spec.role != "child" or spec.child is None:
            raise MongoStoredPropertyConfigurationError(
                f"{parent.schema.cls.__name__}.{name} is not a child or reference scope"
            )
        return self._new_scope(
            resolve_schema(spec.target) if spec.target is not None else parent.schema,
            parent,
            spec,
            scalar_child=spec.target is None,
        )

    @staticmethod
    def _coerce_literals(left: MongoValue, right: MongoValue) -> tuple[MongoValue, MongoValue]:
        if left.spec is not None and right.kind == "constant":
            right = replace(right, literal=_literal_for(left.spec, right.literal))
        elif right.spec is not None and left.kind == "constant":
            left = replace(left, literal=_literal_for(right.spec, left.literal))
        return left, right


@dataclass(frozen=True, slots=True)
class _BackingPlan:
    backing: type
    projections: Mapping[str, StoredPropertyProjection]


@dataclass(frozen=True, slots=True)
class MongoStoredPropertyCandidateStream:
    """An ID-only Mongo candidate stream for a configured concrete backing."""

    backing: type
    backing_name: str
    searcher: Any
    sort_count: int


class _ConstantSortSearcher:
    """Inject family-constant sort values into a Mongo candidate projection.

    MongoDB need not sort on a constant ``type`` value, but federation's
    merge contract consumes sort values positionally.  This adapter restores
    that value in exactly the requested position while delegating query
    execution and limits to the real Mongo searcher.
    """

    def __init__(self, searcher: MongoSearcher, sort: Sequence[tuple[str, bool]], entry_type: str) -> None:
        self._searcher = searcher
        self._sort = tuple(sort)
        self._entry_type = entry_type

    def set_limit(self, limit: int) -> None:
        self._searcher.set_limit(limit)

    def __iter__(self) -> Iterator[SearchResult]:
        names = ("sid", "content_id", *(f"sort_{index}" for index in range(len(self._sort))))
        for result in self._searcher:
            values = iter(result.values[2:])
            sort_values = tuple(
                self._entry_type if name == "type" else next(values) for name, _descending in self._sort
            )
            yield SearchResult((result.values[0], result.values[1], *sort_values), names)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._searcher, name)


class MongoStoredPropertyPlan:
    """Stored-property responses and verified Mongo candidate plans for one family."""

    def __init__(
        self,
        store: Any,
        family: type,
        layout: Any,
        entry_type: str,
        definition: EntryTypeDefinition,
        backings: tuple[_BackingPlan, ...],
    ) -> None:
        self.store, self.family, self.layout, self.entry_type, self.definition, self._backings = (
            store,
            family,
            layout,
            entry_type,
            definition,
            backings,
        )

    @property
    def backings(self) -> tuple[type, ...]:
        return tuple(item.backing for item in self._backings)

    def records(self) -> Iterator[Mapping[str, Any]]:
        for backing in self._backings:
            searcher = self.store.searcher()
            variable = searcher.variable(backing.backing)
            searcher.output(variable, "record")
            for result in searcher:
                yield self.response_row(backing.backing, result[0][0])

    def filter_searchers(
        self, filter_string: str | FilterAst, *, sort: Sequence[tuple[str, bool]] = (), public_id_prefix: str = ""
    ) -> tuple[MongoSearcher, ...]:
        ast = parse_optimade_filter(filter_string) if isinstance(filter_string, str) else filter_string
        return tuple(
            self._searcher_for(item, ast, sort, public_id_prefix, candidate=False)[0] for item in self._backings
        )

    def candidate_searchers(
        self,
        filter_string: str | FilterAst | None = None,
        *,
        sort: Sequence[tuple[str, bool]] = (),
        public_id_prefix: str = "",
    ) -> tuple[MongoStoredPropertyCandidateStream, ...]:
        ast = parse_optimade_filter(filter_string) if isinstance(filter_string, str) else filter_string
        streams: list[MongoStoredPropertyCandidateStream] = []
        for backing, name in zip(self._backings, self.layout.record_names, strict=True):
            searcher, variable, sorts = self._searcher_for(backing, ast, sort, public_id_prefix, candidate=True)
            searcher.output(variable.sid, "sid")
            # ``content_id`` remains canonical: StoredEntryFederation applies
            # its source prefix when it turns a candidate into a public id.
            searcher.output(self._public_id_field(variable, ""), "content_id")
            for index, value in enumerate(sorts):
                searcher.output(value, f"sort_{index}")
            candidate_searcher: Any = (
                _ConstantSortSearcher(searcher, sort, self.entry_type)
                if any(sort_name == "type" for sort_name, _descending in sort)
                else searcher
            )
            streams.append(MongoStoredPropertyCandidateStream(backing.backing, name, candidate_searcher, len(sort)))
        return tuple(streams)

    def response_row(self, backing: type, record: object, *, public_id: str | None = None) -> Mapping[str, Any]:
        configured = next((item for item in self._backings if item.backing is backing), None)
        if configured is None:
            raise MongoStoredPropertyConfigurationError(
                f"{backing.__name__} is not a configured backing for {self.family.__name__}"
            )
        result: dict[str, Any] = {"id": content_id(record) if public_id is None else public_id, "type": self.entry_type}
        for name in self.definition.properties:
            if name not in _CORE_PROPERTIES:
                projection = configured.projections.get(name)
                result[name] = None if projection is None else _response_json_value(projection.response(record))
        return result

    def _searcher_for(
        self,
        backing: _BackingPlan,
        ast: FilterAst | None,
        sort: Sequence[tuple[str, bool]],
        public_id_prefix: str,
        *,
        candidate: bool,
    ) -> tuple[MongoSearcher, MongoVariable, tuple[MongoField, ...]]:
        searcher = self.store.searcher()
        variable = searcher.variable(backing.backing)
        context = _MongoQueryContext(backing.backing)
        if ast is not None:
            try:
                predicate = translate_filter_ast(
                    ast,
                    cast(Any, variable),
                    _property_fulltypes(self.definition),
                    self._handlers(backing, context, public_id_prefix),
                    known_definition_prefixes(),
                )
            except QueryLiteralError as error:
                raise FilterTranslationError(str(error), "type-mismatch") from error
            if not isinstance(predicate, MongoPredicate):
                raise MongoStoredPropertyConfigurationError("stored-property filter produced a foreign expression")
            # The candidate query is intentionally unrestrictive unless a
            # renderer can prove a necessary condition.  Verification is the
            # authority; this cannot drop exact or UNKNOWN-sensitive matches.
            searcher.add(variable.always_true())
            identity = canonical_predicate(predicate)
            searcher.set_row_verifier(
                lambda document, p=predicate, cls=backing.backing: (
                    evaluate(p, self.store.fetch(cls, int(document["_id"]))) is True
                ),
                identity,
            )
        elif not candidate:
            searcher.add(variable.always_true())
        sort_fields: list[MongoField] = []
        for name, descending in sort:
            if name == "type":
                continue
            field = self._sort_field(backing, variable, name, public_id_prefix)
            searcher.add_sort(field, descending)
            sort_fields.append(field)
        if not candidate:
            searcher.output(variable, "record")
        return searcher, variable, tuple(sort_fields)

    def _handlers(self, backing: _BackingPlan, context: _MongoQueryContext, prefix: str) -> HandlerTable:
        handlers: dict[str, Mapping[str, Callable[..., Any]]] = {
            "id": _id_handlers(context, prefix),
            "type": _type_handlers(context, self.entry_type),
        }
        for name, definition in self.definition.properties.items():
            if name in _CORE_PROPERTIES:
                continue
            projection = backing.projections.get(name)
            if projection is None:
                assert definition.nullable
                handlers[name] = _null_handlers(context)
            elif projection.query is not None:
                handlers[name] = _projection_handlers(projection, context)
        return handlers

    def _sort_field(self, backing: _BackingPlan, variable: MongoVariable, name: str, prefix: str) -> MongoField:
        if name == "id":
            return self._public_id_field(variable, prefix)
        projection = backing.projections.get(name)
        if projection is None or projection.sort is None:
            raise MongoStoredPropertyConfigurationError(
                f"{backing.backing.__name__} has no sortable projection for {name!r}"
            )
        value = projection.sort(cast(Any, _MongoFieldSortContext(variable)))
        if not isinstance(value, MongoField):
            raise MongoStoredPropertyConfigurationError(
                "stored-property sort callback must return a direct Mongo field"
            )
        if (
            value._codec is not None
            and value._codec.name != "float"
            and any(suffix == "_exact" for suffix, _kind in value._codec.columns)
        ):
            raise MongoStoredPropertyConfigurationError(
                f"{backing.backing.__name__}.{name} cannot sort an exact value through its canonical text channel"
            )
        return value

    @staticmethod
    def _public_id_field(variable: MongoVariable, prefix: str) -> MongoField:
        """Return content identity with its source-specific presentation prefix.

        Mongo sorts the physical ``content_id`` path.  Prefixing every value
        with one constant preserves that ordering, while the field's scalar
        projection exposes the prefixed public value to federation streams.
        """
        return MongoField(
            variable,
            "content_id",
            FieldSpec("content_id", str, "scalar", ()),
            presentation_prefix=prefix,
        )


class _MongoFieldSortContext:
    """Small sort-only context returning native root fields."""

    def __init__(self, variable: MongoVariable) -> None:
        self._variable = variable

    def field(self, name: str) -> MongoField:
        return getattr(self._variable, name)


def stored_property_mongo_plan(store: Any, family: type) -> MongoStoredPropertyPlan:
    layout = next((item for item in store.entry_layout if item.family is family), None)
    if layout is None:
        raise MongoStoredPropertyConfigurationError(
            f"entry family {getattr(family, '__name__', family)!r} is not configured in this MongoStore"
        )
    entry_type = getattr(family, "type", None)
    if not isinstance(entry_type, str) or not entry_type or entry_type != entry_type.strip():
        raise MongoStoredPropertyConfigurationError(f"{family.__name__}.type must be a non-empty stripped entry type")
    definition_id = getattr(family, "definition_id", entry_family_info(layout.name)[1])
    if definition_id != entry_family_info(layout.name)[1]:
        raise MongoStoredPropertyConfigurationError(
            f"{family.__name__}.definition_id does not match the registered family definition id"
        )
    factory = getattr(family, "entry_type_definition", None)
    definition = factory() if callable(factory) else load_entry_type_definition(definition_id)
    if (
        not isinstance(definition, EntryTypeDefinition)
        or (definition.definition_id or definition.extends_id) != definition_id
        or definition.name != entry_type
    ):
        raise MongoStoredPropertyConfigurationError(f"{family.__name__} has an inconsistent entry definition")
    plans: list[_BackingPlan] = []
    names = set(definition.properties)
    for backing in layout.records:
        projections = stored_property_projections(backing)
        reserved, unknown = sorted(set(_CORE_PROPERTIES) & set(projections)), sorted(set(projections) - names)
        if reserved:
            raise MongoStoredPropertyConfigurationError(
                f"{backing.__name__} must not declare intrinsic properties: {', '.join(reserved)}"
            )
        if unknown:
            raise MongoStoredPropertyConfigurationError(
                f"{backing.__name__} projects properties absent from {definition_id!r}: {', '.join(unknown)}"
            )
        required = [
            name
            for name, item in definition.properties.items()
            if name not in _CORE_PROPERTIES and not item.nullable and name not in projections
        ]
        if required:
            raise MongoStoredPropertyConfigurationError(
                f"{backing.__name__} has no response mapping for non-null property/properties: {', '.join(required)}"
            )
        plans.append(_BackingPlan(backing, projections))
    return MongoStoredPropertyPlan(store, family, layout, entry_type, definition, tuple(plans))


def _projection_handlers(
    projection: StoredPropertyProjection, context: _MongoQueryContext
) -> Mapping[str, Callable[..., Any]]:
    query = projection.query
    assert query is not None

    def invoke(operator: str, value: object) -> MongoPredicate:
        result = query(cast(Any, context), operator, value)
        if not isinstance(result, MongoPredicate):
            raise MongoStoredPropertyConfigurationError("stored-property query callback returned a foreign expression")
        return result

    return {
        "comparison": lambda _e, op, value, _v: invoke(op, value),
        "stringmatching": lambda _e, value, op, _v: invoke(op, value),
        "HAS": lambda _e, _o, values, _v, op: invoke(op, tuple(values)),
        "length": lambda _e, op, value, _v: invoke(f"LENGTH {op}", value),
        "unknown": lambda _e, _v, op: invoke(op, None),
    }


def _null_handlers(context: _MongoQueryContext) -> Mapping[str, Callable[..., Any]]:
    unknown = lambda: MongoPredicate("constant", (None,))
    return {
        "comparison": lambda *_: unknown(),
        "stringmatching": lambda *_: unknown(),
        "HAS": lambda *_: unknown(),
        "length": lambda *_: unknown(),
        "unknown": lambda _e, _v, op: context.always_true() if op == "IS_UNKNOWN" else context.always_false(),
    }


def _id_handlers(context: _MongoQueryContext, prefix: str) -> Mapping[str, Callable[..., Any]]:
    value = context._field(context._root, "__content_id__" + prefix)
    return {
        "comparison": lambda _e, op, literal, _v: context.compare(value, op, context.constant(literal)),
        "stringmatching": lambda _e, literal, op, _v: context.compare(value, op, context.constant(literal)),
        "unknown": lambda _e, _v, op: context.always_false() if op == "IS_UNKNOWN" else context.always_true(),
    }


def _type_handlers(context: _MongoQueryContext, entry_type: str) -> Mapping[str, Callable[..., Any]]:
    def compare(_e: object, op: str, literal: object, _v: object) -> MongoPredicate:
        if not isinstance(literal, str):
            raise QueryLiteralError("type comparison needs a string literal")
        result = {
            "=": entry_type == literal,
            "!=": entry_type != literal,
            "<": entry_type < literal,
            "<=": entry_type <= literal,
            ">": entry_type > literal,
            ">=": entry_type >= literal,
        }[op]
        return context.always_true() if result else context.always_false()

    return {
        "comparison": compare,
        "stringmatching": lambda _e, literal, op, _v: (
            context.always_true()
            if {
                "CONTAINS": str(literal) in entry_type,
                "STARTS": entry_type.startswith(str(literal)),
                "ENDS": entry_type.endswith(str(literal)),
            }[op]
            else context.always_false()
        ),
        "unknown": lambda _e, _v, op: context.always_false() if op == "IS_UNKNOWN" else context.always_true(),
    }


def _property_fulltypes(definition: EntryTypeDefinition) -> Mapping[str, str]:
    return MappingProxyType({name: _definition_fulltype(item) for name, item in definition.properties.items()})


def _definition_fulltype(definition: PropertyDefinition) -> str:
    document = definition.as_optimade()
    value = document["x-optimade-type"]
    return (
        "list of " + _fulltype_from_document(cast(Mapping[str, Any], document["items"]))
        if value == "list"
        else cast(str, value)
    )


def _fulltype_from_document(document: Mapping[str, Any]) -> str:
    value = document["x-optimade-type"]
    return (
        "list of " + _fulltype_from_document(cast(Mapping[str, Any], document["items"]))
        if value == "list"
        else cast(str, value)
    )


def _literal_for(spec: FieldSpec, value: object) -> object:
    if spec.codec_name == "datetime" and isinstance(value, str):
        if _RFC3339_TIMESTAMP.fullmatch(value) is None:
            raise QueryLiteralError("timestamp property requires an RFC 3339 literal")
        normalized = value.replace("t", "T", 1)
        normalized = normalized[:-1] + "+00:00" if normalized.endswith(("Z", "z")) else normalized
        try:
            parsed = datetime.datetime.fromisoformat(normalized)
        except ValueError as error:
            raise QueryLiteralError("timestamp property requires an RFC 3339 literal") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise QueryLiteralError("timestamp property requires an RFC 3339 UTC offset")
        return parsed
    if spec.codec_name == "float" and isinstance(value, (int, float)):
        return float(value)
    return value


def _response_json_value(value: object) -> Any:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, fractions.Fraction | decimal.Decimal):
        return float(value)
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("stored-property response timestamps must be timezone-aware")
        return value.astimezone(datetime.UTC).isoformat()
    if isinstance(value, FracVector):
        return [] if value.dim in ((), (0,)) else _response_json_value(value.to_fractions())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _response_json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("stored-property response dictionaries must have string keys")
        return {key: _response_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_response_json_value(item) for item in value]
    to_float = getattr(value, "to_float", None)
    if callable(to_float):
        return float(cast(Any, to_float)())
    raise TypeError(f"stored-property response cannot serialize {type(value).__name__}")


def _scope(value: object) -> MongoScope:
    if not isinstance(value, MongoScope):
        raise MongoStoredPropertyConfigurationError("stored-property callback received a foreign scope")
    return value


def _value(value: object) -> MongoValue:
    if not isinstance(value, MongoValue):
        raise MongoStoredPropertyConfigurationError("stored-property callback received a foreign value")
    return value


def _predicate(value: object) -> MongoPredicate:
    if not isinstance(value, MongoPredicate):
        raise MongoStoredPropertyConfigurationError("stored-property callback received a foreign predicate")
    return value
