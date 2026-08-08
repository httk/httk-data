"""Exact three-valued evaluation for Mongo stored-property predicates.

The Mongo query path deliberately evaluates this small, frozen AST over
hydrated records.  That is important for child records: document storage keeps
their content-addressed SID links, while ``MongoStore.fetch`` gives the
evaluator their exact Python values.
"""

import dataclasses
import datetime
import fractions
import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from httk.core.storage import QueryLiteralError, content_id

from httk.data.db.codecs import ValueCodec, codec_named
from httk.data.db.schema import FieldSpec, TableSchema

__all__ = ["MongoPredicate", "MongoScope", "MongoValue", "canonical_predicate", "evaluate"]

_NO_LITERAL = object()


@dataclass(frozen=True, slots=True)
class MongoScope:
    """A root, child, reference, or filtered record-object scope."""

    identifier: int
    schema: TableSchema
    parent: "MongoScope | None" = None
    relationship: FieldSpec | None = None
    filter_predicate: "MongoPredicate | None" = None
    scalar_child: bool = False
    context: Any = dataclasses.field(default=None, compare=False, repr=False)

    def field(self, name: str) -> "MongoValue":
        """Select a scalar field from this scope."""
        return self.context._field(self, name)

    def scope(self, name: str) -> "MongoScope":
        """Follow one child or reference relationship from this scope."""
        return self.context._scope(self, name)


@dataclass(frozen=True, slots=True)
class MongoValue:
    """A field, aggregate, null, or literal value in the neutral AST."""

    kind: Literal["field", "present", "constant", "null", "count", "distinct_count"]
    scope: MongoScope | None = None
    field: str | None = None
    spec: FieldSpec | None = None
    literal: object = _NO_LITERAL
    value: "MongoValue | None" = None


@dataclass(frozen=True, slots=True)
class MongoPredicate:
    """A frozen three-valued predicate in the neutral stored-property AST."""

    kind: Literal["constant", "compare", "is_null", "exists", "and", "or", "not", "when_known", "scaled"]
    operands: tuple[object, ...] = ()

    def __and__(self, other: object) -> "MongoPredicate":
        return MongoPredicate("and", (self, _predicate(other)))

    def __or__(self, other: object) -> "MongoPredicate":
        return MongoPredicate("or", (self, _predicate(other)))

    def __invert__(self) -> "MongoPredicate":
        return MongoPredicate("not", (self,))


def _predicate(value: object) -> MongoPredicate:
    if not isinstance(value, MongoPredicate):
        raise TypeError("stored-property callback received a foreign predicate")
    return value


def evaluate(predicate: MongoPredicate, record: object) -> bool | None:
    """Evaluate ``predicate`` exactly over a hydrated backing object.

    :param predicate: Frozen predicate produced by ``_MongoQueryContext``.
    :param record: Hydrated backing record for the candidate SID.
    :return: ``True``, ``False``, or ``None`` for SQL UNKNOWN.
    """
    root = _root_scope(predicate)
    return _eval_predicate(predicate, {} if root is None else {root.identifier: record})


def _root_scope(predicate: MongoPredicate) -> MongoScope | None:
    stack: list[object] = [predicate]
    while stack:
        item = stack.pop()
        if isinstance(item, MongoScope) and item.parent is None:
            return item
        if isinstance(item, MongoScope) and item.parent is not None:
            stack.append(item.parent)
        if isinstance(item, MongoValue) and item.scope is not None:
            stack.append(item.scope)
        elif isinstance(item, MongoPredicate):
            stack.extend(item.operands)
        elif isinstance(item, tuple):
            stack.extend(item)
    return None


def _eval_predicate(predicate: MongoPredicate, environment: dict[int, object]) -> bool | None:
    kind = predicate.kind
    values = predicate.operands
    if kind == "constant":
        return values[0]  # type: ignore[return-value]
    if kind == "compare":
        left, operator, right = values
        left_value, right_value = _value(left), _value(right)
        return _compare(
            _eval_value(left_value, environment),
            left_value,
            str(operator),
            _eval_value(right_value, environment),
            right_value,
        )
    if kind == "is_null":
        return _eval_value(_value(values[0]), environment) is None
    if kind == "exists":
        scope, nested = _scope(values[0]), _predicate(values[1])
        for item in _items(scope, environment):
            result = _eval_predicate(nested, environment | {scope.identifier: item})
            if result is True:
                return True
        return False
    if kind == "and":
        left, right = (
            _eval_predicate(_predicate(values[0]), environment),
            _eval_predicate(_predicate(values[1]), environment),
        )
        return False if False in (left, right) else None if None in (left, right) else True
    if kind == "or":
        left, right = (
            _eval_predicate(_predicate(values[0]), environment),
            _eval_predicate(_predicate(values[1]), environment),
        )
        return True if True in (left, right) else None if None in (left, right) else False
    if kind == "not":
        value = _eval_predicate(_predicate(values[0]), environment)
        return None if value is None else not value
    if kind == "when_known":
        known = _eval_predicate(_predicate(values[0]), environment)
        return _eval_predicate(_predicate(values[1]), environment) if known is True else None
    assert kind == "scaled"
    left, left_factor, right, right_factor = (_eval_value(_value(item), environment) for item in values)
    if None in (left, left_factor, right, right_factor):
        return None
    try:
        return fractions.Fraction(cast(Any, left)) * fractions.Fraction(cast(Any, left_factor)) == fractions.Fraction(
            cast(Any, right)
        ) * fractions.Fraction(cast(Any, right_factor))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _items(scope: MongoScope, environment: dict[int, object]) -> tuple[object, ...]:
    if scope.identifier in environment:
        return (environment[scope.identifier],)
    if scope.parent is None or scope.relationship is None:
        return ()
    parents = _items(scope.parent, environment)
    values: list[object] = []
    for parent in parents:
        child = getattr(parent, scope.relationship.field, None)
        if child is None:
            continue
        if scope.relationship.role == "reference":
            values.append(child)
        else:
            values.extend(child)
    if scope.filter_predicate is not None:
        values = [
            item
            for item in values
            if _eval_predicate(scope.filter_predicate, environment | {scope.identifier: item}) is True
        ]
    return tuple(values)


def _eval_value(value: MongoValue, environment: dict[int, object]) -> object:
    if value.kind == "constant":
        return value.literal
    if value.kind == "null":
        return None
    if value.kind == "present":
        assert value.scope is not None and value.field is not None
        items = _items(value.scope, environment)
        return len(items) == 1 and getattr(items[0], value.field, None) is not None
    if value.kind == "count":
        assert value.scope is not None
        return len(_items(value.scope, environment))
    if value.kind == "distinct_count":
        assert value.scope is not None and value.value is not None
        found: set[object] = set()
        for item in _items(value.scope, environment):
            candidate = _eval_value(value.value, environment | {value.scope.identifier: item})
            if candidate is not None:
                found.add(_exact_key(value.value, candidate))
        return len(found)
    assert value.scope is not None and value.field is not None
    items = _items(value.scope, environment)
    if len(items) != 1:
        return None
    if value.scope.scalar_child:
        return items[0]
    if value.field.startswith("__content_id__"):
        return value.field.removeprefix("__content_id__") + content_id(items[0])
    return getattr(items[0], value.field, None)


def _compare(
    left: object,
    left_value: MongoValue,
    operator: str,
    right: object,
    right_value: MongoValue,
) -> bool | None:
    left_is_null, right_is_null = left_value.kind == "null", right_value.kind == "null"
    if operator == "=" or operator == "==":
        if left_is_null or right_is_null:
            return (right if left_is_null else left) is None
        if left is None or right is None:
            return None
        codec = _comparison_codec(left_value, right_value)
        return _exact_key(left_value, left, codec) == _exact_key(right_value, right, codec)
    if operator == "!=":
        if left_is_null or right_is_null:
            return (right if left_is_null else left) is not None
        if left is None or right is None:
            return None
        codec = _comparison_codec(left_value, right_value)
        return _exact_key(left_value, left, codec) != _exact_key(right_value, right, codec)
    if left is None or right is None:
        return None
    try:
        if operator == "<":
            return left < right  # type: ignore[operator]
        if operator == "<=":
            return left <= right  # type: ignore[operator]
        if operator == ">":
            return left > right  # type: ignore[operator]
        if operator == ">=":
            return left >= right  # type: ignore[operator]
        if operator in {"CONTAINS", "STARTS", "ENDS"}:
            if not isinstance(left, str) or not isinstance(right, str):
                return None
            return {"CONTAINS": right in left, "STARTS": left.startswith(right), "ENDS": left.endswith(right)}[operator]
    except TypeError:
        return None
    raise QueryLiteralError(f"unsupported stored-property comparison operator {operator!r}")


def _comparison_codec(left: MongoValue, right: MongoValue) -> ValueCodec | None:
    """Return the exact codec channel selected by either comparison operand."""
    return _exact_codec(left) or _exact_codec(right)


def _exact_key(value: MongoValue, raw: object, codec: ValueCodec | None = None) -> object:
    """Return a canonical equality/distinct key, using an exact codec channel."""
    selected = codec if codec is not None else _exact_codec(value)
    if selected is None:
        return raw
    try:
        exact_index = next(index for index, (suffix, _kind) in enumerate(selected.columns) if suffix == "_exact")
    except StopIteration:
        return raw
    try:
        return selected.encode(cast(Any, raw))[exact_index]
    except (TypeError, ValueError):
        return raw


def _exact_codec(value: MongoValue) -> ValueCodec | None:
    if value.spec is None or value.spec.codec_name is None:
        return None
    codec = codec_named(value.spec.codec_name)
    return codec if any(suffix == "_exact" for suffix, _kind in codec.columns) else None


def canonical_predicate(predicate: MongoPredicate) -> str:
    """Return canonical JSON used as the verifier identity and cursor input."""
    return json.dumps(_canonical(predicate), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical(value: object) -> object:
    if isinstance(value, MongoPredicate):
        return {"predicate": value.kind, "operands": [_canonical(item) for item in value.operands]}
    if isinstance(value, MongoValue):
        return {
            "value": value.kind,
            "scope": None if value.scope is None else _canonical(value.scope),
            "field": value.field,
            "literal": _canonical_literal(value.literal),
            "nested": None if value.value is None else _canonical(value.value),
        }
    if isinstance(value, MongoScope):
        return {
            "scope": value.identifier,
            "class": f"{value.schema.cls.__module__}.{value.schema.cls.__qualname__}",
            "parent": None if value.parent is None else _canonical(value.parent),
            "relationship": None if value.relationship is None else value.relationship.field,
            "filter": None if value.filter_predicate is None else _canonical(value.filter_predicate),
            "scalar_child": value.scalar_child,
        }
    return _canonical_literal(value)


def _canonical_literal(value: object) -> object:
    if value is _NO_LITERAL:
        return "<no-literal>"
    if isinstance(value, fractions.Fraction):
        return {"fraction": f"{value.numerator}/{value.denominator}"}
    if isinstance(value, datetime.datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, tuple):
        return [_canonical_literal(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return {"repr": repr(value), "type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _scope(value: object) -> MongoScope:
    if not isinstance(value, MongoScope):
        raise TypeError("stored-property callback received a foreign scope")
    return value


def _value(value: object) -> MongoValue:
    if not isinstance(value, MongoValue):
        raise TypeError("stored-property callback received a foreign value")
    return value
