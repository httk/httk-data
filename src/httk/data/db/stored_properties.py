"""SQL plans for property-mapped durable entry backings.

This module translates the backend-neutral
:class:`httk.core.StoredPropertyProjection` callbacks declared by one concrete
record backing into SQLAlchemy predicates.  A logical entry family supplies its
entry type and OPTIMADE definition; every backing configured for that family in
the :class:`~httk.data.db.store.SqlStore` supplies only the properties it can
actually represent.

The plan is deliberately independent of serving.  It exposes concrete-record
responses and one SQL searcher per backing, leaving a later protocol adapter to
apply public ids, merge backing result streams, and construct envelopes.
"""

import dataclasses
import datetime
import decimal
import fractions
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

import sqlalchemy
from httk.core import (
    EntryTypeDefinition,
    FilterAst,
    FracVector,
    PropertyDefinition,
    QueryLiteralError,
    StoredPropertyProjection,
    content_id,
    entry_family_info,
    known_definition_prefixes,
    load_entry_type_schema,
    parse_optimade_filter,
    stored_property_projections,
)
from sqlalchemy.sql.elements import Null
from sqlalchemy.sql.selectable import Exists, ScalarSelect
from sqlalchemy.sql.visitors import replacement_traverse

from httk.data.db.codecs import ValueCodec, codec_named
from httk.data.db.mapping import CONTENT_ID_COLUMN, SID_COLUMN
from httk.data.db.rows import RowHydrator
from httk.data.db.schema import FieldSpec, SchemaError, TableSchema, resolve_schema
from httk.data.db.searcher import SqlColumn, SqlExpression, SqlSearcher, SqlVariable, _bool_clause
from httk.data.db.store import SqlStore
from httk.data.optimade_query import (
    FilterTranslationError,
    HandlerTable,
    constant_comparison_handler,
    constant_stringmatching_handler,
    translate_filter_ast,
)

__all__ = [
    "StoredPropertySqlCandidateStream",
    "StoredPropertySqlConfigurationError",
    "StoredPropertySqlPlan",
    "stored_property_sql_plan",
]


_CORE_PROPERTIES: Final[frozenset[str]] = frozenset(("id", "type"))
_EXACT_CODEC_NAMES: Final = frozenset(("float", "fraction", "fracscalar", "surdscalar"))
_NO_LITERAL: Final = object()
_RFC3339_TIMESTAMP: Final = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})\Z",
    re.IGNORECASE,
)


class StoredPropertySqlConfigurationError(ValueError):
    """A configured family/backing cannot realize its declared entry definition."""


@dataclass(frozen=True)
class _SqlValue:
    """A SQL scalar together with its optional canonical exact representation."""

    element: sqlalchemy.ColumnElement[Any]
    exact_element: sqlalchemy.ColumnElement[Any] | None = None
    codec: ValueCodec | None = None
    scope: "_SqlScope | None" = None
    literal: object = _NO_LITERAL

    @property
    def exact(self) -> sqlalchemy.ColumnElement[Any]:
        return self.exact_element if self.exact_element is not None else self.element


@dataclass(frozen=True)
class _SqlPredicate:
    """A correlated SQL predicate implementing core's ``QueryExpression`` protocol."""

    clause: sqlalchemy.ColumnElement[bool]

    # ``SqlSearcher`` deliberately owns execution, but its expression shape
    # is simple enough for this correlated-query implementation to satisfy.
    # Stored-property predicates never need its child-join/HAVING machinery:
    # scopes are represented by correlated subqueries instead.
    where_clause: sqlalchemy.ColumnElement[bool] = dataclasses.field(init=False)
    having_clause: sqlalchemy.ColumnElement[bool] = dataclasses.field(init=False)
    post: bool = dataclasses.field(init=False, default=False)
    set_derived: bool = dataclasses.field(init=False, default=False)
    group_columns: tuple[sqlalchemy.ColumnElement[Any], ...] = dataclasses.field(init=False, default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "where_clause", self.clause)
        object.__setattr__(self, "having_clause", self.clause)

    def __and__(self, other: object) -> "_SqlPredicate":
        return _SqlPredicate(_bool_clause(sqlalchemy.and_(self.clause, _predicate(other).clause)))

    def __or__(self, other: object) -> "_SqlPredicate":
        return _SqlPredicate(_bool_clause(sqlalchemy.or_(self.clause, _predicate(other).clause)))

    def __invert__(self) -> "_SqlPredicate":
        return _SqlPredicate(_bool_clause(sqlalchemy.not_(self.clause)))


@dataclass(frozen=True)
class _SqlScope:
    """One root, reference, or child relation plus its correlation conditions."""

    context: "_SqlQueryContext"
    schema: TableSchema
    alias: sqlalchemy.FromClause
    froms: tuple[sqlalchemy.FromClause, ...]
    ancestors: tuple[sqlalchemy.FromClause, ...]
    conditions: tuple[sqlalchemy.ColumnElement[bool], ...]
    scalar_child: FieldSpec | None = None
    singleton: bool = True

    def field(self, name: str) -> _SqlValue:
        if self.scalar_child is not None:
            if name not in {"value", self.scalar_child.field}:
                raise StoredPropertySqlConfigurationError(
                    f"{self.scalar_child.field!r} is a scalar child scope; use field('value')"
                )
            return self.context._child_scalar_value(self, self.scalar_child)
        try:
            spec = self.schema.field(name)
        except SchemaError as error:
            raise StoredPropertySqlConfigurationError(str(error)) from error
        if spec.role == "scalar":
            return self.context._scoped_scalar(
                self,
                _SqlValue(self.alias.c[spec.columns[0].name], scope=self),
            )
        if spec.role == "encoded":
            assert spec.codec_name is not None
            codec = codec_named(spec.codec_name)
            exact = self.alias.c.get(f"{spec.field}_exact") if spec.codec_name in _EXACT_CODEC_NAMES else None
            return self.context._scoped_scalar(
                self,
                _SqlValue(
                    self.alias.c[spec.field + codec.query_suffix],
                    exact_element=exact,
                    codec=codec,
                    scope=self,
                ),
            )
        raise StoredPropertySqlConfigurationError(
            f"{self.schema.cls.__name__}.{name} is {spec.role!r}; query a scalar through a compatible scope"
        )

    def scope(self, name: str) -> "_SqlScope":
        try:
            spec = self.schema.field(name)
        except SchemaError as error:
            raise StoredPropertySqlConfigurationError(str(error)) from error
        return self.context._related_scope(self, spec)


class _SqlQueryContext:
    """SQLAlchemy implementation of httk-core's property ``QueryContext`` protocol."""

    def __init__(self, searcher: SqlSearcher, variable: SqlVariable) -> None:
        self._searcher = searcher
        # The root alias belongs to the outer SqlSearcher SELECT, never to a
        # correlated child query's local FROM clause.
        self._root = _SqlScope(self, variable._schema, variable._alias, (), (), ())

    def field(self, name: str) -> _SqlValue:
        return self._root.field(name)

    def scope(self, name: str) -> _SqlScope:
        return self._root.scope(name)

    def constant(self, value: object) -> _SqlValue:
        if isinstance(value, fractions.Fraction):
            return _SqlValue(
                sqlalchemy.literal(value),
                sqlalchemy.literal(f"{value.numerator}/{value.denominator}"),
                literal=value,
            )
        return _SqlValue(sqlalchemy.literal(value), literal=value)

    def null(self) -> _SqlValue:
        return _SqlValue(sqlalchemy.null())

    def always_true(self) -> _SqlPredicate:
        return _SqlPredicate(sqlalchemy.true())

    def always_false(self) -> _SqlPredicate:
        return _SqlPredicate(sqlalchemy.false())

    def compare(self, left: _SqlValue, operator: str, right: _SqlValue) -> _SqlPredicate:
        left_value, right_value = _codec_literals(_value(left), _value(right))
        if operator == "=":
            return self.equal(left_value, right_value)
        if operator == "!=":
            return _SqlPredicate(sqlalchemy.not_(self.equal(left_value, right_value).clause))
        if (
            left_value.exact_element is not None or right_value.exact_element is not None
        ) and not _exact_ordering_uses_float_companion(left_value, right_value):
            raise QueryLiteralError("ordering an exact stored value is not implemented")
        if operator == "<":
            return _SqlPredicate(_bool_clause(left_value.element < right_value.element))
        if operator == "<=":
            return _SqlPredicate(_bool_clause(left_value.element <= right_value.element))
        if operator == ">":
            return _SqlPredicate(_bool_clause(left_value.element > right_value.element))
        if operator == ">=":
            return _SqlPredicate(_bool_clause(left_value.element >= right_value.element))
        if operator in {"CONTAINS", "STARTS", "ENDS"}:
            literal = getattr(right_value.element, "value", None)
            if not isinstance(literal, str):
                raise QueryLiteralError(f"{operator} needs a string literal")
            escaped = _escape_like(literal)
            pattern = {
                "CONTAINS": f"%{escaped}%",
                "STARTS": f"{escaped}%",
                "ENDS": f"%{escaped}",
            }[operator]
            return _SqlPredicate(_bool_clause(left_value.element.like(pattern, escape="\\")))
        raise StoredPropertySqlConfigurationError(f"unsupported stored-property comparison operator {operator!r}")

    def equal(self, left: _SqlValue, right: _SqlValue) -> _SqlPredicate:
        left_value, right_value = _codec_literals(_value(left), _value(right))
        if _is_null(left_value.element) or _is_null(right_value.element):
            value = right_value.element if _is_null(left_value.element) else left_value.element
            return _SqlPredicate(_bool_clause(value.is_(None)))
        if left_value.exact_element is not None or right_value.exact_element is not None:
            return self.exact_equal(left_value, right_value)
        return _SqlPredicate(_bool_clause(left_value.element == right_value.element))

    def exact_equal(self, left: _SqlValue, right: _SqlValue) -> _SqlPredicate:
        left_value, right_value = _codec_literals(_value(left), _value(right))
        if _is_null(left_value.element) or _is_null(right_value.element):
            value = right_value.element if _is_null(left_value.element) else left_value.element
            return _SqlPredicate(_bool_clause(value.is_(None)))
        return _SqlPredicate(_bool_clause(left_value.exact == right_value.exact))

    def is_null(self, value: _SqlValue) -> _SqlPredicate:
        return _SqlPredicate(_bool_clause(_value(value).element.is_(None)))

    def exists(self, scope: _SqlScope, predicate: _SqlPredicate) -> _SqlPredicate:
        target = _scope(scope)
        condition = _predicate(predicate)
        nested_condition = _correlate_nested(condition.clause, _scope_from(target))
        conditions = tuple(_correlate_nested(item, _scope_from(target)) for item in target.conditions)
        statement = (
            sqlalchemy.select(sqlalchemy.literal(1))
            .select_from(*_scope_from(target))
            .where(*conditions, nested_condition)
            .correlate(*target.ancestors)
        )
        return _SqlPredicate(_bool_clause(sqlalchemy.exists(statement)))

    def filtered(self, scope: _SqlScope, predicate: _SqlPredicate) -> _SqlScope:
        target = _scope(scope)
        return dataclasses.replace(target, conditions=(*target.conditions, _predicate(predicate).clause))

    def count(self, scope: _SqlScope) -> _SqlValue:
        target = _scope(scope)
        conditions = tuple(_correlate_nested(item, _scope_from(target)) for item in target.conditions)
        statement = (
            sqlalchemy.select(sqlalchemy.func.count())
            .select_from(*_scope_from(target))
            .where(*conditions)
            .correlate(*target.ancestors)
        )
        return _SqlValue(statement.scalar_subquery())

    def distinct_count(self, scope: _SqlScope, value: _SqlValue) -> _SqlValue:
        target = _scope(scope)
        selected = _value(value)
        if selected.scope is None or selected.scope.alias is not target.alias:
            raise StoredPropertySqlConfigurationError("distinct_count value must belong to its scope")
        conditions = tuple(_correlate_nested(item, _scope_from(target)) for item in target.conditions)
        statement = (
            sqlalchemy.select(sqlalchemy.func.count(sqlalchemy.distinct(selected.exact)))
            .select_from(*_scope_from(target))
            .where(*conditions)
            .correlate(*target.ancestors)
        )
        return _SqlValue(statement.scalar_subquery())

    def scaled_exact_equal(
        self,
        left: _SqlValue,
        left_factor: _SqlValue,
        right: _SqlValue,
        right_factor: _SqlValue,
    ) -> _SqlPredicate:
        return _SqlPredicate(
            _bool_clause(
                sqlalchemy.func.httk_fraction_scaled_equal(
                    _value(left).exact,
                    _value(left_factor).exact,
                    _value(right).exact,
                    _value(right_factor).exact,
                )
            )
        )

    def and_(self, *predicates: _SqlPredicate) -> _SqlPredicate:
        if not predicates:
            return self.always_true()
        return _SqlPredicate(_bool_clause(sqlalchemy.and_(*(_predicate(item).clause for item in predicates))))

    def or_(self, *predicates: _SqlPredicate) -> _SqlPredicate:
        if not predicates:
            return self.always_false()
        return _SqlPredicate(_bool_clause(sqlalchemy.or_(*(_predicate(item).clause for item in predicates))))

    def not_(self, predicate: _SqlPredicate) -> _SqlPredicate:
        return ~_predicate(predicate)

    def when_known(self, known: _SqlPredicate, predicate: _SqlPredicate) -> _SqlPredicate:
        """Preserve SQL's unknown value when a conditional backing fact is absent."""
        return _SqlPredicate(
            _bool_clause(
                sqlalchemy.case(
                    (_predicate(known).clause, _predicate(predicate).clause),
                    else_=sqlalchemy.null(),
                )
            )
        )

    def _scoped_scalar(self, scope: _SqlScope, value: _SqlValue) -> _SqlValue:
        """Read a scalar reference path through a correlated one-row subquery.

        Adding a related alias directly to an outer ``SqlSearcher`` predicate
        lets SQLAlchemy add it as an uncorrelated FROM term.  A reference path
        has at most one target row, so selecting its field through a correlated
        scalar subquery both avoids that cartesian product and retains SQL NULL
        when the optional reference is absent.  Child scopes are deliberately
        not scalarized: they are collections and must remain inside
        ``exists``/aggregate operations that own their local FROM tree.
        """
        if scope is self._root or not scope.singleton:
            return value
        froms = _scope_from(scope)
        conditions = tuple(_correlate_nested(item, froms) for item in scope.conditions)

        def select_scalar(element: sqlalchemy.ColumnElement[Any]) -> sqlalchemy.ColumnElement[Any]:
            statement = sqlalchemy.select(element).select_from(*froms).where(*conditions).correlate(*scope.ancestors)
            return statement.scalar_subquery()

        return _SqlValue(
            select_scalar(value.element),
            exact_element=None if value.exact_element is None else select_scalar(value.exact_element),
            codec=value.codec,
            literal=value.literal,
        )

    def _related_scope(self, parent: _SqlScope, spec: FieldSpec) -> _SqlScope:
        if spec.role == "reference":
            assert spec.target is not None
            schema = resolve_schema(spec.target)
            alias = self._searcher._store._table(schema.table_name).alias()
            condition = _bool_clause(parent.alias.c[spec.columns[0].name] == alias.c[SID_COLUMN])
            return _SqlScope(
                self,
                schema,
                alias,
                (*parent.froms, alias),
                _scope_ancestors(parent),
                (*parent.conditions, condition),
                singleton=parent.singleton,
            )
        if spec.role != "child" or spec.child is None:
            raise StoredPropertySqlConfigurationError(
                f"{parent.schema.cls.__name__}.{spec.field} is {spec.role!r}, not a reference or child scope"
            )
        alias = self._searcher._store._table(spec.child.table_name).alias()
        condition = _bool_clause(alias.c[f"{parent.schema.table_name}_sid"] == parent.alias.c[SID_COLUMN])
        if spec.target is not None:
            schema = resolve_schema(spec.target)
            target = self._searcher._store._table(schema.table_name).alias()
            target_condition = _bool_clause(alias.c[f"{spec.field}_sid"] == target.c[SID_COLUMN])
            # Keep aliases separately selectable so a nested scope can
            # correlate the parent aliases out of its inner SELECT. Every
            # local alias remains linked by an explicit predicate below.
            return _SqlScope(
                self,
                schema,
                target,
                (*parent.froms, alias, target),
                _scope_ancestors(parent),
                (*parent.conditions, condition, target_condition),
                singleton=False,
            )
        return _SqlScope(
            self,
            parent.schema,
            alias,
            (*parent.froms, alias),
            _scope_ancestors(parent),
            (*parent.conditions, condition),
            scalar_child=spec,
            singleton=False,
        )

    def _child_scalar_value(self, scope: _SqlScope, spec: FieldSpec) -> _SqlValue:
        assert spec.child is not None
        if spec.codec_name is None:
            return _SqlValue(scope.alias.c[spec.child.element_columns[0].name], scope=scope)
        codec = codec_named(spec.codec_name)
        exact = scope.alias.c.get(f"{spec.field}_exact") if spec.codec_name in _EXACT_CODEC_NAMES else None
        return _SqlValue(
            scope.alias.c[spec.field + codec.query_suffix],
            exact_element=exact,
            codec=codec,
            scope=scope,
        )


@dataclass(frozen=True)
class _BackingPlan:
    backing: type
    projections: Mapping[str, StoredPropertyProjection]


@dataclass(frozen=True)
class StoredPropertySqlCandidateStream:
    """One raw, SQL-bounded backing stream for a later federation merge.

    ``searcher`` outputs only ``sid``, canonical ``content_id``, and one raw
    SQL value per requested sort property.  Iterating it therefore never
    hydrates a record; a federation can select its final page before fetching
    any object graph.
    """

    backing: type
    backing_name: str
    searcher: SqlSearcher
    sort_count: int


class StoredPropertySqlPlan:
    """Validated responses and SQL queries for one configured logical entry family.

    The plan has no federation semantics: :meth:`filter_searchers` returns one
    independent searcher per configured concrete backing.  That explicit shape
    preserves backing-local property semantics for a future protocol adapter.
    """

    def __init__(
        self,
        store: SqlStore,
        family: type,
        layout: Any,
        entry_type: str,
        definition: EntryTypeDefinition,
        backings: tuple[_BackingPlan, ...],
    ) -> None:
        self.store = store
        self.family = family
        self.layout = layout
        self.entry_type = entry_type
        self.definition = definition
        self._backings = backings

    @property
    def backings(self) -> tuple[type, ...]:
        """The configured concrete record classes, in persisted backing order."""
        return tuple(item.backing for item in self._backings)

    def records(self) -> Iterator[Mapping[str, Any]]:
        """Yield protocol-boundary rows projected from concrete backing records."""
        for backing in self._backings:
            yield from self._records_for(backing)

    def filter_searchers(
        self,
        filter_string: str | FilterAst,
        *,
        sort: Sequence[tuple[str, bool]] = (),
        public_id_prefix: str = "",
    ) -> tuple[SqlSearcher, ...]:
        """Return one concrete-backing SQL searcher for an OPTIMADE filter and sort list."""
        ast = parse_optimade_filter(filter_string) if isinstance(filter_string, str) else filter_string
        return tuple(self._filter_searcher(backing, ast, sort, public_id_prefix) for backing in self._backings)

    def candidate_searchers(
        self,
        filter_string: str | FilterAst | None = None,
        *,
        sort: Sequence[tuple[str, bool]] = (),
        public_id_prefix: str = "",
    ) -> tuple[StoredPropertySqlCandidateStream, ...]:
        """Return ID-only concrete streams for a bounded federated page.

        ``None`` emits the query context's portable true predicate.  It never
        adds an ``ORDER BY`` unless a sort was explicitly requested.  The
        supplied public-id prefix participates in both the intrinsic id
        filter handlers and id sort expression.
        """
        ast = parse_optimade_filter(filter_string) if isinstance(filter_string, str) else filter_string
        streams: list[StoredPropertySqlCandidateStream] = []
        for backing, backing_name in zip(self._backings, self.layout.backing_names, strict=True):
            searcher, variable, sort_values = self._candidate_searcher(backing, ast, sort, public_id_prefix)
            searcher.output(SqlColumn(searcher, variable._alias.c[SID_COLUMN]), "sid")
            searcher.output(SqlColumn(searcher, variable._alias.c[CONTENT_ID_COLUMN]), "content_id")
            for index, value in enumerate(sort_values):
                searcher.output(SqlColumn(searcher, value.element), f"sort_{index}")
            streams.append(StoredPropertySqlCandidateStream(backing.backing, backing_name, searcher, len(sort_values)))
        return tuple(streams)

    def response_row(
        self,
        backing: type,
        record: object,
        *,
        public_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Render one hydrated backing record at the protocol boundary."""
        configured = next((item for item in self._backings if item.backing is backing), None)
        if configured is None:
            raise StoredPropertySqlConfigurationError(
                f"{backing.__name__} is not a configured backing for {self.family.__name__}"
            )
        row: dict[str, Any] = {"id": content_id(record) if public_id is None else public_id, "type": self.entry_type}
        for name in self.definition.properties:
            if name in _CORE_PROPERTIES:
                continue
            projection = configured.projections.get(name)
            row[name] = None if projection is None else _response_json_value(projection.response(record))
        return row

    def _records_for(self, backing: _BackingPlan) -> Iterator[Mapping[str, Any]]:
        searcher = self.store.searcher()
        variable = searcher.variable(backing.backing)
        sid = SqlColumn(searcher, variable._alias.c[SID_COLUMN])
        searcher.output(sid, "sid")
        sids = tuple(int(values[0]) for values, _names in searcher)
        hydrator = RowHydrator(self.store, backing.backing, sids)
        for record in hydrator.materialize_many():
            yield self.response_row(backing.backing, record)

    def _filter_searcher(
        self,
        backing: _BackingPlan,
        ast: FilterAst,
        sort: Sequence[tuple[str, bool]],
        public_id_prefix: str,
    ) -> SqlSearcher:
        searcher, variable, _sort_values = self._candidate_searcher(backing, ast, sort, public_id_prefix)
        searcher.output(variable, "record")
        return searcher

    def _candidate_searcher(
        self,
        backing: _BackingPlan,
        ast: FilterAst | None,
        sort: Sequence[tuple[str, bool]],
        public_id_prefix: str,
    ) -> tuple[SqlSearcher, SqlVariable, tuple[_SqlValue, ...]]:
        searcher = self.store.searcher()
        variable = searcher.variable(backing.backing)
        context = _SqlQueryContext(searcher, variable)
        if ast is None:
            searcher.add(cast(SqlExpression, context.always_true()))
        else:
            handlers = self._handlers(backing, context, public_id_prefix)
            try:
                predicate = translate_filter_ast(
                    ast,
                    cast(Any, variable),
                    self.entry_type,
                    _property_fulltypes(self.definition),
                    handlers,
                    known_definition_prefixes(),
                )
            except QueryLiteralError as error:
                raise FilterTranslationError(str(error), "type-mismatch") from error
            searcher.add(cast(SqlExpression, predicate))
        sort_values: list[_SqlValue] = []
        for name, descending in sort:
            value = self._sort_value(backing, context, name, public_id_prefix)
            # SQLite orders nulls first in ascending order while DuckDB's
            # default differs.  Make the cross-dialect NULLS LAST contract
            # explicit before the actual user key in both directions.
            null_rank = sqlalchemy.case((value.element.is_(None), 1), else_=0)
            searcher.add_sort(SqlColumn(searcher, null_rank), False)
            order_element = value.element if value.codec is not None and value.codec.name == "float" else value.exact
            searcher.add_sort(SqlColumn(searcher, order_element), descending)
            sort_values.append(value)
        return searcher, variable, tuple(sort_values)

    def _handlers(
        self,
        backing: _BackingPlan,
        context: _SqlQueryContext,
        public_id_prefix: str,
    ) -> HandlerTable:
        handlers: dict[str, Mapping[str, Callable[..., Any]]] = {
            "id": _id_handlers(context, public_id_prefix),
            "type": _type_handlers(self.entry_type),
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

    def _sort_value(
        self,
        backing: _BackingPlan,
        context: _SqlQueryContext,
        name: str,
        public_id_prefix: str,
    ) -> _SqlValue:
        if name == "id":
            return _public_id_value(context, public_id_prefix)
        if name == "type":
            return context.constant(self.entry_type)
        if name not in self.definition.properties:
            raise StoredPropertySqlConfigurationError(f"{self.entry_type} has no property {name!r} to sort")
        projection = backing.projections.get(name)
        if projection is None or projection.sort is None:
            raise StoredPropertySqlConfigurationError(
                f"{backing.backing.__name__} has no sortable projection for {name!r}"
            )
        sorter = projection.sort
        assert sorter is not None
        value = _value(sorter(cast(Any, context)))
        if value.exact_element is not None and (value.codec is None or value.codec.name != "float"):
            raise StoredPropertySqlConfigurationError(
                f"{backing.backing.__name__}.{name} cannot sort an exact value through its canonical text column"
            )
        return value


def stored_property_sql_plan(store: SqlStore, family: type) -> StoredPropertySqlPlan:
    """Validate and return the SQL property plan for one configured logical family.

    The family must be present in ``store.entry_layout``; unconfigured family
    classes and their records cannot accidentally become part of a durable
    entry source.  ``id`` and ``type`` are intrinsic: a concrete backing's
    canonical ``content_id`` and the family's fixed entry type respectively.
    Backings must not try to redeclare either property.
    """
    layout = next((item for item in store.entry_layout if item.family is family), None)
    if layout is None:
        raise StoredPropertySqlConfigurationError(
            f"entry family {getattr(family, '__name__', family)!r} is not configured in this SqlStore"
        )
    entry_type = getattr(family, "type", None)
    if not isinstance(entry_type, str) or not entry_type or entry_type != entry_type.strip():
        raise StoredPropertySqlConfigurationError(f"{family.__name__}.type must be a non-empty stripped entry type")
    registered_definition_id = entry_family_info(layout.name)[1]
    definition_id = getattr(family, "definition_id", registered_definition_id)
    if definition_id != registered_definition_id:
        raise StoredPropertySqlConfigurationError(
            f"{family.__name__}.definition_id does not match the registered family definition id"
        )
    if not isinstance(definition_id, str) or not definition_id:
        raise StoredPropertySqlConfigurationError(f"{family.__name__} needs a registered entry definition id")
    factory = getattr(family, "entry_type_definition", None)
    definition = factory() if callable(factory) else load_entry_type_schema(definition_id)
    if not isinstance(definition, EntryTypeDefinition):
        raise StoredPropertySqlConfigurationError(
            f"{family.__name__}.entry_type_definition() must return EntryTypeDefinition"
        )
    source_id = definition.definition_id or definition.extends_id
    if source_id != definition_id:
        raise StoredPropertySqlConfigurationError(
            f"{family.__name__}.entry_type_definition() does not describe {definition_id!r}"
        )
    if definition.name != entry_type:
        raise StoredPropertySqlConfigurationError(
            f"{family.__name__}.type is {entry_type!r}, but {definition_id!r} defines {definition.name!r}"
        )

    backing_plans: list[_BackingPlan] = []
    definition_names = set(definition.properties)
    for backing in layout.backings:
        projections = stored_property_projections(backing)
        reserved = sorted(_CORE_PROPERTIES & set(projections))
        if reserved:
            raise StoredPropertySqlConfigurationError(
                f"{backing.__name__} must not declare intrinsic properties: {', '.join(reserved)}"
            )
        unknown = sorted(set(projections) - definition_names)
        if unknown:
            raise StoredPropertySqlConfigurationError(
                f"{backing.__name__} projects properties absent from {definition_id!r}: {', '.join(unknown)}"
            )
        required = sorted(
            name
            for name, property_definition in definition.properties.items()
            if name not in _CORE_PROPERTIES and not property_definition.nullable and name not in projections
        )
        if required:
            raise StoredPropertySqlConfigurationError(
                f"{backing.__name__} has no response mapping for non-null property/properties: {', '.join(required)}"
            )
        backing_plans.append(_BackingPlan(backing, projections))
    return StoredPropertySqlPlan(store, family, layout, entry_type, definition, tuple(backing_plans))


def _property_fulltypes(definition: EntryTypeDefinition) -> Mapping[str, str]:
    return MappingProxyType({name: _definition_fulltype(item) for name, item in definition.properties.items()})


def _definition_fulltype(definition: PropertyDefinition) -> str:
    document = definition.as_optimade()
    value = document["x-optimade-type"]
    if value == "list":
        return "list of " + _fulltype_from_document(cast(Mapping[str, Any], document["items"]))
    if value == "dictionary":
        return "dict"
    return cast(str, value)


def _fulltype_from_document(document: Mapping[str, Any]) -> str:
    value = document["x-optimade-type"]
    if value == "list":
        return "list of " + _fulltype_from_document(cast(Mapping[str, Any], document["items"]))
    if value == "dictionary":
        return "dict"
    return cast(str, value)


def _projection_handlers(
    projection: StoredPropertyProjection, context: _SqlQueryContext
) -> Mapping[str, Callable[..., Any]]:
    query = projection.query
    assert query is not None

    def invoke(operator: str, literal: object) -> _SqlPredicate:
        result = query(cast(Any, context), operator, literal)
        if not isinstance(result, _SqlPredicate):
            raise StoredPropertySqlConfigurationError("stored-property query callback returned a foreign expression")
        return result

    return {
        "comparison": lambda entry, operator, value, _variable: invoke(operator, value),
        "stringmatching": lambda entry, value, operator, _variable: invoke(operator, value),
        "HAS": lambda entry, _ops, values, _variable, operator: invoke(operator, tuple(values)),
        "length": lambda entry, operator, value, _variable: invoke(f"LENGTH {operator}", value),
        "unknown": lambda entry, _variable, operator: invoke(operator, None),
    }


def _null_handlers(context: _SqlQueryContext) -> Mapping[str, Callable[..., Any]]:
    return {
        "comparison": lambda entry, operator, value, variable: _sql_unknown(),
        "stringmatching": lambda entry, value, operator, variable: _sql_unknown(),
        "HAS": lambda entry, ops, values, variable, operator: _sql_unknown(),
        "length": lambda entry, operator, value, variable: _sql_unknown(),
        "unknown": lambda entry, variable, operator: (
            context.always_true() if operator == "IS_UNKNOWN" else context.always_false()
        ),
    }


def _public_id_value(context: _SqlQueryContext, prefix: str) -> _SqlValue:
    """The source-prefixed public id as one portable SQL string expression."""
    if not prefix:
        return _SqlValue(context._root.alias.c[CONTENT_ID_COLUMN])
    return _SqlValue(sqlalchemy.literal(prefix) + context._root.alias.c[CONTENT_ID_COLUMN])


def _id_handlers(context: _SqlQueryContext, prefix: str) -> Mapping[str, Callable[..., Any]]:
    value = _public_id_value(context, prefix)
    return {
        "comparison": lambda entry, operator, literal, variable: context.compare(
            value, operator, context.constant(literal)
        ),
        "stringmatching": lambda entry, literal, operator, variable: context.compare(
            value, operator, context.constant(literal)
        ),
        "unknown": lambda entry, variable, operator: (
            context.always_false() if operator == "IS_UNKNOWN" else context.always_true()
        ),
    }


def _type_handlers(entry_type: str) -> Mapping[str, Callable[..., Any]]:
    return {
        "comparison": lambda entry, operator, literal, variable: constant_comparison_handler(
            entry_type, operator, literal, variable
        ),
        "stringmatching": lambda entry, literal, operator, variable: constant_stringmatching_handler(
            entry_type, literal, operator, variable
        ),
        "unknown": lambda entry, variable, operator: (
            variable.always_false() if operator == "IS_UNKNOWN" else variable.always_true()
        ),
    }


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
        if value.dim in ((), (0,)):
            return []
        return _response_json_value(value.to_fractions())
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


def _scope(value: object) -> _SqlScope:
    if not isinstance(value, _SqlScope):
        raise StoredPropertySqlConfigurationError("stored-property query callback received a foreign scope")
    return value


def _scope_from(scope: _SqlScope) -> tuple[sqlalchemy.FromClause, ...]:
    """Return local aliases, falling back to the root alias for a root scope."""
    return (scope.alias,) if not scope.froms else scope.froms


def _scope_ancestors(scope: _SqlScope) -> tuple[sqlalchemy.FromClause, ...]:
    """All aliases supplied by the parent query, kept in stable identity order."""
    result: list[sqlalchemy.FromClause] = []
    for alias in (*scope.ancestors, *scope.froms, scope.alias):
        if not any(alias is existing for existing in result):
            result.append(alias)
    return tuple(result)


def _codec_literals(left: _SqlValue, right: _SqlValue) -> tuple[_SqlValue, _SqlValue]:
    """Encode a raw literal through the codec of the field it is compared with."""
    if left.codec is not None and right.literal is not _NO_LITERAL:
        right = _codec_literal(right, left.codec, left.exact_element is not None)
    elif right.codec is not None and left.literal is not _NO_LITERAL:
        left = _codec_literal(left, right.codec, right.exact_element is not None)
    return left, right


def _exact_ordering_uses_float_companion(left: _SqlValue, right: _SqlValue) -> bool:
    """Whether exact operands belong to float fields whose DOUBLE remains orderable."""
    codecs = tuple(value.codec for value in (left, right) if value.codec is not None)
    return bool(codecs) and all(codec.name == "float" for codec in codecs)


def _codec_literal(value: _SqlValue, codec: ValueCodec, needs_exact: bool) -> _SqlValue:
    """Render one callback literal in a field codec's persisted query domain."""
    raw = value.literal
    if codec.name == "datetime" and isinstance(raw, str):
        raw = _parse_rfc3339_datetime(raw)
    if not isinstance(raw, codec.python_type):
        raise QueryLiteralError(f"{codec.name} property requires a {codec.python_type.__name__} literal")
    try:
        encoded = codec.encode(raw)
    except (TypeError, ValueError) as error:
        raise QueryLiteralError(f"invalid {codec.name} property literal") from error
    try:
        query_index = next(index for index, (suffix, _kind) in enumerate(codec.columns) if suffix == codec.query_suffix)
    except StopIteration as error:  # pragma: no cover - ValueCodec registration validates this convention
        raise StoredPropertySqlConfigurationError(f"{codec.name} codec has no query column") from error
    exact_element = None
    if needs_exact:
        try:
            exact_index = next(index for index, (suffix, _kind) in enumerate(codec.columns) if suffix == "_exact")
        except StopIteration as error:  # pragma: no cover - field/schema codec inconsistency
            raise StoredPropertySqlConfigurationError(f"{codec.name} codec has no exact column") from error
        exact_element = sqlalchemy.literal(encoded[exact_index])
    return _SqlValue(sqlalchemy.literal(encoded[query_index]), exact_element=exact_element)


def _parse_rfc3339_datetime(value: str) -> datetime.datetime:
    """Parse an OPTIMADE timestamp literal so the datetime codec can canonicalize it."""
    if _RFC3339_TIMESTAMP.fullmatch(value) is None:
        raise QueryLiteralError("timestamp property requires an RFC 3339 literal")
    normalized_text = value.replace("t", "T", 1)
    if normalized_text.endswith(("Z", "z")):
        normalized_text = normalized_text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(normalized_text)
    except ValueError as error:
        raise QueryLiteralError("timestamp property requires an RFC 3339 literal") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise QueryLiteralError("timestamp property requires an RFC 3339 UTC offset")
    return parsed


def _correlate_nested(
    clause: sqlalchemy.ColumnElement[bool], aliases: tuple[sqlalchemy.FromClause, ...]
) -> sqlalchemy.ColumnElement[bool]:
    """Correlate subqueries in one scope predicate to that scope's local aliases.

    A declaration can construct a descendant scope before placing it inside
    ``exists(parent, ...)``.  The descendant query must then see the parent
    aliases from that outer query rather than add same-named local aliases of
    its own.  SQLAlchemy only applies ``correlate`` to the immediate SELECT,
    so walk the predicate tree and make that relationship explicit for every
    nested EXISTS or scalar subquery.
    """

    def replace(node: sqlalchemy.ClauseElement) -> sqlalchemy.ClauseElement | None:
        if isinstance(node, Exists | ScalarSelect):
            return cast(Any, node).correlate(*aliases)
        return None

    return cast(
        sqlalchemy.ColumnElement[bool],
        cast(Any, replacement_traverse)(clause, {}, replace),
    )


def _value(value: object) -> _SqlValue:
    if not isinstance(value, _SqlValue):
        raise StoredPropertySqlConfigurationError("stored-property query callback received a foreign value")
    return value


def _predicate(value: object) -> _SqlPredicate:
    if not isinstance(value, _SqlPredicate):
        raise StoredPropertySqlConfigurationError("stored-property query callback received a foreign predicate")
    return value


def _sql_unknown() -> _SqlPredicate:
    """A SQL UNKNOWN predicate, which stays unknown under boolean negation."""
    return _SqlPredicate(cast(sqlalchemy.ColumnElement[bool], sqlalchemy.null()))


def _is_null(value: sqlalchemy.ColumnElement[Any]) -> bool:
    return isinstance(value, Null)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
