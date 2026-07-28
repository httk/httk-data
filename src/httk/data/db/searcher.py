"""The query DSL: build and run searches over stored dataclasses through SQLAlchemy Core.

:class:`SqlSearcher` (obtained from :meth:`~httk.data.db.store.SqlStore.searcher`)
implements the backend-agnostic search protocols of :mod:`httk.data.query` —
:class:`~httk.data.query.Searcher`, :class:`~httk.data.query.SearchVariable`,
:class:`~httk.data.query.SearchField`, :class:`~httk.data.query.SearchExpression`
— porting the query semantics of the v1 ``httk.db`` ``FilteredCollection``
searchers onto SQLAlchemy Core:

- :meth:`SqlSearcher.variable` binds a storable class to a **fresh alias** of
  its table; two variables of the same class therefore make a self-join.
- Attribute access on a :class:`SqlVariable` follows the class's resolved
  :class:`~httk.data.db.schema.TableSchema`: scalar and encoded fields yield a
  :class:`SqlColumn` over the field's query column; reference fields yield a
  chainable :class:`SqlReference` that compares by foreign key
  (``v.ref == other_variable`` / ``== stored_object`` / ``== None``) and, on
  further attribute access, lazily LEFT OUTER JOINs a fresh alias of the target
  table (``v.ref.doi`` chains arbitrarily deep; one join alias per reference
  path per variable); variable-length (child-table) fields LEFT OUTER JOIN the
  child table (a fresh alias per attribute access, so independent set
  predicates on one field — e.g. ``v.symbols.has_any('O') &
  v.symbols.has_any('Ca')`` — constrain independent joined rows, as in httk
  v1) and switch the searcher into grouped mode (GROUP BY the root rows).
- Comparisons and set operations on columns produce :class:`SqlExpression`
  objects carrying **two** renderings — a WHERE-position clause and a
  HAVING-position clause — exactly as the v1 searcher rendered one expression
  per position. Each expression also carries **its own placement**, so
  :meth:`SqlSearcher.add` is the only way to apply one: the WHERE rendering
  always applies, and the HAVING rendering additionally applies when the
  expression is flagged :attr:`SqlExpression.post` (the for-all forms
  ``has_only`` and child-field ``is_in``, and any ``~`` over a set-derived
  subtree — see :class:`SqlExpression`'s ``__invert__``, which lets ``~`` say
  "no joined row matches" and so replaces the removed ``has_inv_*`` methods).
  Alongside the two renderings each expression carries the **non-aggregated
  columns its HAVING rendering references** (:attr:`SqlExpression.group_columns`), unioned
  by ``&``/``|`` and preserved by ``~``. A grouped query GROUP BYs those
  columns in addition to the root ``sid``\\ s (``SqlSearcher._grouping``, which
  both ``count()`` and iteration go through): a plain comparison reaching HAVING
  position mentions a root-table column directly, and strict dialects (DuckDB)
  reject such a column unless it is grouped, while permissive ones (SQLite)
  accept it via functional dependency on the grouped primary key. Grouping by
  it is sound for exactly that reason — one distinct value per group — and the
  same applies to the sort keys, which the same helper adds.

The neutral protocol's string matching is ``contains``/``startswith``/
``endswith`` over **literal** text; this backend renders them as SQL LIKE with
backslash as the escape character, escaping ``%`` and ``_`` in the given text
first. The LIKE rendering itself is private (``SqlColumn._like``) precisely
because pattern syntax is a dialect detail that must not leak into the
contract.

Encoded (codec) fields compare against their query column: for rationals that
is the float companion column, so SQL comparisons on them are documented
float-approximate (stored values themselves round-trip exactly). Comparison
values are encoded through the field's codec, e.g. comparing a
:class:`fractions.Fraction` field against ``Fraction(1, 3)`` compares the
float column against ``float(Fraction(1, 3))``.

Set-operation semantics (ported from the v1 ``BinaryBooleanOp._sql``,
translated to portable SQLAlchemy aggregates): in WHERE position ``has_any``
renders as ``column IN values`` while ``has_only`` renders as constant true; in
HAVING position the aggregate forms
``SUM(CASE WHEN column [NOT] IN values THEN 1 ELSE 0 END)``
compare per-group match counts. A NULL child value — the LEFT OUTER JOIN row
of a parent with no children — never satisfies ``IN`` (nor ``NOT IN``), so a
record with an empty child list matches ``has_only`` and fails ``has_any``,
the exact set semantics of the reference in-memory store.
"""

import dataclasses
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, NoReturn, cast

import sqlalchemy

from httk.data.db.codecs import ValueCodec, codec_named
from httk.data.db.mapping import SID_COLUMN
from httk.data.db.schema import FieldSpec, SchemaError, TableSchema, resolve_schema
from httk.data.query import SearchResult

if TYPE_CHECKING:
    from httk.data.db.store import SqlStore

__all__ = [
    "SqlColumn",
    "SqlExpression",
    "SqlReference",
    "SqlSearcher",
    "SqlVariable",
]


def _bool_clause(clause: Any) -> sqlalchemy.ColumnElement[bool]:
    """Contain the SQLAlchemy operator-return typing at one place."""
    return cast("sqlalchemy.ColumnElement[bool]", clause)


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards in a literal string (backslash escape, as v1 used)."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SqlExpression:
    """A search condition carrying both its WHERE-position and HAVING-position renderings.

    Plain comparisons render identically in both positions; the set operations
    differ (constant true/false in WHERE, aggregate match counts in HAVING).
    The combinators ``&``, ``|`` and ``~`` combine the two renderings pairwise.
    :attr:`group_columns` travels along so a grouped query can GROUP BY every
    non-aggregated column its HAVING clauses mention.
    """

    __slots__ = ("group_columns", "having_clause", "post", "set_derived", "where_clause")

    def __init__(
        self,
        where_clause: sqlalchemy.ColumnElement[bool],
        having_clause: sqlalchemy.ColumnElement[bool],
        *,
        post: bool = False,
        set_derived: bool = False,
        group_columns: tuple[sqlalchemy.ColumnElement[Any], ...] = (),
    ) -> None:
        self.where_clause = where_clause
        """The rendering applied in WHERE position (always applied)."""
        self.having_clause = having_clause
        """The rendering applied in HAVING position when :attr:`post` is set."""
        self.post = post
        """Whether :meth:`SqlSearcher.add` must *also* apply :attr:`having_clause`.

        Set when the WHERE rendering alone is incomplete, i.e. by the for-all
        forms (``has_only``, ``is_in`` on a child field) and by ``~`` over a
        set-derived subtree. Those all render WHERE as constant true, so the
        pair "WHERE plus HAVING" is never a double restriction. ``has_any``
        deliberately does **not** set it: its WHERE rendering is exact, and
        forcing it into HAVING measurably slows DuckDB down.
        """
        self.set_derived = set_derived
        """Whether this expression's truth is a property of a *set* of joined rows.

        Negating such an expression cannot be done row-by-row in WHERE
        position (no single joined row can witness "no row matches"), so
        ``__invert__`` negates the aggregate instead — see there.
        """
        self.group_columns = group_columns
        """Non-aggregated columns of :attr:`having_clause` that must be grouped by.

        Empty for the set operations (their HAVING rendering is fully
        aggregated) and for child-table comparisons (those columns *are* the
        aggregated rows); a root-table comparison contributes its own column.
        """

    def __and__(self, other: "SqlExpression") -> "SqlExpression":
        return SqlExpression(
            sqlalchemy.and_(self.where_clause, other.where_clause),
            sqlalchemy.and_(self.having_clause, other.having_clause),
            post=self.post or other.post,
            set_derived=self.set_derived or other.set_derived,
            group_columns=self.group_columns + other.group_columns,
        )

    def __or__(self, other: "SqlExpression") -> "SqlExpression":
        return SqlExpression(
            sqlalchemy.or_(self.where_clause, other.where_clause),
            sqlalchemy.or_(self.having_clause, other.having_clause),
            post=self.post or other.post,
            set_derived=self.set_derived or other.set_derived,
            group_columns=self.group_columns + other.group_columns,
        )

    def __invert__(self) -> "SqlExpression":
        """Negate, aggregating first when the subtree's truth depends on a set of rows.

        For a set-derived subtree the WHERE rendering is dropped to constant
        true and only the (fully aggregated) HAVING rendering is negated: "no
        joined row matches" is not expressible per row. This deliberately gives
        up a potentially index-assisted WHERE prefilter — the same trade the
        removed ``has_inv_*`` methods made — so do not "optimize" the narrowing
        back in: negating a per-row restriction would drop exactly the rows the
        aggregate must count.
        """
        if self.set_derived:
            return SqlExpression(
                sqlalchemy.true(),
                sqlalchemy.not_(self.having_clause),
                post=True,
                set_derived=True,
                group_columns=self.group_columns,
            )
        return SqlExpression(
            sqlalchemy.not_(self.where_clause),
            sqlalchemy.not_(self.having_clause),
            post=self.post,
            group_columns=self.group_columns,
        )


def _same(clause: Any, group_columns: tuple[sqlalchemy.ColumnElement[Any], ...] = ()) -> SqlExpression:
    """One clause serving as both renderings (every plain comparison)."""
    rendered = _bool_clause(clause)
    return SqlExpression(rendered, rendered, group_columns=group_columns)


class SqlColumn:
    """A queryable column of a search variable.

    Rich comparisons (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``),
    ``contains``/``startswith``/``endswith``/``is_in`` and the set operations
    return :class:`SqlExpression`. Comparison values are encoded through the
    field's value codec when the field is codec-encoded, so e.g. rational
    comparisons run on the float companion column (documented approximate).
    Comparing against another :class:`SqlColumn` compares the two columns.
    """

    def __init__(
        self,
        searcher: "SqlSearcher",
        element: sqlalchemy.ColumnElement[Any],
        *,
        codec: ValueCodec | None = None,
        query_index: int = 0,
        from_child: bool = False,
    ) -> None:
        self._searcher = searcher
        self._element = element
        self._codec = codec
        self._query_index = query_index
        self._from_child = from_child

    def _encode(self, value: Any) -> Any:
        if isinstance(value, SqlColumn):
            return value._element
        if value is None or self._codec is None:
            return value
        if isinstance(value, self._codec.python_type):
            return self._codec.encode(value)[self._query_index]
        return value

    def _plain(self, clause: Any) -> SqlExpression:
        """A comparison on this column, rendered for both clause positions.

        On a **root-table** column the same clause serves both positions, and it
        must appear in GROUP BY to reach HAVING position (DuckDB rejects an
        unaggregated column there; SQLite tolerates it via functional dependency
        on the grouped primary key).

        On a **child-table** column the two positions genuinely differ, exactly
        as they do for the set operations. Row-wise, ``child.value = x`` means
        "some joined row matches" — the right reading, and what WHERE keeps. But
        that reading cannot be negated row-wise (no single joined row witnesses
        "no row matches"), and it cannot stand in HAVING position unaggregated.
        So the HAVING rendering is the aggregate existential
        ``SUM(CASE WHEN clause THEN 1 ELSE 0 END) > 0`` and the expression is
        marked :attr:`~SqlExpression.set_derived`, which makes ``~`` negate the
        aggregate: ``~(v.symbols == 'O')`` then means "no symbol is O" rather
        than "some symbol is not O", agreeing with ``~v.symbols.has_any('O')``.
        ``post`` stays unset, so an un-negated comparison still filters in WHERE
        alone and does not force grouped mode.
        """
        if not self._from_child:
            return _same(clause, (self._element,))
        return SqlExpression(
            _bool_clause(clause),
            _bool_clause(self._match_count(clause) > 0),
            set_derived=True,
        )

    def _match_count(self, condition: Any) -> Any:
        """``SUM(CASE WHEN condition THEN 1 ELSE 0 END)`` — NULL rows count as 0."""
        return sqlalchemy.func.sum(sqlalchemy.case((_bool_clause(condition), 1), else_=0))

    # ------------------------------------------------------------------ comparisons

    def __eq__(self, other: object) -> SqlExpression:  # type: ignore[override]
        encoded = self._encode(other)
        if encoded is None:
            return self._plain(self._element.is_(None))
        return self._plain(self._element == encoded)

    def __ne__(self, other: object) -> SqlExpression:  # type: ignore[override]
        encoded = self._encode(other)
        if encoded is None:
            return self._plain(self._element.is_not(None))
        return self._plain(self._element != encoded)

    def __hash__(self) -> int:
        return id(self)

    def __lt__(self, other: Any) -> SqlExpression:
        return self._plain(self._element < self._encode(other))

    def __le__(self, other: Any) -> SqlExpression:
        return self._plain(self._element <= self._encode(other))

    def __gt__(self, other: Any) -> SqlExpression:
        return self._plain(self._element > self._encode(other))

    def __ge__(self, other: Any) -> SqlExpression:
        return self._plain(self._element >= self._encode(other))

    # ------------------------------------------------------------------ string matching

    def _like(self, pattern: str) -> SqlExpression:
        """SQL LIKE with backslash as the escape character (as the v1 searcher used).

        Private: LIKE syntax is this backend's own business. The neutral
        protocol only ever passes literal text, which the three public methods
        below escape before assembling their pattern.
        """
        return self._plain(self._element.like(pattern, escape="\\"))

    def contains(self, text: str) -> SqlExpression:
        """Match values containing the literal ``text`` (LIKE wildcards escaped)."""
        return self._like("%" + _escape_like(text) + "%")

    def startswith(self, prefix: str) -> SqlExpression:
        """Match values beginning with the literal ``prefix`` (LIKE wildcards escaped)."""
        return self._like(_escape_like(prefix) + "%")

    def endswith(self, suffix: str) -> SqlExpression:
        """Match values ending with the literal ``suffix`` (LIKE wildcards escaped)."""
        return self._like("%" + _escape_like(suffix))

    # ------------------------------------------------------------------ set operations

    def is_in(self, *values: Any) -> SqlExpression:
        """Membership in ``values``.

        On a root column this is plain ``column IN values``. On a *child* field
        it is the for-all reading — every child value is in ``values`` — which,
        exactly as :meth:`has_only`, is an aggregate over the group and so
        renders as constant true in WHERE position and forces the HAVING
        rendering (:attr:`SqlExpression.post`).

        Only the child form is :attr:`~SqlExpression.set_derived`: on a root
        column ``~column.is_in(...)`` is exactly ``column NOT IN values``
        row-wise, so negating it aggregate-style would switch the query into
        grouped mode for no gain.
        """
        member = self._element.in_([self._encode(value) for value in values])
        where: sqlalchemy.ColumnElement[bool] = sqlalchemy.true() if self._from_child else _bool_clause(member)
        return SqlExpression(
            where,
            _bool_clause(self._match_count(sqlalchemy.not_(member)) == 0),
            post=self._from_child,
            set_derived=self._from_child,
        )

    def has_any(self, *values: Any) -> SqlExpression:
        """Some child value is in ``values``: WHERE ``IN``; HAVING a positive match count.

        The WHERE rendering is exact, so this does not set
        :attr:`SqlExpression.post` — pushing it into HAVING as well is a
        measured DuckDB slowdown for no semantic gain. It is nonetheless
        set-derived, so ``~`` negates the aggregate.
        """
        member = self._element.in_([self._encode(value) for value in values])
        return SqlExpression(
            _bool_clause(member),
            _bool_clause(self._match_count(member) > 0),
            set_derived=True,
        )

    def has_only(self, *values: Any) -> SqlExpression:
        """Every child value is in ``values``: constant true in WHERE, zero outsiders in HAVING.

        A record with no child rows at all satisfies this (its single LEFT
        OUTER JOIN row is NULL, which never matches ``NOT IN``) — the empty
        set is a subset of any value set.
        """
        outside = self._element.notin_([self._encode(value) for value in values])
        return SqlExpression(
            sqlalchemy.true(),
            _bool_clause(self._match_count(outside) == 0),
            post=True,
            set_derived=True,
        )

    def __getattr__(self, name: str) -> NoReturn:
        """Refuse to chain: this column holds a value, not a reference.

        The :class:`~httk.data.query.SearchField` contract allows attribute
        access because a field *may* refer to another record (see
        :class:`SqlReference`); a plain value column cannot, and says so rather
        than letting the default lookup failure suggest a typo in the method
        name.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"{self._element.name!r} holds a value, not a reference to another record, "
            f"so {name!r} cannot be looked up through it"
        )


class SqlReference:
    """A reference (foreign key) field of a search variable, chainable into the target.

    Supports ``== other_variable`` (join condition), ``== stored_object``
    (the object must be known to the store), and ``== None`` (no referent);
    ``!=`` gives the negated forms. The set operations (``has_any``/``has_only``)
    treat the reference as the (at most one-element) set of its referent,
    rendering directly over the foreign-key column (WHERE-position ``IN``);
    their values are stored instances or raw sids (:class:`int`, as returned
    by :meth:`~httk.data.db.store.SqlStore.save`). Attribute access LEFT OUTER
    JOINs the target class's table (once per reference path per variable) and
    delegates to the joined sub-variable, so chains like ``v.ref.doi`` — or
    deeper — work and repeated access hits the same join alias.
    """

    def __init__(self, variable: "SqlVariable", spec: FieldSpec) -> None:
        self._variable = variable
        self._spec = spec

    @property
    def _fk(self) -> sqlalchemy.ColumnElement[Any]:
        return self._variable._alias.c[self._spec.columns[0].name]

    def _target_sid(self, other: Any) -> int:
        sid = self._variable._searcher._store.sid_of(other)
        if sid is None:
            raise ValueError(
                f"the {type(other).__name__} instance compared against "
                f"{self._variable._cls.__name__}.{self._spec.field} has not been stored or fetched "
                f"through this store"
            )
        return sid

    def __eq__(self, other: object) -> SqlExpression:  # type: ignore[override]
        # The foreign key lives on the (root or reference-joined) parent row, so
        # it is one value per group and safe — and, on DuckDB, necessary — to
        # GROUP BY when this comparison reaches HAVING position.
        if isinstance(other, SqlVariable):
            return _same(self._fk == other._alias.c[SID_COLUMN], (self._fk,))
        if other is None:
            return _same(self._fk.is_(None), (self._fk,))
        return _same(self._fk == self._target_sid(other), (self._fk,))

    def __ne__(self, other: object) -> SqlExpression:  # type: ignore[override]
        if isinstance(other, SqlVariable):
            return _same(self._fk != other._alias.c[SID_COLUMN], (self._fk,))
        if other is None:
            return _same(self._fk.is_not(None), (self._fk,))
        return _same(self._fk != self._target_sid(other), (self._fk,))

    def __hash__(self) -> int:
        return id(self)

    # ------------------------------------------------------------------ set operations

    def _fk_column(self) -> SqlColumn:
        return SqlColumn(self._variable._searcher, self._fk)

    def _sid_value(self, value: Any) -> int:
        return value if isinstance(value, int) else self._target_sid(value)

    def has_any(self, *values: Any) -> SqlExpression:
        """The referent is among ``values`` (stored instances or raw sids): FK ``IN``."""
        return self._fk_column().has_any(*[self._sid_value(value) for value in values])

    def has_only(self, *values: Any) -> SqlExpression:
        """The referent (when set) is among ``values`` (see :meth:`SqlColumn.has_only`)."""
        return self._fk_column().has_only(*[self._sid_value(value) for value in values])

    def __getattr__(self, name: str) -> "SqlColumn | SqlReference":
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._variable._reference_variable(self._spec), name)


class SqlVariable:
    """A query variable bound to a fresh alias of a storable class's table.

    Attribute access resolves stored fields (including stored properties) into
    :class:`SqlColumn` / :class:`SqlReference` objects per the class's
    :class:`~httk.data.db.schema.TableSchema`; ``sid`` (a reserved field name)
    yields the store-managed integer primary key column; accessing a
    variable-length (child-table) field registers a LEFT OUTER JOIN and
    switches the searcher into grouped mode. Unknown names raise
    :class:`AttributeError`;
    fixed-shape tensor fields raise :class:`~httk.data.db.schema.SchemaError`
    (they are not queryable as a whole).

    :meth:`always_true` and :meth:`always_false` are — like ``sid`` — reserved
    names that never resolve to a stored field: they are real methods declared
    before ``__getattr__``, so no query column is involved at all.
    """

    def __init__(self, searcher: "SqlSearcher", cls: type, schema: TableSchema, alias: sqlalchemy.FromClause) -> None:
        self._searcher = searcher
        self._cls = cls
        self._schema = schema
        self._alias = alias
        self._joins: list[tuple[sqlalchemy.FromClause, sqlalchemy.ColumnElement[bool], SqlVariable | None]] = []
        self._reference_variables: dict[str, SqlVariable] = {}

    def always_true(self) -> SqlExpression:
        """A condition matching every row (SQL ``TRUE`` in both positions)."""
        return SqlExpression(sqlalchemy.true(), sqlalchemy.true())

    def always_false(self) -> SqlExpression:
        """A condition matching no row (SQL ``FALSE`` in both positions)."""
        return SqlExpression(sqlalchemy.false(), sqlalchemy.false())

    def __getattr__(self, name: str) -> "SqlColumn | SqlReference":
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "sid":
            # The store-managed integer primary key ('sid' is a reserved field
            # name, so this never shadows a stored field).
            return SqlColumn(self._searcher, self._alias.c[SID_COLUMN])
        try:
            spec = self._schema.field(name)
        except SchemaError:
            raise AttributeError(f"{self._cls.__name__} has no stored field {name!r} to query") from None
        if spec.role == "scalar":
            return SqlColumn(self._searcher, self._alias.c[spec.columns[0].name])
        if spec.role == "encoded":
            assert spec.codec_name is not None
            codec = codec_named(spec.codec_name)
            return SqlColumn(
                self._searcher,
                self._alias.c[spec.field + codec.query_suffix],
                codec=codec,
                query_index=_query_index(codec),
            )
        if spec.role == "reference":
            return SqlReference(self, spec)
        if spec.role == "child":
            return self._child_column(spec)
        raise SchemaError(
            f"{self._cls.__name__}.{spec.field} is a fixed-shape tensor field and cannot be queried "
            f"as a whole (querying individual components is not implemented yet)"
        )

    def _child_column(self, spec: FieldSpec) -> SqlColumn:
        # A fresh alias per attribute access, as in httk v1: AND-composing
        # independent set predicates on one child field (the translation
        # layer's HAS ALL pattern) must constrain independent joined rows.
        assert spec.child is not None
        table = self._searcher._store._table(spec.child.table_name)
        alias = table.alias()
        onclause = _bool_clause(alias.c[f"{self._schema.table_name}_sid"] == self._alias.c[SID_COLUMN])
        self._joins.append((alias, onclause, None))
        self._searcher._grouped = True
        codec: ValueCodec | None = None
        if spec.target is not None:
            column_name = f"{spec.field}_sid"
        elif spec.codec_name is not None:
            codec = codec_named(spec.codec_name)
            column_name = spec.field + codec.query_suffix
        else:
            column_name = spec.child.element_columns[0].name
        return SqlColumn(
            self._searcher,
            alias.c[column_name],
            codec=codec,
            query_index=_query_index(codec) if codec is not None else 0,
            from_child=True,
        )

    def _reference_variable(self, spec: FieldSpec) -> "SqlVariable":
        sub = self._reference_variables.get(spec.field)
        if sub is None:
            assert spec.target is not None
            target_schema = resolve_schema(spec.target)
            alias = self._searcher._store._table(target_schema.table_name).alias()
            onclause = _bool_clause(self._alias.c[spec.columns[0].name] == alias.c[SID_COLUMN])
            sub = SqlVariable(self._searcher, spec.target, target_schema, alias)
            self._reference_variables[spec.field] = sub
            self._joins.append((alias, onclause, sub))
        return sub

    def _flat_joins(self) -> Iterator[tuple[sqlalchemy.FromClause, sqlalchemy.ColumnElement[bool]]]:
        for alias, onclause, sub in self._joins:
            yield alias, onclause
            if sub is not None:
                yield from sub._flat_joins()


@dataclasses.dataclass(frozen=True)
class _Output:
    """One declared output: a column to select and, for objects, the class to reconstruct."""

    name: str
    element: sqlalchemy.ColumnElement[Any]
    target: type | None
    from_child: bool


class SqlSearcher:
    """One query under construction against a :class:`~httk.data.db.store.SqlStore`.

    Build the query with :meth:`variable`, :meth:`add` (AND-joined conditions,
    each placed by the expression itself), :meth:`output`, :meth:`add_sort`,
    :meth:`set_limit` (``-1`` clears the limit) and :meth:`add_offset` (the
    public :attr:`offset` attribute is readable and writable). Iterating
    yields one :class:`~httk.data.query.SearchResult` per match, whose
    ``values`` holds one entry per declared output — the reconstructed instance for variable
    outputs (via :meth:`~httk.data.db.store.SqlStore.fetch`, so the identity
    cache applies), the raw column value for column outputs. :meth:`count`
    returns the number of matches, disregarding any limit and offset.
    """

    def __init__(self, store: "SqlStore") -> None:
        self._store = store
        self._variables: list[SqlVariable] = []
        self._where: list[SqlExpression] = []
        self._having: list[SqlExpression] = []
        self._outputs: list[_Output] = []
        self._sorts: list[tuple[SqlColumn, bool]] = []
        self._grouped = False
        self._limit: int | None = None
        self.offset: int = 0
        """Row offset applied when iterating; mutable (:meth:`add_offset` adds to it)."""

    def variable(self, target: type) -> SqlVariable:
        """A new query variable over ``target``'s table (a fresh alias; self-joins allowed).

        The class's tables are created on first use (via
        :meth:`~httk.data.db.store.SqlStore.ensure_tables`).
        """
        self._store.ensure_tables(target)
        schema = resolve_schema(target)
        alias = self._store._table(schema.table_name).alias()
        variable = SqlVariable(self, target, schema, alias)
        self._variables.append(variable)
        return variable

    def output(self, variable: "SqlVariable | SqlColumn", name: str) -> None:
        """Append an output: a variable (yields the reconstructed instance) or a column (raw value)."""
        if isinstance(variable, SqlVariable):
            self._outputs.append(_Output(name, variable._alias.c[SID_COLUMN], variable._cls, False))
        elif isinstance(variable, SqlColumn):
            self._outputs.append(_Output(name, variable._element, None, variable._from_child))
        else:
            raise TypeError(f"output() takes a search variable or a search column, got {type(variable).__name__}")

    def add(self, expression: SqlExpression) -> None:
        """Add a condition; all added conditions must hold.

        The expression decides its own placement: it always applies in WHERE
        position, and an expression flagged :attr:`SqlExpression.post` — a
        for-all form, or a negated set-derived subtree — additionally applies
        in HAVING position, which switches the searcher into grouped mode.
        """
        self._where.append(expression)
        if expression.post:
            self._having.append(expression)
            self._grouped = True

    def add_sort(self, field: SqlColumn, descending: bool = False) -> None:
        """Append a sort key; the first-declared key is the most significant."""
        self._sorts.append((field, descending))

    def set_limit(self, limit: int) -> None:
        """Limit the number of iterated matches; a negative value (e.g. ``-1``) clears the limit."""
        self._limit = None if limit < 0 else limit

    def add_offset(self, offset: int) -> None:
        """Add to the row :attr:`offset` applied when iterating."""
        self.offset += offset

    # ------------------------------------------------------------------ execution

    def _joined(self, variable: SqlVariable) -> sqlalchemy.FromClause:
        clause: sqlalchemy.FromClause = variable._alias
        for alias, onclause in variable._flat_joins():
            clause = clause.outerjoin(alias, onclause)
        return clause

    def _grouping(self, base: list[sqlalchemy.ColumnElement[Any]]) -> list[sqlalchemy.ColumnElement[Any]]:
        """``base`` plus every column a grouped query must additionally GROUP BY.

        Those are (a) the :attr:`SqlExpression.group_columns` of the conditions
        applied in HAVING position — a plain comparison reaching HAVING names a
        root-table column, which strict dialects (DuckDB) reject unless it is
        grouped — and (b) the non-child sort keys, which ORDER BY names for the
        same reason. Both kinds hold one distinct value per group (they are
        functionally dependent on the grouped root ``sid``), so grouping by them
        cannot split a group.

        Columns are de-duplicated by :func:`id`, never by ``==``/``in``:
        :class:`sqlalchemy.ColumnElement` overloads ``==`` to build a SQL clause
        whose ``__bool__`` raises, so a containment test would blow up here.
        """
        columns = list(base)
        seen = {id(column) for column in columns}

        def append(column: sqlalchemy.ColumnElement[Any]) -> None:
            if id(column) not in seen:
                seen.add(id(column))
                columns.append(column)

        for expression in self._having:
            for column in expression.group_columns:
                append(column)
        for sort_column, _descending in self._sorts:
            if not sort_column._from_child:
                append(sort_column._element)
        return columns

    def _base_select(
        self,
        columns: list[sqlalchemy.ColumnElement[Any]],
        group_columns: list[sqlalchemy.ColumnElement[Any]],
    ) -> sqlalchemy.Select[Any]:
        if not self._variables:
            raise ValueError("this searcher has no query variables; call variable() first")
        statement = sqlalchemy.select(*columns).select_from(*[self._joined(v) for v in self._variables])
        if self._where:
            statement = statement.where(*[expression.where_clause for expression in self._where])
        if self._grouped:
            # _grouping() applies to count() and __iter__ alike, so the two can
            # never disagree about which rows form a group.
            statement = statement.group_by(*self._grouping(group_columns))
            if self._having:
                statement = statement.having(*[expression.having_clause for expression in self._having])
        return statement

    def count(self) -> int:
        """The number of matches, disregarding any limit and offset (grouped: one per group)."""
        sids = [cast("sqlalchemy.ColumnElement[Any]", v._alias.c[SID_COLUMN]) for v in self._variables]
        statement = self._base_select(sids, sids)
        count_statement = sqlalchemy.select(sqlalchemy.func.count()).select_from(statement.subquery())
        with self._store._read_connection() as connection:
            return int(connection.execute(count_statement).scalar_one())

    def __iter__(self) -> Iterator[SearchResult]:
        """Run the query; yield a :class:`~httk.data.query.SearchResult` per match.

        ``values`` holds one entry per declared output (reconstructed instance
        or raw column value), ``names`` the names they were declared under.
        """
        if not self._outputs:
            raise ValueError("this searcher has no outputs; call output() before iterating")
        columns = [output.element for output in self._outputs]
        group_columns = [cast("sqlalchemy.ColumnElement[Any]", v._alias.c[SID_COLUMN]) for v in self._variables]
        group_columns += [output.element for output in self._outputs if output.target is None and not output.from_child]
        statement = self._base_select(columns, group_columns)
        for column, descending in self._sorts:
            statement = statement.order_by(column._element.desc() if descending else column._element.asc())
        if self._limit is not None:
            statement = statement.limit(self._limit)
        if self.offset > 0:
            statement = statement.offset(self.offset)
        names = tuple(output.name for output in self._outputs)
        results: list[SearchResult] = []
        with self._store._read_connection() as connection:
            # Materialize the match rows before reconstructing any object output:
            # reconstruction issues nested queries on this same connection, and a
            # DuckDB connection carries only one active result set, so streaming
            # the outer cursor across a nested fetch silently truncates it after
            # the first row. (Under SQLite it merely worked by accident.)
            for row in connection.execute(statement).fetchall():
                values: list[Any] = []
                for output, value in zip(self._outputs, row, strict=True):
                    if output.target is None:
                        values.append(value)
                    elif value is None:
                        break  # An object output with no row (outer-joined away); skip the match.
                    else:
                        values.append(self._store._fetch(connection, output.target, int(value)))
                else:
                    results.append(SearchResult(tuple(values), names))
        return iter(results)


def _query_index(codec: ValueCodec) -> int:
    """The index of the codec's query column among its columns."""
    for i, (suffix, _kind) in enumerate(codec.columns):
        if suffix == codec.query_suffix:
            return i
    return 0
