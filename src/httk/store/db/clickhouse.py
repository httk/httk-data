"""ClickHouse-specific SQLAlchemy decoration and KeeperMap primitives.

The rest of :mod:`httk.store.db` deliberately deals in generic SQLAlchemy
objects.  This module is the adapter boundary for ClickHouse's physical types,
MergeTree sorting keys, system catalogue, and KeeperMap metadata protocol.
"""

import contextlib
import datetime
import functools
import json
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import sqlalchemy
import sqlalchemy.util
from sqlalchemy import event
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BinaryExpression, ColumnElement
from sqlalchemy.sql.functions import Function
from sqlalchemy.sql.selectable import Select

from httk.store.query import UnsupportedQueryError

_MIN_SERVER_VERSION = (26, 7)
_BOOTSTRAP_TABLE = "_httk_bootstrap"
_BOOTSTRAP_PATH = "/_httk_bootstrap"
_METADATA_MARKER = "httk_metadata"
_LEASE_KEY = "lease"
_INGEST_STATE_KEY = "ingest_state"
_BOOTSTRAP_LOCK_MESSAGE = "ClickHouse Keeper bootstrap lock is unavailable; Keeper is required"

__all__ = [
    "ClickHouseUnsupportedQueryError",
    "acquire_lease",
    "actual_columns",
    "actual_schema_objects",
    "bootstrap_fence",
    "clear_ingest_marker",
    "decorate_table",
    "ensure_bootstrap_table",
    "install_connection_guards",
    "keeper_database_uuid",
    "keeper_metadata_path",
    "load_parquet_stages",
    "null_order_rank",
    "null_safe_difference",
    "release_lease",
    "stamp_store_metadata",
    "swap_finalizer_map",
    "validate_bulk_tables",
    "validate_metadata_table",
    "verify_bulk_integrity",
    "verify_clickhouse_connection",
    "verify_lease",
    "write_ingest_marker",
]


@compiles(Select, "clickhousedb")
def _compile_clickhouse_select(element: Select[Any], compiler: Any, **kwargs: Any) -> str:
    """Render ClickHouse's offset form without accidentally combining FETCH and LIMIT."""

    if element._fetch_clause is not None:
        raise ClickHouseUnsupportedQueryError("clickhousedb does not support SQL FETCH; use LIMIT/OFFSET for paging")
    if element._offset_clause is not None and element._limit_clause is None and element._fetch_clause is None:
        element = element.limit(sqlalchemy.literal_column("18446744073709551615"))
    return compiler.visit_select(element, **kwargs)


class ClickHouseUncertainInsertError(RuntimeError):
    """The client lost the acknowledgement for an Arrow stage insert."""


class ClickHouseBulkIntegrityError(RuntimeError):
    """A metadata-derived ClickHouse bulk invariant was violated."""


class ClickHouseUnsupportedQueryError(UnsupportedQueryError):
    """A ClickHouse query is outside the shipped correlated-query profile."""


_BINARY_QUERY_FORMATS_KEY = "_httk_query_formats"
_BINARY_QUERY_FORMATS = {"String": "bytes"}


def _install_binary_query_format_hook() -> None:
    """Teach the pinned clickhouse-connect DBAPI to honor per-query formats."""
    from clickhouse_connect.dbapi import cursor as clickhouse_cursor

    if getattr(clickhouse_cursor.Cursor, "_httk_binary_query_formats", False):
        return
    original_execute = clickhouse_cursor.Cursor.execute

    @functools.wraps(original_execute)
    def execute(self: Any, operation: str, parameters: Any = None, settings: dict[str, Any] | None = None) -> None:
        if settings is None or _BINARY_QUERY_FORMATS_KEY not in settings:
            return original_execute(self, operation, parameters, settings=settings)
        settings = dict(settings)
        query_formats = settings.pop(_BINARY_QUERY_FORMATS_KEY)
        if not parameters and isinstance(operation, str):
            operation = operation.replace("%%", "%")
        query_result = self.client.query(operation, parameters, settings=settings, query_formats=query_formats)
        self.data = query_result.result_set
        self._rowcount = len(self.data)
        self._summary.append(query_result.summary)
        self._ix = 0
        if query_result.column_names:
            self.names = query_result.column_names
            self.types = [item.name for item in query_result.column_types]

    clickhouse_cursor.Cursor.execute = execute  # type: ignore[method-assign]  # deliberate driver monkeypatch
    clickhouse_cursor.Cursor._httk_binary_query_formats = True  # type: ignore[attr-defined]  # our install marker on the driver Cursor


def _statement_selects_binary(statement: Any) -> bool:
    columns = getattr(statement, "selected_columns", ())
    return any(isinstance(getattr(column, "type", None), sqlalchemy.LargeBinary) for column in columns)


def _install_binary_query_event(engine: sqlalchemy.Engine) -> None:
    marker = "_httk_clickhouse_binary_query_event"
    if getattr(engine, marker, False):
        return

    @event.listens_for(engine, "before_cursor_execute")
    def _mark_binary_query(
        _connection: Any, _cursor: Any, _statement: str, _parameters: Any, context: Any, _many: bool
    ) -> None:
        statement = getattr(context, "invoked_statement", None)
        if statement is None or not _statement_selects_binary(statement):
            return
        options = dict(context.execution_options)
        settings = dict(options.get("settings") or {})
        settings[_BINARY_QUERY_FORMATS_KEY] = dict(_BINARY_QUERY_FORMATS)
        options["settings"] = settings
        context.execution_options = sqlalchemy.util.immutabledict(options)

    setattr(engine, marker, True)


def normalize_clickhouse_value(value: Any, type_: Any) -> Any:
    """Decode adapter-owned String reads while preserving raw binary bytes."""
    if isinstance(value, bytes) and not isinstance(type_, sqlalchemy.LargeBinary):
        return value.decode("utf-8")
    return value


def _q(name: str) -> str:
    """Quote an identifier owned by the storage protocol."""
    return '"' + name.replace('"', '""') + '"'


def null_safe_difference(left: str, right: str) -> str:
    """The ClickHouse S9 comparator (true precisely when values differ)."""
    return f"(xor(isNull({left}), isNull({right})) OR ifNull({left} != {right}, false))"


class _ClickHouseNullRank(ColumnElement[int]):
    """Dialect-local rank expression with explicit NULL and NaN buckets."""

    inherit_cache = True
    type = sqlalchemy.Integer()

    def __init__(self, element: ColumnElement[Any], nulls: str) -> None:
        super().__init__()
        self.element = element
        self.nulls = nulls
        self.has_nan = isinstance(element.type, sqlalchemy.Float)


@compiles(_ClickHouseNullRank, "clickhousedb")
def _compile_clickhouse_null_rank(element: _ClickHouseNullRank, compiler: Any, **kwargs: Any) -> str:
    value = compiler.process(element.element, **kwargs)
    if element.nulls == "first":
        if element.has_nan:
            return f"if(isNull({value}), 0, if(isNaN({value}), 2, 1))"
        return f"if(isNull({value}), 0, 1)"
    if element.has_nan:
        return f"if(isNull({value}), 2, if(isNaN({value}), 1, 0))"
    return f"if(isNull({value}), 1, 0)"


def null_order_rank(column: ColumnElement[Any], nulls: str, *, dialect_name: str) -> ColumnElement[int]:
    """Return a deterministic NULL/NaN rank for one result-order expression."""
    if dialect_name == "clickhousedb":
        return _ClickHouseNullRank(column, nulls)
    null_rank = 0 if nulls == "first" else 1
    value_rank = 1 - null_rank
    return sqlalchemy.case((column.is_(None), null_rank), else_=value_rank)


def _fraction_component(text: str, component: str) -> str:
    """Parse one already-validated textual fraction component as Int256."""
    del component
    return f"toInt256({text})"


def _fraction_text_component(argument: str, component: str) -> str:
    """Return one fraction component as text without narrowing it first."""
    text = f"toString({argument})"
    index = 1 if component == "numerator" else 2
    default = "'1'" if component == "denominator" else text
    return f"if(position({text}, '/') = 0, {default}, splitByChar('/', {text})[{index}])"


def _fraction_inline(arguments: list[str]) -> str:
    """Render the G0-probed four-fraction, bounded Int256 equality."""
    if len(arguments) != 4:
        raise TypeError("httk_fraction_scaled_equal expects four fraction arguments")
    text_fractions = [
        (
            _fraction_text_component(argument, "numerator"),
            _fraction_text_component(argument, "denominator"),
        )
        for argument in arguments
    ]
    lengths = [
        (
            f"length(replaceRegexpAll({numerator}, '^[-+]', ''))",
            f"length(replaceRegexpAll({denominator}, '^[-+]', ''))",
        )
        for numerator, denominator in text_fractions
    ]
    over_budget = " OR ".join(f"({length}) > 19" for pair in lengths for length in pair)
    zero_denominators = " OR ".join(
        f"(replaceRegexpAll({denominator}, '^[-+]', '') = '0')" for _numerator, denominator in text_fractions
    )
    fractions = [
        (_fraction_component(numerator, "numerator"), _fraction_component(denominator, "denominator"))
        for numerator, denominator in text_fractions
    ]
    left = " * ".join((fractions[0][0], fractions[1][0], fractions[2][1], fractions[3][1]))
    right = " * ".join((fractions[2][0], fractions[3][0], fractions[0][1], fractions[1][1]))
    return (
        "if("
        f"({over_budget}), throwIf(1, 'httk_fraction_scaled_equal: Int256 component exceeds 19 digits'), "
        f"if(({zero_denominators}), throwIf(1, 'httk_fraction_scaled_equal: zero denominator'), "
        f"(({left}) = ({right})))"
        ")"
    )


@compiles(Function, "clickhousedb")
def _compile_clickhouse_function(element: Function, compiler: Any, **kwargs: Any) -> str:
    if element.name == "httk_fraction_scaled_equal":
        arguments = [compiler.process(argument, **kwargs) for argument in element.clause_expr.clauses]
        return _fraction_inline(arguments)
    return compiler.visit_function(element, **kwargs)


@compiles(BinaryExpression, "clickhousedb")
def _compile_clickhouse_binary(element: BinaryExpression[Any], compiler: Any, **kwargs: Any) -> str:
    left = compiler.process(element.left, **kwargs)
    right = compiler.process(element.right, **kwargs)
    if element.operator in {operators.like_op, operators.not_like_op}:
        operator = "NOT LIKE" if element.operator is operators.not_like_op else "LIKE"
        return f"{left} {operator} {right}"
    if element.operator is operators.is_distinct_from:
        return null_safe_difference(left, right)
    if element.operator is operators.is_not_distinct_from:
        return f"NOT {null_safe_difference(left, right)}"
    return compiler.visit_binary(element, **kwargs)


def _stage_table(source: sqlalchemy.Table, name: str) -> sqlalchemy.Table:
    """Clone record metadata for a disposable MergeTree stage relation."""
    metadata = sqlalchemy.MetaData()
    stage = source.to_metadata(metadata, name=name)
    # Stage rows retain the ordinary ``sid`` spelling.  It is their stage sid,
    # and ``_order_by`` deliberately gives it the deterministic MergeTree key.
    stage.info["_httk_stage"] = True
    return stage


def _client_for_url(url: sqlalchemy.URL) -> Any:
    """Build the separate Arrow-loading client with the G0 join setting."""
    import clickhouse_connect

    if not url.host:
        raise RuntimeError("ClickHouse client-stream staging requires a URL host")
    client = clickhouse_connect.get_client(
        host=url.host,
        port=url.port or 8123,
        username=url.username or "default",
        password=url.password or "",
        database=url.database or "default",
        settings={"join_use_nulls": 1},
    )
    value = client.query("SELECT getSetting('join_use_nulls')").result_rows
    if not value or not _setting_enabled(value[0][0]):
        client.close()
        raise RuntimeError("ClickHouse Arrow loader could not enforce join_use_nulls=1")
    return client


def load_parquet_stages(store: Any, manifests: list[Any]) -> dict[str, str]:
    """Create and stream every Parquet shard into local ClickHouse stage tables.

    Every shard is verified by its before/after row count.  This makes a
    transport retry or a response lost after send observable before the next
    shard is admitted.
    """
    try:
        from pyarrow import parquet
    except ImportError as error:  # pragma: no cover - P3 dependency gate
        raise ImportError("ClickHouse bulk_ingest needs pyarrow for Parquet staging") from error

    files: dict[str, list[str]] = {}
    for manifest in manifests:
        for table, paths in manifest.shards.items():
            if isinstance(paths, list):
                files.setdefault(table, []).extend(str(path) for path in paths)
    stage_names = {table: f"_httk_stage_{table}" for table in files}
    engine = store._database.engine
    with engine.begin() as connection:
        for table, stage_name in stage_names.items():
            if table == "_httk_roots":
                connection.execute(
                    sqlalchemy.text(
                        f"CREATE TABLE {_q(stage_name)} (token Int64, tbl String, stage_sid Int64) "
                        "ENGINE = MergeTree ORDER BY stage_sid"
                    )
                )
            elif table == "_httk_dispatch_payload":
                connection.execute(
                    sqlalchemy.text(
                        f"CREATE TABLE {_q(stage_name)} "
                        "(dispatch_name String, content_id String, column String, block_sid Int64) "
                        "ENGINE = MergeTree ORDER BY (dispatch_name, content_id)"
                    )
                )
            elif table == "_httk_nan_content":
                connection.execute(
                    sqlalchemy.text(
                        f"CREATE TABLE {_q(stage_name)} "
                        "(table_name String, content_id String, field_name String) "
                        "ENGINE = MergeTree ORDER BY (table_name, content_id)"
                    )
                )
            else:
                connection.execute(sqlalchemy.schema.CreateTable(_stage_table(store._table(table), stage_name)))

    client = _client_for_url(engine.url)
    try:
        for table, paths in files.items():
            for path in paths:
                arrow = parquet.read_table(path)
                rows = arrow.num_rows
                before = int(client.query(f"SELECT count() FROM {_q(stage_names[table])}").result_rows[0][0])
                try:
                    client.insert_arrow(stage_names[table], arrow)
                except BaseException as error:
                    after = int(client.query(f"SELECT count() FROM {_q(stage_names[table])}").result_rows[0][0])
                    if after == before + rows:
                        continue
                    if after != before:
                        raise ClickHouseUncertainInsertError(
                            f"ClickHouse Arrow stage insert has ambiguous row count for {table!r}"
                        ) from error
                    try:
                        client.insert_arrow(stage_names[table], arrow)
                    except BaseException as retry_error:
                        after = int(client.query(f"SELECT count() FROM {_q(stage_names[table])}").result_rows[0][0])
                        if after == before + rows:
                            continue
                        raise ClickHouseUncertainInsertError(
                            f"ClickHouse Arrow stage retry has ambiguous row count for {table!r}"
                        ) from retry_error
                after = int(client.query(f"SELECT count() FROM {_q(stage_names[table])}").result_rows[0][0])
                if after != before + rows:
                    raise ClickHouseUncertainInsertError(
                        f"ClickHouse Arrow stage insert changed {table!r} by {after - before}, expected {rows}"
                    )
    except ClickHouseUncertainInsertError:
        raise
    except BaseException as error:
        raise RuntimeError(f"ClickHouse Arrow stage insert failed cleanly: {error}") from error
    finally:
        client.close()
    return stage_names


def swap_finalizer_map(finalizer: Any, table: str, candidate: str) -> None:
    """Atomically replace a finalizer map and update its bookkeeping.

    ClickHouse has no transactional UPDATE protocol for this use.  The
    candidate is checked for the same stage key cardinality, renamed over the
    old name in one RENAME statement, and the retired relation is dropped.
    Keeping the public map name stable means every already-constructed query
    continues to address the current map.
    """
    old = finalizer.maps[table]
    retired = f"{old}_retired"
    connection = finalizer.connection
    old_count = int(connection.execute(sqlalchemy.text(f"SELECT count(*) FROM {_q(old)}")).scalar_one())
    new_count = int(connection.execute(sqlalchemy.text(f"SELECT count(*) FROM {_q(candidate)}")).scalar_one())
    if old_count != new_count:
        raise RuntimeError(f"ClickHouse finalizer map swap for {table!r} changed stage-key cardinality")
    finalizer.ingest._before_clickhouse_map_swap(table, "before-rename")
    connection.execute(sqlalchemy.text(f"RENAME TABLE {_q(old)} TO {_q(retired)}, {_q(candidate)} TO {_q(old)}"))
    # The rename is durable.  Make cleanup bookkeeping reflect both live
    # relations before exposing the post-rename fault seam.
    with contextlib.suppress(ValueError):
        finalizer.objects.remove(candidate)
    if retired not in finalizer.objects:
        finalizer.objects.append(retired)
    finalizer.ingest._before_clickhouse_map_swap(table, "after-rename")
    finalizer.ingest._before_clickhouse_map_swap(table, "before-drop")
    connection.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {_q(retired)}"))
    exists = connection.execute(
        sqlalchemy.text("SELECT count() FROM system.tables WHERE database = currentDatabase() AND name = :name"),
        {"name": retired},
    ).scalar_one()
    if exists:
        raise RuntimeError(f"ClickHouse finalizer map swap for {table!r} did not remove retired map")
    with contextlib.suppress(ValueError):
        finalizer.objects.remove(retired)
    finalizer.ingest._before_clickhouse_map_swap(table, "after-drop")


def validate_bulk_tables(connection: sqlalchemy.Connection, tables: list[sqlalchemy.Table]) -> None:
    """Validate ClickHouse's non-enforcing physical and logical constraints."""
    for table in tables:
        name = table.name
        info = connection.execute(
            sqlalchemy.text(
                "SELECT engine, sorting_key, primary_key FROM system.tables "
                "WHERE database = currentDatabase() AND name = :name"
            ),
            {"name": name},
        ).one_or_none()
        if "sid" in table.c:
            expected_key = "sid"
        elif "content_id" in table.c:
            expected_key = "content_id"
        else:
            index_columns = [column.name for column in table.columns if column.name.endswith("_index")]
            parent_columns = [column.name for column in table.columns if column.name.endswith("_sid")]
            expected_key = ",".join(parent_columns[:1] + index_columns[:1])
        canonical = lambda value: str(value).replace("`", "").replace('"', "").replace(" ", "").strip("()")
        if (
            info is None
            or str(info[0]) != "MergeTree"
            or canonical(info[1]) != expected_key
            or canonical(info[2]) != expected_key
        ):
            raise RuntimeError(f"ClickHouse bulk physical validation failed for {name!r}: MergeTree sorting key")
        actual = connection.execute(
            sqlalchemy.text(
                "SELECT name, type, default_kind, default_expression FROM system.columns "
                "WHERE database = currentDatabase() AND table = :table ORDER BY position"
            ),
            {"table": name},
        ).all()
        expected = [
            (column.name, str(_ch_type(column).compile(dialect=connection.dialect)), "", "") for column in table.columns
        ]
        normalized = [
            (str(column), str(type_name), str(kind or ""), str(expression or ""))
            for column, type_name, kind, expression in actual
        ]
        if normalized != expected:
            raise RuntimeError(
                f"ClickHouse bulk physical validation failed for {name!r}: column types/nullability/defaults"
            )
        extras = (
            connection.execute(
                sqlalchemy.text(
                    "SELECT name FROM system.data_skipping_indices WHERE database = currentDatabase() AND table = :name"
                ),
                {"name": name},
            )
            .scalars()
            .all()
        )
        if extras:
            raise RuntimeError(f"ClickHouse bulk physical validation failed for {name!r}: unexpected indexes")


def verify_bulk_integrity(connection: sqlalchemy.Connection, tables: list[sqlalchemy.Table]) -> None:
    """Run the SQLAlchemy-metadata-derived logical constraints ClickHouse cannot enforce."""
    for table in tables:
        name = _q(table.name)
        columns = {column.name for column in table.columns}
        if "sid" in columns:
            duplicate = connection.execute(
                sqlalchemy.text(f"SELECT sid FROM {name} GROUP BY sid HAVING count() > 1 LIMIT 1")
            ).first()
            count, low, high = connection.execute(
                sqlalchemy.text(f"SELECT count(), min(sid), max(sid) FROM {name}")
            ).one()
            if duplicate is not None or (count and (int(low) != 1 or int(high) != int(count))):
                raise ClickHouseBulkIntegrityError(
                    f"ClickHouse bulk integrity failed for {table.name!r}: sid uniqueness/density"
                )
            if "content_id" in columns:
                duplicate = connection.execute(
                    sqlalchemy.text(f"SELECT content_id FROM {name} GROUP BY content_id HAVING count() > 1 LIMIT 1")
                ).first()
                if duplicate is not None:
                    raise ClickHouseBulkIntegrityError(
                        f"ClickHouse bulk integrity failed for {table.name!r}: content_id uniqueness"
                    )
            if "_httk_role" in columns:
                invalid = connection.execute(
                    sqlalchemy.text(f"SELECT 1 FROM {name} WHERE _httk_role NOT IN (0, 1) LIMIT 1")
                ).first()
                if invalid is not None:
                    raise ClickHouseBulkIntegrityError(
                        f"ClickHouse bulk integrity failed for {table.name!r}: _httk_role domain"
                    )
        unique_sets = {
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, sqlalchemy.UniqueConstraint)
        }
        unique_sets.add(tuple(column.name for column in table.primary_key.columns))
        unique_sets.update(tuple(column.name for column in index.columns) for index in table.indexes if index.unique)
        is_dispatch = "sid" not in columns and "content_id" in columns
        unique_sets.update((column.name,) for column in table.columns if column.unique)
        for unique_columns in unique_sets:
            if not unique_columns:
                continue
            predicate = " AND ".join(f"{_q(column)} IS NOT NULL" for column in unique_columns)
            grouping = ", ".join(_q(column) for column in unique_columns)
            duplicate = connection.execute(
                sqlalchemy.text(
                    f"SELECT 1 FROM {name} WHERE {predicate} GROUP BY {grouping} HAVING count() > 1 LIMIT 1"
                )
            ).first()
            if duplicate is not None:
                raise ClickHouseBulkIntegrityError(
                    f"ClickHouse bulk integrity failed for {table.name!r}: unique {unique_columns!r}"
                )
        # Entry dispatch relations have a content-id key and nullable backing
        # sid columns.  Metadata supplies the exact-one check as a named CHECK.
        if is_dispatch:
            backing = [column for column in columns if column.endswith("_sid")]
            if backing:
                expression = " + ".join(f"if({_q(column)} IS NULL, 0, 1)" for column in backing)
                invalid = connection.execute(
                    sqlalchemy.text(f"SELECT 1 FROM {name} WHERE ({expression}) != 1 LIMIT 1")
                ).first()
                if invalid is not None:
                    raise ClickHouseBulkIntegrityError(
                        f"ClickHouse bulk integrity failed for {table.name!r}: dispatch exactly-one"
                    )


def _keeper_engine(path: str, *, primary_key: str = "key") -> Any:
    """Build the clickhouse-connect table-engine object without importing it at module load."""
    from clickhouse_connect.cc_sqlalchemy.ddl.tableengine import TableEngine

    class KeeperMap(TableEngine):
        arg_names: Sequence[str] = ["path"]
        quoted_args: set[str] = {"path"}  # noqa: RUF012 — must match TableEngine's set[str] instance-var type exactly (ClassVar breaks the override for mypy/pyright)
        eng_params: Sequence[str] = ["primary_key"]

        def __init__(self, keeper_path: str, key: str) -> None:
            super().__init__({"path": keeper_path, "primary_key": key})

    return KeeperMap(path, primary_key)


def keeper_metadata_path(database_uuid: str) -> str:
    """Return the KeeperMap path scoped by the ClickHouse database UUID."""
    try:
        identity = str(uuid.UUID(str(database_uuid)))
    except (AttributeError, ValueError) as error:
        raise RuntimeError(f"cannot derive a KeeperMap path from invalid database UUID {database_uuid!r}") from error
    if identity == str(uuid.UUID(int=0)):
        raise RuntimeError("cannot derive a KeeperMap path from the nil ClickHouse database UUID")
    return f"/{identity}/_httk_store_metadata"


def _database_name(connection: sqlalchemy.Connection) -> str:
    database = connection.engine.url.database
    if database:
        return str(database)
    return str(connection.execute(sqlalchemy.text("SELECT currentDatabase()")).scalar_one())


def keeper_database_uuid(connection: sqlalchemy.Connection) -> str:
    """Return the stable UUID assigned to the current ClickHouse database."""
    row = connection.execute(
        sqlalchemy.text("SELECT uuid FROM system.databases WHERE name = currentDatabase()")
    ).one_or_none()
    if row is None or row[0] is None:
        raise RuntimeError("ClickHouse system.databases.uuid is missing; refusing the KeeperMap backend")
    try:
        identity = uuid.UUID(str(row[0]))
    except (AttributeError, ValueError) as error:
        raise RuntimeError(
            f"ClickHouse system.databases.uuid is malformed ({row[0]!r}); refusing the KeeperMap backend"
        ) from error
    if identity.int == 0:
        raise RuntimeError("ClickHouse system.databases.uuid is nil; refusing the KeeperMap backend")
    return str(identity)


def _ch_type(column: sqlalchemy.Column[Any]) -> Any:
    """Compile one generic SQLAlchemy type into the fixed P1 ClickHouse vocabulary."""
    from clickhouse_connect.cc_sqlalchemy import types

    generic = column.type
    if isinstance(generic, sqlalchemy.Boolean):
        result = types.Bool()
    elif isinstance(generic, sqlalchemy.Integer):
        result = types.Int64()
    elif isinstance(generic, sqlalchemy.Float):
        result = types.Float64()
    elif isinstance(generic, (sqlalchemy.LargeBinary, sqlalchemy.Text, sqlalchemy.String)):
        result = types.String()
    else:
        if isinstance(generic, sqlalchemy.types.NullType):
            raise TypeError(f"ClickHouse DDL cannot compile untyped column {column.name!r}")
        raise TypeError(f"ClickHouse DDL has no P1 type mapping for {type(generic).__name__}")
    return types.Nullable(result) if column.nullable else result


def _order_by(table: sqlalchemy.Table) -> Any:
    columns = table.c
    # Stage rows are append-only and must sort by stage_sid when that key is present.
    if "stage_sid" in columns:
        return "stage_sid"
    if "sid" in columns:
        return "sid"
    if "content_id" in columns:
        return "content_id"
    index_columns = [column for column in columns if column.name.endswith("_index")]
    parent_columns = [column for column in columns if column.name.endswith("_sid")]
    if index_columns and parent_columns:
        return (parent_columns[0].name, index_columns[0].name)
    raise TypeError(f"ClickHouse DDL cannot derive an ORDER BY key for table {table.name!r}")


def _clone_table(table: sqlalchemy.Table) -> sqlalchemy.Table:
    """Copy a source table into private metadata for dialect-local decoration."""
    metadata = sqlalchemy.MetaData()
    clone = table.to_metadata(metadata)
    clone.info.update(table.info)
    return clone


def decorate_table(
    table: sqlalchemy.Table,
    *,
    database: str = "default",
    database_uuid: str | None = None,
) -> sqlalchemy.Table:
    """Return a decorated private clone; the source table is never modified."""
    clone = _clone_table(table)
    for column in clone.columns:
        column.type = _ch_type(column)
        if column.name == "sid":
            column.default = None
            column.server_default = None
            column.autoincrement = False
    if clone.info.get(_METADATA_MARKER):
        identity = database_uuid or clone.info.get("httk_clickhouse_database_uuid")
        if identity is None:
            raise RuntimeError("ClickHouse metadata DDL requires the current database UUID")
        clone.engine = _keeper_engine(keeper_metadata_path(str(identity)))  # type: ignore[attr-defined]  # clickhouse-connect dialect reads Table.engine
    else:
        from clickhouse_connect.cc_sqlalchemy.ddl.tableengine import MergeTree

        clone.engine = MergeTree(order_by=_order_by(clone))  # type: ignore[attr-defined]  # clickhouse-connect dialect reads Table.engine
    clone.info["_httk_clickhouse_decorated"] = True
    return clone


@compiles(sqlalchemy.schema.CreateTable, "clickhousedb")
def _compile_clickhouse_create_table(element: sqlalchemy.schema.CreateTable, compiler: Any, **kwargs: Any) -> str:
    source = element.element
    clone = decorate_table(
        source,
        database_uuid=source.info.get("httk_clickhouse_database_uuid"),
    )
    clone_element = sqlalchemy.schema.CreateTable(clone, if_not_exists=element.if_not_exists)
    rendered = compiler.visit_create_table(clone_element, **kwargs)
    checks = [constraint for constraint in clone.constraints if isinstance(constraint, sqlalchemy.CheckConstraint)]
    if not checks:
        return rendered
    rendered_checks = ", ".join(
        f"CONSTRAINT {compiler.preparer.quote(str(constraint.name))} CHECK {constraint.sqltext}"
        for constraint in checks
    )
    close = rendered.rfind(") ")
    if close < 0:
        raise RuntimeError(f"ClickHouse DDL compiler did not render a table body for {source.name!r}")
    return f"{rendered[:close]}, {rendered_checks}{rendered[close:]}"


@compiles(sqlalchemy.schema.CreateIndex, "clickhousedb")
def _compile_clickhouse_index(_: sqlalchemy.schema.CreateIndex, __: Any, **___: Any) -> str:
    """Keep logical mapping indexes out of ClickHouse's physical schema."""
    return "SELECT 1"


def _raw_verify(dbapi_connection: Any) -> tuple[str, Any]:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SELECT version(), getSetting('join_use_nulls')")
        row = cursor.fetchone()
    finally:
        cursor.close()
    if row is None or len(row) != 2:
        raise RuntimeError("ClickHouse connection did not return version and join_use_nulls")
    return str(row[0]), row[1]


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split(".")[:4])
    except ValueError as error:
        raise RuntimeError(f"ClickHouse returned an invalid server version {version!r}") from error


def _setting_enabled(value: Any) -> bool:
    return value is True or value == 1 or (isinstance(value, str) and value.lower() in {"1", "true"})


def verify_clickhouse_connection(connection: sqlalchemy.Connection) -> str:
    """Verify version and the NULL-producing outer-join setting on one connection."""
    version, join_use_nulls = connection.execute(
        sqlalchemy.text("SELECT version(), getSetting('join_use_nulls')")
    ).one()
    version = str(version)
    if _version_tuple(version) < _MIN_SERVER_VERSION:
        required = ".".join(str(part) for part in _MIN_SERVER_VERSION)
        raise RuntimeError(f"ClickHouse server {version} is too old; httk-store requires ClickHouse >= {required}")
    if not _setting_enabled(join_use_nulls):
        raise RuntimeError("ClickHouse join_use_nulls=1 could not be enforced; refusing the ClickHouse backend")
    return version


def install_connection_guards(engine: sqlalchemy.Engine) -> str:
    """Install pool checkout verification and return the first server version."""
    _install_binary_query_format_hook()
    _install_binary_query_event(engine)
    state = getattr(engine, "_httk_clickhouse_guard", None)
    if state is None:
        state = {"version": None, "lock": threading.Lock()}

        @event.listens_for(engine, "checkout")
        def _verify_checkout(dbapi_connection: Any, *_: Any) -> None:
            version, join_use_nulls = _raw_verify(dbapi_connection)
            if _version_tuple(version) < _MIN_SERVER_VERSION:
                required = ".".join(str(part) for part in _MIN_SERVER_VERSION)
                raise RuntimeError(
                    f"ClickHouse server {version} is too old; httk-store requires ClickHouse >= {required}"
                )
            if not _setting_enabled(join_use_nulls):
                raise RuntimeError("ClickHouse join_use_nulls=1 could not be enforced; refusing the ClickHouse backend")
            with state["lock"]:
                if state["version"] is None:
                    state["version"] = version

        engine._httk_clickhouse_guard = state  # type: ignore[attr-defined]
    with engine.connect() as connection:
        return verify_clickhouse_connection(connection)


def _bootstrap_table() -> sqlalchemy.Table:
    metadata = sqlalchemy.MetaData()
    return sqlalchemy.Table(
        _BOOTSTRAP_TABLE,
        metadata,
        sqlalchemy.Column("key", sqlalchemy.Text, primary_key=True, nullable=False),
        sqlalchemy.Column("value", sqlalchemy.Text, nullable=False),
    )


def _metadata_table() -> sqlalchemy.Table:
    metadata = sqlalchemy.MetaData()
    return sqlalchemy.Table(
        "_httk_store_metadata",
        metadata,
        sqlalchemy.Column("key", sqlalchemy.Text, primary_key=True, nullable=False),
        sqlalchemy.Column("value", sqlalchemy.Text, nullable=False),
    )


def _metadata_value(connection: sqlalchemy.Connection, key: str) -> str | None:
    table = _metadata_table()
    return connection.execute(sqlalchemy.select(table.c.value).where(table.c.key == key)).scalar_one_or_none()


def _lease_description(value: str) -> str:
    try:
        parsed = json.loads(value)
        acquired = datetime.datetime.fromisoformat(str(parsed["acquired_at"]))
        age = datetime.datetime.now(datetime.UTC) - acquired.astimezone(datetime.UTC)
        return f"holder {parsed['owner']!r}, age {age}"
    except (KeyError, TypeError, ValueError, OverflowError):
        return f"holder value {value!r} (unparseable age)"


def _manual_lease_recovery(key: str, value: str | None) -> str:
    if value is None:
        return (
            "manual recovery (lease): inspect _httk_store_metadata for a stale lease, "
            "verify the writer is dead, then issue an exact-value lease DELETE"
        )
    escaped = value.replace("'", "''")
    return (
        "manual recovery (lease): after verifying the writer is dead, issue "
        f"SET keeper_map_strict_mode = 1; DELETE FROM _httk_store_metadata WHERE key = '{key}' AND value = '{escaped}'"
    )


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its DBAPI/SQLAlchemy causes once each."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attribute in ("orig", "__cause__", "__context__"):
            cause = getattr(current, attribute, None)
            if isinstance(cause, BaseException):
                pending.append(cause)


def _keeper_error_code(error: BaseException) -> int | None:
    for current in _exception_chain(error):
        try:
            code = getattr(current, "code", None)
        except BaseException:
            code = None
        if code is None:
            continue
        try:
            return int(code)
        except (TypeError, ValueError):
            continue
    return None


def _is_keeper_node_exists(error: BaseException) -> bool:
    code = _keeper_error_code(error)
    if code is not None:
        return code == 999
    text = " ".join(str(current) for current in _exception_chain(error)).casefold()
    return "node exists" in text or "already exists" in text


def _strict_insert(
    connection: sqlalchemy.Connection,
    key: str,
    value: str,
) -> None:
    table = _metadata_table()
    connection.execute(
        sqlalchemy.insert(table).values(key=key, value=value).execution_options(settings={"keeper_map_strict_mode": 1})
    )


def _strict_delete(table: sqlalchemy.Table, *conditions: Any) -> sqlalchemy.Delete:
    """Build a value-conditioned KeeperMap delete with atomic strict semantics."""
    return sqlalchemy.delete(table).where(*conditions).execution_options(settings={"keeper_map_strict_mode": 1})


def acquire_lease(connection: sqlalchemy.Connection, owner: str) -> str:
    """Strictly acquire the store lease and return its complete JSON value."""
    token = uuid.uuid4().hex
    payload = json.dumps(
        {
            "owner": owner,
            "token": token,
            "acquired_at": datetime.datetime.now(datetime.UTC).isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        _strict_insert(connection, _LEASE_KEY, payload)
    except BaseException as error:
        held = None
        diagnostic_error: BaseException | None = None
        try:
            held = _metadata_value(connection, _LEASE_KEY)
        except BaseException as diagnostic:
            diagnostic_error = diagnostic
        if held is not None and _is_keeper_node_exists(error):
            raise RuntimeError(
                f"ClickHouse lease is held ({_lease_description(str(held))}); "
                f"{_manual_lease_recovery(_LEASE_KEY, str(held))}"
            ) from error
        if held is not None:
            raise RuntimeError(
                f"ClickHouse lease acquisition failed; {_lease_description(str(held))}; "
                f"{_manual_lease_recovery(_LEASE_KEY, str(held))}"
            ) from error
        raise RuntimeError(
            f"ClickHouse lease acquisition failed; Keeper is required; {_manual_lease_recovery(_LEASE_KEY, None)}"
            + (f" (diagnostic read failed: {diagnostic_error})" if diagnostic_error is not None else "")
        ) from error
    return payload


def verify_lease(connection: sqlalchemy.Connection, lease_value: str) -> None:
    """Verify that the current KeeperMap lease still contains ``lease_value``."""
    current = _metadata_value(connection, _LEASE_KEY)
    if current != lease_value:
        holder = "missing lease" if current is None else _lease_description(str(current))
        raise RuntimeError(f"ClickHouse lease ownership was lost ({holder})")


def release_lease(connection: sqlalchemy.Connection, lease_value: str | None) -> None:
    """Idempotently delete only the exact lease value owned by this writer."""
    if lease_value is None:
        return
    table = _metadata_table()
    connection.execute(_strict_delete(table, table.c.key == _LEASE_KEY, table.c.value == lease_value))


def write_ingest_marker(connection: sqlalchemy.Connection, lease_value: str) -> str:
    """Strictly insert the token-carrying bulk-ingest marker."""
    verify_lease(connection, lease_value)
    lease = json.loads(lease_value)
    marker = json.dumps(
        {
            "state": "bulk-ingest",
            "token": lease["token"],
            "acquired_at": lease["acquired_at"],
            "nonce": uuid.uuid4().hex,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        _strict_insert(connection, _INGEST_STATE_KEY, marker)
    except BaseException:
        # KeeperMap writes are durable independently of the client response.
        # If an ambiguous insert did land this exact token, remove only that
        # observed value; a different marker is foreign crash residue.
        try:
            observed = _metadata_value(connection, _INGEST_STATE_KEY)
        except BaseException:
            observed = None
        if observed == marker:
            try:
                clear_ingest_marker(connection, marker)
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "ClickHouse ingest marker insert was ambiguous and exact-token cleanup failed"
                ) from cleanup_error
        raise
    return marker


def clear_ingest_marker(connection: sqlalchemy.Connection, marker_value: str | None) -> None:
    """Idempotently clear only the exact marker value from this ingest."""
    if marker_value is None:
        return
    table = _metadata_table()
    connection.execute(_strict_delete(table, table.c.key == _INGEST_STATE_KEY, table.c.value == marker_value))


def _expected_bootstrap_engine() -> str:
    return f"KeeperMap('{_BOOTSTRAP_PATH}') PRIMARY KEY key"


def _validate_bootstrap_table(connection: sqlalchemy.Connection) -> None:
    rows = connection.execute(
        sqlalchemy.text(
            "SELECT engine, engine_full FROM system.tables WHERE database = currentDatabase() AND name = :name"
        ),
        {"name": _BOOTSTRAP_TABLE},
    ).all()
    if not rows:
        raise RuntimeError(
            f"{_BOOTSTRAP_LOCK_MESSAGE}: deployment table {_BOOTSTRAP_TABLE!r} is absent; "
            "create it before opening the backend"
        )
    if any(str(row[0]) != "KeeperMap" or str(row[1]) != _expected_bootstrap_engine() for row in rows):
        raise RuntimeError(f"{_BOOTSTRAP_LOCK_MESSAGE}: {_BOOTSTRAP_TABLE!r} has the wrong engine or Keeper path")
    connection.execute(sqlalchemy.text("SELECT count(*) FROM _httk_bootstrap")).scalar_one()


def ensure_bootstrap_table(engine: sqlalchemy.Engine) -> None:
    """Require and validate the deployment-created KeeperMap bootstrap table."""
    try:
        with engine.begin() as connection:
            _validate_bootstrap_table(connection)
    except RuntimeError:
        raise
    except BaseException as error:
        raise RuntimeError(f"{_BOOTSTRAP_LOCK_MESSAGE}: {error}") from error


@contextmanager
def bootstrap_fence(connection: sqlalchemy.Connection) -> Iterator[tuple[str, str]]:
    """Hold the strict UUID-keyed bootstrap fence across one complete operation."""
    try:
        _validate_bootstrap_table(connection)
    except RuntimeError:
        raise
    except BaseException as error:
        raise RuntimeError(f"{_BOOTSTRAP_LOCK_MESSAGE}: {error}") from error
    key = keeper_database_uuid(connection)
    token = uuid.uuid4().hex
    bootstrap = _bootstrap_table()
    lock_statement = (
        sqlalchemy.insert(bootstrap)
        .values(key=key, value=token)
        .execution_options(settings={"keeper_map_strict_mode": 1})
    )
    try:
        connection.execute(lock_statement)
    except BaseException as error:
        held: Any = None
        diagnostic_error: BaseException | None = None
        try:
            held = connection.execute(
                sqlalchemy.text("SELECT value FROM _httk_bootstrap WHERE key = :key"), {"key": key}
            ).scalar_one_or_none()
        except BaseException as diagnostic:
            diagnostic_error = diagnostic
        cleanup_error: BaseException | None = None
        if held == token:
            try:
                connection.execute(_strict_delete(bootstrap, bootstrap.c.key == key, bootstrap.c.value == token))
            except BaseException as cleanup:
                cleanup_error = cleanup
        if held == token:
            recovery = (
                "the fresh token landed despite the failed INSERT; exact-value cleanup was attempted; "
                f"manual recovery is SET keeper_map_strict_mode = 1; DELETE FROM _httk_bootstrap WHERE key = '{key}' AND value = '{token}'"
            )
        elif held is not None:
            recovery = (
                f"manual recovery: after verifying the writer is dead, "
                f"SET keeper_map_strict_mode = 1; DELETE FROM _httk_bootstrap WHERE key = '{key}' AND value = '{held}'"
            )
        else:
            recovery = "manual recovery: inspect _httk_bootstrap for a stale UUID-keyed row"
        diagnostics = []
        if diagnostic_error is not None:
            diagnostics.append(f"diagnostic read failed: {diagnostic_error}")
        if cleanup_error is not None:
            diagnostics.append(f"fresh-token cleanup failed: {cleanup_error}")
        suffix = f" ({'; '.join(diagnostics)})" if diagnostics else ""
        raise RuntimeError(
            f"{_BOOTSTRAP_LOCK_MESSAGE}: database UUID {key!r} acquisition failed; {recovery}{suffix}"
        ) from error
    try:
        yield key, token
    finally:
        try:
            connection.execute(_strict_delete(bootstrap, bootstrap.c.key == key, bootstrap.c.value == token))
        except BaseException as error:
            raise RuntimeError(
                f"ClickHouse Keeper bootstrap lock could not be released; {_BOOTSTRAP_LOCK_MESSAGE}; "
                f"manual recovery requires the exact key/value DELETE for {key!r}"
            ) from error


def _stamp_rows(connection: sqlalchemy.Connection, metadata_table: sqlalchemy.Table, rows: Mapping[str, str]) -> None:
    for metadata_key, value in rows.items():
        connection.execute(
            sqlalchemy.insert(metadata_table)
            .values(key=metadata_key, value=value)
            .execution_options(settings={"keeper_map_strict_mode": 1})
        )


def stamp_store_metadata(
    connection: sqlalchemy.Connection,
    metadata_table: sqlalchemy.Table,
    rows: Mapping[str, str],
    *,
    fence_held: bool = False,
) -> None:
    """Strictly stamp metadata, acquiring the bootstrap fence unless already held."""
    try:
        if fence_held:
            _stamp_rows(connection, metadata_table, rows)
        else:
            with bootstrap_fence(connection):
                _stamp_rows(connection, metadata_table, rows)
    except RuntimeError:
        raise
    except BaseException as error:
        raise RuntimeError(f"ClickHouse KeeperMap metadata stamp failed; Keeper is required: {error}") from error


def validate_metadata_table(connection: sqlalchemy.Connection) -> None:
    """Validate the physical KeeperMap metadata table before any metadata read."""
    database_uuid = keeper_database_uuid(connection)
    expected_path = keeper_metadata_path(database_uuid)
    expected_engine = f"KeeperMap('{expected_path}') PRIMARY KEY key"
    rows = connection.execute(
        sqlalchemy.text(
            "SELECT engine, engine_full, primary_key FROM system.tables "
            "WHERE database = currentDatabase() AND name = :name"
        ),
        {"name": "_httk_store_metadata"},
    ).all()
    if (
        len(rows) != 1
        or str(rows[0][0]) != "KeeperMap"
        or str(rows[0][1]) != expected_engine
        or str(rows[0][2]).replace("`", "").replace('"', "").strip() != "key"
    ):
        raise RuntimeError(
            f"ClickHouse metadata table _httk_store_metadata failed physical validation: expected {expected_engine}"
        )
    columns = connection.execute(
        sqlalchemy.text(
            "SELECT name, type, default_kind, default_expression FROM system.columns "
            "WHERE database = currentDatabase() AND table = :table ORDER BY position"
        ),
        {"table": "_httk_store_metadata"},
    ).all()
    expected_columns = [("key", "String", "", ""), ("value", "String", "", "")]
    actual_columns = [
        (str(name), str(type_name), str(default_kind or ""), str(default_expression or ""))
        for name, type_name, default_kind, default_expression in columns
    ]
    if actual_columns != expected_columns:
        raise RuntimeError(
            "ClickHouse metadata table _httk_store_metadata failed physical validation: "
            f"expected columns {expected_columns!r}"
        )


def actual_schema_objects(connection: sqlalchemy.Connection) -> Mapping[str, frozenset[str]]:
    """Read the current database's application objects from ClickHouse's catalogue."""
    rows = connection.execute(
        sqlalchemy.text(
            "SELECT name, engine FROM system.tables WHERE database = currentDatabase() AND name != :bootstrap"
        ),
        {"bootstrap": _BOOTSTRAP_TABLE},
    )
    result: dict[str, set[str]] = {}
    for name, engine in rows:
        kind = "view" if str(engine).lower() == "view" else "table"
        result.setdefault(str(name), set()).add(kind)
    return {name: frozenset(kinds) for name, kinds in result.items()}


def actual_columns(connection: sqlalchemy.Connection, table_name: str) -> Mapping[str, str]:
    """Return a ClickHouse system.columns name/type mapping for one table."""
    rows = connection.execute(
        sqlalchemy.text(
            "SELECT name, type FROM system.columns "
            "WHERE database = currentDatabase() AND table = :table ORDER BY position"
        ),
        {"table": table_name},
    )
    return {str(name): str(type_name) for name, type_name in rows}
