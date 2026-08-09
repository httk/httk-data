"""ClickHouse-specific SQLAlchemy decoration and KeeperMap primitives.

The rest of :mod:`httk.data.db` deliberately deals in generic SQLAlchemy
objects.  This module is the adapter boundary for ClickHouse's physical types,
MergeTree sorting keys, system catalogue, and KeeperMap metadata protocol.
"""

import datetime
import json
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

import sqlalchemy
from sqlalchemy import event
from sqlalchemy.ext.compiler import compiles

_MIN_SERVER_VERSION = (26, 8, 1, 1028)
_BOOTSTRAP_TABLE = "_httk_bootstrap"
_BOOTSTRAP_PATH = "/_httk_bootstrap"
_METADATA_MARKER = "httk_metadata"
_LEASE_KEY = "lease"
_INGEST_STATE_KEY = "ingest_state"
_BOOTSTRAP_LOCK_MESSAGE = "ClickHouse Keeper bootstrap lock is unavailable; Keeper is required"

__all__ = [
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
    "release_lease",
    "stamp_store_metadata",
    "validate_metadata_table",
    "verify_clickhouse_connection",
    "verify_lease",
    "write_ingest_marker",
]


def _keeper_engine(path: str, *, primary_key: str = "key") -> Any:
    """Build the clickhouse-connect table-engine object without importing it at module load."""
    from clickhouse_connect.cc_sqlalchemy.ddl.tableengine import TableEngine

    class KeeperMap(TableEngine):
        arg_names: ClassVar[list[str]] = ["path"]
        quoted_args: ClassVar[set[str]] = {"path"}
        eng_params: ClassVar[list[str]] = ["primary_key"]

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
        if isinstance(generic, sqlalchemy.NullType):
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
        clone.engine = _keeper_engine(keeper_metadata_path(str(identity)))
    else:
        from clickhouse_connect.cc_sqlalchemy.ddl.tableengine import MergeTree

        clone.engine = MergeTree(order_by=_order_by(clone))
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
        raise RuntimeError(f"ClickHouse server {version} is too old; httk-data requires ClickHouse >= {required}")
    if not _setting_enabled(join_use_nulls):
        raise RuntimeError("ClickHouse join_use_nulls=1 could not be enforced; refusing the ClickHouse backend")
    return version


def install_connection_guards(engine: sqlalchemy.Engine) -> str:
    """Install pool checkout verification and return the first server version."""
    state = getattr(engine, "_httk_clickhouse_guard", None)
    if state is None:
        state = {"version": None, "lock": threading.Lock()}

        @event.listens_for(engine, "checkout")
        def _verify_checkout(dbapi_connection: Any, *_: Any) -> None:
            version, join_use_nulls = _raw_verify(dbapi_connection)
            if _version_tuple(version) < _MIN_SERVER_VERSION:
                required = ".".join(str(part) for part in _MIN_SERVER_VERSION)
                raise RuntimeError(
                    f"ClickHouse server {version} is too old; httk-data requires ClickHouse >= {required}"
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
        f"DELETE FROM _httk_store_metadata WHERE key = '{key}' AND value = '{escaped}'"
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
    connection.execute(sqlalchemy.delete(table).where(table.c.key == _LEASE_KEY, table.c.value == lease_value))


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
    connection.execute(sqlalchemy.delete(table).where(table.c.key == _INGEST_STATE_KEY, table.c.value == marker_value))


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
                connection.execute(
                    sqlalchemy.delete(bootstrap).where(bootstrap.c.key == key, bootstrap.c.value == token)
                )
            except BaseException as cleanup:
                cleanup_error = cleanup
        if held == token:
            recovery = (
                "the fresh token landed despite the failed INSERT; exact-value cleanup was attempted; "
                f"manual recovery is DELETE FROM _httk_bootstrap WHERE key = '{key}' AND value = '{token}'"
            )
        elif held is not None:
            recovery = (
                f"manual recovery: after verifying the writer is dead, "
                f"DELETE FROM _httk_bootstrap WHERE key = '{key}' AND value = '{held}'"
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
            connection.execute(sqlalchemy.delete(bootstrap).where(bootstrap.c.key == key, bootstrap.c.value == token))
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
            "SELECT engine, engine_full FROM system.tables WHERE database = currentDatabase() AND name = :name"
        ),
        {"name": "_httk_store_metadata"},
    ).all()
    if len(rows) != 1 or str(rows[0][0]) != "KeeperMap" or str(rows[0][1]) != expected_engine:
        raise RuntimeError(
            f"ClickHouse metadata table _httk_store_metadata failed physical validation: expected {expected_engine}"
        )
    columns = connection.execute(
        sqlalchemy.text(
            "SELECT name, type FROM system.columns "
            "WHERE database = currentDatabase() AND table = :table ORDER BY position"
        ),
        {"table": "_httk_store_metadata"},
    ).all()
    expected_columns = [("key", "String"), ("value", "String")]
    if [(str(name), str(type_name)) for name, type_name in columns] != expected_columns:
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
