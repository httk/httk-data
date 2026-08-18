"""Backend engine lifecycle: :class:`Backend` wraps a SQLAlchemy engine behind an httk-facing API.

A :class:`Backend` names *where* data lives — an SQLite file, an in-memory
SQLite database, a DuckDB file, a PostgreSQL server, or a Keeper-backed
ClickHouse cluster — and owns the connection pool that reaches it.
It deliberately exposes no SQL surface of its own: the store layer
(:class:`~httk.store.backend.sql.store.SqlStore`) asks it for connections internally, and
user code only constructs one (usually via :meth:`Backend.sqlite` or
:meth:`Backend.duckdb`) and passes it on. There is no global engine registry
and no interpreter-exit hook; dispose of a database explicitly with
:meth:`Backend.dispose` or use it as a context manager.
"""

import importlib
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from fractions import Fraction
from types import TracebackType
from typing import Any, Literal, Self

import sqlalchemy
from sqlalchemy import event

__all__ = [
    "Backend",
]

_LOGGER = logging.getLogger(__name__)

_EXACT_FRACTION_FUNCTIONS_ATTRIBUTE = "_httk_exact_fraction_functions_installed"
_EXACT_FRACTION_FUNCTIONS_KEY = "httk_exact_fraction_functions_installed"
_DISPOSE_WAIT_WARNING_SECONDS = 60.0


class Backend:
    """A relational database reachable through a wrapped SQLAlchemy engine.

    Construct one with :meth:`sqlite`, :meth:`duckdb`, :meth:`postgresql`, or
    :meth:`clickhouse` (or, for other SQLAlchemy-supported backends, by passing a
    preconfigured engine directly). The instance is a context manager; leaving
    the ``with`` block disposes the engine's connection pool.

    :param engine: The configured SQLAlchemy engine to wrap.
    :param degraded: Open with autocommit isolation for degraded-mode access
        (recovery and inspection) instead of the default transactional isolation.
    :param write_profile: Explicit permanentization profile, or ``None`` to
        derive it from the backend and ``degraded`` flag.
    """

    def __init__(
        self,
        engine: sqlalchemy.Engine,
        *,
        degraded: bool = False,
        write_profile: Literal["transactional", "degraded", "bulk-fenced"] | None = None,
    ) -> None:
        self._engine = engine
        self._degraded = degraded
        if write_profile is None:
            write_profile = "degraded" if degraded else "transactional"
            if engine.dialect.name == "clickhousedb":
                write_profile = "bulk-fenced"
        from httk.store.backend.sql.layout import WRITE_PROFILE_VOCABULARY, backend_facts_for_dialect

        if write_profile not in WRITE_PROFILE_VOCABULARY:
            raise ValueError(f"unknown storage write profile {write_profile!r}")
        try:
            facts = backend_facts_for_dialect(engine.dialect.name)
        except ValueError:
            facts = None
        if facts is not None and write_profile not in facts.write_profiles:
            raise ValueError(f"write profile {write_profile!r} is not supported by the {engine.dialect.name!r} backend")
        if degraded and write_profile != "degraded":
            raise ValueError("degraded=True requires the degraded write profile")
        # Explicit annotation: the constructor above guarantees membership in the
        # literal set, but an inferred attribute type would widen back to ``str``.
        self._write_profile: Literal["transactional", "degraded", "bulk-fenced"] = write_profile
        self._server_version: str | None = None
        self._dispose_callbacks: list[Callable[[], None]] = []
        self._dispose_lock = threading.RLock()
        self._lifecycle_condition = threading.Condition(self._dispose_lock)
        self._lifecycle_holders: dict[str, int] = {}
        self._lifecycle_owner: int | None = None
        self._disposed = False
        self._lifecycle_generation = 0
        if engine.dialect.name == "clickhousedb":
            from httk.store.backend.clickhouse.support import install_connection_guards

            self._server_version = install_connection_guards(engine)
        if engine.dialect.name == "postgresql":
            # Register the @compiles hook that rewrites httk_fraction_scaled_equal
            # to inline SQL. Done here (not only in Backend.postgresql) so a
            # preconfigured PostgreSQL engine passed straight to Backend(...) is
            # covered too — PostgreSQL has no per-connection UDF to fall back on.
            importlib.import_module("httk.store.backend.postgresql.compiler")
        _install_engine_functions(engine)

    @classmethod
    def sqlite(cls, path: str | os.PathLike[str] | None = None, *, degraded: bool = False) -> "Backend":
        """Create an SQLite database stored in ``path``, or in memory when ``path`` is None.

        The in-memory variant is configured (via a static connection pool with a
        shared, thread-unrestricted connection) so that every connection drawn
        from the engine sees the one and same database; file-backed databases
        use SQLAlchemy's default pooling.

        :param path: The database file path, or ``None`` for an in-memory database.
        :param degraded: Open with autocommit isolation for degraded-mode access
            (recovery and inspection) instead of the default transactional isolation.
        :return: The configured database wrapper.
        """
        from httk.store.backend.sqlite.engine import database

        return database(cls, path, degraded=degraded)

    @classmethod
    def duckdb(cls, path: str | os.PathLike[str] | None = None, *, memory_limit: str | None = None) -> "Backend":
        """Create a DuckDB database stored in ``path``, or in memory when ``path`` is None.

        :param path: The database file path, or ``None`` for an in-memory database.
        :param memory_limit: An optional DuckDB ``memory_limit`` setting such as
            ``"1GB"``. DuckDB's own default allows every instance up to about 80%
            of system RAM, which multiplies dangerously across parallel test or
            ingest processes; when this parameter is ``None`` the
            ``HTTK_DUCKDB_MEMORY_LIMIT`` environment variable (if set) supplies
            the cap instead, so process trees can be memory-guarded wholesale.
        :return: The configured database wrapper.
        :raises ImportError: If the ``duckdb_engine`` SQLAlchemy dialect is not installed;
            install the ``httk-store[duckdb]`` extra to use it.
        """
        from httk.store.backend.duckdb.engine import database

        return database(cls, path, memory_limit=memory_limit)

    @classmethod
    def clickhouse(cls, url: str | sqlalchemy.URL, *, database: str | None = None) -> "Backend":
        """Create a ClickHouse database from a ``clickhousedb://`` URL.

        The URL uses the SQLAlchemy ``clickhouse-connect`` dialect, for example
        ``clickhousedb://default:@host:8123/my_database``.  ``database``
        replaces the URL path when supplied.  The constructor always merges
        ``join_use_nulls=1`` into the URL query and selects the ``bulk-fenced``
        storage profile before any :class:`~httk.store.backend.sql.store.SqlStore`
        initialization occurs.

        :param url: ClickHouse SQLAlchemy URL or URL string.
        :param database: The database name overriding the URL path, if supplied.
        :return: Connected ClickHouse database wrapper using the bulk-fenced profile.
        :raises ImportError: If ``clickhouse-connect`` is not installed; install
            the ``httk-store[clickhouse]`` extra.
        :raises RuntimeError: If Keeper is unavailable, the server is too old,
            or ``join_use_nulls`` cannot be enforced.
        """
        from httk.store.backend.clickhouse.engine import database as clickhouse_database

        return clickhouse_database(cls, url, database=database)

    @classmethod
    def postgresql(cls, url: str | sqlalchemy.URL, *, database: str | None = None) -> "Backend":
        """Create a PostgreSQL database from a ``postgresql://`` URL.

        PostgreSQL is fully transactional and rides the existing
        ``"transactional"`` write profile with no special-casing. Only the
        psycopg 3 driver is supported: a bare ``postgresql://`` URL is
        normalized to ``postgresql+psycopg://`` (SQLAlchemy 2.0 would otherwise
        select psycopg2), and any other explicit driver is rejected.

        :param url: PostgreSQL SQLAlchemy URL or URL string.
        :param database: The database name overriding the URL path, if supplied.
        :return: The configured PostgreSQL database wrapper.
        :raises ImportError: If ``psycopg`` (psycopg 3) is not installed; install
            the ``httk-store[postgresql]`` extra to use ``Backend.postgresql()``.
        :raises ValueError: If the URL names a driver other than
            ``postgresql+psycopg`` (psycopg 3).
        """
        from httk.store.backend.postgresql.engine import database as postgresql_database

        return postgresql_database(cls, url, database=database)

    @property
    def degraded(self) -> bool:
        """Whether this wrapper deliberately uses the SQLite autocommit vehicle."""
        return self._degraded

    @property
    def write_profile(self) -> Literal["transactional", "degraded", "bulk-fenced"]:
        """Return the profile selected before store initialization."""
        return self._write_profile

    @property
    def server_version(self) -> str | None:
        """Return the server version captured at the first ClickHouse connection."""
        return self._server_version

    @property
    def lifecycle_generation(self) -> int:
        """Return the active lifecycle generation for guarded storage callbacks."""
        with self._dispose_lock:
            if self._disposed:
                raise RuntimeError("cannot obtain a lifecycle generation from a disposed Backend")
            return self._lifecycle_generation

    def add_dispose_callback(self, callback: Callable[[], None], *, generation: int | None = None) -> int:
        """Register a best-effort callback for the active lifecycle generation.

        A disposed wrapper deliberately rejects late registration: accepting a
        callback after :meth:`dispose` snapshots its callback list can strand a
        store-owned lease on a newly recreated pool.
        """
        with self._dispose_lock:
            if self._disposed:
                raise RuntimeError("cannot register a disposal callback on a disposed Backend")
            if generation is not None and generation != self._lifecycle_generation:
                raise RuntimeError("Backend lifecycle generation changed before callback registration")
            self._dispose_callbacks.append(callback)
            return self._lifecycle_generation

    @contextmanager
    def lifecycle_guard(self, generation: int, *, holder: str | None = None) -> Any:
        """Prevent disposal while one named store mutation uses ``generation``."""
        label = holder or threading.current_thread().name
        owner = threading.get_ident()
        with self._lifecycle_condition:
            while self._lifecycle_holders and self._lifecycle_owner != owner:
                if self._disposed:
                    raise RuntimeError("Backend has been disposed; create a new Backend before mutating this store")
                self._lifecycle_condition.wait()
            if self._disposed or generation != self._lifecycle_generation:
                raise RuntimeError("Backend has been disposed; create a new Backend before mutating this store")
            self._lifecycle_owner = owner
            self._lifecycle_holders[label] = self._lifecycle_holders.get(label, 0) + 1
        try:
            yield
        finally:
            with self._lifecycle_condition:
                count = self._lifecycle_holders.get(label, 0)
                if count <= 1:
                    self._lifecycle_holders.pop(label, None)
                else:
                    self._lifecycle_holders[label] = count - 1
                if not self._lifecycle_holders:
                    self._lifecycle_owner = None
                self._lifecycle_condition.notify_all()

    @property
    def engine(self) -> sqlalchemy.Engine:
        """Return the underlying SQLAlchemy engine for the storage layer.

        :return: The wrapped SQLAlchemy engine.
        """
        return self._engine

    def dispose(self) -> None:
        """Dispose the current connection pool; later use creates a new pool.

        :return: None.
        """
        with self._lifecycle_condition:
            if self._disposed:
                return
            if self._lifecycle_holders and self._lifecycle_owner == threading.get_ident():
                raise RuntimeError("cannot dispose from within an active bulk context")
            self._disposed = True
            self._lifecycle_generation += 1
            callbacks, self._dispose_callbacks = self._dispose_callbacks, []
            next_warning = time.monotonic() + _DISPOSE_WAIT_WARNING_SECONDS
            while self._lifecycle_holders:
                remaining = max(0.0, next_warning - time.monotonic())
                self._lifecycle_condition.wait(timeout=remaining)
                if self._lifecycle_holders and time.monotonic() >= next_warning:
                    holders = ", ".join(f"{name} ({count})" for name, count in sorted(self._lifecycle_holders.items()))
                    _LOGGER.warning("database disposal is waiting for in-flight lifecycle guard holder(s): %s", holders)
                    next_warning = time.monotonic() + _DISPOSE_WAIT_WARNING_SECONDS
        for callback in callbacks:
            try:
                callback()
            except Exception:
                _LOGGER.exception("database disposal callback failed", extra={"context": "storage"})
        self._engine.dispose()

    def __enter__(self) -> Self:
        """Enter a context that owns this database's connection pool.

        :return: This database wrapper.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Dispose the database when leaving its context.

        :param exc_type: The exception class raised in the context, if any.
        :param exc_value: The exception instance raised in the context, if any.
        :param traceback: The traceback for the context exception, if any.
        :return: None.
        """
        self.dispose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._engine.url!r})"


_DIALECT_HOOKS: dict[str, str] = {
    "sqlite": "httk.store.backend.sqlite.hooks",
    "duckdb": "httk.store.backend.duckdb.hooks",
    "postgresql": "httk.store.backend.postgresql.hooks",
}


def _dialect_hooks(dialect_name: str) -> Any:
    """Import the per-dialect hooks module, or return ``None`` when there is none.

    :param dialect_name: The SQLAlchemy dialect name (``engine.dialect.name``).
    :return: The imported ``<dialect>.hooks`` module, or ``None`` if the dialect
        is not registered in :data:`_DIALECT_HOOKS`.
    """
    module_path = _DIALECT_HOOKS.get(dialect_name)
    if module_path is None:
        return None
    return importlib.import_module(module_path)


def _install_engine_functions(engine: sqlalchemy.Engine) -> None:
    """Run the dialect's ``install_engine_functions`` hook, if it defines one.

    A dialect with no hooks module, or a hooks module that defines no
    ``install_engine_functions``, is a documented no-op: only SQLite and DuckDB
    register the exact-fraction UDF, so PostgreSQL and ClickHouse install
    nothing.  Called from :meth:`Backend.__init__` so a preconfigured engine
    passed straight to :class:`Backend` is covered too.

    :param engine: The SQLAlchemy engine to install dialect functions on.
    :return: None.
    """
    hooks = _dialect_hooks(engine.dialect.name)
    install = getattr(hooks, "install_engine_functions", None)
    if install is not None:
        install(engine)


def connection_uses_autocommit(connection: sqlalchemy.Connection) -> bool:
    """Return the DBAPI connection's actual autocommit state.

    SQLAlchemy's execution options describe an engine's intent, but a caller
    can wrap any preconfigured engine in :class:`Backend`, so permanentization
    inspects the live DBAPI connection.  The check is dialect specific and
    delegated to the dialect's ``connection_uses_autocommit`` hook; a dialect
    with no hooks module (or no such function) is never in autocommit mode.

    :param connection: The live SQLAlchemy connection to inspect.
    :return: Whether the underlying DBAPI connection is in autocommit mode.
    """
    hooks = _dialect_hooks(connection.dialect.name)
    uses_autocommit = getattr(hooks, "connection_uses_autocommit", None)
    if uses_autocommit is None:
        return False
    return bool(uses_autocommit(connection))


def _install_scalar_function(engine: sqlalchemy.Engine, register: Callable[[Any], None]) -> None:
    """Register a per-connection scalar function on every connection of ``engine``.

    Neutral wiring shared by the SQLite and DuckDB hooks: ``register`` performs
    the dialect-specific ``create_function`` call on a freshly-connected or
    checked-out DBAPI connection.  The registration is idempotent per engine and
    per connection, so wrapping an already-open engine in :class:`Backend` still
    covers its pooled handles.

    :param engine: The SQLAlchemy engine to install the function on.
    :param register: Callback installing the function on one DBAPI connection.
    :return: None.
    """
    if getattr(engine, _EXACT_FRACTION_FUNCTIONS_ATTRIBUTE, False):
        return

    def install(dbapi_connection: Any, connection_record: Any) -> None:
        if connection_record.info.get(_EXACT_FRACTION_FUNCTIONS_KEY):
            return
        register(dbapi_connection)
        connection_record.info[_EXACT_FRACTION_FUNCTIONS_KEY] = True

    def connect(dbapi_connection: Any, connection_record: Any) -> None:
        install(dbapi_connection, connection_record)

    def checkout(dbapi_connection: Any, connection_record: Any, _connection_proxy: Any) -> None:
        # ``connect`` only observes newly-created DBAPI connections.  This
        # covers an engine that was already in use before ``Backend(engine)``
        # wrapped it and subsequently checks out an existing pooled handle.
        install(dbapi_connection, connection_record)

    event.listen(engine, "connect", connect)
    event.listen(engine, "checkout", checkout)
    setattr(engine, _EXACT_FRACTION_FUNCTIONS_ATTRIBUTE, True)


def _fraction(value: object) -> Fraction | None:
    if value is None:
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        raise ValueError("exact fraction SQL functions do not accept float values")
    return Fraction(str(value))


def _fraction_scaled_equal(
    left: object,
    left_factor: object,
    right: object,
    right_factor: object,
) -> bool | None:
    left_value = _fraction(left)
    left_multiplier = _fraction(left_factor)
    right_value = _fraction(right)
    right_multiplier = _fraction(right_factor)
    if None in (left_value, left_multiplier, right_value, right_multiplier):
        return None
    assert (
        left_value is not None
        and left_multiplier is not None
        and right_value is not None
        and right_multiplier is not None
    )
    return left_value * left_multiplier == right_value * right_multiplier
