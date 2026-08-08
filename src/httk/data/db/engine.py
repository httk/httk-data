"""Database engine lifecycle: :class:`Database` wraps a SQLAlchemy engine behind an httk-facing API.

A :class:`Database` names *where* data lives — an SQLite file, an in-memory
SQLite database, a DuckDB file — and owns the connection pool that reaches it.
It deliberately exposes no SQL surface of its own: the store layer
(:class:`~httk.data.db.store.SqlStore`) asks it for connections internally, and
user code only constructs one (usually via :meth:`Database.sqlite` or
:meth:`Database.duckdb`) and passes it on. There is no global engine registry
and no interpreter-exit hook; dispose of a database explicitly with
:meth:`Database.dispose` or use it as a context manager.
"""

import importlib
import importlib.util
import logging
import os
import sys
from fractions import Fraction
from types import TracebackType
from typing import Any, Self

import sqlalchemy
from sqlalchemy import event

__all__ = [
    "Database",
]

_LOGGER = logging.getLogger(__name__)

_EXACT_FRACTION_FUNCTIONS_ATTRIBUTE = "_httk_exact_fraction_functions_installed"
_EXACT_FRACTION_FUNCTIONS_KEY = "httk_exact_fraction_functions_installed"


class Database:
    """A relational database reachable through a wrapped SQLAlchemy engine.

    Construct one with :meth:`sqlite` or :meth:`duckdb` (or, for other
    SQLAlchemy-supported backends, by passing a preconfigured engine directly).
    The instance is a context manager; leaving the ``with`` block disposes the
    engine's connection pool.

    :param engine: The configured SQLAlchemy engine to wrap.
    """

    def __init__(self, engine: sqlalchemy.Engine) -> None:
        self._engine = engine
        _install_exact_fraction_functions(engine)

    @classmethod
    def sqlite(cls, path: str | os.PathLike[str] | None = None) -> Self:
        """Create an SQLite database stored in ``path``, or in memory when ``path`` is None.

        The in-memory variant is configured (via a static connection pool with a
        shared, thread-unrestricted connection) so that every connection drawn
        from the engine sees the one and same database; file-backed databases
        use SQLAlchemy's default pooling.

        :param path: The database file path, or ``None`` for an in-memory database.
        :return: The configured database wrapper.
        """
        if path is None:
            engine = sqlalchemy.create_engine(
                "sqlite://",
                poolclass=sqlalchemy.StaticPool,
                connect_args={"check_same_thread": False},
            )
        else:
            engine = sqlalchemy.create_engine(f"sqlite:///{os.fspath(path)}")
        return cls(engine)

    @classmethod
    def duckdb(cls, path: str | os.PathLike[str] | None = None) -> Self:
        """Create a DuckDB database stored in ``path``, or in memory when ``path`` is None.

        :param path: The database file path, or ``None`` for an in-memory database.
        :return: The configured database wrapper.
        :raises ImportError: If the ``duckdb_engine`` SQLAlchemy dialect is not installed;
            install the ``httk-data[duckdb]`` extra to use it.
        """
        try:
            importlib.import_module("duckdb_engine")
        except ImportError as error:
            raise ImportError(
                "the DuckDB backend needs the 'duckdb_engine' SQLAlchemy dialect; "
                "install the 'httk-data[duckdb]' extra to use Database.duckdb()"
            ) from error
        _install_missing_pandas_sentinel()
        location = ":memory:" if path is None else os.fspath(path)
        engine = sqlalchemy.create_engine(f"duckdb:///{location}")
        # duckdb_engine derives from the psycopg2 dialect, which doubles
        # backslashes when rendering inline string literals (PostgreSQL's
        # non-standard-conforming-strings legacy). DuckDB always uses
        # standard-conforming string literals, so that doubling corrupts e.g.
        # the LIKE ... ESCAPE '\' clause the search DSL emits; turn it off.
        engine.dialect._backslash_escapes = False  # type: ignore[attr-defined]
        return cls(engine)

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


def _install_missing_pandas_sentinel() -> None:
    """Cache pandas's absence so DuckDB's per-row import probe stops re-searching ``sys.path``.

    DuckDB binds statement parameters through its native ``_duckdb`` extension
    (reached from ``duckdb_engine``'s ``CursorWrapper.execute``/``executemany``,
    which delegate to ``self.__c.execute(...)`` — duckdb_engine ``__init__.py``
    around line 150). For each bound value that path probes for pandas, roughly
    once per row. CPython caches a *successful* ``import`` in :data:`sys.modules`,
    so an installed pandas stays fast; but a *failed* import is not cached, so
    when pandas is absent every probe re-runs the full ``sys.path`` finder search
    — profiled at 40.7 s versus 2.86 s (about 14x) for a 50k-row ``executemany``.

    Installing the standard ``None`` failed-import sentinel makes each subsequent
    ``import pandas`` fail immediately from the :data:`sys.modules` check instead
    of searching the path; DuckDB tolerates that ``ImportError`` (its parameter
    binding is unaffected). This only acts when pandas is genuinely unimportable
    and untouched: an already-imported pandas is left as the real module, and a
    ``None`` (or any other) entry another party placed is left exactly as found.
    """
    if "pandas" in sys.modules:
        return
    if importlib.util.find_spec("pandas") is not None:
        return
    sys.modules["pandas"] = None  # type: ignore[assignment]  # the standard failed-import cache sentinel
    _LOGGER.debug(
        "installed a None sys.modules sentinel for absent pandas to short-circuit DuckDB's per-row import probe",
        extra={"context": "storage"},
    )


def _install_exact_fraction_functions(engine: sqlalchemy.Engine) -> None:
    """Install exact-fraction scalar functions on every SQLite/DuckDB connection.

    Fraction values are persisted canonically as text because SQL integer
    products can overflow long before a valid Python :class:`Fraction` does.
    These functions preserve exact comparisons without touching presentation
    float columns.
    """

    dialect = engine.dialect.name
    if dialect not in {"sqlite", "duckdb"}:
        return
    if getattr(engine, _EXACT_FRACTION_FUNCTIONS_ATTRIBUTE, False):
        return

    def install(dbapi_connection: Any, connection_record: Any) -> None:
        if connection_record.info.get(_EXACT_FRACTION_FUNCTIONS_KEY):
            return
        if dialect == "sqlite":
            dbapi_connection.create_function("httk_fraction_scaled_equal", 4, _fraction_scaled_equal)
        else:
            duckdb = importlib.import_module("duckdb")
            try:
                dbapi_connection.create_function(
                    "httk_fraction_scaled_equal",
                    _fraction_scaled_equal,
                    return_type=duckdb.sqltypes.BOOLEAN,
                )
            except Exception as error:
                # DuckDB registers functions in the database catalog, rather
                # than per DBAPI connection.  A simultaneous peer may already
                # have installed it, in which case this connection can use the
                # same catalog function immediately.  Never remove/recreate:
                # doing so races an active peer's query.
                if not _duckdb_duplicate_function_error(error):
                    raise
        connection_record.info[_EXACT_FRACTION_FUNCTIONS_KEY] = True

    def connect(dbapi_connection: Any, connection_record: Any) -> None:
        install(dbapi_connection, connection_record)

    def checkout(dbapi_connection: Any, connection_record: Any, _connection_proxy: Any) -> None:
        # ``connect`` only observes newly-created DBAPI connections.  This
        # covers an engine that was already in use before ``Database(engine)``
        # wrapped it and subsequently checks out an existing pooled handle.
        install(dbapi_connection, connection_record)

    event.listen(engine, "connect", connect)
    event.listen(engine, "checkout", checkout)
    setattr(engine, _EXACT_FRACTION_FUNCTIONS_ATTRIBUTE, True)


def _duckdb_duplicate_function_error(error: Exception) -> bool:
    """Whether DuckDB reports a catalog function already installed by a peer."""
    text = str(error).casefold()
    return "already exists" in text or "already created" in text


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
