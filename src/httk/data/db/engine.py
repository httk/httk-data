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
import os
from types import TracebackType
from typing import Self

import sqlalchemy

__all__ = [
    "Database",
]


class Database:
    """A relational database reachable through a wrapped SQLAlchemy engine.

    Construct one with :meth:`sqlite` or :meth:`duckdb` (or, for other
    SQLAlchemy-supported backends, by passing a preconfigured engine directly).
    The instance is a context manager; leaving the ``with`` block disposes the
    engine's connection pool.
    """

    def __init__(self, engine: sqlalchemy.Engine) -> None:
        """Wrap an already-configured SQLAlchemy :class:`~sqlalchemy.Engine`."""
        self._engine = engine

    @classmethod
    def sqlite(cls, path: str | os.PathLike[str] | None = None) -> Self:
        """An SQLite database stored in the file at ``path``, or in memory when ``path`` is None.

        The in-memory variant is configured (via a static connection pool with a
        shared, thread-unrestricted connection) so that every connection drawn
        from the engine sees the one and same database; file-backed databases
        use SQLAlchemy's default pooling.
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
        """A DuckDB database stored in the file at ``path``, or in memory when ``path`` is None.

        Raises:
            ImportError: If the ``duckdb_engine`` SQLAlchemy dialect is not
                installed; install the ``httk-data[duckdb]`` extra to get it.
        """
        try:
            importlib.import_module("duckdb_engine")
        except ImportError as error:
            raise ImportError(
                "the DuckDB backend needs the 'duckdb_engine' SQLAlchemy dialect; "
                "install the 'httk-data[duckdb]' extra to use Database.duckdb()"
            ) from error
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
        """The underlying SQLAlchemy engine (for use by the storage layer itself)."""
        return self._engine

    def dispose(self) -> None:
        """Close the engine's connection pool; the database can no longer be used after this."""
        self._engine.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.dispose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._engine.url!r})"
