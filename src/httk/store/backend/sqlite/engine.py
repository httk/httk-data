"""SQLite database construction for :class:`~httk.store.backend.sql.engine.Backend`."""

import os
from typing import TYPE_CHECKING, Any

import sqlalchemy

if TYPE_CHECKING:
    from httk.store.backend.sql.engine import Backend


def database(
    cls: "type[Backend]",
    path: str | os.PathLike[str] | None = None,
    *,
    degraded: bool = False,
) -> "Backend":
    """Build an SQLite-backed backend stored in ``path``, or in memory when ``path`` is None.

    The in-memory variant is configured (via a static connection pool with a
    shared, thread-unrestricted connection) so that every connection drawn from
    the engine sees the one and same database; file-backed databases use
    SQLAlchemy's default pooling.

    :param cls: The backend class to instantiate.
    :param path: The database file path, or ``None`` for an in-memory database.
    :param degraded: Open with autocommit isolation for degraded-mode access
        (recovery and inspection) instead of the default transactional isolation.
    :return: The configured backend wrapper.
    """
    if path is None:
        options: dict[str, Any] = {
            "poolclass": sqlalchemy.StaticPool,
            "connect_args": {"check_same_thread": False},
        }
        if degraded:
            options["isolation_level"] = "AUTOCOMMIT"
        engine = sqlalchemy.create_engine("sqlite://", **options)
    else:
        options = {"isolation_level": "AUTOCOMMIT"} if degraded else {}
        engine = sqlalchemy.create_engine(f"sqlite:///{os.fspath(path)}", **options)
    return cls(engine, degraded=degraded)
