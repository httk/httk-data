"""PostgreSQL database construction for :class:`~httk.store.backend.sql.engine.Backend`."""

import importlib
from typing import TYPE_CHECKING

import sqlalchemy

if TYPE_CHECKING:
    from httk.store.backend.sql.engine import Backend


def database(
    cls: "type[Backend]",
    url: str | sqlalchemy.URL,
    *,
    database: str | None = None,
) -> "Backend":
    """Build a PostgreSQL-backed backend from a ``postgresql://`` URL.

    PostgreSQL is fully transactional and rides the existing ``"transactional"``
    write profile with no special-casing. Only the psycopg 3 driver is
    supported: a bare ``postgresql://`` URL is normalized to
    ``postgresql+psycopg://`` (SQLAlchemy 2.0 would otherwise select psycopg2),
    and any other explicit driver is rejected.

    :param cls: The backend class to instantiate.
    :param url: PostgreSQL SQLAlchemy URL or URL string.
    :param database: The database name overriding the URL path, if supplied.
    :return: The configured PostgreSQL backend wrapper.
    :raises ImportError: If ``psycopg`` (psycopg 3) is not installed; install the
        ``httk-store[postgresql]`` extra to use ``Backend.postgresql()``.
    :raises ValueError: If the URL names a driver other than
        ``postgresql+psycopg`` (psycopg 3).
    """
    try:
        importlib.import_module("psycopg")
    except ImportError as error:
        raise ImportError(
            "the PostgreSQL backend needs psycopg (psycopg 3); install the 'httk-store[postgresql]' extra "
            "to use Backend.postgresql()"
        ) from error
    from sqlalchemy.engine import make_url

    postgres_url = make_url(url) if isinstance(url, str) else url
    if postgres_url.drivername == "postgresql":
        postgres_url = postgres_url.set(drivername="postgresql+psycopg")
    elif postgres_url.drivername != "postgresql+psycopg":
        raise ValueError(
            f"Backend.postgresql() supports only the 'postgresql+psycopg' driver (psycopg 3), "
            f"not {postgres_url.drivername!r}"
        )
    if database is not None:
        postgres_url = postgres_url.set(database=database)
    engine = sqlalchemy.create_engine(postgres_url)
    return cls(engine)
