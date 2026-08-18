"""ClickHouse database construction for :class:`~httk.store.backend.sql.engine.Backend`."""

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
    """Build a ClickHouse-backed backend from a ``clickhousedb://`` URL.

    The URL uses the SQLAlchemy ``clickhouse-connect`` dialect, for example
    ``clickhousedb://default:@host:8123/my_database``.  ``database`` replaces the
    URL path when supplied.  The constructor always merges ``join_use_nulls=1``
    into the URL query and selects the ``bulk-fenced`` storage profile before any
    :class:`~httk.store.backend.sql.store.SqlStore` initialization occurs.

    :param cls: The backend class to instantiate.
    :param url: ClickHouse SQLAlchemy URL or URL string.
    :param database: The database name overriding the URL path, if supplied.
    :return: Connected ClickHouse backend wrapper using the bulk-fenced profile.
    :raises ImportError: If ``clickhouse-connect`` is not installed; install the
        ``httk-store[clickhouse]`` extra.
    :raises RuntimeError: If Keeper is unavailable, the server is too old, or
        ``join_use_nulls`` cannot be enforced.
    """
    try:
        importlib.import_module("clickhouse_connect")
        # Importing this module registers the clickhousedb SQLAlchemy URL.
        importlib.import_module("clickhouse_connect.cc_sqlalchemy")
    except ImportError as error:
        raise ImportError(
            "the ClickHouse backend needs clickhouse-connect; install the 'httk-store[clickhouse]' extra "
            "to use Backend.clickhouse()"
        ) from error
    from sqlalchemy.engine import make_url

    clickhouse_url = make_url(url) if isinstance(url, str) else url
    if clickhouse_url.drivername.split("+")[0] != "clickhousedb":
        raise ValueError("Backend.clickhouse() requires a clickhousedb:// SQLAlchemy URL")
    if database is not None:
        clickhouse_url = clickhouse_url.set(database=database)
    clickhouse_url = clickhouse_url.update_query_dict({"join_use_nulls": "1"})
    engine = sqlalchemy.create_engine(clickhouse_url)
    try:
        result = cls(engine, write_profile="bulk-fenced")
        from httk.store.backend.clickhouse.support import ensure_bootstrap_table

        ensure_bootstrap_table(result.engine)
        return result
    except BaseException:
        engine.dispose()
        raise
