"""Small, isolated ClickHouse helpers for the parametrized read suites."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from conftest import clickhouse_test_uri
from test_clickhouse_bulk import _clickhouse_bulk_database

from httk.store.backend.sql import SqlStore

CLICKHOUSE_PARAM = pytest.param("clickhousedb", marks=pytest.mark.xdist_group("clickhouse_read"))


@contextmanager
def clickhouse_database() -> Iterator[Any]:
    """Yield one isolated, deployment-validated ClickHouse database."""
    with _clickhouse_bulk_database(clickhouse_test_uri()) as database:
        yield database


@contextmanager
def bulk_store(records: Iterable[Any], *, entry_records: dict[type, Any] | None = None) -> Iterator[SqlStore]:
    """Build one isolated ClickHouse store from a suite's save plan."""

    with clickhouse_database() as database:
        store = SqlStore(database, entry_records=entry_records or {})
        with store.bulk_ingest(finalize="deferred") as bulk:
            for record in records:
                bulk.save(record)
        yield store
