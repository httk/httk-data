"""Shared PostgreSQL scaffolding for the parametrized store test suites.

PostgreSQL, like ClickHouse, needs a running server and per-test *isolated*
databases (it has no in-memory mode).  Every test that opts in creates a freshly
named ``httk_test_<uuid>`` database from the admin URI, runs against it, and
drops it on teardown so nothing leaks.  The ``POSTGRES_PARAM`` xdist group pins
all such tests to a single worker, so parallel workers never race on
create/drop.
"""

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import sqlalchemy
from sqlalchemy.engine import make_url

from httk.store.backend.sql import Backend


def postgres_admin_uri() -> str:
    """Return the configured PostgreSQL admin URI or skip with the setup pointer."""
    uri = os.environ.get("HTTK_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip("HTTK_TEST_POSTGRES_URI is not set; a reachable PostgreSQL admin URI is required")
    return uri


POSTGRES_PARAM = pytest.param("postgresql", marks=pytest.mark.xdist_group("postgres"))


class IsolatedPostgresDatabase:
    """A uniquely named PostgreSQL database, created on construction.

    ``uri`` is the per-database URL callers turn into ``Backend.postgresql(uri)``.
    Dispose any ``Backend`` pools you opened, then call :meth:`drop`; the drop
    uses ``WITH (FORCE)`` so a leftover backend connection cannot block it.
    """

    def __init__(self) -> None:
        self._admin_url = make_url(postgres_admin_uri())
        self._admin_engine = sqlalchemy.create_engine(self._admin_url, isolation_level="AUTOCOMMIT")
        self.name = f"httk_test_{uuid.uuid4().hex}"
        with self._admin_engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'CREATE DATABASE "{self.name}"'))
        self.uri = self._admin_url.set(database=self.name)

    def drop(self) -> None:
        """Drop the database (forcing off any remaining backends) and dispose the admin pool."""
        with self._admin_engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'DROP DATABASE IF EXISTS "{self.name}" WITH (FORCE)'))
        self._admin_engine.dispose()


@contextmanager
def postgres_database() -> Iterator[Backend]:
    """Yield one ``Backend.postgresql`` over a fresh isolated database, dropped on exit."""
    isolated = IsolatedPostgresDatabase()
    database = Backend.postgresql(isolated.uri)
    try:
        yield database
    finally:
        database.dispose()
        isolated.drop()
