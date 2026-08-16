"""Live PostgreSQL acceptance: open, incremental save/fetch, dedup, and reopen.

Skipped unless ``HTTK_TEST_POSTGRES_URI`` names a reachable admin database. Each
test runs against a freshly created, uniquely named database so parallel workers
never collide, and drops it on teardown.
"""

import os
import uuid

import pytest
import sqlalchemy
from sqlalchemy.engine import make_url

from httk.store.db import Database, SqlStore

from test_db_store import Sample, make_sample


def _admin_uri() -> str:
    uri = os.environ.get("HTTK_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip("HTTK_TEST_POSTGRES_URI is not set; a reachable PostgreSQL admin URI is required")
    return uri


@pytest.fixture
def postgres_uri():
    """Create and drop a uniquely named database, yielding its psycopg URI."""
    admin_url = make_url(_admin_uri())
    admin_engine = sqlalchemy.create_engine(admin_url, isolation_level="AUTOCOMMIT")
    name = f"httk_smoke_{uuid.uuid4().hex}"
    try:
        with admin_engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'CREATE DATABASE "{name}"'))
        yield admin_url.set(database=name)
    finally:
        with admin_engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin_engine.dispose()


def test_postgres_save_fetch_dedup_and_reopen(postgres_uri):
    sample = make_sample()
    database = Database.postgres(postgres_uri)
    try:
        store = SqlStore(database, entry_records={})

        sid = store.save(sample)
        assert isinstance(sid, int)

        fetched = store.fetch(Sample, sid)
        assert fetched == sample

        # Content-id dedup: the same content reuses the existing row and sid.
        assert store.save(make_sample()) == sid

        # Reopen with the identical declaration: metadata is trusted, and the
        # previously saved object is still reachable.
        reopened = SqlStore(database, entry_records={})
        assert reopened.fetch(Sample, sid) == sample
    finally:
        database.dispose()
