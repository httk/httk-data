"""Live PostgreSQL coverage for inline exact fraction-equality compilation.

Skipped unless ``HTTK_TEST_POSTGRES_URI`` names a reachable admin database.
PostgreSQL cannot register the per-connection Python UDF that SQLite/DuckDB use,
so ``httk_fraction_scaled_equal`` is rewritten to inline ``numeric`` SQL by the
``@compiles`` hook in :mod:`httk.store.backend.postgresql.compiler`.  These tests confirm the
inline SQL matches the Python reference and that the real stored-property filter
path works against a PostgreSQL store.  Each test runs against a freshly created,
uniquely named database and drops it on teardown.
"""

import os
import uuid

import pytest
import sqlalchemy
from sqlalchemy.engine import make_url

from httk.store.backend.sql import Backend, SqlStore, stored_property_sql_plan
from httk.store.backend.sql.engine import _fraction_scaled_equal

from test_db_stored_properties import (
    FIRST,
    SECOND,
    CalculationEntry,
    GenericCalculationFirst,
    GenericCalculationSecond,
    _records,
)


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
    name = f"httk_fraction_{uuid.uuid4().hex}"
    try:
        with admin_engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'CREATE DATABASE "{name}"'))
        yield admin_url.set(database=name)
    finally:
        with admin_engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin_engine.dispose()


# (left_value, left_factor, right_value, right_factor): canonical "p/q" text,
# integer-form factors (implicit denominator 1), None, and a large in-bound value.
_ARGUMENT_TABLE = (
    ("1/3", "2", "2/3", "1"),  # equal
    ("1/3", "1", "2/3", "1"),  # unequal
    ("-1/3", "2", "-2/3", "1"),  # negative numerators, equal
    ("1/1", "5", "5/1", "1"),  # 1/1 canonical, equal
    ("1/2", 3, "3/4", 2),  # mixed text with integer factors, equal
    ("1/2", 2, "3/4", 2),  # unequal
    (None, "1", "2/3", "1"),  # None argument propagates to NULL
    ("100000000000000000000/3", "1", "100000000000000000000/3", "1"),  # large in-bound, equal
)


@pytest.mark.parametrize("arguments", _ARGUMENT_TABLE)
def test_inline_fraction_equality_matches_python_reference(postgres_uri, arguments):
    database = Backend.postgresql(postgres_uri)
    try:
        with database.engine.connect() as connection:
            statement = sqlalchemy.select(sqlalchemy.func.httk_fraction_scaled_equal(*arguments))
            # Opening Backend.postgresql must self-register the @compiles hook, so
            # the call compiles to inline SQL rather than a bare function call.
            compiled = str(statement.compile(connection, compile_kwargs={"literal_binds": True}))
            assert "split_part" in compiled
            assert "httk_fraction_scaled_equal(" not in compiled
            result = connection.execute(statement).scalar_one()
        assert result == _fraction_scaled_equal(*arguments)
    finally:
        database.dispose()


def test_stored_property_exact_fraction_filter_selects_matching_rows(postgres_uri):
    # Mirrors test_db_stored_properties' 'immutable_id = "one-third"' case, whose
    # query emits httk_fraction_scaled_equal against the energy_exact column.
    database = Backend.postgresql(postgres_uri)
    try:
        store = SqlStore(
            database,
            entry_records={CalculationEntry: (GenericCalculationFirst, GenericCalculationSecond)},
        )
        store.save(FIRST)
        store.save(SECOND)
        plan = stored_property_sql_plan(store, CalculationEntry)
        matched = {record.label for record in _records(plan.filter_searchers('immutable_id = "one-third"'))}
        assert matched == {"first"}
    finally:
        database.dispose()
