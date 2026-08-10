"""Session-wide setup shared by this repository's tests.

``tests/test_examples.py`` runs each example script in a *subprocess* whose
working directory is a fresh temporary one, so an example that writes files
cannot pollute the checkout. That interacts badly with one thing: Python
resolves a **relative** ``PYTHONPATH`` entry against each process's own working
directory, not against the directory pytest was started in.

It matters whenever a repository is tested straight from a source checkout
rather than from an install — the sibling httk repositories are developed that
way, with invocations such as ``PYTHONPATH=src:../httk-data/src pytest``. Left
alone, those relative entries would resolve against the temporary directory in
the child process and point at nothing, so every example would fail to import
its own package: a false failure that says nothing about the example.

Absolutizing the inherited entries once, up front, makes them mean what the
caller meant — in this process and in every subprocess it spawns. It is a no-op
when ``PYTHONPATH`` is unset (the installed case, including CI) or when its
entries are already absolute.
"""

import os
import uuid

import pytest


def clickhouse_test_uri() -> str:
    """Return the configured ClickHouse URI or skip with the setup pointer."""
    uri = os.environ.get("HTTK_TEST_CLICKHOUSE_URI")
    if not uri:
        pytest.skip(
            "HTTK_TEST_CLICKHOUSE_URI is not set; run `make clickhouse-dev-server` and see tests/clickhouse/README.md"
        )
    return uri


# DuckDB's default memory_limit is ~80% of system RAM PER INSTANCE; across
# parallel pytest workers that multiplies into machine-wide OOM. Divide a fixed
# suite-wide budget across the xdist workers (each worker process sees the
# worker count in PYTEST_XDIST_WORKER_COUNT), clamped so a serial run still
# gets full single-test performance. An explicitly exported
# HTTK_DUCKDB_MEMORY_LIMIT always wins.
def _duckdb_test_memory_limit() -> str:
    budget_mb = int(os.environ.get("HTTK_DUCKDB_TEST_MEMORY_BUDGET_MB", "4096"))
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    return f"{max(512, min(4096, budget_mb // max(1, workers)))}MB"


os.environ.setdefault("HTTK_DUCKDB_MEMORY_LIMIT", _duckdb_test_memory_limit())

from httk.data.db import Database, SqlStore


@pytest.fixture(autouse=True)
def _fail_loudly_for_abandoned_bulk_contexts(monkeypatch):
    """Clean up and fail a test that abandoned a successfully entered bulk context.

    Database disposal must still wait for a legitimate long finalizer.  A test
    that bypasses ``__exit__`` is different: release its local ownership before
    fixture teardown so the failure is reported instead of turning into a
    deadlock in ``Database.dispose()``.
    """
    from httk.data.db.bulk import BulkIngest

    entered: list[BulkIngest] = []
    original_enter = BulkIngest.__enter__

    def tracked_enter(bulk: BulkIngest):
        result = original_enter(bulk)
        entered.append(bulk)
        return result

    monkeypatch.setattr(BulkIngest, "__enter__", tracked_enter)
    yield
    abandoned = [
        bulk
        for bulk in entered
        if bulk._store._bulk_active or bulk._bulk_lifecycle_guard is not None or bulk._bulk_lock_held
    ]
    cleanup_errors: list[BaseException] = []
    for bulk in abandoned:
        try:
            if bulk._entered and not bulk._closed:
                bulk.__exit__(RuntimeError, RuntimeError("test abandoned bulk context"), None)
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            bulk._release_connection(bulk._connection)
            bulk._connection = None
            bulk._close_workers()
            bulk._release_bulk_ownership()
            bulk._store._release_bulk_context()
    if abandoned:
        detail = ", ".join(type(bulk._store).__name__ for bulk in abandoned)
        if cleanup_errors:
            detail += f" (cleanup errors: {cleanup_errors!r})"
        raise AssertionError("test left claimed bulk_ingest context(s): " + detail)


_PYTHONPATH = os.environ.get("PYTHONPATH")
if _PYTHONPATH:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        os.path.abspath(entry) if entry else entry for entry in _PYTHONPATH.split(os.pathsep)
    )


@pytest.fixture(scope="session")
def mongo_test_client():
    """Reuse one live-client pool while each test still receives a new database.

    A ``MongoClient`` owns topology-monitor and pool allocations. Closing one
    after every parametrized test releases its sockets but not necessarily the
    worker allocator's RSS high-water mark; the neutral behavior suite alone
    used to create dozens. A session client is safe here because databases are
    unique and are dropped after each test.
    """
    uri = os.environ.get("HTTK_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HTTK_TEST_MONGODB_URI is not set")
    from pymongo import MongoClient

    client = MongoClient(
        uri,
        w="majority",
        journal=True,
        readConcernLevel="majority",
        serverSelectionTimeoutMS=1000,
    )
    try:
        client.admin.command("ping")
    except Exception as error:
        client.close()
        pytest.skip(f"MongoDB test server is unreachable: {error}")
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(params=["sqlite", "duckdb", "mongo"])
def store_backend(request):
    """Select each backend supported by the neutral store behavior suite."""
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
    if request.param == "mongo":
        yield request.param, request.getfixturevalue("mongo_test_client")
        return
    yield request.param, None


class _StoreFactory:
    """Callable factory returning real stores plus a same-database reopen path."""

    def __init__(self, backend, mongo_client, databases):
        self._backend = backend
        self._mongo_client = mongo_client
        self._databases = databases
        self._stores = {}

    def __call__(self, *, entry_records=None):
        if self._backend == "sqlite":
            database = Database.sqlite()
            declaration = entry_records if entry_records is not None else {}
            store = SqlStore(database, entry_records=declaration)
        elif self._backend == "duckdb":
            database = Database.duckdb()
            declaration = entry_records if entry_records is not None else {}
            store = SqlStore(database, entry_records=declaration)
        else:
            from httk.data.mongo import MongoDatabase, MongoStore

            name = f"httk_behavior_{uuid.uuid4().hex}"
            assert self._mongo_client is not None
            database = MongoDatabase(self._mongo_client, name, transactions="never")
            declaration = entry_records if entry_records is not None else {}
            store = MongoStore(database, entry_records=declaration)
        self._databases.append(database)
        self._stores[id(store)] = (store, database, declaration)
        return store

    def reopen(self, store):
        """Return a fresh real store over the database used by ``store``."""
        try:
            original, database, declaration = self._stores[id(store)]
        except KeyError as error:
            raise ValueError("store was not created by this store_factory") from error
        if original is not store:
            raise ValueError("store was not created by this store_factory")
        if self._backend == "mongo":
            from httk.data.mongo import MongoDatabase, MongoStore

            assert self._mongo_client is not None
            mongo_database = MongoDatabase(self._mongo_client, database.database.name, transactions="never")
            self._databases.append(mongo_database)
            return MongoStore(mongo_database, entry_records=declaration)
        return SqlStore(database, entry_records=declaration)


@pytest.fixture
def store_factory(store_backend):
    """Build fresh stores on fresh in-memory databases and dispose them at teardown."""
    backend, mongo_client = store_backend
    databases = []
    factory = _StoreFactory(backend, mongo_client, databases)

    try:
        yield factory
    finally:
        for database in databases:
            if factory._backend == "mongo":
                database.client.drop_database(database.database.name)
            else:
                database.dispose()


@pytest.fixture
def mongo_test_database(mongo_test_client):
    """Yield a fresh live MongoDB database when the test URI is configured."""
    from httk.data.mongo import MongoDatabase

    name = f"httk_test_{uuid.uuid4().hex}"
    database = MongoDatabase(mongo_test_client, name)
    try:
        yield database
    finally:
        database.client.drop_database(name)
