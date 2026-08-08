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

from httk.data.db import Database, SqlStore

_PYTHONPATH = os.environ.get("PYTHONPATH")
if _PYTHONPATH:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        os.path.abspath(entry) if entry else entry for entry in _PYTHONPATH.split(os.pathsep)
    )


@pytest.fixture(params=["sqlite", "duckdb", "mongo"])
def store_backend(request):
    """Select each backend supported by the neutral store behavior suite."""
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
    if request.param == "mongo":
        uri = os.environ.get("HTTK_TEST_MONGODB_URI")
        if not uri:
            pytest.skip("HTTK_TEST_MONGODB_URI is not set")
        from pymongo import MongoClient

        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=1000)
            client.admin.command("ping")
        except Exception as error:
            pytest.skip(f"MongoDB test server is unreachable: {error}")
        finally:
            try:
                client.close()
            except UnboundLocalError:
                pass
    yield request.param


class _StoreFactory:
    """Callable factory returning real stores plus a same-database reopen path."""

    def __init__(self, backend, databases):
        self._backend = backend
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
            uri = os.environ["HTTK_TEST_MONGODB_URI"]
            database = MongoDatabase.connect(uri, database=name, transactions="never")
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

            mongo_database = MongoDatabase.connect(
                os.environ["HTTK_TEST_MONGODB_URI"], database=database.database.name, transactions="never"
            )
            self._databases.append(mongo_database)
            return MongoStore(mongo_database, entry_records=declaration)
        return SqlStore(database, entry_records=declaration)


@pytest.fixture
def store_factory(store_backend):
    """Build fresh stores on fresh in-memory databases and dispose them at teardown."""
    databases = []
    factory = _StoreFactory(store_backend, databases)

    try:
        yield factory
    finally:
        for database in databases:
            if factory._backend == "mongo":
                database.client.drop_database(database.database.name)
            database.dispose()


@pytest.fixture
def mongo_test_database():
    """Yield a fresh live MongoDB database when the test URI is configured."""
    uri = os.environ.get("HTTK_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HTTK_TEST_MONGODB_URI is not set")
    from httk.data.mongo import MongoDatabase

    name = f"httk_test_{uuid.uuid4().hex}"
    try:
        database = MongoDatabase.connect(uri, database=name)
        database.database.command("ping")
    except Exception as error:
        pytest.skip(f"MongoDB test server is unreachable: {error}")
    try:
        yield database
    finally:
        database.client.drop_database(name)
        database.dispose()
