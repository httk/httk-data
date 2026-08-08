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

import pytest
from httk.data.db import Database, SqlStore

_PYTHONPATH = os.environ.get("PYTHONPATH")
if _PYTHONPATH:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        os.path.abspath(entry) if entry else entry for entry in _PYTHONPATH.split(os.pathsep)
    )


@pytest.fixture(params=["sqlite", "duckdb"])
def store_backend(request):
    """Select each backend supported by the neutral store behavior suite."""
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
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
        else:
            # A later MongoDB backend can join by adding one param + one branch.
            database = Database.duckdb()
        self._databases.append(database)
        declaration = entry_records if entry_records is not None else {}
        store = SqlStore(database, entry_records=declaration)
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
        # A future MongoDB branch returns a new MongoStore over the same server database.
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
            database.dispose()
