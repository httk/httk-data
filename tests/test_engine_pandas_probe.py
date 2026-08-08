"""DuckDB's per-row pandas import probe must not re-search ``sys.path`` every row.

DuckDB binds statement parameters through its native extension, which probes
``import pandas`` roughly once per bound value. CPython does not cache a *failed*
import, so when pandas is absent every probe re-runs the full ``sys.path`` finder
search (profiled at ~14x slower for a 50k-row ``executemany``).
:meth:`~httk.data.db.engine.Database.duckdb` installs the standard ``None``
failed-import sentinel to short-circuit that search.

The pinned test interpreter HAS pandas, so the absence path is exercised by
simulating it: hiding pandas from :data:`sys.modules` and monkeypatching
:func:`importlib.util.find_spec`, restored on teardown.
"""

import builtins
import importlib.util
import sys

import pytest
import sqlalchemy

from httk.data.db import Database

_MISSING = object()


def _require_duckdb() -> None:
    pytest.importorskip("duckdb_engine")


def test_duckdb_executemany_probes_pandas_at_most_once():
    """A 1000-row executemany import-probes pandas at most once (via the cache or the sentinel)."""
    _require_duckdb()
    # Warm sys.modules if pandas is installed so the per-row probe hits the cache;
    # if pandas is absent, Database.duckdb() installs the sentinel that does the same.
    try:
        import pandas  # noqa: F401
    except ImportError:
        pass

    database = Database.duckdb()
    try:
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE probe (a INTEGER, b VARCHAR)"))
            table = sqlalchemy.table("probe", sqlalchemy.column("a"), sqlalchemy.column("b"))
            rows = [{"a": index, "b": f"x{index}"} for index in range(1000)]
            attempts = 0
            real_import = builtins.__import__

            def counting_import(name, *args, **kwargs):
                nonlocal attempts
                if name == "pandas" or name.startswith("pandas."):
                    attempts += 1
                return real_import(name, *args, **kwargs)

            builtins.__import__ = counting_import
            try:
                connection.execute(sqlalchemy.insert(table), rows)
            finally:
                builtins.__import__ = real_import
        assert attempts <= 1, f"pandas was import-probed {attempts} times for 1000 rows"
    finally:
        database.dispose()


def test_duckdb_installs_pandas_absence_sentinel(monkeypatch):
    """When pandas is unimportable, a None sys.modules sentinel is installed once and blocks re-search."""
    _require_duckdb()
    import httk.data.db.engine as engine_module

    # Simulate pandas being absent (the pinned venv has it): drop it from
    # sys.modules and make find_spec report it missing; restore on teardown.
    saved = sys.modules.pop("pandas", _MISSING)
    find_spec_calls: list[str] = []
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "pandas":
            find_spec_calls.append(name)
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    try:
        assert "pandas" not in sys.modules
        engine_module._install_missing_pandas_sentinel()
        assert sys.modules.get("pandas", _MISSING) is None  # the None sentinel is installed
        assert find_spec_calls == ["pandas"]  # probed exactly once
        with pytest.raises(ImportError):
            import pandas  # noqa: F401
        # Idempotent: a second call is a no-op — sys.modules already has the
        # sentinel, so find_spec is not consulted again.
        engine_module._install_missing_pandas_sentinel()
        assert sys.modules.get("pandas", _MISSING) is None
        assert find_spec_calls == ["pandas"]
    finally:
        if saved is _MISSING:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = saved


def test_duckdb_leaves_installed_pandas_untouched(monkeypatch):
    """When pandas is importable, no sentinel is planted and any prior entry is preserved."""
    _require_duckdb()
    import httk.data.db.engine as engine_module

    saved = sys.modules.pop("pandas", _MISSING)
    try:
        # pandas is genuinely importable in this venv: find_spec (real) finds it,
        # so the helper must not touch sys.modules.
        engine_module._install_missing_pandas_sentinel()
        assert sys.modules.get("pandas", _MISSING) is _MISSING  # nothing planted
    finally:
        if saved is _MISSING:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = saved
