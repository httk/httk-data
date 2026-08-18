"""DuckDB engine hooks: the per-dialect behavior the neutral SQL engine dispatches to.

:mod:`httk.store.backend.sql.engine` keeps only neutral dialect dispatch; the
DuckDB-specific bodies live here and are imported lazily on first use by the
``_DIALECT_HOOKS`` registry in :mod:`httk.store.backend.sql.engine`.
"""

import importlib
from typing import Any

import sqlalchemy

from httk.store.backend.sql.engine import _fraction_scaled_equal, _install_scalar_function


def install_engine_functions(engine: sqlalchemy.Engine) -> None:
    """Install the exact-fraction scalar function on every DuckDB connection.

    Fraction values are persisted canonically as text because SQL integer
    products can overflow long before a valid Python :class:`~fractions.Fraction`
    does; ``httk_fraction_scaled_equal`` preserves exact comparisons without
    touching presentation float columns.

    :param engine: The DuckDB SQLAlchemy engine to install the function on.
    :return: None.
    """

    def register(dbapi_connection: Any) -> None:
        duckdb = importlib.import_module("duckdb")
        try:
            dbapi_connection.create_function(
                "httk_fraction_scaled_equal",
                _fraction_scaled_equal,
                return_type=duckdb.sqltypes.BOOLEAN,
            )
        except Exception as error:
            # DuckDB registers functions in the database catalog, rather than
            # per DBAPI connection.  A simultaneous peer may already have
            # installed it, in which case this connection can use the same
            # catalog function immediately.  Never remove/recreate: doing so
            # races an active peer's query.
            if not _duckdb_duplicate_function_error(error):
                raise

    _install_scalar_function(engine, register)


def _duckdb_duplicate_function_error(error: Exception) -> bool:
    """Whether DuckDB reports a catalog function already installed by a peer."""
    text = str(error).casefold()
    return "already exists" in text or "already created" in text
