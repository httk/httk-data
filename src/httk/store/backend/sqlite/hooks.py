"""SQLite engine hooks: the per-dialect behavior the neutral SQL engine dispatches to.

:mod:`httk.store.backend.sql.engine` keeps only neutral dialect dispatch; the
SQLite-specific bodies live here and are imported lazily on first use by the
``_DIALECT_HOOKS`` registry in :mod:`httk.store.backend.sql.engine`.
"""

from typing import Any

import sqlalchemy

from httk.store.backend.sql.engine import _fraction_scaled_equal, _install_scalar_function


def install_engine_functions(engine: sqlalchemy.Engine) -> None:
    """Install the exact-fraction scalar function on every SQLite connection.

    Fraction values are persisted canonically as text because SQL integer
    products can overflow long before a valid Python :class:`~fractions.Fraction`
    does; ``httk_fraction_scaled_equal`` preserves exact comparisons without
    touching presentation float columns.

    :param engine: The SQLite SQLAlchemy engine to install the function on.
    :return: None.
    """

    def register(dbapi_connection: Any) -> None:
        dbapi_connection.create_function("httk_fraction_scaled_equal", 4, _fraction_scaled_equal)

    _install_scalar_function(engine, register)


def connection_uses_autocommit(connection: sqlalchemy.Connection) -> bool:
    """Return whether the live SQLite DBAPI connection is in autocommit mode.

    ``sqlite3.Connection.isolation_level is None`` is SQLite's documented
    autocommit mode; the SQLAlchemy checks cover alternate DBAPI wrappers while
    still requiring the live connection to agree.

    :param connection: The live SQLite SQLAlchemy connection to inspect.
    :return: Whether the underlying DBAPI connection is in autocommit mode.
    """
    raw = connection.connection.driver_connection
    if getattr(raw, "isolation_level", object()) is None:
        return True
    execution_mode = connection.get_execution_options().get("isolation_level")
    if isinstance(execution_mode, str) and execution_mode.upper() == "AUTOCOMMIT":
        return True
    try:
        return str(connection.get_isolation_level()).upper() == "AUTOCOMMIT"
    except (AttributeError, NotImplementedError):
        return False
