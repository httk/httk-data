"""PostgreSQL engine hooks: the per-dialect behavior the neutral SQL engine dispatches to.

:mod:`httk.store.backend.sql.engine` keeps only neutral dialect dispatch; the
PostgreSQL-specific bodies live here and are imported lazily on first use by the
``_DIALECT_HOOKS`` registry in :mod:`httk.store.backend.sql.engine`.
"""

import sqlalchemy


def connection_uses_autocommit(connection: sqlalchemy.Connection) -> bool:
    """Return whether the live PostgreSQL DBAPI connection is in autocommit mode.

    psycopg 3 exposes the live autocommit flag on the DBAPI connection; the
    transactional-profile validator relies on this to reject a misconfigured
    AUTOCOMMIT PostgreSQL engine.

    :param connection: The live PostgreSQL SQLAlchemy connection to inspect.
    :return: Whether the underlying DBAPI connection is in autocommit mode.
    """
    return bool(getattr(connection.connection.driver_connection, "autocommit", False))
