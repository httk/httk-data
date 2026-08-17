"""Shared lifecycle (time-slice) query predicate for versioned-mode stores.

Every query path that is *entered from the top* — a root search variable, a
:meth:`~httk.store.db.store.SqlStore.referring` scan, an entry provider
enumeration — must see one consistent time slice: the current view by default,
or a half-open ``as_of`` slice ``[ts_start, ts_end)``. This module holds the one
clause builder those callers share so they cannot drift apart.

A versioned family table carries its own lifetime columns, so its predicate is a
direct interval test. A non-family table (a deduplicated dependency or child
content row) has none, so its visibility is derived: it is visible when it was
saved standalone (``_httk_role = 1``) or when some current family row still owns
it through a statically computed ownership chain (an ``EXISTS`` per
:class:`~httk.store.db.graph.LifecyclePath`). Intermediate ownership hops carry
no time predicate: a pinned sub-row is always inserted at-or-before its referrer
in the same transaction, so the referrer's lifetime alone governs the slice.

Known approximation: ``_httk_role`` promotion (a dependency later saved
top-level in its own right) is not timestamped, so the role term under ``as_of``
reflects the *current* role and errs toward visibility.
"""

from typing import TYPE_CHECKING, Any

import sqlalchemy

from httk.store.db.graph import LifecyclePath
from httk.store.db.mapping import ROLE_COLUMN, SID_COLUMN, TS_END_COLUMN, TS_START_COLUMN
from httk.store.store_common import LifecycleScopeError

if TYPE_CHECKING:
    from httk.store.db.store import SqlStore

__all__ = ["LifecycleScopeError", "lifecycle_clause"]


def _family_lifetime(alias: sqlalchemy.FromClause, as_of_units: int | None) -> sqlalchemy.ColumnElement[bool]:
    """The interval predicate for a versioned family row on ``alias``."""
    if as_of_units is None:
        return alias.c[TS_END_COLUMN].is_(None)
    # Half-open [ts_start, ts_end): at T == successor.ts_start the successor is
    # visible and the closed predecessor (ts_end == T) is not.
    return sqlalchemy.and_(
        alias.c[TS_START_COLUMN] <= as_of_units,
        sqlalchemy.or_(alias.c[TS_END_COLUMN].is_(None), alias.c[TS_END_COLUMN] > as_of_units),
    )


def _path_exists(
    store: "SqlStore",
    outer_alias: sqlalchemy.FromClause,
    path: LifecyclePath,
    as_of_units: int | None,
) -> sqlalchemy.ColumnElement[bool]:
    """A correlated ``EXISTS`` asserting a live family row owns ``outer_alias`` via ``path``."""
    froms: list[sqlalchemy.FromClause] = []
    conditions: list[sqlalchemy.ColumnElement[bool]] = []
    previous_sid: Any = outer_alias.c[SID_COLUMN]
    referrer: sqlalchemy.FromClause | None = None
    for hop in path.hops:
        if hop.child_table is not None:
            assert hop.child_element_column is not None and hop.child_parent_column is not None
            child = store._table(hop.child_table).alias()
            referrer = store._table(hop.referrer_table).alias()
            froms.extend((child, referrer))
            conditions.append(child.c[hop.child_element_column] == previous_sid)
            conditions.append(referrer.c[SID_COLUMN] == child.c[hop.child_parent_column])
        else:
            assert hop.referrer_column is not None
            referrer = store._table(hop.referrer_table).alias()
            froms.append(referrer)
            conditions.append(referrer.c[hop.referrer_column] == previous_sid)
        previous_sid = referrer.c[SID_COLUMN]
    assert referrer is not None  # a LifecyclePath always has at least one hop
    conditions.append(_family_lifetime(referrer, as_of_units))
    return sqlalchemy.select(sqlalchemy.literal(1)).select_from(*froms).where(*conditions).exists()


def lifecycle_clause(
    store: "SqlStore",
    alias: sqlalchemy.FromClause,
    table_name: str,
    as_of_units: int | None,
) -> sqlalchemy.ColumnElement[bool]:
    """The current-view (or ``as_of``) predicate for a top-entered scope on ``table_name``.

    A versioned family table yields its own interval predicate. A non-family
    table yields ``role = 1 [AND ts_start <= T] OR EXISTS(path) ...`` over its
    statically computed ownership paths.

    :param store: The versioned SQL store owning the tables.
    :param alias: The table alias (or table) the query binds this scope to.
    :param table_name: The logical name of the table ``alias`` refers to.
    :param as_of_units: The historic cutoff in integer store units, or ``None`` for current view.
    :return: A SQLAlchemy boolean clause selecting the time slice.
    :raises LifecycleScopeError: If ``table_name`` is a non-family table caught in a reference cycle.
    """
    if store._is_versioned_family_table(table_name):
        return _family_lifetime(alias, as_of_units)
    path_map = store._lifecycle_path_map()
    if table_name in path_map and path_map[table_name] is None:
        raise LifecycleScopeError(
            f"the non-family table {table_name!r} is caught in a reference cycle and cannot be scoped "
            "to a time slice; query with scoped=False to disable lifecycle filtering"
        )
    role_term: sqlalchemy.ColumnElement[bool] = alias.c[ROLE_COLUMN] == 1
    if as_of_units is not None:
        role_term = sqlalchemy.and_(role_term, alias.c[TS_START_COLUMN] <= as_of_units)
    paths = path_map.get(table_name) or ()
    return sqlalchemy.or_(role_term, *(_path_exists(store, alias, path, as_of_units) for path in paths))
