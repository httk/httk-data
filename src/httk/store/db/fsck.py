"""Integrity repair and dependency collection for SQL permanentization stores."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import sqlalchemy

from httk.store.db.graph import LogicalEdgeGraph
from httk.store.db.layout import METADATA_TABLE_NAME, actual_table_names
from httk.store.db.mapping import (
    CONTENT_ID_COLUMN,
    REPLACED_BY_COLUMN,
    ROLE_COLUMN,
    SID_COLUMN,
    TS_END_COLUMN,
    TS_START_COLUMN,
    backing_dispatch_column_name,
    entry_dispatch_table_name,
)
from httk.store.db.schema import TableSchema, resolve_schema
from httk.store.store_timestamp import FUTURE_TIMESTAMP_SLACK_NS

if TYPE_CHECKING:
    from httk.store.db.store import SqlStore

__all__ = ["FsckSummary", "FsckTableSummary", "run_fsck"]


@dataclass(frozen=True)
class FsckTableSummary:
    """Counters for one physical record, child, or dispatch table."""

    examined: int = 0
    repaired: int = 0
    conflicts: int = 0
    deleted: int = 0


@dataclass(frozen=True)
class FsckSummary:
    """Immutable SQL fsck report, modeled after Mongo's public summary."""

    tables: Mapping[str, FsckTableSummary]
    violations: tuple[str, ...]


class _Counter:
    def __init__(self) -> None:
        self.examined = self.repaired = self.conflicts = self.deleted = 0

    def freeze(self) -> FsckTableSummary:
        return FsckTableSummary(self.examined, self.repaired, self.conflicts, self.deleted)


def run_fsck(
    store: SqlStore,
    *,
    repair: bool = True,
    collect_garbage: bool = True,
    repair_conflicts: bool = False,
    clamp_future_timestamps: bool = False,
    known_types: tuple[type, ...] = (),
    exclusive: bool = False,
) -> FsckSummary:
    """Repair dispatches, sweep incomplete residue, and report logical dangling references.

    DuckDB does not serialize a read-then-delete fsck against concurrent
    writers.  Callers must therefore pass ``exclusive=True`` there, explicitly
    acknowledging that they have taken the database offline from all writers.
    SQLite transactional stores instead issue ``BEGIN IMMEDIATE`` themselves.
    """
    if store._database.engine.dialect.name == "duckdb" and not exclusive:
        raise RuntimeError("DuckDB fsck requires exclusive=True and offline ownership from all writers")
    verification_only = store.write_profile == "degraded" and not repair and not collect_garbage
    schemas: dict[str, TableSchema] = {}
    pending = [
        *known_types,
        *store._known_record_types,
        *(record for family in store.layout.families for record in family.records),
    ]
    while pending:
        record = pending.pop()
        schema = resolve_schema(record)
        if schema.table_name in schemas:
            continue
        schemas[schema.table_name] = schema
        pending.extend(schema.referenced_classes())
    # Rebuild the in-memory mapping after a reopen; this is deliberately not
    # DDL and therefore does not make missing tables appear.
    store._register_tables(tuple(schema.cls for schema in schemas.values()))
    graph = LogicalEdgeGraph.from_store(store, tuple(schemas.values()))
    counters: defaultdict[str, _Counter] = defaultdict(_Counter)
    violations: list[str] = []
    timestamp_repaired = False
    # A degraded verification has no repair, collection, lease, or dirty-row
    # semantics.  In particular it must not manufacture a metadata ``lease``
    # row merely to inspect an otherwise untouched store.
    connection_scope = store._read_connection() if verification_only else store._fsck_connection()
    with connection_scope as connection:
        present = actual_table_names(connection)
        expected = set(graph.tables) | {METADATA_TABLE_NAME, "_httk_sid_counters"}
        unattributed = sorted(name for name in present if name not in expected and not name.startswith("_httk_"))
        for name in unattributed:
            counters[name].conflicts += 1
            violations.append(f"table {name!r} cannot be attributed to a known schema and blocks fsck sweep")
        # Attribution refusal happens before *any* mutation.  A verification
        # call is read-only by construction, and even a repair call does not
        # touch a partially attributable database.
        mutation_allowed = not unattributed
        timestamp_repaired = _check_future_timestamps(
            store,
            connection,
            present,
            graph,
            counters,
            violations,
            clamp_future_timestamps and repair and mutation_allowed,
        )
        if store.store_timestamp_mode == "versioned":
            _check_version_invariants(store, connection, present, graph, schemas, counters, violations)
        if repair and mutation_allowed:
            _repair_dispatches(store, connection, present, counters, violations, True, repair_conflicts)
        else:
            _repair_dispatches(store, connection, present, counters, violations, False, False)
        marked = _mark(store, connection, graph, present, counters, violations, repair and mutation_allowed)
        if collect_garbage and not unattributed:
            for table_name, schema in schemas.items():
                if table_name not in present:
                    continue
                table = store._table(table_name)
                survivors = marked.get(table_name, set())
                condition = table.c[ROLE_COLUMN] == 0
                if survivors:
                    condition = sqlalchemy.and_(condition, table.c[SID_COLUMN].not_in(survivors))
                result = connection.execute(sqlalchemy.delete(table).where(condition))
                counters[table_name].deleted += max(result.rowcount, 0)
            # Deleting unreachable dependency parents can make their owned
            # element rows ownerless.  Sweep afterwards so one pass reaches a
            # physical fixpoint.
            _sweep_ownerless_children(connection, graph, present, counters)
        elif collect_garbage:
            violations.append("sweep aborted because unattributed application tables exist")
        store._clear_identity_caches()
    if timestamp_repaired:
        with store._mutation_lock, store._read_connection() as connection:
            store._initialize_store_timestamp_mark(connection)
    return FsckSummary(
        MappingProxyType({name: value.freeze() for name, value in sorted(counters.items())}), tuple(violations)
    )


def _check_future_timestamps(
    store: SqlStore,
    connection: sqlalchemy.Connection,
    present: set[str] | frozenset[str],
    graph: LogicalEdgeGraph,
    counters: defaultdict[str, _Counter],
    violations: list[str],
    clamp: bool,
) -> bool:
    """Report parent timestamps beyond the current clock plus writer/checker slack.

    Clamping is destructive to historic-query fidelity and is a last-resort
    repair for data written with a badly skewed clock.
    """
    if not store.store_timestamps:
        return False
    now_ns = store._clock()
    resolution = store.store_timestamp_resolution
    limit_units = (now_ns + FUTURE_TIMESTAMP_SLACK_NS) // resolution
    now_units = now_ns // resolution
    repaired_any = False
    for name in graph.tables:
        if name not in present or name not in store._metadata.tables:
            continue
        table = store._table(name)
        if ROLE_COLUMN not in table.c or TS_START_COLUMN not in table.c:
            continue
        # Both lifecycle bounds are checked; ts_end exists only on versioned
        # family tables. Clamping ts_end down to now can make a successor's
        # ts_start exceed its predecessor's (now clamped) ts_end — the lineage
        # check reports that as a <= violation, which is acceptable for a store
        # written under a badly skewed clock.
        for column_name in (TS_START_COLUMN, TS_END_COLUMN):
            if column_name not in table.c:
                continue
            column = table.c[column_name]
            rows = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN], column).where(column.is_not(None), column > limit_units)
            ).all()
            for sid, value in rows:
                counters[name].examined += 1
                future_ns = int(value) * resolution
                limit_ns = limit_units * resolution
                if clamp:
                    connection.execute(
                        sqlalchemy.update(table).where(table.c[SID_COLUMN] == sid).values({column_name: now_units})
                    )
                    repaired_any = True
                    counters[name].repaired += 1
                    violations.append(
                        f"table {name!r} sid {sid} {column_name} {future_ns} ns exceeds {limit_ns} ns; "
                        f"clamped to {now_ns} ns"
                    )
                else:
                    counters[name].conflicts += 1
                    violations.append(f"table {name!r} sid {sid} {column_name} {future_ns} ns exceeds {limit_ns} ns")
    return repaired_any


def _check_version_invariants(
    store: SqlStore,
    connection: sqlalchemy.Connection,
    present: set[str] | frozenset[str],
    graph: LogicalEdgeGraph,
    schemas: Mapping[str, TableSchema],
    counters: defaultdict[str, _Counter],
    violations: list[str],
) -> None:
    """Report versioning-invariant violations on versioned family tables (report-only).

    Checks the half-open lifetime interval bookkeeping: ``ts_end`` and
    ``replaced_by_sid`` are set together (pairing), a closed interval is
    non-empty (ordering), the successor exists and does not start after the
    predecessor closed (lineage), and — only on backends without partial unique
    indexes — no author-Unique value repeats among current rows.
    """
    for name in graph.tables:
        if name not in store._versioned_tables or name not in present or name not in store._metadata.tables:
            continue
        table = store._table(name)
        sid_col = table.c[SID_COLUMN]
        start_col = table.c[TS_START_COLUMN]
        end_col = table.c[TS_END_COLUMN]
        repl_col = table.c[REPLACED_BY_COLUMN]
        # Pairing: ts_end and replaced_by_sid must both be set or both be null.
        for sid, end, repl in connection.execute(
            sqlalchemy.select(sid_col, end_col, repl_col).where(
                sqlalchemy.or_(
                    sqlalchemy.and_(end_col.is_(None), repl_col.is_not(None)),
                    sqlalchemy.and_(end_col.is_not(None), repl_col.is_(None)),
                )
            )
        ):
            counters[name].conflicts += 1
            if end is None:
                violations.append(f"table {name!r} sid {sid} has replaced_by_sid {repl} but no ts_end")
            else:
                violations.append(f"table {name!r} sid {sid} has ts_end {end} but no replaced_by_sid")
        # Ordering: a closed interval must be non-empty (ts_end > ts_start).
        for sid, start, end in connection.execute(
            sqlalchemy.select(sid_col, start_col, end_col).where(end_col.is_not(None), end_col <= start_col)
        ):
            counters[name].conflicts += 1
            violations.append(f"table {name!r} sid {sid} ts_end {end} is not after ts_start {start}")
        # Lineage: replaced_by_sid must reference an existing row in the same
        # table whose ts_start does not fall after the predecessor's ts_end
        # (equality is a fresh insert; earlier is a converged lineage).
        successor = table.alias("successor")
        for sid, repl, end, succ_sid, succ_start in connection.execute(
            sqlalchemy.select(sid_col, repl_col, end_col, successor.c[SID_COLUMN], successor.c[TS_START_COLUMN])
            .select_from(table.outerjoin(successor, repl_col == successor.c[SID_COLUMN]))
            .where(repl_col.is_not(None))
        ):
            if succ_sid is None:
                counters[name].conflicts += 1
                violations.append(f"table {name!r} sid {sid} replaced_by_sid {repl} references a missing row")
            elif end is not None and succ_start > end:
                counters[name].conflicts += 1
                violations.append(
                    f"table {name!r} sid {sid} successor {repl} ts_start {succ_start} is after ts_end {end}"
                )
        # Unique-among-current: partial unique indexes (ts_end IS NULL) enforce
        # this in the engine, so it is only re-checked on backends without them.
        if store.backend_facts.supports_partial_unique_indexes:
            continue
        schema = schemas.get(name)
        if schema is None:
            continue
        for spec in schema.fields:
            if spec.role == "child":
                continue
            for column_spec in spec.columns:
                if not column_spec.unique:
                    continue
                column = table.c[column_spec.name]
                # NULLs never conflict, mirroring SQL unique-index semantics.
                for value, count in connection.execute(
                    sqlalchemy.select(column, sqlalchemy.func.count())
                    .where(end_col.is_(None), column.is_not(None))
                    .group_by(column)
                    .having(sqlalchemy.func.count() > 1)
                ):
                    counters[name].conflicts += 1
                    violations.append(
                        f"table {name!r} column {column_spec.name!r} value {value!r} appears in {count} current rows"
                    )


def _repair_dispatches(
    store: SqlStore,
    connection: sqlalchemy.Connection,
    present: set[str] | frozenset[str],
    counters,
    violations,
    repair: bool,
    repair_conflicts: bool,
) -> None:
    for family in store.layout.families:
        if len(family.records) < 2:
            continue
        name = entry_dispatch_table_name(family.name)
        backing_names = [resolve_schema(record).table_name for record in family.records]
        if name not in present:
            nonempty_backings = [
                backing_name
                for backing_name in backing_names
                if backing_name in present
                and connection.execute(
                    sqlalchemy.select(sqlalchemy.func.count()).select_from(store._table(backing_name))
                ).scalar_one()
            ]
            if nonempty_backings:
                counters[name].conflicts += 1
                violations.append(
                    f"dispatch {name!r} is missing while backing rows exist in {tuple(nonempty_backings)!r}"
                )
            continue
        dispatch = store._table(name)
        for row in connection.execute(sqlalchemy.select(dispatch)).mappings():
            counters[name].examined += 1
            populated = [
                (record, int(row[backing_dispatch_column_name(record_name)]))
                for record_name, record in zip(family.record_names, family.records, strict=True)
                if row[backing_dispatch_column_name(record_name)] is not None
            ]
            valid = len(populated) == 1
            if valid:
                record, sid = populated[0]
                backing_name = resolve_schema(record).table_name
                if backing_name not in present:
                    valid = False
                else:
                    backing = store._table(backing_name)
                    valid = (
                        connection.execute(
                            sqlalchemy.select(backing.c[CONTENT_ID_COLUMN]).where(backing.c[SID_COLUMN] == sid)
                        ).scalar_one_or_none()
                        == row[CONTENT_ID_COLUMN]
                    )
            if valid:
                continue
            counters[name].conflicts += 1
            violations.append(f"dispatch {name!r} has an invalid backing association")
            if repair and repair_conflicts:
                connection.execute(
                    sqlalchemy.delete(dispatch).where(dispatch.c[CONTENT_ID_COLUMN] == row[CONTENT_ID_COLUMN])
                )
                counters[name].deleted += 1
        if not repair:
            continue
        for record_name, record in zip(family.record_names, family.records, strict=True):
            backing = store._table(resolve_schema(record).table_name)
            if backing.name not in present:
                continue
            column = backing_dispatch_column_name(record_name)
            for sid, content in connection.execute(
                sqlalchemy.select(backing.c[SID_COLUMN], backing.c[CONTENT_ID_COLUMN]).where(
                    backing.c[ROLE_COLUMN] == 1
                )
            ):
                if (
                    connection.execute(
                        sqlalchemy.select(dispatch.c[CONTENT_ID_COLUMN]).where(dispatch.c[CONTENT_ID_COLUMN] == content)
                    ).first()
                    is not None
                ):
                    continue
                values = {dispatch_column.name: None for dispatch_column in dispatch.columns}
                values[CONTENT_ID_COLUMN] = content
                values[column] = sid
                connection.execute(sqlalchemy.insert(dispatch).values(values))
                counters[name].repaired += 1


def _sweep_ownerless_children(
    connection: sqlalchemy.Connection, graph: LogicalEdgeGraph, present: set[str] | frozenset[str], counters
) -> None:
    for edge in graph.ownership():
        if edge.source_table not in present or edge.target_table not in present:
            continue
        assert edge.target_column is not None
        child = sqlalchemy.table(edge.target_table, sqlalchemy.column(edge.target_column))
        parent = sqlalchemy.table(edge.source_table, sqlalchemy.column(SID_COLUMN))
        result = connection.execute(
            sqlalchemy.delete(child).where(
                ~sqlalchemy.exists(sqlalchemy.select(1).where(parent.c[SID_COLUMN] == child.c[edge.target_column]))
            )
        )
        counters[edge.target_table].deleted += max(result.rowcount, 0)


def _mark(
    store: SqlStore,
    connection: sqlalchemy.Connection,
    graph: LogicalEdgeGraph,
    present,
    counters,
    violations,
    repair_roles: bool,
) -> dict[str, set[int]]:
    marked: dict[str, set[int]] = defaultdict(set)
    queue: deque[tuple[str, int]] = deque()

    def add(name: str, sid: object) -> None:
        if name in marked and isinstance(sid, int) and sid not in marked[name]:
            marked[name].add(sid)
            queue.append((name, sid))

    # Initialize every parent table key so dependencies can be queued.
    for name in graph.tables:
        if name in present and name in store._metadata.tables and SID_COLUMN in store._table(name).c:
            marked[name]
            table = store._table(name)
            for sid, role in connection.execute(sqlalchemy.select(table.c[SID_COLUMN], table.c[ROLE_COLUMN])):
                if role in (0, 1):
                    continue
                counters[name].conflicts += 1
                violations.append(f"table {name!r} sid {sid} has invalid _httk_role {role!r}")
                if repair_roles:
                    # Invalid roles are normalized to dependency.  This is the
                    # conservative repair: corrupt data never becomes a root
                    # merely because fsck was asked to repair it.
                    connection.execute(
                        sqlalchemy.update(table).where(table.c[SID_COLUMN] == sid).values({ROLE_COLUMN: 0})
                    )
                    counters[name].repaired += 1
            for sid in connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[ROLE_COLUMN] == 1)
            ).scalars():
                counters[name].examined += 1
                add(name, sid)
    for edge in graph.edges:
        if (
            edge.kind != "dispatch"
            or edge.source_table not in present
            or edge.source_table not in store._metadata.tables
        ):
            continue
        table = store._table(edge.source_table)
        assert edge.source_column is not None
        for sid in connection.execute(
            sqlalchemy.select(table.c[edge.source_column]).where(table.c[edge.source_column].is_not(None))
        ).scalars():
            add(edge.target_table, sid)
    outgoing: defaultdict[str, list] = defaultdict(list)
    for edge in graph.edges:
        outgoing[edge.source_table].append(edge)
    while queue:
        table_name, sid = queue.popleft()
        if table_name not in present:
            violations.append(f"dangling logical reference to absent table {table_name!r}/{sid}")
            continue
        table = store._table(table_name)
        row = connection.execute(sqlalchemy.select(table).where(table.c[SID_COLUMN] == sid)).mappings().one_or_none()
        if row is None:
            violations.append(f"dangling logical reference to {table_name!r}/{sid}")
            continue
        for edge in outgoing[table_name]:
            if edge.kind == "reference":
                assert edge.source_column is not None
                add(edge.target_table, row[edge.source_column])
            elif edge.kind == "ownership":
                assert edge.target_column is not None
                child = store._table(edge.target_table)
                for child_row in connection.execute(
                    sqlalchemy.select(child).where(child.c[edge.target_column] == sid)
                ).mappings():
                    for child_edge in outgoing[edge.target_table]:
                        if child_edge.kind == "child_element":
                            assert child_edge.source_column is not None
                            add(child_edge.target_table, child_row[child_edge.source_column])
    return marked
