"""Stage-side finalization for fresh SQL bulk ingests.

Unlike :mod:`httk.data.db.bulk_parallel`'s parity merger this module never
loads provisional rows into a record table.  It builds temporary maps over the
external stage, computes the reachable canonical rows, and projects each real
table exactly once with final dense sids.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import sqlalchemy

from httk.data.db.mapping import CONTENT_ID_COLUMN, SID_COLUMN
from httk.data.store_common import EntryMetadataConflictError

if TYPE_CHECKING:
    from httk.data.db.bulk import BulkIngest


class DeferredFinalizer:
    """Set-wise, non-destructive finalizer for one empty-store ingest."""

    def __init__(self, ingest: BulkIngest, manifests: list[Any]) -> None:
        self.ingest = ingest
        self.store = ingest._store
        assert ingest._connection is not None
        self.connection = ingest._connection
        self.manifests = manifests
        self.graph = ingest._logical_graph()
        self.parents = dict(ingest._parent_schema)
        self.stage_views: dict[str, str] = {}
        self.maps: dict[str, str] = {}
        self.finals: dict[str, str] = {}
        self.objects: list[str] = []
        self.finalize_timings: dict[str, float] = {}
        self._final_by_stage: dict[str, dict[int, int]] = {}
        self.root_stage: str | None = None
        self.root_occurrences: str | None = None

    # ------------------------------------------------------------------ lifecycle

    def run(self) -> None:
        self._timed("attach_views", self._make_stage_views)
        self._timed("maps", self._make_maps)
        self._timed("fixpoint", self._collapse_by_value_to_fixpoint)
        self._timed("conflicts", self._verify_metadata)
        self._timed("survivors", self._make_survivors)
        self._timed("final_sids", self._make_final_sids)
        self._timed("load", self._load_real_tables)
        self._timed("resolved_sids", self._populate_returned_sids)
        self._timed("dispatch", self._rebuild_dispatch)
        if self.connection.dialect.name == "clickhousedb":
            self.ingest._after_clickhouse_projection()

    def _timed(self, name: str, operation: Any) -> None:
        started = time.perf_counter()
        operation()
        self.finalize_timings[name] = time.perf_counter() - started

    def cleanup(self) -> None:
        """Drop every main-database temporary relation before marker clear."""
        started = time.perf_counter()
        for name in reversed(self.objects):
            with contextlib.suppress(Exception):
                self.connection.execute(sqlalchemy.text(f'DROP VIEW IF EXISTS "{name}"'))
            with contextlib.suppress(Exception):
                self.connection.execute(sqlalchemy.text(f'DROP TABLE IF EXISTS "{name}"'))
            with contextlib.suppress(Exception):
                self.connection.execute(sqlalchemy.text(f'DROP INDEX IF EXISTS "{name}"'))
        self.objects.clear()
        self.finalize_timings["cleanup"] = time.perf_counter() - started

    # ------------------------------------------------------------------ stage views

    @staticmethod
    def _q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _temp_name(self, kind: str, table: str) -> str:
        return f"_httk_deferred_{kind}_{table}"

    def _create_view(self, name: str, query: str) -> None:
        if self.connection.dialect.name == "clickhousedb":
            self.connection.execute(sqlalchemy.text(f'CREATE VIEW {self._q(name)} AS {query}'))
        else:
            self.connection.execute(sqlalchemy.text(f'CREATE TEMP VIEW {self._q(name)} AS {query}'))
        self.objects.append(name)

    def _create_table(self, name: str, query: str) -> None:
        if self.connection.dialect.name == "clickhousedb":
            # ClickHouse has no connection-local temporary CTAS relations.
            # ``tuple()`` is the neutral key for maps/finals whose projected
            # shape has no stage sid; stage tables themselves use ``sid``.
            self.connection.execute(
                sqlalchemy.text(f'CREATE TABLE {self._q(name)} ENGINE = MergeTree ORDER BY tuple() AS {query}')
            )
        else:
            self.connection.execute(sqlalchemy.text(f'CREATE TEMP TABLE {self._q(name)} AS {query}'))
        self.objects.append(name)

    def _index_relation(self, name: str, *columns: str, unique: bool = False) -> None:
        if not self.store.backend_facts.supports_adhoc_indexes:
            return
        suffix = "_".join(columns)
        index = f"{name}_{suffix}_idx"
        unique_sql = "UNIQUE " if unique else ""
        self.connection.execute(
            sqlalchemy.text(
                f"CREATE {unique_sql}INDEX IF NOT EXISTS {self._q(index)} ON {self._q(name)} ({', '.join(self._q(c) for c in columns)})"
            )
        )
        self.objects.append(index)

    def _make_stage_views(self) -> None:
        """Expose every shard relation as one read-only logical stage view."""
        sources: dict[str, list[str]] = {}
        backend = self.connection.dialect.name
        if backend == "clickhousedb":
            self.stage_views.update(self.ingest._clickhouse_stage_tables)
            self.objects.extend(self.ingest._clickhouse_stage_tables.values())
            self.root_stage = self.stage_views.pop("_httk_roots", None)
            return
        if backend == "duckdb":
            parquet: dict[str, list[str]] = {}
            duckdb_stages: list[tuple[str, list[str]]] = []
            for index, manifest in enumerate(self.manifests):
                if manifest.shards.get("format") == "duckdb":
                    duckdb_stages.append((str(manifest.shards["db"]), list(manifest.shards.get("tables", ()))))
                for table, files in manifest.shards.items():
                    if table not in {"db", "tables", "format"} and isinstance(files, list):
                        parquet.setdefault(table, []).extend(files)
            for index, (path, tables) in enumerate(duckdb_stages):
                alias = f"httk_deferred_stage_{index}"
                literal = "'" + path.replace("'", "''") + "'"
                self.connection.execute(sqlalchemy.text(f"ATTACH {literal} AS {alias}"))
                self.ingest._parallel_attached.append(alias)
                for table in tables:
                    sources.setdefault(table, []).append(f"{alias}.{self._q(table)}")
            for table, files in parquet.items():
                parameters = {f"p{index}": path for index, path in enumerate(files)}
                values = ", ".join(f":p{index}" for index in range(len(files)))
                # The paths are bound while defining a view is not supported by
                # DuckDB, so use a controlled SQL string after literal quoting.
                quoted = ", ".join("'" + path.replace("'", "''") + "'" for path in files)
                del parameters, values
                sources.setdefault(table, []).append(f"read_parquet([{quoted}])")
        else:
            for index, manifest in enumerate(self.manifests):
                path = manifest.shards.get("db")
                tables = manifest.shards.get("tables", ())
                if not path:
                    continue
                alias = f"httk_deferred_stage_{index}"
                self.connection.execute(sqlalchemy.text(f"ATTACH DATABASE :path AS {alias}").bindparams(path=str(path)))
                self.ingest._parallel_attached.append(alias)
                for table in tables:
                    sources.setdefault(table, []).append(f"{alias}.{self._q(table)}")
        for table, relations in sources.items():
            view = self._temp_name("stage", table)
            self._create_view(view, " UNION ALL ".join(f"SELECT * FROM {relation}" for relation in relations))
            self.stage_views[table] = view
        self.root_stage = self.stage_views.pop("_httk_roots", None)

    # ------------------------------------------------------------------ maps and conflict scan

    def _make_maps(self) -> None:
        for table, schema in self.parents.items():
            stage = self.stage_views.get(table)
            if stage is None:
                continue
            name = self._temp_name("map", table)
            if schema.dedup == "content_id":
                query = (
                    f"SELECT {self._q(SID_COLUMN)} AS stage_sid, "
                    f"MIN({self._q(SID_COLUMN)}) OVER (PARTITION BY {self._q(CONTENT_ID_COLUMN)}) AS canonical_sid "
                    f"FROM {self._q(stage)}"
                )
            else:
                query = (
                    f"SELECT {self._q(SID_COLUMN)} AS stage_sid, {self._q(SID_COLUMN)} AS canonical_sid "
                    f"FROM {self._q(stage)}"
                )
            self._create_table(name, query)
            self._index_relation(name, "stage_sid", unique=True)
            self._index_relation(name, "canonical_sid")
            self.maps[table] = name

    def _collapse_by_value_to_fixpoint(self) -> None:
        by_value = {name for name, schema in self.parents.items() if schema.dedup == "by_value" and name in self.maps}
        if not by_value:
            return
        # Every iteration computes keys through the current target maps.  A
        # whole pass with no changed map is the global graph fixpoint; SCC order
        # provides deterministic work ordering without assuming acyclicity.
        order = [name for name in self.graph.dependency_order(self.parents) if name in by_value]
        while True:
            changed = False
            for table in order:
                candidate = self._temp_name("candidate", table)
                self._drop(candidate)
                columns, joins = self._normalized_columns(table, "s")
                partition = ", ".join(columns)
                query = (
                    f"SELECT s.{self._q(SID_COLUMN)} AS stage_sid, "
                    f"MIN(s.{self._q(SID_COLUMN)}) OVER (PARTITION BY {partition}) AS canonical_sid "
                    f"FROM {self._q(self.stage_views[table])} AS s {' '.join(joins)}"
                )
                self._create_table(candidate, query)
                self._index_relation(candidate, "stage_sid", unique=True)
                self._index_relation(candidate, "canonical_sid")
                map_name = self.maps[table]
                comparator = "old_map.canonical_sid IS DISTINCT FROM candidate_map.canonical_sid"
                if self.connection.dialect.name == "clickhousedb":
                    from httk.data.db.clickhouse import null_safe_difference

                    comparator = null_safe_difference("old_map.canonical_sid", "candidate_map.canonical_sid")
                different = self.connection.execute(
                    sqlalchemy.text(
                        f"SELECT 1 FROM {self._q(map_name)} AS old_map JOIN {self._q(candidate)} AS candidate_map "
                        "ON old_map.stage_sid = candidate_map.stage_sid "
                        f"WHERE {comparator} LIMIT 1"
                    )
                ).first()
                if different is not None:
                    if self.store.backend_facts.finalize_map_maintenance == "swap":
                        from httk.data.db.clickhouse import swap_finalizer_map

                        swap_finalizer_map(self, table, candidate)
                    else:
                        self.connection.execute(
                            sqlalchemy.text(
                                f"UPDATE {self._q(map_name)} AS old_map SET canonical_sid = "
                                f"(SELECT candidate_map.canonical_sid FROM {self._q(candidate)} AS candidate_map "
                                "WHERE candidate_map.stage_sid = old_map.stage_sid)"
                            )
                        )
                        self._drop(candidate)
                    changed = True
                else:
                    self._drop(candidate)
            if not changed:
                return

    def _normalized_columns(self, table: str, alias: str) -> tuple[list[str], list[str]]:
        real = self.store._table(table)
        reference_columns = {column: target for column, target in self.graph.sid_columns().get(table, ())}
        columns: list[str] = []
        joins: list[str] = []
        for index, column in enumerate(real.columns):
            if column.name == SID_COLUMN:
                continue
            target = reference_columns.get(column.name)
            if target is None or target not in self.maps:
                columns.append(f"{alias}.{self._q(column.name)}")
                continue
            map_alias = f"m{index}"
            joins.append(
                f"LEFT JOIN {self._q(self.maps[target])} AS {map_alias} "
                f"ON {alias}.{self._q(column.name)} = {map_alias}.stage_sid"
            )
            columns.append(f"{map_alias}.canonical_sid")
        return columns, joins

    def _verify_metadata(self) -> None:
        if not self.ingest._verify_metadata:
            return
        from httk.data.db.bulk_parallel import _Merger

        nan_stage = self.stage_views.get("_httk_nan_content")
        nan = {} if nan_stage is not None else self._nan_by_content()
        for table, schema in self.parents.items():
            if schema.dedup != "content_id" or table not in self.maps:
                continue
            reported = nan.get(table, {})
            if nan_stage is not None:
                duplicate = self.connection.execute(
                    sqlalchemy.text(
                        f"SELECT s.{self._q(CONTENT_ID_COLUMN)} FROM {self._q(self.stage_views[table])} s "
                        f"JOIN {self._q(nan_stage)} n ON n.table_name = :table "
                        f"AND n.content_id = s.{self._q(CONTENT_ID_COLUMN)} "
                        f"GROUP BY s.{self._q(CONTENT_ID_COLUMN)} HAVING count(DISTINCT s.{self._q(SID_COLUMN)}) > 1 LIMIT 1"
                    ),
                    {"table": table},
                ).first()
                if duplicate is not None:
                    fields = set(
                        self.connection.execute(
                            sqlalchemy.text(
                                f"SELECT DISTINCT field_name FROM {self._q(nan_stage)} "
                                "WHERE table_name = :table AND content_id = :content_id"
                            ),
                            {"table": table, "content_id": duplicate[0]},
                        ).scalars()
                    )
                    compare = _Merger._metadata_compare_columns(schema)
                    field = next((field for _column, field in compare if field in fields), schema.cls.__name__)
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {schema.cls.__name__}.{field}: content id {duplicate[0]!r} occurs with "
                        "a NaN identity-excluded value that never equals itself"
                    )
            elif reported:
                duplicate = self.connection.execute(
                    sqlalchemy.text(
                        f"SELECT {self._q(CONTENT_ID_COLUMN)} FROM {self._q(self.stage_views[table])} "
                        f"WHERE {self._q(CONTENT_ID_COLUMN)} IN ({', '.join(':k' + str(i) for i in range(len(reported)))}) "
                        f"GROUP BY {self._q(CONTENT_ID_COLUMN)} HAVING count(*) > 1 LIMIT 1"
                    ).bindparams(**{f"k{i}": key for i, key in enumerate(sorted(reported))})
                ).first()
                if duplicate is not None:
                    fields = reported[str(duplicate[0])]
                    compare = _Merger._metadata_compare_columns(schema)
                    field = next((field for _column, field in compare if field in fields), schema.cls.__name__)
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {schema.cls.__name__}.{field}: content id {duplicate[0]!r} occurs with "
                        "a NaN identity-excluded value that never equals itself"
                    )
            compare = _Merger._metadata_compare_columns(schema)
            if not compare:
                continue
            expressions, joins = self._metadata_expressions(table, compare, "s")
            distinct = self._temp_name("metadata", table)
            self._create_view(
                distinct,
                f"SELECT DISTINCT m.canonical_sid AS identity, {', '.join(expressions)} "
                f"FROM {self._q(self.stage_views[table])} AS s "
                f"JOIN {self._q(self.maps[table])} AS m ON s.{self._q(SID_COLUMN)} = m.stage_sid {' '.join(joins)}",
            )
            conflict = self.connection.execute(
                sqlalchemy.text(
                    f"SELECT identity FROM {self._q(distinct)} GROUP BY identity HAVING count(*) > 1 LIMIT 1"
                )
            ).first()
            if conflict is not None:
                field = compare[0][1]
                key = self.connection.execute(
                    sqlalchemy.text(
                        f"SELECT s.{self._q(CONTENT_ID_COLUMN)} FROM {self._q(self.stage_views[table])} s "
                        f"JOIN {self._q(self.maps[table])} m ON s.{self._q(SID_COLUMN)} = m.stage_sid "
                        "WHERE m.canonical_sid = :sid LIMIT 1"
                    ),
                    {"sid": conflict[0]},
                ).scalar_one()
                raise EntryMetadataConflictError(
                    f"metadata conflict for {schema.cls.__name__}.{field}: content id {key!r} occurs with "
                    "differing identity-excluded metadata"
                )

    def _metadata_expressions(
        self, table: str, compare: Iterable[tuple[str, str]], alias: str
    ) -> tuple[list[str], list[str]]:
        references = {column: target for column, target in self.graph.sid_columns().get(table, ())}
        expressions: list[str] = []
        joins: list[str] = []
        for index, (column, _field) in enumerate(compare):
            target = references.get(column)
            if target is None or target not in self.maps:
                expressions.append(f"{alias}.{self._q(column)} AS c{index}")
                continue
            map_alias = f"metadata_map_{index}"
            joins.append(
                f"LEFT JOIN {self._q(self.maps[target])} AS {map_alias} "
                f"ON {alias}.{self._q(column)} = {map_alias}.stage_sid"
            )
            expressions.append(f"{map_alias}.canonical_sid AS c{index}")
        return expressions, joins

    def _nan_by_content(self) -> dict[str, dict[str, set[str]]]:
        result: dict[str, dict[str, set[str]]] = {}
        for manifest in self.manifests:
            for table, key, field in manifest.nan_content:
                result.setdefault(table, {}).setdefault(key, set()).add(field)
        return result

    # ------------------------------------------------------------------ reachability and final sid maps

    def _make_survivors(self) -> None:
        self.survivors: dict[str, str] = {}
        roots = self._temp_name("roots", "rows")
        root_empty = "SELECT CAST(NULL AS TEXT) AS tbl, CAST(NULL AS INTEGER) AS stage_sid WHERE 0"
        if self.connection.dialect.name == "clickhousedb":
            root_empty = "SELECT CAST('' AS String) AS tbl, toInt64(0) AS stage_sid WHERE 0"
        self._create_table(roots, root_empty)
        self.root_occurrences = roots
        if self.root_stage is not None:
            self.connection.execute(
                sqlalchemy.text(
                    f"INSERT INTO {self._q(roots)} (tbl, stage_sid) "
                    f"SELECT tbl, stage_sid FROM {self._q(self.root_stage)}"
                )
            )
        else:
            root_rows = [
                {"tbl": table, "stage_sid": sid}
                for manifest in self.manifests
                for table, sid in manifest.token_sid.values()
                if table in self.maps
            ]
            if root_rows:
                self.connection.execute(
                    sqlalchemy.text(f"INSERT INTO {self._q(roots)} (tbl, stage_sid) VALUES (:tbl, :stage_sid)"),
                    root_rows,
                )
        self._index_relation(roots, "tbl", "stage_sid")
        for component in self.graph.reachability_scc_order():
            cyclic = len(component) > 1 or any(
                edge.source_table == edge.target_table and edge.source_table in component for edge in self.graph.edges
            )
            if cyclic:
                self._build_cyclic_survivors(component, roots)
                continue
            for table in component:
                if table in self.maps:
                    self._build_table_survivors(table, roots)

    def _build_table_survivors(self, table: str, roots: str) -> None:
        terms = [
            (
                f"SELECT m.canonical_sid FROM {self._q(self.maps[table])} m JOIN {self._q(roots)} r "
                f"ON r.tbl = '{table}' AND r.stage_sid = m.stage_sid"
            )
        ]
        for edge in self.graph.edges:
            if edge.target_table != table:
                continue
            if edge.kind == "reference" and edge.source_table in self.survivors:
                assert edge.source_column is not None
                terms.append(
                    f"SELECT tm.canonical_sid FROM {self._q(self.survivors[edge.source_table])} ss "
                    f"JOIN {self._q(self.stage_views[edge.source_table])} s ON 1 = 1 "
                    f"JOIN {self._q(self.maps[edge.source_table])} sm ON s.{self._q(SID_COLUMN)} = sm.stage_sid "
                    f"AND sm.canonical_sid = ss.canonical_sid AND sm.stage_sid = sm.canonical_sid "
                    f"JOIN {self._q(self.maps[table])} tm ON s.{self._q(edge.source_column)} = tm.stage_sid"
                )
            elif edge.kind == "child_element":
                assert edge.source_column is not None
                ownership = next(
                    (value for value in self.graph.ownership() if value.target_table == edge.source_table), None
                )
                if ownership is None or ownership.target_column is None or ownership.source_table not in self.survivors:
                    continue
                parent = ownership.source_table
                terms.append(
                    f"SELECT tm.canonical_sid FROM {self._q(self.survivors[parent])} ps "
                    f"JOIN {self._q(self.stage_views[edge.source_table])} c ON 1 = 1 "
                    f"JOIN {self._q(self.maps[parent])} pm ON c.{self._q(ownership.target_column)} = pm.stage_sid "
                    "AND pm.canonical_sid = ps.canonical_sid AND pm.stage_sid = pm.canonical_sid "
                    f"JOIN {self._q(self.maps[table])} tm ON c.{self._q(edge.source_column)} = tm.stage_sid"
                )
        name = self._temp_name("survivors", table)
        self._create_table(name, f"SELECT DISTINCT canonical_sid FROM ({' UNION ALL '.join(terms)})")
        self._index_relation(name, "canonical_sid", unique=True)
        self.survivors[table] = name

    def _build_cyclic_survivors(self, component: tuple[str, ...], roots: str) -> None:
        """Scoped semi-naive fallback for a genuinely cyclic reachability SCC."""
        for table in component:
            if table in self.maps:
                self._build_table_survivors(table, roots)
        edges = [edge for edge in self.graph.edges if edge.source_table in component and edge.target_table in component]
        while True:
            before = sum(self._count(self.survivors[table]) for table in component if table in self.survivors)
            for edge in edges:
                if edge.kind == "reference":
                    if edge.source_table not in self.survivors or edge.target_table not in self.survivors:
                        continue
                    assert edge.source_column is not None
                    target = self.survivors[edge.target_table]
                    query = (
                        f"SELECT DISTINCT tm.canonical_sid FROM {self._q(self.survivors[edge.source_table])} ss "
                        f"JOIN {self._q(self.stage_views[edge.source_table])} s ON 1 = 1 "
                        f"JOIN {self._q(self.maps[edge.source_table])} sm ON s.{self._q(SID_COLUMN)} = sm.stage_sid "
                        "AND sm.canonical_sid = ss.canonical_sid AND sm.stage_sid = sm.canonical_sid "
                        f"JOIN {self._q(self.maps[edge.target_table])} tm ON s.{self._q(edge.source_column)} = tm.stage_sid"
                    )
                elif edge.kind == "child_element":
                    assert edge.source_column is not None
                    ownership = next(
                        (value for value in self.graph.ownership() if value.target_table == edge.source_table), None
                    )
                    if (
                        ownership is None
                        or ownership.target_column is None
                        or ownership.source_table not in self.survivors
                        or edge.target_table not in self.survivors
                    ):
                        continue
                    target = self.survivors[edge.target_table]
                    query = (
                        f"SELECT DISTINCT tm.canonical_sid FROM {self._q(self.survivors[ownership.source_table])} ps "
                        f"JOIN {self._q(self.stage_views[edge.source_table])} c ON 1 = 1 "
                        f"JOIN {self._q(self.maps[ownership.source_table])} pm "
                        f"ON c.{self._q(ownership.target_column)} = pm.stage_sid "
                        "AND pm.canonical_sid = ps.canonical_sid AND pm.stage_sid = pm.canonical_sid "
                        f"JOIN {self._q(self.maps[edge.target_table])} tm ON c.{self._q(edge.source_column)} = tm.stage_sid"
                    )
                else:
                    continue
                self.connection.execute(
                    sqlalchemy.text(
                        f"INSERT INTO {self._q(target)} {query} WHERE NOT EXISTS "
                        f"(SELECT 1 FROM {self._q(target)} existing WHERE existing.canonical_sid = tm.canonical_sid)"
                    )
                )
            after = sum(self._count(self.survivors[table]) for table in component if table in self.survivors)
            if after == before:
                return

    def _make_final_sids(self) -> None:
        for table, map_name in self.maps.items():
            final = self._temp_name("final", table)
            query = (
                "SELECT canonical_sid, ROW_NUMBER() OVER (ORDER BY canonical_sid) AS final_sid FROM "
                f"(SELECT DISTINCT m.canonical_sid FROM {self._q(map_name)} m JOIN {self._q(self.survivors[table])} r "
                "ON r.canonical_sid = m.canonical_sid)"
            )
            self._create_table(final, query)
            self._index_relation(final, "canonical_sid", unique=True)
            self.finals[table] = final

    # ------------------------------------------------------------------ one-shot projection and dispatch

    def _load_real_tables(self) -> None:
        for table in self._projection_order():
            real = self.store._table(table)
            if SID_COLUMN in real.c:
                self._insert_parent(table)
            else:
                self._insert_child(table)

    def _projection_order(self) -> list[str]:
        names = set(self.stage_views) & set(self.store._metadata.tables)
        # Insert order is physically unconstrained; a stable order is useful
        # for reproducible diagnostics.
        return sorted(names)

    def _insert_parent(self, table: str) -> None:
        real = self.store._table(table)
        expressions, joins = self._projection_columns(table, "s", "m", "f")
        if "_httk_role" in real.c:
            role = "role_occurrences"
            assert self.root_occurrences is not None
            table_literal = table.replace("'", "''")
            joins.append(
                f"LEFT JOIN (SELECT m2.canonical_sid, MAX(1) AS role "
                f"FROM {self._q(self.root_occurrences)} r2 JOIN {self._q(self.maps[table])} m2 "
                f"ON r2.stage_sid = m2.stage_sid WHERE r2.tbl = '{table_literal}' "
                f"GROUP BY m2.canonical_sid) {role} ON {role}.canonical_sid = m.canonical_sid"
            )
            role_index = [column.name for column in real.columns].index("_httk_role")
            expressions[role_index] = f"COALESCE({role}.role, 0)"
        columns = ", ".join(self._q(column.name) for column in real.columns)
        statement = (
            f"INSERT INTO {self._q(table)} ({columns}) SELECT {', '.join(expressions)} "
            f"FROM {self._q(self.stage_views[table])} s "
            f"JOIN {self._q(self.maps[table])} m ON s.{self._q(SID_COLUMN)} = m.stage_sid "
            f"JOIN {self._q(self.finals[table])} f ON m.canonical_sid = f.canonical_sid {' '.join(joins)} "
            "WHERE m.stage_sid = m.canonical_sid"
        )
        expected = self._count(self.finals[table])
        self.connection.execute(sqlalchemy.text(statement))
        self._assert_loaded_count(table, expected)
        self.ingest._inserted_count[table] = expected
        self.ingest._next_sid[table] = expected + 1

    def _insert_child(self, table: str) -> None:
        ownership = next((edge for edge in self.graph.ownership() if edge.target_table == table), None)
        if ownership is None or ownership.target_column is None or ownership.source_table not in self.maps:
            return
        parent = ownership.source_table
        real = self.store._table(table)
        expressions, joins = self._projection_columns(table, "s", "pm", "pf")
        columns = ", ".join(self._q(column.name) for column in real.columns)
        statement = (
            f"INSERT INTO {self._q(table)} ({columns}) SELECT {', '.join(expressions)} "
            f"FROM {self._q(self.stage_views[table])} s "
            f"JOIN {self._q(self.maps[parent])} pm ON s.{self._q(ownership.target_column)} = pm.stage_sid "
            f"JOIN {self._q(self.finals[parent])} pf ON pm.canonical_sid = pf.canonical_sid {' '.join(joins)} "
            "WHERE pm.stage_sid = pm.canonical_sid"
        )
        expected = int(
            self.connection.execute(
                sqlalchemy.text(
                    f"SELECT count(*) FROM {self._q(self.stage_views[table])} s "
                    f"JOIN {self._q(self.maps[parent])} pm ON s.{self._q(ownership.target_column)} = pm.stage_sid "
                    "WHERE pm.stage_sid = pm.canonical_sid"
                )
            ).scalar_one()
        )
        self.connection.execute(sqlalchemy.text(statement))
        self._assert_loaded_count(table, expected)
        self.ingest._inserted_count[table] = expected

    def _assert_loaded_count(self, table: str, expected: int) -> None:
        actual = self._count(table)
        if actual != expected:
            raise RuntimeError(
                f"deferred finalize projection count failed for {table!r}: expected {expected}, stored {actual}"
            )

    def _projection_columns(self, table: str, source: str, own_map: str, own_final: str) -> tuple[list[str], list[str]]:
        real = self.store._table(table)
        refs = {column: target for column, target in self.graph.sid_columns().get(table, ())}
        expressions: list[str] = []
        joins: list[str] = []
        for index, column in enumerate(real.columns):
            if column.name == SID_COLUMN:
                expressions.append(f"{own_final}.final_sid")
                continue
            target = refs.get(column.name)
            if target is None or target not in self.maps:
                expressions.append(f"{source}.{self._q(column.name)}")
                continue
            if target == table and column.name == SID_COLUMN:
                expressions.append(f"{own_final}.final_sid")
                continue
            map_alias = f"rm{index}"
            final_alias = f"rf{index}"
            joins.append(
                f"LEFT JOIN {self._q(self.maps[target])} {map_alias} "
                f"ON {source}.{self._q(column.name)} = {map_alias}.stage_sid "
                f"LEFT JOIN {self._q(self.finals[target])} {final_alias} ON {map_alias}.canonical_sid = {final_alias}.canonical_sid"
            )
            expressions.append(f"{final_alias}.final_sid")
        return expressions, joins

    def _rebuild_dispatch(self) -> None:
        from httk.data.store_common import EntryDispatchIntegrityError

        payload_stage = self.stage_views.get("_httk_dispatch_payload")
        grouped: dict[str, list[Any]] = {}
        if payload_stage is None:
            # SQLite/DuckDB-native serial stages retain this list because the
            # serial duplicate-return contract needs their in-memory indexes.
            for manifest in self.manifests:
                for row in manifest.dispatch:
                    grouped.setdefault(row.dispatch_name, []).append(row)
            dispatches = ((name, rows) for name, rows in grouped.items())
        else:
            dispatches = (
                (self._dispatch_name(family), None)
                for family in self.store.layout.families
                if self.connection.execute(
                    sqlalchemy.text(f"SELECT 1 FROM {self._q(payload_stage)} WHERE dispatch_name = :name LIMIT 1"),
                    {"name": self._dispatch_name(family)},
                ).first()
                is not None
            )
        for dispatch_name, rows in dispatches:
            real = self.store._table(dispatch_name)
            stage = self._temp_name("dispatch", dispatch_name)
            columns = ", ".join(self._q(column.name) for column in real.columns)
            self._create_table(stage, f"SELECT {columns} FROM {self._q(dispatch_name)} WHERE 0")
            if payload_stage is None:
                payload: list[dict[str, Any]] = []
                assert rows is not None
                for row in rows:
                    built: dict[str, Any] = {"content_id": row.key}
                    for column in row.all_columns:
                        built[column] = None
                    built[row.column] = row.block_sid
                    payload.append(built)
                self.connection.execute(
                    sqlalchemy.insert(
                        sqlalchemy.Table(
                            stage,
                            sqlalchemy.MetaData(),
                            *[sqlalchemy.Column(column.name, column.type) for column in real.columns],
                        )
                    ),
                    payload,
                )
            else:
                payload_columns = []
                for column in real.columns:
                    if column.name == CONTENT_ID_COLUMN:
                        payload_columns.append(f"p.content_id AS {self._q(column.name)}")
                    elif column.name.endswith("_sid"):
                        payload_columns.append(
                            f"CASE WHEN p.column = '{column.name}' THEN p.block_sid ELSE NULL END AS {self._q(column.name)}"
                        )
                    else:
                        raise RuntimeError(f"deferred dispatch {dispatch_name!r} has unexpected column {column.name!r}")
                self.connection.execute(
                    sqlalchemy.text(
                        f"INSERT INTO {self._q(stage)} ({columns}) SELECT {', '.join(payload_columns)} "
                        f"FROM {self._q(payload_stage)} p WHERE p.dispatch_name = :name"
                    ),
                    {"name": dispatch_name},
                )
            projected, joins = self._dispatch_projection(dispatch_name, stage)
            distinct = self._temp_name("dispatch_rows", dispatch_name)
            self._create_view(
                distinct,
                f"SELECT DISTINCT {', '.join(projected)} FROM {self._q(stage)} d {' '.join(joins)}",
            )
            conflict = self.connection.execute(
                sqlalchemy.text(
                    f"SELECT content_id FROM {self._q(distinct)} GROUP BY content_id HAVING count(*) > 1 LIMIT 1"
                )
            ).first()
            if conflict is not None:
                family = next(
                    family for family in self.store.layout.families if self._dispatch_name(family) == dispatch_name
                )
                raise EntryDispatchIntegrityError(
                    f"entry dispatch {family.name!r} maps content_id {conflict[0]!r} to a conflicting backing row"
                )
            expected = self._count(distinct)
            self.connection.execute(
                sqlalchemy.text(
                    f"INSERT INTO {self._q(dispatch_name)} ({columns}) SELECT {columns} FROM {self._q(distinct)}"
                )
            )
            self._assert_loaded_count(dispatch_name, expected)
            self.ingest._inserted_count[dispatch_name] = expected

    def _dispatch_name(self, family: Any) -> str:
        from httk.data.db.mapping import entry_dispatch_table_name

        return entry_dispatch_table_name(family.name)

    def _dispatch_projection(self, dispatch: str, stage: str) -> tuple[list[str], list[str]]:
        edges = {edge.source_column: edge.target_table for edge in self.graph.edges if edge.source_table == dispatch}
        expressions: list[str] = []
        joins: list[str] = []
        for index, column in enumerate(self.store._table(dispatch).columns):
            target = edges.get(column.name)
            if target is None:
                expressions.append(f"d.{self._q(column.name)}")
                continue
            if target not in self.maps or target not in self.finals:
                unexpected = self.connection.execute(
                    sqlalchemy.text(f"SELECT 1 FROM {self._q(stage)} WHERE {self._q(column.name)} IS NOT NULL LIMIT 1")
                ).first()
                if unexpected is not None:
                    raise RuntimeError(f"deferred dispatch {dispatch!r} refers to unstaged backing table {target!r}")
                expressions.append(f"NULL AS {self._q(column.name)}")
                continue
            map_alias, final_alias = f"dm{index}", f"df{index}"
            joins.append(
                f"LEFT JOIN {self._q(self.maps[target])} {map_alias} ON d.{self._q(column.name)} = {map_alias}.stage_sid "
                f"LEFT JOIN {self._q(self.finals[target])} {final_alias} ON {map_alias}.canonical_sid = {final_alias}.canonical_sid"
            )
            expressions.append(f"{final_alias}.final_sid AS {self._q(column.name)}")
        return expressions, joins

    def _populate_returned_sids(self) -> None:
        if not self.ingest._track_sids:
            return
        for table, map_name in self.maps.items():
            rows = self.connection.execute(
                sqlalchemy.text(
                    f"SELECT m.stage_sid, f.final_sid FROM {self._q(map_name)} m "
                    f"JOIN {self._q(self.finals[table])} f ON m.canonical_sid = f.canonical_sid"
                )
            ).all()
            self._final_by_stage[table] = {int(stage): int(final) for stage, final in rows}
        for manifest in self.manifests:
            for token, (table, stage_sid) in manifest.token_sid.items():
                try:
                    self.ingest._resolved_map[(table, token)] = self._final_by_stage[table][stage_sid]
                except KeyError:
                    raise RuntimeError(f"deferred finalize lost staged {table!r} sid {stage_sid}") from None

    def _final_for_stage(self, table: str, stage_sid: int) -> int:
        try:
            return self._final_by_stage[table][stage_sid]
        except KeyError:
            raise RuntimeError(f"deferred finalize lost staged {table!r} sid {stage_sid}") from None

    def _drop(self, name: str) -> None:
        with contextlib.suppress(Exception):
            self.connection.execute(sqlalchemy.text(f'DROP VIEW IF EXISTS {self._q(name)}'))
        with contextlib.suppress(Exception):
            self.connection.execute(sqlalchemy.text(f'DROP TABLE IF EXISTS {self._q(name)}'))
        with contextlib.suppress(ValueError):
            self.objects.remove(name)

    def _count(self, relation: str) -> int:
        return int(self.connection.execute(sqlalchemy.text(f"SELECT count(*) FROM {self._q(relation)}")).scalar_one())
