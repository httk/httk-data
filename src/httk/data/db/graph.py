"""Logical storage edges derived from the resolved schema declarations.

The SQL schema deliberately has more than one relationship shape.  Keeping
those relationships here makes the bulk algorithms independent of whether a
backend happens to expose physical foreign-key constraints.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Literal

from httk.data.db.mapping import backing_dispatch_column_name, entry_dispatch_table_name
from httk.data.db.schema import TableSchema, resolve_schema

__all__ = ["EdgeKind", "LogicalEdge", "LogicalEdgeGraph"]

EdgeKind = Literal["reference", "ownership", "child_element", "dispatch"]


@dataclasses.dataclass(frozen=True)
class LogicalEdge:
    """One typed relationship in the logical storage graph.

    ``source_table`` and ``target_table`` describe the traversal direction.
    ``source_column`` is the forward sid column for reference, child-element,
    and dispatch edges.  Ownership is traversed from parent to child, so its
    parent sid column is carried as ``target_column`` on the child table.
    """

    kind: EdgeKind
    source_table: str
    target_table: str
    source_column: str | None = None
    target_column: str | None = None


@dataclasses.dataclass(frozen=True)
class LogicalEdgeGraph:
    """Deterministic typed edges for a set of resolved table declarations."""

    edges: tuple[LogicalEdge, ...]
    tables: tuple[str, ...]

    @classmethod
    def from_schemas(
        cls,
        schemas: Iterable[TableSchema],
        dispatches: Iterable[tuple[str, Sequence[tuple[str, TableSchema]]]] = (),
    ) -> LogicalEdgeGraph:
        """Build edges from parent schemas, recursively including their targets."""
        by_table: dict[str, TableSchema] = {}
        pending = list(schemas)
        while pending:
            schema = pending.pop()
            if schema.table_name in by_table:
                continue
            by_table[schema.table_name] = schema
            for target in schema.referenced_classes():
                pending.append(resolve_schema(target))

        edges: set[LogicalEdge] = set()
        tables: set[str] = set(by_table)
        for schema in by_table.values():
            parent = schema.table_name
            for spec in schema.fields:
                if spec.role == "reference":
                    assert spec.target is not None and spec.columns
                    target = resolve_schema(spec.target).table_name
                    edges.add(LogicalEdge("reference", parent, target, spec.columns[0].name))
                if spec.child is None:
                    continue
                child = spec.child.table_name
                parent_column = f"{parent}_sid"
                tables.add(child)
                edges.add(LogicalEdge("ownership", parent, child, target_column=parent_column))
                if spec.target is not None:
                    target = resolve_schema(spec.target).table_name
                    element_column = spec.child.element_columns[0].name
                    edges.add(LogicalEdge("child_element", child, target, element_column))

        for dispatch_name, backings in dispatches:
            tables.add(dispatch_name)
            for backing_name, schema in backings:
                target = schema.table_name
                tables.add(target)
                edges.add(
                    LogicalEdge(
                        "dispatch",
                        dispatch_name,
                        target,
                        backing_dispatch_column_name(backing_name),
                    )
                )
        return cls(tuple(sorted(edges, key=_edge_key)), tuple(sorted(tables)))

    @classmethod
    def from_store(cls, store: object, schemas: Iterable[TableSchema]) -> LogicalEdgeGraph:
        """Build a graph using the store's configured entry-family dispatches."""
        dispatches: list[tuple[str, Sequence[tuple[str, TableSchema]]]] = []
        for family in store.entry_layout:  # type: ignore[attr-defined]
            if len(family.records) < 2:
                continue
            dispatches.append(
                (
                    entry_dispatch_table_name(family.name),
                    tuple((name, resolve_schema(record)) for name, record in zip(family.record_names, family.records)),
                )
            )
        return cls.from_schemas(schemas, dispatches)

    def sid_columns(self) -> dict[str, list[tuple[str, str]]]:
        """Return ``table -> (sid_column, referenced_table)`` mappings.

        This is the compatibility view used by remapping and reachability
        code; it includes the parent column of ownership edges as well as the
        forward columns of the other edge kinds.
        """
        result: dict[str, list[tuple[str, str]]] = {}
        for edge in self.edges:
            if edge.kind == "ownership":
                assert edge.target_column is not None
                table, column, target = edge.target_table, edge.target_column, edge.source_table
            else:
                assert edge.source_column is not None
                table, column, target = edge.source_table, edge.source_column, edge.target_table
            result.setdefault(table, []).append((column, target))
        return {name: sorted(columns) for name, columns in sorted(result.items())}

    def referrers(self, target_table: str) -> tuple[tuple[str, str], ...]:
        """Return all ``(referrer_table, sid_column)`` pairs for a target."""
        return tuple(
            sorted(
                (table, column)
                for table, columns in self.sid_columns().items()
                for column, target in columns
                if target == target_table
            )
        )

    def ownership(self) -> tuple[LogicalEdge, ...]:
        """Return parent-to-child ownership edges in deterministic order."""
        return tuple(edge for edge in self.edges if edge.kind == "ownership")

    def dependency_order(self, table_names: Iterable[str] | None = None) -> tuple[str, ...]:
        """Return a deterministic dependency order, including cyclic schemas.

        Reference and child-element dependencies are condensed into strongly
        connected components.  The physical load order places referenced rows
        before referrers, parents before owned child rows, and backing rows
        before dispatch rows.  Both component selection and members use the
        lexicographically smallest available table name as their tiebreaker.
        """
        nodes = set(self.tables if table_names is None else table_names)
        allowed = nodes if table_names is not None else None
        for edge in self.edges:
            if allowed is None or {edge.source_table, edge.target_table} <= allowed:
                nodes.update((edge.source_table, edge.target_table))
        constraints: dict[str, set[str]] = {node: set() for node in nodes}
        for edge in self.edges:
            if allowed is not None and {edge.source_table, edge.target_table} - allowed:
                continue
            if edge.kind in ("reference", "child_element", "dispatch"):
                # target must be available before the table that points to it
                constraints.setdefault(edge.target_table, set()).add(edge.source_table)
            elif edge.kind == "ownership":
                constraints.setdefault(edge.source_table, set()).add(edge.target_table)

        components = _strongly_connected_components(nodes, constraints)
        component_of = {node: index for index, component in enumerate(components) for node in component}
        outgoing: dict[int, set[int]] = {index: set() for index in range(len(components))}
        indegree = {index: 0 for index in range(len(components))}
        for source, targets in constraints.items():
            for target in targets:
                left, right = component_of[source], component_of[target]
                if left != right and right not in outgoing[left]:
                    outgoing[left].add(right)
                    indegree[right] += 1
        ready = {index for index, degree in indegree.items() if degree == 0}
        result: list[str] = []
        while ready:
            component = min(ready, key=lambda index: min(components[index]))
            ready.remove(component)
            result.extend(sorted(components[component]))
            for target in outgoing[component]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.add(target)
        assert len(result) == len(nodes)
        return tuple(result)

    def reachability_scc_order(self) -> tuple[tuple[str, ...], ...]:
        """Return SCCs in forward logical-reachability order.

        Unlike :meth:`dependency_order`, this follows a reference from its
        source to its target and ownership from parent to child.  Consumers
        propagating root reachability can therefore complete acyclic SCCs in
        one wave, reserving iteration for genuine cycles.
        """
        nodes = set(self.tables)
        forward: dict[str, set[str]] = {node: set() for node in nodes}
        for edge in self.edges:
            forward.setdefault(edge.source_table, set()).add(edge.target_table)
            nodes.update((edge.source_table, edge.target_table))
        components = _strongly_connected_components(nodes, forward)
        component_of = {node: index for index, component in enumerate(components) for node in component}
        outgoing: dict[int, set[int]] = {index: set() for index in range(len(components))}
        indegree = {index: 0 for index in range(len(components))}
        for source, targets in forward.items():
            for target in targets:
                left, right = component_of[source], component_of[target]
                if left != right and right not in outgoing[left]:
                    outgoing[left].add(right)
                    indegree[right] += 1
        ready = {index for index, degree in indegree.items() if degree == 0}
        ordered: list[tuple[str, ...]] = []
        while ready:
            index = min(ready, key=lambda value: min(components[value]))
            ready.remove(index)
            ordered.append(components[index])
            for target in outgoing[index]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.add(target)
        assert len(ordered) == len(components)
        return tuple(ordered)


def _edge_key(edge: LogicalEdge) -> tuple[str, str, str, str, str]:
    return (edge.kind, edge.source_table, edge.target_table, edge.source_column or "", edge.target_column or "")


def _strongly_connected_components(nodes: set[str], constraints: dict[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    """Tarjan SCCs for the dependency constraints."""
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    found: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for target in sorted(constraints.get(node, ())):
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], indices[target])
        if low[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            target = stack.pop()
            active.remove(target)
            component.append(target)
            if target == node:
                break
        found.append(tuple(sorted(component)))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return tuple(found)
