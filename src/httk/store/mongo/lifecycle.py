"""Lifecycle (time-slice) predicate construction for versioned MongoStores.

Every query path entered from the top must see one consistent time slice: the
current view by default, or the half-open ``as_of`` slice ``[ts_start, ts_end)``.
A versioned family collection carries its own lifetime fields, so its predicate
is a direct interval test. A non-family (deduplicated dependency/content)
collection has none, so its visibility is derived: it is visible when it was
saved standalone (``_httk_role == "main"``) or when some current family document
still owns it through a statically computed ownership chain.

The SQL backend expresses that ``EXISTS`` with correlated subqueries; here the
same shape is built as chained ``$lookup`` stages in the aggregation pipeline.
The ownership-chain analysis itself (:func:`~httk.store.db.graph.lifecycle_paths`)
is backend-neutral and reused directly. MongoDB stores child elements as
embedded arrays rather than separate tables, so a child-element hop collapses to
a single dotted array path (``f.<field>.<element_key>``) instead of a join
through a child collection.
"""

from typing import TYPE_CHECKING, Any

from httk.store.db.graph import LifecycleHop, LifecyclePath
from httk.store.db.schema import TableSchema, resolve_schema
from httk.store.store_common import LifecycleScopeError

if TYPE_CHECKING:
    from httk.store.mongo.store import MongoStore

__all__ = ["LifecycleScopeError", "family_interval_match", "reachability_stages"]


def family_interval_match(as_of_units: int | None) -> dict[str, Any]:
    """The plain ``$match`` predicate for a versioned family document being live.

    Operates on top-level ``ts_start``/``ts_end`` document fields, so it is used
    inside a ``$lookup`` sub-pipeline against the referrer collection.

    :param as_of_units: The historic cutoff in integer store units, or ``None`` for current view.
    :return: A MongoDB ``$match`` filter selecting the current (or ``as_of``) document.
    """
    if as_of_units is None:
        return {"ts_end": None}
    return {
        "$and": [
            {"ts_start": {"$lte": as_of_units}},
            {"$or": [{"ts_end": None}, {"ts_end": {"$gt": as_of_units}}]},
        ]
    }


def _collection_schemas(store: "MongoStore") -> dict[str, TableSchema]:
    """The closure of schemas reachable from the store's known records, by collection."""
    result: dict[str, TableSchema] = {}
    pending = [
        *store._known_record_types,
        *(record for family in store.layout.families for record in family.records),
    ]
    seen: set[type] = set()
    while pending:
        record = pending.pop()
        if record in seen:
            continue
        seen.add(record)
        schema = resolve_schema(record)
        result[schema.table_name] = schema
        pending.extend(schema.referenced_classes())
    return result


def _mongo_hop(schemas: dict[str, TableSchema], hop: LifecycleHop) -> tuple[str, str, bool]:
    """Translate one ownership hop into ``(referrer_collection, key_path, is_array)``.

    A plain reference hop yields the scalar foreign-key path ``f.<column>``. A
    child-element hop yields the embedded-array path ``f.<field>.<element_key>``,
    which MongoDB matches element-wise.
    """
    if hop.child_table is not None:
        assert hop.child_element_column is not None
        parent_schema = schemas[hop.referrer_table]
        field = next(
            spec.field
            for spec in parent_schema.fields
            if spec.child is not None and spec.child.table_name == hop.child_table
        )
        return hop.referrer_table, f"f.{field}.{hop.child_element_column}", True
    assert hop.referrer_column is not None
    return hop.referrer_table, f"f.{hop.referrer_column}", False


def _path_lookup(
    schemas: dict[str, TableSchema],
    path: LifecyclePath,
    as_name: str,
    local_id_expr: str,
    as_of_units: int | None,
) -> dict[str, Any]:
    """Build one chained ``$lookup`` that is non-empty iff a live owner reaches this path."""
    hops = [_mongo_hop(schemas, hop) for hop in path.hops]

    def sub(index: int, let_expr: str, output_name: str) -> dict[str, Any]:
        collection, key_path, is_array = hops[index]
        variable = f"p{index}"
        if is_array:
            match_expr: dict[str, Any] = {"$in": [f"$${variable}", {"$ifNull": [f"${key_path}", []]}]}
        else:
            match_expr = {"$eq": [f"${key_path}", f"$${variable}"]}
        pipeline: list[dict[str, Any]] = [{"$match": {"$expr": match_expr}}]
        if index == len(hops) - 1:
            pipeline.append({"$match": family_interval_match(as_of_units)})
        else:
            nested_name = f"_httk_reach_next_{index + 1}"
            pipeline.append(sub(index + 1, "$_id", nested_name))
            pipeline.append({"$match": {nested_name: {"$ne": []}}})
        pipeline.append({"$limit": 1})
        return {"$lookup": {"from": collection, "let": {variable: let_expr}, "pipeline": pipeline, "as": output_name}}

    return sub(0, local_id_expr, as_name)


def reachability_stages(
    store: "MongoStore",
    table_name: str,
    *,
    local_id_expr: str,
    role_path: str,
    ts_start_path: str,
    as_of_units: int | None,
    reach_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the non-family ``role OR EXISTS(ownership chain)`` lifecycle stages.

    :param store: The versioned MongoStore owning the collections.
    :param table_name: The non-family collection the scope binds to.
    :param local_id_expr: The aggregation expression yielding this scope's sid (e.g. ``"$_id"``).
    :param role_path: The document path of this scope's role marker.
    :param ts_start_path: The document path of this scope's ``ts_start`` field.
    :param as_of_units: The historic cutoff in store units, or ``None`` for current view.
    :param reach_prefix: A unique prefix naming this scope's reachability lookup outputs.
    :return: The ``$lookup`` stages to prepend and the combined ``$match`` predicate.
    :raises LifecycleScopeError: If ``table_name`` is caught in a non-family reference cycle.
    """
    path_map = store._lifecycle_path_map()
    if table_name in path_map and path_map[table_name] is None:
        raise LifecycleScopeError(
            f"the non-family collection {table_name!r} is caught in a reference cycle and cannot be scoped "
            "to a time slice; query with scoped=False to disable lifecycle filtering"
        )
    schemas = _collection_schemas(store)
    paths: tuple[LifecyclePath, ...] = path_map.get(table_name) or ()
    stages: list[dict[str, Any]] = []
    reach_terms: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        as_name = f"{reach_prefix}_{index}"
        stages.append(_path_lookup(schemas, path, as_name, local_id_expr, as_of_units))
        reach_terms.append({as_name: {"$ne": []}})
    if as_of_units is None:
        role_term: dict[str, Any] = {role_path: "main"}
    else:
        role_term = {"$and": [{role_path: "main"}, {ts_start_path: {"$lte": as_of_units}}]}
    match = {"$or": [role_term, *reach_terms]} if reach_terms else role_term
    return stages, match
