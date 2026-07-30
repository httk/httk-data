# Federated stores

`FederatedStore` presents two or more existing `httk.data.Store` instances as
one read-only, source-major union. It is a data-management capability in
*httk-data*: it has no dependency on a serving protocol or on *httk-optimade*.

```python
from contextlib import ExitStack

from httk.data import FederatedStore

with ExitStack() as stack:
    first = stack.enter_context(open_first_store())
    second = stack.enter_context(open_second_store())
    combined = FederatedStore({"first": first, "second": second})
    # Use combined while the caller-owned child stores remain open.
```

The federation borrows its children: constructing it makes no requests and it
never closes a child. The caller owns remote-client and database lifecycles;
`contextlib.ExitStack` is useful when there are several independently managed
stores.

## Targets, filters, and outputs

Constructor mapping order is preserved. With no sort, all matches from the
first source are returned in that source's native order, then those from the
second, and so on. The federation is a union, not a deduplicating merge: equal
IDs from different sources remain separate rows.

When every source accepts the same target, bind it directly:

```python
search = combined.searcher()
record = search.variable(MyRecord)
search.add(record.energy >= threshold)
rows = search.results(record=record, energy=record.energy, origin=search.origin)
```

For sources that need distinct concrete descriptors, create a
`FederatedTarget`. Its mapping can deliberately select only a subset of the
federation's sources:

```python
records = combined.target(
    "records",
    {"first": first_descriptor, "second": second_descriptor},
)
search = combined.searcher()
record = search.variable(records)
```

The opaque `search.origin` projection returns the stable source name without
wrapping or changing the child record. Record and scalar projections otherwise
retain the exact values returned by the child store.

Federation supports the portable single-root filter profile: literal scalar
comparisons (including `None`), `contains`, `startswith`, `endswith`, `has`,
`has_any`, `has_only`, `is_in`, boolean `&`, `|`, `~`, and
`always_true()`/`always_false()`. Each operation and projection is validated
against every participating source before execution. Field-to-field
comparisons, a second root, and ordinary `add_sort()` are rejected; global sort
semantics are not yet part of the neutral store contract.

## Paging and exact counts

`add_offset()` and `set_limit()` apply globally after the source-major union,
not once per source. Iteration is lazy and sequential; a zero global limit
contacts no child, and a satisfied limit need not contact later sources.

`search.count()` requests each participating child's fresh, unpaged filtered
exact count and sums those totals. It ignores global offset and limit. A frozen
result set caches that successful total, and `len(result)` applies the frozen
plan's global offset and limit to it; slices share the same exact-count cache.
Counting never crawls result pages or returns a partial total.

## Failures and boundaries

Each child is executed sequentially with a fresh child searcher. A child
failure raises `FederatedSourceError` naming the source and operation and
chains the original exception; unsupported-query and exact-count-unavailable
categories are retained. Earlier streamed rows do not turn a later failure
into a successful partial result.

There is no best-effort or `ignore_errors` mode. Federation does not implement
writes, distributed transactions, deduplication, sorting, concurrency, or
cross-store joins.
