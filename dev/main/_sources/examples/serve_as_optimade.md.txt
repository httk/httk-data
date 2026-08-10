# Serving a database as an OPTIMADE API

The last step of the chain: a store full of frozen dataclasses becomes an
OPTIMADE API. `StoreEntryProvider` implements httk-core's neutral
`httk.core.EntryProvider` contract over an `SqlStore`; *httk-serve*'s
`adapter_from_providers` consumes that contract. *httk-store* does not depend on
*httk-serve*. The provider handoff uses the httk-core contract, while
*httk-serve* also consumes *httk-store*'s neutral query and store APIs.

## What the provider derives from the schema

`StoreEntryProvider(store, {"books": Book, "writers": Writer})` needs nothing
but the mapping from entry-type name to storable class. From each class's
storage schema it derives:

- an `EntryTypeDefinition` with one OPTIMADE property per servable stored
  field, named with a registered database-specific prefix (`_httk_` by
  default), plus the mandatory `id` and `type`. You may pass your own
  `definitions=` instead, as long as they describe everything served.
- JSON-able records: rationals are served as their nearest floats (the exact
  values stay in the database), datetimes as ISO text, fixed-shape tensors as
  nested lists, and `stored_property` values as ordinary properties.
- relationships, for reference fields and `list[Storable]` fields **whose
  target class is also served**. A `Related` marker on the field adds the
  OPTIMADE `role`/`description` metadata; `StorageInfo(links=...)` lets a
  separate join class contribute relationships without being served itself.

Fields with no OPTIMADE value representation — `bytes`, custom codecs — are
simply not served, and a reference field whose target class is not served
becomes neither a property nor a relationship.

## Running a query

`adapter_from_providers([provider])` builds a backend adapter: it reads the
definitions, loads the records, and wires the filter handlers and the
relationship id fields. `execute_query(adapter, entries, response_fields, ...)`
is then the whole OPTIMADE query pipeline minus HTTP — the same call an
OPTIMADE web endpoint makes after parsing a request. It takes the filter as an
already-parsed AST from `parse_optimade_filter`, so this example goes
end to end from a filter *string* to served records.

Note the difference from the *OPTIMADE filters* example: there the filter runs
as SQL against the store and yields dataclass instances; here it runs through
the serving layer and yields **records** — id, type and the selected response
fields — which is what an API actually returns.

## Ids

By default an entry is identified as `"<entry type>-<sid>"`, using the row's
integer sid: `books-1`, `writers-3`. An `id_of` callback replaces that scheme
everywhere at once, including inside relationships.

```{literalinclude} ../../examples/serve_as_optimade.py
:language: python
:lines: 53-
```
