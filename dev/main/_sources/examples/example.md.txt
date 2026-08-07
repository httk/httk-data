# Serving records and validating them against their OPTIMADE definitions

This is the smallest useful tour of *httk-data*: the two capabilities that need
no database at all. A **provider** takes data you already have and answers the
neutral `httk.core.EntryProvider` contract about it; **validation** checks
concrete values against the OPTIMADE property definitions that describe them.
Both run fully offline — no network, no server, no registry lookup.

## The in-memory providers

`ReferenceEntryProvider`, `FileEntryProvider` and `CalculationEntryProvider`
each wrap a plain `{id: record}` mapping of one of httk-core's stdlib-only
record models (`Reference`, `File`, `Calculation`) and serve it as the
corresponding OPTIMADE entry type. A record may be given as a model instance or
as a mapping — the provider calls `Reference.create(...)` on whatever it gets,
so the two forms below are interchangeable.

Reading a provider is always the same three steps, and this example does them
in order:

1. `entry_types()` — what is served, described by an `EntryTypeDefinition`. The
   providers use the *vendored standard* definition (`standard_entry_type`), so
   the descriptions and types are the ones in the OPTIMADE specification, not
   ad-hoc local guesses.
2. `property_keys(entry_type)` — the map from a served property name to the key
   it occupies in a record. It exists so records can keep whatever internal
   shape they already had: here `id` lives under the record key `__id`.
3. `records(entry_type)` — JSON-able mappings, read *through* that map.

`relationships(entry_type)` is the optional fourth step, mapping an entry id to
a flat tuple of `RelatedEntry` values.

## Validation

`validate_record(entry_type, record)` checks a whole record: every key must be
a property the entry type describes, `id` and `type` must be present, and every
value present is validated against its property definition. `validate_property`
does one value on its own. Both raise `PropertyValidationError`, which carries
the offending property `name`.

The definitions *are* JSON Schema documents (that is what
`PropertyDefinition.as_optimade()` returns), so validation is a plain
`jsonschema` Draft 2020-12 run against a self-contained schema. The one subtle
point the failures below illustrate: OPTIMADE's `references.year` is a
**string** property, so the integer `2021` is rejected while `"2021"` passes —
a mistake that is easy to make, and easy to catch here.

For storing and querying data in a database rather than holding it in memory,
see the other examples, which build on `httk.data.db`.

```{literalinclude} ../../examples/example.py
:language: python
:lines: 51-
```
