# Querying stored dataclasses with OPTIMADE filter strings

`optimade_filter_searcher(store, cls, filter_string)` takes an **OPTIMADE
filter string** — the query language of the OPTIMADE API, as it arrives in a
URL — and returns a searcher over the stored rows of `cls`. It is the bridge
between the text a user (or a remote client) writes and the search DSL of the
*searching* example: the filter is parsed by httk-core's OPTIMADE grammar,
translated by `httk.data.query.optimade_filters` against a handler table derived from
the class's storage schema, and executed as an ordinary search. Each match
is available as the friendly `searcher.results().scalars()` stream.

This is useful well before you serve anything: it lets a filter string be
evaluated against a local database with no HTTP server, no adapter and no
OPTIMADE endpoint in sight. (For the full serving path, see the
*serve as OPTIMADE* example.)

## Property names are the *served* names

An OPTIMADE filter may only mention properties the API declares, and a
database-specific property must carry a registered provider prefix. The stored
fields of `cls` are therefore filterable as `_httk_custom_<field>` — `_httk_custom_name`,
`_httk_custom_x`, `_httk_custom_symbols` — and **not** as the bare field name. This holds
inside dotted relationship paths too: `refs._httk_custom_doi`, not `refs.doi`.

The rules for names that do not resolve follow the OPTIMADE specification
rather than being uniformly strict:

- an unknown name *carrying* the prefix is an error
  (`FilterTranslationError`, category `"unrecognized-property"`), because a
  client asking for `_httk_bananas` has made a mistake worth reporting;
- an unknown name *without* a prefix matches nothing, silently, because it may
  be a standard property this implementation does not support.

## Operators

Everything the grammar offers is available. Comparisons on numeric and string
properties; `STARTS WITH`, `ENDS WITH` and `CONTAINS` for substrings; and the
list operators `HAS`, `HAS ALL`, `HAS ANY`, `HAS ONLY` on variable-length
fields, which map onto the set semantics described in the *searching* example.
`NOT` negates, with the same set reading: `NOT _httk_custom_symbols HAS "O"` selects
the records where *no* symbol is O.

Rational fields (`Fraction` and friends) are stored exactly but compared on
their float companion columns, so a numeric filter on one is
documented-approximate.

## Relationship filters

`related_classes={"refs": Publication}` declares that the relationship type
`refs` is served from the `Publication` field of `cls` — either a reference
field or a `list[Publication]` child field. That unlocks depth-1 dotted
filters, resolved as a **two-phase semi-join**: a nested filter search over
`Publication` collects the sids of the matching publications, and those are
then matched against the relationship's own column. Two consequences worth
knowing:

- `NOT refs._httk_custom_year >= 2000` is the complement over the *parent* rows, so it
  includes rows with no related publication at all.
- A related filter matching nothing yields an empty result, not an error.

Relationship *ids* are filterable as `refs.id HAS "refs-<sid>"`, using the same
`"<entry type>-<sid>"` ids `StoreEntryProvider` mints by default. Nesting
deeper than one level (`refs.other._httk_custom_x`) raises `FilterTranslationError`
with category `"not-implemented"`.

```{literalinclude} ../../examples/optimade_filters.py
:language: python
:lines: 66-
```
