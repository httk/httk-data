"""Serving records and validating them against their OPTIMADE definitions

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
"""

from typing import Any

from httk.core import Reference, RelatedEntry, standard_entry_type

from httk.data import (
    PropertyValidationError,
    ReferenceEntryProvider,
    validate_property,
    validate_record,
)

#: Two bibliographic references: one as a model instance, one as a plain mapping.
ENTRIES: dict[str, Reference | dict[str, Any]] = {
    "ref-1": Reference(
        title="Notes on the Analytical Engine",
        year="1843",
        journal="Scientific Memoirs",
        doi="10.1000/analytical",
    ),
    "ref-2": {"title": "An Investigation of the Laws of Thought", "year": "1854"},
}

#: A related entry, served flat per id; the serving layer groups them by type.
RELATIONSHIPS = {
    "ref-1": (RelatedEntry("files", "file-1", description="Scanned original", role="input"),),
}

#: Property values that must NOT validate, and why.
REJECTED: list[tuple[str, Any, str]] = [
    ("year", 2021, "the OPTIMADE 'references.year' property is a string, not an integer"),
    ("id", None, "'id' is required in the response, hence not nullable"),
    ("authors", "Ada Lovelace", "'authors' is a list of dicts, not a bare string"),
]


def show_provider(provider: ReferenceEntryProvider) -> None:
    """Walk the entry-provider contract the way a serving module does."""
    print("== What is served ==")
    for name, definition in provider.entry_types().items():
        print(f"  {name}: {definition.description.splitlines()[0]}")
        print(f"  described properties: {len(definition.properties)}")
    print()

    print("== Served property -> record key ==")
    property_keys = provider.property_keys("references")
    for name in ("id", "type", "title", "year"):
        print(f"  {name!r} is read from record key {property_keys[name]!r}")
    print()

    print("== The records, read through that map ==")
    for record in provider.records("references"):
        served = {name: record[key] for name, key in property_keys.items()}
        interesting = {name: value for name, value in served.items() if value is not None}
        print(f"  {interesting}")
    print()

    print("== Relationships ==")
    for entry_id, related in provider.relationships("references").items():
        for entry in related:
            print(f"  {entry_id} -> {entry.entry_type}/{entry.id} (role={entry.role})")
    print()


def show_validation(provider: ReferenceEntryProvider) -> None:
    """Validate every served record, then show values the definitions reject."""
    references = standard_entry_type("references")
    property_keys = provider.property_keys("references")

    print("== Every served record validates against the standard definition ==")
    for record in provider.records("references"):
        served = {name: record[key] for name, key in property_keys.items()}
        validate_record(references, served)
        print(f"  {served['id']}: valid")
    print()

    print("== Values the definitions reject ==")
    for name, value, reason in REJECTED:
        try:
            validate_property(references.properties[name], value)
        except PropertyValidationError as error:
            print(f"  {name}={value!r}: rejected -- {reason}")
            print(f"    {error}")
        else:
            raise AssertionError(f"expected {name}={value!r} to be rejected")
    print()

    print("== A record carrying a property the entry type does not describe ==")
    try:
        validate_record(references, {"id": "ref-3", "type": "references", "sprocket": 3})
    except PropertyValidationError as error:
        print(f"  rejected -- {error}")
    else:
        raise AssertionError("expected an unknown property to be rejected")


def main() -> None:
    provider = ReferenceEntryProvider(ENTRIES, relationships=RELATIONSHIPS)
    show_provider(provider)
    show_validation(provider)


if __name__ == "__main__":
    main()
