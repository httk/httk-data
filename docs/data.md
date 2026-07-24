# Data management

*httk-data* is the **data-management capability layer** of *httk₂*. The guiding
principle of the workspace is *contracts and models in httk-core; capabilities
in modules*: httk-core defines the neutral `httk.core.EntryProvider` contract,
the OPTIMADE property/entry-type definition model, and the stdlib-only record
dataclasses (`Reference`, `File`, `Calculation`) — but ships no concrete
providers and no third-party dependencies. httk-data builds the capabilities on
top of those models:

- **entry providers** that serve the record models through the provider
  contract,
- **property-definition validation** on `jsonschema`, and
- the **database storage layer** `httk.data.db` (see {doc}`db`), which stores
  plain frozen dataclasses relationally and serves them through the same
  provider contract.

## Entry providers

`ReferenceEntryProvider`, `FileEntryProvider`, and `CalculationEntryProvider`
map a `{id: record}` mapping — where each record is a httk-core dataclass (or a
plain mapping coerced into one) — onto the neutral contract. A provider answers
three questions: which entry types it serves (each described by a first-class
`httk.core.EntryTypeDefinition`), which record column holds each served property
(`columns()`), and what the records are (`records()`, plain JSON-able mappings).

```python
from httk.core import Reference
from httk.data import ReferenceEntryProvider

provider = ReferenceEntryProvider(
    {
        "ref-1": Reference(title="A study of gallium titanium compounds", doi="10.1234/demo.2021.1"),
        "ref-2": {"title": "Silicon dioxide polymorphs revisited", "year": "2019"},
    }
)

# The served entry type is the vendored OPTIMADE 'references' definition:
entry_types = provider.entry_types()
assert list(entry_types) == ["references"]
assert entry_types["references"].properties["title"].optimade_type == "string"

# columns() maps served property name -> record column; records() are JSON-able:
columns = provider.columns("references")
assert columns["id"] == "__id" and columns["type"] == "type"

records = list(provider.records("references"))
assert {r["__id"] for r in records} == {"ref-1", "ref-2"}
assert records[0]["type"] == "references"
```

### Discovery through the registry

The three providers self-register when `httk.core` discovers the module, under
the names `data-references`, `data-files`, and `data-calculations` (the
database-backed provider of {doc}`db` registers alongside them as
`data-db-store`). A serving module (such as *httk-optimade*) can therefore
find them through the registry without importing httk-data directly:

```python
import httk.core
from httk.core import known_entry_providers
from httk.core._plugins import resolve_callable
from httk.core.register import entry_providers

registered = set(known_entry_providers())
assert {"data-references", "data-files", "data-calculations"} <= registered

# The registered value is a lazy factory reference; resolve and instantiate it:
factory = resolve_callable(entry_providers.require("data-references").handler)
provider = factory({"ref-1": {"title": "A study"}})
assert list(provider.entry_types()) == ["references"]
```

To actually serve these providers over HTTP as an OPTIMADE API, hand them to
*httk-optimade*'s `adapter_from_providers` (see that module's documentation);
neither module imports the other — the httk-core contract is the only coupling.

## Validation

`validate_property` checks a single value against one property definition, and
`validate_record` checks a whole record against an entry-type definition. Both
build a `jsonschema` **Draft 2020-12** validator directly from the definition's
OPTIMADE document. The definitions are self-contained (no `$ref`), and the
document's `$schema` *meta*-schema reference is removed before validation, so
**no network access or schema resolution ever happens** — validation is fully
offline.

```python
from httk.core import standard_entry_type
from httk.data import validate_property, validate_record

references = standard_entry_type("references")

# A single value against one property definition (references.year is a string):
validate_property(references.properties["year"], "2021")

# Nullable properties accept None:
assert references.properties["year"].nullable
validate_property(references.properties["year"], None)

# A whole record against the entry-type definition. Properties absent from the
# record are simply not checked (serving a subset is normal); 'id' and 'type'
# must be present:
validate_record(
    references,
    {"id": "ref-1", "type": "references", "title": "A study", "year": "2021"},
)
```

### Failing validation

A value that does not match its definition raises `PropertyValidationError`
(a `ValueError`). The underlying `jsonschema` error is kept as the chained
cause:

```python
from httk.core import standard_entry_type
from httk.data import PropertyValidationError, validate_property, validate_record

references = standard_entry_type("references")

# 'year' is a string property, so an integer is rejected:
try:
    validate_property(references.properties["year"], 2021)
except PropertyValidationError as exc:
    assert exc.name == "year"
    assert exc.__cause__ is not None  # the jsonschema ValidationError

# Unknown property names are rejected, naming them and the entry type:
try:
    validate_record(references, {"id": "r", "type": "references", "sprocket": 3})
except PropertyValidationError as exc:
    assert "sprocket" in str(exc)

# 'id' and 'type' are required:
try:
    validate_record(references, {"type": "references"})
except PropertyValidationError as exc:
    assert exc.name == "id"
```

## Database-backed serving

The entry providers above are in-memory. To store records in a database and
serve them the same way, see {doc}`db`: `httk.data.db.SqlStore` stores plain
frozen dataclasses in SQLite or DuckDB, and `StoreEntryProvider` (registered
as `data-db-store`) serves them through the identical provider contract.
