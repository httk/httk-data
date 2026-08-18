# *httk-store*

This site documents specifically the *httk-store* module. For the full
documentation of *httk₂* as a whole, see [docs.httk.org](https://docs.httk.org).

*httk-store* is a [*httk₂*](https://github.com/httk/httk2) module for **data
management**. It is the *capability* layer built on the stdlib-only *contracts
and models* in *httk-core*: it serves httk-core's record models through the
neutral `httk.core.EntryProvider` contract, validates values against their
OPTIMADE property definitions with `jsonschema`, and provides the database
storage layer `httk.store.backend.sql` — relational storage and querying of plain frozen
dataclasses over SQLite, DuckDB, or PostgreSQL (via the `httk-store[db]` /
`httk-store[duckdb]` / `httk-store[postgresql]` extras).

```{admonition} Quick links
:class: tip

- **Data management guide**: {doc}`data`
- **Backend storage guide**: {doc}`db`
- **MongoDB storage guide**: {doc}`mongo`
- **Federated stores guide**: {doc}`federation`
- **Migrating from httk v1**: {doc}`migrating_from_v1`
- **API reference**: {doc}`reference/index`
- **Runnable examples**: {doc}`examples/index`
- **Examples notebook**: {doc}`notebooks/examples`

The topic pages above are short and practical; the ones with a full guide link
onward to it in the **Details** section of the sidebar.
````

## Install

Preferably work in a Python virtual environment, then do:
```bash
git clone https://github.com/httk/httk-store
cd httk-store
python -m pip install -e .
```

## Usage example

```python
from httk.core import standard_entry_type
from httk.store import ReferenceEntryProvider, validate_record

# Serve a bibliographic reference through the entry-provider contract.
provider = ReferenceEntryProvider({"ref-1": {"title": "A study", "year": "2021"}})
assert list(provider.entry_types()) == ["references"]

# Validate an OPTIMADE 'references' record against its vendored definition.
validate_record(
    standard_entry_type("references"),
    {"id": "ref-1", "type": "references", "title": "A study", "year": "2021"},
)
```

```{toctree}
:maxdepth: 2
:caption: Documentation

data
db
mongo
federation
migrating_from_v1
testing
clickhouse-testing
postgres-testing
reference/index
examples/index
notebooks/examples
```

```{toctree}
:maxdepth: 1
:caption: Details

details/db
details/mongo
details/migrating_from_v1
```
