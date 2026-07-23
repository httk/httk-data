# *httk-data*

This site documents specifically the *httk-data* module. For the full
documentation of *httk₂* as a whole, see [docs.httk.org](https://docs.httk.org).

*httk-data* is a [*httk₂*](https://github.com/httk/httk2) module for **data
management**. It is the *capability* layer built on the stdlib-only *contracts
and models* in *httk-core*: it serves httk-core's record models through the
neutral `httk.core.EntryProvider` contract, and validates values against their
OPTIMADE property definitions with `jsonschema`. It is also the intended future
home of the v1-style sqlite/database storage layer (not built yet).

```{admonition} Quick links
:class: tip

- **Data management guide**: {doc}`data`
- **API reference**: {doc}`reference/index`
- **Examples notebook**: {doc}`notebooks/examples`
````

## Install

Preferably work in a Python virtual environment, then do:
```bash
git clone https://github.com/httk/httk-data
cd httk-data
python -m pip install -e .
```

## Usage example

```python
from httk.core import standard_entry_type
from httk.data import ReferenceEntryProvider, validate_record

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
reference/index
notebooks/examples
```
