# Migrating from httk v1 in detail

The database layer of httk v1 (`httk.db`, with `httk.db.backend.Sqlite` and
`httk.db.store.SqlStore`) and the storage layer of *httk₂* (`httk.data.db`, see
{doc}`db`) solve the same problem, and the shape of a query survives the port
almost unchanged: open a searcher, bind classes to variables, add conditions,
declare outputs, iterate. What changed is how a class is declared storable, and
that several v1 constructs whose meaning depended on context — most of all
`add` versus `add_all` — were replaced by ones that mean the same thing
everywhere.

```{admonition} v1 snippets on this page are historical
:class: warning

Every block marked *httk v1* is shown only so you can recognise it in your own
scripts. It requires httk v1 and does **not** run under *httk₂*. The blocks
marked *httk₂* are the ones to copy.
```

## What stays the same

`store.searcher()` still opens a query; `search.variable(Cls)` still binds a
class to a variable; `search.add(...)` still adds a condition; `search.output(
variable, name)` still declares an output; iterating still runs the query.
Attribute access on a variable still builds the joins for you, two variables of
the same class still self-join, comparing a reference field against another
variable (`result.structure == structure`) is still how you write a join
condition, and `set_limit` / `add_offset` / `count()` are unchanged.

## First: `add` versus `add_all`

This is the one v1 idiom that silently changes meaning if you port it
literally. In v1 the *same* expression meant different things depending on
which method it was passed to:

```python
# httk v1 — historical, does not run under httk₂
search.add(search_struct.formula_symbols.is_in('Na'))              # contains ANY of
search.add_all(search_struct.formula_symbols.is_in('O','Ca','Ti')) # contains ONLY these
```

Both lines are real v1 code (`Examples/4_database/1_store_retrieve.py` and
`Tutorial/Step6/step6_part5.py` respectively), and the first even carries the
comment *"cf. the search for a material system in tutorial step6, where all
symbols must be in a set, 'add' vs. 'add_all'"*.

In *httk₂* the reading is in the expression, not in the method. There is one
`add()`, and each expression carries its own placement:

```python
# httk₂
search.add(s.symbols.has_any("Na"))               # existential: some symbol is Na
search.add(s.symbols.has_only("O", "Ca", "Ti"))   # for-all: every symbol is in the set
```

`s.symbols.is_in("O", "Ca", "Ti")` on a child (list-valued) field is the same
for-all reading as `has_only`; on a root field `is_in` is plain membership.
Against records `NaCl`, `CaTiO3` and `NaTiO2`, `has_any` matches `NaCl` and
`NaTiO2`, while `has_only` and child-field `is_in` both match `CaTiO3` alone.

## Old → new

| httk v1 | *httk₂* |
| --- | --- |
| `httk.db.backend.Sqlite(path)` then `httk.db.store.SqlStore(backend)` | `Database.sqlite(path)` then `SqlStore(database, entry_records={})` for first use, `SqlStore(database)` when reopening |
| `httk.db.backend.Duckdb(path)` | `Database.duckdb(path)` |
| `store.delay_commit()` … `store.commit()` | `with store.transaction():` |
| subclass `httk.HttkObject`, `@httk.httk_typed_init({...}, index=[...], skip=[...])` | plain frozen dataclass with `Annotated` markers (`Indexed`, `Unique`, `Skip`, `Shape`, `Related`) and an optional `__httk_storage__: ClassVar[StorageInfo]` |
| `@httk_typed_property(t)` | `@stored_property` (value type read from the return annotation) |
| `store.save(obj)`, `obj.db.store(store)` | `store.save(obj)` |
| `search.add_all(expr)` (a separate post-filter position) | gone — one `add()`; the expression decides where it applies |
| `col.has_inv_any(...)`, `col.has_inv_only(...)` | `~col.has_any(...)`, `~col.has_only(...)` |
| `col.like('%x%')` with raw SQL wildcards | `col.contains("x")`, `col.startswith(...)`, `col.endswith(...)`, matching **literal** text (`%` and `_` match themselves) |
| `add_sort(expr, direction='ASC')` | `add_sort(field, descending=False)` |
| `for match, header in search:` then `match[0]` | iteration yields a `SearchResult`; use `result.values` / `result.names` — `result[0][0]` still works |
| `obj.db.sid` | `store.sid_of(obj)` |
| `hexhash` deduplication | content-id deduplication; look a record up with `store.fetch_by_content_id(cls, key)` |
| `struct.get_tags()`, `struct.get_refs()` (lazy codependent fetch) | explicit `store.referring(TagCls, field="structure", to=struct)` |

One v1 trap is worth naming because it looked like a query bug: because the v1
table name came from `cls.__name__`, the storable class had to be re-declared
verbatim in the querying script, and `search.variable(cls)` would silently
*create* the table when it did not exist — so a renamed or slightly mistyped
class returned no matches instead of an error. *httk₂* still creates a class's
tables on first use, but you import the class rather than re-declare it, and
`StorageInfo(storage_name="…")` pins the stored name so renaming the class
cannot orphan the data.

## Converting a store-and-search script

The v1 original (`Examples/4_database/1_store_retrieve.py`) stored two
structures with a tag and searched for anything containing sodium:

```python
# httk v1 — historical, does not run under httk₂
import httk, httk.db
from httk.atomistic import Structure

backend = httk.db.backend.Sqlite('example.sqlite')
store = httk.db.store.SqlStore(backend)
tablesalt = httk.load('NaCl.cif')
tablesalt.add_tag('common name', 'salt')
arsenic = httk.load('As.cif')
store.save(tablesalt)
store.save(arsenic)

search = store.searcher()
search_struct = search.variable(Structure)
search.add(search_struct.formula_symbols.is_in('Na'))
search.output(search_struct, 'structure')

for match, header in list(search):
    struct = match[0]
    print("Found structure", struct.formula, [str(struct.get_tags()[x]) for x in struct.get_tags()])
```

In *httk₂* the record class is declared where you use it — a plain frozen
dataclass — and the tag becomes an ordinary join class stored like any other,
looked up with `store.referring`:

```python
# httk₂
from dataclasses import dataclass
from typing import Annotated, ClassVar

from httk.core.storage import Indexed, StorageInfo
from httk.data.db import Database, SqlStore


@dataclass(frozen=True)
class Structure:
    formula: Annotated[str, Indexed()]
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class StructureTag:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    structure: Structure
    tag: Annotated[str, Indexed()]
    value: str


store = SqlStore(Database.sqlite("example.sqlite"), entry_records={})

tablesalt = Structure("NaCl", ("Na", "Cl"))
arsenic = Structure("As", ("As",))
with store.transaction():
    store.save(tablesalt)
    store.save(arsenic)
    store.save(StructureTag(tablesalt, "common name", "salt"))

search = store.searcher()
s = search.variable(Structure)
search.add(s.symbols.has_any("Na"))
search.output(s, "structure")
for result in search:
    structure = result.values[0]
    tags = store.referring(StructureTag, field="structure", to=structure)
    print("Found structure", structure.formula, [f"{tag.tag}={tag.value}" for tag in tags])
```

Point by point: `add(... .is_in('Na'))` became `add(... .has_any("Na"))`;
`delay_commit`/`commit` became `transaction()`; `add_tag`/`get_tags` became a
stored join class and `store.referring`; and `for match, header in list(search)`
became a plain loop over `SearchResult` objects, of which `result.values[0]` is
the first declared output. Note that `symbols` is a `tuple`, not a `list`: an
instance with a list field is unhashable, and the store's object→sid cache
(which `store.referring` and `store.sid_of` consult) is equality-keyed.

## Converting a result class and its search

The v1 original (`Tutorial/Step6/step6_part4.py` and `step6_part5.py`) declared
a result class with `httk_typed_init` — once in the storing script and again,
verbatim, in the querying one — and searched it joined against `Structure`:

```python
# httk v1 — historical, does not run under httk₂
class TotalEnergyResult(httk.Result):

    @httk.httk_typed_init({'computation': httk.Computation, 'structure': Structure, 'total_energy': float})
    def __init__(self, computation, structure, total_energy):
        self.computation = computation
        self.structure = structure
        self.total_energy = total_energy

backend = httk.db.backend.Sqlite('example.sqlite')
store = httk.db.store.SqlStore(backend)

search = store.searcher()
search_total_energy = search.variable(TotalEnergyResult)
search_struct = search.variable(Structure)
search.add(search_total_energy.structure == search_struct)
search.add_all(search_struct.formula_symbols.is_in('O', 'Ca', 'Ti'))
search.output(search_total_energy, 'total_energy_result')

for match, header in search:
    total_energy_result = match[0]
    structures += [total_energy_result.structure]
    energies += [total_energy_result.total_energy]
```

The *httk₂* version declares the classes once, as dataclasses, and translates
the `add_all(... .is_in(...))` post-filter into a single `add(... .has_only(...))`:

```python
# httk₂
from dataclasses import dataclass
from typing import Annotated, ClassVar

from httk.core.storage import Indexed, StorageInfo, stored_property
from httk.data.db import Database, SqlStore


@dataclass(frozen=True)
class Structure:
    formula: Annotated[str, Indexed()]
    symbols: tuple[str, ...]

    @stored_property
    def nsymbols(self) -> int:                    # queryable, recomputed on load
        return len(self.symbols)


@dataclass(frozen=True)
class Computation:
    code: str
    version: str


@dataclass(frozen=True)
class TotalEnergyResult:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(indexes=(("structure", "total_energy"),))

    computation: Computation
    structure: Structure
    total_energy: float


store = SqlStore(Database.sqlite("results.sqlite"), entry_records={})

vasp = Computation("VASP", "5.4.4")
runs = [
    (Structure("CaTiO3", ("Ca", "Ti", "O")), -39.2),
    (Structure("TiO2", ("Ti", "O")), -26.5),
    (Structure("NaCl", ("Na", "Cl")), -7.1),
]
with store.transaction():
    for structure, energy in runs:
        store.save(TotalEnergyResult(vasp, structure, energy))   # saves referents recursively

search = store.searcher()
result = search.variable(TotalEnergyResult)
structure = search.variable(Structure)
search.add(result.structure == structure)                        # the join condition, as in v1
search.add(structure.symbols.has_only("O", "Ca", "Ti"))          # was add_all(... .is_in(...))
search.add_sort(result.total_energy, descending=False)           # was direction='ASC'
search.output(result, "total_energy_result")

structures = []
energies = []
for match in search:
    total_energy_result = match.values[0]
    structures.append(total_energy_result.structure)
    energies.append(total_energy_result.total_energy)
```

This prints `['CaTiO3', 'TiO2']` and `[-39.2, -26.5]`: `NaCl` is excluded by the
for-all condition, and the sort is by total energy ascending. There is no
`store.commit()` at the end — the `transaction()` block committed on exit.

## Looking records up by identity

v1 identified a stored object through the `obj.db` plugin and deduplicated on
`hexhash`. *httk₂* has no plugin on the object: the store answers both
questions.

```python
# httk₂
from httk.core.storage import content_id

nacl = structures[0]
sid = store.sid_of(nacl)                                  # was nacl.db.sid
found = store.fetch_by_content_id(Structure, content_id(nacl))   # was the hexhash lookup
```

For content-addressed records, `sid_of` queries the database when its local
cache has no answer, so an equal object and a reopened store resolve the same
store-local sid. `fetch_by_content_id` retrieves the corresponding record.

## No longer available

These v1 features have no *httk₂* replacement:

- the `obj.db` plugin and the `Storable` base class — storability is
  non-intrusive, and all storage operations go through the store;
- `DictStore` and `TrivialStore`;
- `searcher.store_table(name)`;
- `searcher.sql()` and `searcher.sql_query()` SQL introspection;
- `search.function(name)` custom SQL functions;
- `output(..., concat=True)` (`GROUP_CONCAT`);
- arithmetic operators on columns, and the `^` (xor) and `//` (string
  concatenation) operators;
- `searcher.reset()` and `searcher.duplicate(other)` — build a new searcher;
- explicit join keys, i.e.
  `variable(Cls, parent=..., parentkey='Compound_id', subkey='compound_Compound_sid')`;
  `variable()` takes only the class and derives the joins from the schema;
- the `DATABASE_DEBUG` and `DATABASE_DEBUG_SLOW` environment variables.
