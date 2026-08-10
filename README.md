# httk-store

*httk-store* is a [*httk₂*](https://github.com/httk/httk2) module for data
management. Built on the stdlib-only contracts and models in *httk-core*, it
provides in-memory `httk.core.EntryProvider` implementations for the standard
OPTIMADE entry types (`references`, `files`, `calculations`),
property-definition validation on `jsonschema`, and a database storage layer
(`httk.store.db`, via the `httk-store[db]` extra) that stores plain frozen
dataclasses in SQLite or DuckDB, makes them queryable through a backend-agnostic
search DSL, and serves them through the entry-provider contract.
