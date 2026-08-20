# httk-store

![Status: Early beta](https://img.shields.io/badge/status-early--beta-orange)

> **⚠️ EARLY BETA**
>
> This is an early beta release of *httk₂*. The organization of the packages
> and their APIs should not yet be regarded as stable, and may change between
> releases.

*httk-store* is a [*httk₂*](https://github.com/httk/httk2) module for data
management. Built on the stdlib-only contracts and models in *httk-core*, it
provides in-memory `httk.core.EntryProvider` implementations for the standard
OPTIMADE entry types (`references`, `files`, `calculations`),
property-definition validation on `jsonschema`, and a database storage layer
(`httk.store.backend.sql`, via the `httk-store[db]` extra) that stores plain frozen
dataclasses in SQLite or DuckDB, makes them queryable through a backend-agnostic
search DSL, and serves them through the entry-provider contract.
