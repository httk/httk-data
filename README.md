# httk-data

*httk-data* is a [*httk₂*](https://github.com/httk/httk2) module for data
management. Built on the stdlib-only contracts and models in *httk-core*, it
provides in-memory `httk.core.EntryProvider` implementations for the standard
OPTIMADE entry types (`references`, `files`, `calculations`) and
property-definition validation on `jsonschema`. It is also the intended future
home of the v1-style sqlite/database storage layer.
