"""Versioned physical layout for :class:`httk.data.db.store.SqlStore`.

The SQL object mapper deliberately permits ordinary frozen dataclasses to be
introduced on demand.  A persisted store, however, also needs a small stable
contract for the entry families it serves.  This module owns that contract:
the protocol marker, its registry-name declaration, deterministic dispatch
tables, and read-only SQLite/DuckDB layout comparison.
"""

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

import sqlalchemy
from httk.core import (
    entry_backing_info,
    entry_family_info,
    known_entry_backings,
    known_entry_families,
    resolve_entry_backing,
    resolve_entry_family,
)

from httk.data.db.mapping import dispatch_table_for, entry_dispatch_table_name, table_for
from httk.data.db.schema import resolve_schema

__all__ = [
    "METADATA_TABLE_NAME",
    "STORAGE_PROTOCOL_VERSION",
    "EntryFamilyLayout",
    "StorageLayout",
    "StorageLayoutUpgradeRequiredError",
    "actual_schema_objects",
    "actual_table_names",
    "declaration_json",
    "expected_metadata",
    "metadata_table_for",
    "normalize_entry_backings",
    "read_store_metadata",
    "validate_expected_tables",
]

STORAGE_PROTOCOL_VERSION: Final = "v2.0.2"
"""The persisted SqlStore layout protocol implemented by this package."""

METADATA_TABLE_NAME: Final = "_httk_store_metadata"
"""Reserved key/value table holding the store protocol and entry declaration."""

_METADATA_PROTOCOL_KEY: Final = "protocol"
_METADATA_DECLARATION_KEY: Final = "entry_declaration"
_RESERVED_PREFIX: Final = "_httk_"


class StorageLayoutUpgradeRequiredError(RuntimeError):
    """A database does not exactly implement the current persisted store layout.

    ``diff`` is immutable and JSON-shaped.  Its top-level keys are stable
    categories (currently ``protocol``, ``declaration`` and ``schema``), so a
    caller can present a precise upgrade diagnostic without parsing
    the human-readable exception message.
    """

    def __init__(self, diff: Mapping[str, object]) -> None:
        frozen = _freeze_mapping(diff)
        self.diff: Mapping[str, object] = frozen
        categories = ", ".join(frozen) or "unknown layout difference"
        super().__init__(f"SqlStore layout upgrade is required ({categories})")


@dataclasses.dataclass(frozen=True)
class EntryFamilyLayout:
    """One immutable configured entry family and its concrete record backings."""

    name: str
    family: type
    backing_names: tuple[str, ...]
    backings: tuple[type, ...]
    dispatch_table_name: str | None


@dataclasses.dataclass(frozen=True)
class StorageLayout:
    """The immutable normalized entry declaration of an initialized store."""

    protocol_version: str
    families: tuple[EntryFamilyLayout, ...]

    @property
    def entry_backings(self) -> Mapping[type, tuple[type, ...]]:
        """Configured family classes mapped to their ordered concrete record classes."""
        return MappingProxyType({family.family: family.backings for family in self.families})

    @property
    def declaration(self) -> Mapping[str, tuple[str, ...]]:
        """Configured stable family names mapped to their ordered stable backing names."""
        return MappingProxyType({family.name: family.backing_names for family in self.families})


@dataclasses.dataclass(frozen=True)
class _TableDescription:
    columns: tuple[tuple[str, str, bool], ...]
    primary_key: tuple[str, ...]
    foreign_keys: frozenset[tuple[tuple[str, ...], str, tuple[str, ...], str, str, str, str, str]]
    uniques: frozenset[tuple[str, ...]]
    indexes: frozenset[tuple[str, tuple[tuple[str, str, str], ...], bool, str]]
    checks: frozenset[str]


@dataclasses.dataclass(frozen=True)
class _SqlToken:
    kind: str
    text: str
    value: str
    start: int
    end: int


def normalize_entry_backings(entry_backings: Mapping[type, type | tuple[type, ...]]) -> StorageLayout:
    """Validate an explicit class declaration and replace it with stable registry names.

    Registry aliases are rejected rather than selected arbitrarily: a
    persistent declaration must have exactly one stable spelling for every
    supplied class.
    """
    if not isinstance(entry_backings, Mapping):
        raise TypeError("entry_backings must be a mapping from entry-family classes to record backing classes")
    entries: list[EntryFamilyLayout] = []
    for family, supplied_backings in entry_backings.items():
        if not isinstance(family, type):
            raise TypeError("entry_backings keys must be entry-family classes")
        family_name = _registered_family_name(family)
        backings: tuple[type, ...]
        if isinstance(supplied_backings, type):
            backings = (supplied_backings,)
        elif isinstance(supplied_backings, tuple):
            backings = supplied_backings
        else:
            raise TypeError(f"entry_backings[{family.__name__}] must be a backing class or a tuple of backing classes")
        if not backings:
            raise ValueError(f"entry_backings[{family.__name__}] cannot be an empty tuple")
        if any(not isinstance(backing, type) for backing in backings):
            raise TypeError(f"entry_backings[{family.__name__}] contains a non-class backing")
        if len(set(backings)) != len(backings):
            raise ValueError(f"entry_backings[{family.__name__}] repeats a backing class")
        backing_names: list[str] = []
        for backing in backings:
            backing_name = _registered_backing_name(backing)
            registered_family_name, _ = entry_backing_info(backing_name)
            if registered_family_name != family_name:
                raise ValueError(
                    f"entry backing {backing.__name__} belongs to registered family {registered_family_name!r}, "
                    f"not {family_name!r}"
                )
            schema = resolve_schema(backing)
            if schema.dedup != "content_id":
                raise ValueError(
                    f"configured entry backing {backing.__name__} must use dedup='content_id', got {schema.dedup!r}"
                )
            backing_names.append(backing_name)
        entries.append(
            EntryFamilyLayout(
                name=family_name,
                family=family,
                backing_names=tuple(backing_names),
                backings=backings,
                dispatch_table_name=entry_dispatch_table_name(family_name) if len(backings) > 1 else None,
            )
        )
    entries.sort(key=lambda entry: entry.name)
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("entry_backings contains multiple registered aliases for the same family")
    layout = StorageLayout(STORAGE_PROTOCOL_VERSION, tuple(entries))
    _validate_physical_names(layout)
    return layout


def declaration_json(layout: StorageLayout) -> str:
    """Serialize a normalized declaration in its exact deterministic persisted form."""
    document = {
        "families": [{"backings": list(family.backing_names), "family": family.name} for family in layout.families],
        "format": 1,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _layout_from_declaration(value: str) -> StorageLayout:
    try:
        document = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("stored entry declaration is not valid JSON") from error
    if not isinstance(document, dict) or set(document) != {"families", "format"} or document["format"] != 1:
        raise ValueError("stored entry declaration does not use format 1")
    families = document["families"]
    if not isinstance(families, list):
        raise ValueError("stored entry declaration families must be a list")
    supplied: dict[type, tuple[type, ...]] = {}
    previous = ""
    for item in families:
        if not isinstance(item, dict) or set(item) != {"backings", "family"}:
            raise ValueError("stored entry declaration family entry is malformed")
        family_name = item["family"]
        backing_names = item["backings"]
        if not isinstance(family_name, str) or not isinstance(backing_names, list) or not backing_names:
            raise ValueError("stored entry declaration has an invalid family or backing list")
        if family_name <= previous:
            raise ValueError("stored entry declaration families are not deterministically ordered")
        previous = family_name
        family = resolve_entry_family(family_name)
        resolved_backings: list[type] = []
        for backing_name in backing_names:
            if not isinstance(backing_name, str):
                raise ValueError("stored entry declaration backing names must be strings")
            declared_family, _ = entry_backing_info(backing_name)
            if declared_family != family_name:
                raise ValueError(
                    f"stored entry backing {backing_name!r} is registered for {declared_family!r}, not {family_name!r}"
                )
            resolved_backings.append(resolve_entry_backing(backing_name))
        supplied[family] = tuple(resolved_backings)
    layout = normalize_entry_backings(supplied)
    if declaration_json(layout) != value:
        raise ValueError("stored entry declaration is not in its canonical deterministic encoding")
    return layout


def expected_metadata(layout: StorageLayout) -> sqlalchemy.MetaData:
    """Return SQLAlchemy metadata for all protocol-owned tables of ``layout``."""
    metadata = sqlalchemy.MetaData()
    metadata_table_for(metadata)
    for family in layout.families:
        schemas = tuple(resolve_schema(backing) for backing in family.backings)
        for schema in schemas:
            table_for(schema, metadata)
        if len(schemas) > 1:
            dispatch_table_for(family.name, tuple(zip(family.backing_names, schemas, strict=True)), metadata)
    return metadata


def metadata_table_for(metadata: sqlalchemy.MetaData) -> sqlalchemy.Table:
    """Return the reserved protocol key/value table in ``metadata``."""
    existing = metadata.tables.get(METADATA_TABLE_NAME)
    if existing is not None:
        return existing
    return sqlalchemy.Table(
        METADATA_TABLE_NAME,
        metadata,
        sqlalchemy.Column("key", sqlalchemy.Text, primary_key=True, nullable=False),
        sqlalchemy.Column("value", sqlalchemy.Text, nullable=False),
    )


def actual_schema_objects(connection: sqlalchemy.Connection) -> Mapping[str, frozenset[str]]:
    """Return application schema-object names mapped to their stable object kinds.

    The DuckDB SQLAlchemy inspector presently routes column inspection through
    a PostgreSQL catalogue relation DuckDB does not expose, so the whole
    layout path intentionally uses the dialect catalogues directly.
    """
    if connection.dialect.name == "sqlite":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
            )
        )
    elif connection.dialect.name == "duckdb":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT table_name, lower(table_type) FROM information_schema.tables "
                "WHERE table_schema = current_schema() "
                "UNION ALL "
                "SELECT sequence_name, 'sequence' FROM duckdb_sequences() WHERE schema_name = current_schema()"
            )
        )
    else:
        raise ValueError(f"SqlStore layout validation does not support dialect {connection.dialect.name!r}")
    result: dict[str, set[str]] = {}
    for name, kind in rows:
        result.setdefault(str(name), set()).add(str(kind).lower().replace("base ", ""))
    return MappingProxyType({name: frozenset(kinds) for name, kinds in result.items()})


def actual_table_names(connection: sqlalchemy.Connection) -> frozenset[str]:
    """Return application base-table names without SQLAlchemy reflection."""
    return frozenset(name for name, kinds in actual_schema_objects(connection).items() if "table" in kinds)


def read_store_metadata(connection: sqlalchemy.Connection) -> Mapping[str, str] | None:
    """Read the marker values, or return ``None`` when no metadata table exists."""
    if METADATA_TABLE_NAME not in actual_table_names(connection):
        return None
    table = metadata_table_for(sqlalchemy.MetaData())
    rows = connection.execute(sqlalchemy.select(table.c.key, table.c.value)).all()
    result: dict[str, str] = {}
    for key, value in rows:
        if not isinstance(key, str) or not isinstance(value, str) or key in result:
            raise ValueError("stored metadata rows are malformed")
        result[key] = value
    return MappingProxyType(result)


def validate_expected_tables(
    connection: sqlalchemy.Connection,
    metadata: sqlalchemy.MetaData,
    *,
    table_names: Sequence[str] | None = None,
) -> Mapping[str, object]:
    """Return a stable schema diff for expected tables absent or different on disk."""
    actual_names = actual_table_names(connection)
    expected_names = tuple(table_names) if table_names is not None else tuple(metadata.tables)
    result: dict[str, object] = {}
    for name in expected_names:
        table = metadata.tables[name]
        if name not in actual_names:
            result[name] = {"missing": True}
            continue
        expected = _expected_description(table, connection.dialect)
        actual = _actual_description(connection, name)
        difference = _description_diff(expected, actual)
        if difference:
            result[name] = difference
    return result


def _registered_family_name(family: type) -> str:
    matches: list[str] = []
    for name in known_entry_families():
        entry_family_info(name)
        if resolve_entry_family(name) is family:
            matches.append(name)
    if len(matches) != 1:
        found = ", ".join(matches) or "none"
        raise ValueError(f"entry family {family.__name__} must resolve to exactly one registered name (found {found})")
    return matches[0]


def _registered_backing_name(backing: type) -> str:
    matches: list[str] = []
    for name in known_entry_backings():
        entry_backing_info(name)
        if resolve_entry_backing(name) is backing:
            matches.append(name)
    if len(matches) != 1:
        found = ", ".join(matches) or "none"
        raise ValueError(
            f"entry backing {backing.__name__} must resolve to exactly one registered name (found {found})"
        )
    return matches[0]


def _validate_physical_names(layout: StorageLayout) -> None:
    owners: dict[str, type] = {}
    visited: set[type] = set()

    def visit(record: type) -> None:
        if record in visited:
            return
        visited.add(record)
        schema = resolve_schema(record)
        names = [schema.table_name]
        names.extend(spec.child.table_name for spec in schema.fields if spec.child is not None)
        for name in names:
            if name.startswith(_RESERVED_PREFIX):
                raise ValueError(f"record {record.__name__} claims reserved SqlStore table name {name!r}")
            previous = owners.get(name)
            if previous is not None and previous is not record:
                raise ValueError(
                    f"records {previous.__name__} and {record.__name__} collide on physical table name {name!r}"
                )
            owners[name] = record
        for target in schema.referenced_classes():
            visit(target)

    for family in layout.families:
        for backing in family.backings:
            visit(backing)
        if family.dispatch_table_name is not None:
            if family.dispatch_table_name in owners:
                raise ValueError(
                    f"entry family {family.name!r} dispatch table collides with record table {family.dispatch_table_name!r}"
                )
            owners[family.dispatch_table_name] = family.family


def _expected_description(table: sqlalchemy.Table, dialect: sqlalchemy.engine.Dialect) -> _TableDescription:
    columns = tuple(
        (column.name, _normalized_type(str(column.type.compile(dialect=dialect))), bool(column.nullable))
        for column in table.columns
    )
    primary_key = tuple(column.name for column in table.primary_key.columns)
    foreign_keys = frozenset(
        (
            tuple(element.parent.name for element in constraint.elements),
            constraint.elements[0].column.table.name,
            tuple(element.column.name for element in constraint.elements),
            _normalized_fk_action(constraint.onupdate),
            _normalized_fk_action(constraint.ondelete),
            _normalized_match(constraint.match),
            _normalized_deferrable(constraint.deferrable),
            _normalized_initially(constraint.initially),
        )
        for constraint in table.foreign_key_constraints
    )
    constraint_uniques = frozenset(
        tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, sqlalchemy.UniqueConstraint)
    )
    indexes = frozenset(
        _index_description_from_sql(
            str(sqlalchemy.schema.CreateIndex(index).compile(dialect=dialect)),
            name=index.name or "",
            unique=bool(index.unique),
        )
        for index in table.indexes
    )
    uniques = constraint_uniques | frozenset(
        tuple(term[0] for term in terms) for _name, terms, unique, predicate in indexes if unique and not predicate
    )
    checks = frozenset(
        _normalized_check(str(constraint.sqltext))
        for constraint in table.constraints
        if isinstance(constraint, sqlalchemy.CheckConstraint)
    )
    return _TableDescription(columns, primary_key, foreign_keys, uniques, indexes, checks)


def _actual_description(connection: sqlalchemy.Connection, name: str) -> _TableDescription:
    if connection.dialect.name == "sqlite":
        return _sqlite_description(connection, name)
    if connection.dialect.name == "duckdb":
        return _duckdb_description(connection, name)
    raise ValueError(f"SqlStore layout validation does not support dialect {connection.dialect.name!r}")


def _sqlite_description(connection: sqlalchemy.Connection, name: str) -> _TableDescription:
    quoted = _quote_identifier(name)
    columns_rows = connection.execute(sqlalchemy.text(f"PRAGMA table_info({quoted})")).mappings().all()
    columns = tuple(
        (
            str(row["name"]),
            _normalized_type(str(row["type"])),
            not (bool(row["notnull"]) or bool(row["pk"])),
        )
        for row in columns_rows
    )
    primary_key = tuple(str(row["name"]) for row in sorted(columns_rows, key=lambda row: int(row["pk"])) if row["pk"])
    sql = connection.execute(
        sqlalchemy.text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :name"), {"name": name}
    ).scalar_one()
    foreign_key_rows = tuple(connection.execute(sqlalchemy.text(f"PRAGMA foreign_key_list({quoted})")).mappings())
    foreign_key_options = _sqlite_foreign_key_options(str(sql))
    foreign_keys: set[tuple[tuple[str, ...], str, tuple[str, ...], str, str, str, str, str]] = set()
    grouped_foreign_keys: dict[int, list[Mapping[str, Any]]] = {}
    for foreign_key_row in foreign_key_rows:
        values = {str(key): value for key, value in foreign_key_row.items()}
        grouped_foreign_keys.setdefault(int(values["id"]), []).append(values)
    for rows in grouped_foreign_keys.values():
        ordered = sorted(rows, key=lambda item: int(item["seq"]))
        local_columns = tuple(str(item["from"]) for item in ordered)
        target_table = str(ordered[0]["table"])
        target_columns = tuple(str(item["to"]) for item in ordered)
        match, deferrable, initially = foreign_key_options.get(
            (local_columns, target_table, target_columns),
            ("none", "not deferrable", "immediate"),
        )
        foreign_keys.add(
            (
                local_columns,
                target_table,
                target_columns,
                _normalized_fk_action(str(ordered[0]["on_update"])),
                _normalized_fk_action(str(ordered[0]["on_delete"])),
                match,
                deferrable,
                initially,
            )
        )
    unique_columns: set[tuple[str, ...]] = set()
    indexes: set[tuple[str, tuple[tuple[str, str, str], ...], bool, str]] = set()
    for row in connection.execute(sqlalchemy.text(f"PRAGMA index_list({quoted})")).mappings():
        index_name = str(row["name"])
        index_sql = connection.execute(
            sqlalchemy.text("SELECT sql FROM sqlite_master WHERE type = 'index' AND name = :name"),
            {"name": index_name},
        ).scalar_one_or_none()
        parsed = _index_description_from_sql(
            str(index_sql or ""),
            name=index_name,
            unique=bool(row["unique"]),
        )
        parsed_terms = parsed[1]
        xinfo = tuple(
            item
            for item in connection.execute(
                sqlalchemy.text(f"PRAGMA index_xinfo({_quote_identifier(index_name)})")
            ).mappings()
            if bool(item["key"])
        )
        index_terms = tuple(
            (
                str(item["name"]) if item["name"] is not None else parsed_terms[position][0],
                "desc" if bool(item["desc"]) else "asc",
                _normalized_collation(item["coll"]),
            )
            for position, item in enumerate(xinfo)
        )
        unique = bool(row["unique"])
        predicate = parsed[3]
        if unique and str(row["origin"]) != "pk" and not bool(row["partial"]):
            unique_columns.add(tuple(term[0] for term in index_terms))
        # SQLite labels named CREATE INDEX indexes as origin "c"; unnamed
        # UNIQUE constraints are implementation indexes and have no stable
        # expected name to compare.
        if str(row["origin"]) == "c":
            indexes.add((index_name, index_terms, unique, predicate))
    checks = frozenset(_normalized_check(expression) for expression in _sqlite_checks(str(sql)))
    return _TableDescription(
        columns, primary_key, frozenset(foreign_keys), frozenset(unique_columns), frozenset(indexes), checks
    )


def _duckdb_description(connection: sqlalchemy.Connection, name: str) -> _TableDescription:
    parameters = {"name": name}
    columns_rows = (
        connection.execute(
            sqlalchemy.text(
                "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = :name ORDER BY ordinal_position"
            ),
            parameters,
        )
        .mappings()
        .all()
    )
    columns = tuple(
        (str(row["column_name"]), _normalized_type(str(row["data_type"])), str(row["is_nullable"]) == "YES")
        for row in columns_rows
    )
    constraints = (
        connection.execute(
            sqlalchemy.text(
                "SELECT constraint_name, constraint_type, constraint_text, expression, constraint_column_names, "
                "referenced_table, referenced_column_names FROM duckdb_constraints() "
                "WHERE schema_name = current_schema() AND table_name = :name"
            ),
            parameters,
        )
        .mappings()
        .all()
    )
    foreign_key_attributes = {
        str(row["constraint_name"]): row
        for row in connection.execute(
            sqlalchemy.text(
                "SELECT tc.constraint_name, tc.is_deferrable, tc.initially_deferred, "
                "rc.match_option, rc.update_rule, rc.delete_rule "
                "FROM information_schema.table_constraints AS tc "
                "LEFT JOIN information_schema.referential_constraints AS rc "
                "ON rc.constraint_catalog = tc.constraint_catalog "
                "AND rc.constraint_schema = tc.constraint_schema "
                "AND rc.constraint_name = tc.constraint_name "
                "WHERE tc.table_schema = current_schema() AND tc.table_name = :name "
                "AND tc.constraint_type = 'FOREIGN KEY'"
            ),
            parameters,
        ).mappings()
    }
    primary_key: tuple[str, ...] = ()
    foreign_keys: set[tuple[tuple[str, ...], str, tuple[str, ...], str, str, str, str, str]] = set()
    uniques: set[tuple[str, ...]] = set()
    checks: set[str] = set()
    for row in constraints:
        kind = str(row["constraint_type"]).upper()
        column_names = tuple(str(item) for item in (row["constraint_column_names"] or ()))
        if kind == "PRIMARY KEY":
            primary_key = column_names
        elif kind == "UNIQUE":
            uniques.add(column_names)
        elif kind == "FOREIGN KEY":
            referenced = tuple(str(item) for item in (row["referenced_column_names"] or ()))
            if len(column_names) != len(referenced):
                raise ValueError(f"DuckDB reported malformed foreign key metadata for {name!r}")
            attributes = foreign_key_attributes.get(str(row["constraint_name"]))
            if attributes is None:
                raise ValueError(f"DuckDB omitted referential-constraint metadata for {name!r}")
            foreign_keys.add(
                (
                    column_names,
                    str(row["referenced_table"]),
                    referenced,
                    _normalized_fk_action(attributes["update_rule"]),
                    _normalized_fk_action(attributes["delete_rule"]),
                    _normalized_match(attributes["match_option"]),
                    _normalized_deferrable(attributes["is_deferrable"]),
                    _normalized_initially(attributes["initially_deferred"]),
                )
            )
        elif kind == "CHECK":
            expression = row["expression"] or row["constraint_text"]
            checks.add(_normalized_check(str(expression)))
    indexes: set[tuple[str, tuple[tuple[str, str, str], ...], bool, str]] = set()
    index_rows = connection.execute(
        sqlalchemy.text(
            "SELECT index_name, is_unique, sql FROM duckdb_indexes() "
            "WHERE schema_name = current_schema() AND table_name = :name"
        ),
        parameters,
    ).mappings()
    for row in index_rows:
        index = _index_description_from_sql(
            str(row["sql"]),
            name=str(row["index_name"]),
            unique=bool(row["is_unique"]),
        )
        indexes.add(index)
        if row["is_unique"] and not index[3]:
            uniques.add(tuple(term[0] for term in index[1]))
    return _TableDescription(
        columns, primary_key, frozenset(foreign_keys), frozenset(uniques), frozenset(indexes), frozenset(checks)
    )


def _description_diff(expected: _TableDescription, actual: _TableDescription) -> dict[str, object]:
    result: dict[str, object] = {}
    expected_names = tuple(column[0] for column in expected.columns)
    actual_names = tuple(column[0] for column in actual.columns)
    if expected_names != actual_names:
        result["column_order"] = {"expected": expected_names, "actual": actual_names}
    expected_by_name = {column[0]: column[1:] for column in expected.columns}
    actual_by_name = {column[0]: column[1:] for column in actual.columns}
    column_difference = {
        name: {"expected": expected_by_name.get(name), "actual": actual_by_name.get(name)}
        for name in sorted(set(expected_by_name) | set(actual_by_name))
        if expected_by_name.get(name) != actual_by_name.get(name)
    }
    if column_difference:
        result["columns"] = column_difference
    for key, expected_value, actual_value in (
        ("primary_key", expected.primary_key, actual.primary_key),
        ("foreign_keys", tuple(sorted(expected.foreign_keys)), tuple(sorted(actual.foreign_keys))),
        ("unique", tuple(sorted(expected.uniques)), tuple(sorted(actual.uniques))),
        ("indexes", tuple(sorted(expected.indexes)), tuple(sorted(actual.indexes))),
        ("checks", tuple(sorted(expected.checks)), tuple(sorted(actual.checks))),
    ):
        if expected_value != actual_value:
            result[key] = {"expected": expected_value, "actual": actual_value}
    return result


def _normalized_type(value: str) -> str:
    upper = re.sub(r"\s+", " ", value.strip().upper())
    if any(token in upper for token in ("INT", "SERIAL")):
        return "integer"
    if any(token in upper for token in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC")):
        return "float"
    if any(token in upper for token in ("CHAR", "TEXT", "STRING", "VARCHAR")):
        return "text"
    if "BOOL" in upper:
        return "bool"
    if any(token in upper for token in ("BLOB", "BINARY", "BYTEA")):
        return "bytes"
    return upper.lower()


def _normalized_check(expression: str) -> str:
    tokens = list(_sql_tokens(expression.strip()))
    if tokens and _is_keyword(tokens[0], "CHECK"):
        opening = next((index for index, token in enumerate(tokens[1:], 1) if token.text == "("), None)
        if opening is not None:
            closing = _matching_parenthesis(tokens, opening)
            if closing is not None:
                tokens = tokens[opening : closing + 1]
    tokens = list(_without_outer_parentheses(tokens))
    # DuckDB adds parentheses around atomic CASE tests/results. Remove only
    # those dialect decorations: parentheses around boolean/arithmetic groups
    # remain significant and therefore continue to produce a schema diff.
    while True:
        normalized = _without_outer_parentheses(_without_redundant_atomic_parentheses(tokens))
        if tuple(tokens) == normalized:
            return "".join(_normalized_sql_token(token) for token in tokens)
        tokens = list(normalized)


def _without_outer_parentheses(tokens: Sequence[_SqlToken]) -> tuple[_SqlToken, ...]:
    result = tuple(tokens)
    while result and result[0].text == "(" and _matching_parenthesis(result, 0) == len(result) - 1:
        result = result[1:-1]
    return result


def _without_redundant_atomic_parentheses(tokens: Sequence[_SqlToken]) -> tuple[_SqlToken, ...]:
    removed: set[int] = set()
    for opening, token in enumerate(tokens):
        if token.text != "(":
            continue
        closing = _matching_parenthesis(tokens, opening)
        if closing is None:
            continue
        inner = tokens[opening + 1 : closing]
        numeric = (
            len(inner) == 1 and inner[0].kind == "word" and re.fullmatch(r"\d+(?:\.\d+)?", inner[0].text) is not None
        ) or (
            len(inner) == 2
            and inner[0].text in {"+", "-"}
            and inner[1].kind == "word"
            and re.fullmatch(r"\d+(?:\.\d+)?", inner[1].text) is not None
        )
        null_test = (
            len(inner) in {3, 4}
            and inner[0].kind in {"word", "identifier"}
            and _is_keyword(inner[1], "IS")
            and _is_keyword(inner[-1], "NULL")
            and (len(inner) == 3 or _is_keyword(inner[2], "NOT"))
        )
        if numeric or null_test:
            removed.update((opening, closing))
    return tuple(token for index, token in enumerate(tokens) if index not in removed)


def _sqlite_checks(sql: str) -> tuple[str, ...]:
    values: list[str] = []
    tokens = _sql_tokens(sql)
    position = 0
    while position < len(tokens):
        if not _is_keyword(tokens[position], "CHECK"):
            position += 1
            continue
        opening = position + 1
        if opening >= len(tokens) or tokens[opening].text != "(":
            position += 1
            continue
        closing = _matching_parenthesis(tokens, opening)
        if closing is None:
            break
        values.append(sql[tokens[opening].start : tokens[closing].end])
        position = closing + 1
    return tuple(values)


def _normalized_fk_action(value: object | None) -> str:
    if value is None:
        return "no action"
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _normalized_match(value: object | None) -> str:
    if value is None:
        return "none"
    return str(value).strip().lower()


def _normalized_deferrable(value: object | None) -> str:
    if value is True or str(value).strip().upper() == "YES":
        return "deferrable"
    return "not deferrable"


def _normalized_initially(value: object | None) -> str:
    if str(value).strip().upper() in {"DEFERRED", "YES"}:
        return "deferred"
    return "immediate"


def _normalized_collation(value: object | None) -> str:
    if value is None:
        return "binary"
    return str(value).strip().strip('"`').lower()


def _index_description_from_sql(
    sql: str,
    *,
    name: str,
    unique: bool,
) -> tuple[str, tuple[tuple[str, str, str], ...], bool, str]:
    tokens = _sql_tokens(sql)
    opening = next((index for index, token in enumerate(tokens) if token.text == "("), None)
    if opening is None:
        return (name, (), unique, "")
    closing = _matching_parenthesis(tokens, opening)
    if closing is None:
        return (name, (), unique, "")
    terms_sql = sql[tokens[opening].end : tokens[closing].start]
    terms = tuple(_normalized_index_term(term) for term in _split_index_terms(terms_sql))
    remainder = sql[tokens[closing].end :].strip().rstrip(";").strip()
    predicate = ""
    remainder_tokens = _sql_tokens(remainder)
    if remainder_tokens and _is_keyword(remainder_tokens[0], "WHERE"):
        predicate = _normalized_check(remainder[remainder_tokens[0].end :])
    return (name, terms, unique, predicate)


def _split_index_terms(value: str) -> tuple[str, ...]:
    terms: list[str] = []
    start = 0
    depth = 0
    for token in _sql_tokens(value):
        if token.text == "(":
            depth += 1
        elif token.text == ")":
            depth -= 1
        elif token.text == "," and depth == 0:
            terms.append(value[start : token.start])
            start = token.end
    terms.append(value[start:])
    return tuple(terms)


def _normalized_index_term(value: str) -> tuple[str, str, str]:
    tokens = list(_sql_tokens(value.strip()))
    direction = "asc"
    if tokens and (_is_keyword(tokens[-1], "ASC") or _is_keyword(tokens[-1], "DESC")):
        direction = tokens.pop().text.lower()
    collation = "binary"
    if len(tokens) >= 2 and _is_keyword(tokens[-2], "COLLATE"):
        collation = _normalized_collation(tokens[-1].value)
        del tokens[-2:]
    expression = "".join(_normalized_sql_token(token) for token in tokens)
    return (expression, direction, collation)


def _sqlite_foreign_key_options(
    sql: str,
) -> Mapping[tuple[tuple[str, ...], str, tuple[str, ...]], tuple[str, str, str]]:
    """Recover FK options SQLite's PRAGMA omits from the stored CREATE statement."""
    tokens = _sql_tokens(sql)
    result: dict[tuple[tuple[str, ...], str, tuple[str, ...]], tuple[str, str, str]] = {}
    parsed_references: set[int] = set()
    position = 0
    while position + 2 < len(tokens):
        if not (_is_keyword(tokens[position], "FOREIGN") and _is_keyword(tokens[position + 1], "KEY")):
            position += 1
            continue
        local_opening = position + 2
        if tokens[local_opening].text != "(":
            position += 1
            continue
        local_closing = _matching_parenthesis(tokens, local_opening)
        if local_closing is None:
            break
        local_columns = _identifier_list(tokens[local_opening + 1 : local_closing])
        references = local_closing + 1
        if references >= len(tokens) or not _is_keyword(tokens[references], "REFERENCES"):
            position = local_closing + 1
            continue
        tail = _sqlite_foreign_key_tail(tokens, references)
        if tail is None:
            position = local_closing + 1
            continue
        target_table, target_columns, match, deferrable, initially, target_closing = tail
        result[(local_columns, target_table, target_columns)] = (match, deferrable, initially)
        parsed_references.add(references)
        position = target_closing + 1

    depths: list[int] = []
    depth = 0
    for token in tokens:
        depths.append(depth)
        if token.text == "(":
            depth += 1
        elif token.text == ")":
            depth -= 1
    for references, token in enumerate(tokens):
        if references in parsed_references or not _is_keyword(token, "REFERENCES"):
            continue
        reference_depth = depths[references]
        segment_start = 0
        for previous in range(references - 1, -1, -1):
            if tokens[previous].text == "," and depths[previous] == reference_depth:
                segment_start = previous + 1
                break
            if tokens[previous].text == "(" and depths[previous] == reference_depth - 1:
                segment_start = previous + 1
                break
        segment = tokens[segment_start:references]
        if any(_is_keyword(member, "FOREIGN") for member in segment):
            continue
        local = next((member.value for member in segment if member.kind in {"word", "identifier"}), None)
        tail = _sqlite_foreign_key_tail(tokens, references)
        if local is None or tail is None:
            continue
        target_table, target_columns, match, deferrable, initially, _target_closing = tail
        result[((local,), target_table, target_columns)] = (match, deferrable, initially)
    return MappingProxyType(result)


def _sqlite_foreign_key_tail(
    tokens: Sequence[_SqlToken], references: int
) -> tuple[str, tuple[str, ...], str, str, str, int] | None:
    target_position = references + 1
    target_parts: list[str] = []
    while target_position < len(tokens) and tokens[target_position].text != "(":
        token = tokens[target_position]
        if token.kind in {"word", "identifier"}:
            target_parts.append(token.value)
        target_position += 1
    if not target_parts or target_position >= len(tokens):
        return None
    target_table = target_parts[-1]
    target_closing = _matching_parenthesis(tokens, target_position)
    if target_closing is None:
        return None
    target_columns = _identifier_list(tokens[target_position + 1 : target_closing])
    match = "none"
    deferrable = "not deferrable"
    initially = "immediate"
    option = target_closing + 1
    while option < len(tokens) and tokens[option].text not in {",", ")"}:
        if _is_keyword(tokens[option], "MATCH") and option + 1 < len(tokens):
            match = _normalized_match(tokens[option + 1].value)
            option += 2
            continue
        if (
            _is_keyword(tokens[option], "NOT")
            and option + 1 < len(tokens)
            and _is_keyword(tokens[option + 1], "DEFERRABLE")
        ):
            deferrable = "not deferrable"
            option += 2
            continue
        if _is_keyword(tokens[option], "DEFERRABLE"):
            deferrable = "deferrable"
            option += 1
            continue
        if _is_keyword(tokens[option], "INITIALLY") and option + 1 < len(tokens):
            initially = _normalized_initially(tokens[option + 1].value)
            option += 2
            continue
        option += 1
    return target_table, target_columns, match, deferrable, initially, target_closing


def _identifier_list(tokens: Sequence[_SqlToken]) -> tuple[str, ...]:
    return tuple(token.value for token in tokens if token.kind in {"word", "identifier"})


def _sql_tokens(sql: str) -> tuple[_SqlToken, ...]:
    """Tokenize enough SQL DDL to distinguish syntax from quoted contents."""
    tokens: list[_SqlToken] = []
    position = 0
    while position < len(sql):
        character = sql[position]
        if character.isspace():
            position += 1
            continue
        if sql.startswith("--", position):
            newline = sql.find("\n", position + 2)
            position = len(sql) if newline < 0 else newline + 1
            continue
        if sql.startswith("/*", position):
            closing = sql.find("*/", position + 2)
            position = len(sql) if closing < 0 else closing + 2
            continue
        if character in {'"', "'", "`"}:
            start = position
            quote = character
            position += 1
            value: list[str] = []
            while position < len(sql):
                if sql[position] == quote:
                    if position + 1 < len(sql) and sql[position + 1] == quote:
                        value.append(quote)
                        position += 2
                        continue
                    position += 1
                    break
                value.append(sql[position])
                position += 1
            kind = "string" if quote == "'" else "identifier"
            tokens.append(_SqlToken(kind, sql[start:position], "".join(value), start, position))
            continue
        if character == "[":
            start = position
            position += 1
            value = []
            while position < len(sql):
                if sql[position] == "]":
                    if position + 1 < len(sql) and sql[position + 1] == "]":
                        value.append("]")
                        position += 2
                        continue
                    position += 1
                    break
                value.append(sql[position])
                position += 1
            tokens.append(_SqlToken("identifier", sql[start:position], "".join(value), start, position))
            continue
        if character.isalnum() or character in {"_", "$"}:
            start = position
            position += 1
            while position < len(sql) and (sql[position].isalnum() or sql[position] in {"_", "$"}):
                position += 1
            text = sql[start:position]
            tokens.append(_SqlToken("word", text, text, start, position))
            continue
        tokens.append(_SqlToken("symbol", character, character, position, position + 1))
        position += 1
    return tuple(tokens)


def _is_keyword(token: _SqlToken, keyword: str) -> bool:
    return token.kind == "word" and token.text.upper() == keyword


def _matching_parenthesis(tokens: Sequence[_SqlToken], opening: int) -> int | None:
    if opening >= len(tokens) or tokens[opening].text != "(":
        return None
    depth = 0
    for position in range(opening, len(tokens)):
        if tokens[position].text == "(":
            depth += 1
        elif tokens[position].text == ")":
            depth -= 1
            if depth == 0:
                return position
    return None


def _normalized_sql_token(token: _SqlToken) -> str:
    if token.kind == "word":
        return token.text.lower()
    if token.kind == "identifier":
        value = token.value.lower()
        if re.fullmatch(r"[a-z_][a-z0-9_$]*", value):
            return value
        return '"' + value.replace('"', '""') + '"'
    return token.text


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(member) for key, member in item.items()})
        if isinstance(item, list):
            return tuple(freeze(member) for member in item)
        if isinstance(item, tuple):
            return tuple(freeze(member) for member in item)
        if isinstance(item, set | frozenset):
            return tuple(sorted((freeze(member) for member in item), key=repr))
        return item

    return MappingProxyType({str(key): freeze(member) for key, member in value.items()})
