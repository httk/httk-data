"""Serve stored dataclasses through the httk-core entry-provider contract.

:class:`StoreEntryProvider` bridges the SQL storage layer to the neutral
:class:`~httk.core.EntryProvider` contract: it serves the rows of one or more
storable classes in a :class:`~httk.data.db.store.SqlStore` as described,
JSON-able entry-type records, so a serving module (such as *httk-optimade*)
can expose a database as an OPTIMADE API without either side depending on the
other.

For each served class the provider either passes through a supplied
:class:`~httk.core.EntryTypeDefinition` (validated to describe every served
property) or auto-generates one from the class's resolved
:class:`~httk.data.db.schema.TableSchema`: the OPTIMADE core ``id``/``type``
properties plus one :meth:`~httk.core.PropertyDefinition.from_simple`
definition per servable stored field, each named with a registered
database-specific ``prefix`` (default ``"_httk_"``) and merged in via
:meth:`~httk.core.EntryTypeDefinition.extended` — the same construction route
the other httk entry providers use.

The schema-to-OPTIMADE type mapping is:

- ``str``/``int``/``bool``/``float`` fields — ``string``/``integer``/
  ``boolean``/``float``;
- rational fields (:class:`fractions.Fraction`, :class:`~httk.core.FracScalar`,
  :class:`~httk.core.SurdScalar`) — ``float``, served as the nearest float
  (stored values themselves remain exact; only the *served* value is
  approximate);
- :class:`datetime.datetime` fields — ``timestamp``, served as ISO-8601 text;
- fixed-shape (``Shape(r, c)``) and variable-rows (``Shape(0, c)``)
  :class:`~httk.core.FracVector` fields — ``list of list of float`` (the fixed
  shape also declares its dimension sizes);
- ``list``/``tuple`` fields of scalars or of the codec types above — ``list
  of`` the mapped element type.

Not every stored field can be served as a property: ``bytes`` fields (and
fields encoded by a custom, non-built-in value codec) have no OPTIMADE value
representation and are skipped, while reference fields and child fields of
storable elements surface through :meth:`StoreEntryProvider.relationships`
instead — when their target class is itself served, each record declares its
related entries as a flat tuple of :class:`~httk.core.RelatedEntry` values,
carrying the ``role``/``description`` metadata of an optional
:class:`~httk.core.Related` field marker (``Related(serve=False)`` suppresses
the field as a relationship). Class-level
:class:`~httk.core.RelationshipLink` declarations contribute further
relationships: each stored row of the declaring class — a served class, or an
unserved join class passed via ``link_classes`` — expresses one FROM→TO
relationship between the entries its link endpoints resolve to, carrying the
link's metadata.
"""

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Final

import sqlalchemy
from httk.core import (
    EntryProvider,
    EntryTypeDefinition,
    FracVector,
    PropertyDefinition,
    RelatedEntry,
    RelationshipLink,
    known_definition_prefixes,
)

from httk.data.db.codecs import codec_named
from httk.data.db.mapping import SID_COLUMN
from httk.data.db.schema import FieldSpec, TableSchema, resolve_schema
from httk.data.db.searcher import SqlColumn, _query_index
from httk.data.db.store import SqlStore, _as_fixed_tensor

__all__ = [
    "StoreEntryProvider",
    "served_specs",
    "auto_definition",
]

#: How a scalar column kind maps onto an OPTIMADE fulltype (bytes has none).
_SCALAR_FULLTYPES: Final[dict[str, str]] = {
    "str": "string",
    "int": "integer",
    "bool": "boolean",
    "float": "float",
}

#: How a built-in value codec maps onto an OPTIMADE fulltype (custom codecs have none).
_CODEC_FULLTYPES: Final[dict[str, str]] = {
    "fraction": "float",
    "fracscalar": "float",
    "surdscalar": "float",
    "datetime": "timestamp",
}


def _default_id(entry_type: str, sid: int, obj: Any) -> str:
    return f"{entry_type}-{sid}"


@dataclasses.dataclass(frozen=True)
class _LinkScan:
    """One relationship-link scan: which table to read and how to interpret its rows.

    Each stored row of ``declaring`` expresses one FROM→TO relationship: the
    FROM-side sid is read from ``source_column`` and the TO-side sid from
    ``target_column`` (a reference field's foreign-key column, or the declaring
    table's own sid column for a ``None`` endpoint).
    """

    declaring: type
    schema: TableSchema
    link: RelationshipLink
    source_column: str
    target_column: str
    from_cls: type
    to_cls: type
    to_type: str


def _fulltype_of(spec: FieldSpec) -> str | None:
    """The OPTIMADE fulltype a field is served as, or None when it is not servable."""
    if spec.role == "scalar":
        return _SCALAR_FULLTYPES.get(spec.columns[0].kind)
    if spec.role == "encoded":
        assert spec.codec_name is not None
        return _CODEC_FULLTYPES.get(spec.codec_name)
    if spec.role == "fixed_array":
        return "list of list of float"
    if spec.role == "child":
        if spec.shape is not None:
            return "list of list of float"
        if spec.target is not None:
            return None  # storable elements surface through relationships()
        if spec.codec_name is not None:
            element = _CODEC_FULLTYPES.get(spec.codec_name)
        else:
            element = _SCALAR_FULLTYPES.get(spec.child.element_columns[0].kind) if spec.child is not None else None
        return None if element is None else f"list of {element}"
    return None  # reference: surfaces through relationships()


def served_specs(schema: TableSchema, prefix: str) -> list[tuple[str, FieldSpec, str]]:
    """The served ``(property name, field spec, fulltype)`` triples of a storable class.

    One triple per servable stored field of ``schema`` (see the module
    docstring for which fields are servable and how their types map), each
    named ``{prefix}{field}``.
    """
    served: list[tuple[str, FieldSpec, str]] = []
    for spec in schema.fields:
        fulltype = _fulltype_of(spec)
        if fulltype is not None:
            served.append((f"{prefix}{spec.field}", spec, fulltype))
    return served


def auto_definition(entry_type: str, schema: TableSchema, prefix: str) -> EntryTypeDefinition:
    """Auto-generate the :class:`~httk.core.EntryTypeDefinition` of a storable class.

    The definition carries the OPTIMADE core ``id``/``type`` properties plus
    one :meth:`~httk.core.PropertyDefinition.from_simple` definition per triple
    of :func:`served_specs`, merged in via
    :meth:`~httk.core.EntryTypeDefinition.extended` (so ``prefix`` must be a
    registered definition prefix).
    """
    cls = schema.cls
    base = EntryTypeDefinition(
        entry_type,
        f"The '{entry_type}' entry type, generated from the stored class {cls.__name__}.",
        {
            "id": PropertyDefinition.from_simple("id", description="The unique entry id.", required_response=True),
            "type": PropertyDefinition.from_simple(
                "type", description="The name of the entry type.", required_response=True
            ),
        },
    )
    extra: dict[str, PropertyDefinition] = {}
    for name, spec, fulltype in served_specs(schema, prefix):
        kind = "stored property" if spec.derived else "stored field"
        dimensions: dict[str, Any] | None = None
        if spec.role == "fixed_array":
            assert spec.shape is not None
            dimensions = {"names": ["rows", "cols"], "sizes": [spec.shape.rows, spec.shape.cols]}
        extra[name] = PropertyDefinition.from_simple(
            name,
            description=f"The {kind} '{spec.field}' of {cls.__name__}.",
            fulltype=fulltype,
            dimensions=dimensions,
        )
    return base.extended(extra)


class StoreEntryProvider(EntryProvider):
    """Serves the stored rows of storable classes as httk-core entry types.

    ``classes`` maps each served entry-type name to its storable dataclass;
    the classes' tables are read through ``store``. ``definitions`` optionally
    supplies the :class:`~httk.core.EntryTypeDefinition` of an entry type
    (validated: it must describe every property the provider serves for it);
    entry types without a supplied definition get one auto-generated from the
    class's schema, with every schema-derived property name carrying
    ``prefix`` (which must be registered, see
    :func:`~httk.core.register_definition_prefix`). ``id_of`` maps
    ``(entry_type, sid, instance)`` to the served entry id; the default is
    ``"<entry_type>-<sid>"``. ``link_classes`` names storable *join-object*
    classes that are not served as entries themselves but whose
    :class:`~httk.core.RelationshipLink` declarations contribute relationships
    between served entries; every link endpoint (on a served class or a link
    class) must resolve to a served entry type, and a link class without link
    declarations is rejected.

    See the module docstring for which stored fields are served as properties
    (and how their types map), which are skipped, and which surface through
    :meth:`relationships` instead.
    """

    def __init__(
        self,
        store: SqlStore,
        classes: Mapping[str, type],
        *,
        definitions: Mapping[str, EntryTypeDefinition] | None = None,
        prefix: str = "_httk_",
        id_of: Callable[[str, int, Any], str] | None = None,
        link_classes: Iterable[type] = (),
    ) -> None:
        if prefix not in known_definition_prefixes():
            raise ValueError(
                f"the property-name prefix {prefix!r} is not registered; register it with "
                f"httk.core.register_definition_prefix() (registered prefixes: "
                f"{', '.join(known_definition_prefixes())})"
            )
        self._store = store
        self._classes: dict[str, type] = dict(classes)
        self._prefix = prefix
        self._id_of: Callable[[str, int, Any], str] = id_of if id_of is not None else _default_id
        self._entry_type_of: dict[type, str] = {cls: name for name, cls in self._classes.items()}
        self._schemas: dict[str, TableSchema] = {name: resolve_schema(cls) for name, cls in self._classes.items()}
        self._links_by_from: dict[str, list[_LinkScan]] = self._build_link_inventory(tuple(link_classes))
        self._definitions: dict[str, EntryTypeDefinition] = dict(definitions or {})
        unknown = sorted(name for name in self._definitions if name not in self._classes)
        if unknown:
            raise ValueError(
                f"definitions were supplied for entry types this provider does not serve: {', '.join(unknown)}"
            )
        for entry_type, definition in self._definitions.items():
            described = definition.properties
            missing = sorted(name for name in self.property_keys(entry_type) if name not in described)
            if missing:
                raise ValueError(
                    f"the supplied definition for entry type {entry_type!r} does not describe the "
                    f"served propert{'y' if len(missing) == 1 else 'ies'}: {', '.join(missing)}"
                )

    def _build_link_inventory(self, link_classes: tuple[type, ...]) -> dict[str, list[_LinkScan]]:
        """Collect every RelationshipLink of the served and link classes, indexed by FROM-side entry type.

        Both endpoint classes of every link must be served; a link class whose
        schema declares no links is a likely user error and rejected.
        """
        served = set(self._classes.values())
        inventory: dict[str, list[_LinkScan]] = {}
        seen: set[type] = set()
        for declaring in (*self._classes.values(), *link_classes):
            if declaring in seen:
                continue
            seen.add(declaring)
            schema = resolve_schema(declaring)
            if declaring not in served and not schema.links:
                raise ValueError(
                    f"link class {declaring.__name__} declares no relationship links (StorageInfo.links "
                    f"is empty); remove it from link_classes or declare its links"
                )
            for link in schema.links:
                from_cls = schema.field(link.source).target if link.source is not None else declaring
                to_cls = schema.field(link.target).target if link.target is not None else declaring
                assert from_cls is not None and to_cls is not None  # link endpoints are reference fields
                from_type = self._entry_type_of.get(from_cls)
                to_type = self._entry_type_of.get(to_cls)
                if from_type is None or to_type is None:
                    missing = from_cls if from_type is None else to_cls
                    side = "FROM" if from_type is None else "TO"
                    raise ValueError(
                        f"RelationshipLink({link.source!r}, {link.target!r}) on {declaring.__name__}: "
                        f"the {side}-side class {missing.__name__} is not served by this provider; every "
                        f"link endpoint must resolve to a served entry type"
                    )
                inventory.setdefault(from_type, []).append(
                    _LinkScan(
                        declaring=declaring,
                        schema=schema,
                        link=link,
                        source_column=(
                            schema.field(link.source).columns[0].name if link.source is not None else SID_COLUMN
                        ),
                        target_column=(
                            schema.field(link.target).columns[0].name if link.target is not None else SID_COLUMN
                        ),
                        from_cls=from_cls,
                        to_cls=to_cls,
                        to_type=to_type,
                    )
                )
        return inventory

    # ------------------------------------------------------------------ the served subset

    def _require_entry_type(self, entry_type: str) -> type:
        try:
            return self._classes[entry_type]
        except KeyError:
            raise KeyError(
                f"StoreEntryProvider serves only the entry type(s): {', '.join(sorted(self._classes))}"
            ) from None

    def _served_specs(self, entry_type: str) -> list[tuple[str, FieldSpec, str]]:
        """The served ``(property name, field spec, fulltype)`` triples of ``entry_type``."""
        return served_specs(self._schemas[entry_type], self._prefix)

    def _relationship_specs(self, entry_type: str) -> list[tuple[FieldSpec, str]]:
        """The ``(field spec, related entry type)`` pairs whose target class is also served.

        Fields suppressed with ``Related(serve=False)`` are excluded.
        """
        pairs: list[tuple[FieldSpec, str]] = []
        for spec in self._schemas[entry_type].fields:
            if spec.role not in ("reference", "child") or spec.target is None:
                continue
            if spec.related is not None and not spec.related.serve:
                continue
            related = self._entry_type_of.get(spec.target)
            if related is not None:
                pairs.append((spec, related))
        return pairs

    # ------------------------------------------------------------------ the provider contract

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {
            entry_type: (
                self._definitions[entry_type] if entry_type in self._definitions else self._auto_definition(entry_type)
            )
            for entry_type in self._classes
        }

    def _auto_definition(self, entry_type: str) -> EntryTypeDefinition:
        return auto_definition(entry_type, self._schemas[entry_type], self._prefix)

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        self._require_entry_type(entry_type)
        property_keys = {"id": "__id", "type": "type"}
        property_keys.update({name: name for name, _spec, _fulltype in self._served_specs(entry_type)})
        return property_keys

    def records(self, entry_type: str) -> Iterator[Mapping[str, Any]]:
        cls = self._require_entry_type(entry_type)
        served = self._served_specs(entry_type)
        schema = self._schemas[entry_type]
        searcher = self._store.searcher()
        variable = searcher.variable(cls)
        searcher.output(variable, "obj")
        searcher.output(SqlColumn(searcher, variable._alias.c[SID_COLUMN]), "sid")
        for (obj, sid), _names in searcher:
            row: dict[str, Any] = {"__id": self._id_of(entry_type, int(sid), obj), "type": entry_type}
            for name, spec, _fulltype in served:
                row[name] = _json_value(schema, spec, getattr(obj, spec.field))
            yield row

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        cls = self._require_entry_type(entry_type)
        relation_specs = self._relationship_specs(entry_type)
        link_scans = self._links_by_from.get(entry_type, [])
        if not relation_specs and not link_scans:
            return {}
        store = self._store
        store.ensure_tables(cls, *(scan.declaring for scan in link_scans))
        schema = self._schemas[entry_type]
        table = store._table(schema.table_name)
        reference_specs = [(spec, related) for spec, related in relation_specs if spec.role == "reference"]
        child_specs = [(spec, related) for spec, related in relation_specs if spec.role == "child"]
        # With the default id function in use, ids depend only on (entry type,
        # sid) and per-object fetches would be pure overhead.
        fast_ids = self._id_of is _default_id
        result: dict[str, tuple[RelatedEntry, ...]] = {}
        with store._read_connection() as connection:
            # The related (target class, target sid, related entry type,
            # description, role) tuples per record sid: forward reference fields
            # first (in field order), then child fields (by row order), then
            # link-derived relationships (link declaration order, row-sid order).
            related_by_sid: dict[int, list[tuple[type, int, str, str | None, str | None]]] = {}
            if reference_specs:
                columns = [table.c[SID_COLUMN]] + [table.c[spec.columns[0].name] for spec, _related in reference_specs]
                for row in connection.execute(sqlalchemy.select(*columns).order_by(table.c[SID_COLUMN])):
                    sid = int(row[0])
                    for (spec, related), value in zip(reference_specs, row[1:], strict=True):
                        if value is not None:
                            assert spec.target is not None
                            marker = spec.related
                            related_by_sid.setdefault(sid, []).append(
                                (
                                    spec.target,
                                    int(value),
                                    related,
                                    marker.description if marker is not None else None,
                                    marker.role if marker is not None else None,
                                )
                            )
            for spec, related in child_specs:
                assert spec.child is not None and spec.target is not None
                child_table = store._table(spec.child.table_name)
                parent_column = child_table.c[f"{schema.table_name}_sid"]
                element_column = child_table.c[spec.child.element_columns[0].name]
                statement = sqlalchemy.select(parent_column, element_column).order_by(
                    parent_column, child_table.c[f"{spec.field}_index"]
                )
                marker = spec.related
                for parent_sid, element_sid in connection.execute(statement):
                    related_by_sid.setdefault(int(parent_sid), []).append(
                        (
                            spec.target,
                            int(element_sid),
                            related,
                            marker.description if marker is not None else None,
                            marker.role if marker is not None else None,
                        )
                    )
            for scan in link_scans:
                link_table = store._table(scan.schema.table_name)
                statement = sqlalchemy.select(
                    link_table.c[scan.source_column], link_table.c[scan.target_column]
                ).order_by(link_table.c[SID_COLUMN])
                for from_sid, to_sid in connection.execute(statement):
                    if from_sid is None or to_sid is None:
                        continue
                    related_by_sid.setdefault(int(from_sid), []).append(
                        (scan.to_cls, int(to_sid), scan.to_type, scan.link.description, scan.link.role)
                    )
            for sid in sorted(related_by_sid):
                record_id = self._id_of(entry_type, sid, None if fast_ids else store._fetch(connection, cls, sid))
                entries = [
                    RelatedEntry(
                        related,
                        self._id_of(
                            related,
                            target_sid,
                            None if fast_ids else store._fetch(connection, target_cls, target_sid),
                        ),
                        description=description,
                        role=role,
                    )
                    for target_cls, target_sid, related, description, role in related_by_sid[sid]
                ]
                # Dedup by exact RelatedEntry equality, preserving first
                # occurrence; entries differing only in metadata are both kept.
                result[record_id] = tuple(dict.fromkeys(entries))
        return result


def _json_value(schema: TableSchema, spec: FieldSpec, value: Any) -> Any:
    """The JSON-able served form of one stored field value (see the module docstring)."""
    if value is None:
        return None
    if spec.role == "scalar":
        return value
    if spec.role == "encoded":
        assert spec.codec_name is not None
        codec = codec_named(spec.codec_name)
        return codec.encode(value)[_query_index(codec)]
    if spec.role == "fixed_array":
        assert spec.shape is not None
        return _as_fixed_tensor(schema, spec, spec.shape, value).to_floats()
    assert spec.role == "child"
    if spec.shape is not None:
        tensor = FracVector.use(value)
        if tensor.dim in ((), (0,)):
            return []
        return tensor.to_floats()
    if spec.codec_name is not None:
        codec = codec_named(spec.codec_name)
        index = _query_index(codec)
        return [codec.encode(element)[index] for element in value]
    return list(value)
