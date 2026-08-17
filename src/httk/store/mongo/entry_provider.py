"""Serve MongoDB-backed records through the neutral entry-provider contract.

Mongo entry identities are content identities rather than SQL row identities.
Configured entry families are rendered through their Mongo stored-property plan;
configured backing records are also accepted for the schema-derived provider
surface used by the SQL provider's parity tests.
"""

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from httk.core import (
    EntryProvider,
    EntryTypeDefinition,
    FracVector,
    PropertyDefinition,
    RelatedEntry,
    content_id,
    known_definition_prefixes,
)
from httk.core.storage import RelationshipLink

from httk.store.db.codecs import codec_named
from httk.store.db.schema import FieldSpec, TableSchema, resolve_schema
from httk.store.served_specs import served_specs

from .documents import _as_fixed_tensor
from .stored_properties import MongoStoredPropertyPlan

__all__ = ["StoreEntryProvider", "auto_definition", "served_specs"]


def _default_id(_entry_type: str, _sid: int, obj: Any) -> str:
    return content_id(obj)


def _query_index(codec: Any) -> int:
    """Return the stored codec component used for query/response values."""
    return next((index for index, (suffix, _kind) in enumerate(codec.columns) if suffix == codec.query_suffix), 0)


def auto_definition(entry_type: str, schema: TableSchema, prefix: str) -> EntryTypeDefinition:
    """Build a definition for the JSON-able fields of one backing class."""
    base = EntryTypeDefinition(
        entry_type,
        f"The '{entry_type}' entry type, generated from the stored class {schema.cls.__name__}.",
        {
            "id": PropertyDefinition.from_simple("id", description="The unique entry id.", required_response=True),
            "type": PropertyDefinition.from_simple(
                "type", description="The name of the entry type.", required_response=True
            ),
        },
    )
    extra: dict[str, PropertyDefinition] = {}
    for name, spec, fulltype in served_specs(schema, prefix):
        dimensions: dict[str, Any] | None = None
        if spec.role == "fixed_array":
            assert spec.shape is not None
            dimensions = {"names": ["rows", "cols"], "sizes": [spec.shape.rows, spec.shape.cols]}
        kind = "stored property" if spec.derived else "stored field"
        extra[name] = PropertyDefinition.from_simple(
            name,
            description=f"The {kind} '{spec.field}' of {schema.cls.__name__}.",
            fulltype=fulltype,
            dimensions=dimensions,
        )
    return base.extended(extra)


@dataclasses.dataclass(frozen=True)
class _LinkScan:
    declaring: type
    link: RelationshipLink
    from_cls: type
    to_cls: type
    from_type: str
    to_type: str


class StoreEntryProvider(EntryProvider):
    """Serve configured Mongo entry families or their concrete backings.

    ``classes`` maps public entry-type names to either configured entry-family
    classes or configured concrete backing classes. Family classes use the
    family's :class:`~httk.store.mongo.stored_properties.MongoStoredPropertyPlan`; backing classes use the same
    schema-derived property contract as the SQL provider. ``id_of`` receives
    ``(entry_type, sid, hydrated_record)`` and defaults to the record's
    content identity. ``only_latest`` restricts served searchers' root
    variables to the latest document of each lineage.
    """

    def __init__(
        self,
        store: Any,
        classes: Mapping[str, type],
        *,
        definitions: Mapping[str, EntryTypeDefinition] | None = None,
        prefix: str = "_httk_",
        id_of: Callable[[str, int, Any], str] | None = None,
        link_classes: Iterable[type] = (),
        only_latest: bool = False,
    ) -> None:
        if prefix not in known_definition_prefixes():
            raise ValueError(
                f"the property-name prefix {prefix!r} is not registered; register it with "
                f"httk.core.register_definition_prefix() (registered prefixes: "
                f"{', '.join(known_definition_prefixes())})"
            )
        self._store = store
        self._classes = dict(classes)
        self._only_latest = only_latest
        self._prefix = prefix
        self._id_of = id_of if id_of is not None else _default_id
        self._definitions = dict(definitions or {})
        unknown = sorted(name for name in self._definitions if name not in self._classes)
        if unknown:
            raise ValueError(
                f"definitions were supplied for entry types this provider does not serve: {', '.join(unknown)}"
            )

        self._families: dict[str, type] = {}
        self._plans: dict[str, MongoStoredPropertyPlan] = {}
        self._record_classes: dict[str, tuple[type, ...]] = {}
        self._schemas: dict[type, TableSchema] = {}
        for entry_type, selected in self._classes.items():
            family = self._family_for(selected, entry_type)
            if family is not None:
                plan = store.stored_property_plan(family)
                self._families[entry_type] = family
                self._plans[entry_type] = plan
                self._record_classes[entry_type] = plan.backings
            else:
                self._record_classes[entry_type] = (selected,)
            for record in self._record_classes[entry_type]:
                self._schemas[record] = resolve_schema(record)

        self._type_for_class: dict[type, str] = {}
        for entry_type, records in self._record_classes.items():
            for record in records:
                self._type_for_class[record] = entry_type
        self._links_by_from = self._build_link_inventory(tuple(link_classes))

        for entry_type in self._classes:
            definition = self._definitions.get(entry_type)
            if definition is not None:
                actual_keys = self._actual_property_keys(entry_type)
                missing = sorted(set(actual_keys) - set(definition.properties))
                if missing:
                    raise ValueError(
                        f"the supplied definition for entry type {entry_type!r} does not describe the served "
                        f"propert{'y' if len(missing) == 1 else 'ies'}: {', '.join(missing)}"
                    )

    def _family_for(self, selected: type, entry_type: str) -> type | None:
        for layout in self._store.entry_layout:
            if selected is layout.family:
                family_type = getattr(selected, "type", entry_type)
                if entry_type != family_type:
                    raise ValueError(f"entry family {selected.__name__} has type {family_type!r}, not {entry_type!r}")
                return selected
            if selected in layout.records:
                return None
        raise ValueError(
            f"record class {selected.__name__} is not configured in this MongoStore; "
            "serve a configured entry family or one of its configured backing records"
        )

    def _build_link_inventory(self, link_classes: tuple[type, ...]) -> dict[str, list[_LinkScan]]:
        inventory: dict[str, list[_LinkScan]] = {}
        seen: set[type] = set()
        declaring_classes = [record for records in self._record_classes.values() for record in records]
        declaring_classes.extend(link_classes)
        for declaring in declaring_classes:
            if declaring in seen:
                continue
            seen.add(declaring)
            schema = self._schemas.setdefault(declaring, resolve_schema(declaring))
            if declaring not in self._type_for_class and not schema.links:
                raise ValueError(
                    f"link class {declaring.__name__} declares no relationship links (StorageInfo.links is empty); "
                    "remove it from link_classes or declare its links"
                )
            for link in schema.links:
                from_cls = schema.field(link.source).target if link.source is not None else declaring
                to_cls = schema.field(link.target).target if link.target is not None else declaring
                assert from_cls is not None and to_cls is not None
                from_type = self._type_for_class.get(from_cls)
                to_type = self._type_for_class.get(to_cls)
                if from_type is None or to_type is None:
                    missing = from_cls if from_type is None else to_cls
                    side = "FROM" if from_type is None else "TO"
                    raise ValueError(
                        f"RelationshipLink({link.source!r}, {link.target!r}) on {declaring.__name__}: the {side}-side "
                        f"class {missing.__name__} is not served by this provider; every link endpoint must resolve "
                        "to a served entry type"
                    )
                inventory.setdefault(from_type, []).append(
                    _LinkScan(declaring, link, from_cls, to_cls, from_type, to_type)
                )
        return inventory

    def _require_entry_type(self, entry_type: str) -> None:
        if entry_type not in self._classes:
            raise KeyError(f"StoreEntryProvider serves only the entry type(s): {', '.join(sorted(self._classes))}")

    def _definition(self, entry_type: str) -> EntryTypeDefinition:
        supplied = self._definitions.get(entry_type)
        if supplied is not None:
            return supplied
        plan = self._plans.get(entry_type)
        if plan is not None:
            return plan.definition
        return auto_definition(entry_type, self._schemas[self._record_classes[entry_type][0]], self._prefix)

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        """Return definitions for all served entry types."""
        return {entry_type: self._definition(entry_type) for entry_type in self._classes}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        """Return public property names mapped to Mongo response keys."""
        self._require_entry_type(entry_type)
        return self._actual_property_keys(entry_type)

    def _actual_property_keys(self, entry_type: str) -> Mapping[str, str]:
        """Return keys actually emitted by :meth:`records`, independent of overrides."""
        if entry_type in self._plans:
            return {name: name for name in self._plans[entry_type].definition.properties}
        return {"id": "id", "type": "type", **{name: name for name, _spec, _ in self._served_specs(entry_type)}}

    def _served_specs(self, entry_type: str) -> list[tuple[str, FieldSpec, str]]:
        record = self._record_classes[entry_type][0]
        return served_specs(self._schemas[record], self._prefix)

    def _iter_records(self, record: type) -> Iterator[tuple[Any, int]]:
        searcher = self._store.searcher(only_latest=self._only_latest)
        variable = searcher.variable(record)
        searcher.add_sort(variable.sid)
        searcher.output(variable, "record")
        searcher.output(variable.sid, "sid")
        for result in searcher:
            yield result[0][0], int(result[0][1])

    def records(self, entry_type: str) -> Iterator[Mapping[str, Any]]:
        """Yield JSON-able records for one served entry type."""
        self._require_entry_type(entry_type)
        plan = self._plans.get(entry_type)
        if plan is not None:
            for backing in plan.backings:
                for record, sid in self._iter_records(backing):
                    yield plan.response_row(backing, record, public_id=self._id_of(entry_type, sid, record))
            return
        specs = self._served_specs(entry_type)
        schema = self._schemas[self._record_classes[entry_type][0]]
        for record, sid in self._iter_records(self._record_classes[entry_type][0]):
            row: dict[str, Any] = {"id": self._id_of(entry_type, sid, record), "type": entry_type}
            for name, spec, _fulltype in specs:
                row[name] = _json_value(schema, spec, getattr(record, spec.field))
            yield row

    def _relationship_specs(self, record: type) -> list[tuple[FieldSpec, str]]:
        result: list[tuple[FieldSpec, str]] = []
        for spec in self._schemas[record].fields:
            if spec.role not in {"reference", "child"} or spec.target is None:
                continue
            if spec.related is not None and not spec.related.serve:
                continue
            related = self._type_for_class.get(spec.target)
            if related is not None:
                result.append((spec, related))
        return result

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        """Return direct and link-derived relationships grouped by source id."""
        self._require_entry_type(entry_type)
        result: dict[str, list[RelatedEntry]] = {}
        for record_type in self._record_classes[entry_type]:
            for source, sid in self._iter_records(record_type):
                entries: list[RelatedEntry] = []
                relation_specs = self._relationship_specs(record_type)
                relation_specs.sort(key=lambda item: item[0].role != "reference")
                for spec, related_type in relation_specs:
                    values = (
                        (getattr(source, spec.field),)
                        if spec.role == "reference"
                        else (getattr(source, spec.field) or ())
                    )
                    marker = spec.related
                    for target in values:
                        if target is None:
                            continue
                        target_type = self._type_for_class.get(type(target))
                        if target_type != related_type:
                            continue
                        target_sid = self._store.sid_of(target, as_record=type(target))
                        if target_sid is None:
                            continue
                        entries.append(
                            RelatedEntry(
                                related_type,
                                self._id_of(related_type, target_sid, target),
                                description=marker.description if marker is not None else None,
                                role=marker.role if marker is not None else None,
                            )
                        )
                if entries:
                    result[self._id_of(entry_type, sid, source)] = entries

        for scan in self._links_by_from.get(entry_type, ()):
            for link_obj, link_sid in self._iter_records(scan.declaring):
                source = link_obj if scan.link.source is None else getattr(link_obj, scan.link.source)
                target = link_obj if scan.link.target is None else getattr(link_obj, scan.link.target)
                if source is None or target is None:
                    continue
                source_sid = (
                    link_sid if scan.link.source is None else self._store.sid_of(source, as_record=scan.from_cls)
                )
                target_sid = self._store.sid_of(target, as_record=scan.to_cls)
                if source_sid is None or target_sid is None:
                    continue
                source_id = self._id_of(entry_type, source_sid, source)
                result.setdefault(source_id, []).append(
                    RelatedEntry(
                        scan.to_type,
                        self._id_of(scan.to_type, target_sid, target),
                        description=scan.link.description,
                        role=scan.link.role,
                    )
                )

        return {key: tuple(dict.fromkeys(values)) for key, values in result.items()}


def _json_value(schema: TableSchema, spec: FieldSpec, value: Any) -> Any:
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
        tensor = FracVector(value)
        return [] if tensor.dim in {(), (0,)} else tensor.to_floats()
    if spec.codec_name is not None:
        codec = codec_named(spec.codec_name)
        index = _query_index(codec)
        return [codec.encode(element)[index] for element in value]
    return list(value)
