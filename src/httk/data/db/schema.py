"""Schema IR: resolve a storable dataclass into the relational schema that drives storage.

A *storable* class is a plain **frozen dataclass** declared with the stdlib-only
marker vocabulary from httk-core (:class:`~httk.core.Indexed`,
:class:`~httk.core.Unique`, :class:`~httk.core.Skip`, :class:`~httk.core.Shape`,
:class:`~httk.core.StorageInfo`, :class:`~httk.core.stored_property`).
:func:`resolve_schema` reads the class once — dataclass fields, ``Annotated``
markers, stored properties, and the optional class-level or externally
registered :class:`~httk.core.StorageInfo` — and produces a
:class:`TableSchema`, the single source of truth from which the SQL layer
derives DDL, inserts, selects, and reconstruction alike.

Resolution rules (field annotation, then the resulting relational shape):

- ``int``/``float``/``str``/``bool``/``bytes`` — one scalar column named after
  the field (``bool`` is its own column kind, never folded into ``int``).
- ``X | None`` — the field is optional; all of its columns become nullable.
- a type with a registered :class:`~httk.data.db.codecs.ValueCodec`
  (:class:`fractions.Fraction`, :class:`~httk.core.FracScalar`,
  :class:`~httk.core.SurdScalar`, :class:`datetime.datetime`, ...) — the
  codec's columns, named by appending each suffix to the field name.
- ``Annotated[FracVector, Shape(r, c)]`` with ``r >= 1`` — a fixed-shape tensor
  stored inline: ``r*c`` float columns ``name_0 .. name_{r*c-1}`` plus one
  ``name_exact`` text column holding the canonical exact tensor text.
- ``Annotated[FracVector, Shape(0, c)]`` — variable rows in a child table
  ``<parent>_<name>``, each row ``c`` float columns plus a ``name_exact`` text
  column with the same exact encoding per row.
- ``list[T]`` / homogeneous ``tuple[T, ...]`` — a child table: scalar or
  codec-typed elements store their columns per row; storable-dataclass elements
  store one ``name_sid`` foreign-key column per row.
- another storable frozen dataclass (optionally ``| None``) — a reference:
  one ``name_sid`` foreign-key column.
- ``Annotated[..., Skip()]`` — omitted from storage (the field must have a
  default so instances can be reconstructed without it).
- a :class:`~httk.core.stored_property` — resolved like a field from its return
  annotation, flagged derived: stored and queryable, recomputed (not passed to
  ``__init__``) on reconstruction.

The store layer additionally manages a ``sid`` integer primary key and a
``content_id`` text column on every table (and ``<parent>_sid`` /
``<name>_index`` columns on child tables); those never appear in
:attr:`TableSchema.fields`, and declaring a field named ``sid`` or
``content_id`` is an error.
"""

import collections.abc
import dataclasses
import re
import types
import typing
from typing import Annotated, Any, Final, Literal

from httk.core import (
    STORAGE_INFO_ATTRIBUTE,
    DedupPolicy,
    FracVector,
    Indexed,
    Shape,
    Skip,
    StorageInfo,
    Unique,
    stored_property,
)

from httk.data.db.codecs import ScalarKind, codec_for, codec_named

__all__ = [
    "ScalarKind",
    "FieldRole",
    "SchemaError",
    "ColumnSpec",
    "ChildTableSpec",
    "FieldSpec",
    "TableSchema",
    "resolve_schema",
    "register_schema_override",
    "snake_case",
]

type FieldRole = Literal["scalar", "encoded", "fixed_array", "child", "reference"]
"""How a stored field maps onto the relational model."""


class SchemaError(Exception):
    """A class or field cannot be resolved into a storage schema; the message names both."""


@dataclasses.dataclass(frozen=True)
class ColumnSpec:
    """One scalar column of a table."""

    name: str
    """The column name."""

    kind: ScalarKind
    """The scalar kind of the column."""

    nullable: bool = False
    """Whether the column accepts NULL (all columns of an optional field do)."""

    indexed: bool = False
    """Whether a single-column index is requested (:class:`~httk.core.Indexed`)."""

    unique: bool = False
    """Whether a unique index is requested (:class:`~httk.core.Unique`)."""


@dataclasses.dataclass(frozen=True)
class ChildTableSpec:
    """The out-of-line child table backing a variable-length field.

    Only the per-element value columns are listed; the store layer adds the
    ``<parent>_sid`` foreign key and ``<name>_index`` ordering columns.
    """

    table_name: str
    """The child table name, ``<parent table>_<field>``."""

    element_columns: tuple[ColumnSpec, ...]
    """The value column(s) of one element row."""

    target: type | None = None
    """The storable element class when rows are foreign keys, else None."""


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """The resolved storage shape of one stored field (or stored property)."""

    field: str
    """The dataclass field (or stored property) name."""

    python_type: Any
    """The field's value type with markers and optionality stripped."""

    role: FieldRole
    """How the field maps onto the relational model."""

    columns: tuple[ColumnSpec, ...] = ()
    """The field's columns in the parent table (empty for the child role)."""

    codec_name: str | None = None
    """The value codec encoding this field (or its child elements), if any."""

    shape: Shape | None = None
    """The :class:`~httk.core.Shape` marker for tensor-valued fields, if any."""

    child: ChildTableSpec | None = None
    """The child table specification for the child role, else None."""

    target: type | None = None
    """The referenced storable class for reference (and child-of-storable) fields."""

    derived: bool = False
    """True for :class:`~httk.core.stored_property` values: stored and queryable,
    recomputed rather than passed to ``__init__`` on reconstruction."""

    optional: bool = False
    """True when the annotation was ``X | None``; all columns are then nullable."""


@dataclasses.dataclass(frozen=True)
class TableSchema:
    """The resolved relational schema of one storable class."""

    cls: type
    """The storable dataclass this schema was resolved from."""

    table_name: str
    """The table name (:attr:`~httk.core.StorageInfo.table_name` or the snake-cased class name)."""

    fields: tuple[FieldSpec, ...]
    """The stored fields, dataclass fields first (in declaration order), then stored properties.

    The store-managed ``sid`` and ``content_id`` columns are not fields and do
    not appear here.
    """

    composite_indexes: tuple[tuple[str, ...], ...]
    """The :attr:`~httk.core.StorageInfo.indexes` declarations, resolved to column names."""

    dedup: DedupPolicy
    """The deduplication policy applied when instances are saved."""

    def field(self, name: str) -> FieldSpec:
        """Return the :class:`FieldSpec` named ``name``; raise :class:`SchemaError` if absent."""
        for spec in self.fields:
            if spec.field == name:
                return spec
        raise SchemaError(f"{self.cls.__name__} has no stored field named {name!r}")

    def referenced_classes(self) -> tuple[type, ...]:
        """The distinct storable classes this schema references (field order, no duplicates)."""
        seen: list[type] = []
        for spec in self.fields:
            if spec.target is not None and spec.target not in seen:
                seen.append(spec.target)
        return tuple(seen)


_RESERVED_FIELD_NAMES: Final = frozenset({"sid", "content_id"})

_SCALAR_KINDS: Final[dict[type, ScalarKind]] = {
    int: "int",
    float: "float",
    str: "str",
    bool: "bool",
    bytes: "bytes",
}

_SNAKE_BOUNDARY_1: Final = re.compile(r"(.)([A-Z][a-z]+)")
_SNAKE_BOUNDARY_2: Final = re.compile(r"([a-z0-9])([A-Z])")

_schema_cache: dict[tuple[type, StorageInfo | None], TableSchema] = {}
_schema_overrides: dict[type, StorageInfo] = {}
_in_progress: set[type] = set()


def snake_case(name: str) -> str:
    """The default table name for a class name: CamelCase to camel_case (acronym-aware)."""
    return _SNAKE_BOUNDARY_2.sub(r"\1_\2", _SNAKE_BOUNDARY_1.sub(r"\1_\2", name)).lower()


def register_schema_override(cls: type, info: StorageInfo) -> None:
    """Register an external :class:`~httk.core.StorageInfo` for a class that cannot declare one.

    The registered info is used by :func:`resolve_schema` whenever no explicit
    ``override`` argument is passed, and takes precedence over a
    ``__httk_storage__`` declaration on the class itself.
    """
    _schema_overrides[cls] = info


def resolve_schema(cls: type, *, override: StorageInfo | None = None) -> TableSchema:
    """Resolve (and cache) the :class:`TableSchema` of a storable dataclass.

    The effective :class:`~httk.core.StorageInfo` is, in order of precedence:
    the explicit ``override`` argument, an info registered via
    :func:`register_schema_override`, the class's own ``__httk_storage__``
    attribute, or defaults. Results are cached per ``(class, effective
    override)``, so repeated calls return the same :class:`TableSchema` object.
    Reference cycles (a class referencing itself, or mutually referencing
    classes) are allowed and resolve without recursion loops.

    Raises:
        SchemaError: If the class is not a frozen dataclass or any field cannot
            be resolved; the message names the class and field.
    """
    effective = override if override is not None else _schema_overrides.get(cls)
    key = (cls, effective)
    cached = _schema_cache.get(key)
    if cached is not None:
        return cached

    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise SchemaError(f"{getattr(cls, '__name__', cls)!r} is not a dataclass; storable classes are dataclasses")
    if not getattr(cls, "__dataclass_params__").frozen:
        raise SchemaError(
            f"{cls.__name__} is not a frozen dataclass; storable classes are immutable records "
            f"(declare it with @dataclass(frozen=True))"
        )

    _in_progress.add(cls)
    try:
        schema = _build_schema(cls, effective)
    finally:
        _in_progress.discard(cls)
    _schema_cache[key] = schema
    return schema


def _build_schema(cls: type, override: StorageInfo | None) -> TableSchema:
    info = override if override is not None else getattr(cls, STORAGE_INFO_ATTRIBUTE, None)
    if info is None:
        info = StorageInfo()
    table_name = info.table_name if info.table_name is not None else snake_case(cls.__name__)

    hints = typing.get_type_hints(cls, include_extras=True)
    specs: list[FieldSpec] = []
    for field in dataclasses.fields(cls):
        has_default = field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING
        spec = _resolve_field(cls, table_name, field.name, hints[field.name], derived=False, has_default=has_default)
        if spec is not None:
            specs.append(spec)

    field_names = {spec.field for spec in specs}
    for name, prop in _stored_properties(cls):
        if name in field_names:
            raise SchemaError(f"{cls.__name__}.{name} is both a dataclass field and a stored property")
        spec = _resolve_field(
            cls, table_name, name, _property_annotation(cls, name, prop), derived=True, has_default=True
        )
        if spec is not None:
            specs.append(spec)
            field_names.add(name)

    composite_indexes = _resolve_composite_indexes(cls, info, specs)
    return TableSchema(
        cls=cls,
        table_name=table_name,
        fields=tuple(specs),
        composite_indexes=composite_indexes,
        dedup=info.dedup,
    )


def _stored_properties(cls: type) -> list[tuple[str, stored_property]]:
    """The class's stored properties in definition order, nearest override first per name."""
    found: dict[str, stored_property] = {}
    for klass in cls.__mro__:
        for name, attribute in vars(klass).items():
            if isinstance(attribute, stored_property) and name not in found:
                found[name] = attribute
    return list(found.items())


def _property_annotation(cls: type, name: str, prop: stored_property) -> Any:
    if prop.fget is None:
        raise SchemaError(f"{cls.__name__}.{name}: stored_property has no getter")
    hints = typing.get_type_hints(prop.fget, include_extras=True)
    if "return" not in hints:
        raise SchemaError(f"{cls.__name__}.{name}: stored_property getter needs a return annotation")
    return hints["return"]


def _resolve_field(
    cls: type, table_name: str, name: str, annotation: Any, *, derived: bool, has_default: bool
) -> FieldSpec | None:
    if name in _RESERVED_FIELD_NAMES:
        raise SchemaError(
            f"{cls.__name__}.{name}: the field names {sorted(_RESERVED_FIELD_NAMES)} are reserved for "
            f"store-managed columns"
        )

    base = annotation
    optional = False
    indexed = False
    unique = False
    shape: Shape | None = None
    skipped = False
    while True:
        origin = typing.get_origin(base)
        if origin is Annotated:
            arguments = typing.get_args(base)
            for marker in arguments[1:]:
                if isinstance(marker, Indexed):
                    indexed = True
                elif isinstance(marker, Unique):
                    unique = True
                elif isinstance(marker, Skip):
                    skipped = True
                elif isinstance(marker, Shape):
                    shape = marker
            base = arguments[0]
            continue
        if origin is typing.Union or origin is types.UnionType:
            arguments = typing.get_args(base)
            others = tuple(argument for argument in arguments if argument is not types.NoneType)
            if len(others) == len(arguments):
                raise SchemaError(f"{cls.__name__}.{name}: union types are not storable (only 'X | None')")
            if len(others) != 1:
                raise SchemaError(f"{cls.__name__}.{name}: only unions of one type with None are storable")
            optional = True
            base = others[0]
            continue
        break

    if skipped:
        if not derived and not has_default:
            raise SchemaError(
                f"{cls.__name__}.{name}: a Skip'd field must have a default so instances can be "
                f"reconstructed without it"
            )
        return None

    if shape is not None:
        return _resolve_shape_field(cls, table_name, name, base, shape, optional, indexed, unique, derived)

    kind = _SCALAR_KINDS.get(base)
    if kind is not None:
        column = ColumnSpec(name, kind, nullable=optional, indexed=indexed, unique=unique)
        return FieldSpec(
            field=name, python_type=base, role="scalar", columns=(column,), derived=derived, optional=optional
        )

    codec = codec_for(base)
    if codec is not None:
        columns = tuple(
            ColumnSpec(
                f"{name}{suffix}",
                column_kind,
                nullable=optional,
                indexed=indexed and suffix == codec.query_suffix,
                unique=unique and suffix == codec.query_suffix,
            )
            for suffix, column_kind in codec.columns
        )
        return FieldSpec(
            field=name,
            python_type=base,
            role="encoded",
            columns=columns,
            codec_name=codec.name,
            derived=derived,
            optional=optional,
        )

    origin = typing.get_origin(base)
    if origin is list or origin is tuple:
        return _resolve_sequence_field(cls, table_name, name, base, origin, optional, derived)

    if _is_mapping_type(base, origin):
        raise SchemaError(f"{cls.__name__}.{name}: mapping-typed fields are not storable (yet)")

    if isinstance(base, type) and issubclass(base, FracVector):
        raise SchemaError(
            f"{cls.__name__}.{name}: a FracVector field needs a Shape marker, e.g. "
            f"Annotated[FracVector, Shape(3, 3)]"
        )

    if isinstance(base, type) and dataclasses.is_dataclass(base):
        _validate_target(cls, base)
        column = ColumnSpec(f"{name}_sid", "int", nullable=optional, indexed=indexed, unique=unique)
        return FieldSpec(
            field=name,
            python_type=base,
            role="reference",
            columns=(column,),
            target=base,
            derived=derived,
            optional=optional,
        )

    raise SchemaError(
        f"{cls.__name__}.{name}: type {base!r} is not storable; expected a scalar "
        f"(int/float/str/bool/bytes), a codec type, a Shape-annotated FracVector, a list/tuple, "
        f"or a storable frozen dataclass"
    )


def _resolve_shape_field(
    cls: type,
    table_name: str,
    name: str,
    base: Any,
    shape: Shape,
    optional: bool,
    indexed: bool,
    unique: bool,
    derived: bool,
) -> FieldSpec:
    if base is not FracVector:
        raise SchemaError(f"{cls.__name__}.{name}: Shape markers apply to FracVector fields only, got {base!r}")
    if shape.rows >= 1:
        size = shape.rows * shape.cols
        columns = tuple(
            ColumnSpec(f"{name}_{i}", "float", nullable=optional, indexed=indexed, unique=unique) for i in range(size)
        ) + (
            ColumnSpec(f"{name}_exact", "str", nullable=optional, indexed=indexed, unique=unique),
        )
        return FieldSpec(
            field=name,
            python_type=base,
            role="fixed_array",
            columns=columns,
            shape=shape,
            derived=derived,
            optional=optional,
        )
    element_columns = tuple(ColumnSpec(f"{name}_{i}", "float") for i in range(shape.cols)) + (
        ColumnSpec(f"{name}_exact", "str"),
    )
    child = ChildTableSpec(table_name=f"{table_name}_{name}", element_columns=element_columns)
    return FieldSpec(
        field=name, python_type=base, role="child", shape=shape, child=child, derived=derived, optional=optional
    )


def _resolve_sequence_field(
    cls: type, table_name: str, name: str, base: Any, origin: type, optional: bool, derived: bool
) -> FieldSpec:
    arguments = typing.get_args(base)
    if origin is tuple:
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise SchemaError(f"{cls.__name__}.{name}: only homogeneous variable tuples (tuple[T, ...]) are storable")
        element_type = arguments[0]
    else:
        if len(arguments) != 1:
            raise SchemaError(f"{cls.__name__}.{name}: a storable list needs exactly one element type")
        element_type = arguments[0]

    codec_name: str | None = None
    target: type | None = None
    kind = _SCALAR_KINDS.get(element_type)
    if kind is not None:
        element_columns: tuple[ColumnSpec, ...] = (ColumnSpec(name, kind),)
    else:
        codec = codec_for(element_type)
        if codec is not None:
            codec_name = codec.name
            element_columns = tuple(ColumnSpec(f"{name}{suffix}", column_kind) for suffix, column_kind in codec.columns)
        elif isinstance(element_type, type) and dataclasses.is_dataclass(element_type):
            _validate_target(cls, element_type)
            target = element_type
            element_columns = (ColumnSpec(f"{name}_sid", "int"),)
        else:
            raise SchemaError(
                f"{cls.__name__}.{name}: element type {element_type!r} is not storable; expected a "
                f"scalar, a codec type, or a storable frozen dataclass"
            )
    child = ChildTableSpec(table_name=f"{table_name}_{name}", element_columns=element_columns, target=target)
    return FieldSpec(
        field=name,
        python_type=base,
        role="child",
        codec_name=codec_name,
        child=child,
        target=target,
        derived=derived,
        optional=optional,
    )


def _is_mapping_type(base: Any, origin: Any) -> bool:
    candidate = origin if origin is not None else base
    return isinstance(candidate, type) and issubclass(candidate, collections.abc.Mapping)


def _validate_target(cls: type, target: type) -> None:
    """Resolve a referenced storable class, tolerating reference cycles."""
    if target is cls or target in _in_progress:
        return
    resolve_schema(target)


def _resolve_composite_indexes(cls: type, info: StorageInfo, specs: list[FieldSpec]) -> tuple[tuple[str, ...], ...]:
    by_name = {spec.field: spec for spec in specs}
    resolved: list[tuple[str, ...]] = []
    for index in info.indexes:
        columns: list[str] = []
        for field_name in index:
            spec = by_name.get(field_name)
            if spec is None:
                raise SchemaError(f"{cls.__name__}: composite index names unknown field {field_name!r}")
            if spec.role == "scalar" or spec.role == "reference":
                columns.append(spec.columns[0].name)
            elif spec.role == "encoded":
                assert spec.codec_name is not None
                columns.append(f"{spec.field}{codec_named(spec.codec_name).query_suffix}")
            else:
                raise SchemaError(
                    f"{cls.__name__}: composite index field {field_name!r} has role {spec.role!r}; only "
                    f"scalar, encoded, and reference fields can be composite-indexed"
                )
        resolved.append(tuple(columns))
    return tuple(resolved)
