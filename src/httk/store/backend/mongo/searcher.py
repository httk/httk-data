"""MongoDB query planning for the neutral :mod:`httk.store.query` protocols.

The planner keeps a truth/falsity pair for each predicate.  This is important
for embedded arrays: the negation of "some element matches" is "no element
matches", not MongoDB's row-like ``$ne`` interpretation.
"""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, cast

from httk.store.backend.sql.codecs import ValueCodec, codec_named
from httk.store.backend.sql.schema import FieldSpec, SchemaError, TableSchema, resolve_schema
from httk.store.query import SearchResult, UnsupportedQueryError
from httk.store.store_timestamp import ns_operand_to_store_units

if TYPE_CHECKING:
    from httk.store.backend.mongo.store import MongoStore

__all__ = [
    "AlwaysFalseNode",
    "AlwaysTrueNode",
    "AndNode",
    "ComparisonNode",
    "IsInNode",
    "MongoExpression",
    "MongoField",
    "MongoSearcher",
    "MongoVariable",
    "NotNode",
    "OrNode",
    "StringMatchNode",
]


@dataclass(frozen=True, slots=True)
class _FieldReference:
    """A field used as the right side of a comparison."""

    field: "MongoField"


@dataclass(frozen=True, slots=True)
class ComparisonNode:
    """Compare one stored field with a literal or another field."""

    field: "MongoField"
    op: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    literal: Any


@dataclass(frozen=True, slots=True)
class IsInNode:
    """Test a scalar field for explicit membership."""

    field: "MongoField"
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ChildSetNode:
    """Test containment or universality over one embedded child array."""

    field: "MongoField"
    operation: Literal["has_any", "has_only"]
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ChildComparisonNode:
    """Compare the value channel of any element in an embedded child array."""

    field: "MongoField"
    op: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    value: Any


@dataclass(frozen=True, slots=True)
class ChildStringNode:
    """Match literal text in the value channel of an embedded child array."""

    field: "MongoField"
    mode: Literal["contains", "startswith", "endswith"]
    text: str


@dataclass(frozen=True, slots=True)
class ReferenceEqualityNode:
    """Compare a reference key with a declared query variable's sid."""

    reference: "MongoReference"
    variable: "MongoVariable"


@dataclass(frozen=True, slots=True)
class StringMatchNode:
    """Match literal text in a scalar string field."""

    field: "MongoField"
    mode: Literal["contains", "startswith", "endswith"]
    text: str


@dataclass(frozen=True, slots=True)
class AndNode:
    """Conjoin two AST nodes."""

    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class OrNode:
    """Disjoin two AST nodes."""

    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class NotNode:
    """Negate an AST node by swapping its truth and falsity filters."""

    child: Any


@dataclass(frozen=True, slots=True)
class AlwaysTrueNode:
    """An AST constant that is true for every document."""


@dataclass(frozen=True, slots=True)
class AlwaysFalseNode:
    """An AST constant that is false for every document."""


Node = (
    ComparisonNode
    | IsInNode
    | ChildSetNode
    | ChildComparisonNode
    | ChildStringNode
    | ReferenceEqualityNode
    | StringMatchNode
    | AndNode
    | OrNode
    | NotNode
    | AlwaysTrueNode
    | AlwaysFalseNode
)


def _and(*filters: dict[str, Any]) -> dict[str, Any]:
    """Conjoin Mongo filters without relying on duplicate mapping keys."""
    present = [item for item in filters if item]
    if not present:
        return {}
    if len(present) == 1:
        return present[0]
    return {"$and": present}


def _or(*filters: dict[str, Any]) -> dict[str, Any]:
    """Disjoin Mongo filters."""
    if any(not item for item in filters):
        return {}
    if not filters:
        return _unknown_constant()
    if len(filters) == 1:
        return filters[0]
    return {"$or": list(filters)}


def _unknown_constant() -> dict[str, Any]:
    """Return a filter that cannot match a normal Mongo document."""
    return {"_id": {"$exists": False}}


def _type_in(path: str, types: list[str]) -> dict[str, Any]:
    """Match a path whose aggregation type belongs to ``types``.

    ``$type`` is deliberately used here instead of ``$exists``/``$ne``: Mongo
    treats missing fields specially for several query operators.
    """
    return {"$expr": {"$in": [{"$type": f"${path}"}, types]}}


def _known(path: str) -> dict[str, Any]:
    """Return the explicit SQL-known predicate for a scalar path."""
    return _type_in(
        path,
        [
            "double",
            "string",
            "object",
            "array",
            "binData",
            "undefined",
            "objectId",
            "bool",
            "date",
            "regex",
            "int",
            "timestamp",
            "long",
            "decimal",
        ],
    )


def _null(path: str) -> dict[str, Any]:
    """Match Mongo null and absent values, the scalar NULL state here."""
    return _type_in(path, ["missing", "null"])


def _safe_expr(path: str, other: str, expression: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a field comparison only after both operands are non-null."""
    return {
        "$expr": {
            "$cond": [
                {
                    "$and": [
                        {"$not": [{"$in": [{"$type": f"${path}"}, ["missing", "null"]]}]},
                        {"$not": [{"$in": [{"$type": f"${other}"}, ["missing", "null"]]}]},
                    ]
                },
                expression,
                False,
            ]
        }
    }


def _render_comparison(node: ComparisonNode) -> tuple[dict[str, Any], dict[str, Any]]:
    path = node.field._path
    if node.literal is None and not isinstance(node.literal, _FieldReference):
        if node.op == "eq":
            return _null(path), _known(path)
        if node.op == "ne":
            return _known(path), _null(path)
        return _unknown_constant(), _known(path)

    operators = {"eq": "$eq", "ne": "$ne", "lt": "$lt", "le": "$lte", "gt": "$gt", "ge": "$gte"}
    complements = {"eq": "ne", "ne": "eq", "lt": "ge", "le": "gt", "gt": "le", "ge": "lt"}
    value = node.literal
    if isinstance(value, _FieldReference):
        right = value.field._path
        return (
            _safe_expr(path, right, {operators[node.op]: [f"${path}", f"${right}"]}),
            _safe_expr(path, right, {operators[complements[node.op]]: [f"${path}", f"${right}"]}),
        )
    truth_op = {path: {operators[node.op]: value}}
    false_op = {path: {operators[complements[node.op]]: value}}
    return _and(truth_op, _known(path)), _and(false_op, _known(path))


def _render_is_in(node: IsInNode) -> tuple[dict[str, Any], dict[str, Any]]:
    path = node.field._path
    non_null = tuple(value for value in node.values if value is not None)
    includes_null = len(non_null) != len(node.values)
    if non_null:
        member = _and({path: {"$in": list(non_null)}}, _known(path))
        truth = _or(_null(path), member) if includes_null else member
        false = _and({path: {"$nin": list(non_null)}}, _known(path))
    elif includes_null:
        truth = _null(path)
        false = _known(path)
    else:
        truth = _unknown_constant()
        false = _known(path)
    return truth, false


def _array_domain(path: str) -> dict[str, Any]:
    """Match the canonical array/missing/null states of a child field."""
    return _type_in(path, ["array", "missing", "null"])


def _child_element_condition(field: "MongoField", values: tuple[Any, ...], *, outside: bool) -> dict[str, Any]:
    """Return an ``$elemMatch`` body for members or outsiders."""
    encoded = [field._encode_child(value) for value in values]
    keys = field._child_keys
    assert keys
    if not encoded:
        # Every element is outside an empty allowed set; no element is a
        # member.  Avoid Mongo's invalid empty $or/$nor forms.
        return {} if outside else {keys[0]: {"$in": []}}
    if len(keys) == 1:
        return {keys[0]: {"$nin" if outside else "$in": [value[keys[0]] for value in encoded]}}
    clauses = [{key: encoded_value[key] for key in keys} for encoded_value in encoded]
    if outside:
        return {"$nor": clauses}
    return {"$or": clauses}


def _render_child_set(node: ChildSetNode) -> tuple[dict[str, Any], dict[str, Any]]:
    path = node.field._path
    member = {path: {"$elemMatch": _child_element_condition(node.field, node.values, outside=False)}}
    no_member = {path: {"$not": {"$elemMatch": _child_element_condition(node.field, node.values, outside=False)}}}
    outsider = {path: {"$elemMatch": _child_element_condition(node.field, node.values, outside=True)}}
    no_outsider = {path: {"$not": {"$elemMatch": _child_element_condition(node.field, node.values, outside=True)}}}
    if node.operation == "has_any":
        return _and(_type_in(path, ["array"]), member), _and(_array_domain(path), no_member)
    return _and(_array_domain(path), no_outsider), _and(_type_in(path, ["array"]), outsider)


def _child_value_key(field: "MongoField") -> str:
    """Return the physical key used for an element's query value."""
    assert field._child_keys
    if field._codec is None:
        return field._child_keys[0]
    return next(key for key in field._child_keys if key == field._spec.field + field._codec.query_suffix)


def _render_child_predicate(field: "MongoField", predicate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render existential child-element truth and universal falsity."""
    path = field._path
    member = {path: {"$elemMatch": predicate}}
    no_member = {path: {"$not": {"$elemMatch": predicate}}}
    return _and(_type_in(path, ["array"]), member), _and(_array_domain(path), no_member)


def _render_child_comparison(node: ChildComparisonNode) -> tuple[dict[str, Any], dict[str, Any]]:
    key = _child_value_key(node.field)
    operators = {"lt": "$lt", "le": "$lte", "gt": "$gt", "ge": "$gte"}
    predicate: dict[str, Any]
    if node.op == "ne":
        predicate = {key: {"$nin": [None, node.value]}}
    elif node.op == "eq":
        predicate = {key: {"$eq": node.value}}
    else:
        # The explicit null guard makes a null element unknown rather than a
        # witness for a rich comparison.
        predicate = {key: {"$ne": None, operators[node.op]: node.value}}
    return _render_child_predicate(node.field, predicate)


def _render_child_string(node: ChildStringNode) -> tuple[dict[str, Any], dict[str, Any]]:
    key = _child_value_key(node.field)
    escaped = re.escape(node.text)
    pattern = {"contains": f".*{escaped}.*", "startswith": f"^{escaped}", "endswith": f"{escaped}$"}[node.mode]
    return _render_child_predicate(node.field, {key: {"$regex": pattern}})


def _render_string(node: StringMatchNode) -> tuple[dict[str, Any], dict[str, Any]]:
    path = node.field._path
    escaped = re.escape(node.text)
    pattern = {"contains": f".*{escaped}.*", "startswith": f"^{escaped}", "endswith": f"{escaped}$"}[node.mode]
    regex = {"$regex": pattern}
    return _and({path: regex}, _known(path)), _and({path: {"$not": regex}}, _known(path))


def render_node(node: Node) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render an AST node to its TRUE and FALSE Mongo filters."""
    if isinstance(node, ComparisonNode):
        return _render_comparison(node)
    if isinstance(node, IsInNode):
        return _render_is_in(node)
    if isinstance(node, ChildSetNode):
        return _render_child_set(node)
    if isinstance(node, ChildComparisonNode):
        return _render_child_comparison(node)
    if isinstance(node, ChildStringNode):
        return _render_child_string(node)
    if isinstance(node, ReferenceEqualityNode):
        left = node.reference._field
        right = node.variable.sid
        return _render_comparison(ComparisonNode(left, "eq", _FieldReference(right)))
    if isinstance(node, StringMatchNode):
        return _render_string(node)
    if isinstance(node, AndNode):
        left_truth, left_false = render_node(node.left)
        right_truth, right_false = render_node(node.right)
        return _and(left_truth, right_truth), _or(left_false, right_false)
    if isinstance(node, OrNode):
        left_truth, left_false = render_node(node.left)
        right_truth, right_false = render_node(node.right)
        return _or(left_truth, right_truth), _and(left_false, right_false)
    if isinstance(node, NotNode):
        false, truth = render_node(node.child)
        return truth, false
    if isinstance(node, AlwaysTrueNode):
        return {}, _unknown_constant()
    if isinstance(node, AlwaysFalseNode):
        return _unknown_constant(), {}
    raise TypeError(f"unsupported Mongo query node {type(node).__name__}")


class MongoExpression:
    """A composable MongoDB expression backed by a neutral AST node."""

    __slots__ = ("node",)

    def __init__(self, node: Node) -> None:
        self.node = node

    def __and__(self, other: "MongoExpression") -> "MongoExpression":
        if not isinstance(other, MongoExpression):
            return NotImplemented
        return MongoExpression(AndNode(self.node, other.node))

    def __or__(self, other: "MongoExpression") -> "MongoExpression":
        if not isinstance(other, MongoExpression):
            return NotImplemented
        return MongoExpression(OrNode(self.node, other.node))

    def __invert__(self) -> "MongoExpression":
        return MongoExpression(NotNode(self.node))


class MongoField:
    """A scalar field, or the value channel of an embedded child field."""

    __slots__ = (
        "_child_keys",
        "_codec",
        "_key_path",
        "_operand_converter",
        "_presentation_converter",
        "_presentation_prefix",
        "_spec",
        "_variable",
    )

    def __init__(
        self,
        variable: "MongoVariable",
        key_path: str,
        spec: FieldSpec,
        codec: ValueCodec | None = None,
        child_keys: tuple[str, ...] = (),
        presentation_prefix: str = "",
        operand_converter: Callable[[Any], Any] | None = None,
        presentation_converter: Callable[[Any], Any] | None = None,
    ) -> None:
        self._variable = variable
        self._key_path = key_path
        self._spec = spec
        self._codec = codec
        self._child_keys = child_keys
        self._presentation_prefix = presentation_prefix
        self._operand_converter = operand_converter
        self._presentation_converter = presentation_converter

    @property
    def _path(self) -> str:
        """Return the aggregate path after this variable's lookup alias."""
        return f"{self._variable._document_path}.{self._key_path}" if self._variable._document_path else self._key_path

    def _encode(self, value: Any) -> Any:
        if isinstance(value, MongoField):
            return _FieldReference(value)
        if value is None:
            return value
        if self._operand_converter is not None:
            value = self._operand_converter(value)
        if self._codec is None:
            return value
        if isinstance(value, self._codec.python_type):
            query_index = next(
                index for index, (suffix, _kind) in enumerate(self._codec.columns) if suffix == self._codec.query_suffix
            )
            return self._codec.encode(value)[query_index]
        return value

    def _encode_child(self, value: Any) -> dict[str, Any]:
        if self._codec is not None and isinstance(value, self._codec.python_type):
            encoded = self._codec.encode(value)
            return dict(zip(self._child_keys, encoded, strict=True))
        if len(self._child_keys) != 1:
            if not isinstance(value, tuple) or len(value) != len(self._child_keys):
                raise TypeError(f"{self._spec.field} child values require {len(self._child_keys)} components")
            return dict(zip(self._child_keys, value, strict=True))
        return {self._child_keys[0]: value}

    def _comparison(self, op: Literal["eq", "ne", "lt", "le", "gt", "ge"], value: Any) -> MongoExpression:
        if self._child_keys:
            if value is None:
                raise ValueError("None is not a valid member of a child-field set operation")
            return MongoExpression(ChildComparisonNode(self, op, self._encode(value)))
        return MongoExpression(ComparisonNode(self, op, self._encode(value)))

    def __eq__(self, other: object) -> MongoExpression:  # type: ignore[override]
        return self._comparison("eq", other)

    def __ne__(self, other: object) -> MongoExpression:  # type: ignore[override]
        return self._comparison("ne", other)

    def __hash__(self) -> int:
        return id(self)

    def __lt__(self, other: Any) -> MongoExpression:
        return self._comparison("lt", other)

    def __le__(self, other: Any) -> MongoExpression:
        return self._comparison("le", other)

    def __gt__(self, other: Any) -> MongoExpression:
        return self._comparison("gt", other)

    def __ge__(self, other: Any) -> MongoExpression:
        return self._comparison("ge", other)

    def _set_values(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        if any(value is None for value in values):
            raise ValueError("None is not a valid member of a child-field set operation")
        return values

    def is_in(self, *values: Any) -> MongoExpression:
        """Match a scalar member, or universally constrain a child field."""
        if self._child_keys:
            return MongoExpression(ChildSetNode(self, "has_only", self._set_values(values)))
        return MongoExpression(IsInNode(self, tuple(self._encode(value) for value in values)))

    def has(self, value: Any) -> MongoExpression:
        """Match a child collection containing ``value``."""
        return MongoExpression(ChildSetNode(self, "has_any", self._set_values((value,))))

    def has_any(self, *values: Any) -> MongoExpression:
        """Match a child collection containing any supplied value."""
        return MongoExpression(ChildSetNode(self, "has_any", self._set_values(values)))

    def has_only(self, *values: Any) -> MongoExpression:
        """Match a child collection containing no value outside those supplied."""
        return MongoExpression(ChildSetNode(self, "has_only", self._set_values(values)))

    def contains(self, text: str) -> MongoExpression:
        """Match a literal substring, case-sensitively."""
        if self._child_keys:
            return MongoExpression(ChildStringNode(self, "contains", text))
        return MongoExpression(StringMatchNode(self, "contains", text))

    def startswith(self, prefix: str) -> MongoExpression:
        """Match a literal, case-sensitive prefix."""
        if self._child_keys:
            return MongoExpression(ChildStringNode(self, "startswith", prefix))
        return MongoExpression(StringMatchNode(self, "startswith", prefix))

    def endswith(self, suffix: str) -> MongoExpression:
        """Match a literal, case-sensitive suffix."""
        if self._child_keys:
            return MongoExpression(ChildStringNode(self, "endswith", suffix))
        return MongoExpression(StringMatchNode(self, "endswith", suffix))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(f"{self._path!r} holds a value, not a reference to another record")


class MongoReference:
    """A chainable single-sid reference field."""

    __slots__ = ("_field", "_target_variable", "_variable")

    def __init__(self, variable: "MongoVariable", spec: FieldSpec) -> None:
        assert spec.target is not None
        self._variable = variable
        self._field = MongoField(variable, f"f.{spec.columns[0].name}", spec)
        self._target_variable: MongoVariable | None = None

    @property
    def _spec(self) -> FieldSpec:
        return self._field._spec

    def _target(self) -> "MongoVariable":
        if self._target_variable is None:
            assert self._spec.target is not None
            self._target_variable = self._variable._searcher._reference_variable(self)
        return self._target_variable

    def _target_sid(self, other: Any) -> int:
        assert self._spec.target is not None
        sid = self._variable._searcher._store.sid_of(other, as_record=self._spec.target)
        if sid is None:
            raise ValueError(
                f"the {type(other).__name__} instance compared against {self._variable._cls.__name__}."
                f"{self._spec.field} has not been stored or fetched through this store"
            )
        return sid

    def __eq__(self, other: object) -> MongoExpression:  # type: ignore[override]
        if isinstance(other, MongoVariable):
            target = self._spec.target
            assert target is not None
            if other._cls is not target:
                raise SchemaError(
                    f"{self._variable._cls.__name__}.{self._spec.field} references "
                    f"{target.__name__}, not {other._cls.__name__}"
                )
            return MongoExpression(ReferenceEqualityNode(self, other))
        value = None if other is None else self._target_sid(other)
        return MongoExpression(ComparisonNode(self._field, "eq", value))

    def __ne__(self, other: object) -> MongoExpression:  # type: ignore[override]
        if isinstance(other, MongoVariable):
            return ~self.__eq__(other)
        value = None if other is None else self._target_sid(other)
        return MongoExpression(ComparisonNode(self._field, "ne", value))

    def __hash__(self) -> int:
        return id(self)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._target(), name)


class MongoVariable:
    """A query variable, optionally produced by a lookup from another variable."""

    __slots__ = ("_alias", "_cls", "_references", "_schema", "_searcher", "_source")

    def __init__(self, searcher: "MongoSearcher", cls: type, schema: TableSchema, alias: str) -> None:
        self._searcher = searcher
        self._cls = cls
        self._schema = schema
        self._alias = alias
        self._source: MongoReference | tuple[MongoField, MongoField] | None = None
        self._references: dict[str, MongoReference] = {}

    @property
    def _document_path(self) -> str:
        return "" if self is self._searcher._root else self._alias

    @property
    def sid(self) -> MongoField:
        """Return the store-managed sid field."""
        return MongoField(self, "_id", FieldSpec("sid", int, "scalar", ()))

    def always_true(self) -> MongoExpression:
        """Return an expression matching every stored document."""
        return MongoExpression(AlwaysTrueNode())

    def always_false(self) -> MongoExpression:
        """Return an expression matching no stored document."""
        return MongoExpression(AlwaysFalseNode())

    def __getattr__(self, name: str) -> MongoField | MongoReference:
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "sid":
            return self.sid
        if name == "logical_id":
            # The store-managed lineage id: a top-level document field (like sid
            # and store_timestamp), carrying no unit conversion and, unlike
            # store_timestamp, no store_timestamps=True requirement.
            return MongoField(self, "logical_id", FieldSpec("logical_id", int, "scalar", ()))
        if name == "store_timestamp":
            if not self._searcher._store.store_timestamps:
                raise AttributeError("store_timestamp queries require MongoStore(store_timestamps=True)")
            return MongoField(
                self,
                "store_timestamp",
                FieldSpec("store_timestamp", int, "scalar", ()),
                operand_converter=lambda value: ns_operand_to_store_units(
                    value, cast(int, self._searcher._store.store_timestamp_resolution)
                ),
                presentation_converter=lambda value: (
                    None if value is None else value * cast(int, self._searcher._store.store_timestamp_resolution)
                ),
            )
        try:
            spec = self._schema.field(name)
        except SchemaError:
            raise AttributeError(f"{self._cls.__name__} has no stored field {name!r} to query") from None
        if spec.role == "scalar":
            return MongoField(self, f"f.{spec.columns[0].name}", spec)
        if spec.role == "encoded":
            assert spec.codec_name is not None
            codec = codec_named(spec.codec_name)
            query_column = next(column for column in spec.columns if column.name == spec.field + codec.query_suffix)
            return MongoField(self, f"f.{query_column.name}", spec, codec)
        if spec.role == "reference":
            reference = self._references.get(spec.field)
            if reference is None:
                reference = MongoReference(self, spec)
                self._references[spec.field] = reference
            return reference
        if spec.role == "child":
            assert spec.child is not None
            child_codec: ValueCodec | None = codec_named(spec.codec_name) if spec.codec_name is not None else None
            return MongoField(
                self, f"f.{spec.field}", spec, child_codec, tuple(column.name for column in spec.child.element_columns)
            )
        raise SchemaError(
            f"{self._cls.__name__}.{spec.field} is a fixed-shape tensor field and cannot be queried as a whole"
        )


@dataclass(frozen=True, slots=True)
class _MongoOutput:
    """One object or scalar projection in a frozen Mongo result plan."""

    name: str
    value: MongoVariable | MongoField


class MongoSearcher:
    """Build and execute a MongoDB query over reference-connected variables.

    An historic cutoff is injected for every root and lookup variable; visible
    rows' dependencies are always visible because references only point at
    earlier-or-equal rows from the same transaction. When ``only_latest`` is set,
    every declared (root) variable is additionally restricted to the latest
    document of its ``logical_id`` lineage by sid (bounded by ``as_of`` when
    given); reference/lookup variables stay unfiltered so pinned references may
    still resolve replaced documents.
    """

    def __init__(self, store: "MongoStore", *, as_of: object = None, only_latest: bool = False) -> None:
        self._store = store
        self._as_of = as_of
        self._only_latest = only_latest
        self._variables: list[MongoVariable] = []
        self._hidden_variables: list[MongoVariable] = []
        self._root: MongoVariable | None = None
        self._expressions: list[MongoExpression] = []
        self._outputs: list[_MongoOutput] = []
        self._sorts: list[tuple[MongoField, bool, str]] = []
        self._limit: int | None = None
        self.offset = 0
        # Phase 5 attaches its client-authoritative predicate evaluator here.
        # MongoResultSet owns the single candidate iterator that applies it.
        self._row_verifier: Callable[[dict[str, Any]], bool] | None = None
        self._row_verifier_identity: str | bytes | None = None

    def set_row_verifier(self, verifier: Callable[[dict[str, Any]], bool], identity: str | bytes) -> None:
        """Attach the client-authoritative candidate verifier and its frozen identity.

        The identity must be the canonical logical predicate payload, including
        every verifier constant.  It is folded into continuation fingerprints,
        so cursors never cross between otherwise identical server plans.

        :param verifier: Return whether a server candidate is a real match.
        :param identity: Canonical text or bytes identifying the verifier logic.
        :raises TypeError: If either attachment component has the wrong type.
        """
        if not callable(verifier):
            raise TypeError("row verifier must be callable")
        if not isinstance(identity, (str, bytes)):
            raise TypeError("row verifier identity must be str or bytes")
        if not identity:
            raise ValueError("row verifier identity must not be empty")
        self._row_verifier = verifier
        self._row_verifier_identity = identity

    def _require_verifier_identity(self) -> None:
        """Reject an unsafe direct verifier attachment lacking its identity."""
        if self._row_verifier is not None and self._row_verifier_identity is None:
            raise ValueError("row verifier requires a canonical identity payload")
        if self._row_verifier is None and self._row_verifier_identity is not None:
            raise ValueError("row verifier identity requires a row verifier")

    def variable(self, target: type) -> MongoVariable:
        """Bind a query variable; each additional one must be join-connected."""
        variable = MongoVariable(self, target, resolve_schema(target), f"_httk_var_{len(self._variables)}")
        self._variables.append(variable)
        if self._root is None:
            self._root = variable
        if self._as_of is not None:
            self.add(cast(MongoField, variable.store_timestamp) <= self._as_of)
        return variable

    def _root_sid_field(self) -> MongoField:
        """Return the store-managed SID field of this searcher's root variable.

        Backend wiring may project a nested searcher's SID without depending on
        the searcher's private variable-list layout.

        :return: The root variable's SID query field.
        :raises ValueError: If no root variable has been declared.
        """
        if self._root is None:
            raise ValueError("this searcher has no query variables; call variable() first")
        return self._root.sid

    def _reference_variable(self, reference: MongoReference) -> MongoVariable:
        assert reference._spec.target is not None
        variable = MongoVariable(
            self,
            reference._spec.target,
            resolve_schema(reference._spec.target),
            f"_httk_ref_{len(self._hidden_variables)}",
        )
        variable._source = reference
        self._hidden_variables.append(variable)
        if self._as_of is not None:
            self.add(cast(MongoField, variable.store_timestamp) <= self._as_of)
        return variable

    def output(self, variable: MongoVariable | MongoField, name: str) -> None:
        """Declare an object variable or scalar field output."""
        if not isinstance(variable, (MongoVariable, MongoField)):
            raise TypeError(f"output() takes a Mongo variable or field, got {type(variable).__name__}")
        self._outputs.append(_MongoOutput(name, variable))

    def add(self, expression: MongoExpression) -> None:
        """Add a filter expression, conjoined with earlier filters."""
        if not isinstance(expression, MongoExpression):
            raise TypeError(f"add() takes a MongoExpression, got {type(expression).__name__}")
        self._expressions.append(expression)

    def add_sort(
        self, field: MongoField, descending: bool = False, *, nulls: Literal["first", "last"] = "last"
    ) -> None:
        """Append a stable scalar sort with an explicit null rank."""
        if not isinstance(field, MongoField):
            raise TypeError(f"add_sort() takes a MongoField, got {type(field).__name__}")
        if nulls not in {"first", "last"}:
            raise ValueError("nulls must be 'first' or 'last'")
        self._sorts.append((field, descending, nulls))

    def set_limit(self, limit: int) -> None:
        """Set the iteration limit; negative values clear it."""
        self._limit = None if limit < 0 else limit

    def add_offset(self, offset: int) -> None:
        """Add rows to the iteration offset."""
        self.offset += offset

    def _required_nodes(self, node: Node) -> Iterator[Node]:
        """Yield predicates in positive conjunctive positions only.

        A lookup is a physical restriction, so an equality below ``OR`` or
        ``NOT`` cannot establish a join: that branch may be irrelevant to a
        matching result, and constraining it would silently lose rows.
        """
        if isinstance(node, AndNode):
            yield from self._required_nodes(node.left)
            yield from self._required_nodes(node.right)
        elif not isinstance(node, (OrNode, NotNode)):
            yield node

    def _bind_joins(self) -> None:
        """Infer lookup sources from reference and scalar equality conditions."""
        if self._root is None:
            raise ValueError("this searcher has no query variables; call variable() first")
        nodes = [node for expression in self._expressions for node in self._required_nodes(expression.node)]
        changed = True
        while changed:
            changed = False
            for node in nodes:
                if isinstance(node, ReferenceEqualityNode) and node.variable._source is None:
                    chained = node.reference._target_variable
                    if chained is not None:
                        # A chain may have named this lookup before a second
                        # declared variable compares equal to it.  Both names
                        # then denote the one reference path/lookup alias.
                        node.variable._alias = chained._alias
                    node.variable._source = node.reference
                    changed = True
                if (
                    isinstance(node, ComparisonNode)
                    and node.op == "eq"
                    and isinstance(node.literal, _FieldReference)
                    and node.field._variable is not node.literal.field._variable
                ):
                    left, right = node.field._variable, node.literal.field._variable
                    if (
                        (left is self._root or left._source is not None)
                        and right._source is None
                        and right is not self._root
                    ):
                        right._source = (node.field, node.literal.field)
                        changed = True
                    elif (
                        (right is self._root or right._source is not None)
                        and left._source is None
                        and left is not self._root
                    ):
                        left._source = (node.literal.field, node.field)
                        changed = True
        disconnected = [
            variable for variable in self._variables if variable is not self._root and variable._source is None
        ]
        if disconnected:
            raise UnsupportedQueryError(
                "Mongo search supports reference-connected variables and equality joins; disconnected cartesian "
                "queries are not supported"
            )

    def _lookup_stages(self) -> list[dict[str, Any]]:
        self._bind_joins()
        pending: list[MongoVariable] = []
        aliases: set[str] = set()
        for variable in [*self._variables, *self._hidden_variables]:
            if variable is self._root or variable._alias in aliases:
                continue
            aliases.add(variable._alias)
            pending.append(variable)
        stages: list[dict[str, Any]] = []
        emitted: set[MongoVariable] = set()
        while pending:
            progress = False
            for variable in tuple(pending):
                source = variable._source
                if isinstance(source, MongoReference):
                    parent = source._variable
                    if parent is not self._root and parent not in emitted:
                        continue
                    local = source._field._path
                    pipeline = [{"$match": {"$expr": {"$eq": ["$_id", "$$local"]}}}]
                elif isinstance(source, tuple):
                    outer, foreign = source
                    if outer._variable is not self._root and outer._variable not in emitted:
                        continue
                    local = outer._path
                    pipeline = [{"$match": {"$expr": {"$eq": [f"${foreign._key_path}", "$$local"]}}}]
                else:
                    continue
                from httk.store.backend.mongo.mapping import collection_name_for

                stages.extend(
                    [
                        {
                            "$lookup": {
                                "from": collection_name_for(variable._schema),
                                "let": {"local": f"${local}"},
                                "pipeline": pipeline,
                                "as": variable._alias,
                            }
                        },
                        {"$unwind": {"path": f"${variable._alias}", "preserveNullAndEmptyArrays": True}},
                    ]
                )
                emitted.add(variable)
                pending.remove(variable)
                progress = True
            if not progress:
                raise UnsupportedQueryError("Mongo search contains a cyclic or disconnected variable join")
        return stages

    def _truth_filter(self) -> dict[str, Any]:
        return _and(*(render_node(expression.node)[0] for expression in self._expressions))

    def _latest_lineage_stages(self) -> list[dict[str, Any]]:
        """Restrict each declared variable to the latest document of its lineage.

        For every declared (root) variable a correlated self-``$lookup`` finds a
        newer sibling — a same-collection document with equal ``logical_id`` and a
        greater sid, additionally bounded by ``store_timestamp <= as_of`` when a
        cutoff is set — and the following ``$match`` keeps only rows with no such
        sibling. Reference/lookup variables are deliberately left unfiltered.

        :return: The correlated self-join and anti-match stages, in order.
        """
        from httk.store.backend.mongo.mapping import collection_name_for

        as_of_units = (
            ns_operand_to_store_units(self._as_of, cast(int, self._store.store_timestamp_resolution))
            if self._as_of is not None
            else None
        )
        stages: list[dict[str, Any]] = []
        for index, variable in enumerate(self._variables):
            prefix = "" if variable is self._root else f"{variable._alias}."
            newer: list[dict[str, Any]] = [
                {"$eq": ["$logical_id", "$$lid"]},
                {"$gt": ["$_id", "$$sid"]},
            ]
            if as_of_units is not None:
                newer.append({"$lte": ["$store_timestamp", as_of_units]})
            alias = f"_httk_latest_{index}"
            stages.append(
                {
                    "$lookup": {
                        "from": collection_name_for(variable._schema),
                        "let": {"lid": f"${prefix}logical_id", "sid": f"${prefix}_id"},
                        "pipeline": [{"$match": {"$expr": {"$and": newer}}}, {"$limit": 1}],
                        "as": alias,
                    }
                }
            )
            stages.append({"$match": {alias: {"$eq": []}}})
        return stages

    def _pipeline(
        self,
        *,
        count: bool = False,
        limit: int | None = None,
        apply_window: bool = True,
    ) -> list[dict[str, Any]]:
        if self._root is None:
            raise ValueError("this searcher has no query variables; call variable() first")
        self._require_verifier_identity()
        pipeline = self._lookup_stages()
        pipeline.append({"$match": self._truth_filter()})
        if self._only_latest:
            pipeline.extend(self._latest_lineage_stages())
        if count:
            pipeline.append({"$count": "count"})
            return pipeline
        for index, (field, _descending, nulls) in enumerate(self._sorts):
            rank = 0 if nulls == "first" else 1
            pipeline.append(
                {
                    "$addFields": {
                        f"_httk_sort_{index}_rank": {
                            "$cond": [{"$in": [{"$type": f"${field._path}"}, ["missing", "null"]]}, rank, 1 - rank]
                        }
                    }
                }
            )
        if self._sorts:
            order: dict[str, int] = {}
            for index, (field, descending, _nulls) in enumerate(self._sorts):
                order[f"_httk_sort_{index}_rank"] = 1
                order[field._path] = -1 if descending else 1
            order["_id"] = 1
            pipeline.append({"$sort": order})
        server_window = apply_window and self._row_verifier is None
        if server_window and self.offset > 0:
            pipeline.append({"$skip": self.offset})
        effective_limit = self._limit if limit is None else limit
        if server_window and effective_limit is not None:
            pipeline.append({"$limit": effective_limit})
        return pipeline

    def count(self) -> int:
        """Return the exact filtered count, ignoring offset and limit."""
        if self._row_verifier is not None:
            return sum(1 for _document in self._verified_documents(self._pipeline(apply_window=False)))
        row = next(
            iter(self._collection().aggregate(self._pipeline(count=True), **self._store._session_kwargs())), None
        )
        return 0 if row is None else int(row["count"])

    def _collection(self) -> Any:
        if self._root is None:
            raise ValueError("this searcher has no query variables; call variable() first")
        from httk.store.backend.mongo.mapping import collection_name_for

        return self._store._database.database[collection_name_for(self._root._schema)]

    def _candidate_documents(self, pipeline: list[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], bool]]:
        """Yield every server candidate and whether the optional verifier accepts it."""
        self._require_verifier_identity()
        verifier = self._row_verifier
        for document in self._collection().aggregate(pipeline, **self._store._session_kwargs()):
            yield document, verifier is None or verifier(document)

    def _verified_documents(
        self, pipeline: list[dict[str, Any]], *, apply_window: bool = False
    ) -> Iterator[dict[str, Any]]:
        """Yield accepted candidates, applying offset and limit after verification."""
        documents = (document for document, verified in self._candidate_documents(pipeline) if verified)
        if not apply_window or self._row_verifier is None:
            return documents
        start = max(self.offset, 0)
        stop = None if self._limit is None else start + self._limit
        return islice(documents, start, stop)

    def _execute(self, outputs: list[_MongoOutput] | None = None) -> list[tuple[Any, ...]]:
        chosen = self._outputs if outputs is None else outputs
        if not chosen:
            raise ValueError("this search has no outputs; declare outputs or pass them to results()")
        if self._row_verifier is None:
            documents = list(self._collection().aggregate(self._pipeline(), **self._store._session_kwargs()))
        else:
            documents = list(self._verified_documents(self._pipeline(apply_window=False), apply_window=True))
        values: list[tuple[Any, ...]] = []
        for document in documents:
            row: list[Any] = []
            for output in chosen:
                if isinstance(output.value, MongoVariable):
                    object_document = _variable_document(document, output.value)
                    if object_document is None:
                        row.append(None)
                    else:
                        row.append(self._store.fetch(output.value._cls, int(object_document["_id"])))
                else:
                    row.append(_scalar_value(document, output.value))
            values.append(tuple(row))
        return values

    def __iter__(self) -> Iterator[SearchResult]:
        """Yield declared outputs as neutral search results."""
        names = tuple(output.name for output in self._outputs)
        return iter(SearchResult(values, names) for values in self._execute())

    def results(self, **outputs: Any) -> Any:
        """Return a materialized :class:`~httk.store.backend.mongo.results.MongoResultSet` for this query."""
        selected = [_MongoOutput(name, value) for name, value in outputs.items()] if outputs else None
        return __import__("httk.store.backend.mongo.results", fromlist=["MongoResultSet"]).MongoResultSet(
            self, selected
        )


def _variable_document(document: dict[str, Any], variable: MongoVariable) -> dict[str, Any] | None:
    """Return the root or lookup document represented by ``variable``."""
    if variable is variable._searcher._root:
        return document
    value = document.get(variable._alias)
    return value if isinstance(value, dict) else None


def _scalar_value(document: dict[str, Any], field: MongoField) -> Any:
    """Decode one scalar projection from a root or lookup record document."""
    source = _variable_document(document, field._variable)
    if source is None:
        return None
    if field._key_path == "_id":
        return source.get("_id")
    if field._key_path == "logical_id":
        return source.get("logical_id")
    if field._key_path == "content_id":
        value = source.get("content_id")
        return None if value is None else field._presentation_prefix + value
    if field._key_path == "store_timestamp":
        value = source.get("store_timestamp")
        return (
            None
            if value is None
            else (field._presentation_converter(value) if field._presentation_converter is not None else value)
        )
    embedded = source.get("f", {})
    spec = field._spec
    if spec.role == "scalar":
        return embedded.get(field._key_path.removeprefix("f."))
    assert field._codec is not None
    parts = [embedded.get(spec.field + suffix) for suffix, _kind in field._codec.columns]
    return None if all(part is None for part in parts) else field._codec.decode(tuple(parts))
