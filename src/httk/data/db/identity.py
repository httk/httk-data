"""Content identity: a canonical text form and SHA-256 identity for storable instances.

:func:`content_id` gives every storable instance a deterministic identity
derived purely from its stored, non-derived field values — the basis of the
store layer's ``"content_id"`` deduplication policy and of its store-managed
``content_id`` column. Two instances (even of distinct object identity, even
holding distinct-but-equal nested instances) get the same identity exactly when
their stored values are equal.

The identity hashes :func:`canonical_form`: a compact JSON text of
``[table_name, [[field, canonical value], ...]]`` with the stored non-derived
fields sorted by field name. Canonical values are chosen to be exact and
deterministic per field role:

- rationals (:class:`fractions.Fraction`, :class:`~httk.core.FracScalar`,
  :class:`~httk.core.SurdScalar`) — their canonical exact text;
- :class:`~httk.core.FracVector` tensors — the canonical exact tensor text;
- ``float`` — its ``repr`` (shortest exact round-trip text);
- ``bytes`` — its hex text;
- ``str``/``int``/``bool``/``None`` — as-is;
- ``datetime`` (and any single-text-column codec) — its encoded text;
- lists/tuples — JSON arrays of their elements' canonical values, in order;
- nested storable instances — their :func:`content_id`, recursively.

Derived (:class:`~httk.core.stored_property`) values are functions of the
fields and are excluded, so they can never skew the identity.
"""

import hashlib
import json
from typing import Any

from httk.data.db.codecs import ValueCodec, codec_named, encode_fracvector_exact
from httk.data.db.schema import FieldSpec, resolve_schema

__all__ = [
    "canonical_form",
    "content_id",
]


def canonical_form(obj: Any) -> str:
    """The deterministic canonical JSON text of a storable instance's stored values.

    Resolves the schema of ``obj``'s class (so the class must be storable) and
    serializes ``[table_name, [[field, canonical value], ...]]`` over the
    non-derived stored fields, sorted by field name, as compact JSON.
    """
    schema = resolve_schema(type(obj))
    entries = [
        [spec.field, _canonical_value(spec, getattr(obj, spec.field))]
        for spec in sorted((spec for spec in schema.fields if not spec.derived), key=lambda spec: spec.field)
    ]
    return json.dumps([schema.table_name, entries], sort_keys=True, separators=(",", ":"))


def content_id(obj: Any) -> str:
    """The SHA-256 hex digest (64 hex characters) of :func:`canonical_form` of ``obj``."""
    return hashlib.sha256(canonical_form(obj).encode("utf-8")).hexdigest()


def _canonical_value(spec: FieldSpec, value: Any) -> Any:
    if value is None:
        return None
    if spec.role == "scalar":
        return _canonical_scalar(value)
    if spec.role == "encoded":
        assert spec.codec_name is not None
        return _canonical_encoded(codec_named(spec.codec_name), value)
    if spec.role == "fixed_array":
        return encode_fracvector_exact(value)
    if spec.role == "reference":
        return content_id(value)
    # Child role: a Shape(0, c) child holds one FracVector of variable rows;
    # list/tuple children hold their elements in insertion order.
    if spec.shape is not None:
        return encode_fracvector_exact(value)
    if spec.target is not None:
        return [content_id(element) for element in value]
    if spec.codec_name is not None:
        codec = codec_named(spec.codec_name)
        return [_canonical_encoded(codec, element) for element in value]
    return [_canonical_scalar(element) for element in value]


def _canonical_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, bytes):
        return value.hex()
    return value


def _canonical_encoded(codec: ValueCodec, value: Any) -> Any:
    encoded = codec.encode(value)
    texts = [encoded[i] for i, (_, kind) in enumerate(codec.columns) if kind == "str"]
    if len(texts) == 1:
        return texts[0]
    return [_canonical_scalar(part) for part in encoded]
