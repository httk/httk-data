"""Encoding and decoding of MongoStore record documents."""

import inspect
import typing
from collections.abc import Callable, Mapping
from typing import Any

from bson import encode as bson_encode
from httk.core import FracVector
from httk.core.storage import Shape

from httk.data.db.codecs import (
    codec_named,
    decode_fracvector_exact,
    encode_fracvector_exact,
    encode_fracvector_floats,
)
from httk.data.db.schema import FieldSpec, TableSchema

from .mapping import document_fields_for

__all__ = ["RecordTooLargeError", "decode_record", "encode_record", "preflight_document"]


class RecordTooLargeError(ValueError):
    """A record document exceeds the connected MongoDB BSON limit."""


def _as_fixed_tensor(schema: TableSchema, spec: FieldSpec, shape: Shape, value: Any) -> FracVector:
    tensor = FracVector(value)
    if tensor.dim == (shape.rows, shape.cols):
        return tensor
    if shape.rows == 1 and tensor.dim == (shape.cols,):
        return FracVector.from_noms_and_denom((tensor.noms,), tensor.denom)
    raise ValueError(
        f"{schema.cls.__name__}.{spec.field}: expected a FracVector of shape "
        f"({shape.rows}, {shape.cols}), got {tensor.dim}"
    )


def _tensor_rows(schema: TableSchema, spec: FieldSpec, shape: Shape, value: Any) -> list[FracVector]:
    if value is None:
        return []
    tensor = FracVector(value)
    if tensor.dim in {(), (0,)}:
        return []
    if len(tensor.dim) != 2 or tensor.dim[1] != shape.cols:
        raise ValueError(
            f"{schema.cls.__name__}.{spec.field}: expected a FracVector with {shape.cols} columns per row, "
            f"got shape {tensor.dim}"
        )
    rows = typing.cast(tuple[tuple[int, ...], ...], tensor.noms)
    return [FracVector.from_noms_and_denom(row, tensor.denom) for row in rows]


def _value(record_type: type, source: Any, projected: Mapping[str, object], spec: FieldSpec) -> Any:
    if spec.field in projected:
        return projected[spec.field]
    if spec.derived:
        try:
            return getattr(source, spec.field)
        except AttributeError:
            raise TypeError(
                f"projecting {type(source).__name__} as {record_type.__name__} requires the source "
                f"to expose derived stored property {spec.field!r}"
            ) from None
    raise ValueError(f"projection for {type(source).__name__} omitted stored field {spec.field!r}")


def _resolve_reference(callback: Callable[..., int], target: type, value: Any, field: str) -> int:
    """Call both the public two-argument and store-internal path-aware forms."""
    if len(inspect.signature(callback).parameters) >= 3:
        return callback(target, value, field)
    return callback(target, value)


def encode_record(
    schema: TableSchema,
    projected: Mapping[str, object],
    source: Any,
    record_type: type,
    resolve_reference: Callable[..., int],
) -> dict[str, Any]:
    """Encode projected field values into the record's ``f`` document.

    :param schema: The resolved record schema.
    :param projected: The backend-neutral storage projection.
    :param source: The source object used for derived stored properties.
    :param record_type: The record representation being encoded.
    :param resolve_reference: Callback saving a reference and returning its sid.
    :return: The embedded ``f`` document.
    """
    fields = {item.field: item for item in document_fields_for(schema)}
    result: dict[str, Any] = {}
    for spec in schema.fields:
        plan = fields[spec.field]
        value = _value(record_type, source, projected, spec)
        if value is None:
            if spec.role == "child" and not spec.optional:
                value = ()
            elif not spec.optional:
                raise ValueError(f"{record_type.__name__}.{spec.field} cannot be None")
            else:
                continue
        if spec.role == "child":
            assert spec.child is not None
            elements: list[Any] = []
            if spec.shape is not None:
                assert spec.shape is not None
                for row in _tensor_rows(schema, spec, spec.shape, value):
                    parts = encode_fracvector_floats(row)
                    elements.append(
                        {
                            key: part
                            for key, part in zip(
                                (key for key in plan.element_keys if not key.endswith("_exact")), parts, strict=True
                            )
                        }
                        | {f"{spec.field}_exact": encode_fracvector_exact(row)}
                    )
            else:
                codec = codec_named(spec.codec_name) if spec.codec_name is not None else None
                for element in typing.cast(typing.Iterable[Any], value):
                    if spec.target is not None:
                        elements.append(
                            {
                                plan.element_keys[0]: _resolve_reference(
                                    callback=resolve_reference,
                                    target=spec.target,
                                    value=element,
                                    field=f"{spec.field}[{len(elements)}]",
                                )
                            }
                        )
                    elif codec is not None:
                        elements.append(
                            {key: part for key, part in zip(plan.element_keys, codec.encode(element), strict=True)}
                        )
                    else:
                        elements.append({plan.element_keys[0]: element})
            result[spec.field] = elements
        elif spec.role == "scalar":
            result[plan.keys[0]] = value
        elif spec.role == "encoded":
            assert spec.codec_name is not None
            result.update(dict(zip(plan.keys, codec_named(spec.codec_name).encode(value), strict=True)))
        elif spec.role == "fixed_array":
            assert spec.shape is not None
            tensor = _as_fixed_tensor(schema, spec, spec.shape, value)
            result.update(
                {f"{spec.field}_{index}": part for index, part in enumerate(encode_fracvector_floats(tensor))}
            )
            result[f"{spec.field}_exact"] = encode_fracvector_exact(tensor)
        else:
            assert spec.target is not None
            result[plan.keys[0]] = _resolve_reference(
                callback=resolve_reference, target=spec.target, value=value, field=spec.field
            )
    return result


def _decode_child(spec: FieldSpec, value: Any, resolve_reference: Callable[[type, int], Any]) -> Any:
    if value is None:
        return None
    assert spec.child is not None
    if spec.shape is not None:
        rows = [
            decode_fracvector_exact(element[f"{spec.field}_exact"], 1, spec.shape.cols).to_fractions()[0]
            for element in value
        ]
        return FracVector(rows)
    if spec.target is not None:
        elements = [
            resolve_reference(spec.target, int(element[spec.child.element_columns[0].name])) for element in value
        ]
    elif spec.codec_name is not None:
        codec = codec_named(spec.codec_name)
        elements = [
            codec.decode(tuple(element[column.name] for column in spec.child.element_columns)) for element in value
        ]
    else:
        elements = [element[spec.child.element_columns[0].name] for element in value]
    return tuple(elements) if typing.get_origin(spec.python_type) is tuple else elements


def decode_record(
    schema: TableSchema, document: Mapping[str, Any], resolve_reference: Callable[[type, int], Any]
) -> Any:
    """Decode a MongoDB record document into its concrete storable instance.

    :param schema: The resolved record schema.
    :param document: The complete MongoDB record document.
    :param resolve_reference: Callback hydrating a referenced sid.
    :return: A concrete record instance.
    """
    fields = {item.field: item for item in document_fields_for(schema)}
    embedded = document.get("f", {})
    values: dict[str, Any] = {}
    for spec in schema.fields:
        if spec.derived:
            continue
        plan = fields[spec.field]
        if spec.role == "child":
            values[spec.field] = _decode_child(spec, embedded.get(spec.field), resolve_reference)
        elif spec.role == "scalar":
            values[spec.field] = embedded.get(plan.keys[0])
        elif spec.role == "encoded":
            assert spec.codec_name is not None
            parts = tuple(embedded.get(key) for key in plan.keys)
            values[spec.field] = (
                None if all(part is None for part in parts) else codec_named(spec.codec_name).decode(parts)
            )
        elif spec.role == "fixed_array":
            assert spec.shape is not None
            exact = embedded.get(f"{spec.field}_exact")
            values[spec.field] = (
                None if exact is None else decode_fracvector_exact(exact, spec.shape.rows, spec.shape.cols)
            )
        else:
            assert spec.target is not None
            sid = embedded.get(plan.keys[0])
            values[spec.field] = None if sid is None else resolve_reference(spec.target, int(sid))
    return schema.cls(**values)


def preflight_document(document: Mapping[str, Any], max_bson_size: int, record_type: type) -> None:
    """Raise when ``document`` cannot be accepted by MongoDB's BSON limit.

    :param document: Candidate record document.
    :param max_bson_size: The server BSON limit read at store construction.
    :param record_type: The record class used in the diagnostic.
    :return: None.
    :raises RecordTooLargeError: If BSON encoding exceeds the server limit.
    """
    size = len(bson_encode(dict(document)))
    if size > max_bson_size:
        raise RecordTooLargeError(
            f"{record_type.__name__} document is {size} bytes, exceeding MongoDB's {max_bson_size}-byte BSON limit"
        )
