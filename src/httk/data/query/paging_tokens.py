"""Private canonical continuation-token primitives for any keyset-paging backend.

The token is intentionally an opaque, transport-safe value rather than a SQL
fragment.  Its payload carries only typed anchor values and a digest of the
frozen result plan; every decoded anchor is subsequently passed to SQLAlchemy
as a normal bound parameter.
"""

import base64
import datetime
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from httk.data.query import ContinuationToken, PaginationCursorError

__all__ = []

_TOKEN_VERSION = 1
_MAX_TOKEN_CHARS = 12_000
_MAX_JSON_BYTES = 9_000
_MAX_ANCHORS = 32
_MAX_STRING_CHARS = 4_096
_MAX_BYTES = 4_096
_MAX_INT_DIGITS = 128
_URLSAFE = re.compile(r"[A-Za-z0-9_-]+\Z")


@dataclass(frozen=True, slots=True)
class _DecodedContinuation:
    direction: Literal["forward", "backward"]
    anchors: tuple[Any, ...]
    sid: int


def _encode_continuation(
    *,
    direction: Literal["forward", "backward"],
    anchors: tuple[Any, ...],
    sid: int,
    fingerprint: str,
) -> ContinuationToken:
    """Encode a bounded canonical continuation payload."""
    if direction not in {"forward", "backward"}:
        raise ValueError(f"unknown continuation direction {direction!r}")
    _validate_sid(sid)
    if not _valid_fingerprint(fingerprint):
        raise ValueError("continuation fingerprint must be a SHA-256 hexadecimal digest")
    if len(anchors) > _MAX_ANCHORS:
        raise ValueError(f"continuation supports at most {_MAX_ANCHORS} order anchors")
    payload = {
        "a": [_encode_value(value) for value in anchors],
        "d": direction,
        "f": fingerprint,
        "s": sid,
        "v": _TOKEN_VERSION,
    }
    encoded = _encode_payload(payload)
    if len(encoded) > _MAX_TOKEN_CHARS:
        raise ValueError("continuation token exceeds the supported size")
    return ContinuationToken(encoded)


def _decode_continuation(token: ContinuationToken, *, fingerprint: str, anchors: int) -> _DecodedContinuation:
    """Strictly decode ``token`` and verify that it belongs to this result plan."""
    if not isinstance(token, ContinuationToken):
        raise PaginationCursorError("cursor must be a ContinuationToken returned by page()")
    if not _valid_fingerprint(fingerprint):
        raise PaginationCursorError("paging configuration has an invalid fingerprint")
    if not isinstance(anchors, int) or isinstance(anchors, bool) or not 0 <= anchors <= _MAX_ANCHORS:
        raise PaginationCursorError("paging configuration has an invalid order-key count")
    text = str(token)
    if not text or len(text) > _MAX_TOKEN_CHARS or _URLSAFE.fullmatch(text) is None:
        raise PaginationCursorError("cursor is not a valid URL-safe continuation token")
    try:
        padding = "=" * (-len(text) % 4)
        raw = base64.urlsafe_b64decode(text + padding)
    except (ValueError, UnicodeError) as error:
        raise PaginationCursorError("cursor is not valid base64url data") from error
    if len(raw) > _MAX_JSON_BYTES:
        raise PaginationCursorError("cursor payload exceeds the supported size")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PaginationCursorError("cursor does not contain a valid continuation payload") from error
    if not isinstance(payload, dict) or set(payload) != {"a", "d", "f", "s", "v"}:
        raise PaginationCursorError("cursor has an invalid continuation payload shape")
    if isinstance(payload["v"], bool) or not isinstance(payload["v"], int) or payload["v"] != _TOKEN_VERSION:
        raise PaginationCursorError(f"unsupported continuation cursor version {payload['v']!r}")
    if payload["f"] != fingerprint:
        raise PaginationCursorError("cursor belongs to a different query, order, schema, or SQL dialect")
    direction = payload["d"]
    if direction not in {"forward", "backward"}:
        raise PaginationCursorError("cursor has an invalid continuation direction")
    raw_anchors = payload["a"]
    if not isinstance(raw_anchors, list) or len(raw_anchors) != anchors:
        raise PaginationCursorError("cursor has a different number of order anchors")
    try:
        values = tuple(_decode_value(value) for value in raw_anchors)
        sid = payload["s"]
        _validate_sid(sid)
    except (TypeError, ValueError) as error:
        raise PaginationCursorError("cursor contains an invalid anchor value") from error
    # The bytes must be canonical too: accepting alternate JSON spellings would
    # undermine the advertised strict, deterministic wire value.
    if _encode_payload(payload) != text:
        raise PaginationCursorError("cursor is not canonically encoded")
    return _DecodedContinuation(direction, values, sid)


def _plan_fingerprint(payload: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON-safe plan description."""
    normalized = _fingerprint_value(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _encode_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _encode_value(value: Any) -> dict[str, str | int | bool | None]:
    if value is None:
        return {"t": "null", "v": None}
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        _validate_int(value)
        return {"t": "int", "v": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("continuation anchors cannot be non-finite floats")
        # ``float.hex`` is exact, portable, and unlike decimal rendering never
        # changes the key selected on the next request.
        return {"t": "float", "v": value.hex()}
    if isinstance(value, str):
        if len(value) > _MAX_STRING_CHARS:
            raise ValueError("continuation string anchor exceeds the supported size")
        return {"t": "str", "v": value}
    if isinstance(value, bytes):
        if len(value) > _MAX_BYTES:
            raise ValueError("continuation bytes anchor exceeds the supported size")
        return {"t": "bytes", "v": base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")}
    if isinstance(value, datetime.datetime):
        return {"t": "datetime", "v": value.isoformat()}
    if isinstance(value, datetime.date):
        return {"t": "date", "v": value.isoformat()}
    raise ValueError(f"continuation does not support anchor values of type {type(value).__name__}")


def _decode_value(value: Any) -> Any:
    if not isinstance(value, dict) or set(value) != {"t", "v"} or not isinstance(value["t"], str):
        raise ValueError("invalid typed anchor")
    tag = value["t"]
    raw = value["v"]
    if tag == "null" and raw is None:
        return None
    if tag == "bool" and isinstance(raw, bool):
        return raw
    if tag == "int" and isinstance(raw, str):
        if not raw or len(raw.lstrip("-")) > _MAX_INT_DIGITS or (raw.startswith("-") and raw == "-"):
            raise ValueError("invalid integer anchor")
        integer = int(raw)
        if str(integer) != raw:
            raise ValueError("non-canonical integer anchor")
        return integer
    if tag == "float" and isinstance(raw, str):
        floating = float.fromhex(raw)
        if not math.isfinite(floating) or floating.hex() != raw:
            raise ValueError("invalid float anchor")
        return floating
    if tag == "str" and isinstance(raw, str) and len(raw) <= _MAX_STRING_CHARS:
        return raw
    if tag == "bytes" and isinstance(raw, str) and len(raw) <= 2 * _MAX_BYTES:
        if raw and _URLSAFE.fullmatch(raw) is None:
            raise ValueError("invalid bytes anchor")
        decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        if len(decoded) > _MAX_BYTES or base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != raw:
            raise ValueError("non-canonical bytes anchor")
        return decoded
    if tag == "datetime" and isinstance(raw, str) and len(raw) <= _MAX_STRING_CHARS:
        datetime_value = datetime.datetime.fromisoformat(raw)
        if datetime_value.isoformat() != raw:
            raise ValueError("non-canonical datetime anchor")
        return datetime_value
    if tag == "date" and isinstance(raw, str) and len(raw) <= _MAX_STRING_CHARS:
        date_value = datetime.date.fromisoformat(raw)
        if date_value.isoformat() != raw:
            raise ValueError("non-canonical date anchor")
        return date_value
    raise ValueError("unknown or malformed anchor tag")


def _fingerprint_value(value: Any) -> Any:
    """Normalize SQLAlchemy compile parameters without putting them in a token.

    Query predicates eventually bind one of the storage scalar types.  The
    fallback deliberately includes the concrete class and ``repr`` so a custom
    value cannot make two different frozen plans share a fingerprint; it is
    hashed locally and never exposed in a continuation payload.
    """
    try:
        return _encode_value(value)
    except ValueError:
        pass
    if isinstance(value, tuple):
        return {"t": "tuple", "v": [_fingerprint_value(item) for item in value]}
    if isinstance(value, list):
        return {"t": "list", "v": [_fingerprint_value(item) for item in value]}
    if isinstance(value, dict):
        pairs = sorted((repr(key), _fingerprint_value(item)) for key, item in value.items())
        return {"t": "dict", "v": pairs}
    cls = type(value)
    return {"t": "repr", "c": f"{cls.__module__}.{cls.__qualname__}", "v": repr(value)}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value!r} is not allowed")


def _validate_int(value: int) -> None:
    if len(str(abs(value))) > _MAX_INT_DIGITS:
        raise ValueError("continuation integer anchor exceeds the supported size")


def _validate_sid(sid: Any) -> None:
    if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
        raise ValueError("continuation sid must be a positive integer")
    _validate_int(sid)


def _valid_fingerprint(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
