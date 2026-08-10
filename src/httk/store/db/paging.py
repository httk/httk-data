"""Compatibility shim for the neutral continuation-token codec."""

from httk.store.query.paging_tokens import (
    _decode_continuation,  # noqa: F401
    _DecodedContinuation,  # noqa: F401
    _encode_continuation,  # noqa: F401
    _plan_fingerprint,  # noqa: F401
)

__all__ = []
