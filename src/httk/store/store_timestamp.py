"""Backend-neutral store-managed timestamp helpers."""

import datetime
import time
from collections.abc import Callable

__all__ = [
    "FUTURE_TIMESTAMP_SLACK_NS",
    "_CLOCK_REGRESSION_GRACE_NS",
    "StoreClockRegressionError",
    "advance_store_timestamp_mark",
    "capture_store_timestamp",
    "encode_store_timestamp_state",
    "ns_operand_to_store_units",
    "parse_store_timestamp_state",
]

_CLOCK_REGRESSION_GRACE_NS = 1_000_000
FUTURE_TIMESTAMP_SLACK_NS = 2_000_000_000
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)


class StoreClockRegressionError(RuntimeError):
    """A writable store clock is behind its process-local timestamp mark."""

    def __init__(self, mark_ns: int, clock_ns: int) -> None:
        super().__init__(
            f"store clock regressed: high-water mark is {mark_ns} ns but clock is {clock_ns} ns; "
            "wait for the clock to catch up or enable allow_clock_regression=True"
        )


def capture_store_timestamp(
    clock: Callable[[], int],
    resolution: int,
    mark: int | None,
    *,
    allow_clock_regression: bool = False,
    clock_regression_grace: bool = True,
) -> int:
    """Capture one guarded store-unit timestamp from a nanosecond clock."""
    clock_ns = clock()
    if allow_clock_regression:
        return clock_ns // resolution
    captured = clock_ns // resolution
    if mark is not None and captured < mark:
        deficit_ns = mark * resolution - clock_ns
        if clock_regression_grace and deficit_ns < _CLOCK_REGRESSION_GRACE_NS:
            time.sleep(deficit_ns / 1_000_000_000 + 0.000001)
            clock_ns = clock()
            captured = clock_ns // resolution
        if captured < mark:
            raise StoreClockRegressionError(mark * resolution, clock_ns)
    return captured


def advance_store_timestamp_mark(mark: int | None, captured: int | None, *, allow_clock_regression: bool) -> int | None:
    """Advance a process-local timestamp high-water mark after a successful save."""
    if captured is None or allow_clock_regression:
        return mark
    return captured if mark is None else max(mark, captured)


def encode_store_timestamp_state(enabled: bool, resolution: int) -> str:
    """Encode the persisted timestamp configuration marker."""
    return f"v1:{resolution}" if enabled else "off"


def parse_store_timestamp_state(value: object) -> tuple[bool, int | None] | None:
    """Parse a persisted timestamp marker, returning ``None`` for unknown values."""
    if value == "off":
        return False, None
    if isinstance(value, str) and value.startswith("v1:"):
        encoded_resolution = value[3:]
        try:
            resolution = int(encoded_resolution)
        except ValueError:
            return None
        if resolution <= 0 or str(resolution) != encoded_resolution:
            return None
        return True, resolution
    return None


def ns_operand_to_store_units(value: object, resolution: int) -> int:
    """Convert a nanosecond, aware datetime, or RFC3339 string to store units."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value // resolution
    if isinstance(value, datetime.datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
        try:
            parsed = datetime.datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError("store_timestamp requires an RFC3339/ISO-8601 timestamp") from error
    else:
        raise ValueError("store_timestamp requires an integer nanosecond, aware datetime, or ISO-8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("store_timestamp requires a timezone-aware datetime")
    utc = parsed.astimezone(datetime.UTC)
    delta = utc - EPOCH
    nanoseconds = delta.days * 86_400 * 1_000_000_000 + delta.seconds * 1_000_000_000 + delta.microseconds * 1_000
    return nanoseconds // resolution
