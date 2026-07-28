"""Value codecs: exact, round-trippable encodings of rich Python values into scalar columns.

A :class:`ValueCodec` describes how one field value of a given Python type is
stored across one or more scalar columns and recovered exactly. The registry
(:func:`register_value_codec` / :func:`codec_for` / :func:`codec_named`)
mirrors the leaf-codec registry idiom in :mod:`httk.core.vectors`: a frozen
dataclass plus module-level registration functions, with built-in codecs
registered at import time.

Rational values are stored **losslessly**: a query/index column holds the
nearest ``float`` (comparisons on it are documented float-approximate), while a
companion ``*_exact`` text column holds a canonical exact form that is the
round-trip source of truth. The canonical text formats are:

- :data:`FRACTION_EXACT_FORMAT` — a reduced fraction as ``"p/q"``, always with
  the ``/q`` part (``"1/1"`` for one), denominator positive. Used for
  :class:`fractions.Fraction` and :class:`~httk.core.FracScalar`.
- :data:`SURD_EXACT_FORMAT` — a :class:`~httk.core.SurdScalar` as its canonical
  radicand map: ``"r1:p1/q1;r2:p2/q2;..."`` sorted by squarefree radicand
  (``"0"`` for zero), e.g. ``"1:1/2;2:3/4"`` for ``1/2 + (3/4)*sqrt(2)``.
- :data:`FRACVECTOR_EXACT_FORMAT` — a :class:`~httk.core.FracVector` tensor as
  ``"d;n0,n1,n2,..."``: the least positive common denominator ``d`` followed by
  the integer numerators flattened row-major. Used by the schema layer for
  fixed-shape (``Shape``) fields and per-row child-table storage.

httk-core is a hard dependency of httk-data, so the httk.core vector types are
imported unconditionally; nothing in this module needs sqlalchemy.
"""

import datetime
import fractions
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal

from httk.core import FracScalar, FracVector, SurdScalar

__all__ = [
    "FRACTION_EXACT_FORMAT",
    "FRACVECTOR_EXACT_FORMAT",
    "SURD_EXACT_FORMAT",
    "ScalarKind",
    "ValueCodec",
    "codec_for",
    "codec_named",
    "decode_fraction_exact",
    "decode_fracvector_exact",
    "decode_surdscalar_exact",
    "encode_fraction_exact",
    "encode_fracvector_exact",
    "encode_fracvector_floats",
    "encode_surdscalar_exact",
    "known_value_codecs",
    "register_value_codec",
]

type ScalarKind = Literal["int", "float", "str", "bool", "bytes"]
"""The five scalar column kinds a storage backend must provide."""

FRACTION_EXACT_FORMAT: Final = "p/q"
"""The canonical exact text of a rational — reduced ``"p/q"``, always with the ``/q`` part."""

SURD_EXACT_FORMAT: Final = "r1:p1/q1;r2:p2/q2;..."
"""The canonical exact text of a surd scalar — ``radicand:coefficient`` terms sorted by radicand."""

FRACVECTOR_EXACT_FORMAT: Final = "d;n0,n1,n2,..."
"""The canonical exact text of a rational tensor — common denominator, then numerators row-major."""


@dataclass(frozen=True)
class ValueCodec:
    """An exact encoding of one Python value type across one or more scalar columns.

    The ``columns`` tuple gives ``(suffix, kind)`` pairs: the empty suffix names
    the field's own column (by convention the query column), non-empty suffixes
    are appended to the field name by the schema layer (e.g. ``"_exact"``).
    ``encode`` and ``decode`` map a value to and from the column-value tuple, in
    ``columns`` order, and must round-trip exactly.
    """

    name: str
    """Registry name of the codec (e.g. ``"fraction"``)."""

    python_type: type
    """The Python type this codec stores; matched exactly first, then by subclass."""

    columns: tuple[tuple[str, ScalarKind], ...]
    """The ``(column name suffix, scalar kind)`` pairs the codec encodes into."""

    encode: Callable[[Any], tuple[Any, ...]]
    """Encode a value into one scalar per entry of ``columns``, in order."""

    decode: Callable[[tuple[Any, ...]], Any]
    """Recover the exact value from the tuple produced by ``encode``."""

    query_suffix: str = ""
    """Suffix of the column used for querying/indexing (normally the empty suffix)."""


_codecs_by_name: dict[str, ValueCodec] = {}
_codecs_by_type: dict[type, ValueCodec] = {}


def register_value_codec(codec: ValueCodec) -> None:
    """Register ``codec``; raise :class:`ValueError` on a duplicate name or Python type."""
    if codec.name in _codecs_by_name:
        raise ValueError(f"a value codec named {codec.name!r} is already registered")
    if codec.python_type in _codecs_by_type:
        existing = _codecs_by_type[codec.python_type].name
        raise ValueError(
            f"a value codec for type {codec.python_type.__name__!r} is already registered (as {existing!r})"
        )
    _codecs_by_name[codec.name] = codec
    _codecs_by_type[codec.python_type] = codec


def known_value_codecs() -> list[str]:
    """Return the registered value-codec names, in registration order."""
    return list(_codecs_by_name)


def codec_for(python_type: Any) -> ValueCodec | None:
    """Return the codec for ``python_type``, or None if no codec covers it.

    An exact type match wins; otherwise the registered codecs are scanned once
    in registration order and the first whose ``python_type`` is a base class of
    ``python_type`` is returned (deterministic by registration order).
    """
    if not isinstance(python_type, type):
        return None
    exact = _codecs_by_type.get(python_type)
    if exact is not None:
        return exact
    for codec in _codecs_by_name.values():
        if issubclass(python_type, codec.python_type):
            return codec
    return None


def codec_named(name: str) -> ValueCodec:
    """Return the codec registered as ``name``; raise :class:`ValueError` listing the known names."""
    try:
        return _codecs_by_name[name]
    except KeyError:
        raise ValueError(f"unknown value codec {name!r}; known codecs: {known_value_codecs()}") from None


# --------------------------------------------------------------------- exact text helpers


def encode_fraction_exact(value: fractions.Fraction) -> str:
    """The canonical :data:`FRACTION_EXACT_FORMAT` text of ``value`` (e.g. ``"-7/3"``, ``"1/1"``)."""
    return f"{value.numerator}/{value.denominator}"


def decode_fraction_exact(text: str) -> fractions.Fraction:
    """Parse :data:`FRACTION_EXACT_FORMAT` text back into an exact :class:`fractions.Fraction`."""
    numerator_text, _, denominator_text = text.partition("/")
    if not denominator_text:
        raise ValueError(f"invalid exact rational text {text!r}; expected {FRACTION_EXACT_FORMAT!r}")
    return fractions.Fraction(int(numerator_text), int(denominator_text))


def encode_surdscalar_exact(value: SurdScalar) -> str:
    """The canonical :data:`SURD_EXACT_FORMAT` text of ``value`` (``"0"`` for zero).

    The terms are the canonical radicand map of the surd — squarefree radicands
    in increasing order, each with its reduced rational coefficient — so the
    text is unique per value and round-trips exactly.
    """
    radicands = value.radicands
    if not radicands:
        return "0"
    terms = (f"{radicand}:{encode_fraction_exact(value.coefficient(radicand).to_fraction())}" for radicand in radicands)
    return ";".join(terms)


def decode_surdscalar_exact(text: str) -> SurdScalar:
    """Parse :data:`SURD_EXACT_FORMAT` text back into an exact :class:`~httk.core.SurdScalar`."""
    if text == "0":
        return SurdScalar({}, ())
    components: dict[int, FracVector] = {}
    for term in text.split(";"):
        radicand_text, _, coefficient_text = term.partition(":")
        if not coefficient_text:
            raise ValueError(f"invalid exact surd text {text!r}; expected {SURD_EXACT_FORMAT!r}")
        coefficient = decode_fraction_exact(coefficient_text)
        components[int(radicand_text)] = FracVector(coefficient.numerator, coefficient.denominator)
    return SurdScalar(components, ())


def _flat_fractions(value: FracVector) -> list[fractions.Fraction]:
    """The elements of ``value`` as a flat row-major list of :class:`fractions.Fraction`."""
    flat: list[fractions.Fraction] = []

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            for element in node:
                walk(element)
        else:
            flat.append(node)

    walk(value.to_fractions())
    return flat


def encode_fracvector_exact(value: FracVector) -> str:
    """The canonical :data:`FRACVECTOR_EXACT_FORMAT` text of a rational tensor.

    The tensor is flattened row-major and normalized through
    :meth:`~httk.core.FracVector.to_fractions`: ``d`` is the least positive
    common denominator of the (reduced) elements and the numerators are the
    elements scaled by ``d``. The text is therefore canonical — independent of
    the internal denominator the input happened to carry — and lossless at
    arbitrary precision. The shape itself is not encoded; fixed-shape schema
    fields carry it in their declaration.
    """
    flat = _flat_fractions(FracVector.use(value))
    denominator = math.lcm(*(element.denominator for element in flat)) if flat else 1
    numerators = ",".join(str(element.numerator * (denominator // element.denominator)) for element in flat)
    return f"{denominator};{numerators}"


def decode_fracvector_exact(text: str, rows: int, cols: int) -> FracVector:
    """Parse :data:`FRACVECTOR_EXACT_FORMAT` text into a ``rows`` x ``cols`` :class:`~httk.core.FracVector`.

    The numerators are laid out row-major into a tensor of shape
    ``(rows, cols)``; their count must equal ``rows * cols``. Raises
    :class:`ValueError` on malformed text or a count/shape mismatch.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"decode_fracvector_exact requires rows >= 1 and cols >= 1, got {rows}x{cols}")
    denominator_text, _, numerators_text = text.partition(";")
    denominator = int(denominator_text)
    if denominator < 1:
        raise ValueError(f"invalid exact tensor text {text!r}; denominator must be positive")
    numerators = [int(part) for part in numerators_text.split(",")] if numerators_text else []
    if len(numerators) != rows * cols:
        raise ValueError(f"exact tensor text {text!r} holds {len(numerators)} elements; expected {rows}x{cols}")
    noms = tuple(tuple(numerators[row * cols : (row + 1) * cols]) for row in range(rows))
    return FracVector(noms, denominator)


def encode_fracvector_floats(value: FracVector) -> tuple[float, ...]:
    """The elements of a rational tensor as a flat row-major tuple of nearest floats.

    This is the (documented approximate) companion of
    :func:`encode_fracvector_exact`, used to fill the per-element float
    query/index columns of fixed-shape fields.
    """
    return tuple(float(element) for element in _flat_fractions(FracVector.use(value)))


# --------------------------------------------------------------------- built-in codecs


def _encode_fraction(value: Any) -> tuple[Any, ...]:
    return (float(value), encode_fraction_exact(value))


def _decode_fraction(values: tuple[Any, ...]) -> Any:
    return decode_fraction_exact(values[1])


def _encode_fracscalar(value: Any) -> tuple[Any, ...]:
    exact = value.to_fraction()
    return (float(exact), encode_fraction_exact(exact))


def _decode_fracscalar(values: tuple[Any, ...]) -> Any:
    exact = decode_fraction_exact(values[1])
    return FracScalar(exact.numerator, exact.denominator)


def _encode_surdscalar(value: Any) -> tuple[Any, ...]:
    return (value.to_float(), encode_surdscalar_exact(value))


def _decode_surdscalar(values: tuple[Any, ...]) -> Any:
    return decode_surdscalar_exact(values[1])


def _encode_datetime(value: Any) -> tuple[Any, ...]:
    return (value.isoformat(),)


def _decode_datetime(values: tuple[Any, ...]) -> Any:
    return datetime.datetime.fromisoformat(values[0])


_EXACT_COLUMNS: Final[tuple[tuple[str, ScalarKind], ...]] = (("", "float"), ("_exact", "str"))

register_value_codec(
    ValueCodec(
        name="fraction",
        python_type=fractions.Fraction,
        columns=_EXACT_COLUMNS,
        encode=_encode_fraction,
        decode=_decode_fraction,
    )
)
register_value_codec(
    ValueCodec(
        name="fracscalar",
        python_type=FracScalar,
        columns=_EXACT_COLUMNS,
        encode=_encode_fracscalar,
        decode=_decode_fracscalar,
    )
)
register_value_codec(
    ValueCodec(
        name="surdscalar",
        python_type=SurdScalar,
        columns=_EXACT_COLUMNS,
        encode=_encode_surdscalar,
        decode=_decode_surdscalar,
    )
)
register_value_codec(
    ValueCodec(
        name="datetime",
        python_type=datetime.datetime,
        columns=(("", "str"),),
        encode=_encode_datetime,
        decode=_decode_datetime,
    )
)
