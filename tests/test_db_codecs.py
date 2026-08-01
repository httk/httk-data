"""Tests for the value-codec registry and exact text formats (httk.data.db.codecs)."""

import datetime
import random
from fractions import Fraction

import pytest
from httk.core import FracScalar, FracVector, SurdScalar, SurdVector

from httk.data.db import (
    ValueCodec,
    codec_for,
    codec_named,
    decode_fraction_exact,
    decode_fracvector_exact,
    decode_surdscalar_exact,
    encode_fraction_exact,
    encode_fracvector_exact,
    encode_fracvector_floats,
    encode_surdscalar_exact,
    known_value_codecs,
    register_value_codec,
)


def test_builtin_codecs_are_registered():
    names = known_value_codecs()
    for name in ("fraction", "fracscalar", "surdscalar", "datetime"):
        assert name in names


def test_codec_for_matches_exact_type_then_subclass():
    assert codec_for(Fraction) is codec_named("fraction")
    assert codec_for(FracScalar) is codec_named("fracscalar")
    assert codec_for(SurdScalar) is codec_named("surdscalar")
    assert codec_for(datetime.datetime) is codec_named("datetime")
    assert codec_for(FracVector) is None
    assert codec_for(SurdVector) is None
    assert codec_for(int) is None
    assert codec_for(list[int]) is None

    class MyFraction(Fraction):
        pass

    assert codec_for(MyFraction) is codec_named("fraction")


def test_random_fractions_round_trip_exactly():
    rng = random.Random(20260724)
    codec = codec_named("fraction")
    samples = [
        Fraction(0),
        Fraction(1),
        Fraction(-1, 3),
        Fraction(2**71 + 1, 3**41),
        Fraction(-(2**91) - 7, 10**20 + 9),
    ]
    samples += [Fraction(rng.randint(-(2**80), 2**80), rng.randint(1, 2**80)) for _ in range(200)]
    for value in samples:
        encoded = codec.encode(value)
        assert isinstance(encoded[0], float)
        assert isinstance(encoded[1], str)
        decoded = codec.decode(encoded)
        assert decoded == value
        assert isinstance(decoded, Fraction)


def test_fraction_exact_text_always_has_a_denominator():
    assert encode_fraction_exact(Fraction(1)) == "1/1"
    assert encode_fraction_exact(Fraction(-7, 3)) == "-7/3"
    assert decode_fraction_exact("6/4") == Fraction(3, 2)
    with pytest.raises(ValueError):
        decode_fraction_exact("3")


def test_fracscalar_round_trip():
    codec = codec_named("fracscalar")
    value = FracScalar.create("-7/3")
    encoded = codec.encode(value)
    assert encoded[1] == "-7/3"
    decoded = codec.decode(encoded)
    assert isinstance(decoded, FracScalar)
    assert decoded == value
    assert decoded.to_fraction() == Fraction(-7, 3)


def test_float_companion_approximates():
    codec = codec_named("fraction")
    assert codec.encode(Fraction(1, 3))[0] == pytest.approx(1 / 3)
    assert codec.encode(Fraction(-13, 4))[0] == -3.25


def test_surdscalar_round_trip():
    codec = codec_named("surdscalar")
    value = SurdVector.sqrt_of(8) + FracVector(1, 2)  # 1/2 + 2*sqrt(2), a SurdScalar
    encoded = codec.encode(value)
    assert encoded[1] == "1:1/2;2:2/1"
    assert encoded[0] == pytest.approx(0.5 + 2 * 2**0.5)
    decoded = codec.decode(encoded)
    assert isinstance(decoded, SurdScalar)
    assert decoded == value

    zero = SurdScalar({}, ())
    assert encode_surdscalar_exact(zero) == "0"
    assert decode_surdscalar_exact("0") == zero
    negative = SurdVector.sqrt_of(Fraction(1, 2)) - Fraction(5, 7)
    assert decode_surdscalar_exact(encode_surdscalar_exact(negative)) == negative


def test_datetime_round_trips_through_iso_text():
    codec = codec_named("datetime")
    for value in (
        datetime.datetime(2026, 7, 24, 12, 30, 15, 123456),  # noqa: DTZ001 - naive values stay supported
        datetime.datetime(1999, 1, 1, tzinfo=datetime.UTC),
    ):
        encoded = codec.encode(value)
        assert encoded == (value.isoformat(),)
        assert codec.decode(encoded) == value

    offset = datetime.datetime(2026, 1, 1, 2, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
    assert codec.encode(offset) == ("2026-01-01T00:00:00+00:00",)
    assert codec.decode(codec.encode(offset)) == offset


def test_fracvector_exact_text_round_trips_3x3():
    vector = FracVector.create([[1, "1/2", "-3/7"], ["22/7", 0, "1/3"], [-4, "5/6", 2]])
    text = encode_fracvector_exact(vector)
    decoded = decode_fracvector_exact(text, 3, 3)
    assert decoded == vector
    assert decoded.dim == (3, 3)


def test_fracvector_exact_text_round_trips_1xn():
    vector = FracVector.create([["-1/3", "2/5", 7, "1/1000000007"]])
    text = encode_fracvector_exact(vector)
    decoded = decode_fracvector_exact(text, 1, 4)
    assert decoded == vector


def test_fracvector_exact_text_is_canonical():
    # The same value with a non-minimal internal denominator encodes identically.
    plain = FracVector.create([[1, 2], [3, 4]])
    inflated = FracVector(((6, 12), (18, 24)), 6)
    assert encode_fracvector_exact(plain) == encode_fracvector_exact(inflated)
    assert encode_fracvector_exact(plain) == "1;1,2,3,4"


def test_fracvector_exact_text_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="expected 2x2"):
        decode_fracvector_exact("1;1,2,3", 2, 2)
    with pytest.raises(ValueError):
        decode_fracvector_exact("1;1", 0, 1)


def test_fracvector_float_companions():
    vector = FracVector.create([["1/2", "-1/4"]])
    assert encode_fracvector_floats(vector) == (0.5, -0.25)


def test_duplicate_codec_registration_is_rejected():
    with pytest.raises(ValueError, match="named 'fraction'"):
        register_value_codec(
            ValueCodec(
                name="fraction",
                python_type=bytearray,
                columns=(("", "str"),),
                encode=lambda value: (str(value),),
                decode=lambda values: values[0],
            )
        )
    with pytest.raises(ValueError, match="type 'Fraction'"):
        register_value_codec(
            ValueCodec(
                name="fraction2",
                python_type=Fraction,
                columns=(("", "str"),),
                encode=lambda value: (str(value),),
                decode=lambda values: values[0],
            )
        )
