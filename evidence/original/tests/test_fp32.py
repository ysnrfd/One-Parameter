"""IEEE-754 binary32 forensics: bits, hex, ULP, nextafter, bit transitions."""

from __future__ import annotations

import math
import struct
import unittest

try:  # when discovered as part of the `tests` package
    from . import _bootstrap  # noqa: F401
except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # noqa: F401

import torch

from single_parameter_observatory import fp32_inspector as f32
from single_parameter_observatory.errors import BitDecodeError, NumericalConsistencyError

#: Values chosen to cover normals, negatives, zeros, subnormals and powers of two.
SAMPLE_VALUES: tuple[float, ...] = (
    0.0, -0.0, 1.0, -1.0, 0.5, 0.1, -2.5, 3.14159265, 1e-45, 1.17549435e-38,
    1e38, -1e38, 2.0 ** -126, 2.0 ** 23, float("inf"), float("-inf"),
)


class TestFloat32Bits(unittest.TestCase):
    def test_float32_bits(self) -> None:
        for value in SAMPLE_VALUES:
            with self.subTest(value=value):
                tensor = torch.tensor(value, dtype=torch.float32)
                bits = f32.decode_float32_bits(f32.raw_word_from_tensor(tensor))
                # The decoded value must be the tensor's value, bit for bit.
                self.assertEqual(bits.value, float(tensor.item()))
                # Re-encoding the decoded fields must reproduce the raw word.
                self.assertEqual(
                    f32.encode_float32_bits(
                        bits.sign_bit, bits.exponent_bits, bits.fraction_bits_int
                    ),
                    bits.raw_uint32,
                )
                self.assertEqual(len(bits.binary), 32)
                self.assertEqual(len(bits.fraction_bits_binary), 23)
                if math.isnan(value):
                    continue
                if math.isinf(value):
                    self.assertTrue(bits.is_infinite)
                elif bits.raw_uint32 & 0x7FFFFFFF == 0:
                    self.assertTrue(bits.is_zero)
                elif bits.exponent_bits == 0:
                    self.assertTrue(bits.is_subnormal)
                else:
                    self.assertFalse(bits.is_zero)
                    self.assertFalse(bits.is_subnormal)

    def test_exponent_and_fraction_layout(self) -> None:
        # 1.0 is 0x3F800000: sign 0, biased exponent 127, fraction 0.
        bits = f32.decode_float32_bits(0x3F800000)
        self.assertEqual(bits.sign_bit, 0)
        self.assertEqual(bits.exponent_bits, 127)
        self.assertEqual(bits.unbiased_exponent, 0)
        self.assertEqual(bits.fraction_bits_int, 0)
        self.assertEqual(bits.value, 1.0)
        # -2.0 is 0xC0000000: sign 1, biased exponent 128, fraction 0.
        negative = f32.decode_float32_bits(0xC0000000)
        self.assertEqual(negative.sign_bit, 1)
        self.assertEqual(negative.exponent_bits, 128)
        self.assertEqual(negative.unbiased_exponent, 1)
        self.assertEqual(negative.value, -2.0)
        # 0.5 is 0x3F000000.
        self.assertEqual(f32.decode_float32_bits(0x3F000000).value, 0.5)

    def test_subnormal_and_infinity_classification(self) -> None:
        smallest_subnormal = f32.decode_float32_bits(0x00000001)
        self.assertTrue(smallest_subnormal.is_subnormal)
        self.assertEqual(smallest_subnormal.exponent_bits, 0)
        # The exponent *field* decodes as 0 - 127; the value is nevertheless
        # scaled as if the field were one, so the effective exponent is -126.
        self.assertEqual(smallest_subnormal.unbiased_exponent, -127)
        self.assertEqual(smallest_subnormal.effective_exponent, -126)
        self.assertEqual(smallest_subnormal.value, f32.SMALLEST_SUBNORMAL_F32)
        self.assertEqual(f32.decode_float32_bits(0x00000000).effective_exponent, -126)
        infinity = f32.decode_float32_bits(0x7F800000)
        self.assertTrue(infinity.is_infinite)
        self.assertFalse(infinity.is_finite)
        self.assertFalse(infinity.is_nan)
        nan = f32.decode_float32_bits(0x7FC00000)
        self.assertTrue(nan.is_nan)

    def test_decode_detects_a_bad_word(self) -> None:
        """A decode that cannot round-trip must raise, not return a plausible value."""
        raw = 0x3F800000
        # Corrupt the fraction after decoding and confirm the verifier objects.
        bits = f32.decode_float32_bits(raw)
        tampered = f32.Float32Bits(**{**bits.to_dict(), "fraction_bits_int": 7})
        with self.assertRaises(BitDecodeError):
            f32.verify_decode(tampered)

    def test_hex_encoding(self) -> None:
        self.assertEqual(f32.decode_float32_bits(0x3F800000).hex, "0x3F800000")
        self.assertEqual(f32.decode_float32_bits(0xC0000000).hex, "0xC0000000")
        bits = f32.inspect_float32(torch.tensor(-2.0, dtype=torch.float32))
        self.assertEqual(bits.hex, "0xC0000000")
        self.assertEqual(bits.raw_uint32, 0xC0000000)
        self.assertTrue(bits.hex.startswith("0x"))
        self.assertEqual(len(bits.hex), 10)
        # hex, raw word and binary string must agree with each other.
        self.assertEqual(int(bits.binary, 2), bits.raw_uint32)
        self.assertEqual(f"{bits.raw_uint32:08X}", bits.hex[2:])

    def test_shortest_representation_round_trips(self) -> None:
        for value in (0.1, 1.0 / 3.0, 1e-8, 12345.678):
            with self.subTest(value=value):
                bits = f32.inspect_float32(torch.tensor(value, dtype=torch.float32))
                reparsed = float(bits.decimal_shortest_float32)
                self.assertEqual(
                    f32.raw_word_from_python_float(reparsed), bits.raw_uint32,
                    "the displayed decimal must reproduce the stored float32",
                )

    def test_exact_decimal_is_the_exact_expansion(self) -> None:
        """`decimal_exact` must be the exact rational value, not a short form.

        `float32(0.1)` is 0.100000001490116119384765625; the shortest
        round-tripping decimal is '0.1'. Printing the short form under an
        "exact" label is the conflation this test exists to prevent.
        """
        from decimal import Decimal
        from fractions import Fraction

        bits = f32.inspect_float32(0.1)
        self.assertEqual(bits.decimal_exact, "0.100000001490116119384765625")
        self.assertEqual(bits.decimal_shortest_float32, "0.1")
        self.assertNotEqual(bits.decimal_exact, bits.decimal_shortest_float32)

        cases = [0.1, 1.0 / 3.0, 1e-8, f32.SMALLEST_SUBNORMAL_F32, -2.5, 1.0]
        for value in cases:
            with self.subTest(value=value):
                observed = f32.inspect_float32(value)
                # Independently rebuild the exact value from the stored fields.
                exponent, fraction = observed.exponent_bits, observed.fraction_bits_int
                if exponent == 0:
                    mantissa = fraction
                    power = -149
                else:
                    mantissa = (1 << 23) | fraction
                    power = exponent - 127 - 23
                signed = -mantissa if observed.sign_bit else mantissa
                # The printed decimal must parse back to exactly that rational.
                self.assertEqual(
                    Fraction(Decimal(observed.decimal_exact)),
                    Fraction(signed) * Fraction(2) ** power,
                )
                # And the short form must parse back to the same float32 bits.
                self.assertEqual(
                    f32.raw_word_from_python_float(float(observed.decimal_shortest_float32)),
                    observed.raw_uint32,
                )

    def test_exact_decimal_special_values(self) -> None:
        self.assertEqual(f32.decode_float32_bits(0x00000000).decimal_exact, "0.0")
        self.assertEqual(f32.decode_float32_bits(0x80000000).decimal_exact, "-0.0")
        self.assertEqual(f32.decode_float32_bits(0x7F800000).decimal_exact, "inf")
        self.assertEqual(f32.decode_float32_bits(0xFF800000).decimal_exact, "-inf")
        self.assertEqual(f32.decode_float32_bits(0x7FC00000).decimal_exact, "nan")
        # 1.0 is exactly representable, so the expansion is the whole value.
        self.assertEqual(f32.decode_float32_bits(0x3F800000).decimal_exact, "1.0")
        # The largest finite float32 is an exact integer.
        largest = f32.decode_float32_bits(0x7F7FFFFF)
        self.assertEqual(largest.decimal_exact, "340282346638528859811704183484516925440.0")

    def test_exact_helpers_agree_on_magnitude_and_sign(self) -> None:
        """The two exact helpers are distinct in return type but agree in value.

        `_exact_decimal` returns a str carrying the sign in the string;
        `_exact_fraction` returns a signed Fraction. Both take a raw uint32 and
        apply the sign bit themselves, which the docstrings now state and this
        test makes checkable.
        """
        from decimal import Decimal
        from fractions import Fraction

        words = [
            0x3DCCCCCD,  # float32(0.1), positive normal
            0xBDCCCCCD,  # negative normal
            0x80000001,  # negative subnormal
            0x00000001,  # smallest positive subnormal
            0x80000000,  # -0.0
            0x00000000,  # +0.0
            0xC0200000,  # -2.5
            0x3F800000,  # 1.0
        ]
        for raw in words:
            with self.subTest(raw=hex(raw)):
                exact_text = f32._exact_decimal(raw)
                exact_value = f32._exact_fraction(raw)
                sign_bit_set = bool(raw & 0x80000000)

                if raw in (0x00000000, 0x80000000):
                    # Zero has no sign in a Fraction. The sign bit is still
                    # visible in the decimal string, which is why the two
                    # helpers are not interchangeable: _exact_decimal('-0.0')
                    # keeps the sign, _exact_fraction(0x80000000) is zero.
                    self.assertEqual(exact_value, Fraction(0))
                    self.assertEqual(exact_text, "-0.0" if sign_bit_set else "0.0")
                else:
                    # For a non-zero value the sign of the Fraction follows the
                    # sign bit of the word.
                    self.assertEqual(
                        exact_value < 0, sign_bit_set,
                        f"{hex(raw)}: sign of the fraction disagrees with the sign bit",
                    )
                # The two helpers agree on the magnitude.
                self.assertEqual(
                    abs(Fraction(Decimal(exact_text))), abs(exact_value)
                )
                # And both agree with the decoded value.
                self.assertEqual(exact_value, Fraction(f32.decode_float32_bits(raw).value))

    def test_exact_fraction_matches_the_stored_value(self) -> None:
        from fractions import Fraction

        for raw in (0x3DCCCCCD, 0x3F800000, 0x00000001, 0xC0200000):
            with self.subTest(raw=hex(raw)):
                bits = f32.decode_float32_bits(raw)
                self.assertEqual(
                    f32._exact_fraction(raw),
                    Fraction(bits.value),
                )

    def test_storage_evidence(self) -> None:
        evidence = f32.storage_evidence(torch.tensor(-2.0, dtype=torch.float32))
        self.assertEqual(evidence.element_size_bytes, 4)
        self.assertEqual(evidence.numel, 1)
        self.assertEqual(evidence.total_bytes, 4)
        self.assertEqual(len(evidence.raw_storage_bytes), 4)
        # 0xC0000000 little-endian is 00 00 00 C0.
        self.assertEqual(
            list(evidence.raw_storage_bytes), [0x00, 0x00, 0x00, 0xC0]
        )
        self.assertTrue(evidence.is_floating_point)

    def test_tensor_and_struct_paths_agree(self) -> None:
        for value in (0.1, -2.5, 1e-40, 3.4e38):
            with self.subTest(value=value):
                tensor = torch.tensor(value, dtype=torch.float32)
                self.assertEqual(
                    f32.assert_python_and_tensor_agree(tensor),
                    f32.raw_word_from_tensor(tensor),
                )


class TestULP(unittest.TestCase):
    def test_ulp(self) -> None:
        neighbourhood = f32.ulp_neighbourhood(1.0)
        self.assertEqual(neighbourhood.value, 1.0)
        # The float32 gap above 1.0 is 2**-23, exactly.
        self.assertEqual(neighbourhood.ulp_size, 2.0**-23)
        self.assertGreater(neighbourhood.next_representable, 1.0)
        self.assertLess(neighbourhood.previous_representable, 1.0)
        self.assertEqual(neighbourhood.distance_to_next, neighbourhood.ulp_size)
        # The gap below 1.0 is half the gap above it.
        self.assertEqual(neighbourhood.ulp_size_down, 2.0**-24)
        self.assertEqual(neighbourhood.method, "torch.nextafter")

    def test_ulp_symmetry_at_power_of_two(self) -> None:
        """ULP size must be symmetric under sign at power-of-two boundaries."""
        for mag in (0.5, 1.0, 2.0, 4.0):
            with self.subTest(mag=mag):
                pos = f32.ulp_neighbourhood(mag)
                neg = f32.ulp_neighbourhood(-mag)
                self.assertEqual(pos.ulp_size, neg.ulp_size)
                self.assertEqual(pos.ulp_size_down, neg.ulp_size_down)

    def test_ulp_matches_the_exponent_field(self) -> None:
        for value in (0.5, 1.0, 1.5, 2.0, 3.25, 100.0, -7.5, -1.0, -2.0):
            with self.subTest(value=value):
                bits = f32.inspect_float32(value)
                expected = 2.0 ** (bits.unbiased_exponent - 23)
                # Both sides are exact powers of two, so equality is the right test.
                self.assertEqual(f32.ulp_size(value), expected)

    def test_nextafter(self) -> None:
        value = 1.0
        t = torch.tensor(value, dtype=torch.float32)
        up = torch.nextafter(t, torch.tensor(math.inf, dtype=torch.float32))
        down = torch.nextafter(t, torch.tensor(-math.inf, dtype=torch.float32))
        self.assertGreater(float(up.item()), value)
        self.assertLess(float(down.item()), value)
        self.assertEqual(f32.ulp_neighbourhood(value).next_representable, float(up.item()))
        self.assertEqual(
            f32.ulp_neighbourhood(value).previous_representable, float(down.item())
        )
        # Repeated nextafter is additive in the ULP count.
        walked = t
        for _ in range(5):
            walked = torch.nextafter(walked, torch.tensor(math.inf, dtype=torch.float32))
        self.assertEqual(f32.ulp_distance(value, float(walked.item())), 5)
        # The smallest subnormal is one ULP above zero.
        self.assertEqual(
            f32.ulp_neighbourhood(0.0).next_representable, f32.SMALLEST_SUBNORMAL_F32
        )

    def test_ulp_distance_signs(self) -> None:
        self.assertEqual(f32.ulp_distance(1.0, 1.0), 0)
        self.assertEqual(f32.ulp_distance(-1.0, -1.0), 0)
        positive_up = f32.ulp_neighbourhood(1.0).next_representable
        self.assertEqual(f32.ulp_distance(1.0, positive_up), 1)
        self.assertEqual(f32.ulp_distance(positive_up, 1.0), -1)
        # Moving towards zero from a negative value is an increase.
        negative_towards_zero = float(
            torch.nextafter(
                torch.tensor(-1.0, dtype=torch.float32),
                torch.tensor(math.inf, dtype=torch.float32),
            ).item()
        )
        self.assertEqual(f32.ulp_distance(-1.0, negative_towards_zero), 1)

    def test_ulp_distance_rejects_nan(self) -> None:
        with self.assertRaises(NumericalConsistencyError):
            f32.ulp_distance(1.0, float("nan"))

    def test_ulp_movement_is_zero_when_stored_value_does_not_move(self) -> None:
        self.assertEqual(f32.ulp_movement(1.5, 1.5), 0.0)
        moved = f32.ulp_neighbourhood(1.5).next_representable
        self.assertAlmostEqual(f32.ulp_movement(1.5, moved), 1.0, delta=1e-3)


class TestBitTransitions(unittest.TestCase):
    def test_bit_changes(self) -> None:
        transition = f32.bit_transition(0x3F800000, 0x3F800001)
        self.assertEqual(transition.changed_bit_count, 1)
        self.assertEqual(transition.changed_bit_positions, (31,))
        self.assertEqual(transition.xor_mask, 1)
        self.assertFalse(transition.sign_changed)
        self.assertFalse(transition.exponent_changed)
        self.assertTrue(transition.fraction_changed)
        self.assertEqual(transition.ulp_distance, 1)

        identical = f32.bit_transition(0x3F800000, 0x3F800000)
        self.assertEqual(identical.changed_bit_count, 0)
        self.assertEqual(identical.changed_bit_positions, ())
        self.assertEqual(identical.xor_mask, 0)

        # 1.0 -> -1.0 changes the sign bit only: the exponent field is identical
        # (127) because the magnitude is unchanged.
        sign_flip = f32.bit_transition(0x3F800000, 0xBF800000)
        self.assertTrue(sign_flip.sign_changed)
        self.assertFalse(sign_flip.exponent_changed)
        self.assertFalse(sign_flip.fraction_changed)
        self.assertEqual(sign_flip.changed_bit_count, 1)
        self.assertEqual(sign_flip.changed_bit_positions, (0,))

        # 1.0 -> 2.0 changes the exponent field; both have a zero fraction, so
        # the fraction field is untouched.
        exponent_change = f32.bit_transition(0x3F800000, 0x40000000)
        self.assertTrue(exponent_change.exponent_changed)
        self.assertFalse(exponent_change.fraction_changed)
        self.assertFalse(exponent_change.sign_changed)

        # 1.0 -> 3.0 changes the exponent and the fraction together.
        both = f32.bit_transition(0x3F800000, 0x40400000)
        self.assertTrue(both.exponent_changed)
        self.assertTrue(both.fraction_changed)

    def test_positions_count_from_the_sign_bit(self) -> None:
        transition = f32.bit_transition(0x00000000, 0x80000000)
        self.assertEqual(transition.changed_bit_positions, (0,))
        transition = f32.bit_transition(0x00000000, 0x00000001)
        self.assertEqual(transition.changed_bit_positions, (31,))

    def test_transition_matches_ulp_distance(self) -> None:
        """A one-ULP move is one ULP, but not necessarily one changed bit.

        Incrementing a mantissa can carry: at -2.5 the fraction field runs down
        from 0x200000 to 0x1FFFFF, so a single representable step flips many
        bits. The ULP distance is the reliable measure; the bit count is a
        separate observation and the two are recorded separately for that reason.
        """
        single_bit = 0
        for value in (1.0, -2.5, 0.125, 3.0, 1e-8):
            with self.subTest(value=value):
                start = f32.inspect_float32(value)
                end = f32.inspect_float32(
                    f32.ulp_neighbourhood(value).next_representable
                )
                transition = f32.bit_transition(start.raw_uint32, end.raw_uint32)
                self.assertEqual(transition.ulp_distance, 1)
                self.assertGreaterEqual(transition.changed_bit_count, 1)
                if transition.changed_bit_count == 1:
                    single_bit += 1
        # -2.5 is the carry case, so not every value is a single-bit move.
        self.assertGreater(single_bit, 0)
        carry = f32.bit_transition(0xC0200000, 0xC01FFFFF)
        self.assertEqual(carry.ulp_distance, 1)
        self.assertGreater(carry.changed_bit_count, 1)


class TestEncoding(unittest.TestCase):
    def test_python_float_and_tensor_agree(self) -> None:
        for value in (0.1, -2.5, 1e-40, 3.4e38):
            with self.subTest(value=value):
                tensor = torch.tensor(value, dtype=torch.float32)
                self.assertEqual(
                    f32.raw_word_from_python_float(float(tensor.item())),
                    f32.raw_word_from_tensor(tensor),
                )

    def test_struct_round_trip(self) -> None:
        """A double is not a float32: the round trip must return the rounding."""
        for value in (0.0, -1.0, 1.0, 0.1):
            with self.subTest(value=value):
                raw = f32.raw_word_from_python_float(value)
                expected = float(torch.tensor(value, dtype=torch.float32).item())
                self.assertEqual(f32.python_float_from_raw_word(raw), expected)
                self.assertEqual(struct.unpack("<I", struct.pack("<f", value))[0], raw)
                # Values that are exactly representable do return unchanged.
                if expected == value:
                    self.assertEqual(f32.python_float_from_raw_word(raw), value)

    def test_encode_rejects_out_of_range_fields(self) -> None:
        with self.assertRaises(BitDecodeError):
            f32.encode_float32_bits(2, 0, 0)
        with self.assertRaises(BitDecodeError):
            f32.encode_float32_bits(0, 256, 0)
        with self.assertRaises(BitDecodeError):
            f32.encode_float32_bits(0, 0, 1 << 23)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
