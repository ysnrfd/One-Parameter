"""Numerics of the real transition: deltas, bit changes and float32 rounding."""

from __future__ import annotations

import math
import unittest

try:  # when discovered as part of the `tests` package
    from . import _bootstrap
except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # type: ignore[import-not-found, no-redef]

import torch

from single_parameter_observatory import fp32_inspector as f32
from single_parameter_observatory import numerical_analysis as na
from single_parameter_observatory.config import (
    PRESETS,
    ExperimentConfig,
    ValidationConfig,
)
from single_parameter_observatory.model import ObservatoryModel
from single_parameter_observatory.training_engine import TrainingEngine

X = torch.tensor([[1.25], [-0.5]], dtype=torch.float32)
TARGET = torch.tensor([[0.4], [-0.2]], dtype=torch.float32)


class TestParameterTransition(unittest.TestCase):
    def test_parameter_before_after(self) -> None:
        """Forward and backward must not move the parameter; the step must."""
        config = ExperimentConfig(
            optimizer="adamw", steps=1, initial_value=-0.7,
            run_fp64_reference_track=False,
        )
        model = ObservatoryModel(config.initial_value, config.model)
        parameter = model.assert_single_float32_parameter()
        before_bits = f32.raw_word_from_tensor(parameter)
        loss = model.loss(X, TARGET)
        after_forward_bits = f32.raw_word_from_tensor(parameter)
        loss.backward()
        after_backward_bits = f32.raw_word_from_tensor(parameter)
        self.assertEqual(before_bits, after_forward_bits, "forward must not update w")
        self.assertEqual(before_bits, after_backward_bits, "backward must not update w")
        # The gradient is a separate tensor from the parameter.
        self.assertIsNot(parameter.grad, parameter)
        assert parameter.grad is not None
        self.assertNotEqual(
            f32.raw_word_from_tensor(parameter.grad), before_bits,
            "the gradient tensor must not alias the parameter",
        )

    def test_delta(self) -> None:
        """The recorded delta is stored-after minus stored-before, by definition."""
        config = ExperimentConfig(
            optimizer="adamw", steps=3, initial_value=-0.7,
            run_fp64_reference_track=False,
        )
        engine = TrainingEngine(config)
        records = engine.run()
        self.assertEqual(len(records), 3)
        previous = f32.inspect_float32(torch.tensor(config.initial_value, dtype=torch.float32))
        for record in records:
            after = f32.inspect_float32(torch.tensor(record.parameter_after, dtype=torch.float32))
            self.assertEqual(record.parameter_before, previous.value)
            self.assertEqual(record.actual_delta, after.value - previous.value)
            self.assertEqual(record.absolute_delta, abs(record.actual_delta))
            if record.actual_delta == 0.0:
                self.assertEqual(record.direction, "unchanged")
            elif record.actual_delta > 0.0:
                self.assertEqual(record.direction, "increased")
            else:
                self.assertEqual(record.direction, "decreased")
            # The delta must equal the difference of the two stored words, and
            # the bit distance must agree with it.
            self.assertEqual(
                record.ulp_distance_count,
                f32.ulp_distance(previous.value, after.value),
            )
            previous = after

    def test_bit_changes(self) -> None:
        config = ExperimentConfig(
            optimizer="adamw", steps=3, initial_value=-0.7,
            run_fp64_reference_track=False,
        )
        engine = TrainingEngine(config)
        records = engine.run()
        for record in records:
            transition = f32.bit_transition(
                record.parameter_before_uint32, record.parameter_after_uint32
            )
            self.assertEqual(record.changed_bit_count, transition.changed_bit_count)
            self.assertEqual(
                list(record.changed_bit_positions), list(transition.changed_bit_positions)
            )
            self.assertEqual(record.xor_mask, transition.xor_mask)
            self.assertEqual(record.sign_changed, transition.sign_changed)
            self.assertEqual(record.exponent_changed, transition.exponent_changed)
            self.assertEqual(record.fraction_changed, transition.fraction_changed)
            if record.stored_fp32_unchanged:
                self.assertEqual(record.changed_bit_count, 0)
                self.assertEqual(record.actual_delta, 0.0)

    def test_forward_does_not_update_the_parameter(self) -> None:
        """The snapshot chain A == B == C is an observed fact, per step."""
        config = ExperimentConfig(
            optimizer="adamw", steps=3, run_fp64_reference_track=False
        )
        engine = TrainingEngine(config)
        for record in engine.run():
            self.assertEqual(record.snapshot_a_uint32, record.snapshot_b_uint32)
            self.assertEqual(record.snapshot_b_uint32, record.snapshot_c_uint32)
            self.assertEqual(record.snapshot_c_uint32, record.snapshot_d_uint32)
            # Only the optimizer step changes the stored value.
            self.assertEqual(record.snapshot_d_uint32, record.parameter_before_uint32)
            self.assertEqual(record.snapshot_e_uint32, record.parameter_after_uint32)


class TestRounding(unittest.TestCase):
    def test_rounding_analysis(self) -> None:
        """A float64 candidate is rounded to float32 and the residual measured."""
        for candidate in (0.1, 1.0 / 3.0, -7.125, 2.0**-30, 12345.6789):
            with self.subTest(candidate=candidate):
                stored = f32.float32_tensor(candidate)
                analysis = na.analyse_rounding(candidate, stored)
                self.assertEqual(analysis.stored_float32, float(stored.item()))
                self.assertEqual(
                    analysis.candidate_as_float32, float(stored.item())
                )
                # The candidate rounds to the stored value, so the residual is
                # exactly the float32 rounding error and is under half an ULP.
                self.assertTrue(analysis.candidate_rounds_to_stored)
                self.assertLessEqual(
                    abs(analysis.residual_in_ulp), 0.5 + 1e-12
                )
                self.assertEqual(
                    analysis.absolute_residual,
                    analysis.stored_float32 - analysis.candidate_fp64,
                )
                self.assertIn("rounding", analysis.render())

    def test_exact_half_ulp_tie_is_detected_in_exact_arithmetic(self) -> None:
        """A real tie must be detected, and a one-ULP near miss must not.

        1 + 2**-24 is exactly halfway between 1.0 and 1 + 2**-23, so it is a
        round-to-nearest-even tie. Deciding that in floating point is not
        possible - the midpoint of two float32 values is not generally
        representable as a float64 - so the test is the reason the comparison
        uses exact rationals.
        """
        midpoint = 1.0 + 2.0**-24
        tie = na.analyse_rounding(midpoint, f32.float32_tensor(midpoint))
        self.assertTrue(tie.exact_half_ulp_tie, "an exact midpoint must be a tie")

        # One float64 ULP either side of the midpoint is *not* a tie. The offset
        # has to be 2**-52 here: 2**-60 vanishes at this magnitude, so a smaller
        # offset would not produce a distinct float64 at all.
        for offset in (2.0**-52, -2.0**-52):
            with self.subTest(offset=offset):
                near = midpoint + offset
                self.assertNotEqual(near, midpoint)
                analysis = na.analyse_rounding(near, f32.float32_tensor(near))
                self.assertFalse(
                    analysis.exact_half_ulp_tie, "a near miss must not be a tie"
                )

        # An exactly representable candidate is not a tie either.
        exact = na.analyse_rounding(1.5, f32.float32_tensor(1.5))
        self.assertFalse(exact.exact_half_ulp_tie)
        self.assertEqual(exact.absolute_residual, 0.0)

    def test_ulp_metrics_diverge_at_a_sign_crossing(self) -> None:
        """The two ULP metrics are different quantities and must be labelled as such.

        Flipping the sign of 1.0 means the exact count of representable float32
        values in between passes through the whole subnormal range, which is
        about 2.13 billion. The movement metric divides by the ULP at the
        *starting* value instead, so it stays at the scale of the displacement.
        Summing the first metric over a run in which one step crosses zero
        produces a total that is almost entirely that one step.
        """
        before = f32.float32_tensor(1.0)
        after = f32.float32_tensor(-1.0)
        observed = f32.raw_word_from_tensor(before)
        target = f32.raw_word_from_tensor(after)
        movement = na.analyse_movement(before, after)

        # The sign really did change.
        self.assertNotEqual(observed >> 31, target >> 31)

        # The exact count spans the representable range either side of zero.
        self.assertGreater(abs(movement.ulp_distance), 2_000_000_000)

        # The movement metric is the scale of the displacement, not of the
        # representable range, and is far smaller than the exact count.
        self.assertLess(abs(movement.ulp_distance_same_binade), 1e8)
        self.assertLess(
            abs(movement.ulp_distance_same_binade), abs(movement.ulp_distance) / 10
        )

        # The two are distinct named fields, so neither can silently stand in
        # for the other.
        payload = movement.to_dict()
        self.assertIn("ulp_distance", payload)
        self.assertIn("ulp_distance_same_binade", payload)
        self.assertNotEqual(payload["ulp_distance"], payload["ulp_distance_same_binade"])

    def test_same_binade_metric_matches_the_exact_count_within_a_binade(self) -> None:
        """Away from a boundary the two metrics agree, which is why both exist."""
        value = 1.0
        neighbour = f32.ulp_neighbourhood(value).next_representable
        movement = na.analyse_movement(
            f32.float32_tensor(value), f32.float32_tensor(neighbour)
        )
        self.assertEqual(movement.ulp_distance, 1)
        self.assertEqual(movement.ulp_distance_same_binade, 1)

    def test_rounding_residual_of_a_real_step(self) -> None:
        config = ExperimentConfig(
            optimizer="adamw", steps=4, initial_value=-0.7,
            run_fp64_reference_track=False,
        )
        engine = TrainingEngine(config)
        for record in engine.run():
            self.assertEqual(
                record.fp32_rounding_error_abs,
                record.parameter_after - record.fp64_reconstructed_after,
            )
            self.assertIsInstance(record.fp32_rounding_sign, int)
            if record.fp32_rounding_error_abs == 0.0:
                # Exactly representable: the residual is genuinely zero.
                self.assertEqual(record.fp32_rounding_sign, 0)

    def test_update_below_fp32_resolution_is_reported(self) -> None:
        """An update too small to represent must be labelled, not hidden.

        This is the distinction the project exists to make: the optimizer
        genuinely computes a non-zero update, and the stored float32 value
        genuinely does not move. Neither fact may be reported as the other.
        """
        config = PRESETS["fp32_resolution"].replace(
            run_fp64_reference_track=False, validation=ValidationConfig(enabled=False)
        )
        engine = TrainingEngine(config)
        records = engine.run()
        self.assertTrue(records)
        attempted = [r for r in records if r.update_attempted_but_storage_frozen]
        self.assertTrue(
            attempted,
            "the preset must produce steps with a real but unrepresentable update",
        )
        for record in attempted:
            # The optimizer really did compute a non-zero mathematical
            # displacement ...
            self.assertNotEqual(record.optimizer_attempted_update, 0.0)
            # ... which is smaller than one local ULP ...
            self.assertLess(abs(record.optimizer_attempted_update), record.ulp_size)
            # ... and the float32 arithmetic of the real update landed back on
            # the same value, so the stored parameter did not move.
            self.assertEqual(record.optimizer_total_update, 0.0)
            self.assertEqual(record.actual_delta, 0.0)
            self.assertEqual(record.changed_bit_count, 0)
            self.assertEqual(record.ulp_distance_count, 0)
            self.assertTrue(record.stored_fp32_unchanged)
            self.assertIn(
                "UPDATE_ATTEMPTED_BELOW_FP32_RESOLUTION",
                engine.recorder.events.codes_for_step(record.step),
            )
        for record in records:
            self.assertTrue(record.stored_fp32_unchanged)
            self.assertLess(
                abs(record.ulp_relative_movement), 1.0,
                "the configured regime must stay below one ULP",
            )

    def test_optimizer_state_still_moves_when_the_parameter_cannot(self) -> None:
        """Optimizer memory keeps evolving even when the stored value cannot."""
        config = PRESETS["fp32_resolution"].replace(
            run_fp64_reference_track=False, validation=ValidationConfig(enabled=False)
        )
        engine = TrainingEngine(config)
        records = engine.run()
        first, last = records[0], records[-1]
        self.assertEqual(first.parameter_after, last.parameter_after)
        # exp_avg is a real tensor that autograd-fed data keeps writing to.
        self.assertIsNotNone(first.moment1)
        self.assertIsNotNone(last.moment1)
        self.assertNotEqual(
            (first.moment1, first.moment2), (last.moment1, last.moment2),
            "the optimizer state must keep evolving while w cannot move",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
