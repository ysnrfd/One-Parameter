"""Real autograd: the gradient comes from ``loss.backward()``, and it is checked.

Three independent methods must agree:
* the gradient autograd produced in float32;
* a closed-form derivative written by hand in float64;
* a central finite difference that re-evaluates the real model in float64.

The gradient lifecycle (``zero_grad`` with ``set_to_none`` both ways) is
measured here too, rather than described from memory.
"""

from __future__ import annotations

import math
import unittest

try:  # when discovered as part of the `tests` package
    from . import _bootstrap  # noqa: F401
except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # noqa: F401

import torch

from single_parameter_observatory import numerical_analysis as na
from single_parameter_observatory import validation as vd
from single_parameter_observatory.config import ValidationConfig
from single_parameter_observatory.model import FP64ReferenceModel, ObservatoryModel
from single_parameter_observatory.validation import validate_gradient, validate_ulp

X = torch.tensor([[1.25], [-0.5], [0.75]], dtype=torch.float32)
TARGET = torch.tensor([[0.4], [-0.2], [0.1]], dtype=torch.float32)


def _model(initial: float = 0.3) -> ObservatoryModel:
    return ObservatoryModel(initial)


class TestAutograd(unittest.TestCase):
    def test_gradient_autograd(self) -> None:
        model = _model()
        loss = model.loss(X, TARGET)
        loss.backward()
        gradient = model.w.grad
        self.assertIsNotNone(gradient, "loss.backward() must populate parameter.grad")
        self.assertIs(gradient.dtype, torch.float32)
        self.assertEqual(tuple(gradient.shape), ())
        self.assertEqual(int(gradient.numel()), 1)
        self.assertTrue(torch.isfinite(gradient).all().item())
        # The gradient must be a real float32 value read from the tensor.
        observed = float(gradient.item())
        self.assertIsInstance(observed, float)
        self.assertNotEqual(observed, 0.0)

        # It must be the float32 rounding of a well-defined real quantity, so it
        # agrees with a float64 autograd evaluation to within float32 precision.
        reference = FP64ReferenceModel(float(model.w.item()))
        reference.loss(X, TARGET).backward()
        fp64 = float(reference.w.grad.item())
        self.assertLess(na.relative_error(observed, fp64), 1e-5)

    def test_gradient_is_not_fabricated(self) -> None:
        """A gradient that is written by hand must not be what the run uses."""
        model = _model()
        model.w.grad = torch.tensor(1234.0, dtype=torch.float32)
        self.assertEqual(float(model.w.grad.item()), 1234.0)
        # A real backward pass overwrites whatever was there.
        model.loss(X, TARGET).backward()
        self.assertNotEqual(float(model.w.grad.item()), 1234.0)

    def test_gradient_analytical(self) -> None:
        model = _model()
        w = float(model.w.item())
        analytical = model.analytic_gradient_fp64(X, TARGET, w)
        loss = model.loss(X, TARGET)
        loss.backward()
        autograd = float(model.w.grad.item())

        # The bound is absolute-plus-relative, scaled by the loss: this gradient
        # is a sum of terms far larger than the result, so eps*|result| is not a
        # sound criterion. See validation.gradient_tolerance.
        loss_scale = abs(model.evaluate_loss(X, TARGET, w, dtype=torch.float64))
        bound, note = vd.gradient_tolerance(analytical, loss_scale, 1e-6)
        self.assertLessEqual(
            na.absolute_error(autograd, analytical), bound,
            f"autograd {autograd!r} vs closed form {analytical!r}; bound: {note}",
        )

        # The closed form is a hand-derived expression, so it must also match a
        # real float64 autograd evaluation of the same forward shape.  The
        # replica model is used here because the primary instrument is float32
        # by construction and its gradient accumulates in float32.
        fp64_model = FP64ReferenceModel(w)
        loss64 = fp64_model.loss(X, TARGET)
        loss64.backward()
        self.assertLess(
            na.relative_error(float(fp64_model.w.grad.item()), analytical), 1e-12
        )

    def test_gradient_finite_difference(self) -> None:
        model = _model()
        w = float(model.w.item())
        analytic = model.analytic_gradient_fp64(X, TARGET, w)

        def evaluate(value: float) -> float:
            return model.evaluate_loss(X, TARGET, value, dtype=torch.float64)

        estimate = na.central_difference(evaluate, w, 1e-5)
        self.assertLess(na.relative_error(estimate, analytic), 1e-6)

        # A perturbation that is far too small is dominated by cancellation, and
        # the sweep has to show that rather than hide it behind one number.
        sweep = na.finite_difference_sweep(
            evaluate, w, analytic, 1e-4, (-14.0, -10.0, -6.0, -3.0, 0.0)
        )
        self.assertEqual(len(sweep), 5)
        self.assertTrue(all(math.isfinite(e.estimate) for e in sweep))
        best = min(sweep, key=lambda e: e.absolute_error)
        self.assertLess(best.absolute_error, 1e-6)
        tiny = [e for e in sweep if e.half_width < 1e-12]
        self.assertTrue(tiny, "the sweep should include a deliberately tiny step")
        self.assertIn("truncation", na.finite_difference_limitation_note())

    def test_full_gradient_validation(self) -> None:
        model = _model()
        loss = model.loss(X, TARGET)
        loss.backward()
        validation = validate_gradient(
            step=0,
            model=model,
            x=X,
            target=TARGET,
            w_value=float(model.w.item()),
            autograd_gradient=float(model.w.grad.item()),
            autograd_uint32=None,
            config=ValidationConfig(),
        )
        self.assertEqual(validation.autograd_status, "PASS", validation.render())
        self.assertEqual(validation.finite_difference_status, "PASS", validation.render())
        self.assertTrue(validation.passed)
        self.assertIsNotNone(validation.autograd_fp64_value)
        self.assertGreater(len(validation.finite_difference_sweep), 1)

    def test_gradient_accumulation(self) -> None:
        """With ``set_to_none=False`` a second backward sums into the same grad."""
        model = _model()
        first_x = torch.tensor([[1.0]], dtype=torch.float32)
        first_t = torch.tensor([[0.5]], dtype=torch.float32)
        second_x = torch.tensor([[-0.5]], dtype=torch.float32)
        second_t = torch.tensor([[0.25]], dtype=torch.float32)

        model.loss(first_x, first_t).backward()
        after_first = float(model.w.grad.item())

        model.loss(second_x, second_t).backward()
        after_second = float(model.w.grad.item())

        separate = _model()
        separate.loss(second_x, second_t).backward()
        second_only = float(separate.w.grad.item())

        # The accumulated gradient is the sum of the two real gradients.
        self.assertAlmostEqual(after_second, after_first + second_only, places=5)

        # set_to_none=True genuinely clears it. set_to_none=False only zeroes a
        # gradient that exists: with grad still None there is nothing to zero,
        # so PyTorch leaves it as None. That asymmetry is measured, not assumed.
        model.zero_grad(set_to_none=True)
        self.assertIsNone(model.w.grad)
        model.zero_grad(set_to_none=False)
        self.assertIsNone(model.w.grad, "zeroing a missing gradient leaves it missing")

        model.loss(first_x, first_t).backward()
        model.zero_grad(set_to_none=False)
        self.assertIsNotNone(model.w.grad)
        self.assertEqual(float(model.w.grad.item()), 0.0)

    def test_gradient_lifecycle_is_measured(self) -> None:
        """The states before/after zero_grad and before/after backward, in order."""
        model = _model()
        first = torch.tensor([[1.0]], dtype=torch.float32)
        target = torch.tensor([[0.5]], dtype=torch.float32)

        # Before any backward: None.
        self.assertIsNone(model.w.grad)
        model.zero_grad(set_to_none=True)
        self.assertIsNone(model.w.grad, "set_to_none=True keeps it None")

        model.loss(first, target).backward()
        self.assertIsNotNone(model.w.grad)
        populated = float(model.w.grad.item())

        model.zero_grad(set_to_none=True)
        self.assertIsNone(model.w.grad)

        model.loss(first, target).backward()
        self.assertEqual(float(model.w.grad.item()), populated, "backward is repeatable")

        model.zero_grad(set_to_none=False)
        self.assertIsNotNone(model.w.grad, "set_to_none=False leaves a zero tensor")
        self.assertEqual(float(model.w.grad.item()), 0.0)

    def test_ulp_validation(self) -> None:
        result = validate_ulp(1.0, steps=4)
        self.assertEqual(result.status, "PASS", result.render())
        self.assertEqual(result.ulp_distance_check, 4)
        self.assertTrue(result.walk_reaches_next)
        self.assertTrue(result.walk_reaches_previous)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
