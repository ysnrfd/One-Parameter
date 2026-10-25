"""Invariants of the tracked parameter: what it is, and that it is really float32."""

from __future__ import annotations

import unittest

try:  # when discovered as part of the `tests` package
    from . import _bootstrap
except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # type: ignore[import-not-found, no-redef]

import torch
from torch import nn

from single_parameter_observatory.config import ExperimentConfig, ModelConfig
from single_parameter_observatory.errors import TrackingInvalidError
from single_parameter_observatory.model import ObservatoryModel
from single_parameter_observatory.tracker import TrackedParameter


def _model(initial: float = -2.0) -> ObservatoryModel:
    return ObservatoryModel(initial, ModelConfig())


class TestTrackedParameterIdentity(unittest.TestCase):
    def test_exactly_one_parameter(self) -> None:
        """The module holds exactly one trainable tensor, and it is a scalar."""
        model = _model()
        parameters = list(model.named_parameters())
        self.assertEqual(len(parameters), 1, f"expected one parameter, found {parameters}")
        self.assertEqual(parameters[0][0], "w")
        self.assertIsInstance(parameters[0][1], nn.Parameter)
        # Buffers must not be parameters, or "exactly one parameter" is a claim
        # about the handle rather than about the module.
        self.assertEqual(
            sorted(name for name, _ in model.named_buffers()),
            ["input_scale", "output_scale"],
        )
        self.assertEqual(len(list(model.parameters())), 1)

    def test_parameter_is_float32(self) -> None:
        parameter = _model().w
        self.assertIs(parameter.dtype, torch.float32)
        self.assertEqual(parameter.element_size(), 4, "float32 must occupy four bytes")

    def test_parameter_shape(self) -> None:
        parameter = _model().w
        self.assertEqual(tuple(parameter.shape), (), "a true scalar has shape ()")
        self.assertEqual(int(parameter.numel()), 1)
        self.assertTrue(parameter.is_leaf)

    def test_parameter_requires_grad(self) -> None:
        parameter = _model().w
        self.assertIs(parameter.requires_grad, True)
        # It must be reachable by autograd: a graph built from it must produce
        # a gradient.
        x = torch.tensor([[1.0]], dtype=torch.float32)
        target = torch.tensor([[0.5]], dtype=torch.float32)
        model = _model()
        loss = model.loss(x, target)
        loss.backward()
        self.assertIsNotNone(model.w.grad)

    def test_initialization(self) -> None:
        for value in (-2.0, 0.0, 1.5, 0.1, -1e-7, 3.25e-3):
            with self.subTest(value=value):
                model = _model(value)
                observed = float(model.w.detach().item())
                # The stored value is the float32 rounding of the requested
                # value; the comparison is against that rounding, not the input.
                expected = float(torch.tensor(value, dtype=torch.float32).item())
                self.assertEqual(observed, expected)
                self.assertEqual(
                    int(model.w.detach().view(torch.int32).item())
                    & 0xFFFFFFFF,
                    int(torch.tensor(value, dtype=torch.float32).view(torch.int32).item())
                    & 0xFFFFFFFF,
                )


class TestTrackingInvariants(unittest.TestCase):
    def test_tracker_reports_identity_card(self) -> None:
        tracker = TrackedParameter(_model(-2.0))
        info = tracker.info()
        self.assertEqual(info.dtype, "float32")
        self.assertEqual(info.numel, 1)
        self.assertEqual(info.shape, ())
        self.assertEqual(info.element_size, 4)
        self.assertTrue(info.requires_grad)
        self.assertTrue(info.invariants_satisfied)
        self.assertEqual(len(info.raw_storage_bytes.split()), 4, "four raw bytes expected")

    def test_verifier_rejects_a_non_scalar_parameter(self) -> None:
        model = _model()
        model.bad = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        with self.assertRaises(TrackingInvalidError) as caught:
            model.assert_single_float32_parameter()
        self.assertIn("exactly one parameter", str(caught.exception))

    def test_verifier_rejects_a_float64_parameter(self) -> None:
        """A tracked object that stops being float32 must be rejected loudly."""
        model = _model()
        # Replace the parameter's storage with a float64 tensor, which is the
        # kind of drift the invariant exists to catch.
        model.w.data = model.w.data.double()
        self.assertIs(model.w.dtype, torch.float64)
        with self.assertRaises(TrackingInvalidError) as caught:
            model.assert_single_float32_parameter()
        self.assertIn("float32", str(caught.exception))

    def test_tracker_detects_a_frozen_parameter(self) -> None:
        model = _model()
        tracker = TrackedParameter(model)
        tracker.verify()
        model.w.requires_grad_(False)
        with self.assertRaises(TrackingInvalidError) as caught:
            tracker.verify(step=3)
        self.assertIn("requires_grad", str(caught.exception))

    def test_model_rejects_non_float32_construction(self) -> None:
        with self.assertRaises(Exception):
            ObservatoryModel(0.0, ModelConfig(), dtype=torch.float64)

    def test_forward_uses_the_parameter(self) -> None:
        """Changing the parameter must change the prediction: it is really in the graph."""
        model = _model(0.0)
        x = torch.tensor([[2.0]], dtype=torch.float32)
        first = float(model(x).item())
        with torch.no_grad():
            model.w.fill_(1.5)
        second = float(model(x).item())
        self.assertNotEqual(first, second)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
