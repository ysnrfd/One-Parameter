"""Real optimizers: state inspection and independent reconstruction.

The reconstruction tests are the strict ones.  Each optimizer is run for a few
real steps, and for each step the expected transition is recomputed from the
observed before-state, the observed gradient and the observed hyperparameters
using PyTorch's own operations.  A reconstruction that does not reproduce the
stored float32 word is a reported failure, not a rounded-away detail.
"""

from __future__ import annotations

import unittest

try:  # when discovered as part of the `tests` package
    from . import _bootstrap  # noqa: F401
except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # noqa: F401

import torch

from single_parameter_observatory import fp32_inspector as f32
from single_parameter_observatory import numerical_analysis as na
from single_parameter_observatory import optimizer_inspector as oi
from single_parameter_observatory.config import ExperimentConfig
from single_parameter_observatory.errors import ConfigError
from single_parameter_observatory.model import ObservatoryModel

X = torch.tensor([[1.25], [-0.5]], dtype=torch.float32)
TARGET = torch.tensor([[0.4], [-0.2]], dtype=torch.float32)


def _run(name: str, steps: int = 4, **overrides) -> tuple[ExperimentConfig, list]:
    """Run a real optimizer for a few steps, reconstructing each transition."""
    settings: dict = {
        "optimizer": name,
        "steps": steps,
        "initial_value": -0.7,
        "lr": 0.1,
        "weight_decay": 0.01,
        "run_fp64_reference_track": False,
    }
    settings.update(overrides)
    config = ExperimentConfig(name=f"test-{name}", **settings)
    model = ObservatoryModel(config.initial_value, config.model)
    parameter = model.assert_single_float32_parameter()
    handle = oi.make_optimizer(config, parameter)
    results: list = []

    for _ in range(steps):
        before = parameter.detach().clone()
        state_before = {
            key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
            for key, value in dict(handle.optimizer.state.get(parameter, {}) or {}).items()
        }
        handle.optimizer.zero_grad(set_to_none=True)
        loss = model.loss(X, TARGET)
        loss.backward()
        gradient = parameter.grad.detach().clone()
        lr = handle.current_lr()
        handle.optimizer.step()
        after = parameter.detach().clone()

        results.append(
            oi.reconstruct_update(
                optimizer_name=name,
                weight_before=before,
                weight_after=after,
                gradient=gradient,
                state_before=state_before,
                lr=lr,
                config=config,
                search=True,
                tolerance_ulp=0,
            )
        )
    return config, results


class TestOptimizers(unittest.TestCase):
    def test_sgd(self) -> None:
        _config, results = _run("sgd", steps=4, weight_decay=0.0, momentum=0.0)
        for result in results:
            self.assertEqual(result.status, "PASS", result.render())
            self.assertTrue(result.bit_exact, result.render())
            self.assertEqual(result.ulp_error, 0)
        # Plain SGD with no momentum keeps no per-parameter state, so there is
        # no first moment to report.
        self.assertIsNone(results[0].moment2_after)

    def test_sgd_momentum_and_nesterov(self) -> None:
        _config, results = _run(
            "sgd", steps=4, momentum=0.9, nesterov=True, weight_decay=0.01
        )
        for result in results:
            self.assertTrue(result.bit_exact, result.render())
            self.assertIsNotNone(result.fp64_reference_after)

    def test_adam(self) -> None:
        _config, results = _run("adam", steps=5)
        for result in results:
            self.assertTrue(result.bit_exact, result.render())
            self.assertIsNotNone(result.moment1_after)
            self.assertIsNotNone(result.moment2_after)
        # The moments must actually be evolving, not frozen at zero.
        self.assertNotEqual(results[0].moment1_after, results[-1].moment1_after)

    def test_adamw(self) -> None:
        _config, results = _run("adamw", steps=5)
        for result in results:
            self.assertTrue(result.bit_exact, result.render())
        # AdamW's decay is decoupled: with a non-zero weight decay the update
        # splits into a gradient-driven and a decay-driven part.
        self.assertNotEqual(results[0].weight_decay_update, 0.0)

    def test_optimizer_reconstruction(self) -> None:
        """Across optimizers and configurations, the reconstruction is exact."""
        cases = [
            ("adamw", {}),
            ("adam", {}),
            ("adamw", {"weight_decay": 0.0}),
            ("adam", {"weight_decay": 0.0, "betas": (0.5, 0.99), "eps": 1e-6}),
            ("sgd", {"weight_decay": 0.0}),
            ("sgd", {"momentum": 0.5, "weight_decay": 0.02}),
            ("sgd", {"momentum": 0.9, "nesterov": True, "weight_decay": 0.0}),
            ("adamw", {"amsgrad": True}),
        ]
        for name, overrides in cases:
            with self.subTest(optimizer=name, **overrides):
                _config, results = _run(name, steps=3, **overrides)
                for result in results:
                    self.assertEqual(
                        result.ulp_error, 0, f"{result.render()}\n{overrides}"
                    )
                    self.assertEqual(result.status, "PASS")

    def test_reconstruction_reports_a_mismatch(self) -> None:
        """A deliberately wrong 'actual' value must be reported, not accepted."""
        config = ExperimentConfig(optimizer="adamw", steps=1, run_fp64_reference_track=False)
        model = ObservatoryModel(config.initial_value, config.model)
        parameter = model.assert_single_float32_parameter()
        handle = oi.make_optimizer(config, parameter)
        before = parameter.detach().clone()
        handle.optimizer.zero_grad(set_to_none=True)
        model.loss(X, TARGET).backward()
        gradient = parameter.grad.detach().clone()
        handle.optimizer.step()
        after = parameter.detach().clone()
        # Pretend the stored value is something the optimizer never produced.
        with torch.no_grad():
            after.add_(1.0)
        result = oi.reconstruct_update(
            optimizer_name="adamw",
            weight_before=before,
            weight_after=after,
            gradient=gradient,
            state_before={},
            lr=handle.current_lr(),
            config=config,
            search=True,
            tolerance_ulp=0,
        )
        self.assertFalse(result.matched)
        self.assertIn("MISMATCH", result.status)
        self.assertGreater(result.ulp_error, 0)

    def test_weight_decay(self) -> None:
        """With the gradient suppressed, AdamW's decay must still move w."""
        config = ExperimentConfig(
            optimizer="adamw", steps=1, lr=0.1, weight_decay=0.5,
            run_fp64_reference_track=False,
        )
        model = ObservatoryModel(4.0, config.model)
        parameter = model.assert_single_float32_parameter()
        handle = oi.make_optimizer(config, parameter)
        before = float(parameter.item())
        handle.optimizer.zero_grad(set_to_none=True)
        # A zero gradient: the only force left is the decoupled decay.
        loss = model.loss(X, TARGET)
        loss.backward()
        with torch.no_grad():
            parameter.grad.zero_()
        handle.optimizer.step()
        after = float(parameter.item())
        # AdamW's decoupled decay shrinks the parameter multiplicatively even
        # when the gradient contributes nothing.
        self.assertLess(after, before)
        expected = before * (1.0 - config.lr * config.weight_decay)
        self.assertAlmostEqual(after, expected, delta=abs(before) * 1e-6)
        _ = loss

    def test_weight_decay_is_not_merged_into_the_gradient_for_adamw(self) -> None:
        """AdamW decays the parameter; the recorded gradient must be unchanged."""
        config = ExperimentConfig(
            optimizer="adamw", steps=1, weight_decay=0.5, run_fp64_reference_track=False
        )
        model = ObservatoryModel(2.0, config.model)
        parameter = model.assert_single_float32_parameter()
        handle = oi.make_optimizer(config, parameter)
        handle.optimizer.zero_grad(set_to_none=True)
        model.loss(X, TARGET).backward()
        gradient_with_decay = float(parameter.grad.item())

        plain = ExperimentConfig(
            optimizer="adamw", steps=1, weight_decay=0.0, run_fp64_reference_track=False
        )
        other = ObservatoryModel(2.0, plain.model)
        other.w.grad = None
        other.loss(X, TARGET).backward()
        self.assertEqual(gradient_with_decay, float(other.w.grad.item()))

    def test_optimizer_state_is_read_not_guessed(self) -> None:
        config = ExperimentConfig(optimizer="adamw", steps=1, run_fp64_reference_track=False)
        model = ObservatoryModel(0.5, config.model)
        parameter = model.assert_single_float32_parameter()
        handle = oi.make_optimizer(config, parameter)
        # Before the first step there is genuinely no state entry yet.
        before = oi.read_optimizer_state(handle.optimizer, parameter)
        self.assertFalse(before.present)
        self.assertIn("no entry", before.note)

        handle.optimizer.zero_grad(set_to_none=True)
        model.loss(X, TARGET).backward()
        handle.optimizer.step()
        after = oi.read_optimizer_state(handle.optimizer, parameter)
        self.assertTrue(after.present)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            self.assertIn(key, after.keys, f"{key} missing from {after.keys}")
        self.assertEqual(after.step_value, 1)
        self.assertIsNotNone(after.value_of("exp_avg"))
        # The decoded state must equal the tensor's own value.
        state = handle.optimizer.state[parameter]
        self.assertEqual(after.value_of("exp_avg"), float(state["exp_avg"].item()))
        self.assertEqual(
            after.value_of("exp_avg_sq"), float(state["exp_avg_sq"].item())
        )
        rendered = after.render()
        self.assertIn("exp_avg", rendered)

    def test_sgd_without_momentum_keeps_no_state(self) -> None:
        """Observed behaviour: no state entry is created at all, not an empty one."""
        config = ExperimentConfig(
            optimizer="sgd", steps=1, momentum=0.0, run_fp64_reference_track=False
        )
        model = ObservatoryModel(0.5, config.model)
        parameter = model.assert_single_float32_parameter()
        handle = oi.make_optimizer(config, parameter)
        handle.optimizer.zero_grad(set_to_none=True)
        model.loss(X, TARGET).backward()
        handle.optimizer.step()
        reading = oi.read_optimizer_state(handle.optimizer, parameter)
        self.assertFalse(reading.present)
        self.assertEqual(reading.keys, ())
        self.assertIn("no per-parameter state", reading.note)
        self.assertIsNone(reading.step_value)

        # Momentum, by contrast, does create state.
        with_momentum = ExperimentConfig(
            optimizer="sgd", steps=1, momentum=0.9, run_fp64_reference_track=False
        )
        other_model = ObservatoryModel(0.5, with_momentum.model)
        other_parameter = other_model.assert_single_float32_parameter()
        other_handle = oi.make_optimizer(with_momentum, other_parameter)
        other_handle.optimizer.zero_grad(set_to_none=True)
        other_model.loss(X, TARGET).backward()
        other_handle.optimizer.step()
        momentum_reading = oi.read_optimizer_state(other_handle.optimizer, other_parameter)
        self.assertTrue(momentum_reading.present)
        self.assertIn("momentum_buffer", momentum_reading.keys)

    def test_source_audit_records_the_installed_implementation(self) -> None:
        config = ExperimentConfig(optimizer="adamw", steps=1, run_fp64_reference_track=False)
        model = ObservatoryModel(0.5, config.model)
        handle = oi.make_optimizer(config, model.w)
        audit = handle.audit
        self.assertEqual(audit.class_name, "AdamW")
        self.assertIn("lr", audit.constructor_parameters)
        self.assertIn("weight_decay", audit.constructor_parameters)
        self.assertTrue(audit.source_findings["source_text_available"])
        self.assertEqual(audit.torch_version, str(torch.__version__))
        rendered = audit.render()
        self.assertIn("AdamW", rendered)

    def test_source_audit_sgd(self) -> None:
        config = ExperimentConfig(optimizer="sgd", steps=1, run_fp64_reference_track=False)
        model = ObservatoryModel(0.5, config.model)
        handle = oi.make_optimizer(config, model.w)
        audit = handle.audit
        self.assertEqual(audit.class_name, "SGD")
        self.assertIn("lr", audit.constructor_parameters)
        self.assertIn("momentum", audit.constructor_parameters)

    def test_semantics_probe_measures_an_exact_variant(self) -> None:
        """The probe must find a variant that reproduces the real update exactly."""
        config = ExperimentConfig(optimizer="adamw", steps=1, run_fp64_reference_track=False)
        probe = oi.probe_optimizer_semantics(config)
        self.assertTrue(probe["bit_exact"], probe)
        self.assertIn("adamw", probe["measured_variant"])
        self.assertTrue(probe["candidates"])
        self.assertEqual(min(c["ulp_error"] for c in probe["candidates"]), 0)

    def test_unknown_optimizer_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            ExperimentConfig(optimizer="rmsprop")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
