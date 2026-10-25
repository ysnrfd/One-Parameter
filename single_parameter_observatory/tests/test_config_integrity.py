"""Configuration fails at the public boundary and retains immutable provenance."""

from __future__ import annotations

import unittest
from typing import Any
from collections.abc import Callable

from single_parameter_observatory.config import (
    DatasetConfig,
    ExperimentConfig,
    LandscapeConfig,
    ModelConfig,
    StabilityCriteria,
    ValidationConfig,
)
from single_parameter_observatory.errors import ConfigError


class TestConfigurationIntegrity(unittest.TestCase):
    def test_malformed_configuration_raises_named_error_before_execution(self) -> None:
        cases: tuple[tuple[Callable[..., Any], dict[str, Any]], ...] = (
            (DatasetConfig, {"samples": 1.5}),
            (DatasetConfig, {"batch_size": True}),
            (DatasetConfig, {"seed": 2**64}),
            (DatasetConfig, {"input_max": 1e40}),
            (ModelConfig, {"input_scale": 1e-50}),
            (ModelConfig, {"output_scale": 1e40}),
            (ModelConfig, {"use_tanh": "false"}),
            (LandscapeConfig, {"points": 5.5}),
            (ValidationConfig, {"every": 1.5}),
            (ValidationConfig, {"fd_sweep_decades": []}),
            (ValidationConfig, {"fd_sweep_decades": [float("nan")]}),
            (ValidationConfig, {"fd_sweep_decades": [400.0]}),
            (ValidationConfig, {"fd_sweep_decades": [-400.0]}),
            (StabilityCriteria, {"window": 2.5}),
            (ExperimentConfig, {"steps": True}),
            (ExperimentConfig, {"initial_value": 1e40}),
            (ExperimentConfig, {"lr": "0.1"}),
            (ExperimentConfig, {"lr": 1e40}),
            (ExperimentConfig, {"eps": 1e40}),
            (ExperimentConfig, {"weight_decay": 1e40}),
            (ExperimentConfig, {"lr": 10**1000}),
            (ExperimentConfig, {"seed": -(2**63) - 1}),
            (ExperimentConfig, {"dataset": None}),
            (ExperimentConfig, {"record_full_state_snapshots": False}),
            (ExperimentConfig, {"betas": None}),
            (ExperimentConfig, {"betas": [0.9, "0.999"]}),
            (ExperimentConfig, {"lr_schedule": "cosine", "lr_final_factor": -0.1}),
            (ExperimentConfig, {"lr_schedule": "cosine", "lr_final_factor": float("nan")}),
        )
        for constructor, changes in cases:
            with self.subTest(constructor=constructor.__name__, changes=changes):
                with self.assertRaises(ConfigError):
                    constructor(**changes)

    def test_caller_lists_cannot_change_an_existing_run_configuration(self) -> None:
        betas = [0.9, 0.999]
        decades = [-8.0, -6.0, -4.0]
        config = ExperimentConfig(
            betas=betas,  # type: ignore[arg-type]  # exercise runtime input normalization
            validation=ValidationConfig(fd_sweep_decades=decades),  # type: ignore[arg-type]
        )
        saved = config.to_dict()
        betas[0] = 0.0
        decades.clear()
        self.assertEqual(config.to_dict(), saved)
        self.assertIsInstance(config.betas, tuple)
        self.assertIsInstance(config.validation.fd_sweep_decades, tuple)

    def test_cosine_can_decay_to_zero_and_torch_seed_boundaries_are_accepted(self) -> None:
        config = ExperimentConfig(lr_schedule="cosine", lr_final_factor=0.0, seed=2**64 - 1)
        self.assertEqual(config.lr_final_factor, 0.0)
        self.assertEqual(DatasetConfig(seed=-(2**63)).seed, -(2**63))


if __name__ == "__main__":
    unittest.main()
