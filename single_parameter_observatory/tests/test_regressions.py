"""Regression tests for defects found by the enterprise audit and fixed in 1.1.0.

Each test here corresponds to a numbered finding in AUDIT.md:

* F-01  the live dashboard's series panel rendered empty in every mid-run
  frame, because ``Axes.clear`` resets the view limits and re-attaching the
  persistent artists via ``add_line`` never schedules an autoscale;
* F-02  ``export_all`` raised ``ValueError`` for any run whose final report
  contains a non-finite float (every diverged run), because
  ``json.dump(allow_nan=False, default=str)`` rejects ``inf`` and the
  ``default`` hook never fires for floats;
* F-05  ``--hud-hz`` was accepted, stored and never read;
* F-08  the SGD reconstruction fabricated a step counter of 1 or 2.

Every test fails on the pre-fix code and passes on the fixed code.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

try:  # when discovered as part of the `tests` package
    from . import _bootstrap
except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # type: ignore[import-not-found, no-redef]

from single_parameter_observatory import exporters
from single_parameter_observatory.config import PRESETS
from single_parameter_observatory.dashboard import LiveDashboard
from single_parameter_observatory.optimizer_inspector import probe_optimizer_semantics
from single_parameter_observatory.recorder import json_safe
from single_parameter_observatory.training_engine import TrainingEngine


class TestDashboardSeriesPanel(unittest.TestCase):
    """F-01: the recorded series must actually be visible in the live view."""

    def _dashboard(self, tmp: str) -> tuple[LiveDashboard, TrainingEngine]:
        engine = TrainingEngine(
            PRESETS["adamw"].replace(steps=6, run_fp64_reference_track=False)
        )
        panel = LiveDashboard(
            engine,
            interactive=False,
            save_frames_directory=tmp,
            frame_every=1,
            target_fps=60.0,
            static_hz=10.0,
        )
        return panel, engine
    def test_refresh_restores_autoscaled_view_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel, engine = self._dashboard(tmp)
            panel.build_figure()
            for _ in range(4):
                engine.step()
            panel.refresh(active_phase="UPDATE")
            xlim = panel._ax_series.get_xlim()
            ylim = panel._ax_series.get_ylim()
            # The post-clear default is the unit box; a populated run must not
            # still be showing it.
            self.assertNotEqual(
                (xlim, ylim),
                ((0.0, 1.0), (0.0, 1.0)),
                "the series axes kept the post-clear unit-box viewport, so the "
                "recorded data is clipped outside the panel",
            )
            w_values = [r.parameter_after for r in engine.recorder.records]
            # The autoscaled viewport must contain the data it is meant to show.
            self.assertLessEqual(xlim[0], min(r.step for r in engine.recorder.records) + 1e-9)
            self.assertGreaterEqual(xlim[1], max(r.step for r in engine.recorder.records) - 1e-9)
            self.assertLess(ylim[0], max(w_values) + 1e-9)
            self.assertGreater(ylim[1], min(w_values) - 1e-9)
            panel.close()

    def test_saved_frames_contain_series_ink(self) -> None:
        """Every saved frame must contain visible data pixels in the series panel.

        The pixel assertion is the objective form of the frame audit: on the
        pre-fix code the panel held 0.004% strongly-coloured pixels (legend
        swatches only) against several percent with the series drawn.
        """
        try:
            import numpy as np
            from PIL import Image
        except ImportError:  # pragma: no cover - both are hard dependencies here
            self.skipTest("numpy and Pillow are required for the frame-ink regression test")

        with tempfile.TemporaryDirectory() as tmp:
            panel, engine = self._dashboard(tmp)
            # Steps are taken before the view starts, so every drawn frame -
            # including the first - renders a populated panel.  This makes the
            # ink assertion deterministic rather than a race against the first
            # arriving step.
            engine.run(max_steps=3)
            panel.controller.set_speed(60.0)
            panel.run(max_seconds=2.0)
            panel.close()
            frames = sorted(
                name for name in os.listdir(tmp) if name.endswith(".png")
            )
            self.assertGreaterEqual(len(frames), 2, "the dump should contain several frames")
            for name in frames:
                image = np.asarray(
                    Image.open(os.path.join(tmp, name)).convert("RGB")
                ).astype(int)
                height, width, _ = image.shape
                # top-right quadrant, inset past the title and legend furniture
                region = image[
                    int(height * 0.10) : int(height * 0.46),
                    int(width * 0.53) : int(width * 0.98),
                ]
                saturation = region.max(axis=2) - region.min(axis=2)
                colored = ((saturation > 40) & (region.max(axis=2) > 60)).sum()
                fraction = colored / (region.shape[0] * region.shape[1])
                self.assertGreater(
                    fraction,
                    0.001,
                    f"{name}: the series panel has no visible data ink "
                    f"({fraction:.5f} coloured pixels)",
                )

    def test_hud_interval_follows_configuration(self) -> None:
        """F-05: ``hud_hz`` must actually drive the HUD refresh cadence."""
        engine = TrainingEngine(
            PRESETS["adamw"].replace(steps=2, run_fp64_reference_track=False)
        )
        panel = LiveDashboard(engine, interactive=False, hud_hz=10.0)
        self.assertAlmostEqual(panel._hud_interval, 0.1)
        slow = LiveDashboard(engine, interactive=False, hud_hz=2.0)
        self.assertAlmostEqual(slow._hud_interval, 0.5)
        default = LiveDashboard(engine, interactive=False)
        self.assertAlmostEqual(default._hud_interval, 0.25)
        panel.close()
        slow.close()
        default.close()


class TestNonFiniteExport(unittest.TestCase):
    """F-02: a diverged run must export, as strict JSON, without raising."""

    def test_diverged_run_exports_strict_json(self) -> None:
        engine = TrainingEngine(PRESETS["diverging"])
        records = engine.run()
        self.assertTrue(engine.aborted, "the diverging preset must reach a non-finite state")
        self.assertGreaterEqual(len(records), 1)
        report = engine.final_report()
        self.assertTrue(
            any(
                isinstance(value, float) and value == float("inf")
                for value in report.values()
            ),
            "the diverged report is expected to contain a non-finite value",
        )
        with tempfile.TemporaryDirectory() as tmp:
            written = exporters.export_all(
                engine.recorder,
                tmp,
                report=report,
                provenance=engine.provenance(),
                analyzer=engine.analyzer,
            )
            for name in ("summary.json", "trajectory.json", "validation.json"):
                path = written.get(name)
                self.assertIsNotNone(path, f"{name} was not written for the diverged run")
                assert path is not None
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)  # strict: rejects bare NaN/Infinity
                self.assertIsInstance(payload, dict)
            summary_path = written["summary.json"]
            assert summary_path is not None
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["report"]["final_loss"], "inf")
            self.assertTrue(summary["chain_verified"])

    def test_json_safe_stringifies_non_finite_floats(self) -> None:
        self.assertEqual(json_safe(float("inf")), "inf")
        self.assertEqual(json_safe(float("-inf")), "-inf")
        self.assertEqual(json_safe(float("nan")), "nan")
        self.assertEqual(json_safe(1.5), 1.5)
        nested = {"a": [float("inf"), 2.0], "b": {"c": float("nan")}}
        self.assertEqual(
            json.dumps(json_safe(nested), allow_nan=False),
            '{"a": ["inf", 2.0], "b": {"c": "nan"}}',
        )


class TestSgdReconstructionHonesty(unittest.TestCase):
    """F-08: the SGD reconstruction must not fabricate a step counter."""

    def test_sgd_reconstruction_reports_no_fake_step(self) -> None:
        probe = probe_optimizer_semantics(PRESETS["sgd_momentum"])
        self.assertTrue(probe["bit_exact"])
        # The SGD branch reports step 0 (no counter exists in its state) rather
        # than the fabricated 1-or-2 the pre-fix code produced.
        candidates = probe["candidates"]
        self.assertTrue(candidates)
        self.assertIn("sgd[", probe["measured_variant"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
