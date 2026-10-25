"""Prevent the release checker itself from reporting unexecuted work as passing."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from single_parameter_observatory import run_checks


class TestReleaseChecks(unittest.TestCase):
    def _check_cli_frame_fixture(self, mode: str) -> tuple[bool, run_checks.Report]:
        from PIL import Image, ImageDraw
        from single_parameter_observatory.config import PRESETS

        engine = MagicMock()
        engine.aborted = True
        engine.abort_reason = "release-check fixture"
        engine.recorder.__len__.return_value = 1
        engine.recorder.records[-1].nonfinite_detected = True
        engine.analyzer.report.return_value = {
            "regime": "fixture", "mean_gradient_magnitude": 0.0, "mean_update_magnitude": 0.0,
        }

        def fake_cli(_report: run_checks.Report, argv: list[str], _required: list[str], _label: str) -> bool:
            if "--save-frames" in argv:
                directory = Path(argv[argv.index("--save-frames") + 1])
                ink_run = "--frame-every" in argv
                if ink_run and mode == "missing":
                    return True
                with Image.new("RGB", (100, 100), (30, 30, 30)) as frame:
                    if ink_run and mode == "healthy":
                        ImageDraw.Draw(frame).rectangle((54, 11, 96, 44), fill=(70, 90, 210))
                    frame.save(directory / "frame_000001.png")
            return True

        def fake_presets(_argv: list[str]) -> int:
            print(" ".join(PRESETS))
            return 0

        report = run_checks.Report()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run_checks, "_run_cli", side_effect=fake_cli), \
                patch("single_parameter_observatory.main.main", side_effect=fake_presets), \
                patch("single_parameter_observatory.training_engine.TrainingEngine", return_value=engine), \
                contextlib.redirect_stdout(io.StringIO()):
            result = run_checks.check_cli_surfaces(report, directory)
        return result, report

    def test_single_healthy_final_frame_passes_independently_of_render_cadence(self) -> None:
        healthy, report = self._check_cli_frame_fixture("healthy")
        self.assertTrue(healthy, report.failures)
        self.assertEqual(report.failures, [])
        self.assertTrue(any("observed frame count for the final-frame ink assertion: 1" in line for line in report.lines))
        self.assertTrue(any("the final frame's series panel contains visible data" in line for line in report.lines))

    def test_missing_or_dataless_final_frame_still_fails_the_pixel_check(self) -> None:
        for mode, expected in (("missing", "no final frame was rendered"),
                               ("dataless", "the final frame's series panel is empty")):
            with self.subTest(mode=mode):
                healthy, report = self._check_cli_frame_fixture(mode)
                self.assertFalse(healthy)
                self.assertTrue(any(expected in failure for failure in report.failures), report.failures)

    def test_environment_reports_available_mps_hardware_instead_of_namespace(self) -> None:
        from single_parameter_observatory.optimizer_inspector import audit_environment

        with patch("torch.backends.mps.is_available", return_value=False):
            self.assertIn("mps_available=False", audit_environment().parallel_info)

    def test_empty_discovery_cannot_pass(self) -> None:
        report = run_checks.Report()
        with patch("unittest.TestLoader.discover", return_value=unittest.TestSuite()):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(run_checks.check_tests(report, False))
        self.assertTrue(any("no tests were discovered" in failure for failure in report.failures))

    def test_skipped_tests_are_excluded_from_pass_count(self) -> None:
        class Probe(unittest.TestCase):
            def test_pass(self) -> None:
                pass

            @unittest.skip("intentional probe")
            def test_skip(self) -> None:
                pass

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Probe)
        report = run_checks.Report()
        with patch("unittest.TestLoader.discover", return_value=suite):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(run_checks.check_tests(report, False))
        summary = next(line for line in report.lines if "TESTS: ran" in line)
        self.assertIn("ran 2, 1 passed", summary)
        self.assertIn("1 skipped", summary)

    def test_missing_required_export_is_a_check_failure(self) -> None:
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.exporters import EXPORT_FILES
        from single_parameter_observatory.training_engine import TrainingEngine

        engine = TrainingEngine(ExperimentConfig(steps=1))
        engine.run()
        report = run_checks.Report()
        with tempfile.TemporaryDirectory() as directory:
            written = {}
            for name in EXPORT_FILES:
                path = Path(directory) / name
                path.write_bytes(b"test artifact")
                written[name] = str(path) if name != "trajectory.npz" else None
            with patch("single_parameter_observatory.exporters.export_all", return_value=written):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertFalse(run_checks.check_export(report, engine, directory))
        self.assertTrue(any("trajectory.npz" in failure for failure in report.failures))

    def test_invalid_measurement_arguments_fail_before_running_checks(self) -> None:
        for arguments in (("--steps", "0"), ("--guard-steps", "0"), ("--fps-seconds", "nan")):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    run_checks.main(list(arguments))
                self.assertEqual(caught.exception.code, 2)

    def test_an_incomplete_export_inventory_cannot_pass(self) -> None:
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.training_engine import TrainingEngine

        engine = TrainingEngine(ExperimentConfig(steps=1))
        engine.run()
        for written in ({}, {"summary.json": "not-a-complete-export"}):
            report = run_checks.Report()
            with self.subTest(written=written):
                with patch("single_parameter_observatory.exporters.export_all", return_value=written):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertFalse(run_checks.check_export(report, engine, "unused"))
                self.assertTrue(any("not reported" in failure for failure in report.failures))


if __name__ == "__main__":
    unittest.main()
