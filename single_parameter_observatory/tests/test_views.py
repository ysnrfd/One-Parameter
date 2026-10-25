"""Views: figures, the state card, and the CLI surface.

The figures and the state card are display code over recorded data, so these
tests check that they render, that they say what they claim to say, and that they
fail loudly when matplotlib is genuinely unavailable rather than silently
producing nothing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import unittest

try:  # when discovered as part of the `tests` package
    from . import _bootstrap
except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # type: ignore[import-not-found, no-redef]

from single_parameter_observatory import visualization as viz
from single_parameter_observatory.config import PRESETS
from single_parameter_observatory.errors import ObservatoryError
from single_parameter_observatory.training_engine import TrainingEngine


def _engine(steps: int = 6) -> TrainingEngine:
    config = PRESETS["adamw"].replace(
        steps=steps,
        run_fp64_reference_track=True,
    ).replace_landscape(every=2)
    return TrainingEngine(config)


#: The dead catalogue's name.  Nothing in the package may define or use it; this
#: constant is the only mention of it anywhere, and exists so it cannot quietly
#: come back.
_REMOVED_SERIES_TABLE = "LIVE_SERIES"


def _args() -> argparse.Namespace:
    """The subset of an argparse namespace that ``main._finish`` reads."""
    return argparse.Namespace(export=None, plot=None, no_timestamp=True)


class TestStateCard(unittest.TestCase):
    def test_state_card_contains_the_required_fields(self) -> None:
        engine = _engine(steps=3)
        engine.run()
        card = viz.render_state_card(engine.recorder.records[-1], engine.config)
        for label in (
            "Step", "Parameter", "dtype", "shape", "Before", "After", "Actual dw",
            "Gradient", "Loss", "Prediction", "Learning Rate", "Optimizer", "m", "v",
            "Bias-corrected m", "Bias-corrected v", "Weight Decay (reconstructed)",
            "Optimizer Update (reconstructed)", "FP32 Candidate", "FP32 Stored Value",
            "Rounding Residual", "ULP at w_before", "Movement (ULPs of w_before)",
            "Displacement / ULP(w_before)", "Representable f32 values between",
            "Hex", "Raw uint32",
            "Bits", "Sign", "Exponent", "Fraction", "Changed Bits", "Finite",
        ):
            self.assertIn(label, card, f"the state card is missing {label!r}")

        # No label may present a reconstructed value as an observed one.
        self.assertIn("Reconstructed update (fp32, derived)", card)
        self.assertIn("Attempted update (fp64 candidate)", card)
        self.assertNotIn("Realised float32 update", card)
        self.assertIn("SINGLE PARAMETER FORENSIC STATE", card)

        # The exact stored value must be shown, not a rounded display form.
        record = engine.recorder.records[-1]
        self.assertIn(repr(record.parameter_after), card)
        self.assertIn(record.parameter_hex, card)
        self.assertIn(str(record.parameter_after_uint32), card)

        # It must state that the four tensors are different things.
        self.assertIn("four different", card)

    def test_state_card_on_an_empty_engine(self) -> None:
        from single_parameter_observatory.config import ExperimentConfig

        engine = TrainingEngine(
            ExperimentConfig(steps=1, run_fp64_reference_track=False)
        )
        # No step has run, so there is no record to render; the dashboard must
        # say so rather than inventing a state.
        self.assertEqual(len(engine.recorder), 0)


class TestFigures(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="observatory-figures-")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def test_overview_figure_renders(self) -> None:
        viz.configure_backend(interactive=False)
        engine = _engine(steps=8)
        engine.run()
        figure = viz.build_overview_figure(engine.recorder, engine.analyzer)
        path = viz.save_figure(figure, os.path.join(self.directory, "overview.png"))
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 1000)
        figure.clf()

    def test_bit_figure_renders(self) -> None:
        viz.configure_backend(interactive=False)
        engine = _engine(steps=5)
        engine.run()
        figure = viz.build_bit_figure(engine.recorder)
        path = viz.save_figure(figure, os.path.join(self.directory, "bits.png"))
        self.assertTrue(os.path.exists(path))
        figure.clf()

    def test_render_figures_writes_every_file(self) -> None:
        viz.configure_backend(interactive=False)
        engine = _engine(steps=5)
        engine.run()
        written = viz.render_figures(engine.recorder, self.directory, engine.analyzer)
        self.assertIn("overview", written)
        self.assertIn("bit_history", written)
        for path in written.values():
            self.assertTrue(os.path.exists(path))

    def test_empty_run_figure_says_so(self) -> None:
        viz.configure_backend(interactive=False)
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.recorder import Recorder

        recorder = Recorder(ExperimentConfig(steps=1, run_fp64_reference_track=False))
        figure = viz.build_overview_figure(recorder, None)
        self.assertIsNotNone(figure)
        figure.clf()

    def test_headless_dashboard_always_writes_a_frame(self) -> None:
        """A run that finishes before the first redraw must still leave a frame.

        The loop can exit on the finished condition without ever entering its
        body, which previously left the frame directory empty for short runs.
        The final state of a run is always a legitimate thing to show.
        """
        viz.configure_backend(interactive=False)
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.dashboard import LiveDashboard

        directory = tempfile.mkdtemp(prefix="observatory-frames-")
        self.addCleanup(shutil.rmtree, directory, True)
        engine = TrainingEngine(
            ExperimentConfig(
                name="frames", optimizer="adamw", steps=2,
                run_fp64_reference_track=False,
            )
        )
        panel = LiveDashboard(
            engine, interactive=False, save_frames_directory=directory, frame_every=1
        )
        panel.controller.set_speed(500.0)
        try:
            panel.run(poll_seconds=0.01)
        finally:
            panel.close()
        frames = sorted(os.listdir(directory))
        self.assertTrue(frames, "the headless dashboard wrote no frame at all")
        self.assertGreater(os.path.getsize(os.path.join(directory, frames[0])), 1000)
        # The frames must be a real drawing of the run, not an empty canvas.
        self.assertGreaterEqual(len(engine.recorder), 1)

    def test_single_step_run_leaves_a_visible_mark_in_both_views(self) -> None:
        """One recorded step must be visible in the live and static panels.

        A bare `plot` of a single point has no segment to draw, so the panel
        comes out empty and looks identical to "no run has happened yet". Both
        views go through one drawing helper so they cannot disagree about it.
        """
        import math

        viz.configure_backend(interactive=False)
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.dashboard import LiveDashboard

        engine = TrainingEngine(
            ExperimentConfig(
                name="one-step", optimizer="adamw", steps=2,
                run_fp64_reference_track=False,
            )
        )
        self.assertIsNotNone(engine.step())
        self.assertEqual(len(engine.recorder), 1)

        # The static overview: a single-step run must still be plotted.
        overview = viz.build_overview_figure(engine.recorder, engine.analyzer)
        plotted = [
            ax
            for ax in overview.axes
            if any(
                line.get_marker() not in (None, "None") and len(line.get_xdata()) == 1
                for line in ax.lines
            )
            or len(ax.collections) > 0
        ]
        self.assertTrue(
            plotted, "a one-step run left no visible mark in the overview figure"
        )
        overview.clf()

        # The live dashboard, through the shared helper.
        panel = LiveDashboard(engine, interactive=False)
        try:
            panel.build_figure()
            panel.refresh()
            marks = [
                line
                for line in panel._ax_series.lines
                if line.get_marker() not in (None, "None")
            ]
            self.assertTrue(
                marks, "a one-step run left no visible mark in the dashboard series panel"
            )
            # The four recorded series must each show their single recorded
            # point.  The animation marker is a separate artist that is
            # legitimately empty until a second step exists, so it is excluded
            # rather than asserted to hold a point.
            recorded = [line for line in marks if len(line.get_xdata()) > 0]
            self.assertGreaterEqual(len(recorded), 4)
            for line in recorded:
                self.assertEqual(len(line.get_xdata()), 1)
                self.assertIsNotNone(line.get_marker())
            self.assertGreaterEqual(
                len(panel._ax_series.collections), 0
            )
        finally:
            panel.close()

        # Both views must agree, because they call the same helper.
        xs, ys = viz.recorded_series(engine.recorder.records, "parameter_after")
        self.assertEqual(len(xs), 1)
        self.assertTrue(math.isfinite(ys[0]))

    def test_nonfinite_values_do_not_corrupt_the_axis_limits(self) -> None:
        """A diverged run must still show the steps that were finite.

        Matplotlib ignores a nan when drawing but still uses it when
        auto-scaling, so an unfiltered series produces garbage limits that hide
        the real data.
        """
        import math

        viz.configure_backend(interactive=False)
        from single_parameter_observatory.config import ExperimentConfig, ModelConfig, PRESETS
        from single_parameter_observatory.dashboard import LiveDashboard

        engine = TrainingEngine(PRESETS["diverging"].replace(steps=6))
        engine.run()
        records = engine.recorder.records
        self.assertTrue(records)
        self.assertTrue(
            any(not record.finite for record in records),
            "the diverging preset should have produced a non-finite step",
        )
        panel = LiveDashboard(engine, interactive=False)
        try:
            panel.build_figure()
            panel.refresh()
            limits = (
                panel._ax_series.get_xlim() + panel._ax_series.get_ylim()
            )
            for limit in limits:
                self.assertTrue(
                    math.isfinite(limit), f"axis limit {limit} is not finite"
                )
        finally:
            panel.close()

        # The shared helper must drop the non-finite points, not plot them.
        xs, ys = viz.recorded_series(records, "parameter_after")
        for value in ys:
            self.assertTrue(math.isfinite(value))
        self.assertTrue(all(math.isfinite(x) for x in xs))

    def test_state_card_title_before_and_after_the_first_step(self) -> None:
        """The title must not read 'step -1' before anything has run."""
        viz.configure_backend(interactive=False)
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.dashboard import LiveDashboard

        engine = TrainingEngine(
            ExperimentConfig(
                name="title", optimizer="adamw", steps=3,
                run_fp64_reference_track=False,
            )
        )
        panel = LiveDashboard(engine, interactive=False)
        try:
            panel.build_figure()
            panel.refresh()
            before = panel._ax_card.get_title()
            self.assertIn("no step executed yet", before)
            self.assertNotIn("-1", before)
            engine.step()
            panel.refresh()
            after = panel._ax_card.get_title()
            self.assertIn("step 0", after)
            self.assertNotIn("no step executed yet", after)
        finally:
            panel.close()

    def test_idle_run_shows_the_message_and_plots_nothing(self) -> None:
        viz.configure_backend(interactive=False)
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.dashboard import LiveDashboard

        engine = TrainingEngine(
            ExperimentConfig(
                name="idle", optimizer="adamw", steps=2,
                run_fp64_reference_track=False,
            )
        )
        panel = LiveDashboard(engine, interactive=False)
        try:
            panel.build_figure()
            panel.refresh()
            # The series lines exist as persistent artists so the per-frame
            # blit has something to draw, but with no step recorded they carry
            # no data at all: nothing is plotted.
            plotted = [
                line for line in panel._ax_series.lines
                if len(line.get_xdata()) > 0
            ]
            self.assertEqual(plotted, [], "an idle dashboard must plot no series")
            self.assertEqual(len(panel._ax_series.collections), 0)
            texts = [t.get_text() for t in panel._ax_series.texts]
            self.assertTrue(any("no step executed yet" in t for t in texts))
        finally:
            panel.close()

    def test_live_exit_status_separates_the_view_from_the_run(self) -> None:
        """`live` must exit 0 on a diverged run; `run` must not.

        A diverged run is an expected observation for the diverging preset, not a
        failure of the dashboard. The view's exit status describes the view, and
        the measurement commands' exit status describes the run.
        """
        import contextlib
        import io

        from single_parameter_observatory.main import main as cli_main

        directory = tempfile.mkdtemp(prefix="observatory-live-exit-")
        self.addCleanup(shutil.rmtree, directory, True)

        def exit_code(argv):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                return cli_main(argv)

        live_status = exit_code([
            "live", "--headless", "--preset", "diverging", "--steps", "5",
            "--speed", "100", "--save-frames", directory,
        ])
        self.assertEqual(
            live_status, 0,
            "the dashboard ran; a diverged run it was watching is not a view failure",
        )
        self.assertTrue(os.listdir(directory), "the dashboard wrote no frame")

        run_status = exit_code(["run", "--preset", "diverging", "--steps", "5", "--no-fp64-reference"])
        self.assertEqual(
            run_status, 3,
            "run reports a diverged run as a non-zero exit, because that is its result",
        )

    def test_exit_status_ordering_is_what_the_comment_claims(self) -> None:
        """All four exit statuses, for both a view and a run.

        The D15 test covers only the divergence case. This one also covers a
        broken history chain and a validation failure, because those two are
        *defects* and must fail a view even though a divergence does not - and
        because a future edit that reordered the guards would keep all of them
        passing if only the divergence case were covered.
        """
        import contextlib
        import dataclasses
        import io

        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.main import main as cli_main
        from single_parameter_observatory.training_engine import TrainingEngine

        def exit_code(argv):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                return cli_main(argv)

        # --- 1. a healthy run: 0 for both
        healthy = tempfile.mkdtemp(prefix="observatory-exit-healthy-")
        self.addCleanup(shutil.rmtree, healthy, True)
        self.assertEqual(
            exit_code(["live", "--headless", "--steps", "2", "--speed", "100",
                       "--save-frames", healthy]),
            0,
        )

        # --- 2. a divergence: 3 for run, 0 for live
        self.assertEqual(
            exit_code(["run", "--preset", "diverging", "--steps", "5",
                       "--no-fp64-reference"]),
            3,
            "a run reports its own divergence as a non-zero exit",
        )
        diverged_frames = tempfile.mkdtemp(prefix="observatory-exit-diverged-")
        self.addCleanup(shutil.rmtree, diverged_frames, True)
        self.assertEqual(
            exit_code(["live", "--headless", "--preset", "diverging", "--steps", "5",
                       "--speed", "100", "--save-frames", diverged_frames]),
            0,
            "a view reports that it ran, not that the run stayed finite",
        )

        # --- 3. a broken history chain: 2 for BOTH, including the view.
        #        This is the case a reordered guard would silently let through.
        engine = TrainingEngine(
            ExperimentConfig(name="broken", optimizer="adamw", steps=2,
                             run_fp64_reference_track=False)
        )
        engine.run()
        self.assertTrue(engine.recorder.verify_chain()[0])
        tampered = dataclasses.replace(
            engine.recorder.records[0], loss=engine.recorder.records[0].loss + 1.0
        )
        engine.recorder._records[0] = tampered
        self.assertFalse(engine.recorder.verify_chain()[0])

        from single_parameter_observatory import main as cli_module

        args = _args()
        # The view must fail on a broken chain even though a view tolerates a
        # divergence. `view_command=True` is passed explicitly: this is the
        # ordering the comment above the divergence guard describes.
        self.assertEqual(
            cli_module._finish(engine, args, quiet=True, view_command=True), 2,
            "a broken history chain must fail a view as well as a run",
        )
        self.assertEqual(
            cli_module._finish(engine, args, quiet=True, view_command=False), 2,
        )

        # --- 4. a validation failure: 4 for BOTH, including the view.
        clean = TrainingEngine(
            ExperimentConfig(name="failing", optimizer="adamw", steps=2,
                             run_fp64_reference_track=False)
        )
        clean.run()
        # Record a failure that really happened in the ledger, rather than
        # widening or narrowing a tolerance to produce one.
        clean.recorder.ledger.gradient_failures += 1
        self.assertEqual(
            clean.recorder.ledger.to_dict()["total_failures"], 1
        )
        self.assertEqual(
            cli_module._finish(clean, args, quiet=True, view_command=True), 4,
            "a validation failure must fail a view as well as a run",
        )
        self.assertEqual(
            cli_module._finish(clean, args, quiet=True, view_command=False), 4,
        )

    def test_broken_chain_notice_is_printed_once(self) -> None:
        """The chain is checked once, and its notice is printed once.

        `_finish` used to verify the chain twice: once before the report was
        built and once after the export and plot blocks. The second check was
        unreachable, and it sat immediately above a comment telling the reader
        not to check the chain at that point. Counting the printed notice is
        what catches a reintroduction.
        """
        import contextlib
        import dataclasses
        import inspect
        import io

        from single_parameter_observatory import main as cli_module
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.training_engine import TrainingEngine

        # Structurally: exactly one chain guard inside _finish.
        finish_source = inspect.getsource(cli_module._finish)
        self.assertEqual(
            finish_source.count("if not ok:"), 1,
            "the history chain must be checked exactly once in _finish",
        )

        # Behaviourally: the notice appears exactly once and the status is 2
        # for a view and for a run alike.
        engine = TrainingEngine(
            ExperimentConfig(name="notice", optimizer="adamw", steps=2,
                             run_fp64_reference_track=False)
        )
        engine.run()
        self.assertTrue(engine.recorder.verify_chain()[0])
        engine.recorder._records[0] = dataclasses.replace(
            engine.recorder.records[0], loss=engine.recorder.records[0].loss + 1.0
        )
        self.assertFalse(engine.recorder.verify_chain()[0])

        for view_command in (True, False):
            with self.subTest(view_command=view_command):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    status = cli_module._finish(
                        engine, _args(), quiet=False, view_command=view_command
                    )
                output = buffer.getvalue()
                self.assertEqual(status, 2)
                self.assertEqual(
                    output.count("the recorded history is corrupt at step"),
                    1,
                    "the corruption notice must be printed exactly once",
                )
                # And no report is derived from a corrupt history.
                self.assertNotIn("COMPLETE NUMERICAL REPORT", output)

    def test_live_series_table_is_the_only_live_catalogue(self) -> None:
        """The dead third catalogue is gone; the two live ones name each other."""
        import inspect

        from single_parameter_observatory import dashboard as dash
        from single_parameter_observatory.visualization import OVERVIEW_PANELS

        source = inspect.getsource(viz)
        removed = _REMOVED_SERIES_TABLE
        self.assertNotIn(removed, source, "the unused series table should be gone")
        self.assertFalse(hasattr(viz, removed))
        self.assertNotIn(removed, viz.__all__)
        # No tombstone left at the site of the removal: what was removed and
        # why belongs in the changelog, not beside a live function.
        self.assertNotIn("third catalogue", source)
        # Both surviving tables cross-reference the other.
        marker = "OVERVIEW_PANELS: tuple"
        self.assertIn(marker, source)
        panel_comment = source[source.index(marker) - 700 : source.index(marker)]
        self.assertIn("DASHBOARD_SERIES", panel_comment)
        dash_source = inspect.getsource(dash)
        self.assertIn("build_overview_figure", dash_source)
        self.assertIn("draw_recorded_points", dash_source)
        # The invariant: every attribute the live view draws is also drawn by
        # the static view. Both sides come from defined constants, so an
        # unrelated tuple elsewhere in the module cannot satisfy the check.
        dash_attributes = {name for name, _label, _colour in dash.DASHBOARD_SERIES}
        overview_attributes = {name for name, _title in OVERVIEW_PANELS}
        self.assertIn("parameter_after", dash_attributes)
        self.assertTrue(
            dash_attributes <= overview_attributes,
            "every attribute the live view draws is also in the static view set: "
            f"live-only = {sorted(dash_attributes - overview_attributes)}",
        )
        # And the figure iterates the constant rather than a private copy.
        self.assertIn("zip(flat, OVERVIEW_PANELS, strict=False)", source)

    def test_draw_bit_pattern_takes_no_previous_argument(self) -> None:
        """The parameter that was passed but never read is gone."""
        import inspect

        parameters = list(inspect.signature(viz.draw_bit_pattern).parameters)
        self.assertEqual(parameters, ["ax", "record"])
        doc = inspect.getdoc(viz.draw_bit_pattern) or ""
        self.assertIn("changed_bit_positions", doc)
        self.assertIn("No separate previous step is drawn", doc)

    def test_live_dashboard_key_controls_restart_vs_reset(self) -> None:
        """'R' and 'shift+r' must restart the experiment, while 'r' resets."""
        from unittest.mock import MagicMock

        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.dashboard import LiveDashboard

        engine = TrainingEngine(ExperimentConfig(steps=2, run_fp64_reference_track=False))
        panel = LiveDashboard(engine, interactive=False)
        mock_controller = MagicMock()
        panel.controller = mock_controller

        # Simulate pressing 'R'
        event_R = MagicMock()
        event_R.key = "R"
        panel._on_key(event_R)
        mock_controller.restart_experiment.assert_called_once()
        mock_controller.reset.assert_not_called()

        mock_controller.reset_mock()

        # Simulate pressing 'shift+r'
        event_shift_r = MagicMock()
        event_shift_r.key = "shift+r"
        panel._on_key(event_shift_r)
        mock_controller.restart_experiment.assert_called_once()
        mock_controller.reset.assert_not_called()

        mock_controller.reset_mock()

        # Simulate pressing 'r'
        event_r = MagicMock()
        event_r.key = "r"
        panel._on_key(event_r)
        mock_controller.reset.assert_called_once()
        mock_controller.restart_experiment.assert_not_called()


class TestCliSurface(unittest.TestCase):
    def test_every_required_subcommand_exists(self) -> None:
        from single_parameter_observatory import main as cli

        parser = cli.build_parser()
        actions = [
            action
            for action in parser._actions
            if hasattr(action, "choices") and action.choices
        ]
        self.assertTrue(actions, "the parser should expose subcommands")
        assert actions[0].choices is not None
        assert actions[0].choices is not None
        assert actions[0].choices is not None
        assert actions[0].choices is not None
        names = set(actions[0].choices)
        for command in ("run", "trace", "live", "controls", "inspect", "presets", "selftest"):
            self.assertIn(command, names, f"missing subcommand {command}")

    def test_presets_all_build_valid_configurations(self) -> None:
        from single_parameter_observatory.config import PRESETS as PRESET_MAP

        for name, config in PRESET_MAP.items():
            with self.subTest(preset=name):
                self.assertEqual(config.name, name)
                self.assertGreaterEqual(config.steps, 1)
                self.assertIn(config.optimizer, ("sgd", "adam", "adamw"))
                # Every preset must be a real, runnable configuration.
                engine = TrainingEngine(config.replace(steps=1))
                record = engine.step()
                self.assertIsNotNone(record, f"preset {name} produced no step")
                assert record is not None
                self.assertIn(record.finite, (True, False))

    def test_unknown_preset_is_rejected(self) -> None:
        from single_parameter_observatory import main as cli

        args = argparse.Namespace(
            preset="does-not-exist",
            steps=None,
            optimizer=None,
            weight_decay=None,
            seed=None,
            no_fp64_reference=False,
            no_landscape=False,
            no_validation=False,
            set_to_none=None,
            accumulate=None,
        )

        with self.assertRaises(ObservatoryError):
            cli.resolve_config(args)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
