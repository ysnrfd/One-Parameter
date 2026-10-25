"""Canvas, lifecycle and CLI regressions found in the October repair audit."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import math
import os
import tempfile
import time
import unittest
from typing import Any
from unittest.mock import PropertyMock, patch

import numpy as np

from single_parameter_observatory import main as cli
from single_parameter_observatory import visualization as viz
from single_parameter_observatory.config import ExperimentConfig, PRESETS
from single_parameter_observatory.dashboard import LiveDashboard, run_control_session
from single_parameter_observatory.errors import ConfigError, DependencyMissingError
from single_parameter_observatory.training_engine import TrainingEngine


def make_engine(steps: int = 8) -> TrainingEngine:
    return TrainingEngine(ExperimentConfig(steps=steps, run_fp64_reference_track=False))


class TestCanvasLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = make_engine()
        self.panel = LiveDashboard(self.engine, interactive=False, animate=False)
        self.addCleanup(self.panel.close)
        self.panel.build_figure()

    def pixels(self) -> np.ndarray:
        return np.asarray(self.panel._figure.canvas.buffer_rgba()).copy()

    def test_static_layer_is_rasterized_before_it_is_cached(self) -> None:
        self.panel._frame(redraw_static=True)
        # The causal chain has enough dark lettering to distinguish a real
        # rasterized static layer from the old empty renderer background.
        bbox = self.panel._ax_chain.bbox
        pixels = self.pixels()
        height = pixels.shape[0]
        region = pixels[height-int(bbox.y1):height-int(bbox.y0), int(bbox.x0):int(bbox.x1), :3]
        self.assertGreater(int(np.count_nonzero(np.max(region, axis=2) < 100)), 500)

    def test_recorded_lines_persist_on_motion_only_frames(self) -> None:
        self.engine.run(max_steps=3)
        self.panel._frame(redraw_static=True)
        first = self.pixels()
        self.panel._frame(redraw_static=False, redraw_data=False)
        self.assertTrue(np.array_equal(first, self.pixels()), "a motion-only restore erased recorded lines")

    def test_new_data_rescales_before_the_next_static_clock_tick(self) -> None:
        self.engine.step()
        self.panel._frame(redraw_static=True)
        self.engine.step()
        self.panel._frame(redraw_static=False, redraw_data=True)
        self.assertGreaterEqual(self.panel._ax_series.get_xlim()[1], 1.0)

    def test_no_animation_marker_is_the_latest_recorded_endpoint(self) -> None:
        self.engine.run(max_steps=2)
        self.panel._new_segment_if_needed()
        self.panel._update_animated(0.0)
        last = self.engine.recorder.records[-1]
        self.assertEqual(list(self.panel._animated["marker"].get_xdata()), [last.step])
        self.assertEqual(list(self.panel._animated["marker"].get_ydata()), [last.parameter_after])

    def test_restart_discards_old_segment_and_nonfinite_endpoint_is_hidden(self) -> None:
        self.engine.run(max_steps=2)
        self.panel._new_segment_if_needed()
        self.assertIsNotNone(self.panel._segment)
        self.engine.restart()
        self.panel._new_segment_if_needed()
        self.panel._update_animated(1.0)
        self.assertEqual(list(self.panel._animated["marker"].get_xdata()), [])
        self.engine.run(max_steps=2)
        records = self.engine.recorder.records
        invalid = dataclasses.replace(records[-1], parameter_after=math.inf)
        with patch.object(type(self.engine.recorder), "records", new_callable=PropertyMock) as prop:
            prop.return_value = (records[0], invalid)
            self.panel._new_segment_if_needed()
            self.assertIsNone(self.panel._segment)

    def test_resize_invalidates_both_cached_layers(self) -> None:
        self.engine.step()
        self.panel._frame(redraw_static=True)
        self.assertIsNotNone(self.panel._data_background)
        self.panel._on_resize(None)
        self.assertIsNone(self.panel._background)
        self.assertIsNone(self.panel._data_background)
        self.panel._frame(redraw_data=False)
        self.assertIsNotNone(self.panel._data_background)

    def test_hud_card_updates_independently_from_static_layer(self) -> None:
        self.engine.step()
        self.panel._frame(redraw_static=True)
        self.engine.step()
        self.panel._frame(redraw_data=True, redraw_hud=True)
        self.assertIn("step 1", self.panel._ax_card.get_title())

    def test_cached_data_and_hud_updates_do_not_redraw_the_full_canvas(self) -> None:
        self.engine.run(max_steps=3)
        self.panel._frame(redraw_static=True)
        bounds = (self.panel._ax_series.get_xlim(), self.panel._ax_series.get_ylim())
        with patch.object(self.panel._figure.canvas, "draw", wraps=self.panel._figure.canvas.draw) as draw:
            for _ in range(4):
                self.engine.step()
                self.panel._frame(redraw_data=True, redraw_hud=True)
        draw.assert_not_called()
        self.assertEqual(bounds, (self.panel._ax_series.get_xlim(), self.panel._ax_series.get_ylim()))
        self.assertIn("step 6", self.panel._ax_card.get_title())
        line = self.panel._animated["series:parameter_after"]
        self.assertEqual(list(line.get_xdata())[-1], 6)

    def test_repeated_static_refresh_preserves_one_copy_of_each_artist(self) -> None:
        self.engine.run(max_steps=2)
        self.panel.refresh()
        count = len(self.panel._ax_series.lines)
        self.panel.refresh()
        self.assertEqual(len(self.panel._ax_series.lines), count)

    def test_motion_frame_draws_only_marker_and_trail(self) -> None:
        self.engine.run(max_steps=2)
        self.panel._frame(redraw_static=True)
        with patch.object(self.panel._figure, "draw_artist", wraps=self.panel._figure.draw_artist) as draw:
            self.panel._draw(redraw_data=False)
        self.assertEqual([call.args[0] for call in draw.call_args_list],
                         [self.panel._animated["trail"], self.panel._animated["marker"]])

    def test_interactive_blits_pump_events(self) -> None:
        self.engine.step()
        self.panel.interactive = True
        with patch.object(self.panel._figure.canvas, "flush_events") as flush:
            self.panel._frame(redraw_static=True)
        flush.assert_called()

    def test_dashboard_keys_replace_matplotlib_save_and_home_shortcuts(self) -> None:
        self.panel.close()
        self.panel.interactive = True
        figure, axes = self.panel._plt.subplots(2, 2)
        handler_id = figure.canvas.manager.key_press_handler_id
        with patch.object(self.panel._plt, "subplots", return_value=(figure, axes)), patch.object(figure.canvas, "mpl_disconnect", wraps=figure.canvas.mpl_disconnect) as disconnect:
            self.panel.build_figure()
        disconnect.assert_called_once_with(handler_id)

    def test_close_event_pauses_and_ends_loop(self) -> None:
        with patch.object(self.panel.controller, "pause") as pause:
            self.panel._on_close(None)
        self.assertTrue(self.panel._quit)
        pause.assert_called_once()

    def test_zero_deadline_renders_initial_state_without_a_step(self) -> None:
        self.panel.interactive = True
        with patch.object(self.panel._plt, "show") as show:
            frames = self.panel.run(max_seconds=0.0)
        show.assert_called_once_with(block=False)
        self.assertEqual(frames, 1)
        self.assertEqual(len(self.engine.recorder.records), 0)

    def test_render_failure_stops_background_writer(self) -> None:
        with patch.object(self.panel, "_frame", side_effect=OSError("broken canvas")):
            with self.assertRaisesRegex(OSError, "broken canvas"):
                self.panel.run(max_seconds=0.1)
        count = len(self.engine.recorder.records)
        time.sleep(0.05)
        self.assertEqual(len(self.engine.recorder.records), count)

    def test_close_allows_a_fresh_figure_with_new_artists(self) -> None:
        old_artist = self.panel._animated["marker"]
        self.panel.close()
        self.panel.build_figure()
        self.assertIsNot(self.panel._animated["marker"], old_artist)
        self.assertIs(self.panel._animated["marker"].axes, self.panel._ax_series)

    def test_frame_export_keeps_animated_series_and_invalidates_renderer_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.panel.save_frames_directory = directory
            self.engine.run(max_steps=2)
            self.panel._frame(redraw_static=True)
            self.panel._save_frame()
            image = self.panel._plt.imread(os.path.join(directory, "frame_00001.png"))
            saturation = image[:, :, :3].max(axis=2) - image[:, :, :3].min(axis=2)
            self.assertGreater(int(np.count_nonzero(saturation > 0.3)), 1000)
            self.assertIsNone(self.panel._background)
            self.assertTrue(self.panel._animated["series:parameter_after"].get_animated())


class TestVisualizationHonesty(unittest.TestCase):
    def test_diverged_overview_has_separated_rows_and_readable_log_ticks(self) -> None:
        engine = TrainingEngine(PRESETS["diverging"].replace(run_fp64_reference_track=False))
        engine.run()
        viz.configure_backend(False)
        plt = viz._require_pyplot()
        figure = viz.build_overview_figure(engine.recorder, engine.analyzer)
        self.addCleanup(plt.close, figure)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axes = figure.axes
        for index in range(12):
            upper_xlabel = axes[index].xaxis.label.get_window_extent(renderer)
            lower_title = axes[index+4].title.get_window_extent(renderer)
            self.assertFalse(upper_xlabel.overlaps(lower_title), f"row labels overlap near panel {index}")
        for index in (1, 10):
            boxes = [label.get_window_extent(renderer) for label in axes[index].get_yticklabels() if label.get_visible() and label.get_text()]
            self.assertLessEqual(len(boxes), 9)
            for first, second in zip(boxes, boxes[1:], strict=False):
                self.assertFalse(first.overlaps(second), f"log tick labels overlap in panel {index}")
        card = axes[-1].texts[0].get_window_extent(renderer)
        self.assertLessEqual(card.x1, figure.bbox.x1)
        self.assertGreaterEqual(card.y0, figure.bbox.y0)

    def test_missing_observations_break_lines_without_fabricating_points(self) -> None:
        engine = make_engine(5)
        engine.run()
        records = [dataclasses.replace(record, moment1=value) for record, value in zip(engine.recorder.records, (1.0, None, 3.0, math.inf, 5.0), strict=True)]
        self.assertEqual(viz.recorded_series(records, "moment1"), ([0.0, 2.0, 4.0], [1.0, 3.0, 5.0]))
        xs, ys = viz.recorded_line_series(records, "moment1")
        self.assertEqual(len(xs), 5)
        self.assertTrue(math.isnan(ys[1]))
        self.assertTrue(math.isnan(ys[3]))
        self.assertEqual([value for value in ys if math.isfinite(value)], [1.0, 3.0, 5.0])

    def test_causal_highlight_refresh_does_not_accumulate_labels(self) -> None:
        viz.configure_backend(False)
        plt = viz._require_pyplot()
        figure, ax = plt.subplots()
        self.addCleanup(plt.close, figure)
        viz.draw_causal_chain(ax)
        count = len(ax.texts)
        viz.draw_causal_chain(ax, "UPDATE")
        self.assertEqual(len(ax.texts), count)

    def test_requested_figure_failure_closes_created_figure(self) -> None:
        engine = make_engine(1)
        engine.run()
        viz.configure_backend(False)
        plt = viz._require_pyplot()
        before = set(plt.get_fignums())
        with tempfile.TemporaryDirectory() as directory, patch.object(viz, "save_figure", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                viz.render_figures(engine.recorder, directory)
        self.assertEqual(set(plt.get_fignums()), before)

    def test_gui_backend_is_probed_before_it_is_reported_as_available(self) -> None:
        import matplotlib as mpl
        from unittest.mock import MagicMock

        pyplot = MagicMock()
        pyplot.figure.side_effect = [RuntimeError("display unavailable"), object()]
        with patch.object(mpl, "use"), patch.object(viz, "_require_pyplot", return_value=pyplot):
            self.assertEqual(viz.configure_backend(True), "QtAgg")
        pyplot.close.assert_called_once()


class TestCliRegressions(unittest.TestCase):
    def output(self, argv: list[str]) -> tuple[int, str]:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = cli.main(argv)
        return status, stream.getvalue()

    def test_explicit_zero_configuration_overrides_are_rejected(self) -> None:
        for flag in ("--steps", "--lr", "--accumulate"):
            with self.subTest(flag=flag):
                status, output = self.output(["run", flag, "0"])
                self.assertEqual(status, 1)
                self.assertIn("ConfigError", output)

    def test_cli_seed_controls_the_dataset_generator_as_well_as_experiment_identity(self) -> None:
        args = cli.build_parser().parse_args(["run", "--seed", "0"])
        config = cli.resolve_config(args)
        self.assertEqual(config.seed, 0)
        self.assertEqual(config.dataset.seed, 0)

    def test_invalid_trace_counts_and_ranges_do_not_succeed(self) -> None:
        for flags in (("--every", "0"), ("--count", "-1"), ("--from-step", "-1"), ("--from-step", "3", "--to-step", "2")):
            with self.subTest(flags=flags):
                status, output = self.output(["trace", *flags])
                self.assertEqual(status, 1)
                self.assertIn("ConfigError", output)

    def test_invalid_dashboard_rates_and_limits_raise_configuration_errors(self) -> None:
        engine = make_engine(1)
        for name in ("target_fps", "hud_hz", "static_hz", "data_hz"):
            for value in (-1.0, math.nan, math.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ConfigError):
                    options: dict[str, Any] = {name: value}
                    LiveDashboard(engine, interactive=False, **options)
        panel = LiveDashboard(engine, interactive=False)
        self.addCleanup(panel.close)
        for value in (-1.0, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                panel.run(max_seconds=value)

    def test_inspect_json_is_machine_parseable_without_banners(self) -> None:
        status, output = self.output(["inspect", "--json", "--no-fp64-reference"])
        self.assertEqual(status, 0)
        document = json.loads(output)
        self.assertIn("environment", document)
        self.assertIn("optimizer_semantics_probe", document)

    def test_selftest_does_not_succeed_without_discovered_tests(self) -> None:
        with patch("unittest.TestLoader.discover", return_value=unittest.TestSuite()):
            status, output = self.output(["selftest"])
        self.assertEqual(status, 2)
        self.assertIn("no tests were discovered", output)

    def test_quitting_before_any_step_is_an_explicit_successful_empty_session(self) -> None:
        status, output = self.output(["controls", "--commands", "quit", "--no-fp64-reference"])
        self.assertEqual(status, 0)
        self.assertIn("NO STEPS EXECUTED", output)

    def test_default_exports_are_unique_and_direct_exports_remain_explicit(self) -> None:
        engine = make_engine(1)
        engine.run()
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(export=directory, plot=None, no_timestamp=False)
            with patch.object(cli.exporters, "export_all", return_value={}) as export:
                self.assertEqual(cli._finish(engine, args, quiet=True), 0)
                self.assertEqual(cli._finish(engine, args, quiet=True), 0)
                targets = [call.args[1] for call in export.call_args_list]
                self.assertNotEqual(targets[0], targets[1])
                self.assertTrue(all(os.path.isdir(target) for target in targets))
                args.no_timestamp = True
                cli._finish(engine, args, quiet=True)
                self.assertEqual(export.call_args.args[1], directory)

    def test_plot_dependency_failure_makes_requested_command_fail(self) -> None:
        engine = make_engine(1)
        engine.run()
        args = argparse.Namespace(export=None, plot="unused", no_timestamp=True)
        with patch.object(viz, "render_figures", side_effect=DependencyMissingError("missing matplotlib")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli._finish(engine, args, quiet=True), 1)

    def test_failed_validation_takes_precedence_over_divergence(self) -> None:
        engine = make_engine(1)
        engine.run()
        engine.recorder.ledger.gradient_failures += 1
        engine.aborted = True
        args = argparse.Namespace(export=None, plot=None)
        self.assertEqual(cli._finish(engine, args, quiet=True), 4)

    def test_show_card_uses_the_injected_output_sink(self) -> None:
        engine = make_engine(1)
        output: list[str] = []
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            run_control_session(engine, ["step"], printer=output.append, show_card=True)
        self.assertEqual(stdout.getvalue(), "")
        self.assertTrue(any("SINGLE PARAMETER FORENSIC STATE" in line for line in output))

    def test_interactive_play_returns_to_prompt_so_pause_can_be_read(self) -> None:
        engine = make_engine(1000)
        args = cli.build_parser().parse_args(["controls", "--interactive", "--no-fp64-reference"])
        with patch.object(cli, "TrainingEngine", return_value=engine), patch("builtins.input", side_effect=["play", "pause", "quit"]) as prompt, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.command_controls(args), 0)
        self.assertEqual(prompt.call_count, 3)
        self.assertLess(len(engine.recorder.records), 1000, "play blocked input until the entire run completed")


if __name__ == "__main__":
    unittest.main()
