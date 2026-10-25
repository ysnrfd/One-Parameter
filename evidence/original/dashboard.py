"""The live dashboard.

The dashboard is a *view*, never a source of numbers.  Every redraw reads the
engine's recorded history, so pausing it changes nothing about the experiment
and un-pausing it never fabricates a frame: if no step has been executed, the
figure shows the initial state and says so.

Fluid motion
------------
The view is built in two layers, because the two halves of a frame cost
wildly different amounts:
* the **static layer**   axes, tick labels, the causal chain, the bit panel and
  the state card.  It changes only when a new step is recorded, and on this
  machine it costs on the order of a second to rasterise, almost entirely in
  text layout.
* the **animated layer**   the recorded series and a marker that travels along
  the newest recorded segment.  It costs a few milliseconds.
Each frame restores the cached static layer and draws only the animated layer,
so the window updates at the display's rate instead of once per training step.
``--fps`` sets that rate; 60 and 120 are both comfortable.

Honesty about the marker
------------------------
A marker sliding along a segment is a display animation, not a measurement.  The
segment's two endpoints are recorded float32 values; the position between them
is an interpolation, and the panel says so on the face of the figure.  The
recorded line itself is only ever extended with recorded points.

Keyboard controls (interactive backends only):
    space   play / pause
    s       single step
    n       run 10 steps
    r       reset the live state to the initial condition
    R       restart the experiment, discarding the history
    + / -   increase / decrease the step rate
    q       quit
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Sequence

import torch

from . import numerical_analysis as na
from . import visualization as viz
from .config import ExperimentConfig
from .errors import DependencyMissingError
from .recorder import ParameterStepRecord
from .training_engine import ExperimentController, RunState, TrainingEngine

#: The live series drawn in the dashboard's trajectory panel.  The static view's
#: series list is ``visualization.OVERVIEW_PANELS``, drawn by
#: ``visualization.build_overview_figure``; both name the same recorded
#: attributes of the same records and deliberately draw different subsets of
#: them, so the two are allowed to differ in content.  What they must not do is
#: draw a series differently: both go through
#: ``visualization.draw_recorded_points``, which filters non-finite values and
#: marks every recorded point.
DASHBOARD_SERIES: tuple[tuple[str, str, str], ...] = (
    ("parameter_after", "w", "tab:blue"),
    ("actual_delta", "delta w", "tab:orange"),
    ("gradient", "gradient", "tab:green"),
    ("loss", "loss", "tab:red"),
)

#: Shown when the dashboard has nothing to draw yet.
IDLE_MESSAGE = "no step executed yet - press play or step to run the real experiment"

#: Text placed next to the moving marker, so the figure never presents an
#: interpolated position as a recorded one.
INTERPOLATION_NOTICE = (
    "marker slides between two recorded steps - the position between them is "
    "interpolated for display; only the integer-step points are recorded values"
)

#: Default render rate.  The static layer is cached, so this is cheap.
DEFAULT_TARGET_FPS = 60.0

#: How often the state card is re-rasterised, in hertz.  It is by far the most
#: expensive text in the figure, and it is informational rather than animated.
DEFAULT_HUD_HZ = 4.0

#: How often the whole static layer - card, bit panel, causal chain, axis
#: labels - is re-rasterised.  Measured on this machine, one such redraw costs
#: on the order of 200 ms, so running it per training step pins the view near
#: 5 FPS regardless of how fast the loop itself is.
DEFAULT_STATIC_HZ = 1.0

#: How often the recorded series lines are re-rasterised, in hertz, when steps
#: are arriving faster than that.  The marker carries the motion in between.
DEFAULT_DATA_HZ = 20.0


class LiveDashboard:
    """A live view over a real, still-running experiment."""

    def __init__(
        self,
        engine: TrainingEngine,
        *,
        controller: ExperimentController | None = None,
        interactive: bool = True,
        save_frames_directory: str | None = None,
        frame_every: int = 1,
        target_fps: float = DEFAULT_TARGET_FPS,
        hud_hz: float = DEFAULT_HUD_HZ,
        static_hz: float = DEFAULT_STATIC_HZ,
        data_hz: float = DEFAULT_DATA_HZ,
        animate: bool = True,
    ) -> None:
        self.engine = engine
        self.controller = controller or ExperimentController(engine)
        self.interactive = interactive
        self.save_frames_directory = save_frames_directory
        self.frame_every = max(1, int(frame_every))
        self.target_fps = float(target_fps) if target_fps else 0.0
        self.hud_hz = float(hud_hz) if hud_hz else 0.0
        self.animate = animate
        if self.save_frames_directory:
            os.makedirs(self.save_frames_directory, exist_ok=True)
        self._plt = viz._require_pyplot()  # noqa: SLF001 - internal by design
        self.backend = viz.configure_backend(interactive)
        self._figure: Any = None
        self._last_drawn_step = -1
        self._drawn_phase: str | None = None
        self._frame_index = 0
        self._quit = False
        # animated-layer artists, created with the first record
        self._animated: dict[str, Any] = {}
        self._background: Any = None
        self._blit_available = True
        self._hud_due = 0.0

        # How often the expensive static layer is re-rasterised.  Text dominates
        # the cost of a frame on this machine, so the card and the bit panel are
        # refreshed on this clock while the data updates every frame.
        self._static_hz = float(static_hz) if static_hz else 0.0
        self._static_interval = 1.0 / self._static_hz if self._static_hz else 0.25
        self._static_due = 0.0

        # The recorded series are redrawn when a step arrives, throttled: they
        # are cheap next to text but not free, and the marker already carries the
        # motion between them.
        self._data_hz = float(data_hz) if data_hz else 0.0
        self._data_interval = 1.0 / self._data_hz if self._data_hz else 0.05
        self._data_due = 0.0
        self._data_step = -1

        self._fps_samples: list[float] = []
        self._measured_fps = 0.0
        self._segment_start: float = 0.0
        self._segment: tuple[ParameterStepRecord, ParameterStepRecord] | None = None
        self._title = "Single float32 parameter observatory - live view of a real run"

    # -- figure -------------------------------------------------------------
    def build_figure(self) -> Any:
        plt = self._plt
        self._figure, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=110)
        self._ax_chain = axes[0][0]
        self._ax_series = axes[0][1]
        self._ax_bits = axes[1][0]
        self._ax_card = axes[1][1]
        self._figure.suptitle(self._title, fontsize=12)
        viz.draw_causal_chain(self._ax_chain)
        self._ax_bits.axis("off")
        self._ax_card.axis("off")
        self._ax_series.set_xlabel("step", fontsize=8)
        self._ax_series.grid(True, alpha=0.25, linewidth=0.5)
        self._figure.tight_layout(rect=(0, 0, 1, 0.95))
        if self.interactive:
            self._figure.canvas.mpl_connect("key_press_event", self._on_key)
        self._ensure_animated_artists()
        return self._figure

    def _ensure_animated_artists(self) -> None:
        """Create the artists that change every frame, once.

        The series lines live here rather than on the static layer.  A step can
        arrive every few milliseconds, and re-rasterising the whole figure each
        time is what made this view update about once a second.  Keeping the
        lines as persistent artists lets every frame redraw only the data.
        """
        if self._animated or self._figure is None:
            return
        ax = self._ax_series
        for attribute, label, colour in DASHBOARD_SERIES:
            # marker="." so a single recorded point is still visible; the shared
            # helper below decides which points are real.
            (line,) = ax.plot(
                [], [], linewidth=1.1, label=label, color=colour,
                marker=".", markersize=6,
            )
            self._animated[f"series:{attribute}"] = line
        self._animated["marker"] = ax.plot(
            [], [], marker="o", linestyle="None", markersize=9,
            color="tab:purple", zorder=5, label="animated marker",
        )[0]
        self._animated["trail"] = ax.plot(
            [], [], linewidth=2.0, color="tab:purple", alpha=0.7, zorder=4
        )[0]
        self._animated["notice"] = ax.text(
            0.02, 0.02, INTERPOLATION_NOTICE, transform=ax.transAxes,
            fontsize=6, color="0.35", va="bottom",
        )
        self._animated["fps"] = ax.text(
            0.98, 0.98, "", transform=ax.transAxes, fontsize=7,
            ha="right", va="top", color="0.3",
        )

    def _reattach_animated(self) -> None:
        """Re-add the per-frame artists after the series axes have been cleared."""
        ax = self._ax_series
        for key in ("trail", "marker"):
            ax.add_line(self._animated[key])
        for key, artist in self._animated.items():
            if key.startswith("series:"):
                ax.add_line(artist)
        ax.add_artist(self._animated["notice"])
        ax.add_artist(self._animated["fps"])

    # -- static layer -------------------------------------------------------
    def refresh(self, active_phase: str | None = None) -> None:
        """Full re-render.  Expensive: the run loop throttles this."""
        records = self.engine.recorder.records
        last = records[-1] if records else None
        # The causal chain holds 12 labels and 11 arrows and changes only when
        # the highlighted phase changes, so redrawing it on every static pass
        # spent a large part of the budget on identical pixels.
        if active_phase != self._drawn_phase:
            viz.draw_causal_chain(self._ax_chain, active_phase)
            self._drawn_phase = active_phase
        self._setup_series_axes(records)
        self._update_series(records)
        self._draw_bits(last)
        self._draw_card(last)
        self._ensure_animated_artists()
        self._reattach_animated()
        self._legend()
        self._update_animated(1.0)
        self._snapshot()

    def _legend(self) -> None:
        """Build the series legend from the persistent artists.

        The legend must be made after the artists are re-added: `Axes.clear`
        removes them, and a legend built before that finds nothing to label.
        """
        if self.engine.recorder.records:
            self._ax_series.legend(fontsize=7)
        else:
            self._ax_series.text(
                0.5, 0.5, IDLE_MESSAGE, ha="center", va="center", wrap=True
            )

    def _setup_series_axes(self, records: Sequence[ParameterStepRecord]) -> None:
        """Static furniture of the series panel: frame, ticks, legend, title."""
        self._ax_series.clear()
        self._ax_series.set_xlabel("step", fontsize=8)
        # Fewer ticks: on this machine each tick label costs milliseconds, and
        # the frame rate matters more than the last digit of an axis.
        self._ax_series.xaxis.set_major_locator(
            self._plt.matplotlib.ticker.MaxNLocator(nbins=4)
        )
        self._ax_series.yaxis.set_major_locator(
            self._plt.matplotlib.ticker.MaxNLocator(nbins=4)
        )
        self._ax_series.grid(True, alpha=0.25, linewidth=0.5)
        self._ax_series.set_title("recorded series (live)", fontsize=9)

    def _update_series(self, records: Sequence[ParameterStepRecord]) -> None:
        """Point the per-frame series lines at the recorded points.

        Goes through `visualization.recorded_series`, the same helper the static
        figure uses, so a non-finite value is never plotted and a single
        recorded step is always marked.
        """
        for attribute, _label, _colour in DASHBOARD_SERIES:
            line = self._animated.get(f"series:{attribute}")
            if line is None:
                continue
            xs, ys = viz.recorded_series(records, attribute)
            line.set_data(xs, ys)

    def _draw_bits(self, last: ParameterStepRecord | None) -> None:
        self._ax_bits.clear()
        if last is not None:
            viz.draw_bit_pattern(self._ax_bits, last)
        else:
            self._ax_bits.axis("off")
            self._ax_bits.text(0.5, 0.5, IDLE_MESSAGE, ha="center", va="center", wrap=True)

    def _draw_card(self, last: ParameterStepRecord | None) -> None:
        self._ax_card.clear()
        self._ax_card.axis("off")
        if last is None:
            self._ax_card.text(0.5, 0.5, IDLE_MESSAGE, ha="center", va="center", wrap=True)
            self._ax_card.set_title("state card (no step executed yet)", fontsize=9)
            return
        # A compact HUD rather than the full 45-line forensic card: the full card
        # is what `trace` and the report print, and rendering it every frame
        # costs more than a second on this machine.  The live card names the
        # quantities it shows; `visualization.render_state_card` is unchanged and
        # remains the complete version.
        self._ax_card.text(
            0.0, 1.0, _compact_hud(last, self.engine, self._measured_fps),
            va="top", family="monospace", fontsize=7.5,
            transform=self._ax_card.transAxes,
        )
        self._ax_card.set_title(
            f"state card (step {last.step} of {self.engine.config.steps - 1})",
            fontsize=9,
        )

    # -- animated layer -----------------------------------------------------
    def _update_animated(self, fraction: float) -> None:
        """Position the marker along the newest recorded segment."""
        marker = self._animated.get("marker")
        trail = self._animated.get("trail")
        if marker is None or trail is None:
            return
        records = self.engine.recorder.records
        if not records or self._segment is None:
            marker.set_data([], [])
            trail.set_data([], [])
            return
        first, second = self._segment
        f = 0.0 if not self.animate else min(1.0, max(0.0, float(fraction)))
        xs = [first.step + (second.step - first.step) * f]
        ys = [
            first.parameter_after
            + (second.parameter_after - first.parameter_after) * f
        ]
        marker.set_data(xs, ys)
        trail.set_data([first.step, xs[0]], [first.parameter_after, ys[0]])

    def _new_segment_if_needed(self) -> None:
        records = self.engine.recorder.records
        if len(records) >= 2:
            candidate = (records[-2], records[-1])
            if self._segment != candidate:
                self._segment = candidate
                self._segment_start = time.monotonic()

    def _snapshot(self) -> None:
        """Cache the rasterised static layer, if this backend can blit."""
        if self._figure is None or not self._blit_available:
            return
        try:
            self._background = self._figure.canvas.copy_from_bbox(self._figure.bbox)
        except Exception as exc:  # pragma: no cover - backend dependent
            self._blit_available = False
            print(
                f"[dashboard] background caching unavailable ({type(exc).__name__}: "
                f"{exc}); falling back to full redraws, so the frame rate will be "
                f"limited by the backend rather than by the view."
            )

    def _frame(self, *, redraw_static: bool = False, redraw_data: bool = True,
               active_phase: str | None = None) -> None:
        """Draw one frame: the static layer if it is due, the rest every frame."""
        if self._figure is None:
            self.build_figure()
        now = time.monotonic()
        if redraw_static:
            # `refresh` ends by taking a snapshot, so the frame that follows it
            # blits that snapshot; it must not redraw the whole figure again.
            self.refresh(active_phase=active_phase)
        self._update_series(self.engine.recorder.records)
        self._new_segment_if_needed()
        if self.animate and self._segment is not None:
            elapsed = now - self._segment_start
            seconds = max(1.0 / self.target_fps, 1e-3) if self.target_fps else 0.016
            self._update_animated(elapsed / seconds)
        else:
            self._update_animated(1.0)
        self._draw(redraw_data=redraw_data)

    def _draw(self, *, redraw_data: bool = False) -> None:
        """Blit the animated layer over the cached static layer.

        Three layers, in increasing order of how often they are redrawn:
        * static - card, bit panel, causal chain, ticks, titles - at
          ``static_hz``, because text layout dominates everything else here;
        * data - the four recorded series - whenever a step has been recorded
          since the last draw, throttled to ``data_hz``;
        * motion - the marker and its trail - every frame.
        Only the last two are drawn per frame, and neither contains text, which
        is why a frame costs single-digit milliseconds rather than a second.
        """
        canvas = self._figure.canvas
        if self._blit_available and self._background is not None:
            try:
                canvas.restore_region(self._background)
                if redraw_data:
                    for key, artist in self._animated.items():
                        if key.startswith("series:"):
                            self._figure.draw_artist(artist)
                for key in ("trail", "marker"):
                    self._figure.draw_artist(self._animated[key])
                canvas.blit(self._figure.bbox)
                return
            except Exception:  # pragma: no cover - backend dependent
                self._blit_available = False
        # No cached background: a full redraw is the only option, and it is
        # expensive, so this path is reported rather than hidden.
        canvas.draw_idle()
        if self.interactive and self.backend != "Agg":
            self._plt.pause(0.001)

    def _update_fps(self, now: float) -> None:
        """Track the achieved frame rate and show it, cheaply."""
        self._fps_samples.append(now)
        if len(self._fps_samples) > 240:
            del self._fps_samples[:120]
        if len(self._fps_samples) >= 2:
            span = self._fps_samples[-1] - self._fps_samples[0]
            if span > 0.0:
                self._measured_fps = (len(self._fps_samples) - 1) / span
        label = self._animated.get("fps")
        if label is not None:
            # Re-rasterising text every frame is the single most expensive thing
            # this view can do, so the readout is refreshed at a low rate.  The
            # window title carries the same number for free.
            if now >= self._hud_due:
                self._hud_due = now + 0.25
                label.set_text(
                    f"{self._measured_fps:.0f} FPS drawn"
                    f" | step rate {self.controller.speed:g}/s"
                )
        manager = getattr(self._figure.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            try:
                manager.set_window_title(
                    f"single parameter observatory - {self._measured_fps:.0f} FPS"
                )
            except Exception:  # pragma: no cover - backend dependent
                pass

    # -- frame output -------------------------------------------------------
    def _save_frame(self) -> None:
        """Write the current figure to the frame directory."""
        if not self.save_frames_directory or self._figure is None:
            return
        self._frame_index += 1
        if self._blit_available and self._background is not None:
            # The blit path leaves the canvas holding the animated layer over the
            # cached background, which is exactly what the window shows; a
            # full draw is needed for a correct saved raster of the static text.
            self._figure.canvas.draw()
        viz.save_figure(
            self._figure,
            os.path.join(
                self.save_frames_directory, f"frame_{self._frame_index:05d}.png"
            ),
            dpi=90,
        )

    # -- interaction --------------------------------------------------------
    def _on_key(self, event: Any) -> None:
        raw_key = getattr(event, "key", "") or ""
        lower_key = raw_key.lower()
        if raw_key == "R" or lower_key == "shift+r":
            self.controller.restart_experiment()
        elif lower_key == " ":
            self.controller.toggle()
        elif lower_key == "s":
            self.controller.single_step()
        elif lower_key == "n":
            self.controller.run_n(10)
        elif lower_key == "r":
            self.controller.reset()
        elif lower_key in ("+", "="):
            self.controller.set_speed(self.controller.speed * 2.0)
        elif lower_key == "-":
            self.controller.set_speed(self.controller.speed / 2.0)
        elif lower_key == "q":
            self._quit = True
            self.controller.pause()
        self._last_drawn_step = -1  # force a static redraw

    def _wait_until(self, deadline: float) -> None:
        """Block until ``deadline``, accurately enough to hit a frame budget.

        ``time.sleep`` has a floor of about ten milliseconds on this platform,
        so sleeping the whole gap caps the loop near 50-100 frames a second
        however small the target is.  The coarse part of the wait is slept and
        the last few milliseconds are spent reading the clock, which is what
        makes a 60 or 120 FPS target reachable.  The spin runs only while the
        dashboard is open.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            if remaining > 0.005:
                time.sleep(remaining - 0.004)

    # -- main loop ----------------------------------------------------------
    def run(
        self,
        *,
        max_seconds: float | None = None,
        poll_seconds: float | None = None,
    ) -> int:
        """Run the dashboard until the experiment finishes or the user quits.

        ``poll_seconds`` is derived from ``target_fps`` and is a *ceiling on the
        wait between frames*, not the rate itself: the loop redraws as fast as
        the backend allows, up to the target rate.
        """
        if self._figure is None:
            self.build_figure()
        self.controller.start()
        self.controller.play()
        started = time.monotonic()
        drawn = 0
        frame_interval = (
            1.0 / self.target_fps if self.target_fps else 0.0
        )
        interval = frame_interval if frame_interval else (poll_seconds or 0.001)
        next_frame = time.monotonic()

        while not self._quit and not self.engine.finished and not self.engine.aborted:
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                break
            now = time.monotonic()
            if now < next_frame:
                self._wait_until(next_frame)
                continue
            next_frame = now + interval

            records = self.engine.recorder.records
            current = records[-1].step if records else -1

            # Three independent cadences, because the three layers cost wildly
            # different amounts:
            #   static (all the text)  -> static_hz, ~140 ms when it does run
            #   data   (the series)   -> data_hz, and only if a step arrived
            #   motion (the marker)   -> every frame, single-digit ms
            # Driving the text layer from the step rate is what pinned this view
            # near 1 FPS: at 30 steps a second that is a text layout per frame.
            static_due = (
                self._background is None
                or (current != self._last_drawn_step and now >= self._static_due)
            )
            if static_due:
                self._static_due = now + self._static_interval

            data_due = current != self._data_step and now >= self._data_due
            if data_due:
                self._data_due = now + self._data_interval

            self._frame(
                redraw_static=static_due,
                redraw_data=data_due or static_due,
                active_phase="UPDATE" if current >= 0 else None,
            )
            self._last_drawn_step = current
            if data_due:
                self._data_step = current
            self._update_fps(now)
            drawn += 1
            if self.save_frames_directory and drawn % self.frame_every == 0:
                self._save_frame()

        # One final static frame, so a very short run still shows its end state.
        self._frame(redraw_static=True, active_phase=None)
        self._update_fps(time.monotonic())
        drawn += 1
        if self.save_frames_directory:
            self._save_frame()
        self.controller.pause()
        return drawn

    def close(self) -> None:
        self.controller.stop()
        if self._figure is not None:
            self._plt.close(self._figure)
            self._figure = None


def _compact_hud(
    record: ParameterStepRecord, engine: TrainingEngine, fps: float
) -> str:
    """A short live summary.  The full forensic card is `render_state_card`."""
    rows = [
        f"step {record.step} / {engine.config.steps - 1}   {fps:.0f} FPS drawn",
        "",
        f"w        {record.parameter_after!r}",
        f"hex      {record.parameter_hex}",
        f"delta    {record.actual_delta!r}  ({record.ulp_distance_same_binade:+d} ULP of w_before)",
        f"attempted {record.optimizer_attempted_update!r}  (fp64 candidate)",
        f"loss     {record.loss!r}",
        f"gradient {record.gradient!r}",
        f"lr       {record.learning_rate!r}",
        f"recon    {record.reconstruction_status} ({record.reconstruction_ulp_error} ULP)",
        "",
        f"bits changed {record.changed_bit_count} of 32",
        f"finite   {'YES' if record.finite else 'NO'}",
    ]
    return "\n".join(rows)


def print_step(engine: TrainingEngine, record: ParameterStepRecord) -> None:
    """One-line live status, used by the terminal control session."""
    print(engine.status_line())


def print_state_card(engine: TrainingEngine) -> None:
    records = engine.recorder.records
    if not records:
        print("[state] no step executed yet")
        return
    print(viz.render_state_card(records[-1], engine.config))


def run_control_session(
    engine: TrainingEngine,
    commands: Sequence[str],
    *,
    printer: Callable[[str], None] = print,
    show_card: bool = False,
) -> None:
    """A headless control session over the real engine.

    Every command maps onto a real operation on the real experiment; there is no
    replay, no scripted trajectory and no generated future step.
    """
    controller = ExperimentController(engine)
    printer("CONTROLS: play | pause | step | n <count> | speed <steps/s> | "
            "reset | restart | quit")
    controller.run_blocking(
        commands,
        on_step=(lambda record: print_state_card(engine)) if show_card else None,
        printer=printer,
    )
    controller.stop()


__all__ = [
    "DASHBOARD_SERIES",
    "IDLE_MESSAGE",
    "INTERPOLATION_NOTICE",
    "DEFAULT_TARGET_FPS",
    "DEFAULT_HUD_HZ",
    "DEFAULT_STATIC_HZ",
    "DEFAULT_DATA_HZ",
    "LiveDashboard",
    "print_step",
    "print_state_card",
    "run_control_session",
]
