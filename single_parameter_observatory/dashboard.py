"""The live dashboard.

The dashboard is a *view*, never a source of numbers.  Every redraw reads the
engine's recorded history, so pausing it changes nothing about the experiment
and un-pausing it never fabricates a frame: if no step has been executed, the
figure shows the initial state and says so.

Fluid motion
------------
The view caches static panels, recorded data and the HUD independently. Motion
frames restore those pixels and draw only a marker and trail along the newest
recorded segment. Text is rasterized on its own refresh clocks, and padded axes
expand only when a new finite observation leaves the viewport. ``--fps`` is a
render-rate ceiling; achieved throughput depends on backend, host load, training
and frame export. The execution checker records the measured rate.

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
import math
import time
from typing import Any
from collections.abc import Callable, Sequence


from . import visualization as viz
from .errors import ConfigError
from .recorder import ParameterStepRecord
from .training_engine import ExperimentController, TrainingEngine

#: The live series drawn in the dashboard's trajectory panel.  The static view's
#: series list is ``visualization.OVERVIEW_PANELS``, drawn by
#: ``visualization.build_overview_figure``; both name the same recorded
#: attributes of the same records and deliberately draw different subsets of
#: them, so the two are allowed to differ in content.  What they must not do is
#: draw a series differently: the overview's ``visualization.draw_recorded_points``
#: and these persistent lines share finite-point and missing-step helpers.
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
#: labels - is re-rasterised. Text work is kept off the motion-frame path.
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
        try:
            valid_frame_every = (
                not isinstance(frame_every, bool)
                and int(frame_every) == frame_every and frame_every >= 1
            )
        except (ValueError, TypeError, OverflowError):
            valid_frame_every = False
        if not valid_frame_every:
            raise ConfigError("frame_every must be a positive integer")
        self.frame_every = int(frame_every)
        for name, value in (("target_fps", target_fps), ("hud_hz", hud_hz),
                            ("static_hz", static_hz), ("data_hz", data_hz)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0.0:
                raise ConfigError(f"{name} must be finite and non-negative")
        self.target_fps = float(target_fps)
        # State-card and FPS refresh cadence, independent of the other panels.
        self.hud_hz = float(hud_hz) if hud_hz else 0.0
        #: Interval between HUD (state card / FPS readout) refreshes, derived from
        #: ``hud_hz``.  The hard-coded 0.25 s this replaces made ``--hud-hz`` a
        #: documented-but-inert flag.
        self._hud_interval = 1.0 / self.hud_hz if self.hud_hz > 0.0 else 0.25
        self.animate = animate
        if self.save_frames_directory:
            os.makedirs(self.save_frames_directory, exist_ok=True)
        self.backend = viz.configure_backend(interactive)
        self._plt = viz._require_pyplot()
        self.interactive = bool(interactive and self.backend.lower() != "agg")
        self._figure: Any = None
        self._last_drawn_step = -1
        self._drawn_phase: str | None = None
        self._frame_index = 0
        self._quit = False
        self._window_closed = False
        # animated-layer artists, created with the first record
        self._animated: dict[str, Any] = {}
        self._background: Any = None
        self._background_without_hud: Any = None
        self._data_background: Any = None
        self._blit_available = False
        self._hud_due = 0.0
        self._card_due = 0.0
        self._hud_step = -1
        self._fps_dirty = False
        self._static_step = -1

        # Bit panels, causal chain and static axes use this clock. The HUD has
        # its own cached update, and recorded data have a separate cadence.
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
        self._recorder = engine.recorder
        self._series_recorder = engine.recorder
        self._series_limits_initialized = False
        self._idle_text: Any = None

    # -- figure -------------------------------------------------------------
    def build_figure(self) -> Any:
        if self._figure is not None:
            return self._figure
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
            # Matplotlib also binds s to Save and r to Home. The documented
            # experiment keys must invoke only their dashboard operation.
            default_handler = getattr(self._figure.canvas.manager, "key_press_handler_id", None)
            if default_handler is not None:
                self._figure.canvas.mpl_disconnect(default_handler)
            self._figure.canvas.mpl_connect("key_press_event", self._on_key)
            self._figure.canvas.mpl_connect("close_event", self._on_close)
        self._figure.canvas.mpl_connect("resize_event", self._on_resize)
        self._blit_available = bool(self._figure.canvas.supports_blit)
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
        self._animated["hud:card"] = self._ax_card.text(
            0.0, 1.0, "", va="top", family="monospace", fontsize=7.5,
            transform=self._ax_card.transAxes,
        )
        self._animated["hud:title"] = self._ax_card.title
        for name, artist in self._animated.items():
            artist.set_animated(self._blit_available and name != "notice")

    def _reattach_animated(self) -> None:
        """Re-add the per-frame artists after the series axes have been cleared."""
        ax = self._ax_series
        for key in ("trail", "marker"):
            if self._animated[key] not in ax.lines:
                ax.add_line(self._animated[key])
        for key, artist in self._animated.items():
            if key.startswith("series:"):
                if artist not in ax.lines:
                    ax.add_line(artist)
        for key in ("notice", "fps"):
            if self._animated[key] not in ax.texts:
                ax.add_artist(self._animated[key])

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
        self._rescale_series_axes(records)
        self._legend()
        self._new_segment_if_needed()
        self._update_animated(1.0)
        self._snapshot()
        self._static_step = last.step if last else -1
        self._hud_step = self._static_step

    def _rescale_series_axes(self, records: Sequence[ParameterStepRecord]) -> None:
        """Expand padded limits only when recorded data leave the viewport.

        Exact autoscaling on every sample changes tick labels and invalidates
        the static raster on every data frame. Keep a useful future step margin
        and 25% vertical headroom while still admitting each new finite point.
        """
        ax = self._ax_series
        if self._series_recorder is not self.engine.recorder:
            self._series_recorder = self.engine.recorder
            self._series_limits_initialized = False
        if not records:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            self._series_limits_initialized = False
            return
        ax.relim()
        x0, y0, x1, y1 = ax.dataLim.extents
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
            return
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        if not self._series_limits_initialized or x0 < left or x1 > right:
            ax.set_xlim(min(-0.5, x0 - 0.5), max(10.0, x1 * 2.0 + 1.0))
        if not self._series_limits_initialized or y0 < bottom or y1 > top:
            margin = max((y1 - y0) * 0.25, abs(y0) * 0.05, abs(y1) * 0.05, 1e-9)
            ax.set_ylim(y0 - margin, y1 + margin)
        self._series_limits_initialized = True

    def _legend(self) -> None:
        """Build the series legend from the persistent artists.

        The legend must be made after the artists are re-added: `Axes.clear`
        removes them, and a legend built before that finds nothing to label.
        """
        if self.engine.recorder.records:
            if self._idle_text is not None:
                self._idle_text.remove()
                self._idle_text = None
            if self._ax_series.get_legend() is None:
                self._ax_series.legend(fontsize=7)
        else:
            legend = self._ax_series.get_legend()
            if legend is not None:
                legend.remove()
            if self._idle_text is None:
                self._idle_text = self._ax_series.text(
                    0.5, 0.5, IDLE_MESSAGE, ha="center", va="center", wrap=True,
                    transform=self._ax_series.transAxes,
                )

    def _setup_series_axes(self, records: Sequence[ParameterStepRecord]) -> None:
        """Static furniture of the series panel: frame, ticks, legend, title."""
        self._ax_series.set_xlabel("step", fontsize=8)
        # Keep axis labels readable and their layout cost bounded.
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

        Uses the static figure's line-coordinate helper, so unavailable values
        leave real gaps and a single recorded step is always marked.
        """
        for attribute, _label, _colour in DASHBOARD_SERIES:
            line = self._animated.get(f"series:{attribute}")
            if line is None:
                continue
            xs, ys = viz.recorded_line_series(records, attribute)
            line.set_data(xs, ys)

    def _draw_bits(self, last: ParameterStepRecord | None) -> None:
        self._ax_bits.clear()
        if last is not None:
            viz.draw_bit_pattern(self._ax_bits, last)
        else:
            self._ax_bits.axis("off")
            self._ax_bits.text(0.5, 0.5, IDLE_MESSAGE, ha="center", va="center", wrap=True)

    def _draw_card(self, last: ParameterStepRecord | None) -> None:
        self._ax_card.axis("off")
        text = self._animated["hud:card"]
        if last is None:
            text.set_text(IDLE_MESSAGE)
            self._ax_card.set_title("state card (no step executed yet)", fontsize=9)
            return
        # The live HUD names its quantities. Trace and render_state_card retain
        # the full forensic inspector without placing it on the motion path.
        text.set_text(_compact_hud(last, self.engine, self._measured_fps))
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
        f = 1.0 if not self.animate else min(1.0, max(0.0, float(fraction)))
        xs = [first.step + (second.step - first.step) * f]
        ys = [
            first.parameter_after
            + (second.parameter_after - first.parameter_after) * f
        ]
        marker.set_data(xs, ys)
        trail.set_data([first.step, xs[0]], [first.parameter_after, ys[0]])

    def _new_segment_if_needed(self) -> None:
        records = self.engine.recorder.records
        if self._recorder is not self.engine.recorder or len(records) < 2:
            self._segment = None
            self._recorder = self.engine.recorder
        if len(records) >= 2:
            candidate = (records[-2], records[-1])
            if not all(math.isfinite(r.parameter_after) for r in candidate):
                self._segment = None
            elif self._segment is None or any(a is not b for a, b in zip(self._segment, candidate, strict=True)):
                self._segment = candidate
                self._segment_start = time.monotonic()

    def _snapshot(self) -> None:
        """Cache the rasterised static layer, if this backend can blit."""
        if self._figure is None or not self._blit_available:
            return
        try:
            # A background copied before the renderer draws the static artists
            # contains a blank/stale canvas. Animated artists are excluded from
            # this draw, so subsequent restores cannot leave ghost markers.
            self._figure.canvas.draw()
            self._background_without_hud = self._figure.canvas.copy_from_bbox(self._figure.bbox)
            for key in ("hud:card", "hud:title", "fps"):
                self._figure.draw_artist(self._animated[key])
            self._background = self._figure.canvas.copy_from_bbox(self._figure.bbox)
            self._data_background = None
            self._fps_dirty = False
        except Exception as exc:  # pragma: no cover - backend dependent
            self._blit_available = False
            for artist in self._animated.values():
                artist.set_animated(False)
            print(
                f"[dashboard] background caching unavailable ({type(exc).__name__}: "
                f"{exc}); falling back to full redraws, so the frame rate will be "
                f"limited by the backend rather than by the view."
            )

    def _snapshot_hud(self) -> None:
        """Replace cached HUD pixels without re-rasterizing the other panels."""
        if not self._blit_available or self._background_without_hud is None:
            self._snapshot()
            return
        canvas = self._figure.canvas
        canvas.restore_region(self._background_without_hud)
        for key in ("hud:card", "hud:title", "fps"):
            self._figure.draw_artist(self._animated[key])
        self._background = canvas.copy_from_bbox(self._figure.bbox)
        self._data_background = None
        self._fps_dirty = False

    def _frame(self, *, redraw_static: bool = False, redraw_data: bool = True,
               redraw_hud: bool = False,
               active_phase: str | None = None) -> None:
        """Draw one frame: the static layer if it is due, the rest every frame."""
        if self._figure is None:
            self.build_figure()
        refreshed = False
        if redraw_static or self._background is None:
            # `refresh` ends by taking a snapshot, so the frame that follows it
            # blits that snapshot; it must not redraw the whole figure again.
            self.refresh(active_phase=active_phase)
            redraw_data = True
            refreshed = True
        elif redraw_hud or self._fps_dirty:
            records = self.engine.recorder.records
            self._draw_card(records[-1] if records else None)
            self._hud_step = records[-1].step if records else -1
            self._snapshot_hud()
            redraw_data = True
        if redraw_data and not refreshed:
            records = self.engine.recorder.records
            self._update_series(records)
            previous_limits = (self._ax_series.get_xlim(), self._ax_series.get_ylim())
            self._rescale_series_axes(records)
            if previous_limits != (self._ax_series.get_xlim(), self._ax_series.get_ylim()):
                self._snapshot()
        self._new_segment_if_needed()
        now = time.monotonic()
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
        Data are redrawn only when due; motion frames redraw no text.
        """
        canvas = self._figure.canvas
        if self._blit_available and self._background is not None:
            try:
                if redraw_data or self._data_background is None:
                    canvas.restore_region(self._background)
                    for key, artist in self._animated.items():
                        if key.startswith("series:"):
                            self._figure.draw_artist(artist)
                    # Keep recorded lines on their own layer. Restoring only
                    # the static layer on a motion frame erases those lines.
                    self._data_background = canvas.copy_from_bbox(self._figure.bbox)
                else:
                    canvas.restore_region(self._data_background)
                for key in ("trail", "marker"):
                    self._figure.draw_artist(self._animated[key])
                canvas.blit(self._figure.bbox)
                if self.interactive:
                    canvas.flush_events()
                return
            except Exception:  # pragma: no cover - backend dependent
                self._blit_available = False
                for artist in self._animated.values():
                    artist.set_animated(False)
        # No cached background: a full redraw is the only option, and it is
        # expensive, so this path is reported rather than hidden.
        canvas.draw_idle()
        if self.interactive:
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
                self._hud_due = now + self._hud_interval
                label.set_text(
                    f"{self._measured_fps:.0f} FPS drawn"
                    f" | step rate {self.controller.speed:g}/s"
                )
                self._fps_dirty = True
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
        animated_flags = {name: artist.get_animated() for name, artist in self._animated.items()}
        try:
            # savefig draws a fresh renderer and normally omits animated
            # artists. Include the recorded series and the display marker.
            for artist in self._animated.values():
                artist.set_animated(False)
            viz.save_figure(
                self._figure,
                os.path.join(self.save_frames_directory, f"frame_{self._frame_index:05d}.png"),
                dpi=90,
            )
        finally:
            for name, artist in self._animated.items():
                artist.set_animated(animated_flags[name])
            # savefig may use another DPI/renderer, invalidating cached pixels.
            self._background = None
            self._data_background = None

    def _on_resize(self, _event: Any) -> None:
        self._background = None
        self._data_background = None

    def _on_close(self, _event: Any) -> None:
        self._quit = True
        self._window_closed = True
        self.controller.pause()

    # -- interaction --------------------------------------------------------
    def _on_key(self, event: Any) -> None:
        raw_key = getattr(event, "key", "") or ""
        lower_key = raw_key.lower()
        if raw_key == "R" or lower_key == "shift+r":
            self.controller.restart_experiment()
            self._segment = None
            self._background = None
            self._data_background = None
        elif lower_key in (" ", "space"):
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
        self._static_due = 0.0

    def _wait_until(self, deadline: float) -> None:
        """Wait in short slices while servicing window events and failures."""
        while True:
            remaining = deadline - time.monotonic()
            self.controller.raise_if_failed()
            if remaining <= 0.0 or self._quit or self.engine.finished or self.engine.aborted:
                return
            time.sleep(min(remaining, 0.01))
            if self.interactive and self._figure is not None:
                self._figure.canvas.flush_events()

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
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds < 0.0):
            raise ConfigError("max_seconds must be finite and non-negative")
        if poll_seconds is not None and (not math.isfinite(poll_seconds) or poll_seconds <= 0.0):
            raise ConfigError("poll_seconds must be finite and positive")
        if self._figure is None:
            self.build_figure()
        if self.interactive:
            self._plt.show(block=False)
        self.controller.start()
        if max_seconds != 0.0:
            self.controller.play()
        started = time.monotonic()
        drawn = 0
        frame_interval = (
            1.0 / self.target_fps if self.target_fps else 0.0
        )
        interval = frame_interval if frame_interval else (poll_seconds or 0.001)
        if poll_seconds is not None:
            interval = min(interval, poll_seconds)
        next_frame = time.monotonic()

        try:
            while not self._quit and not self.engine.finished and not self.engine.aborted:
                self.controller.raise_if_failed()
                if max_seconds is not None and time.monotonic() - started >= max_seconds:
                    break
                now = time.monotonic()
                if now < next_frame:
                    deadline = next_frame
                    if max_seconds is not None:
                        deadline = min(deadline, started + max_seconds)
                    self._wait_until(deadline)
                    continue
                next_frame = now + interval

                records = self.engine.recorder.records
                current = records[-1].step if records else -1
                static_due = (
                    self._background is None
                    or (current != self._static_step and now >= self._static_due)
                )
                if static_due:
                    self._static_due = now + self._static_interval
                hud_due = current != self._hud_step and now >= self._card_due
                if hud_due:
                    self._card_due = now + self._hud_interval
                data_due = current != self._data_step and now >= self._data_due
                if data_due:
                    self._data_due = now + self._data_interval

                self._update_fps(now)
                self._frame(
                    redraw_static=static_due,
                    redraw_data=data_due or static_due,
                    redraw_hud=hud_due,
                    active_phase="UPDATE" if current >= 0 else None,
                )
                self._last_drawn_step = current
                if data_due:
                    self._data_step = current
                drawn += 1
                if self.save_frames_directory and not self._window_closed and drawn % self.frame_every == 0:
                    self._save_frame()
        finally:
            # A step can still be in flight when the deadline/window closes.
            # Stop its writer before the final frame or report reads history.
            self.controller.stop()
        self.controller.raise_if_failed()
        if self._window_closed:
            if not self.save_frames_directory:
                return drawn
            # Closing a GUI destroys its native canvas. Render the requested
            # final PNG with Agg without reopening the user's window.
            from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415 - closed GUI fallback

            FigureCanvasAgg(self._figure)
            self.interactive = False
            self.backend = "Agg"
            self._blit_available = True
            self._background = None
            self._data_background = None

        # One final static frame, so a very short run still shows its end state.
        # The final frame keeps the UPDATE phase highlighted: rendering it with
        # no active phase made the last saved frame of every run show
        # "active phase: none", which reads as a defect rather than an end state.
        final_phase = "UPDATE" if self.engine.recorder.records else None
        self._frame(redraw_static=True, active_phase=final_phase)
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
        self._animated.clear()
        self._background = None
        self._background_without_hud = None
        self._data_background = None
        self._segment = None
        self._drawn_phase = None
        self._idle_text = None
        self._series_limits_initialized = False


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
    def show_state(record: ParameterStepRecord) -> None:
        printer(viz.render_state_card(record, engine.config))

    try:
        controller.run_blocking(
            commands,
            on_step=show_state if show_card else None,
            printer=printer,
        )
    finally:
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
