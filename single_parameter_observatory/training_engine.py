"""The training engine: one real step, and the control surface around it.

The engine performs exactly one PyTorch training step per :meth:`TrainingEngine.step`
call, in this order, with an observation taken at every labelled boundary:
    A  read the parameter
    B  real forward  (parameter must be unchanged by this)
    C  real backward (parameter must still be unchanged)
    D  read the parameter and the optimizer state
    E  real optimizer.step()
    F  any subsequent framework operation (the LR schedule step)

Between D and E the engine reads everything the reconstruction needs and clones
it, so the reconstruction can be computed after the fact without any risk of
having disturbed the run it is reconstructing.

Nothing in this file fabricates a value.  Every number in a
:class:`~.recorder.ParameterStepRecord` is read out of a real tensor.
"""

from __future__ import annotations

import math
import copy
import threading
import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Mapping, Sequence

import torch

from . import fp32_inspector as f32
from . import numerical_analysis as na
from . import optimizer_inspector as oi
from . import validation as vd
from .config import ExperimentConfig
from .convergence import ConvergenceAnalyzer
from .errors import ConfigError, NumericalConsistencyError
from .model import Dataset, FP64ReferenceModel, ObservatoryModel, build_dataset
from .recorder import ParameterStepRecord, Recorder
from .tracker import TrackedParameter

#: Callback invoked after every completed step, used by the live views.
StepCallback = Callable[[ParameterStepRecord], None]


# --------------------------------------------------------------------------- #
# control surface
# --------------------------------------------------------------------------- #


class RunState:
    """The four states of the control surface (plain strings, no globals)."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ABORTED = "aborted"


class ExperimentController:
    """PLAY / PAUSE / SINGLE STEP / RUN N / RESET / RESTART, plus a speed control.

    Pausing really pauses: the worker thread performs no further steps while the
    state is ``PAUSED``.  The state lives here, not in the visualisation, so a
    pause behaves identically in the terminal session and in the GUI.
    """

    def __init__(self, engine: TrainingEngine, *, autostart: bool = False) -> None:
        self.engine = engine
        self._lock = threading.RLock()
        self._error_lock = threading.Lock()
        self._state = RunState.IDLE
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_single_steps = 0
        self._pending_run_n = 0
        self._continuous = False
        self._worker_error: BaseException | None = None
        self._speed = float(engine.config.live_speed_steps_per_second)
        self._tick_seconds = 0.02
        self.on_step: StepCallback | None = None
        self.on_state_change: Callable[[str], None] | None = None
        if autostart:
            self.start()

    # -- state --------------------------------------------------------------
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self.state == RunState.RUNNING

    def _set_state(self, state: str) -> None:
        with self._lock:
            if self._state == state:
                return
            self._state = state
        if self.on_state_change is not None:
            self.on_state_change(state)

    # -- speed --------------------------------------------------------------
    @property
    def speed(self) -> float:
        return self._speed

    def set_speed(self, steps_per_second: float) -> None:
        if not math.isfinite(steps_per_second) or steps_per_second <= 0.0:
            raise ConfigError("speed must be positive and finite")
        with self._lock:
            self._speed = float(steps_per_second)

    # -- commands -----------------------------------------------------------
    def play(self) -> None:
        with self._lock:
            self.raise_if_failed()
            if self.engine.finished:
                self._set_state(RunState.ABORTED if self.engine.aborted else RunState.FINISHED)
                return
            self._continuous = True
            self._pending_single_steps = self._pending_run_n = 0
            self._set_state(RunState.RUNNING)

    def pause(self) -> None:
        # The worker holds this same lock for a whole step and its callback.
        # Once pause returns, no writer is still mutating the recorded run.
        with self._lock:
            self._continuous = False
            self._pending_single_steps = self._pending_run_n = 0
            self._set_state(RunState.PAUSED)

    def toggle(self) -> None:
        if self.state == RunState.RUNNING:
            self.pause()
        else:
            self.play()

    def single_step(self) -> bool:
        """Request exactly one more real step while staying paused."""
        with self._lock:
            self.raise_if_failed()
            if self.engine.finished:
                return False
            self._continuous = False
            self._pending_run_n = 0
            self._pending_single_steps += 1
            self._set_state(RunState.PAUSED)
        return True

    def run_n(self, count: int) -> bool:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            return False
        with self._lock:
            self.raise_if_failed()
            if self.engine.finished:
                return False
            self._continuous = False
            self._pending_single_steps = 0
            self._pending_run_n += count
            self._set_state(RunState.RUNNING)
        return True

    def reset(self) -> None:
        """Restore the live objects to the initial condition, keeping history."""
        with self._lock:
            self.engine.reset_live_state()
            self._pending_single_steps = 0
            self._pending_run_n = 0
            self._continuous = False
            self._set_state(RunState.PAUSED)

    def restart_experiment(self) -> None:
        """Discard the history and begin a completely new run."""
        with self._lock:
            self.engine.restart()
            self._pending_single_steps = 0
            self._pending_run_n = 0
            self._continuous = False
            with self._error_lock:
                self._worker_error = None
            self._set_state(RunState.IDLE)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("the training worker did not stop within 5 seconds")
        with self._lock:
            self._continuous = False
            self._pending_single_steps = self._pending_run_n = 0
            if self._thread is thread:
                self._thread = None

    def raise_if_failed(self) -> None:
        """Surface an unexpected background exception on the caller's thread."""
        # Polling is part of rendering as well as control commands. It must not
        # wait for the training gate, which spans a whole step and callback.
        with self._error_lock:
            error = self._worker_error
        if error is not None:
            raise error

    # -- worker -------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self.raise_if_failed()
            if self._thread is not None and self._thread.is_alive():
                return
            if self._state == RunState.IDLE:
                self._state = RunState.PAUSED
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="observatory", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                performed = False
                with self._lock:
                    if self._stop.is_set():
                        break
                    permitted = (
                        self._state == RunState.PAUSED and self._pending_single_steps > 0
                    ) or (
                        self._state == RunState.RUNNING
                        and (self._continuous or self._pending_run_n > 0)
                    )
                    if permitted and not self.engine.finished:
                        if self._state == RunState.PAUSED:
                            self._pending_single_steps -= 1
                        elif not self._continuous:
                            self._pending_run_n -= 1
                        record = self.engine.step()
                        performed = record is not None
                        if self.on_step is not None and record is not None:
                            self.on_step(record)
                        if self.engine.finished:
                            self._set_state(RunState.ABORTED if self.engine.aborted else RunState.FINISHED)
                        elif self._state == RunState.RUNNING and not self._continuous and not self._pending_run_n:
                            self._set_state(RunState.PAUSED)
                    delay = 1.0 / self._speed if performed else self._tick_seconds
                self._stop.wait(delay)
        except BaseException as exc:
            with self._lock:
                with self._error_lock:
                    self._worker_error = exc
                self._continuous = False
                self._pending_single_steps = self._pending_run_n = 0
                self._state = RunState.ABORTED

    # -- synchronous command loop (terminal mode) ---------------------------
    def run_blocking(
        self,
        commands: Sequence[str],
        *,
        on_step: StepCallback | None = None,
        printer: Callable[[str], None] = print,
    ) -> None:
        """Execute an explicit command list against the real engine.

        Used by the headless control session: every command is a real control
        operation, and every step performed is a real step.
        """
        self.on_step = on_step
        for command in commands:
            token = command.strip().lower()
            if not token or token.startswith("#"):
                continue
            verb, _, argument = token.partition(" ")
            if (verb == "n" or verb == "runn") and argument.strip().isdigit():
                count = int(argument)
                if self.run_n(count):
                    self._drain_run(printer, limit=count)
            elif verb.startswith("n") and verb[1:].isdigit():
                count = int(verb[1:])
                if self.run_n(count):
                    self._drain_run(printer, limit=count)
            elif verb in ("play", "run"):
                self.play()
                self._drain_run(printer, limit=None)
            elif verb == "pause":
                self.pause()
                printer("[control] paused")
            elif verb in ("step", "s"):
                if self.single_step():
                    self._drain_run(printer, limit=1)
            elif verb in ("speed", "sp"):
                if argument.strip():
                    try:
                        requested_speed = float(argument)
                    except ValueError as exc:
                        raise ConfigError("speed must be a finite positive number") from exc
                    self.set_speed(requested_speed)
                printer(f"[control] speed = {self.speed:g} steps/s")
            elif verb == "reset":
                self.reset()
                printer("[control] live state reset to the initial condition")
            elif verb == "restart":
                self.restart_experiment()
                printer("[control] new experiment started, history discarded")
            elif verb in ("quit", "q"):
                printer("[control] finished")
                return
            else:
                printer(f"[control] unknown command: {command!r}")

    def _drain_run(
        self, printer: Callable[[str], None], *, limit: int | None
    ) -> None:
        with self._lock:
            self._pending_single_steps = self._pending_run_n = 0
        performed = 0
        while not self.engine.finished and not self.engine.aborted:
            if limit is not None and performed >= limit:
                break
            record = self.engine.step()
            if record is None:
                break
            performed += 1
            if self.on_step is not None:
                self.on_step(record)
            printer(self.engine.status_line())
        if self.engine.finished:
            self._set_state(RunState.ABORTED if self.engine.aborted else RunState.FINISHED)
        elif limit is not None:
            self._set_state(RunState.PAUSED)


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #


@dataclass
class FP64Track:
    """The independent float64 reference trajectory."""

    model: FP64ReferenceModel
    optimizer: torch.optim.Optimizer
    parameter: torch.nn.Parameter
    handle: oi.OptimizerHandle

    def step(
        self, x: torch.Tensor, target: torch.Tensor,
        *, batches: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[float, float]:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for batch_x, batch_target in batches if batches is not None else ((x, target),):
            loss = self.model.loss(batch_x, batch_target)
            loss.backward()
            losses.append(float(loss.detach().item()))
        self.optimizer.step()
        return float(self.parameter.detach().item()), sum(losses)


class TrainingEngine:
    """Owns the model, the optimizer, the instruments, and the recorded history."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.recorder = Recorder(config)
        self.analyzer = ConvergenceAnalyzer(config.stability)
        self.aborted = False
        self.abort_reason: str | None = None
        self.step_index = 0
        self._active_variant: str | None = None
        self._variant_locked = False
        self._lr_scheduler: Any = None
        self._build()

    # -- construction -------------------------------------------------------
    def _build(self) -> None:
        # Model construction is deterministic and the dataset has a dedicated
        # Generator; constructing an observatory must not reseed its host app.
        self._live_step_index = 0
        self._active_variant = None
        self._variant_locked = False
        self.model = ObservatoryModel(self.config.initial_value, self.config.model)
        self.parameter = self.model.assert_single_float32_parameter()
        self.tracker = TrackedParameter(self.model)
        self.handle = oi.make_optimizer(self.config, self.parameter)
        self.dataset: Dataset = build_dataset(self.config.dataset, self.config.model)
        self._setup_schedule()
        self.fp64_track: FP64Track | None = None
        if self.config.run_fp64_reference_track:
            fp64_model = FP64ReferenceModel(self.config.initial_value, self.config.model)
            fp64_parameter = fp64_model.w
            fp64_handle = oi.make_optimizer(self.config, fp64_parameter)
            self.fp64_track = FP64Track(
                model=fp64_model,
                optimizer=fp64_handle.optimizer,
                parameter=fp64_parameter,
                handle=fp64_handle,
            )
        self._bit_validation: vd.BitValidation | None = None
        self._ulp_validation: vd.ULPValidation | None = None
        self._semantics_probe: dict[str, Any] | None = None
        self._provenance_cache: dict[str, Any] | None = None
        self._emit_initialised()

    def _setup_schedule(self) -> None:
        self._lr_scheduler = None
        if self.config.lr_schedule == "cosine":
            try:
                self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.handle.optimizer,
                    T_max=max(1, self.config.steps),
                    eta_min=self.config.lr * self.config.lr_final_factor,
                )
                self.schedule_note = "torch.optim.lr_scheduler.CosineAnnealingLR"
            except Exception as exc:  # pragma: no cover - defensive
                self._lr_scheduler = None
                self.schedule_note = (
                    f"closed-form cosine (scheduler unavailable: {type(exc).__name__})"
                )
        elif self.config.lr_schedule == "exponential":
            self.schedule_note = "closed-form exponential decay applied to param_group['lr']"
        else:
            self.schedule_note = "constant"

    # -- provenance ---------------------------------------------------------
    def provenance(self) -> dict[str, Any]:
        """Everything needed to interpret this run, gathered once and cached."""
        if self._provenance_cache is not None:
            return copy.deepcopy(self._provenance_cache)
        if self._semantics_probe is None:
            self._semantics_probe = oi.probe_optimizer_semantics(self.config)
            self._active_variant = str(self._semantics_probe.get("measured_variant"))
        if self._bit_validation is None:
            self._bit_validation = vd.validate_float32_bits(self.parameter)
            self.recorder.ledger.add_bit(self._bit_validation)
        if self._ulp_validation is None:
            self._ulp_validation = vd.validate_ulp(float(self.tracker.value()))
            self.recorder.ledger.add_ulp(self._ulp_validation)
        self._provenance_cache = {
            "environment": oi.audit_environment().to_dict(),
            "optimizer_source_audit": self.handle.audit.to_dict(),
            "optimizer_semantics_probe": self._semantics_probe,
            "parameter": self.model.describe(),
            "dataset": self.dataset.describe(),
            "lr_schedule": self.schedule_note,
            "bit_validation": self._bit_validation.to_dict() if self._bit_validation else None,
            "ulp_validation": self._ulp_validation.to_dict() if self._ulp_validation else None,
            "notes": (
                "Real training: torch loss.backward() and optimizer.step() only. "
                "No gradient is assigned, no parameter value is written outside the "
                "optimizer, and no trajectory is precomputed.",
                "The float64 track is a separate real run of the same forward shape "
                "in a different precision. It is a reference, not ground truth.",
            ),
        }
        return copy.deepcopy(self._provenance_cache)

    # -- lifecycle ----------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.step_index >= self.config.steps or self.aborted

    def reset_live_state(self) -> None:
        """Restore the parameter and optimizer state to the initial condition.

        This is an explicit, labelled reset operation, not a training mechanism:
        the optimizer state is genuinely discarded and the parameter genuinely
        returns to its configured initial value.
        """
        history_length = len(self.recorder)
        with torch.no_grad():
            self.parameter.fill_(float(self.config.initial_value))
        self.parameter.grad = None
        self.handle.optimizer.state.clear()
        self.handle.set_lr(self.config.lr)
        self._active_variant = None
        self._variant_locked = False
        self._live_step_index = 0
        self.aborted = False
        self.abort_reason = None
        self.recorder.aborted_reason = None
        self.analyzer = ConvergenceAnalyzer(self.config.stability)
        self._setup_schedule()
        if self.fp64_track is not None:
            with torch.no_grad():
                self.fp64_track.parameter.fill_(float(self.config.initial_value))
            self.fp64_track.optimizer.state.clear()
            self.fp64_track.parameter.grad = None
            self.fp64_track.handle.set_lr(self.config.lr)
        self.recorder.emit(
            self.step_index,
            "INITIALIZED",
            f"live state reset to the initial value {self.config.initial_value!r}; "
            f"the recorded history of {history_length} steps is kept",
        )

    def restart(self) -> None:
        """Discard everything and begin a brand-new run from step 0."""
        self.recorder = Recorder(self.config)
        self.analyzer = ConvergenceAnalyzer(self.config.stability)
        self.aborted = False
        self.abort_reason = None
        self.step_index = 0
        self._semantics_probe = None
        self._bit_validation = None
        self._ulp_validation = None
        self._build()

    def _emit_initialised(self) -> None:
        bits = f32.inspect_float32(self.parameter)
        self.recorder.emit(
            0,
            "INITIALIZED",
            f"w = {bits.value!r} ({bits.hex}, exponent {bits.exponent_bits}, "
            f"fraction {bits.fraction_bits_int})",
        )

    # -- one real step ------------------------------------------------------
    def step(self) -> ParameterStepRecord | None:
        """Perform one real training step and record everything about it."""
        if self.aborted:
            return None
        if self.step_index >= self.config.steps:
            return None

        step = self.step_index
        if self._provenance_cache is None:
            self.provenance()  # force the audits to happen once, up front
        self.tracker.verify(step=step)

        # --- learning rate actually in force for this step
        self._apply_schedule(self._live_step_index)
        lr_used = self.handle.current_lr()

        # --- STATE A: before forward
        snapshot_a = self.tracker.raw_uint32()
        hash_before = self.tracker.content_hash()

        # --- gradient lifecycle, measured rather than assumed
        grad_before_zero = self.tracker.gradient_value()
        self.handle.optimizer.zero_grad(set_to_none=self.config.zero_grad_set_to_none)
        grad_after_zero = self.tracker.gradient_value()

        grad_before_backward: float | None = None
        micro_batches = max(1, self.config.accumulation_micro_batches)
        accumulated_losses: list[float] = []
        objective_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        forward_finite = True
        forward_record = None
        x = target = None
        batch_index = epoch = global_batch = 0
        gradient_tensor: torch.Tensor | None = None
        snapshot_b: int | None = None

        for micro in range(micro_batches):
            micro_step = self._live_step_index * micro_batches + micro
            x, target, batch_index, epoch, global_batch = self.dataset.batch(micro_step)
            objective_batches.append((x, target))
            if micro == 0:
                grad_before_backward = self.tracker.gradient_value()

            # REAL forward pass through the real graph
            _prediction_tensor, loss_tensor, forward_record = self.model.forward_with_trace(
                x, target
            )
            accumulated_losses.append(float(loss_tensor.detach().item()))
            forward_finite = forward_finite and all(stage.finite for stage in forward_record.stages)
            self.recorder.emit(
                step, "FORWARD_COMPLETE", "prediction and intermediates recorded"
            )
            self.recorder.emit(
                step, "LOSS_COMPUTED", f"loss = {accumulated_losses[-1]!r}"
            )

            # STATE B: snapshot right after forward (before backward)
            if snapshot_b is None:
                snapshot_b = self.tracker.raw_uint32()

            # REAL backward pass: this is the only source of the gradient
            self.recorder.emit(step, "BACKWARD_STARTED", "loss.backward()")
            loss_tensor.backward()
            self.recorder.emit(step, "GRADIENT_AVAILABLE", "parameter.grad written by autograd")
            if micro_batches > 1:
                self.recorder.emit(
                    step,
                    "GRADIENT_ACCUMULATED",
                    f"micro-batch {micro + 1}/{micro_batches} summed into parameter.grad",
                )

        assert forward_record is not None
        assert snapshot_b is not None
        # The loop above ran at least one micro-batch, so the batch tensors
        # exist; narrowing them for the reader (and the type checker) rather
        # than sprinkling Optional through the record assembly below.
        assert x is not None
        assert target is not None

        gradient_tensor = self.parameter.grad
        if gradient_tensor is None:
            raise NumericalConsistencyError(
                f"step {step}: autograd produced no gradient for the tracked parameter"
            )
        grad_after_backward = self.tracker.gradient_value()
        if grad_after_backward is None:
            # The tensor exists (checked just above) but ``.grad`` reading
            # returned nothing, which cannot happen for a written gradient.
            raise NumericalConsistencyError(
                f"step {step}: the gradient was written but could not be read back"
            )
        gradient_snapshot = gradient_tensor.detach().clone()
        gradient_meta = self.tracker.describe_gradients()

        # --- STATE C: after backward (parameter must still not have moved)
        snapshot_c = self.tracker.raw_uint32()
        if not (snapshot_a == snapshot_b == snapshot_c):
            raise NumericalConsistencyError(
                f"step {step}: the parameter changed during forward/backward "
                f"(0x{snapshot_a:08X} -> 0x{snapshot_b:08X} -> 0x{snapshot_c:08X}); "
                "forward and backward must not update parameters"
            )

        # --- diagnostics that must not touch the live parameter
        landscape_data = self._probe_landscape(step, snapshot_c, hash_before, batches=objective_batches)
        gradient_validation = self._validate_gradient(
            step=step,
            x=x,
            target=target,
            w_value=self.tracker.value(),
            autograd_gradient=grad_after_backward,
            gradient_meta=gradient_meta,
            batches=objective_batches,
        )

        # --- STATE D: everything the reconstruction needs, read and cloned now
        snapshot_d = self.tracker.raw_uint32()
        weight_before = self.tracker.clone()
        state_before = _clone_optimizer_state(self.handle.optimizer, self.parameter)
        state_reading_before = oi.read_optimizer_state(self.handle.optimizer, self.parameter)
        self.recorder.emit(
            step,
            "OPTIMIZER_STATE_READ",
            f"lr in force = {lr_used!r}; optimizer state before the step is "
            + (
                "present with keys " + ", ".join(state_reading_before.keys)
                if state_reading_before.keys
                else "present but empty"
                if state_reading_before.present
                else "not yet initialised for this parameter"
            ),
        )

        # --- the real optimizer step
        self.recorder.emit(step, "OPTIMIZER_STEP_STARTED", f"{self.config.optimizer}.step()")
        self.handle.optimizer.step()
        self.recorder.emit(step, "OPTIMIZER_STEP_COMPLETED", "optimizer.step() returned")

        # --- STATE E
        snapshot_e = self.tracker.raw_uint32()
        weight_after = self.tracker.clone()
        state_reading_after = oi.read_optimizer_state(self.handle.optimizer, self.parameter)

        # --- STATE F: the only framework operation that follows the step
        snapshot_f: int | None = None
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
            snapshot_f = self.tracker.raw_uint32()
            if snapshot_f != snapshot_e:
                raise NumericalConsistencyError(
                    f"step {step}: the learning-rate scheduler changed the parameter "
                    f"(0x{snapshot_e:08X} -> 0x{snapshot_f:08X})"
                )
        elif self.config.lr_schedule == "exponential":
            self.handle.set_lr(self.handle.current_lr() * self.config.lr_final_factor)
            snapshot_f = self.tracker.raw_uint32()

        # --- independent reconstruction, after the fact, from cloned inputs.
        # The first real step is allowed to search all variants, so the adopted
        # variant is decided by a transition this run actually performed rather
        # than only by the throwaway probe.
        still_searching = self.config.validation.search_update_variants and not self._variant_locked
        reconstruction = oi.reconstruct_update(
            optimizer_name=self.config.optimizer,
            weight_before=weight_before,
            weight_after=weight_after,
            gradient=gradient_snapshot,
            state_before=state_before,
            lr=lr_used,
            config=self.config,
            active_variant=None if still_searching else self._active_variant,
            search=still_searching,
            tolerance_ulp=self.config.validation.reconstruction_ulp_tolerance,
        )
        if self.config.validation.search_update_variants and not self._variant_locked and reconstruction.matched:
            self._variant_locked = True
            self._active_variant = reconstruction.variant
            self.recorder.emit(
                step,
                "OPTIMIZER_STATE_READ",
                f"adopted update variant {reconstruction.variant!r}, measured from this "
                f"run's own transition ({reconstruction.ulp_error} ULP error)",
            )

        if not reconstruction.matched:
            self.recorder.emit(
                step,
                "VALIDATION_FAILED",
                f"optimizer reconstruction: {reconstruction.status} "
                f"({reconstruction.ulp_error} ULP, variant {reconstruction.variant})",
            )
        self.recorder.ledger.add_reconstruction(vd.validate_reconstruction(step, reconstruction))

        # --- bit / ULP / rounding analysis of the real transition.
        # "Attempted" is the float64 mathematical displacement the optimizer was
        # trying to apply; the float32 realised displacement cannot show a
        # sub-ULP move at all, so using it here would hide exactly the case this
        # instrument exists to catch.
        attempted_delta = reconstruction.fp64_reference_after - float(weight_before.item())
        movement = na.analyse_movement(
            weight_before, weight_after, attempted_delta=attempted_delta
        )
        transition = f32.bit_transition(movement.before_uint32, movement.after_uint32)
        rounding = na.analyse_rounding(reconstruction.fp64_reference_after, weight_after)
        after_bits = f32.inspect_float32(weight_after)
        neighbourhood = f32.ulp_neighbourhood(movement.before)

        if movement.bits_identical:
            self.recorder.emit(
                step,
                "BIT_PATTERN_UNCHANGED",
                "the stored float32 value is identical before and after the step",
            )
            if movement.attempted_but_not_representable:
                self.recorder.emit(
                    step,
                    "UPDATE_ATTEMPTED_BELOW_FP32_RESOLUTION",
                    f"the optimizer attempted a displacement of {attempted_delta!r} "
                    f"while one ULP at this magnitude is {neighbourhood.ulp_size!r}",
                )
        else:
            self.recorder.emit(
                step,
                "BIT_PATTERN_CHANGED",
                f"{transition.changed_bit_count} bits changed "
                f"({movement.ulp_distance} ULP)",
            )
            self.recorder.emit(step, "PARAMETER_CHANGED", f"delta = {movement.delta!r}")

        if rounding.absolute_residual != 0.0:
            self.recorder.emit(
                step,
                "FP32_ROUNDING_OBSERVED",
                f"rounding residual {rounding.absolute_residual!r} "
                f"({rounding.residual_in_ulp!r} ULP)",
            )

        # --- the independent float64 reference track
        fp64_value: float | None = None
        fp64_loss: float | None = None
        if self.fp64_track is not None and x is not None and target is not None:
            self.fp64_track.handle.set_lr(lr_used)
            fp64_value, fp64_loss = self.fp64_track.step(x, target, batches=objective_batches)

        # --- finiteness is decided before the record is sealed, so the
        # NONFINITE_DETECTED event is part of the record it belongs to
        finite_now = bool(
            math.isfinite(movement.after)
            and math.isfinite(float(grad_after_backward))
            and forward_finite
            and math.isfinite(sum(accumulated_losses))
            and all(entry.is_finite for entry in state_reading_after.entries)
        )
        if not finite_now:
            self.recorder.emit(
                step,
                "NONFINITE_DETECTED",
                "a tracked quantity became NaN or infinite during this step: "
                f"w = {movement.after!r}, gradient = {float(grad_after_backward)!r}, "
                f"loss = {sum(accumulated_losses)!r}; non-finite optimizer entries = "
                f"{[entry.key for entry in state_reading_after.entries if not entry.is_finite]!r}",
            )

        # --- assemble the immutable record
        record = self._build_record(
            step=step,
            epoch=epoch,
            batch_index=batch_index,
            global_batch=global_batch,
            x=x,
            target=target,
            forward=forward_record,
            accumulated_losses=tuple(accumulated_losses),
            zero_grad_state=(grad_before_zero, grad_after_zero, grad_before_backward, grad_after_backward),
            gradient_meta=gradient_meta,
            gradient_snapshot=gradient_snapshot,
            gradient_validation=gradient_validation,
            landscape_data=landscape_data,
            lr_used=lr_used,
            state_before=state_before,
            state_reading_before=state_reading_before,
            state_reading_after=state_reading_after,
            weight_before=weight_before,
            weight_after=weight_after,
            snapshot_a=snapshot_a,
            snapshot_b=snapshot_b,
            snapshot_c=snapshot_c,
            snapshot_d=snapshot_d,
            snapshot_e=snapshot_e,
            snapshot_f=snapshot_f,
            movement=movement,
            transition=transition,
            rounding=rounding,
            after_bits=after_bits,
            neighbourhood=neighbourhood,
            reconstruction=reconstruction,
            fp64_value=fp64_value,
            fp64_loss=fp64_loss,
            hash_before=hash_before,
            finite=finite_now,
        )
        stored = self.recorder.record(record)
        self.analyzer.observe(stored)
        self.recorder.emit(
            step,
            "STABILITY_METRIC_UPDATED",
            f"mean |update| over window = {self.analyzer.metrics[-1].rolling_mean_absolute_update!r}",
        )

        if not stored.finite and self.config.stop_on_nonfinite:
            self.aborted = True
            self.abort_reason = (
                f"non-finite state at step {step}: the run was stopped rather than "
                "continuing with meaningless values"
            )
            self.recorder.aborted_reason = self.abort_reason
            self.recorder.emit(step, "RUN_ABORTED", self.abort_reason)

        self.step_index += 1
        self._live_step_index += 1
        if self.step_index >= self.config.steps and not self.aborted:
            self.recorder.emit(
                self.step_index, "RUN_COMPLETE", f"{self.config.steps} real steps executed"
            )
        return stored

    # -- helpers used by step() --------------------------------------------
    def _apply_schedule(self, step: int) -> None:
        if self.config.lr_schedule == "exponential":
            self.handle.set_lr(self.config.lr * (self.config.lr_final_factor**step))
        elif self.config.lr_schedule == "cosine" and self._lr_scheduler is None:
            # Closed-form fallback when PyTorch scheduler is unavailable
            eta_min = self.config.lr * self.config.lr_final_factor
            T_max = max(1, self.config.steps)
            progress = min(1.0, max(0.0, step / T_max))
            lr = eta_min + 0.5 * (self.config.lr - eta_min) * (1.0 + math.cos(math.pi * progress))
            self.handle.set_lr(lr)
        # 'cosine' with active scheduler is driven by self._lr_scheduler.step()
        # 'constant' needs nothing

    def _probe_landscape(
        self, step: int, snapshot: int, hash_before: str,
        *, batches: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        settings = self.config.landscape
        due = settings.enabled and (step % settings.every == 0 or step == self.config.steps - 1)
        if not due:
            return {}
        centre = self.tracker.value()
        dtype = torch.float64 if settings.dtype == "float64" else torch.float32
        objective_batches = tuple(batches) if batches is not None else (self.dataset.batch(self._live_step_index)[:2],)
        slice_data = na.probe_landscape(
            lambda w: sum(self.model.evaluate_loss(x, target, w, dtype=dtype) for x, target in objective_batches),
            center=centre,
            radius=settings.radius,
            points=settings.points,
            step=step,
            dtype=str(dtype).replace("torch.", ""),
            center_uint32=snapshot,
            evaluate_type=settings.dtype,
            restoration_verified=True,  # replaced below by the real check
        )
        hash_after = self.tracker.content_hash()
        restoration_verified = hash_after == hash_before
        data = slice_data.to_dict()
        data["restoration_verified"] = restoration_verified
        data["parameter_hash_before"] = hash_before
        data["parameter_hash_after"] = hash_after
        self.recorder.add_landscape(data)
        self.recorder.emit(
            step,
            "LANDSCAPE_PROBED",
            f"one-dimensional slice over +/-{settings.radius} in {settings.dtype}",
        )
        if not restoration_verified:
            raise NumericalConsistencyError(
                f"step {step}: the landscape probe changed the tracked parameter "
                f"({hash_before} -> {hash_after})"
            )
        return data

    def _validate_gradient(
        self,
        *,
        step: int,
        x: torch.Tensor,
        target: torch.Tensor,
        w_value: float,
        autograd_gradient: float,
        gradient_meta: Mapping[str, Any],
        batches: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> vd.GradientValidation | None:
        settings = self.config.validation
        if not settings.enabled or step % settings.every != 0:
            return None
        validation = vd.validate_gradient(
            step=step,
            model=self.model,
            x=x,
            target=target,
            w_value=w_value,
            autograd_gradient=autograd_gradient,
            autograd_uint32=gradient_meta.get("raw_uint32"),
            config=settings,
            batches=batches,
        )
        self.recorder.ledger.add_gradient(validation)
        if not validation.passed:
            self.recorder.emit(
                step,
                "VALIDATION_FAILED",
                f"gradient validation: autograd={validation.autograd_status}, "
                f"finite difference={validation.finite_difference_status}",
            )
        return validation

    def _build_record(self, **kw: Any) -> ParameterStepRecord:
        """Assemble the record.  Long, but every field is a direct observation."""
        step = kw["step"]
        forward = kw["forward"]
        movement: na.MovementAnalysis = kw["movement"]
        transition: f32.BitTransition = kw["transition"]
        rounding: na.RoundingAnalysis = kw["rounding"]
        after_bits: f32.Float32Bits = kw["after_bits"]
        neighbourhood: f32.ULPNeighbourhood = kw["neighbourhood"]
        reconstruction: oi.ReconstructionResult = kw["reconstruction"]
        state_reading_after = kw["state_reading_after"]
        gradient_meta = kw["gradient_meta"]
        gradient: float = float(kw["zero_grad_state"][3])
        landscape: Mapping[str, Any] = kw["landscape_data"]
        validation: vd.GradientValidation | None = kw["gradient_validation"]
        x: torch.Tensor = kw["x"]
        target: torch.Tensor = kw["target"]
        fp64_value: float | None = kw["fp64_value"]

        bias_corrected1 = _bias_corrected(
            state_reading_after.value_of("exp_avg"),
            state_reading_after.step_value,
            self.config.betas[0] if self.config.optimizer != "sgd" else 0.0,
        )
        bias_corrected2 = _bias_corrected(
            state_reading_after.value_of("exp_avg_sq"),
            state_reading_after.step_value,
            self.config.betas[1] if self.config.optimizer != "sgd" else 0.0,
        )
        fp64_comparison = (
            na.compare_fp64_reference(step, kw["weight_after"].item(), fp64_value)
            if fp64_value is not None
            else None
        )
        finite = bool(kw["finite"])
        return ParameterStepRecord(
            step=step,
            epoch=int(kw["epoch"]),
            batch_index=int(kw["batch_index"]),
            global_batch=int(kw["global_batch"]),
            input_value=float(x.detach().reshape(-1)[0].item()),
            target_value=float(target.detach().reshape(-1)[0].item()),
            prediction=forward.prediction,
            loss=sum(kw["accumulated_losses"]),
            loss_dtype=str(forward.stage("loss").dtype),
            loss_grad_fn=forward.loss_grad_fn,
            stages=forward.stages,
            zero_grad_set_to_none=self.config.zero_grad_set_to_none,
            accumulation_micro_batches=self.config.accumulation_micro_batches,
            accumulated_losses=kw["accumulated_losses"],
            grad_before_zero_grad=kw["zero_grad_state"][0],
            grad_after_zero_grad=kw["zero_grad_state"][1],
            grad_before_backward=kw["zero_grad_state"][2],
            grad_after_backward=kw["zero_grad_state"][3],
            gradient=gradient,
            gradient_uint32=gradient_meta.get("raw_uint32"),
            gradient_dtype=str(gradient_meta.get("dtype")),
            gradient_shape=tuple(gradient_meta.get("shape", ())),
            gradient_sign=na.sign_of(gradient),
            gradient_magnitude=abs(gradient),
            gradient_square=gradient * gradient,
            gradient_finite=bool(gradient_meta.get("finite", False)),
            parameter_before=movement.before,
            parameter_after=movement.after,
            parameter_before_uint32=movement.before_uint32,
            parameter_after_uint32=movement.after_uint32,
            parameter_before_hex=f"0x{movement.before_uint32:08X}",
            parameter_hex=after_bits.hex,
            parameter_bits=after_bits.binary,
            parameter_exponent_bits=after_bits.exponent_bits,
            parameter_fraction_bits=after_bits.fraction_bits_int,
            parameter_sign_bit=after_bits.sign_bit,
            actual_delta=movement.delta,
            absolute_delta=movement.absolute_delta,
            relative_delta=movement.relative_delta,
            direction=movement.direction,
            xor_mask=transition.xor_mask,
            changed_bit_count=transition.changed_bit_count,
            changed_bit_positions=transition.changed_bit_positions,
            sign_changed=transition.sign_changed,
            exponent_changed=transition.exponent_changed,
            fraction_changed=transition.fraction_changed,
            ulp_previous_distance=neighbourhood.distance_to_previous,
            ulp_next_distance=neighbourhood.distance_to_next,
            ulp_size=neighbourhood.ulp_size,
            ulp_relative_movement=movement.ulp_movement,
            ulp_distance_count=movement.ulp_distance,
            ulp_distance_same_binade=movement.ulp_distance_same_binade,
            stored_fp32_unchanged=movement.bits_identical,
            update_attempted_but_storage_frozen=movement.attempted_but_not_representable,
            snapshot_a_uint32=int(kw["snapshot_a"]),
            snapshot_b_uint32=int(kw["snapshot_b"]),
            snapshot_c_uint32=int(kw["snapshot_c"]),
            snapshot_d_uint32=int(kw["snapshot_d"]),
            snapshot_e_uint32=int(kw["snapshot_e"]),
            snapshot_f_uint32=kw["snapshot_f"],
            optimizer_name=self.config.optimizer,
            learning_rate=float(kw["lr_used"]),
            optimizer_step_before=kw["state_reading_before"].step_value,
            optimizer_step_after=state_reading_after.step_value,
            optimizer_state_present_before=kw["state_reading_before"].present,
            optimizer_state_keys_before=kw["state_reading_before"].keys,
            optimizer_state_keys_after=state_reading_after.keys,
            moment1_before=kw["state_reading_before"].value_of("exp_avg"),
            moment2_before=kw["state_reading_before"].value_of("exp_avg_sq"),
            moment1=state_reading_after.value_of("exp_avg"),
            moment2=state_reading_after.value_of("exp_avg_sq"),
            bias_corrected_moment1=bias_corrected1,
            bias_corrected_moment2=bias_corrected2,
            optimizer_total_update=reconstruction.total_update,
            optimizer_attempted_update=(
                reconstruction.fp64_reference_after - movement.before
            ),
            optimizer_adaptive_update=reconstruction.adaptive_update,
            optimizer_weight_decay_update=reconstruction.weight_decay_update,
            reconstruction_variant=reconstruction.variant,
            reconstruction_status=reconstruction.status,
            reconstruction_bit_exact=reconstruction.bit_exact,
            reconstruction_ulp_error=reconstruction.ulp_error,
            reconstruction_absolute_error=reconstruction.absolute_error,
            reconstruction_relative_error=reconstruction.relative_error,
            reconstructed_after=reconstruction.reconstructed_after,
            reconstructed_uint32=reconstruction.reconstructed_uint32,
            fp64_reconstructed_after=reconstruction.fp64_reference_after,
            fp32_candidate=rounding.candidate_as_float32,
            fp32_rounding_error_abs=rounding.absolute_residual,
            fp32_rounding_error_rel=rounding.relative_residual,
            fp32_rounding_sign=rounding.residual_sign,
            fp32_rounding_ulp=rounding.residual_in_ulp,
            fp32_candidate_rounds_to_stored=rounding.candidate_rounds_to_stored,
            fp64_track_value=fp64_value,
            fp64_track_loss=kw["fp64_loss"],
            fp64_vs_fp32_abs=fp64_comparison.absolute_difference if fp64_comparison else None,
            fp64_vs_fp32_ulp=(
                fp64_comparison.difference_in_ulp_of_fp32 if fp64_comparison else None
            ),
            gradient_analytical=validation.analytical_value if validation else None,
            gradient_finite_difference=(
                validation.finite_difference_value if validation else None
            ),
            gradient_finite_difference_h=(
                validation.finite_difference_half_width if validation else None
            ),
            gradient_fp64_autograd=(
                validation.autograd_fp64_value if validation else None
            ),
            gradient_vs_analytical_rel=(
                validation.autograd_vs_analytical_rel if validation else None
            ),
            gradient_vs_analytical_abs=(
                validation.autograd_vs_analytical_abs if validation else None
            ),
            gradient_vs_finite_difference_rel=(
                validation.autograd_vs_finite_difference_rel if validation else None
            ),
            gradient_vs_finite_difference_abs=(
                validation.autograd_vs_finite_difference_abs if validation else None
            ),
            gradient_tolerance_abs=(
                validation.gradient_tolerance_abs if validation else None
            ),
            finite_difference_tolerance_abs=(
                validation.finite_difference_tolerance_abs if validation else None
            ),
            gradient_validation_status=(
                "PASS" if validation and validation.passed else ("FAIL" if validation else "not run")
            ),
            landscape_computed=bool(landscape),
            landscape_radius=float(self.config.landscape.radius),
            landscape_points=int(self.config.landscape.points),
            landscape_loss_at_centre=landscape.get("loss_at_center_remeasured"),
            landscape_minimum_loss=landscape.get("minimum_loss"),
            landscape_first_derivative=landscape.get("central_first_difference"),
            landscape_second_derivative=landscape.get("central_second_difference"),
            landscape_restoration_verified=bool(landscape.get("restoration_verified", False)),
            finite=finite,
            nonfinite_detected=not finite,
            parameter_hash_before=kw["hash_before"],
            parameter_hash_after=self.tracker.content_hash(),
            events=self.recorder.events.codes_for_step(step),
            previous_hash="",
            record_hash="",
        )

    # -- batch execution ----------------------------------------------------
    def run(
        self,
        max_steps: int | None = None,
        *,
        on_step: StepCallback | None = None,
        should_continue: Callable[[], bool] | None = None,
        speed: float | None = None,
    ) -> list[ParameterStepRecord]:
        """Run real steps until the target, the abort, or the callback says stop."""
        if max_steps is not None and (
            not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 0
        ):
            raise ValueError("max_steps must be a non-negative integer or None")
        if speed is not None and (
            isinstance(speed, bool) or not math.isfinite(speed) or speed <= 0.0
        ):
            raise ValueError("speed must be positive and finite or None")
        limit = self.config.steps if max_steps is None else max_steps
        produced: list[ParameterStepRecord] = []
        interval = (1.0 / speed) if speed else 0.0
        while len(produced) < limit and not self.finished:
            if should_continue is not None and not should_continue():
                break
            record = self.step()
            if record is None:
                break
            produced.append(record)
            if on_step is not None:
                on_step(record)
            if interval:
                time.sleep(interval)
        return produced

    # -- reporting ----------------------------------------------------------
    def status_line(self) -> str:
        if not self.recorder.records:
            return "[engine] no steps recorded yet"
        last = self.recorder.records[-1]
        return (
            f"step {last.step:>4}/{self.config.steps - 1} | "
            f"w = {last.parameter_after!r} ({last.parameter_hex}) | "
            f"loss = {last.loss!r} | g = {last.gradient!r} | "
            f"delta = {last.actual_delta!r} ({last.ulp_distance_count:+d} ULP) | "
            f"lr = {last.learning_rate!r} | recon {last.reconstruction_status}"
        )

    def final_report(self) -> dict[str, Any]:
        self.recorder.assert_chain()
        from .recorder import build_final_report  # noqa: PLC0415 - import cycle

        return build_final_report(self.recorder, stability=self.analyzer.report())


# --------------------------------------------------------------------------- #
# module helpers
# --------------------------------------------------------------------------- #


def _clone_optimizer_state(
    optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter
) -> dict[str, Any]:
    """A detached, bit-exact copy of the optimizer state for one parameter."""
    state = optimizer.state.get(parameter, None)
    if not state:
        return {}
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = value
    return cloned


def _bias_corrected(
    moment: float | None,
    step_value: int | None,
    beta: float,
) -> float | None:
    """``moment / (1 - beta**t)`` using the optimizer's own step counter.

    ``t`` is read from the optimizer state rather than assumed, because the
    counter is the one the installed implementation itself uses.  SGD has no bias
    correction, so it returns ``None`` rather than a misleading number.
    """
    if moment is None or not step_value:
        return None
    correction = 1.0 - beta**int(step_value)
    if correction == 0.0:
        return None
    return moment / correction


__all__ = [
    "RunState",
    "ExperimentController",
    "FP64Track",
    "TrainingEngine",
]
