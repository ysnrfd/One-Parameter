# API reference

Public surface, module by module. Everything not listed here is private by
convention (leading underscore) or an implementation detail that may change.
Record types are frozen dataclasses. `ParameterStepRecord.to_dict()` returns
JSON-safe values; use `json_safe(...)` when serializing other diagnostic dicts,
whose `to_dict()` methods may retain non-finite Python floats.

## `single_parameter_observatory`

| name | kind | description |
|---|---|---|
| `__version__` | `str` | package version |

## `errors` - the taxonomy

`ObservatoryError(RuntimeError)` is the base. Subclasses:
`ConfigError` (also `ValueError`), `TrackingInvalidError` (carries `.reason`
and `.details`), `BitDecodeError`, `NumericalConsistencyError`,
`ValidationError`, `HistoryCorruptionError`, `NonFiniteStateError`,
`ReconstructionMismatch`, `DependencyMissingError`.

## `config` - experiment configuration

| name | description |
|---|---|
| `ExperimentConfig` | frozen; the complete description of one run. `replace(**changes)` returns a new validated config |
| `DatasetConfig`, `ModelConfig`, `LandscapeConfig`, `ValidationConfig`, `StabilityCriteria` | the frozen sub-configs |
| `PRESETS` | `{name: ExperimentConfig}` - eleven real training presets |
| `PRESET_DESCRIPTIONS` | `{name: one-line description}` |
| `SUPPORTED_OPTIMIZERS` | `("sgd", "adam", "adamw")` |

Configuration rejects malformed runtime types, non-finite values and unsupported
float32 optimizer scalars with `ConfigError`. The retained
`record_full_state_snapshots` field must be `True`: snapshots are required to
verify the instrument. CLI `--seed` updates both experiment and dataset seeds;
library callers can configure `DatasetConfig.seed` explicitly.

## `model` - the substrate

| name | description |
|---|---|
| `ObservatoryModel(initial_value, model_config, *, dtype=torch.float32)` | the one-parameter module; rejects non-float32 dtypes |
| `.forward(x)`, `.loss(x, target)`, `.forward_with_trace(x, target)` | real graph paths; the trace records every intermediate |
| `.evaluate_loss(x, target, w_value, *, dtype)` | diagnostic path; builds a throwaway tensor, cannot touch `w` |
| `.analytic_gradient_fp64(x, target, w_value)` | hand-derived closed form, pure Python float64 |
| `.assert_single_float32_parameter() -> nn.Parameter` | raises `TrackingInvalidError` naming the violated invariant |
| `FP64ReferenceModel` | the independent float64 replica (a reference, not truth) |
| `build_dataset(config, model_config) -> Dataset` | deterministic stream; `Dataset.batch(step)` is a pure function of step |
| `describe_tensor(name, tensor, *, with_bits=False) -> TensorSnapshot` | snapshot without detaching from the graph |
| `FORWARD_STAGE_NAMES` | the recorded stage order |

## `fp32_inspector` - IEEE-754 forensics

| name | description |
|---|---|
| `raw_word_from_tensor(t) -> int` | the stored binary32 word, read from the tensor's own memory |
| `raw_word_from_python_float(v) -> int` / `python_float_from_raw_word(w)` | struct-based pair |
| `assert_python_and_tensor_agree(t) -> int` | raises if the two extraction paths disagree |
| `decode_float32_bits(raw) -> Float32Bits` | full forensic decode; verified by re-encoding |
| `encode_float32_bits(sign, exponent, fraction) -> int` | the inverse |
| `inspect_float32(value)` | accepts tensor / float / int; `prefer_tensor_path=True` re-reads through a real tensor |
| `ulp_neighbourhood(v) -> ULPNeighbourhood` | neighbours, both spacings, via `torch.nextafter` |
| `ulp_distance(before, after) -> int` | exact count of representable values between two floats |
| `ulp_distance_same_binade(before, after) -> int` | movement in ULPs of the *starting* value (the per-step metric) |
| `ulp_movement(before, after) -> float` | `|delta| / ULP(before)` |
| `bit_transition(a, b) -> BitTransition` | XOR mask, changed positions, field flags |
| `storage_evidence(t) -> StorageEvidence` | raw storage bytes, nbytes, method used |
| `summarise_bits_sequence(words)` | sign flips / exponent changes over a trajectory |

## `numerical_analysis` - the analysis layer

| name | description |
|---|---|
| `analyse_rounding(candidate_fp64, stored_tensor) -> RoundingAnalysis` | rounding residual, in ULP, with an exact-rational half-ULP tie test |
| `analyse_movement(before, after, attempted_delta=None) -> MovementAnalysis` | the authoritative stored delta plus the below-resolution flag |
| `central_difference(evaluate, w, h)` / `finite_difference_sweep(...)` | finite-difference machinery (with its own limitation note) |
| `probe_landscape(evaluate, center, radius, points, ...) -> LandscapeSlice` | the diagnostic slice; restoration is hash-verified by the engine |
| `compare_fp64_reference(step, fp32, fp64) -> FP64ReferenceComparison` | trajectory divergence as a measured quantity |
| `fp64_candidate_from_state(...)` | the textbook FP64 reconstruction (reference, not truth) |
| `absolute_error`, `relative_error`, `sign_of` | small helpers with explicit zero conventions |

## `tracker` - the tracked parameter

| name | description |
|---|---|
| `TrackedParameter(model)` | captures and re-verifies the single float32 scalar every step |
| `.verify(step=...)` | raises `TrackingInvalidError` with the exact violated invariant |
| `.info() -> TrackedParameterInfo` | the identity card |
| `.snapshot(label, step) -> ParameterSnapshot` | an immutable bit-level observation |
| `.content_hash()`, `.raw_uint32()`, `.value()`, `.gradient_value()` | readouts |
| `REQUIRED_INVARIANTS`, `TENSOR_VS_PARAMETER_NOTES` | the documented invariants |

## `optimizer_inspector` - audit and reconstruction

| name | description |
|---|---|
| `audit_environment() -> EnvironmentAudit` | interpreter, torch build, platform, backends |
| `audit_optimizer_source(name, ...) -> OptimizerSourceAudit` | what the installed source actually contains |
| `make_optimizer(config, parameter) -> OptimizerHandle` | real optimizer, kwargs filtered against the installed signature |
| `read_optimizer_state(optimizer, parameter) -> OptimizerStateReading` | decodes every state tensor, bit patterns included |
| `reconstruct_update(..., search=True, tolerance_ulp=0) -> ReconstructionResult` | independent reconstruction + variant search |
| `probe_optimizer_semantics(config) -> dict` | one isolated step that *measures* the installed update variant |

## `validation`

`validate_gradient(...)` (autograd vs closed form vs finite differences vs FP64
autograd), `validate_float32_bits(t)`, `validate_ulp(v)`,
`validate_reconstruction(step, result)`, `gradient_tolerance(reference, scale,
rtol)` (the documented absolute-plus-relative bound), and the
`ValidationLedger` run-level tally.

## `recorder` - the immutable history

| name | description |
|---|---|
| `ParameterStepRecord` | one frozen step; ~130 fields, raw and derived kept separate |
| `Recorder(config)` | `record(draft)` links into the SHA-256 chain; `verify_chain()` returns `(ok, first_bad_step)`; `assert_chain()` raises |
| `EventLog` / `EVENT_CODES` | the append-only event sequence |
| `build_final_report(recorder, stability=None)` / `render_final_report(report)` | the end-of-run report |
| `canonical_json(payload)` | deterministic form used for hashing |
| `json_safe(value)` | the sanitiser all JSON exports pass through |

## `convergence`

`ConvergenceAnalyzer(criteria)` - `observe(record)` folds a step,
`report()` classifies the observed regime with its rationale. `REGIMES` and
`STABILITY_DISCLAIMER` document what it refuses to say.

## `training_engine` - the engine and the controller

| name | description |
|---|---|
| `TrainingEngine(config)` | owns model, optimizer, instruments, history |
| `.step() -> ParameterStepRecord or None` | one real step through the A-F boundaries |
| `.run(max_steps=None, on_step=..., should_continue=..., speed=...)` | batch execution |
| `.provenance()` | environment + source audit + semantics probe, cached |
| `.reset_live_state()` / `.restart()` | labelled lifecycle operations |
| `.final_report()` | asserts the chain, then builds the report |
| `ExperimentController(engine)` | play / pause / single_step / run_n / reset / restart / set_speed; `run_blocking(commands)` for the terminal session |
| `RunState` | idle / running / paused / finished / aborted |

## `visualization` / `dashboard` / `exporters`

| name | description |
|---|---|
| `build_overview_figure(recorder, analyzer)` / `build_bit_figure(recorder)` | the static figures |
| `render_figures(recorder, directory, analyzer)` | write both |
| `render_state_card(record, config)` | the full textual card |
| `recorded_series(records, attribute)` / `draw_recorded_points(...)` | the single drawing path shared by both views |
| `configure_backend(interactive)` | explicit backend choice, reported |
| `LiveDashboard(engine, ...)` | the three-layer live view; `run(max_seconds=...)` returns frames drawn |
| `run_control_session(engine, commands, ...)` | headless control session |
| `export_all(recorder, directory, report=..., provenance=..., analyzer=...)` | all twelve artefacts; returns `{name: path or None}` |
| `EXPORT_FILES` | the artefact list, in production order |

## `main` - the CLI

`main(argv=None) -> int` is the entry point (also installed as `spobs`).
`build_parser()` exposes the full argument surface; `resolve_config(args)`
maps preset + overrides onto a validated `ExperimentConfig`. Exit codes:
0 success, 1 observatory error, 2 broken chain, 3 non-finite run, 4 validation
failure.
