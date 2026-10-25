# Architecture

This document is the map: what each module owns, what it may depend on, and how
one real training step flows through the system. The reader it assumes is an
engineer who has read the [README](README.md) and wants to change the code
without breaking the invariants the instrument exists to enforce.

## 1. The layering

The package is layered strictly downward. A module may import only from layers
below it (plus the two side-car modules `errors` and `fp32_inspector`, which
depend on nothing project-internal above them):

```
Layer 0   errors                 the exception taxonomy (no project imports)
Layer 1   fp32_inspector         IEEE-754 bits, ULP, storage evidence
Layer 1   config                 frozen ExperimentConfig + presets
Layer 2   model                  the one-parameter module, FP64 replica, dataset
Layer 2   numerical_analysis     rounding/movement analysis, FD, landscape, FP64 reference
Layer 3   tracker                tracked-parameter invariants and evidence
Layer 3   optimizer_inspector    environment/source audit, state decode, reconstruction
Layer 3   validation             gradient/bit/ULP/reconstruction validators
Layer 4   recorder               immutable records, event log, hash chain, final report
Layer 5   convergence            streaming stability analysis (reads records)
Layer 6   training_engine        one real step + the play/pause/step controller
Layer 7   visualization          figures and the textual state card
Layer 7   exporters              CSV / JSON / NPZ / Parquet artefacts
Layer 8   dashboard              the live view and the headless control session
Layer 9   main                   the CLI
Side-car  run_checks             the 11 end-to-end checks (imports everything)
```

Two deliberate import oddities exist and are load-bearing:

* `errors` is separate so every layer can share the taxonomy without a cycle.
* `convergence` imports `recorder.ParameterStepRecord` (layer 4) even though it
  sits above the engine conceptually; the dependency is on the *record type*
  only, never on engine state.
* `optimizer_inspector` imports `model` **inside**
  `probe_optimizer_semantics` (a local import), because `model` already imports
  `fp32_inspector` and the probe must not create an import cycle at module
  load time.

## 2. One real training step

`TrainingEngine.step()` is the entire experiment. Its labelled boundaries are
the audit trail; every one of them is a bit-pattern read, not a belief:

```
 A  read the parameter (raw uint32 + content hash)
    zero_grad(set_to_none=...)          -- gradient lifecycle observed
    for each accumulation micro-batch:
 B     real forward  (forward_with_trace records every intermediate)
       real backward  (loss.backward(); the ONLY gradient source)
 C  read the parameter -- must equal A, or NumericalConsistencyError
    diagnostics (never touch the live parameter):
      - landscape probe  (pure loss function; hash-checked before/after)
      - gradient validation (closed form + finite differences + FP64 autograd)
 D  read the parameter + clone optimizer state (reconstruction inputs)
    optimizer.step()                    -- the ONLY parameter writer
 E  read the parameter
    scheduler step (if any)
 F  read the parameter -- must equal E
    reconstruction, movement analysis, bit transition, rounding analysis
    record sealed, hash-chained, appended to history
```

The engine raises `NumericalConsistencyError` if the parameter moves anywhere
outside the `optimizer.step()` window (A==B==C and E==F). This is the
structural enforcement of "forward and backward do not update parameters".

## 3. The three guarantees and where they live

| Guarantee | Enforced by |
|---|---|
| No fabricated values | `model.describe_tensor` snapshots scalars only; `recorder.ParameterStepRecord` stores raw vs derived in separate fields; `tests/test_training.py::test_gradient_is_never_fabricated` greps `TrainingEngine.step` for `.grad =` |
| Immutable history | frozen dataclasses + SHA-256 chain in `recorder.Recorder.record/verify_chain`; tampering is detected and reported with the first bad step |
| No silent failure | `errors.py` taxonomy; every catch either re-raises or records a named failure in the ledger |

## 4. The reconstruction pipeline

`optimizer_inspector.reconstruct_update` never assumes the installed PyTorch's
update semantics. It:

1. reads the installed optimizer source (`audit_optimizer_source`) and records
   keyword-presence findings, not interpretations;
2. runs one isolated probe step on a throwaway parameter
   (`probe_optimizer_semantics`) and tries every documented update variant
   (moment update via `lerp_` vs `mul_+add_`, decay before/after the adaptive
   step, eps inside/outside the sqrt; buffer init and decay coupling for SGD);
3. adopts the variant that reproduces the real transition **bit for bit** (0
   ULP), and re-uses it for the rest of the run;
4. recomputes the update per step from cloned inputs and reports any mismatch
   as `RECONSTRUCTION MISMATCH` rather than passing it over.

The FP64 candidate (`numerical_analysis.fp64_candidate_from_state`) is a
separate, textbook-flavoured reconstruction used to quantify the gap between
float32 storage and the mathematical update - it is a reference, not truth.

## 5. The live view's three-layer render loop

`dashboard.LiveDashboard` splits a frame by cost, because text rasterisation
dominates:

* **static layer** (axes, causal chain, bit panel, state card) - re-rasterised
  at `static_hz`, then cached via `copy_from_bbox`;
* **data layer** (the four recorded series) - redrawn when a step arrives,
  throttled to `data_hz`;
* **motion layer** (marker + trail) - every frame.

Two invariants discovered the hard way (see AUDIT.md F-01):

1. After `Axes.clear()` the persistent artists are re-attached with
   `add_line`, which does **not** schedule an autoscale; `_rescale_series_axes`
   must call `relim()` + `autoscale_view()` or every series is clipped outside
   the unit-box viewport and the panel renders empty.
2. The marker's interpolated position is labelled as an interpolation on the
   face of the figure; only integer-step points are recorded values.

## 6. Export contract

`exporters.export_all` writes twelve artefacts from the recorder, never from
live tensors. Two rules matter to consumers:

* floats in JSON are real JSON numbers; CSV cells use `repr`, the shortest
  string that round-trips to the identical double;
* non-finite values are rendered as their `repr` strings (`"inf"`, `"nan"`) so
  a diverged run still exports as strict, parseable JSON
  (`recorder.json_safe`); `trajectory.npz` carries numeric columns as arrays,
  and `trajectory.parquet` is written when pandas is installed and reported as
  not-written when it is not.

## 7. What may never change

These are the project's identity, not implementation details:

* the only gradient source is `loss.backward()`; the only parameter writer is
  `optimizer.step()`;
* the primary tracked parameter is float32, scalar, shape `()`, one parameter
  in the whole module (`ObservatoryModel` raises otherwise);
* the regime classifier reports observed behaviour, never convergence;
* `float32(0.1)`'s exact expansion is `0.100000001490116119384765625` and its
  shortest round-trip is `0.1`, and the two are reported as different
  quantities.
