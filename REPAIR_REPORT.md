# Single Parameter Observatory 1.1.1 repair report

Date: 2026-10-23. Scope: the active Python package, command line, controller,
renderer, configuration, exports, packaging, CI and current verification tools.
The historical audit, PDF and `evidence/original` are retained as prior evidence.

## Current verification

The final source and installed package passed verification. Ninety package
regressions were added to the original 128 tests, plus three delivery-tool tests.
Logs from intermediate failures are retained and are explicitly superseded by
the final evidence below.

| Verification | Observed result | Evidence under `evidence/repair_20261023/` |
| --- | --- | --- |
| Package tests from final source | 218 passed; 0 failures/errors/skips, 51.700 s | `execution_checks.txt`, check 3 |
| Installed-wheel selftest | 218 passed; 0 failures/errors/skips, 20.304 s | `unit_tests.txt` |
| Delivery tooling | 3 passed | `delivery_tests.txt` |
| End-to-end execution | All 11 checks passed, 73.5 s | `execution_checks.txt` |
| Static checks | Ruff passed; mypy passed across 34 source files | `ruff.txt`, `mypy.txt` |
| Full preset sweep | 11 presets, 1,863 real steps, zero validation failures, every chain verified | `preset_sweep_final/summary.json` |
| Saved exports | 132 artifacts independently reread, zero size/hash mismatches | `preset_sweep_final/summary.json` |
| Figures and dashboard frames | 22 overview/bit PNGs and 4 GUI PNGs produced; saved representatives visually reviewed | `preset_sweep_final/`, `visual_review_v3/sha256.json` |
| Distribution | Version 1.1.1 sdist and wheel built; wheel installed outside the checkout package path | `build.txt`, `installed_identity.txt` |
| Installed CLI and exports | Console JSON parsed; a real 5-step run exported all formats and both figures from an external working directory | `installed_inspect.json`, `installed_smoke.txt`, `installed_export/` |

The final sweep's 20 tracked source/script/metadata hashes were unchanged during
execution and matched disk afterward. The installed wheel's 34 Python members
were also compared with archive bytes, with zero mismatches. The installation
used a separate environment for package code and reused the already verified
runtime dependency directory; dependencies were not independently re-resolved.

The isolated final dashboard observation drew 167 frames in 5.1616 s (**32.35
FPS**), with 86 real steps and 11 full canvas draws. The integrated execution
check observed **10.7 FPS** over 4.47 s. These are separate host observations,
not a controlled performance comparison or a frame-rate guarantee. See
`dashboard_performance_final.json` and check 11 in `execution_checks.txt`.

Environment: Windows, Python 3.12.15, torch 2.14.1+cpu, NumPy 2.5.4,
Matplotlib 3.11.2, pandas 3.1.0, PyArrow 26.0.0. The project `.venv` provides
the verified interpreter; the machine's default Python 3.15 is not that environment.
See `evidence/repair_20261023/environment.json`.

## Correctness repairs

### Data, gradients and numerical references

- Configured batch sizes are honored, final remainder batches are retained and
  shuffled epochs visit every sample. Alternating streams reach both input signs.
- Gradient accumulation records the sum of the actual microbatch mean losses.
  Gradient validation and landscape slices evaluate that same summed objective.
- The independent float64 track receives the same batches and actual learning
  rates. It intentionally retains float64 configuration constants; independent
  validation of the primary model instead uses its observed float32 buffers.
- Reset restarts dataset position, optimizer/reference state, learning-rate
  progression and stability analysis while retaining history. Restart clears
  history and reconstruction selection without duplicate initialization events.
- NaN transitions, infinite candidates and continued non-finite runs are recorded
  without diagnostic crashes. Infinite optimizer state is detected even when the
  stored parameter, gradient and loss remain finite.
- FP64 update candidates respect maximize and AMSGrad history. ULP calculations
  handle the minimum normal and maximum finite values, and nextafter walks stop
  at infinity. Parameter replacement is detected by identity.
- Engine construction does not reseed the host application's global torch RNG.
  Provenance callers receive detached copies.
- Environment auditing reports actual MPS hardware availability rather than
  treating the presence of PyTorch's MPS namespace as available hardware.

### Controls, rendering and command line

- Queued single steps are preserved; run-N pauses after exactly N steps. Pause
  waits for the active step and callback. Workers remain usable after completion
  and restart; unexpected worker errors reach the calling thread.
- Error polling uses a separate short lock so renderer polls cannot block behind
  a training step. Invalid control speed text/values produce handled `ConfigError`.
- Static pixels are rendered before caching, recorded lines survive motion-only
  frames, and saved PNGs include animated artists. Resize/renderer changes clear
  caches. HUD refresh, restart and non-finite marker behavior are covered.
- Padded viewports avoid rescaling every sample, and the HUD has its own cached
  layer. Divergence overviews use separated rows, bounded logarithmic tick labels
  and a wrapped transition card; saved images were reviewed.
- GUI show/events/close handling is explicit. Matplotlib Save/Home shortcuts no
  longer conflict with single-step/reset controls. Shutdown stops the writer
  before final snapshots, reports and exports.
- CLI zero values are validated rather than silently ignored. `--seed` reaches
  dataset generation. Interactive play returns to the prompt so pause can be read.
- `inspect --json` returns strict JSON; timestamped exports use fresh directories;
  requested plot failures return failure. Empty control sessions report that no
  steps executed. Validation defects take precedence over observed divergence.
- Invalid trace ranges and timing values are rejected. Selftest cannot succeed
  with empty discovery and reports test outcome categories accurately.

### Recorded evidence and exports

- Each record hash includes `previous_hash`. Duplicate/skipped steps and altered
  sealed records are rejected; record schema validation also detects type changes
  with identical JSON spelling, such as numerical NaN replaced by string `"nan"`.
- Caller-owned sequences/shapes are detached. Live tensors are rejected in
  records, and landscape observations are detached on entry and retrieval.
- Exporters verify the record chain before writing. Files are written through
  temporary files and replaced only after successful serialization.
- Missing pandas/Parquet engines skip only the optional Parquet artifact.
  Mandatory NPZ failures and unrelated I/O errors propagate.
- NPZ retains nullable columns with `__present` masks and oversized integers as
  exact decimal text, without pickle. Parquet preserves oversized integer counts.
- Requested NPZ filenames are honored. Reused export directories remove stale
  managed artifacts that the current run did not produce.
- Reports include infinite observations in extrema/means, avoid overflow in
  finite summaries and preserve an explicitly supplied empty report.

### Configuration, release checks and distribution

- Invalid runtime types, fractional/boolean counts, bad seeds, invalid schedules,
  empty finite-difference sweeps and unrepresentable optimizer scalars fail with
  `ConfigError` before training. Caller-owned beta/decade lists become tuples.
- Full invariant snapshots are required. Setting the retained compatibility field
  `record_full_state_snapshots=False` now raises instead of being silently ignored.
- Package and build metadata share version 1.1.1. SPDX licensing replaces
  deprecated metadata. `python -m single_parameter_observatory` is supported,
  guides are shipped with the package, and generated environments/caches are ignored.
- CI includes Windows 3.12 in addition to its Linux Python matrix and exercises
  the delivery tool. Actual local qualification was Windows 3.12; remote CI was
  not executed during this repair.
- Release checks cannot pass zero discovered tests, missing/empty exports or an
  incomplete export inventory. Every frame check uses a fresh session directory.
  FPS is measured separately from renderer correctness and does not infer a
  broken cache merely from a slow host.
- `scripts/run_preset_sweep.py` resolves the current checkout portably, preserves
  past evidence, verifies saved export content and hashes, and rejects a source
  revision that changes during the sweep.
- `scripts/package_delivery.py` creates a source/evidence ZIP without destructive
  staging. Its manifest verifies every included member's size and SHA-256 against
  disk, detects changed source inventories, rejects symlink roots and keeps ZIP
  output outside included source/evidence trees.

## Compatibility and qualification boundaries

Version 1.1.1 record digests differ from 1.1.0 because prefix hashes now participate
in the digest. Batch sizes greater than one, accumulation, alternating streams,
scheduled FP64 runs and live reset now follow their declared behavior, so those
trajectories can differ from the buggy behavior.

Native Windows GUI lifecycle and keys received mocked event-path coverage;
actual saved Agg figures and frame pixels were inspected. No manual native-window
acceptance is claimed. CPU execution was qualified; CUDA/MPS and other Python or
torch builds were not locally qualified. Float32 reproducibility across hardware
and builds remains outside the contract.

Atomic replacement applies to each export file. A whole multi-file export is not
one filesystem transaction. Optional dependencies remain optional. Historical
report recipes retain their original Linux/service/font/helper prerequisites;
see `scripts/README.md`. They are archived audit material, not current verification
commands, and were not used to regenerate historical claims.

## Reproduce the current checks

From the repository root in PowerShell, with the verified `.venv`:

```powershell
.venv\Scripts\ruff.exe check single_parameter_observatory scripts\run_preset_sweep.py scripts\package_delivery.py scripts\tests
.venv\Scripts\mypy.exe single_parameter_observatory --config-file pyproject.toml
.venv\Scripts\python.exe -m unittest discover -s scripts/tests -v
.venv\Scripts\python.exe single_parameter_observatory\run_checks.py --export-dir NEW_EXPORT_DIR --frames-dir NEW_FRAME_DIR --log NEW_CHECK_LOG
.venv\Scripts\python.exe scripts\run_preset_sweep.py --output NEW_SWEEP_DIR --figures --gui
.venv\Scripts\python.exe -m build
.venv\Scripts\python.exe scripts\package_delivery.py
```

Use a new or empty sweep directory. The package includes historical evidence as
well as the dated repair evidence; the current report identifies which logs
qualify the final bytes.
