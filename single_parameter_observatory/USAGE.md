# Usage guide

Everything here runs the *same* real experiment; the subcommands differ only in
how they show it. Install first:

```bash
pip install -e .            # from a checkout (developers)
pip install .               # or a plain install
# optional export extras:
pip install -e .[export]    # pandas + pyarrow, for trajectory.parquet
```

After installation the CLI is on your path as `spobs` (and
`python -m single_parameter_observatory.main` works from anywhere).

## 1. The commands

| command | what it does |
|---|---|
| `spobs` or `spobs run` | a real AdamW run + the full numerical report |
| `spobs inspect` | environment audit, parameter identity, optimizer source audit |
| `spobs trace --count 5` | the detailed per-step forensic trace |
| `spobs live` | the live dashboard (real-time, keyboard controlled) |
| `spobs controls` | headless control session: play / pause / step / n / reset / restart |
| `spobs presets` | list the training presets |
| `spobs selftest` | the current automated test suite, including repair regressions |
| `python run_checks.py` | all 11 end-to-end checks, log written to `checks_output.txt` |

## 2. The knobs that matter

```bash
spobs run --preset fp32_resolution     # the headline phenomenon
spobs run --preset diverging           # exits 3: the run genuinely diverged
spobs run --preset adamw --steps 500 --lr 0.05 --seed 7
spobs run --preset sgd_momentum --no-fp64-reference      # halve the work
spobs run --preset adamw --accumulate 4                  # gradient accumulation
spobs run --export runs --plot figures                   # every artefact
```

* `--preset` picks a real training configuration (see `spobs presets`);
  `--steps`, `--lr`, `--optimizer`, `--weight-decay`, `--seed` override it.
* `--export DIR` writes all twelve artefacts (CSV/JSON/NPZ/Parquet, bit
  history, optimizer-state history, stability, landscape, events, validation,
  provenance, summary). Add `--no-timestamp` to write into DIR directly.
* `--plot DIR` renders the overview and bit-history figures.
* `--set-to-zero` switches `zero_grad(set_to_none=False)` so the gradient
  lifecycle records a zeroed tensor instead of a missing one.

## 3. The live view

```bash
spobs live                              # window, if an interactive backend exists
spobs live --headless                   # renders frames to PNGs instead
spobs live --fps 144 --static-hz 2      # more frame rate / fresher text
spobs live --headless --save-frames d --frame-every 10
```

Keyboard (interactive backends): `space` play/pause - `s` single step - `n`
run 10 - `r` reset live state - `R` restart experiment - `+`/`-` step rate -
`q` quit.

Three independent cadences drive the three render layers (text / series /
marker): `--static-hz`, `--data-hz`, `--hud-hz`. Lower `--static-hz` to trade
state-card freshness for frame rate. The marker that slides between recorded
steps is an interpolation and the figure says so.

`--hud-hz` refreshes the state card and FPS readout independently of the slower
static layer. Saved PNGs include every visible layer. Closing the view or reaching
`--max-seconds` stops its worker before producing the final snapshot or export.
If an interactive backend is unavailable, frames are saved using Agg.

## 4. Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | an `ObservatoryError` (bad configuration, missing dependency, ...) |
| 2 | the history hash chain failed to verify - the record cannot be trusted |
| 3 | the run reached a non-finite state (`run`/`trace`; a *result*, not a defect) |
| 4 | a validation failure was recorded (a *defect*) |

`live` is a view: it exits 0 when the dashboard ran, printing
`NON-FINITE DETECTED` / `RUN ABORTED` if it was watching a diverged run. A
broken chain or validation failure still fails a view.

## 5. Reading the artefacts

* `summary.json` - the final report, validation ledger, event counts.
* `trajectory.csv` - one row per step, every field, unrounded (`repr` floats).
* `trajectory.npz` / `trajectory.parquet` - the numeric columns as arrays /
  a dataframe.
* `bit_history.csv` - before/after words, XOR mask, changed bit positions.
* `optimizer_state_history.csv` - the optimizer's own state per step.
* `validation.json` - the ledger plus per-step gradient/reconstruction checks.
* `provenance.json` - environment, optimizer source audit, measured update
  semantics, parameter identity, dataset description.
* `events.csv` - the append-only event log.
* `landscape.json` - the one-dimensional diagnostic slices (with disclaimers).

In JSON, non-finite values appear as their `repr` strings (`"inf"`, `"nan"`)
so a diverged run still produces strict, parseable JSON.

NPZ exports use companion `<column>__present` arrays for nullable columns. Integers
outside int64 are decimal Unicode strings so loading with `allow_pickle=False`
remains lossless. Parquet also stores oversized integer columns as decimal text.
Missing pandas or a Parquet engine skips only Parquet; mandatory export errors
fail the command. Exporters verify the history chain before writing and replace
each file atomically. Reusing a directory removes stale optional artifacts.

Timestamped exports create a fresh directory on each invocation. Use
`--no-timestamp` when intentionally replacing a previous export in the same directory.

Version 1.1.1 hashes include each record's `previous_hash`. The digest values
therefore differ from 1.1.0 even when the numerical trajectory is identical.

## 6. The library surface

```python
from single_parameter_observatory.config import ExperimentConfig, PRESETS
from single_parameter_observatory.training_engine import TrainingEngine

engine = TrainingEngine(PRESETS["fp32_resolution"])
for record in engine.run():
    print(record.step, record.parameter_after, record.parameter_hex,
          record.update_attempted_but_storage_frozen)

report = engine.final_report()          # asserts the hash chain first
engine.recorder.verify_chain()          # (ok, first_bad_step)
```

Every number on every record was read out of a real tensor during a real run.
See [API.md](API.md) for the full surface and [ARCHITECTURE.md](ARCHITECTURE.md)
for the module map.
