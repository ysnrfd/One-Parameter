# AUDIT.md - The 1.1.0 forensic audit

**Subject.** `single_parameter_observatory` 1.0.0, as uploaded (2.1 MB zip:
17 source modules, 9 test modules, shipped `checks_output.txt`,
`real_run.txt`, 8 rendered frames, one export bundle).

**Method.** (1) A complete read of all ~9,850 lines of source and all test
modules, against the project's own design rules. (2) A full real execution of
the uploaded code on a fresh machine: the 122-test suite, all 11 end-to-end
checks, every preset including `diverging` and `fp32_resolution`, the complete
export bundle, and a headless GUI session writing a multi-frame PNG dump.
(3) A vision-model (VLM) audit of every rendered frame - the 8 uploaded frames
plus 10 freshly rendered ones - cross-checked by objective pixel analysis and
by runtime introspection of the figure object. (4) A rebuild pass fixing every
finding, gated by six new regression tests, ruff, and mypy, and re-verified by
re-running the entire sweep.

**Environment of record.** Python 3.12.14, PyTorch 2.14.1+cpu (the uploaded
logs show 2.14.0+cpu), NumPy 2.1.3, Matplotlib 3.9.2, Linux x86_64, 2 torch
threads. Closest available parity with the original.

---

## 1. Executive summary

The instrument's **numerical core is excellent and reproduced bit-exactly**.
On this machine: 122/122 of the original tests passed; all 11 checks passed;
the AdamW and SGD+Nesterov update variants were measured identically
(`adamw[lerp,before,outside]`, `sgd[clone_grad,coupled_into_grad]`, both
bit-exact, 0 ULP); the 40-step reference run ended at the *same final
parameter* the uploaded README quotes, `1.31132972240448`; the hash chain,
gradient validation, finite differences, the fp32-resolution phenomenon
(60/60 frozen steps) and the near-zero-loss divergence guard all behaved
exactly as documented. The forensic design - labelled snapshots, immutable
records, measured rather than assumed optimizer semantics - is genuine and
holds up under adversarial reading.

Two critical defects and a cluster of smaller ones were found, all outside
the numerical core, all fixed in 1.1.0:

| # | severity | one line |
|---|---|---|
| F-01 | **critical** | every frame the live dashboard saved rendered an empty series panel - including every frame uploaded with 1.0.0 |
| F-02 | **critical** | any run reaching a non-finite value (the `diverging` preset) crashed the export: `summary.json` and `trajectory.json` could never be written |
| F-03 | high | README section 9 quoted verification numbers that were not in the shipped log (118 vs 122 tests, Windows vs Linux, wrong FPS figures, drifting digits) |
| F-04 | medium | check 9 asserted a frame *file existed*, not that it contained data - which is exactly why F-01 could ship |
| F-14 | medium | `trajectory.json` exported every float as a JSON *string* while `summary.json` carried numbers - internally inconsistent |
| F-05 | medium | `--hud-hz` was a documented, accepted, stored and never-read CLI flag |
| F-15/F-16 | medium | no lint/type gates: the codebase carried 140 ruff findings and 98 mypy errors (27 in the package) |
| F-17 | medium | no packaging: no pyproject, no entry point, no LICENSE, unpinned dependency ranges |
| F-06..F-11, F-18 | low | final-frame phase highlight, dead code branches, an `or 0.0` idiom, a fabricated SGD step counter, a missing `__all__` entry, legend/annotation collision, six placeholder-free f-strings |

Every finding is fixed, each critical/high one is pinned by a named
regression test, and the whole sweep was re-run green: **128/128 tests
(122 original + 6 new), all 11 checks including the new frame-content
assertion, ruff clean, mypy clean across 28 files, and a VLM + pixel
re-audit showing every data-bearing frame now renders all four series.**
Numerical behaviour is unchanged: the preset sweep produced identical
trajectories before and after the fixes (e.g. `adamw` 200 steps ends at
`1.4342018365859985` in both runs on this machine).

---

## 2. What was verified by real execution

| claim in 1.0.0 README | verdict on this machine |
|---|---|
| test suite passes | **reproduced** - but 122 tests, not the 118 the README claimed (see F-03) |
| all end-to-end checks pass | **reproduced** - 11 checks, not the "ten" the README claimed |
| AdamW measured variant `adamw[lerp,before,outside]`, bit-exact, 12 candidates | **reproduced exactly** |
| SGD+Nesterov `sgd[clone_grad,coupled_into_grad]`, bit-exact, 4 candidates | **reproduced exactly** |
| 40-step run: final parameter `1.31132972240448`, chain verified, reconstruction 0 ULP on all steps | **reproduced exactly**, including the final parameter digit-for-digit |
| gradient validation margins ~1.4x under the documented tolerance | reproduced with machine-specific error digits (4.30e-05 rel / 6.66e-07 abs vs the tolerance floor 9.56e-07) |
| `fp32_resolution`: attempted updates below 1 ULP, storage frozen | **reproduced** - 60/60 steps |
| `diverging` aborts at step 2 with a non-finite state | **reproduced** |
| near-zero-loss guard fires at 103 steps and refuses the ratio rise | **reproduced** (minimum loss 3.4539e-09 at step 95, ratio 3.19, regime 'still moving') |
| `near_stationary` reports `oscillating` with update >> gradient | **reproduced** (2.37e-05 vs 2.34e-02) |
| live view frame rate near target | reproduced (111.5 FPS against 120) |
| every preset is a real runnable configuration | reproduced - 11/11 run; **but the diverging one could not export** (F-02) |

---

## 3. The GUI audit (VLM + pixel + introspection)

### 3.1 What the frames showed before the fix

Eighteen frames were audited with a vision model (the 7 uploaded
`frames/frame_*.png`, the uploaded `run_checks_frames/frame_00001.png`, and
10 freshly rendered by the same code on this machine), each with the same
structured prompt (layout / series / text / bits / defects / verdict). The
verdict was unambiguous:

- **16 of 18 frames: the "recorded series (live)" panel is empty of data
  lines** while the state card in the same frame shows live values (step 25
  of 199, w = 0.581..., loss = 0.408...). The VLM's words: *"the line plot
  panel is visually empty despite being at step 25; no historical data
  lines"*.
- Objective pixel analysis agrees: the series-panel region of every uploaded
  frame holds **0.004% strongly-coloured pixels** - legend swatches only. A
  healthy panel holds 2-4%.
- The defect is in the **author's own shipped frames**, not introduced by
  this machine.

### 3.2 Root cause (established by runtime introspection, not inference)

`LiveDashboard.refresh()` rebuilds the series axes each static pass:
`_setup_series_axes` calls `Axes.clear()`, which **resets the view limits to
the unit box (0,1)x(0,1)**, and `_reattach_animated()` then re-adds the
persistent `Line2D` artists with `add_line()`. Neither `add_line` nor the
later `Line2D.set_data()` schedules an autoscale - `set_data` updates
neither the axes data limits nor the view limits - and the full
`canvas.draw()` inside `_save_frame` does not rescue it. Measured state at
frame 14 of a live run: series artists hold 37 points spanning x:[0,36],
y:[-4.35,8.33]; `ax.get_xlim() == (0.0, 1.0)`; `ax.get_ylim() == (0.0, 1.0)`.
Every point is transformed outside the viewport and clipped away. The state
card, causal chain and bit panel render correctly because they do not depend
on data limits - which is exactly why the bug hid: the dashboard *looks*
alive, and the author's check 9 asserted only that a frame file existed.

### 3.3 The fix and its verification

`LiveDashboard._rescale_series_axes()` now runs after every re-attach:
`ax.relim()` (rebuild the data limits from the attached lines) followed by
`ax.autoscale_view()` (apply them to the viewport). Verification, all three
layers:

1. **Runtime**: after the fix, `refresh()` leaves the viewport covering the
   recorded data; the regression test
   `test_refresh_restores_autoscaled_view_limits` asserts exactly this,
   including that the unit-box default is gone and the limits contain the
   data.
2. **Pixels**: the re-rendered GUI dump now measures **2.7-3.6% coloured
   pixels** in the series panel of every post-first-step frame (was 0.004%);
   `test_saved_frames_contain_series_ink` asserts the threshold on every
   frame of a pre-seeded run.
3. **VLM re-audit**: 13 frames re-audited (10 GUI-dump frames + 3 check-9
   ink frames). **All 10 data-bearing frames PASS with DEFECTS: NONE** -
   the VLM now reads four series, the marker and the trail. The 3 failures
   are the honest pre-first-step idle frames ("no step executed yet"),
   which are supposed to be empty.

### 3.4 Why the original checks missed it

Check 9 ran `live --headless --steps 2 --speed 100` with the default
`--frame-every 60` and asserted `len(frames) >= 1`: exactly one frame is
saved (the final one), and only its existence was checked. The new check 9
additionally dumps several frames at `--frame-every 2` and asserts the
data-ink fraction of the final frame (deterministically rendered after a
full refresh with all records), reporting the worst frame as an observation.
The empty-panel regression now fails the checks instead of shipping (this
assertion itself was validated: its first draft failed against the fixed
code for a legitimate reason - pre-first-step frames - and was refined to
the deterministic final-frame form, which is the frame that was empty before
the fix).

---

## 4. Findings matrix

Severity reflects impact on the instrument's *claims*: a rendering defect
that makes every frame lie about the data it shows is critical for a
forensic instrument even though no number was wrong.

### F-01 (critical) - live dashboard renders empty series panels
Evidence: section 3. Root cause: `Axes.clear()` + `add_line()` never
autoscale. Fix: `_rescale_series_axes` (relim + autoscale_view), regression
tests `test_refresh_restores_autoscaled_view_limits` and
`test_saved_frames_contain_series_ink`, plus the check-9 ink assertion.
Files: `dashboard.py`.

### F-02 (critical) - non-finite runs cannot be exported
Evidence: running the `diverging` preset through `export_all` raises
`ValueError: Out of range float values are not JSON compliant: inf` from
`summary.json` and `trajectory.json` (the other six artefacts succeed). The
diverged run itself is correct - the export crashes on `final_loss = inf`.
Root cause: `_write_json` used `json.dump(..., allow_nan=False, default=str)`;
the `default` hook never fires for a `float`, and the report dict bypassed
the recorder's sanitiser. The shipped check 5 only ever exported a healthy
run, so the path was latent. Fix: every JSON payload passes through the new
public `recorder.json_safe` (finite floats pass through, non-finite become
their `repr` strings); regression test
`test_diverged_run_exports_strict_json` parses the diverged run's artefacts
with strict `json.load` and asserts `final_loss == "inf"`. Files:
`recorder.py`, `exporters.py`.

### F-03 (high) - README section 9 contradicts the shipped evidence
Evidence (uploaded 1.0.0 artifacts): README says "118 tests"; the shipped
`checks_output.txt` says "Ran 122 tests" (and the source contains 122 test
methods; both pass). README says "all ten end-to-end checks"; `CHECK_COUNT`
is 11 and the log runs `CHECK 1/11`..`CHECK 11/11`. README's environment
block quotes Windows 11 / Python 3.14.3 / tkagg / Matplotlib 3.11.2 as
"(check 2)"; the shipped log's check 2 shows Linux gvisor / Python 3.11.2 /
agg / Matplotlib 3.10.1 (the Windows figures come from `real_run.txt`, a
different machine). README quotes 631 frames at 103.0 FPS over 6.13 s; the
log shows 183 at 81.6 FPS over 2.24 s. Several trajectory digits differ from
the shipped log (TOTAL ULP 121380477 vs 120952667; final-loss and
near-stationary digits) despite the section's own claim that every figure is
"copied from the shipped checks_output.txt". Fix: section 9 regenerated
from the current machine's log, every figure traceable to it, with an
explicit statement that trajectory digits are machine-specific and the
section is rebuildable from `python run_checks.py` output. Files:
`README.md`.

### F-04 (medium) - check 9 verified frame existence, not frame content
See section 3.4. Fix: multi-frame dump + final-frame data-ink assertion +
worst-frame observation. Files: `run_checks.py`.

### F-05 (medium) - `--hud-hz` is inert
Evidence: `LiveDashboard.__init__` stores `self.hud_hz` and never reads it;
`_update_fps` hard-codes `self._hud_due = now + 0.25`. The CLI documents the
flag. Fix: `self._hud_interval = 1.0 / hud_hz if hud_hz > 0 else 0.25`,
used by `_update_fps`; regression test
`test_hud_interval_follows_configuration` (0.1 s at 10 Hz, 0.5 s at 2 Hz,
0.25 s default). Files: `dashboard.py`.

### F-06 (low) - the final frame shows "active phase: none"
`run()`'s closing `_frame(redraw_static=True, active_phase=None)` drops the
causal chain's UPDATE highlight in the last saved frame of every run, which
reads as a defect. Fix: the final frame keeps `"UPDATE"` when records exist.
Files: `dashboard.py`.

### F-07 (low) - dead branches in `main.resolve_config`
`config.landscape.replace(...) if hasattr(config.landscape, "replace") else
...` - frozen dataclass instances never have a `replace` method (that is the
module-level `dataclasses.replace`), so the tested branch is dead. Fix: call
the `replace_landscape` / `replace_validation` helpers directly. Files:
`main.py`.

### F-08 (low) - `or 0.0` conflation in the state reader
`_state_as_floats` used `read("exp_avg") or 0.0`, mapping a legitimately
zero moment onto the same default as a missing one - precisely the
conflation the project's own documentation forbids elsewhere. Behavior was
accidentally identical (0.0 or 0.0); the idiom was not. Fix: explicit
`None` checks. Files: `optimizer_inspector.py`.

### F-09 (low) - fabricated SGD step counter
`_reconstruct_sgd` reported `step_after = 1 if buffer_before is None else 2`
- a synthetic value unrelated to the true step count (SGD keeps no counter
in its state). Display-only, but a forensic instrument should not fabricate
counters. Fix: reports `0` with a comment stating why; the Adam-family
branch (which reads the real counter) is unchanged. Files:
`optimizer_inspector.py`.

### F-10 (low) - missing `__all__` entry
`ulp_distance_same_binade` - the per-step movement metric the run report
aggregates - was absent from `fp32_inspector.__all__`. Fixed.

### F-11 (low) - stability-panel text collision
The overview figure's panel O let `legend(loc='best')` land on the regime
annotation (VLM: "overlapping/crowded text"). Fix: legend pinned upper
right, annotation shortened to its claim. Files: `visualization.py`.

### F-14 (medium) - floats exported as JSON strings
The recorder's `_jsonable` fell through to `repr(value)` for *finite*
floats, so `trajectory.json` shipped `"loss": "3.977567672729492"` (a
string) while `summary.json` carried real numbers - inconsistent, and
contradicting the exporter's own note ("Floats are Python floats").
Fix: finite floats pass through as JSON numbers (the hash chain remains
self-consistent within a run; 1.0.0 hashes do not compare against 1.1.0
artifacts, which the CHANGELOG states). Regression coverage:
`test_json_safe_stringifies_non_finite_floats` asserts both directions.
Files: `recorder.py`.

### F-15 / F-16 (medium) - no static-analysis gates
Baseline: 140 ruff findings (52 unused-noqa, 25 unused imports, 16
deprecated `typing` imports, 7 `zip` without `strict=`, ... ) and 98 mypy
errors (27 in the package proper: buffer types visible only as
`Union[Tensor, Module]`, Optional narrowing in the engine and the
reconstruction, `assert_single_float32_parameter` typed as plain `Tensor`).
Fixed by: declared buffer annotations, honest `nn.Parameter` return type,
explicit `None` checks (including a real one: the controller's drain loop
now stops on a `None` step), `zip(..., strict=False)` to preserve behavior,
import modernisation, and a `pyproject.toml` configuration whose ignores
each carry their rationale. **Both gates now report zero findings across
28 files.** The formatting is deliberately not enforced: several tests grep
engine source text, and reflowing it would break the invariants they pin
(one such test needed its pinned string updated for the `strict=False`
addition - the only test changed by the lint pass).

### F-17 (medium) - no packaging
No `pyproject.toml`, no console entry point, no `LICENSE`, unpinned
`requirements.txt` (`torch>=2.0` admits a major-version jump). Fixed:
`pyproject.toml` with the `spobs` entry point (verified working from a
clean `pip install -e .`), bounded dependency ranges, `[export]` and
`[dev]` extras, `py.typed`, Apache-2.0 `LICENSE`, and a CI workflow
(lint job, 3.10-3.13 test matrix running the suite *and* a smoke
`run_checks.py`, distribution build job).

### F-18 (low) - lint-level cleanup
Six f-strings without placeholders, `open(..., "r")` mode, one redundant
`int()` cast, one unused local, three non-conventional import aliases - all
fixed by the lint pass.

### Observations that are *not* defects

- `sgd_momentum` is classified `diverging` by the regime classifier on this
  machine: the loss genuinely grew by more than the configured factor and
  absolute thresholds over the final window. The classifier reported what
  happened; that is its job.
- The first frames of a headless session show the idle message in the bit
  and card panels before the first step exists - honest, and now documented
  in the check output.
- `_wait_until` deliberately spins for the last ~4 ms of a frame budget to
  reach 60-120 FPS on platforms whose `time.sleep` floor is ~10 ms; the
  cost is documented in the method.
- The renderer-inked `zip(flat, OVERVIEW_PANELS, strict=False)` source-grep
  test update (F-16 side-effect) is the only behavioural test change of the
  lint pass.

---

## 5. Enterprise readiness, before and after

| pillar | 1.0.0 (as uploaded) | 1.1.0 |
|---|---|---|
| installable package + CLI entry point | no (path bootstrap only) | `pip install .` -> `spobs` command, verified |
| licence | none | Apache-2.0 |
| tests | 122, passing | 128, passing (6 new regression tests, each named for its finding) |
| lint gate | none (140 findings) | ruff clean, curated rule set, rationale in config |
| type gate | none (98 errors) | mypy clean, 28 files, `py.typed` shipped |
| CI | none | lint + 3.10-3.13 matrix (suite + smoke checks) + package build, artifacts uploaded |
| end-to-end checks | 11, passing | 11, passing - now including frame-content assertion |
| docs | README only | README + ARCHITECTURE + USAGE + API + CONTRIBUTING + this audit |
| diverged-run export | crashes (F-02) | strict JSON, regression-tested |
| GUI rendering | empty series in every frame (F-01) | verified by VLM + pixels + runtime introspection |

## 6. Residual risks and recommendations

1. **The variant search is finite.** A future PyTorch whose update matches
   none of the twelve AdamW variants will be reported loudly as
   `RECONSTRUCTION MISMATCH` - correct behaviour, but the variant list will
   need extending when that happens. (The project's own README says the
   same; it remains true.)
2. **Source-grep tests couple tests to formatting.** Kept deliberately (they
   pin the no-fabrication invariant to the code), but contributors must
   know: the CONTRIBUTING guide now says so in bold.
3. **The `_jsonable` change alters record hashes** between 1.0.0 and 1.1.0
   artifacts. Within a run the chain is self-consistent; across versions,
   hashes are not comparable. Stated in the CHANGELOG.
4. **Bit-exactness across machines is still not promised** - the README says
   so and this audit confirms it: the uploaded logs' trajectory digits
   differ from this machine's in the last bits, while the deterministic
   update arithmetic (variant measurement, 0-ULP reconstruction) matches
   exactly. Provenance capture (`provenance.json`) remains the correct
   mitigation and is unchanged.
5. **VLM audits are a layer, not a gate.** The frame audit found F-01, but
   its verdicts needed pixel analysis and introspection to become actionable
   (and one VLM "PASS" on an empty pre-fix frame shows the model can be
   generous). The *pixel* assertion is what now runs in the checks; VLM
   auditing remains the human-scale tool for qualitative regressions.
