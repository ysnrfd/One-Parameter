# Change log

One line per defect: what was wrong, and what the fix is. Defect ids refer to the review that prompted this pass.

## Accuracy of the reported numbers

- **D1**   README §9 quoted step 5's values as the end-of-run values (final
  parameter ` 1.4101637601852417` against the log's `1.31132972240448`). Fixed by
  regenerating the whole "What a run actually does" subsection from the shipped
  log and naming the check each figure comes from.
- **D8**   README §9 cited Matplotlib's version, an SGD+Nesterov probe and a test
  runtime that the log did not contain. Fixed by adding Matplotlib and its backend
  to `EnvironmentAudit`, extending the check to probe both AdamW and SGD+Nesterov,
  and removing the pinned runtime figure (it moves between runs; the log is now
  cited instead).
- **D10**  README said "~40 s" where the log said 71.145s. Fixed by not pinning a
  per-run figure at all, and by recording the suite's wall clock twice in the log.
- **D11**  README described `diverging` and `near_stationary` behaviour that the
  log never produced. Fixed by running both presets in check 9, so the claims are
  observations.
- **D14**  README §9 stated numbers without sources. Fixed by citing the check
  for every claim, or naming the test, for each one.

## Structure of the check log

- **D2**   the export check printed `CHECK 8/8` while running inside check 4, so
  the headers read 1, 2, 3, 4, 8, 5, 6, 7. Fixed by making export its own check at
  position 5 and numbering the sequence `CHECK 1/9`   `CHECK 9/9` in execution
  order.
- **D13**  `run_checks.py` never exercised `inspect`, `presets`, `controls` or
  `live --headless`. Fixed by adding check 9, which runs each in-process, asserts
  its exit status and the headers it must print, and verifies the presets listing
  covers `config.PRESETS`.

## Labels that misdescribed their quantity

- **D4**   `render_trace` printed the *reconstructed* update under the label
  "realised float32 displacement", implying it was observed. Fixed by labelling
  every such value as reconstructed (in `main.py`, `visualization.py` and
  `exporters.py`), printing the observed stored delta under its own name, and
  renaming the CSV columns to `reconstructed_optimizer_total_update_fp32` and
  `optimizer_attempted_update_fp64`.
- **D5**   `MAX GRADIENT VALIDATION ERROR` and `MAX FINITE-DIFFERENCE ERROR` did
  not say *relative*, though the values were relative, which is why the README's
  absolute figure could not be reconciled with the log. Fixed by renaming the
  labels, adding the absolute forms as their own reported fields, and printing
  both forms in the ledger.
- **D9**   the trace printed the ULP ratio twice, once as "ULP relative", and
  printed the exact representable-value count as if it were a movement. Fixed by
  showing the movement metrics once, in the `PARAMETER` block, under distinct
  names, and adding a note when they diverge at a binade or sign boundary.

## Numerics

- **D3**   `decimal_exact` and `decimal_shortest_float32` were produced by the
  same helper, so both printed `0.1` for `float32(0.1)`. Fixed by computing the
  exact rational expansion from the stored bits; `decimal_exact` is now
  `0.100000001490116119384765625`, and the field's docstring says which form is
  which.
- **D6**   `TOTAL ULP MOVEMENT` was the exact representable-value count, which one
  sign-crossing step dominated (about 96% of the total). Fixed by adding
  `ulp_distance_same_binade` as the per-step movement metric, keeping the exact
  count as a separate reported quantity, and printing both with the number of
  contributing steps and the steps that crossed zero.
- **D7**   `exact_half_ulp_tie` used floating-point midpoint arithmetic, which
  cannot decide an exact halfway case. Fixed by comparing exact rationals derived
  from the stored mantissas; a constructed tie is now detected and a one-float64-ULP
  near miss is not.

## Two defects found while verifying the above

- The `events:` line in the trace presented a partial list as if it were complete.
  Fixed by stating that it holds the events emitted up to the moment the record was
  sealed, and naming the events that cannot appear there (D12).
- `live --headless` could finish a short run before its first redraw and write no
  frame at all. Fixed by always writing the final state, and covered by
  `tests/test_views.py::test_headless_dashboard_always_writes_a_frame`.

## Second pass: live view fidelity and helper contracts

- **D19**  the live dashboard and the static overview figure each drew a recorded
  series themselves, and disagreed about two behaviours that matter. Fixed by
  making one drawing path: `visualization.recorded_series` and
  `visualization.draw_recorded_points` are now public, and `LiveDashboard.refresh`
  calls `draw_recorded_points` rather than reimplementing the filtering and the
  marks. `visualization.plot_recorded_series` is the single-series panel wrapper
  built on the same helper.
- **D15**  a one- or two-step run left the live series panel empty, because
  `Axes.plot` of a single point has no segment to draw; the frame was
  indistinguishable from "no run has happened yet". Fixed by the shared helper
  putting a marker at every recorded point, plus a scatter, so a single recorded
  step is visible in both views.
- **D16**  the live view plotted `nan` and `inf` unfiltered, and Matplotlib used
  them when auto-scaling, so a diverged run produced garbage axis limits that hid
  the steps that *were* finite. Fixed by filtering through the shared helper, and
  covered by a test that runs the diverging preset and asserts all four axis
  limits are finite.
- **D17**  the state-card title read "step -1" before the first step, because it
  used `step_index - 1`. Fixed to read "no step executed yet" when there is no
  record, and otherwise to take the index from `records[-1].step` rather than from
  arithmetic on `step_index`.
- **D18**  `draw_bit_pattern` declared a `previous` parameter that it never read,
  while every call site passed one. Fixed by removing the parameter and stating in
  the docstring that the highlighted bits are this step's own transition from
  `changed_bit_positions`, not a comparison with the previous step.
- **D20**  `_exact_decimal` and `_exact_fraction` both took a raw `uint32` and
  applied its sign bit, but neither said so, and they return different types. Fixed
  by documenting each one's return type, its raw-word input and its sign handling,
  cross-referencing both from `Float32Bits`, and adding a test that checks they
  agree on magnitude and that the sign follows the sign bit   with `-0.0` as the
  case where the two are not interchangeable, since `_exact_decimal` keeps the
  sign in the string and `_exact_fraction` returns zero.

### One more defect found while verifying the above

- `live` exited 3 on a diverged run, because it shared the exit-status logic with
  `run`. A view's exit status should report whether the view ran; a run's should
  report whether the run stayed finite. Fixed by passing `view_command=True` from
  `live`, which suppresses the "run aborted" status while still printing
  `NON-FINITE DETECTED` and `RUN ABORTED`, and still failing on a broken history
  chain or a validation failure. `run --preset diverging` still exits 3, because
  for that command the divergence is the result. Covered by
  `tests/test_views.py::test_live_exit_status_separates_the_view_from_the_run`.

## Third pass: exit-status ordering, series-table hygiene, near-zero-loss evidence

- **D21**  `_finish`'s four guards were correctly ordered but nothing said so, and
  the D15 test covered only the divergence case, so a future edit could have moved
  the chain check below the divergence guard and let a broken history through
  `live` undetected. Fixed by a comment above the divergence guard naming the
  ordering and why `view_command` guards that check alone, and by
  `tests/test_views.py::test_exit_status_ordering_is_what_the_comment_claims`,
  which covers all four statuses for both a view and a run.
- **D22**  `visualization.LIVE_SERIES` was a 13-entry table referenced nowhere,
  named as if it were the live view's series when the live view uses
  `dashboard.DASHBOARD_SERIES`. Fixed by removing it and its `__all__` entry; the
  two surviving tables   the overview's `panels` tuple and `DASHBOARD_SERIES`    now each carry a comment naming the other and stating that they draw the same
  recorded attributes in deliberately different subsets, both through
  `draw_recorded_points`.
- **D23**  the classifier's most important behaviour, refusing to call a large
  ratio rise at a near-zero loss a divergence, was implemented and tested but
  never appeared in the shipped log, because the check run was 40 steps and the
  guard fires much later. Fixed by adding check 10, which runs the default preset
  for **103 steps**   measured, not chosen: the loss reaches its minimum at step
  95 and the guard first fires at 103, the ratio crossing 1.05 between 102
  (0.951) and 103 (3.19)   and prints the minimum loss and its step, the final
  loss, the regime, and the classifier's `rationale` verbatim. The check records
  what the classifier did and requires no particular regime.

### Two more defects found while fixing the above

- The documented exit status 2 for a broken history chain was **unreachable**:
  `final_report()` asserts the chain and raised before the status guard was
  reached. Fixed by verifying the chain before building the report and returning
  2 there, so the documented status is the one a caller actually gets.
  `Recorder.assert_chain` still raises for anyone calling it directly.
- Writing check 10 initially shadowed the `report` parameter with the classifier's
  output, so the check would have written to a `dict`. Fixed before the first run
  by naming the classifier's report `stability`.

## Fourth pass: dead branches, orphaned prose, and lists that stopped being complete

- **D24**  `_finish` verified the history chain twice: once before the report was
  built and once after the export and plot blocks. The second check was unreachable,
  and it sat immediately above a comment instructing the reader not to check the chain
  there. Fixed by removing the dead branch; the notice is now printed once and the
  status is 2. Covered by
  `tests/test_views.py::test_broken_chain_notice_is_printed_once`, which also asserts
  that `_finish` contains exactly one `if not ok:` guard.
- **D25**  the module-level comment in `visualization.py` that described the removed
  `LIVE_SERIES` table used the `#: ` prefix, so a Sphinx build would have attached it
  to `configure_backend`, and its content was already carried by the two surviving
  tables' comments. Fixed by deleting it; what was removed and why stays in the D22
  entry above.
- **D26**  the "Verification" section named "nine checks" and "108 tests", both stale
  after the third pass. Fixed by replacing the pinned figures with a pointer to the
  shipped log, which is the same treatment README §4's test runtime got in the first
  pass: a number that changes with every added check does not belong in prose.
- **D27**  `test_live_series_table_is_the_only_live_catalogue` compared the live
  view's attributes against every `("word",` match in `visualization.py`, which passed
  by accident rather than by construction. Fixed by extracting the static view's list
  into the module-level constant `visualization.OVERVIEW_PANELS`, which
  `build_overview_figure` now iterates, and by having the test compare two defined
  sets instead of two text scans.
- **D28**  README §4 said `run_checks.py` runs "everything" and then listed eight of
  the ten checks. Fixed by removing the enumeration and pointing at the log, so the
  sentence cannot go stale when a check is added.

## Fifth pass: the live view was not fluid

The live view updated roughly once a second. Measured rather than assumed, the cause was that a single redraw cost about 1.8 seconds:

| what | cost on this machine |
|---|---|
| a bare line, no text | 16.9 ms |
| the same with tick labels | ~85 ms |
| the 45-line state card | ~1200 ms |
| a full redraw of the live figure | ~1847 ms |

Text rasterisation is about 80% of the cost, and the redraw was triggered by *step arrival*, so the frame rate was pinned to the step rate rather than to anything the loop could do.

- **Rebuilt the view in three layers** with independent cadences: the static
  text layer at `static_hz`, the recorded series at `data_hz`, and the marker
  every frame. Only the last two are redrawn per frame, and neither has text.
- **Cached the text layer and blitted over it.** Measured 61.7x on this figure:
  4.3 FPS for a full redraw against 264 FPS for a cached-background frame. A
  blitted frame now costs 5.1 ms.
- **Fixed a self-inflicted stall**: the frame after a static redraw discarded the
  freshly cached background and did another full redraw, so every static pass
  cost a second it did not need to.
- **Stopped redrawing the causal chain on every static pass.** It holds 12 labels
  and 11 arrows and changes only when the highlighted phase changes; that halved
  the static redraw from about 200 ms to about 93 ms.
- **Replaced the 45-line live card with a compact HUD**, leaving
  `render_state_card` unchanged as the complete version used by `trace` and the
  report.
- **Made the frame wait accurate.** `time.sleep` has a floor of about 10 ms on
  this platform, which capped the loop whatever the target; the coarse part of
  the wait is slept and the last few milliseconds are spent reading the clock.
- **Stopped the headless frame dump from throttling the view.** Saving a frame
  needs a full text redraw, so `--frame-every` now defaults to 60 and the dump
  stays a diagnostic rather than becoming a brake on the frame rate.
- **Recorded the frame rate in the shipped log** as check 11, so "the view is
  fluid" is a measurement rather than an impression. It records the achieved
  rate and fails only if the view has collapsed to its pre-caching behaviour.

## Sixth pass: Enterprise-ready hardening, numerical precision, and edge-case defect fixes

- **D29**  `validate_float32_bits` failed on positive powers of two (e.g. `1.0`, `2.0`, `0.5`)
  because `_bit_level_spacings` returned `(toward_zero, away_from_zero)` while `validate_float32_bits`
  compared them directly to `(spacing, spacing_down)`. For positive values, `spacing` (`distance_next`)
  is away from zero and `spacing_down` (`distance_prev`) is toward zero. Fixed by consistently mapping
  binade spacing and towards-zero spacing symmetrically across both positive and negative values.
- **D30**  `fp64_candidate_from_state` accepted `coupled_decay: bool` but ignored it in the `sgd`
  branch, computing decoupled decay rather than coupling weight decay into the gradient and momentum
  buffer as PyTorch SGD actually does. Fixed by adding coupled decay handling to SGD in `fp64_candidate_from_state`.
- **D31**  `_reconstruct_sgd` erroneously set `adaptive_only = weight_before.clone()` when
  `decay_coupling == "decoupled_after"`, zeroing out `adaptive_delta` and falsely attributing the
  entire update to weight decay. Fixed by removing the redundant overwrite so the gradient update is preserved.
- **D32**  `audit_optimizer_source` searched for `_single_tensor_adam` in `sgd.py` when auditing SGD.
  Fixed to search for `_single_tensor_sgd` (or `sgd`).
- **D33**  `_moments_from_state` called `.item()` unconditionally on `state_before["step"]`, raising
  `AttributeError` if `step` was an `int`. Fixed by safely inspecting `.item()` only when present.
- **D34**  `_extract_function_source` appended the definition line of the next function before breaking.
  Fixed by testing boundary conditions before appending.
- **D35**  `LiveDashboard._on_key` lowered `event.key` before checking `'shift+r'`, causing uppercase `'R'`
  (restart) to match `'r'` (reset). Fixed by checking `'R'` and `'shift+r'` before `'r'`.
- **D36**  `StepStability` and `ConvergenceAnalyzer.report` used `abs(record.ulp_distance_count)` for
  `ulp_movement` and `total_ulp_movement_in_window`, causing window ULP movement to explode to 2.14 billion
  on sign crossings. Fixed by using `abs(record.ulp_distance_same_binade)`.
- **D37**  `REGIMES` lacked `"not enough data"`, which is returned when no steps are recorded. Fixed
  by adding `"not enough data"` to `REGIMES`.
- **D38**  `_growth_ratio` in `convergence.py` filtered out `v > 0.0`, returning `None` if loss started
  at 0.0 and grew. Fixed to support `0.0` initial loss gracefully.
- **D39**  `build_overview_figure` declared a `figsize` parameter with default `(17.0, 20.0)` but
  hardcoded `(18.0, 16.0)` in `plt.subplots`. Fixed to respect the passed `figsize`.
- **D40**  `_draw_stability` displayed `ONE_DIMENSIONAL_DISCLAIMER` instead of `STABILITY_DISCLAIMER`.
  Fixed by rendering `STABILITY_DISCLAIMER`.
- **D41**  `_draw_landscape` raised `ValueError` if `last["samples"]` was empty. Fixed with a safety check.
- **D42**  `ObservatoryModel.assert_single_float32_parameter` omitted the check for `is_leaf`. Fixed
  by verifying `parameter.is_leaf is True`.
- **D43**  `loss_fp64_python` lacked input validation for empty or mismatched tensors, causing unhandled
  division by zero. Fixed with strict validation matching `analytic_gradient_fp64`.
- **D44**  `_apply_schedule` lacked the closed-form cosine fallback promised when `CosineAnnealingLR` is
  unavailable. Fixed by implementing the closed-form cosine schedule fallback.
- **D45**  `TrainingEngine.step` captured `snapshot_b` and `snapshot_c` consecutively after all backward
  passes instead of capturing `snapshot_b` after forward and before backward. Fixed by taking `snapshot_b`
  immediately following forward.
- **D46**  `command_trace` in `main.py` ignored `args.verbose`, always printing the full end-of-run report.
  Fixed by passing `quiet=not getattr(args, "verbose", False)`.
- **D47**  `run_blocking` in `ExperimentController` failed to parse command strings like `n2` without space.
  Fixed by supporting both `n <count>` and `n<count>`.
- **D48**  `export_all` omitted `stability.csv` and `provenance.json` keys when analyzer/provenance was
  omitted. Fixed by guaranteeing all `EXPORT_FILES` entries exist in the returned dictionary.
- **D49**  `_is_exact_half_ulp_tie` in `numerical_analysis.py` used `stored.raw_uint32` instead of
  `candidate_f32`'s raw word when computing the candidate midpoint. Fixed to use `candidate_raw`.
- **D50**  `config.py` threw away rich preset descriptions in `specs` and defined shortened duplicates
  in `PRESET_DESCRIPTIONS`. Fixed by deriving `PRESET_DESCRIPTIONS` directly from the specifications.
