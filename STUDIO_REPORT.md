# Parameter Studio 1.2.0 verification

Built and verified on Windows on 2026-10-23. This report covers the new local
studio and its integration with the existing numerical engine. The 1.1.1
repair report remains a separate historical record.

## Delivered behavior

- Six functional browser views: spatial World, numerical Timeline, float32
  Microscope, CSV Data lab, Journal, and Artifacts.
- Independent real PyTorch experiments with play, quiescent pause, single-step,
  exact run-N, configurable playback, reset with retained history and remaining
  budget, fresh-history restart, presets, complete nested
  configuration, and explicit numerical abort/error reporting.
- Fresh configuration/data branches and exact-state branches/checkpoints.
  Optimizer/training/model edits retain actual imported samples when dataset
  generation fields are unchanged; requested batch-size changes rebatch them.
- Explicit parameter, IEEE-754 bit, learning-rate, and optimizer-memory edits,
  recorded as interventions rather than invented optimizer observations.
- Actual trajectories, eight chart metrics, comparisons, exact historical
  record inspection, recorded causal stages, input/target/prediction plots,
  and a separate real full-dataset float64 diagnostic loss landscape.
- Workspace and experiment notes, draggable cards, pan/zoom, branch connections,
  fit/focus controls, keyboard commands, and fullscreen where the browser allows it.
- Strict JSON project files with exact tensor bytes and integer state keys;
  complete optimizer/reference/scheduler/gradient/data/history restoration;
  atomic autosave; chain verification; numerical evidence ZIP downloads.
- A packaged loopback HTTP server, main `studio` subcommand, `spobs-studio`
  entry point, Windows launcher, and a complete workflow guide.

## Automated verification

The final package suite contains **260 passing tests**: the prior 218 tests,
25 studio workspace tests, and 17 studio HTTP/lifecycle tests. The final full
run completed in 25.158 seconds. The three delivery-tooling tests also pass.
Logs are saved in `evidence/studio_20261023/`.

Workspace tests execute real training and compare continuation records across
SGD, Adam, AdamW, accumulation, schedules, and the float64 reference track.
They cover forks/checkpoints, custom-data retention and rebatching, reset and
restart, optimizer interventions, signed zero, exact unsigned 64-bit seeds,
independent live workers, worker failures, ZIP evidence, autosave and shutdown,
and rejection of corrupt projects before replacing the live workspace.

HTTP tests cover assets, request bounds, strict JSON, duplicate keys, host and
origin checks, invalid routes/actions, actual project/export round trips,
real non-finite runs, startup failure, and worker cleanup. A Windows-specific
exclusive-bind regression prevents two studios from claiming the same port.
Port binding happens before opening autosave storage. Disconnected browser
responses, including Windows error 10053, do not create false backend failures.

Additional real-run verification reached 2,005 recorded steps. The browser state
sent 2,000 actual sampled observations, retained the first/latest points, and
still served an exact intermediate record independently. The state response was
374,482 bytes. Complete history is retained for project files and exports.

Ruff passes for the package, delivery scripts/tests, and installed-studio
verification script. Mypy passes across all 38 package Python files. Node's
syntax check passes for the shipped browser JavaScript.

The numerical execution checker completed **all 11 checks without an error**
using `--steps 12 --trace-steps 1 --guard-steps 103 --fps-seconds 4.0`.
This includes the existing Matplotlib dashboard. Its measured performance in
that run was 2.9 FPS against a 120 FPS target; this is an observation of that
separate renderer, not a browser studio frame-rate result.

## Browser verification

The real in-app browser was exercised against an isolated local preview
workspace at port 8766. No simulated backend or precomputed trajectory was used.
The browser verification included:

1. Create an independent SGD experiment from the preset chooser and execute
   an actual optimizer step.
2. Advance AdamW by five updates, inspect historical record zero, capture a
   checkpoint, and restore it into an independent branch with the same parameter.
3. Play a configured experiment continuously to its 30-update budget, display
   the real chart, compare experiments, and select additional moment metrics.
4. Apply parameter/learning-rate, optimizer-memory, and bit interventions;
   inspect the actual `MANUAL_INTERVENTION` journal entries and live parameter.
5. Import four CSV samples, change learning rate/step budget/batch size and a
   seed of `18446744073709551615`, and verify the child retains the actual
   imported targets with batch size two and the exact seed digits.
6. Save workspace notes and name, download a complete project and a recorded
   experiment's evidence ZIP, and reopen the downloaded project paused.
7. Inspect all six views, the real data/prediction/landscape output, integrity
   verification, checkpoint controls, and the browser error log.

Browser checks found and corrected dropped queued mutations, input drafts lost
between fields, stale polling responses, chart containers collapsed by flex
shrinking, hidden narrow-panel data previews, hero/card overlap, and browser
64-bit seed rounding. Mutations and project operations now share a queue;
configuration drafts persist; stale state requests cannot replace newer state.

Screenshots and browser-derived evidence are saved with the release evidence.
The observed browser panel was 884 by 884 pixels. This does not claim testing
on every browser, device, or display size.

## Packaging and reproducibility

`scripts/verify_studio_install.py` is run with the installed wheel's Python and
`-I`. It asserts that the application module comes from that environment,
then serves all three packaged assets, performs real training, retains custom
data through a configuration change, reopens a project with an identical sealed
record, and validates the numerical evidence ZIP. Its JSON result records the
installed module path, versions, asset hashes, and artifact identities.

The wheel and source distribution are built from the final sources. The portable
source/evidence archive has an inventory and SHA-256 entry for every member
except the manifest itself; the delivery script verifies archived and live
source bytes before publication. The previous 1.1.1 archive remains available
as the preserved baseline.

## Scope

The studio operates on the real single float32 CPU parameter, linear/tanh model,
MSE objective, and existing Adam/AdamW/SGD implementations. It supports the
configuration and state controls described in `STUDIO.md`; arbitrary Python,
multi-parameter models, remote multi-user hosting, and unlimited resource usage
are outside this release. Manual parameter edits require finite float32 values.

Exact checkpoint continuation is tested on this installed runtime (Python
3.12.15, PyTorch 2.14.1+cpu). Arithmetic equivalence across other runtime versions
is not asserted. Project hashes provide integrity checking, not an authorship
signature. Practical memory/project size limits can be reached before the
configured ceilings because full forensic histories are intentionally extensive.
