# Editable Llama component release qualification · 2.2.0

The studio now compiles the connected forward graph into an actual registered
PyTorch causal language model. RMSNorm, rotary positions, grouped-query causal
attention, expanded or fused SwiGLU, projections, masks, positions, embeddings,
residuals and vocabulary logits are editable tensor operations. Named custom
tensor programs preserve autograd and can register trainable layer recipes.

## Browser work

The actual in-app browser exercised both connection directions, blank-canvas
compatible-node discovery, search, known rank/dtype filtering, keyboard opening,
Escape cancellation, atomic undo/redo, direct edits and invalid-draft execution
guards. The visual builder exercised expanded, fused and custom architectures,
multiple decoder layers, invalid query/KV geometry and recovery. Library and
inspector panes collapse on desktop; visual results expose real operation cards,
loss charts, tensor observations, token sequences and decoded text. Advanced
JSON is collapsed and lazy. Runtime inspection opens the selected execution.

| Actual browser workflow | Nodes / wires | Outcome |
| --- | --- | --- |
| Expanded one-layer Llama | 36 / 47 | Training, checkpoint reload, generation and decoding succeeded |
| Custom two-layer Llama, gate scaled by 0.9 | 52 / 71 | 25 updates; evaluated loss 3.555287 to 1.213807; exact reload |
| Fused SwiGLU Llama | 32 / 42 | Training, reload and generation succeeded |
| Manually connected token model | 6 / 5 | Constructed through port discovery; compiled and inspected; epsilon edit committed before execution |
| Plugin-authored source plus live V8 tensor pass-through | 37 / 48 | Every node succeeded; actual training, safe reload, generation and decoding |

All 13 new node types executed successfully across these workflows.
`browser-verification.json` names exact jobs and verifies all per-node statuses.
The custom browser download was explicitly executed as an isolated Python
project; its final model hash matched the browser exactly. The plugin job was
also downloaded through the browser and retained alongside its API snapshot.
Final QA console checks reported no errors.

The real-browser tests found and fixed immediate inline edit timing, stale
builder validation, inert desktop inspector closing, misleading functional
module labels, overly dense low-zoom grid dots, typed plugin default rendering,
and a false missing-type description for a valid plugin with empty description.
The backend audit additionally fixed semantic plugin default/option validation
and collisions between imported expression IDs and live worker IDs. Imported
recipes have deterministic namespaces; genuine live pass-through and fanout
keep their registered identities and shared weights.

## Automated and numerical checks

- Final source suite: **414 Python tests passed**, zero failures/errors/skips.
- Full execution runner: **11 of 11 checks passed**, including real scalar
  optimizer observations, exports, command surfaces and near-zero-loss guards.
- Editor model: **51 tests passed**. Controller DOM: **50 groups passed**, zero
  unexpected errors. These use mocked HTTP and are separate from browser proof.
- Builder/inspection DOM: **13 groups passed against each of two actual jobs**.
- Ten component tests include native HF Llama parity for logits, causal loss and
  every trainable gradient after copying identical weights; causality/padding;
  actual learning, safe reload and generation; deterministic Python replay;
  custom trainable/integer/boolean operations; invalid recipes and budgets; and
  actual isolated worker/plugin integration. The final affected subset passed
  all **43 tests** (33 plugin and 10 component tests).
- Ruff passed. Mypy passed across **71 files**.
- Isolated wheel installation passed: **23 packaged workflow examples** executed
  (7 core and 16 Transformers), all five exposed pipeline task kinds ran, and
  native GPT-2/Llama Python exports executed. All **113 package source files**
  matched both the wheel and installed copy; **161 source-distribution file
  comparisons** passed. The receipt names the actual installed module path.

`final-qualification/verification.json` records expanded, custom two-layer and
fused standalone execution and each exported Python replay. Expanded/fused
evaluated loss fell from 3.592102 to 1.398548 after 20 updates. The separate
two-layer default custom-gate variant fell from 3.553470 to 1.192609 after 25.
Reloaded losses and hashes were exact. These use eight local educational
sentences and random initialization, not pretrained Meta weights.

The execution runner measured the separate Matplotlib live view at **16.8 FPS**
on this host during the final run. Its 120 FPS target was not achieved. The
runner reports this performance observation; correctness checks do not establish
that every workload meets a frame-rate target.

## Preservation and delivery

Fresh main-workspace backups and browser draft-recovery downloads were saved
before restarting the server. `final-restart-preservation.json` verifies retained
graph definitions, parameters, connections, scalar records, jobs, artifacts and
drafts. Revision counters and the active viewport changed through reconnection
and UI updates; these are recorded separately from retained content. Existing
and newly created user graphs remain in the main workspace.

The compact portable bundle includes three qualified workflows, disposable
projects, trained checkpoints, Python exports and exact jobs. Its independent
review passed **281 checks**. The installed distribution check and final archive
inventory/hash receipts are separate evidence; see `installed_verification.json`
and the release index under `download/`.

This release is a bounded local CPU tensor studio. Custom tensor code uses a
restricted mathematical language and plugin JavaScript runs in disposable V8
workers. Arbitrary operating-system Python, unlimited resources, distributed/GPU
training, complete upstream Trainer coverage, and exact optimizer/RNG resume are
outside this release. Functional attention/RoPE slots report zero parameters;
their connected projections own the trainable weights.
