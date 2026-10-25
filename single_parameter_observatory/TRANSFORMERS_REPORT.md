# Parameter Studio 2.0.0 qualification record

Date: 2026-10-23. Platform: Windows, Python 3.12.15, PyTorch 2.14.1 CPU, Transformers 5.20.2. This report describes the implemented and exercised scope. It is not a claim that every possible architecture, third-party extension, or upstream model is qualified.

## Delivered behavior

The new Transformers workspace provides an architecture wizard, editable templates, a searchable block library, observed training curves, tensor tables, and checkpoint downloads. Node lab now has 48 built-in types: 21 existing experiment/JSON/script types and 27 HF/tensor types. Plugins add further typed blocks, templates, and declarative GUI panels.

Actual tensor workflows construct many-parameter PyTorch and Hugging Face models, prepare local tensor or tokenized data, train with AdamW/Adam/SGD/RMSprop, evaluate, run forward passes, generate tokens, execute five text pipeline tasks, inspect model state, and save/reload safe checkpoints. Named custom layers use an interpreted Python-syntax tensor-forward program. JavaScript extensions can produce architecture/data recipes. The Workflow menu exports a runnable Python project with an integrity-checked complete project snapshot; importing that Python module is inert, and explicit execution queues its selected graph.

Models are not restricted to one parameter. The default model budget is 2,000,000 parameters, configurable up to 100,000,000 within finite process/memory/tensor/job limits. The runtime catalog reports the resolved limits. This is a CPU implementation; hardware capacity and the 8 MiB graph artifact quota still apply.

## Upstream reading and audit

The official Transformers 5.20.2 source archive was acquired and hash-verified. All 2,764 runtime Python modules were parsed and inventoried: 1,166,534 source lines, 50,979,457 bytes. Selected configuration, AutoModel, modeling, tokenizer, collator, Trainer/TrainingArguments, optimizer, generation, pipeline, loading/saving and architecture paths received deeper manual review. Complete static archive coverage and selected manual reading are separate evidence claims. No claim of manually reading or security-auditing every line is made.

See `TRANSFORMERS_AUDIT.md` and `evidence/transformers_20261023/hf_audit_summary.json` for the pinned source identity, methods, counts, consulted paths and limitations. See `TRANSFORMERS_STUDIO.md` for the exact executable contracts and supported operations.

## Automated execution

- Full Python suite: **396 tests passed**, zero failures, errors or skips, in 82.016 seconds. All **11 execution checks** completed in 96.9 seconds. Evidence: `run_checks.txt` and `run_checks_console.txt` under `evidence/transformers_20261023/`.
- Graph model: **41 tests passed**. Browser-controller DOM qualification: **39 workflow groups passed**, zero unexpected controller errors. These offline controller tests are distinct from the actual browser replay below.
- All **13 supplied Transformers example workflows** completed as real source-runtime graph jobs; per-node outputs and input hashes are in `examples_execution.json`. Twelve build local models from recipes; the separate cached-pretrained workflow requires its declared pinned fixture. Seven pre-existing Node lab examples remain in the installed-release qualification.
- Code-generation tests cover import inertness, integrity checks, literal marker-shaped IDs, explicit graph execution and timeout/input errors. The complete suite includes those seven tests. Actual optimizer training through an exported Python program is separately demonstrated by the browser export and installed-release qualification.
- Ruff passed across the entire package and all Python scripts. Mypy passed across all 66 Python files, and compileall passed. Three delivery regressions and five isolated image/PDF helper smoke checks passed. The optional `audit` dependency extra declares the PDF libraries and ReportLab stubs required to reproduce those broad checks; install `.[dev,audit,export]` for the combined development environment.
- Lint/type-check evidence and installed-package qualification are saved alongside the execution receipts. `installed_verification.json` is the authoritative post-build receipt for HTTP behavior, all 20 portable example workflows, generated Python execution, artifact integrity and source/wheel/installed/sdist byte identity. `download/RELEASE-2.0.0.json` records the final artifact identities.

The legacy Matplotlib live renderer measured 6.2 FPS against a 120 FPS target on this host. Frame correctness and cache lifecycle passed their regressions; the measured rate is a performance limitation, not a 120 FPS claim.

## Actual browser replay

The normal HTTP connection recovered in the in-app browser. Tests ran in a separate QA server/storage directory on port 8766, preserving the user's original port-8765 experiment. The browser replay exercised:

- Experiment step/bounded run, timeline metrics, historical record selection, IEEE-754 microscope, negative-zero intervention, follow-live, checkpoint/branch, CSV data branching, notes, record-chain verification, project and evidence downloads, and server-workspace replacement recovery.
- Templates, search, palette additions, actual pointer dragging, port connection, undo/redo, validation, queueing and observed outputs.
- Custom plugin authoring, manifest-file import/install, enable/disable, generated GUI values, custom JavaScript execution, deliberate script failure and cancellation of an infinite loop.
- Architecture-wizard validation and custom widths/LayerNorm/SiLU/RMSprop training; editing and executing a custom tensor-forward program; real Python project download followed by explicit CLI execution.
- A **135-node / 124-edge workflow** covering **every one of the 48 built-in node types**, plus `local.transformer.recipes/mlp_layers`. Job `ecd514c58415415c991ba538abae8be1` succeeded with **135/135 nodes**, including all five exposed HF pipeline tasks, custom attention, pretrained-cache loading, fine-tuning, safetensors round-trips and scalar lifecycle/export actions. Its actual browser-downloaded immutable result is `browser/all-nodes.job.json`.

The separate RMSNorm/AdamW browser job trained **33 parameters in 5 tensors** for 20 steps: loss **1.5218572616577148 → 0.05354292690753937**, with changed parameter hashes and 15 optimizer-state tensors. The cached GPT-2 fixture trained **111,968 parameters in 64 tensors**. The browser-wizard RMSprop model trained 81 parameters; its downloaded Python export independently executed the six-node workflow successfully.

Browser screenshots, downloaded bytes, verification receipts and the interaction ledger are in `evidence/transformers_20261023/browser/`. Layout dimensions were checked at 1280×720, 900×700 and 390×844 with no document overflow; screenshot capture after explicit viewport overrides timed out in the browser automation service, so narrow-screen visual acceptance is not claimed. The default viewport was restored. Browser console warning/error collection was empty after the final replay.

This qualifies the listed workflows and the complete built-in node-type catalog. It does not prove every possible input combination or manually click every variant of every control. Failure, cancellation, malformed input, concurrency, persistence, sandbox and HTTP admission cases also have automated regression coverage.

## Defects corrected during this release

Runtime repairs include scalar-tensor hashing, strict JSON conversion of pipeline NumPy scalars, evaluation/dropout state, fractional token-ID rejection, attention-mask validation, padded generation, context/logit allocation limits, cached tokenizer padding/mask tokens, and invalid layer-list wiring.

UI repairs include a nine-view sidebar exceeding the viewport and covering experiment/transport controls, stale selected-run actions during pending selection, unregistered artifact rendering, tensor/count/hash field mappings, and consecutive palette additions occupying the same position. The placement repair preserves imported, template and explicitly dragged node positions. Focused regressions and actual browser replays cover the affected behavior.

## Boundaries and persistence

JavaScript plugins execute in isolated V8 child processes with bounded memory/time and JSON-only inputs/outputs. Custom tensor code is interpreted from an allowed AST; it cannot import, reflect on host objects, or execute arbitrary Python. HF operations run in a job-owned isolated CPU worker with process memory limits, finite deadlines, cancellation and process reaping. Cached models/tokenizers require a pinned commit and already prepared local files; the worker does not download, trust remote code or load pickle weights. These controls reduce the exposed surface; they are not a claim of a formal OS security boundary for arbitrary native code.

Safe-model checkpoints contain weights and a declarative architecture recipe. They do not contain a complete Trainer/optimizer/RNG resume state. Job-local live tensors disappear after the job. Save a safe checkpoint to retain weights. Project export preserves recipes, code, observations and registered artifacts.

The browser recovery and new runtime preserve the existing scalar experiment subsystem and its numerical forensic records. The original user's AdamW experiment retained all 142 recorded steps. Parsed pre/post-upgrade snapshots compare equal for complete runs, workspace settings, unapplied drafts and extension contents; only application version, save time and the extension admission revision changed. The old tab's recovery draft was downloaded before reloading it. Evidence is in `upgrade_preservation.json` and its named snapshots. Test-only projects and generated artifacts are kept separate from the user's active project.
