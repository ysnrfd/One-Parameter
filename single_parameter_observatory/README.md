# Parameter Studio — Transformers Visual Coding

A local visual coding studio for **multi-parameter PyTorch and Hugging Face Transformers models**, with typed node workflows, editable architecture and forward-code blocks, real training and inference, plugins, safe checkpoints, and runnable Python project exports. The original scalar observatory remains available for detailed float32 storage, optimizer, rounding, and historical-record inspection.

> **Current release: 2.2.0 — Editable Llama Component Graphs, 2026-10-23.** Open the
> local browser workspace with **Launch Studio.cmd** or `spobs studio`, then choose
> **Transformers** or **Node lab**. Read the [studio guide](STUDIO.md),
> [Transformers workflow guide](TRANSFORMERS_STUDIO.md),
> [HF source audit and support boundary](TRANSFORMERS_AUDIT.md), and
> [editable Llama block guide](LLAMA_BLOCKS.md),
> [native HF Llama implementation and 2.1.0 qualification](LLAMA_STUDIO.md), and
> [HF Viewer / Llama source review](LLAMA_SOURCE_NOTES.md).
> [TRANSFORMERS_REPORT.md](TRANSFORMERS_REPORT.md) preserves the historical 2.0.0
> Transformers release qualification.
> [NODE_LAB_REPORT.md](NODE_LAB_REPORT.md) preserves the historical 1.3.0 node/plugin
> results; [STUDIO_REPORT.md](STUDIO_REPORT.md) preserves 1.2.0 studio results;
> [REPAIR_REPORT.md](REPAIR_REPORT.md) preserves the 1.1.1 numerical repairs.
> Section 9 retains historical 1.1.0 Linux measurements, which do not describe
> the current release or machine.

## 1. What this is, and what it is not

Training uses actual PyTorch modules, autograd, and optimizer steps. Transformers
workflows can instantiate supported HF AutoModels from configuration or load
pinned, cached safetensors weights; prepare tensors or tokenize text; compose
layers; train, evaluate, generate, and run supported pipelines; inspect actual
parameter tensors; and export or reload safe model checkpoints. Models created
from config start with newly initialized weights. Workflow execution performs
no Hub downloads or remote-code loading.

The visual editor includes separate architecture, data, optimizer, training,
inference, observation, and artifact blocks. JavaScript plugins can generate
recipes and data. Custom tensor-forward bodies use a restricted Python AST over
named trainable layers and tensor operations. **Download runnable Python project**
exports a normal editable Python entrypoint that requires the installed Studio
runtime; importing that file is inert, and execution is explicit.

Llama component graphs expose token inputs, embeddings, RMSNorm, Q/K/V and output
projections, position IDs, RoPE, grouped-query causal attention, SiLU, gated
products, residual additions and vocabulary logits as separate connected blocks.
**Compile visual causal model** registers the trainable modules once and replays
the connected tensor graph on each forward, training update and generated token.
The visual builder offers expanded, fused or custom-activation SwiGLU branches.
Drag either an input or output port onto empty canvas to search compatible blocks,
then add and connect a chosen port in one undoable edit. Suggestions reject known
dtype/rank mismatches; displayed widths are advisory and the compiler checks
the complete geometry. Custom-operation outputs remain flexible until compiled.
Inline settings commit valid edits immediately, while invalid text remains an
explicit draft that blocks queueing. Inspector and Library controls can collapse
either pane at every screen width to give the canvas more room.

HF execution results use visual operation cards, data/model descriptors, actual
loss charts, token strips and decoded text. **Inspect model visually** opens the
same observed execution in Transformers. Exact output JSON and worker logs are
collapsed and generated only when expanded; downloading the original result
retains its complete contents.

The original scalar engine remains deliberately small: a real
`torch.float32` parameter in `y = o * tanh(s * w)` or the linear variant.
Its detailed bit timeline and independent optimizer reconstruction describe that
scalar experiment. They are not an exhaustive forensic trace of every weight in
a Transformers model. The many-parameter runtime is a bounded local CPU
laboratory, with configurable resource budgets rather than unlimited hardware
or an operating-system virtual machine. See the current guides and report for
supported operations and what has actually been qualified.

## 2. Design rules

The scalar forensic engine retains the following evidence contracts. The
Transformers worker separately reports actual model/tensor descriptors, losses,
parameter hashes, and safe checkpoint artifacts; it does not invent scalar-style
forensic records for all model weights.

| Rule | How it is enforced |
|---|---|
| No simulated training | Training updates use `loss.backward()` and `optimizer.step()`. A test greps `TrainingEngine.step` to confirm it contains no `.grad =`. Studio manual interventions are separate, explicit events with before/after provenance. |
| No fabricated values | Every recorded number is read out of a real tensor via `.item()`/`.detach()`; nothing is generated, smoothed or interpolated. |
| Raw != derived | `ParameterStepRecord` stores raw observations and derived analysis in separate fields; a derived number never overwrites the raw one. |
| Immutable history | Records are frozen dataclasses chained by SHA-256. `verify_chain()` reports the first step whose hash no longer matches. |
| No silent failure | Every failure mode has a named exception in `errors.py`. Invariant violations raise with the exact violated condition. |
| Version honesty | The installed optimizer's source is read and audited; the update order is *measured*, not assumed. |

## 3. Layout

    single_parameter_observatory/
      main.py                  CLI entry point (studio / run / trace / live / controls / inspect / presets / selftest)
      studio_session.py        independent experiment worlds, branches, exact checkpoints, project persistence
      studio_server.py         loopback HTTP server over the real workspace
      studio_graph.py          typed workflow registry, immutable execution queue, native experiment nodes
      studio_plugins.py        validated custom-node manifests and isolated JavaScript execution
      studio_transformers.py   HF/tensor node catalog and bounded job-worker client
      studio_transformers_worker.py  actual model/tensor execution and restricted forward interpreter
      studio_components.py     editable Llama operation definitions and portable component templates
      studio_components_runtime.py  validated tensor-DAG compiler and replayable trainable model
      studio_codegen.py        inert portable project snapshots in runnable Python entrypoints
      studio_web/              complete local browser interface, no JavaScript build dependencies
      config.py                frozen ExperimentConfig + the training presets
      errors.py                the error taxonomy (one exception per failure mode)
      model.py                 the single-parameter model, the float64 replica, the dataset
      tracker.py               tracked-parameter invariants and evidence
      training_engine.py       one real step; play/pause/step/run-N/reset/restart
      recorder.py              immutable records, event log, hash chain, final report
      optimizer_inspector.py   version audit, state decoding, independent reconstruction
      fp32_inspector.py        IEEE-754 bits, hex, ULP neighbourhood, bit transitions
      numerical_analysis.py    rounding residuals, finite differences, landscape probe
      validation.py            autograd vs closed form vs finite differences, bit/ULP checks
      convergence.py           rolling statistics and the observed-regime classifier
      visualization.py         figures and the textual state card
      dashboard.py             the live view and the headless control session
      exporters.py             CSV / JSON / NPZ (+ bit, optimizer-state, summary artefacts)
      run_checks.py            runs every check and writes checks_output.txt
      tests/                   standard-library unittest and adversarial regression tests

At the repository root: `pyproject.toml` (packaging, the `spobs` console entry
point, ruff and mypy configuration), `LICENSE` (Apache-2.0),
`.github/workflows/ci.yml` (lint + Python 3.10-3.13 test matrix + package
build), and the documentation set: `ARCHITECTURE.md`, `USAGE.md`, `API.md`,
`single_parameter_observatory/CONTRIBUTING.md`, `AUDIT.md` (the historical
1.1.0 audit), `REPAIR_REPORT.md` (the 1.1.1 numerical fixes), `STUDIO.md`
(the current studio guide), `STUDIO_REPORT.md` (the 1.2.0 studio evidence),
`NODE_LAB_REPORT.md` (historical 1.3.0 node and extension evidence),
`TRANSFORMERS_STUDIO.md` (the current model/workflow guide),
`TRANSFORMERS_AUDIT.md` (source audit and support boundary),
`TRANSFORMERS_REPORT.md` (historical 2.0.0 release qualification),
`LLAMA_STUDIO.md` (native HF Llama workflow and historical 2.1.0 qualification),
`LLAMA_BLOCKS.md` (editable component graphs and current qualification),
`LLAMA_SOURCE_NOTES.md` (HF Viewer and native Llama source review), and
`single_parameter_observatory/CHANGELOG.md`. The architecture, usage, and API
guides also live inside `single_parameter_observatory/`.

`errors.py` is not in the original layout; it exists so every other module can share the error taxonomy without creating an import cycle.

## 4. Running it

```bash
pip install -e .                 # or: pip install .
spobs                            # a real AdamW run + the full numerical report
spobs studio                     # open the interactive local experiment studio
spobs inspect                    # environment, parameter identity, optimizer source audit
spobs trace --count 5            # the detailed per-step forensic trace
spobs live                       # the live dashboard (real-time, keyboard controlled)
spobs controls                   # headless: play / pause / step / n / reset / restart
spobs presets                    # list the presets
spobs selftest                   # the automated test suite
```

On Windows, use Python 3.12 for the environment verified in this release. With uv:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -e '.[dev,export]'
.venv\Scripts\python.exe -m single_parameter_observatory run --steps 40
.venv\Scripts\python.exe -m single_parameter_observatory studio
```

**Parameter Studio** provides a spatial experiment canvas, independent playback,
all configuration fields, exact-state branches and checkpoints, parameter/bit
and optimizer interventions, custom CSV datasets, historical records,
comparison charts, notes, autosave, portable projects, and numerical exports.
Its **Node lab** provides 61 built-in node types: 21 core operations and 40
Transformers/tensor operations. The visual editor supports
typed connections, editable literal inputs, multi-node selection, copy/paste,
undo/redo, reusable workflow selections, templates, an execution queue,
cancellation, observed outputs, and downloadable artifacts. **Plugins** adds
portable custom-node manifests, a node-authoring wizard, sandboxed JavaScript,
versioned registry updates, enable/disable controls, and declarative add-on
panels that produce editable Object nodes. The code editor and UI run locally;
there is no browser build step or cloud account.

Every training trajectory comes from the real PyTorch engine. The JavaScript
sandbox computes strict JSON without filesystem, network, DOM, or direct host
access. Explicit experiment nodes create, train, branch, checkpoint, inspect,
intervene in, control, verify, or export actual experiments. Saving a workflow
or installing a plugin does not execute it. Read the
[complete studio guide](STUDIO.md) for workflows, storage, and practical limits.

For many parameters, open **Transformers → + Build dense architecture** to create an
editable dense network, dataset, optimizer, training, inspection, and checkpoint
graph. The Transformers gallery also includes a tiny HF GPT-2 training and
generation workflow and an editable tensor-forward workflow. Choose **Queue
workflow** in Node lab to execute. AdamW connects to Train alongside the model
and data; RMSNorm is a forward layer inside the architecture. Actual losses,
parameter counts, tensor reports, and checkpoint downloads appear under
**Observed executions**. Read [TRANSFORMERS_STUDIO.md](TRANSFORMERS_STUDIO.md)
and the [runnable examples](examples/transformers/README.md) for complete graphs,
plugin-generated architectures, classification, tokenization, pipelines, and
cached pretrained model prerequisites.

To build Llama visually, choose **Transformers → + Build Llama from blocks**.
Set dimensions, heads, layers, corpus and training settings, choose an expanded,
fused or custom activation branch, and create the connected graph. Its operation
cards show formulas, colored tensor ports and inline settings; the inspector
exposes every parameter and custom expression. The supplied
[expanded component workflow](examples/transformers/llama-blocks.workflow.json)
and [custom component workflow](examples/transformers/llama-blocks-custom.workflow.json)
train, checkpoint, reconstruct and generate through the actual connected model.
Read [LLAMA_BLOCKS.md](LLAMA_BLOCKS.md) for editing and execution contracts.

The existing **Llama decoder training and architecture** workflow
([llama.workflow.json](examples/transformers/llama.workflow.json)) constructs a
native `LlamaForCausalLM` with two decoder layers and 78,144 parameters. Its
module inspector remains available alongside the new component editor. Both
examples start from new tiny weights, without Meta pretrained weights or
pretrained language ability. [LLAMA_STUDIO.md](LLAMA_STUDIO.md) retains that
native workflow's architecture and historical 2.1.0 measurements.

For a first node workflow, choose **Node lab → Templates → Sandbox playground**,
create the workflow, and choose **Queue workflow**. The actual script computes
`21 * 2`; its named output and Preview node report `42`. Choose **Training
laboratory** for configuration → real experiment → training → checkpoint →
observation. Existing experiment inputs can also be selected in a node's inspector.

The separately distributed source-and-evidence release is
`single_parameter_observatory_2.2.0.zip`.
The installable wheel is
`single_parameter_observatory-2.2.0-py3-none-any.whl`.
Release identities and archive verification belong to
`RELEASE-2.2.0.json`; current component execution results
belong to [LLAMA_BLOCKS.md](LLAMA_BLOCKS.md). The historical 2.0.0
qualification remains in [TRANSFORMERS_REPORT.md](TRANSFORMERS_REPORT.md).

The **2.2.0 source qualification** passed **414 Python tests**, all **11 execution
checks**, **51 graph-model tests** and **50 DOM workflow groups**. The component
builder's **13 DOM groups** also passed against each of two actual execution
receipts. Ten deep numerical tests cover the connected runtime, including
comparison of logits, causal loss and every trainable gradient with a matching
native HF Llama after copying identical weights. Runtime tests, mocked-controller
tests, actual browser actions and installed-distribution checks have separate
receipts under `evidence/blocks_20261023/`; their scope is described in
[LLAMA_BLOCKS.md](LLAMA_BLOCKS.md).

`python main.py <subcommand>` from inside the package directory is equivalent
(`main.py` bootstraps `sys.path` itself). See
[the usage guide](USAGE.md).

Add `--export runs --plot figures` to any run to write every artefact.
`--headless` runs the dashboard without a window, writing frames as PNGs.
`--fps` sets the render-rate ceiling (default 120). Achieved FPS is measured and
depends on the host, backend, diagnostic load and frame-saving frequency.
`--static-hz` and `--data-hz` control how often the text layer and the recorded series are re-rasterised: lowering them trades card freshness for frame rate.
`--frame-every` is 60 by default, because saving a frame costs a full text redraw and dumping every frame would throttle the view it is dumping.

The view is built in three layers because they cost wildly different amounts: the static text layer is cached and redrawn only when a step has been recorded, the recorded series are redrawn when a step arrives, and a marker moves along the newest recorded segment every frame. A marker position between two recorded steps is an interpolation and the figure says so on its face; only the integer-step points are recorded values.

The render loop and the engine share one interpreter, so a higher frame rate costs training throughput. The default is a balance.

Exit status follows the kind of command. `run` and `trace` report the *run*, so `run --preset diverging` exits 3 because the run reached a non-finite state and that is its result. `live` is a *view*, so it exits 0 when the dashboard ran and the frames were written, printing `NON-FINITE DETECTED` and `RUN ABORTED` when it was watching a run that diverged. A broken history chain or a validation failure still fails a view, because those are defects rather than observations.

Run the tests from the repository root:

```bash
python -m single_parameter_observatory selftest
python -m unittest discover -s single_parameter_observatory/tests -t single_parameter_observatory
```

Or run everything it checks   with one command that runs every check in `run_checks.py` and writes its complete log to `checks_output.txt`. The log is where to look for what those checks do; this README deliberately does not enumerate them, because a list here would fall out of step with the code the first time a check is added:

```bash
python single_parameter_observatory/run_checks.py
```

`run_checks.py` has no path that prints a passing summary that did not happen: it reproduces the test runner's own output verbatim and exits non-zero on any error.

## 5. The presets

Each preset is a real training configuration, executed by the same code path as any hand-written one. None of them loads a trajectory.

| preset | what it is for |
|---|---|
| `adamw` | the default: decoupled weight decay, moderate steps |
| `adamw_decay_only` | weight decay as the dominant force in the update |
| `adam` | coupled L2 weight decay inside Adam |
| `sgd` | the simplest real update, no state at all |
| `sgd_momentum` | SGD + Nesterov momentum |
| `small_gradient` | tiny inputs -> tiny gradients |
| `large_gradient` | large inputs and a distant start |
| `near_stationary` | starts exactly at the known optimum |
| `oscillatory` | a sign-alternating stream with an oversized learning rate |
| `fp32_resolution` | **candidate updates below one ULP: the optimizer attempts a move the stored float32 value cannot represent** |
| `diverging` | a learning rate that drives the real run to non-finite values (uses the linear model, because a saturated `tanh` kills the gradient and the run would stall at a large finite value instead) |

`fp32_resolution` is the one worth running first for the phenomenon the project is about. It produces steps where `stored_fp32_unchanged = True` **and** `update_attempted_but_storage_frozen = True`: the optimizer really did compute an update, and the parameter really did not move, and the record says so explicitly.

## 6. How the interesting parts work

### 6.1 IEEE-754 forensics (`fp32_inspector.py`)
Bit patterns are read out of the tensor's own memory (`view(torch.int32)`), then cross-checked against `struct` and against numpy; a disagreement raises rather than returning a plausible answer. Every decode is verified by re-encoding the decoded sign/exponent/fraction and comparing the raw word. ULP uses `torch.nextafter` and is cross-checked against the spacing implied by the exponent field alone. Exact ULP counts come from a monotonic ordering of the raw words, so "this step moved 3 ULPs" is a statement about stored values, not about a subtraction.

### 6.2 The optimizer reconstruction (`optimizer_inspector.py`)
The installed optimizer's semantics are **not** assumed. The module reads `torch/optim/adamw.py` from the installed package, records what the source actually contains, and then runs an isolated probe step on a throwaway parameter to see which update variant reproduces the real update bit for bit. The variants span the parts that have changed between PyTorch releases: the first moment updated with `lerp_` vs `mul_`/`add_`, decoupled decay applied before or after the adaptive step, and `eps` added outside vs inside the square root.

For each real step the reconstruction is recomputed and compared with the stored word; a mismatch beyond the configured tolerance is reported as `RECONSTRUCTION MISMATCH` in the record rather than passed over. The weight-decay contribution is not an algebraic split: the same update is reconstructed with decay on and decay off, and the difference between the two float32 results *is* the measured decay term, rounding included.

### 6.3 The float64 reference
Two separate things, both labelled as references rather than truth:
* a **parallel real float64 run** of the same forward shape with its own float64
  parameter and its own real optimizer, so the divergence between the float32 and
  float64 trajectories is a measured quantity;
* a **float64 reconstruction of each update** from the float32 before-state, whose
  difference from the stored float32 value is the rounding residual.

### 6.4 Diagnostic perturbations
The landscape probe never touches the live parameter: it evaluates a pure loss function that builds a throwaway tensor, so the parameter cannot be perturbed even in principle. The parameter's content hash is recorded before and after each probe, and the engine raises if they differ. A FAILING probe therefore cannot be mistaken for a passing one.

### 6.5 The live controls
`ExperimentController` owns the run state, and the real training runs on a worker thread it governs. `pause` stops the thread from stepping - it is not a rendering pause. The same controller drives the GUI dashboard and the headless terminal session, so both behave identically, and a test asserts that a paused engine really stops taking steps.

## 7. Scientific honesty

These statements are load-bearing, not decoration:
1. **A single parameter does not "know" anything.** Representations in a network are
   distributed; one coordinate means nothing on its own.
2. **Parameter, gradient, loss and optimizer state are four different tensors.**
   `exp_avg` and `exp_avg_sq` are memory *about* the update history, not copies of
   the parameter, and they keep evolving when the parameter barely moves.
3. **Forward and backward do not update parameters.** The snapshot chain (A before
   forward, B after forward, C after backward, D before `step()`, E after, F after
   the scheduler) is recorded per step and the engine raises if the parameter moves
   outside the optimizer step.
4. **A one-dimensional landscape plot is a diagnostic slice**, not the loss surface
   of a network, and says nothing about any other direction in weight space.
5. **Small updates do not prove convergence.** "Small update", "small gradient",
   "float32 storage stagnation", "oscillation" and "numerical plateau" are five
   different observations, and the analyser reports them separately.
6. **float32 has finite resolution.** Below one ULP an update cannot be stored,
   no matter how real the computation that produced it was.
7. **Optimizer state can keep changing while the stored parameter does not move**
   `m` and `v` are still updated every step.
8. **The regime classifier reports observed behaviour, never convergence.** Local
   evidence about one coordinate cannot establish a global property.
9. **Bit-for-bit reproducibility is not promised across hardware or PyTorch
   builds.** What *is* recorded, every run, is the exact Python, PyTorch, platform,
   optimizer configuration, model configuration and the measured optimizer semantics.

## 8. Known limitations

* The float64 track doubles the work of a run; disable it with `--no-fp64-reference`
  when only the float32 trajectory matters.
* The variant search is exhaustive over a documented variant list, not over all
  possible implementations. A PyTorch build whose update matches none of the variants
  would be reported as `RECONSTRUCTION MISMATCH` - loudly, but unhelpfully.
* The model is one-dimensional by construction. That is what makes the instrument
  legible; it is not a statement about real networks.
* Visual workflows compose bounded JSON transformations and explicit actions on
  the existing scalar engine. Plugin code uses isolated synchronous JavaScript;
  Python imports, arbitrary host programs, remote services, and third-party
  ComfyUI Python nodes are outside this plugin format. Workflow documents,
  queued execution snapshots, and actual numerical observations remain separate.
* `live` needs an interactive Matplotlib backend. Without one it renders frames to
  PNG instead of opening a window; it does not fake a GUI.
* Bit-level ULP analysis assumes IEEE-754 binary32 semantics. That holds for every
  CPU float32 PyTorch targets, but the code verifies the layout it finds rather than
  assuming it.

## 9. Historical verification of version 1.1.0

Everything in this section is measured on the machine that shipped this
`checks_output.txt`, and every figure below is copied from that log, with the
check that produced it named in parentheses. Where a claim comes from the test
suite rather than the check log, the test is named instead. Trajectory digits
are machine-specific (BLAS summation order differs between CPUs), which is
precisely why the log is shipped: regenerate it with `python run_checks.py`
and this section can be rebuilt from the output rather than trusted.

**Environment** (check 2, environment audit): Python 3.12.14, PyTorch 2.14.1+cpu,
NumPy 2.1.3, Matplotlib 3.9.2 on backend `agg`, Linux x86_64, CPU-only,
2 torch threads.

**Static gates** (not part of the 11 checks; run separately):
`ruff check single_parameter_observatory/` reports *All checks passed*, and
`mypy single_parameter_observatory/` reports *no issues found in 28 source
files*, both with the configuration in `pyproject.toml`.

**Test suite** (check 3, automated test suite): `Ran 128 tests` and `OK` -
128 passed, 0 failures, 0 errors, 0 skipped, in 15.0 s on this machine. The
122 tests of 1.0.0 plus the 6 regression tests added by the 1.1.0 audit
(`tests/test_regressions.py`, each named for the finding it pins).

**Measured update semantics** (check 4). The probe reproduces a real step bit
for bit on throwaway parameters, so the variant it names is a measurement of
the installed PyTorch rather than an assumption:

| optimizer | measured variant | bit-exact | candidates tried |
|---|---|---|---|
| AdamW | `adamw[lerp,before,outside]` | yes, 0 ULP | 12 |
| SGD + Nesterov | `sgd[clone_grad,coupled_into_grad]` | yes, 0 ULP | 4 |

`before` confirms this build applies decoupled weight decay with
`param.mul_(1 - lr * weight_decay)` *before* the adaptive step; `outside`
confirms `eps` is added after the `sqrt(v_hat)` division. Both on
PyTorch 2.14.1+cpu.

**Validation, 40-step AdamW run from w = -2.0** (check 4, validation ledger):
40/40 gradient checks, 40/40 finite-difference checks, 40/40 reconstructions,
1/1 bit check, 1/1 ULP check, and `total_failures : 0`. Reconstruction was
bit-exact on every step: `MAX RECONSTRUCTION ERROR (ULP) : 0`, with
`MAX ROUNDING RESIDUAL : 8.709889187663578e-08`.

The report gives both error forms, each named:

    MAX GRADIENT VALIDATION RELATIVE ERROR : 4.2979431768784344e-05
    MAX GRADIENT VALIDATION ABSOLUTE ERROR : 6.655366515673222e-07
    MAX FINITE-DIFFERENCE RELATIVE ERROR   : 4.2979899559960494e-05
    MAX FINITE-DIFFERENCE ABSOLUTE ERROR   : 6.652551425290198e-07

The two are the same disagreement in different forms. The relative figure is
large only because the gradient here is around 0.01 while the terms producing
it are around 4; the absolute form is the one to read against a tolerance.

**Why the tolerance is what it is.** The gradient check failed on every step
of the first run against a pure relative criterion of 1e-6. The cause is real,
not a bug: the gradient is a sum of terms of magnitude about 4 that cancel
down to about 0.01, so `err <= 1e-6 * |g|` demands an accuracy float32 cannot
deliver. The criterion now carries an absolute floor scaled by the loss,
`8 * eps32 * max(1, |loss|)`. The report prints the tightest bound actually
applied - `TIGHTEST GRADIENT TOLERANCE (absolute) : 9.558796445570145e-07` in
this run, against an observed maximum absolute error of 6.655366515673222e-07,
so the margin is about 1.4x rather than an order of magnitude. The rule is
written out in `validation.gradient_tolerance`, printed in the validation
output, and stored per step as `gradient_tolerance_abs`. Tightening the
relative number until it passed would have been choosing a constant to produce
the answer the author wanted.

**The headline phenomenon** (check 8, FP32-resolution preset, 60 steps):

    steps whose stored float32 did not change: 60
    steps where an update was attempted below one ULP: 60

The optimizer genuinely computes sub-ULP displacements and the stored float32
genuinely does not move, on every step of the preset. Each such step is labelled
both `stored_fp32_unchanged` and `update_attempted_but_storage_frozen` in the
exported trajectory.

Reporting that honestly required separating two quantities the first version
conflated. The *reconstructed float32* displacement is a difference of two
float32 values, so it can never show a sub-ULP move: it reads 0.0 whether or
not an update was attempted. Only the *attempted* float64 candidate can report
the phenomenon. They are separate fields, and every display that shows one says
which it is.

**Divergence detection** (check 9): the `diverging` preset reached a
non-finite state at step 2 and the run stopped - `aborted=True, steps
recorded=3, reason='non-finite state at step 2: the run was stopped rather
than continuing with meaningless values'`. A preset named for divergence
actually diverges, and the log shows it. (1.1.0 also fixed the export path
here: a diverged run's `summary.json` and `trajectory.json` previously failed
with a JSON `inf` error; they now export as strict JSON with the non-finite
values rendered as their `repr` strings - see AUDIT.md F-02.)

**The near-zero-loss guard** (check 10, a 103-step default-preset run). The
classifier's most important behaviour is that it refuses to call a large
*ratio* rise a divergence when the absolute change is tiny, because a loss
sitting near zero changes by a large factor without anything diverging. That
behaviour needs a long run to appear: the loss reaches its minimum at step 95
and then climbs back, and the guard first fires at 103 steps. Check 10 runs
exactly that and records it:

    minimum loss                     : 3.453934027675132e-09 at step 95
    final loss                       : 1.3817137869409635e-06
    loss ratio over the final window : 3.1915262426768134
    observed regime                  : 'still moving'

The classifier **refused** to call the ratio rise a divergence, and said so in
its own words:

> the loss ratio rose to 3.19 but the absolute increase is only
> 9.487818033449003e-07, below the divergence threshold 0.001: a near-zero loss
> changes by a large factor without anything diverging

That sentence is in the shipped log, and the check records what the classifier
did rather than requiring any particular regime - a run that ended differently
would still be a legitimate run, and the log would say so.

The step count was measured, not chosen: while surveying step counts for this
check, the ratio over the final window crossed the 1.05 threshold between 102
steps (0.951) and 103 (3.19). Those two boundary figures come from that survey
rather than from the shipped log, which runs the single 103-step configuration
above.

**A near-stationary run** (check 9): the `near_stationary` preset reports
`regime='oscillating'`, with `mean |gradient|=2.366117317365024e-05` and
`mean |update|=0.023431193828582764`. The update is roughly a thousand times
the gradient, because Adam divides by the square root of v. An earlier version
of the test expected "approximately stable" and failed; the expectation was
wrong, not the classifier. Small gradient and small update are separate
observations and are asserted separately, in
`tests/test_training.py::test_stability_reports_a_stationary_run`.

### What a run actually does

All figures below are from **check 4, the 40-step AdamW run whose report
appears verbatim in the shipped `checks_output.txt`**. It starts at w = -2.0,
takes 40 real steps, and ends here:

    INITIAL PARAMETER           : -2.0
    FINAL PARAMETER             : 1.31132972240448
    TOTAL PARAMETER CHANGE      : 3.31132972240448
    NUMBER OF STEPS             : 40
    MINIMUM LOSS                : 0.00016929797129705548
    INITIAL GRADIENT            : -0.010696807876229286
    HISTORY HASH CHAIN VERIFIED : True

The parameter ends at `1.31132972240448`, approaching the dataset's known
`w_true = 1.5`.

**On the ULP figures** (check 4, same report). The run contains one step that
crosses zero, step 20, and the two ULP metrics behave very differently there,
so both are printed:

    TOTAL ULP MOVEMENT (in ULPs of each step's start) : 120952652
    TOTAL REPRESENTABLE VALUES TRAVELLED              : 2141706663
      ... steps that crossed zero                     : [20]

The second number is the exact count of representable float32 values lying
between the before and after words of each step. It is correct, and it is not
a summary of how far the parameter travelled. The report says so in as many
words, and names step 20 as the one that crosses zero: a step whose endpoints
straddle zero must count every representable value in between, which passes
through the entire subnormal range. The first number divides each step's
displacement by the ULP at that step's starting value, which is why all 40
steps contribute on a comparable scale. Both are in the report, with the
sign-crossing step named.

**The live view frame rate** (check 11, measured over 6.04 s):

    target frame rate                   : 120 FPS
    frames drawn                        : 675
    achieved frame rate                 : 111.5 FPS
    real steps recorded during the view: 128
    background caching in use           : True

The target is a ceiling, not a guarantee: the periodic text redraw costs tens
of milliseconds, so the achieved rate runs below the target. The render loop
shares the interpreter with the engine, so a higher frame rate costs training
throughput. The default of 120 is a balance.

**Frame content, not just frame count** (check 9, new in 1.1.0). The original
check asserted only that the headless dashboard had written a frame; the
1.1.0 audit found that every frame the 1.0.0 renderer saved showed an empty
series panel (AUDIT.md F-01). Check 9 now dumps several frames and measures
the data ink in the series panel of each:

    multi-frame dump for the ink assertion: 3 frames
    lowest series-panel data-ink fraction across frames: 0.00000
    final frame data-ink fraction: 0.02523
    ok  the final frame's series panel contains visible data

The zero is the first frame, saved before the first training step exists - the
idle message, which is honest. The assertion is on the final frame, which is
rendered after a full refresh with every record present and is the frame that
was empty before the fix.

**Live dashboard** (check 9): `live --headless --steps 2 --speed 100` exited 0
and reported `frames rendered by the headless dashboard: 1`.

**Control session** (check 7): the command script `step, step, n 3, pause,
step, reset, step, restart, step` ran against the real engine and left exactly
one recorded step, because `restart` discards the history.
`tests/test_training.py::test_pause_really_stops_the_engine` additionally runs
the controller on a worker thread and asserts that a paused engine performs no
further steps.

**CLI surfaces** (check 9): `inspect`, `presets`, `controls` and
`live --headless` each exited 0 and printed their expected headers, and the
presets listing was checked to contain every name in `config.PRESETS`.

**Export** (check 5): `summary.json`, `trajectory.csv`, `trajectory.json`,
`trajectory.npz`, `trajectory.parquet`, `bit_history.csv`,
`optimizer_state_history.csv`, `stability.csv`, `landscape.json`, `events.csv`,
`validation.json` and `provenance.json` were all written, with byte sizes in
the log.

**Check ordering.** The log's section headers run `CHECK 1/11` through
`CHECK 11/11` in ascending order.

### Claims that are not observations

Two things in this README are design statements rather than measurements, and are labelled as such wherever they appear: that the tracked parameter is one coordinate of a high-dimensional state, and the decision to report `float32(0.1)` as `0.1` in `decimal_shortest_float32` while its `decimal_exact` is `0.100000001490116119384765625`. The second is asserted by `tests/test_fp32.py::test_exact_decimal_is_the_exact_expansion`.
