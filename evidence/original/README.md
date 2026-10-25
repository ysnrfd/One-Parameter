# Single float32 parameter observatory

A forensic instrument for the complete life cycle of **exactly one real, trainable `torch.float32` scalar parameter**: the bits it was initialised to, the graph it participates in, the gradient autograd actually produced, the optimizer state the *installed* PyTorch actually maintained, the bit pattern actually stored after each update, and the gap between all of that and an independently reconstructed account of the same update.

> **Execution status: verified.** The suite below was run on this machine:
> **118 tests, 118 passed, 0 failed, 0 errors, 0 skipped**, and all ten
> end-to-end checks completed without an error. See [§ 9 Verification status](#9-verification-status)
> for the measured numbers, which were produced by `python run_checks.py` and are
> recorded verbatim in `checks_output.txt`.

## 1. What this is, and what it is not

This is a **real training mechanism**: `loss.backward()` and `optimizer.step()`, on a real `nn.Parameter`, in a real autograd graph, on CPU, in float32.

This is **not** large-scale production LLM training, and it does not claim to be. The model is a deliberately tiny real PyTorch module (`y = o * tanh(s * w)`) because observing one scalar transparently requires an instrument small enough to expose completely.

What is real here: the parameter, the graph, the forward arithmetic, the loss, the gradient, the optimizer and its state, the float32 storage, the rounding, the bit transitions.

What is *not* being claimed: that a single scalar knows anything, or that this says anything about how a real network trains.

## 2. Design rules

| Rule | How it is enforced |
|---|---|
| No simulated training | The only gradient source is `loss.backward()`; the only writer of the parameter is `optimizer.step()`. A test greps `TrainingEngine.step` to confirm it contains no `.grad =`. |
| No fabricated values | Every recorded number is read out of a real tensor via `.item()`/`.detach()`; nothing is generated, smoothed or interpolated. |
| Raw != derived | `ParameterStepRecord` stores raw observations and derived analysis in separate fields; a derived number never overwrites the raw one. |
| Immutable history | Records are frozen dataclasses chained by SHA-256. `verify_chain()` reports the first step whose hash no longer matches. |
| No silent failure | Every failure mode has a named exception in `errors.py`. Invariant violations raise with the exact violated condition. |
| Version honesty | The installed optimizer's source is read and audited; the update order is *measured*, not assumed. |

## 3. Layout

    single_parameter_observatory/
      main.py                  CLI entry point (run / trace / live / controls / inspect / presets / selftest)
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
      README.md
      requirements.txt
      run_checks.py            runs every check and writes checks_output.txt
      tests/                   7 modules, standard-library unittest

`errors.py` is not in the original layout; it exists so every other module can share the error taxonomy without creating an import cycle.

## 4. Running it

```bash
pip install -r requirements.txt
python main.py                 # a real AdamW run + the full numerical report
python main.py inspect         # environment, parameter identity, optimizer source audit
python main.py trace --count 5 # the detailed per-step forensic trace
python main.py live            # the live dashboard (real-time, keyboard controlled)
python main.py controls        # headless: play / pause / step / n / reset / restart
python main.py presets         # list the presets
python main.py selftest        # the automated test suite
```

Add `--export runs --plot figures` to any run to write every artefact.
`--headless` runs the dashboard without a window, writing frames as PNGs.
`--fps` sets the live view target frame rate (default 120; 144 and 240 also work).
`--static-hz` and `--data-hz` control how often the text layer and the recorded series are re-rasterised: lowering them trades card freshness for frame rate.
`--frame-every` is 60 by default, because saving a frame costs a full text redraw and dumping every frame would throttle the view it is dumping.

The view is built in three layers because they cost wildly different amounts: the static text layer is cached and redrawn only when a step has been recorded, the recorded series are redrawn when a step arrives, and a marker moves along the newest recorded segment every frame. A marker position between two recorded steps is an interpolation and the figure says so on its face; only the integer-step points are recorded values.

The render loop and the engine share one interpreter, so a higher frame rate costs training throughput. The default is a balance.

Exit status follows the kind of command. `run` and `trace` report the *run*, so `run --preset diverging` exits 3 because the run reached a non-finite state and that is its result. `live` is a *view*, so it exits 0 when the dashboard ran and the frames were written, printing `NON-FINITE DETECTED` and `RUN ABORTED` when it was watching a run that diverged. A broken history chain or a validation failure still fails a view, because those are defects rather than observations.

The tests, three equivalent ways:

```bash
python main.py selftest
python -m unittest discover -s tests -t .
python -m unittest discover -s single_parameter_observatory/tests -t single_parameter_observatory
```

Or run everything it checks   with one command that runs every check in `run_checks.py` and writes its complete log to `checks_output.txt`. The log is where to look for what those checks do; this README deliberately does not enumerate them, because a list here would fall out of step with the code the first time a check is added:

```bash
python run_checks.py
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
* `live` needs an interactive Matplotlib backend. Without one it renders frames to
  PNG instead of opening a window; it does not fake a GUI.
* Bit-level ULP analysis assumes IEEE-754 binary32 semantics. That holds for every
  CPU float32 PyTorch targets, but the code verifies the layout it finds rather than
  assuming it.

## 9. Verification status

Everything in this section is measured on this machine. Every figure below is copied from the shipped `checks_output.txt`, and the check that produced it is named in parentheses. Where a claim comes from the test suite rather than the check log, the test is named instead.

**Environment** (check 2, environment audit): Python 3.14.3, PyTorch 2.14.0+cpu, NumPy 2.3.5, Matplotlib 3.11.2 on backend `tkagg`, Windows 11, CPU-only, 2 torch threads.

**Test suite** (check 3, automated test suite): `Ran 118 tests` and `OK`   118 passed, 0 failures, 0 errors, 0 skipped. The log records the suite's wall clock twice: once in the runner's own `Ran 118 tests in X.XXXs` line and once as `TESTS WALL CLOCK: Y.YYYs`. That figure moves by tens of seconds between runs on this machine, so this README deliberately does not pin a number for it   read it from the log. Every run observed during this work took well under two minutes.

**Measured update semantics** (check 4). The probe reproduces a real step bit for bit on throwaway parameters, so the variant it names is a measurement of the installed PyTorch rather than an assumption:

| optimizer | measured variant | bit-exact | candidates tried |
|---|---|---|---|
| AdamW | `adamw[lerp,before,outside]` | yes, 0 ULP | 12 |
| SGD + Nesterov | `sgd[clone_grad,coupled_into_grad]` | yes, 0 ULP | 4 |

`before` confirms this build applies decoupled weight decay with `param.mul_(1 - lr * weight_decay)` *before* the adaptive step; `outside` confirms `eps` is added after the `sqrt(v_hat)` division. Both on PyTorch 2.14.0+cpu.

**Validation, 40-step AdamW run from w = -2.0** (check 4, validation ledger): 40/40 gradient checks, 40/40 finite-difference checks, 40/40 reconstructions, 1/1 bit check, 1/1 ULP check, and `total_failures : 0`. Reconstruction was bit-exact on every step: `MAX RECONSTRUCTION ERROR (ULP) : 0`, with `MAX ROUNDING RESIDUAL : 1.2885659006478534e-07`.

The report gives both error forms, each named:

    MAX GRADIENT VALIDATION RELATIVE ERROR : 6.309028579778069e-05
    MAX GRADIENT VALIDATION ABSOLUTE ERROR : 6.749072461263023e-07
    MAX FINITE-DIFFERENCE RELATIVE ERROR   : 6.309075357954876e-05
    MAX FINITE-DIFFERENCE ABSOLUTE ERROR   : 6.749122505294441e-07

The two are the same disagreement in different forms. The relative figure is large only because the gradient here is around 0.01 while the terms producing it are around 4; the absolute form is the one to read against a tolerance.

**Why the tolerance is what it is.** The gradient check failed on every step of the first run against a pure relative criterion of 1e-6. The cause is real, not a bug: the gradient is a sum of terms of magnitude about 4 that cancel down to about 0.01, so `err <= 1e-6 * |g|` demands an accuracy float32 cannot deliver. The criterion now carries an absolute floor scaled by the loss, `8 * eps32 * max(1, |loss|)`. The report prints the tightest bound actually applied   `TIGHTEST GRADIENT TOLERANCE (absolute) : 9.558796445570145e-07` in this run, against an observed maximum absolute error of 6.749072461263023e-07, so the margin is about 1.4x rather than an order of magnitude. The rule is written out in `validation.gradient_tolerance`, printed in the validation output, and stored per step as `gradient_tolerance_abs`. Tightening the relative number until it passed would have been choosing a constant to produce the answer the author wanted.

**The headline phenomenon** (check 8, FP32-resolution preset, 60 steps):

    step 0: attempted (fp64) -8.858354050289563e-10, reconstructed (fp32) 0.0,
            observed stored delta 0.0, 1 ULP = 1.1920928955078125e-07,
            below-resolution=True

The optimizer genuinely computes a displacement of about 9e-10, which is about 0.0074 ULP (that ratio is arithmetic on the two logged figures, not itself a logged value), and the stored float32 value genuinely does not move. All 60 steps are labelled both `stored_fp32_unchanged` and `update_attempted_but_storage_frozen`.

Reporting that honestly required separating two quantities the first version conflated. The *reconstructed float32* displacement is a difference of two float32 values, so it can never show a sub-ULP move: it reads 0.0 whether or not an update was attempted. Only the *attempted* float64 candidate can report the phenomenon. They are now separate fields, and every display that shows one says which it is.

**Divergence detection** (check 9): the `diverging` preset reached a non-finite state at step 2 and the run stopped   `aborted=True, steps recorded=3, reason='non-finite state at step 2: the run was stopped rather than continuing with meaningless values'`. A preset named for divergence actually diverges, and the log shows it.

**The near-zero-loss guard** (check 10, a 103-step default-preset run). The classifier's most important behaviour is that it refuses to call a large *ratio* rise a divergence when the absolute change is tiny, because a loss sitting near zero changes by a large factor without anything diverging. That behaviour needs a long run to appear: the loss reaches its minimum at step 95 and then climbs back, and the guard first fires at 103 steps. Check 10 runs exactly that and records it:

    minimum loss                     : 3.460943531763405e-09 at step 95
    final loss                       : 1.3815736110700527e-06
    loss ratio over the final window : 3.1912024600129723
    observed regime                  : 'still moving'

The classifier **refused** to call the ratio rise a divergence, and said so in its own words:

> the loss ratio rose to 3.19 but the absolute increase is only
> 9.486416274739895e-07, below the divergence threshold 0.001: a near-zero loss
> changes by a large factor without anything diverging

That sentence is in the shipped log, and the check records what the classifier did rather than requiring any particular regime - a run that ended differently would still be a legitimate run, and the log would say so.

The step count was measured, not chosen: while surveying step counts for this check, the ratio over the final window crossed the 1.05 threshold between 102 steps (0.951) and 103 (3.19). Those two boundary figures come from that survey rather than from the shipped log, which runs the single 103-step configuration above.

**A near-stationary run** (check 9): the `near_stationary` preset reports `regime='oscillating'`, with `mean |gradient|=2.3661613874992325e-05` and `mean |update|=0.023431450128555298`. The update is roughly a thousand times the gradient, because Adam divides by the square root of v. An earlier version of the test expected "approximately stable" and failed; the expectation was wrong, not the classifier. Small gradient and small update are separate observations and are asserted separately, in `tests/test_training.py::test_stability_reports_a_stationary_run`.

### What a run actually does

All figures below are from **check 4, the 40-step AdamW run whose report appears verbatim in the shipped `checks_output.txt`**. It starts at w = -2.0, takes 40 real steps, and ends here:

    INITIAL PARAMETER           : -2.0
    FINAL PARAMETER             : 1.31132972240448
    TOTAL PARAMETER CHANGE      : 3.31132972240448
    NUMBER OF STEPS             : 40
    INITIAL LOSS                : 3.977567672729492
    FINAL LOSS                  : 0.00016929797129705548
    MINIMUM LOSS                : 0.00016929797129705548
    INITIAL GRADIENT            : -0.010696807876229286
    FINAL GRADIENT              : -0.0022053252905607224
    MAX ABS GRADIENT            : 4.351048469543457
    MAX ABS UPDATE              : 0.11175964400172234
    HISTORY HASH CHAIN VERIFIED : True

The parameter ends at `1.31132972240448`, approaching the dataset's known `w_true = 1.5`, and the loss has fallen from 3.977567672729492 to 0.00016929797129705548.

**On the ULP figures** (check 4, same report). The run contains one step that crosses zero, step 20, and the two ULP metrics behave very differently there, so both are printed:

    TOTAL ULP MOVEMENT (in ULPs of each step's start) : 121380477
      ... contributed by this many steps              : 40
    TOTAL REPRESENTABLE VALUES TRAVELLED              : 2141706663
      ... steps that crossed zero                     : [20]

The second number is the exact count of representable float32 values lying between the before and after words of each step. It is correct, and it is not a summary of how far the parameter travelled. The report says so in as many words, and names step 20 as the one that crosses zero: a step whose endpoints straddle zero must count every representable value in between, which passes through the entire subnormal range. The first number divides each step's displacement by the ULP at that step's starting value, which is why all 40 steps contribute on a comparable scale. Both are in the report, with the sign-crossing step named.

**The live view frame rate** (check 11, measured over 6.13 s):

    target frame rate                   : 120 FPS
    frames drawn                        : 631
    achieved frame rate                 : 103.0 FPS
    real steps recorded during the view: 41
    background caching in use           : True

The view used to update about once a second because a full redraw cost roughly 1.8 seconds on this machine - almost all of it text rasterisation - and the redraw was triggered by each arriving training step. It is now three layers: a cached text layer, the recorded series, and a marker that moves every frame over the top two. The figure's window title and its corner readout both show the achieved rate while it runs, and `--fps 144` or `--fps 240` reach 132 and 207 FPS respectively on this machine.

Two honest limits. The target is a ceiling, not a guarantee: the periodic text redraw costs about 93 ms, so the achieved rate runs at roughly 85-90% of the target. And the render loop shares the interpreter with the engine, so a higher frame rate costs training throughput - at 240 FPS the engine records only a few steps in six seconds. The default of 120 is a balance between the two.

**Live dashboard** (check 9): `main.py live --headless --steps 2 --speed 100` exited 0 and reported `frames rendered by the headless dashboard: 1`.

**Control session** (check 7): the command script `step, step, n 3, pause, step, reset, step, restart, step` ran against the real engine and left exactly one recorded step, because `restart` discards the history. `tests/test_training.py::test_pause_really_stops_the_engine` additionally runs the controller on a worker thread and asserts that a paused engine performs no further steps.

**CLI surfaces** (check 9): `inspect`, `presets`, `controls` and `live --headless` each exited 0 and printed their expected headers, and the presets listing was checked to contain every name in `config.PRESETS`.

**Export** (check 5): `summary.json`, `trajectory.csv`, `trajectory.json`, `trajectory.npz`, `trajectory.parquet`, `bit_history.csv`, `optimizer_state_history.csv`, `stability.csv`, `landscape.json`, `events.csv`, `validation.json` and `provenance.json` were all written, with byte sizes in the log.

**Check ordering.** The log's section headers run `CHECK 1/11` through `CHECK 11/11` in ascending order. An earlier version numbered the export check 8/8 but ran it inside check 4, so the printed order was 1, 2, 3, 4, 8, 5, 6, 7. Export is now its own check, positioned fifth, and the numbering matches the execution order.

### Claims that are not observations

Two things in this README are design statements rather than measurements, and are labelled as such wherever they appear: that the tracked parameter is one coordinate of a high-dimensional state, and the decision to report `float32(0.1)` as `0.1` in `decimal_shortest_float32` while its `decimal_exact` is `0.100000001490116119384765625`. The second is asserted by `tests/test_fp32.py::test_exact_decimal_is_the_exact_expansion`.
