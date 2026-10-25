# Contributing

## The one rule

Every displayed value must be a value a real PyTorch run produced. If a change
introduces a number that was computed to *look* right rather than read out of a
tensor, it does not belong here. The tests grep the engine source for exactly
this class of sin (`test_gradient_is_never_fabricated`), and grepping source
text is deliberate: it pins the invariant to the code, not to a review.

## Setup

```bash
git clone <repository> && cd single-parameter-observatory
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev,export]
```

## Before you push

All three gates must be green, in this order:

```bash
ruff check single_parameter_observatory/
mypy single_parameter_observatory/ --config-file pyproject.toml
python -m unittest discover -s single_parameter_observatory/tests \
                             -t single_parameter_observatory
```

Then the end-to-end checks (a few minutes):

```bash
cd single_parameter_observatory && python run_checks.py
```

Exit code 0 means all 11 checks completed without an error; the full log lands
in `checks_output.txt`. CI runs the same gates on every push.

## What the checks protect

1. **The A-F snapshot chain.** The parameter's raw word is read at six
   labelled instants per step; the engine raises if it moves anywhere other
   than inside `optimizer.step()`. Never insert a parameter write between the
   labels.
2. **The hash chain.** Records are frozen and SHA-256-linked; `verify_chain()`
   must stay able to name the first tampered step. If you add a field to
   `ParameterStepRecord`, it joins the hash payload automatically - never
   populate it with a non-observed value.
3. **The reconstruction contract.** The variant search measures the installed
   PyTorch; do not pin a variant by name as an optimisation without keeping
   the measured path.
4. **The renderer ink.** A saved dashboard frame must contain visible data in
   its series panel (check 9 asserts on the pixels). If you touch
   `_setup_series_axes` / `_reattach_animated`, run the frame-ink test first.
5. **Strict JSON.** Every JSON export goes through `recorder.json_safe`;
   `allow_nan=False` stays.

## Style

The code is hand-laid-out on purpose: aligned value tables, comment towers
that carry audit rationale, and long lines where splitting would obscure a
comparison. Formatting is therefore NOT enforced; ruff runs with `E501`
ignored and no formatter is applied. Keep diffs surgical - reflowing a block
you did not otherwise change breaks the source-grep tests and buries the
review.

Line length is 110 where it matters; new code should read like the code
around it.

## Adding a check

Add it to `run_checks.py` in execution order, bump `CHECK_COUNT`, and make it
report what it observed rather than what it hoped for. A check that cannot
fail is not a check. If it produces numbers worth quoting in the README,
regenerate the README's verification section from the new log rather than
editing numbers by hand - the section's whole value is that it is copied, not
remembered.

## Reporting findings

The project's own audit trail (`AUDIT.md`, `CHANGELOG.md`) is written the way
the instrument records steps: what was wrong, what the evidence was, what the
fix is. Bug reports in the same shape get fixed faster.
