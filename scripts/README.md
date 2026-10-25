# Current verification tools and historical audit scripts

`run_preset_sweep.py` verifies the **current checkout**. It resolves paths from
the script location, runs on Windows or Linux, exports each requested preset,
checks the saved JSON hash chain, CSV parameter values and NPZ parameter words,
and records SHA-256 hashes of source files and delivered artifacts. The summary
uses strict JSON; nonfinite observations remain explicit strings. Required
export failures and numerical validation failures make the process exit with
status 1. Parquet is optional when its dependencies are unavailable.

Run from the checkout with its Python environment:

```powershell
& .venv\Scripts\python.exe scripts\run_preset_sweep.py
```

The default output is `evidence/repair_20261023/preset_sweep`. An output directory
must be empty or new; use `--output PATH` for another run. Existing audit evidence
is never replaced. `--presets adamw diverging --steps 3` performs an explicitly
identified smoke run. Omit `--steps` for the presets' full configured lengths.
`--figures` renders saved plots, and `--gui --gui-seconds 5` records a headless
dashboard session. GUI frame production alone is not a visual acceptance claim.
If source files change during a sweep, its summary records this and fails so a
mixed checkout cannot be presented as one verified revision.

`package_delivery.py` is the current portable package builder. Consult its
`--help` output for supported options.

The following Node Lab verification tools are also current:

| Tool | What it actually verifies |
| --- | --- |
| `verify_studio_install.py` | The installed package, real HTTP routes, real training and V8 execution, cancellation, project reopening, exact artifact bytes, all seven installed workflow examples, and optional wheel/sdist/source byte identity. Run it with the installed environment's Python and `-I`; supply `--output PATH` and `--distributions PATH`. The installed environment must already contain the runtime dependencies. |
| `export_studio_fixtures.py` | Creates backend-generated read fixtures in an isolated in-memory workspace. `--steps 2` performs two real training steps; `--with-example-plugin` installs the example manifest without executing its scripts. It does not edit the user's saved studio. |
| `test_graph_ui.mjs` | Pure graph model, strict JSON, exact-number transport, editing and history regressions. Run with `node scripts/test_graph_ui.mjs`. |
| `test_studio_dom.mjs` | Offline controller workflows using actual read fixtures and deterministic mocked HTTP mutations. Requires a separately installed JSDOM and the `--fixtures`, `--jsdom`, and `--output` arguments. It does not verify browser rendering, pointer hit testing, real HTTP transport, training, or sandbox execution. |

For example, after installing JSDOM 26.1.0 in a separate development directory:

```powershell
& .venv\Scripts\python.exe scripts\export_studio_fixtures.py --steps 2 --with-example-plugin --output evidence\local-ui\fixtures.json
node scripts/test_studio_dom.mjs --fixtures evidence/local-ui/fixtures.json --jsdom "C:/path/to/node_modules/jsdom/lib/api.js" --output evidence/local-ui/dom.json
```

JSDOM and Node.js are development tools, not studio runtime dependencies.

The scripts in the table below are retained as **historical audit recipes**, not current
portable commands. Their original inputs, results, report wording, version
labels and environmental assumptions belong to the earlier audit. They are not
automatically run by package installation, the test suite, or the current sweep.
Do not run them against current source or use historical measurements as new
verification evidence:

| Historical script group | Original dependencies and limitations |
| --- | --- |
| `_run_preset_sweep_enterprise.py` | Linux checkout at `/home/z/my-project`; writes earlier enterprise evidence locations. Use the current sweep instead. |
| `diagnose_series_bug.py`, `probe_diverging_export.py`, `pixel_analyze_frames.py` | Fixed Linux paths and assumptions about earlier defects, recorded frames, and dashboard layouts. Current regression tests cover corresponding repaired behavior. |
| `fix_test_typing.py` | One-time source rewrite for an earlier test suite. It is not idempotent for already annotated imports and must not be rerun on current tests. |
| `make_audit_charts.py`, `make_frame_compare.py` | Fixed historical measurements or frame names, Linux DejaVu font paths, and fixed report output locations. Rebuilding them does not rerun measurements. |
| `build_audit_pdf.py` | ReportLab, Pillow, original source images, Linux Noto/FreeSerif/DejaVu fonts, and external `skills/pdf/scripts/pdf.py` helper providing `install_font_fallback`. That helper is absent from this repository. Report prose contains historical claims and version numbers. |
| `merge_audit_pdf.py`, `rasterize_audit_pdf.py` | Original cover/body PDFs and Linux paths; require `pypdf` or `pypdfium2` respectively. |
| `vlm_verify_*.sh`, `vlm_consolidate.py` | Bash, original Linux frame/report paths and external image-review service tooling. Service and original credentials are not repository dependencies. |

Existing `AUDIT.md`, audit PDF, historical `evidence/` trees and archived source
snapshots remain historical records. New repair evidence belongs in its own
dated directory and must identify the code and environment it actually ran.
