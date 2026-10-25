# Node Lab and studio repair qualification — 1.3.0

Date: 2026-10-23. Scope: current Single Parameter Studio, its eight views, Node Lab, declarative plugins and isolated JavaScript execution. This report records observed checks and their limits; it is not a claim that every possible input or future plugin is defect-free.

## Implemented behavior

Node Lab has 21 built-in types, typed ports and editable literals, DAG validation, a serial execution queue, immutable graph/code/version fingerprints, explicit cancellation, per-node observations, logs and errors. Its host nodes create and train real experiments, inspect records, branch, checkpoint, intervene, verify, control transport and export exact evidence ZIP bytes. Plugin manifests add namespaced typed JavaScript nodes, templates and form panels. Workflows, plugin definitions, jobs, artifacts and unapplied editor drafts travel with projects.

The editor supports palette search, node selection and movement, click/drag connections and an accessible connection form, copy/paste, duplicate, undo/redo, pan/zoom/fit, templates, workflow JSON import/export, reusable selections, execution history and result downloads. The plugin interface includes a node wizard, manifest editor, validation, install/update, enable/disable/remove and portable manifest downloads.

## Repaired defects

- Custom dataset creation now validates and materializes the actual tensors before publishing an experiment; invalid inputs leave no partial experiment.
- Automatic branches and sibling experiments avoid stacking at the same position.
- Current parameter interventions, optimizer memory and next-step learning rates are distinct from historical records; reset clears stale intervention displays. Float32 inspection uses the actual uint32 storage word, including signed zero, infinities and distinct NaN payloads.
- Configuration, CSV, notes, optimizer, bit, plugin, node and add-on panel draft buffers persist without implicitly applying or executing them. Invalid drafts remain editable.
- Draft writes use revision checks. Conflicting tabs retain their local buffers and require an explicit choice before replacement. Keeping a conflicted graph as a new workflow also preserves invalid raw fields. Workspace epoch headers reject stale reads, downloads and writes after another tab replaces the project.
- Imported CSV and programmatically authored plugin text now mark drafts dirty.
- Invalid node edits no longer block cancellation or plugin management.
- A strict lossless JSON codec preserves wide integers and negative zero through graph import/export, clipboard operations, undo/redo, cache cloning and imported drafts. Invalid seed syntax or unsafe floating seed values fail explicitly.
- Scripts reject input integers that would silently round in JavaScript. Wide integer job downloads use server JSON directly, preserving exact integer tokens.
- Plugin implementation changes require a changed version, and queued jobs retain their admitted code and version.
- Failed or cancelled workflows preserve completed real host effects and skip later work; no automatic retries or project-open execution occur.
- Saved jobs reject false success metadata, modified immutable snapshots, duplicate replay keys and inconsistent observations.
- Evidence artifacts retain their exact bytes after later training and project reopening, with validated owner, length, MIME and SHA-256.
- Accepted artifact identifiers have matching HTTP routes.
- Project replacement is blocked while jobs are active or shutdown has begun. Shutdown attempts remaining cleanup even if graph shutdown or persistence fails.
- Corrupt autosaves are preserved as exact recovery copies before replacement; an inability to preserve them blocks overwrite.
- Windows scripts launch the actual CPython worker in a resource job. Trusted V8 startup is separate from evaluation limits. A dedicated I/O thread keeps input-pipe writes from blocking cancellation; stderr is drained from process creation so startup diagnostics cannot deadlock readiness.
- The health endpoint reports the actual running version, allowing the launcher to reject an older backend.
- The release checker now evaluates the guaranteed final frame by its pixels independently of the timing-dependent count of intermediate paints. Missing and dataless output still fail.

## Sandbox and scope

JavaScript runs in a fresh disposable process and V8 isolate, without a host filesystem, network, environment, DOM, Python, Node.js or timer bridge. On Windows the worker has one-process containment, kill on job close and a 256 MiB committed-memory cap. V8 heap limits are 16–64 MiB and evaluation limits are 10–10,000 ms. Startup has a separate bounded allowance. This is a local resource and capability boundary, not an operating-system virtual machine; high-security hostile-code deployments require a stronger surrounding boundary. See [PLUGIN_SDK.md](PLUGIN_SDK.md) for exact budgets and authoring rules.

This is an independently authored node editor for the observatory's supported engine and data operations. ComfyUI is a design reference; ComfyUI Python nodes and extensions are not a compatible plugin format.

## Verification status

Evidence paths below are relative to the source/evidence delivery. The wheel includes the runtime, guides and portable examples.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Complete backend unittest suite | 355 passed, 0 failures, 0 errors, 0 skipped | evidence/node_lab_20261023/run_checks.txt |
| Execution checks | All 11 completed without an error, including the unittest suite, real training, exports, forensic trace, transport controls, presets, headless rendering and measured FPS | evidence/node_lab_20261023/run_checks.txt |
| Graph model regressions | 32 passed | evidence/node_lab_20261023/graph_model_checks.txt |
| Offline DOM workflow groups | 32 passed, zero unexpected controller errors | evidence/node_lab_20261023/dom_checks.json |
| Production JavaScript syntax and hashes | 3 scripts passed | evidence/node_lab_20261023/frontend_checks.json |
| Delivery tooling regressions | 3 passed | evidence/node_lab_20261023/delivery_tests.txt |
| Python static checks | Maintained package/scripts pass Ruff; mypy reports no issues in 44 source files | evidence/node_lab_20261023/ruff_final.txt and mypy_final.txt |

The final measured standalone Matplotlib view rate was 33.3 FPS over 4.17 seconds, with 139 frames drawn and 72 real steps recorded. This is an observation on this host, not a frame-rate guarantee or a measurement of the web node canvas. The final saved-frame ink check and a separate saved-frame visual review verify visible data rather than merely file existence.

The offline DOM suite uses JSDOM, actual backend-generated read fixtures, and deterministic mocked HTTP mutations. It covers all eight views, experiment controls, records, configuration, CSV and bit drafts, graph editing, script/plugin controls, exact-number downloads, revision ordering, both conflict choices, project Open races and stale-tab recovery. It does not establish real-browser rendering, pointer hit testing, responsive layout, native download UI acceptance, actual training or V8 execution; the latter two are checked separately by real backend tests.

The installed-release verifier in scripts/verify_studio_install.py exercises actual HTTP, training and V8, executes all seven installed workflow examples unchanged, checks cancellation and reopening, validates exact ZIP bytes, and compares wheel, installed files and sdist with the final source. Its generated results and distribution hashes are recorded separately in evidence/node_lab_20261023/installed_wheel.json. The installed environment reuses the existing dependency runtime while loading the studio package itself from the installed wheel. The source/evidence ZIP has a complete member-size/SHA-256 manifest verified by scripts/package_delivery.py.

Earlier diagnostics remain visible. The 347-test development run had three failures corrected before the final suite. A later heavily paged 353-test run had one bounded V8 startup failure and a timing-dependent frame-count oracle failure; the isolated plugin rerun passed without changing the 30-second startup limit, and two actual-PNG regressions now enforce the corrected frame oracle. A misplaced new health assertion in a shared test helper was corrected before the final clean run. None of these intermediate logs is used as the final pass.

## Real-browser limitation

Repeated browser attachment attempts initially timed out. A later inventory call recovered, but the existing tab remained an internal data-protocol connection-refused error page. The browser security gate explicitly rejected attaching to that page and prohibited workarounds. After the user confirmed opening the test URL, inventory still reported the internal error page, although direct local HTTP health checks succeeded. No alternate browser surface or policy bypass was used. Fresh real-browser rendering, pointer gestures, responsive layout and browser download acceptance remain unverified. See evidence/node_lab_20261023/browser_connection.json. The old 1.2.0 screenshots are historical and are not 1.3.0 browser evidence.

## References

- [ComfyUI project](https://github.com/comfy-org/comfyui)
- [MiniRacer API](https://bpcreech.com/PyMiniRacer/api/)
- [MiniRacer architecture](https://bpcreech.com/PyMiniRacer/architecture/)
