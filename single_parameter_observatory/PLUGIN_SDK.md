# Plugin SDK and custom-node authoring

This guide documents Parameter Studio's declarative extension format and local graph API. A plugin can supply typed custom JavaScript nodes, workflow templates, and safe GUI control panels. A plugin is JSON data; installation does not import Python or execute JavaScript.

Start with [NODE_LAB.md](NODE_LAB.md) for the canvas and execution model. The runnable [example collection](examples/node_lab/README.md) contains a plugin produced directly by `example_plugin()` and built-in workflow templates produced by the current catalog.

Parameter Studio uses an original `spobs-plugin/1` manifest and `spobs-graphs/1` saved-state format. The [ComfyUI project](https://github.com/comfy-org/ComfyUI) is a visual-workflow reference. This SDK does not claim compatibility with ComfyUI workflows, Python node classes, or extension APIs.

## Create and use a custom node

In **Plugins**, choose **Create custom node**. In Node Lab the equivalent entry is **Build custom node**. Enter a plugin ID, node ID/title, JSON port and parameter declarations, and a JavaScript function body. **Build plugin manifest** opens the complete manifest for review; choose **Install / update plugin** to commit it to the registry.

Alternatively, import a `.plugin.json` file or paste it into **Plugin manifest JSON**. **Load sample manifest** supplies an editable built-in UI example. The downloadable `example.plugin.json` is a separate, richer SDK example with a ready-made workflow template.

After installation, enabled custom node types appear immediately in the Node Lab library. Add one, set its literal inputs/parameters or connect upstream nodes, and queue a workflow to execute it.

## Minimal complete manifest

```json
{
  "schema_version": "spobs-plugin/1",
  "id": "local.my-tools",
  "name": "My tools",
  "version": "1.0.0",
  "description": "A reusable numerical transform.",
  "nodes": [
    {
      "id": "scale",
      "title": "Scale a number",
      "category": "My tools",
      "description": "Multiply an input by an editable factor.",
      "inputs": [{"name": "value", "type": "number", "required": true}],
      "outputs": [{"name": "value", "type": "number"}],
      "parameters": [
        {"name": "factor", "type": "number", "default": 2, "min": -100, "max": 100}
      ],
      "code": "console.log('Scaling', inputs.value); return {value: inputs.value * params.factor};"
    }
  ],
  "templates": [],
  "panels": []
}
```

The catalog type for this node is **`local.my-tools/scale`**. IDs are namespaced by their plugin, so another plugin can also define a local node named `scale`.

Unknown manifest/schema fields are rejected. For example, `python`, `permissions`, `html`, browser `onclick` code, and arbitrary package-import declarations do not grant capabilities.

## Manifest and node schema

| Manifest field | Contract |
| --- | --- |
| `schema_version` | Required exact string `spobs-plugin/1`. |
| `id` | Required ASCII identifier beginning with a letter; remaining characters may be letters, digits, `.`, `_`, or `-`; maximum 96 characters. |
| `name` | Required nonempty text, maximum 256 characters. |
| `version` | Required nonempty version text, maximum 64 characters. Semantic versions such as `1.0.0` are recommended. |
| `description` | Optional text, maximum 8,000 characters. |
| `nodes` | Up to 64 custom-node definitions. Defaults to an empty array. |
| `templates` | Up to 16 `{id,name,graph}` entries. Defaults to an empty array. |
| `panels` | Up to 16 declarative control panels. Defaults to an empty array. |

At least one node, template, or panel is required. A complete standalone manifest is limited to 4 MiB of the validator's encoded JSON; the installed registry also has its stricter aggregate 4,000,000-byte budget and 64-plugin limit.

Each node accepts exactly these fields:

| Node field | Contract |
| --- | --- |
| `id` | Required safe ASCII identifier with the same identifier rule as plugin IDs; unique within this plugin. |
| `title` | Required nonempty text, maximum 256 characters. |
| `category` | Optional nonempty library category, default `Custom`; maximum 256 characters. |
| `description` | Optional text, maximum 8,000 characters. |
| `inputs` | Array of up to 64 input declarations; default `[]`. |
| `outputs` | Array of up to 64 output declarations; default `[]`. |
| `parameters` | Array of up to 64 parameter declarations; default `[]`. |
| `code` | Required JavaScript function body, at most 65,536 UTF-8 bytes. |

IDs `__proto__`, `prototype`, and `constructor` are reserved. Port and parameter names use a narrower rule: start with an ASCII letter or `_`, continue with ASCII letters/digits/`_`, and stay within 64 characters. Names must be unique within each declaration array. Use distinct input and parameter names to keep their meanings clear.

### Port declarations and types

An input declaration is `{name,type,description?,required?,default?}`. `required` defaults to true. A default satisfies an unwired input. An optional input with no default and no supplied value is omitted from `inputs`.

An output declaration is `{name,type,description?}`. Custom node code must return **exactly** the declared output names. Returning an extra name or omitting one fails that node.

| Type | JSON value |
| --- | --- |
| `any` | Any supported strict JSON value. |
| `number` | A finite number; booleans do not count as numbers. |
| `integer` | An integer; booleans do not count as integers. |
| `string` | Text. |
| `boolean` | `true` or `false`. |
| `object` | A JSON object. |
| `array` | A JSON array. |
| `run` | An opaque experiment-ID string referring to a run in this workspace when consumed by a host node. |
| `config` | An experiment-configuration object; native host validation determines valid settings. |
| `dataset` | An object with matching finite numeric `inputs` and `targets` arrays; supported metadata includes `name`, `batch_size`, and `provenance`. |
| `tensor-expression` | A JSON object tagged `kind: "tensor-expression"` containing the connected tensor recipe. Use it to author or pass through editable model operations; the compiler validates the complete graph before allocation. |

The graph rechecks actual values at execution. For example, a script may declare a number output but return a string; that is a failed execution, not an implicit numeric conversion.

### Parameter declarations

Parameters accept `{name,type,description?,default?,min?,max?,options?}`. Their type uses the types above. `min` and `max` apply to numeric/integer parameters, must be finite, and must form a valid range. Integer bounds must be integers. `options` is a nonempty array of up to 256 typed choices. A declared default must satisfy its type, bounds, and options.

A parameter without a default must be supplied before executing a node that requires it. In code, `params` contains declared parameters; input-port literals remain in `inputs`, rather than being duplicated into `params`.

The built-in `core.script` additionally uses the inspector's special `code` editor type. For custom plugin declarations, source belongs in the node's `code` field; a user-editable textual parameter can use `string`.

The runtime uses `params.timeout_ms` and `params.memory_mb` as per-invocation execution limits when a plugin declares and supplies these names. Their values must still fit the sandbox's enforced ranges. Reserve those names for execution budgets rather than unrelated numerical data.

## Sandbox contract

Code is a synchronous function body with three arguments:

```javascript
// inputs: this node's resolved named input-port values
// params: this node's declared parameter values
// console: bounded log/info/warn/error methods
console.log("Input length", inputs.values.length);
return {value: inputs.values.reduce((sum, item) => sum + item, 0)};
```

Return a plain JSON object. Its nested values may be null, strings, booleans, finite numbers, plain objects, or dense arrays. Functions, undefined, symbols, BigInt values, Promises, dates, cyclic objects, accessors/getters, hidden properties, and sparse arrays are rejected as outputs. Convert an intentional BigInt result to a decimal string.

Each invocation gets a fresh V8 isolate in a disposable worker process. Mutation of `inputs`, global variables, or local objects cannot mutate parent Python objects or persist into the next invocation. Some JavaScript intrinsics/prototypes are frozen to protect the output boundary; prototype modification is not a supported customization mechanism.

There are no Python callbacks or exposed host bindings. `process`, `require`, `fetch`, the DOM, filesystem APIs, and network APIs are unavailable. Timers, WebAssembly, and typed-array/ArrayBuffer constructors are disabled. Asynchronous jobs/Promises are not a supported output or scheduling mechanism. Use normal finite synchronous JavaScript loops and data transforms.

On Windows the trusted parent launches the actual CPython interpreter with `-I -S`, a minimal environment, and a separate temporary working directory. This avoids virtual-environment launcher children and skips `.pth`/`sitecustomize` execution in the worker. A Job Object enforces one process, a 256 MiB committed-process memory cap, and kill-on-close. V8 has its own heap and evaluation limits. Cancellation kills and reaps the real worker.

This is process and V8 isolation, **not an operating-system virtual machine**. It does not claim the stronger boundary of a purpose-built container or VM. The official [MiniRacer architecture and security goals](https://bpcreech.com/PyMiniRacer/architecture/) explain the upstream isolation model and limitations; the [API reference](https://bpcreech.com/PyMiniRacer/api/) documents the V8 memory and timeout controls.

### Actual limits

| Boundary | Enforced value |
| --- | --- |
| Source | 65,536 UTF-8 bytes. |
| Combined inputs and parameters | 1,048,576 encoded JSON bytes. |
| Worker response | 1,048,576 bytes, including output metadata/logs. |
| JSON nesting | Maximum depth 64 inside the script boundary; graph JSON uses a shallower maximum of 50. |
| Output structure | At most 500,000 visited values; each array has at most 100,000 entries; the byte budget still applies. |
| V8 evaluation | Default 1,000 ms; configurable integer 10–10,000 ms. |
| V8 heap | Default 32 MiB; configurable integer 16–64 MiB. |
| Trusted runtime startup | Separate maximum 30,000 ms, before untrusted input is sent; runtime readiness is explicit. |
| Parent execution/transfer deadline | Evaluation budget plus 3,000 ms after readiness. The V8 evaluation timeout remains its configured value. |
| Parent cancellation poll | 50 ms while waiting for the worker, followed by kill/reap. OS scheduling can affect observed latency. |
| Logs | At most 100 lines, 2,048 characters per line, and 16,384 characters in total. |
| Windows process boundary | One process, at most 256 MiB committed process memory. |

`duration_ms` returned by `execute_script` measures elapsed invocation time through the validated worker response, including worker startup. `initialization_ms` measures runtime initialization inside the worker, `execution_ms` measures the V8 evaluation/result conversion, and `startup_phases_ms` records the parent-observed bootstrap/import/V8 milestones. These values distinguish a cold/paged Windows startup from user-code execution. The startup allowance does not allow JavaScript to exceed its V8 evaluation budget. Native worker stderr is drained from process creation, including before readiness, so a runtime diagnostic cannot fill that pipe and stall startup. The parent retains only the last 8 KiB and includes at most 1,000 bytes of that tail in failure diagnostics.

Console object arguments are deliberately rendered as `[object]`; stringify a small selected value yourself when needed:

```javascript
console.info(JSON.stringify({count: inputs.values.length, factor: params.factor}));
return {value: inputs.values.map(value => value * params.factor)};
```

Large logging loops do not produce unlimited retained logs. A script that fails or exceeds a bound does not corrupt the next invocation; the next node invocation receives a new process and isolate.

### Numerical precision and wide integers

JavaScript Number is IEEE-754 binary64. Python integer inputs that cannot be represented exactly by Number are rejected before worker launch. Exactly representable integers such as powers of two remain usable. Trusted output serialization preserves the actual integral Number value when an ordinary shortest decimal representation would otherwise be read back as a different Python integer. It also preserves the sign of negative zero from inputs and computations; `-0.0` returns as a negative floating-point zero instead of being silently normalized to positive zero.

For arbitrary-width integer intent, use decimal strings:

```javascript
// inputs.seed is "9223372036854775807", not a rounded browser number.
return {value: (BigInt(inputs.seed) + 1n).toString()};
```

The native **Experiment configuration** node emits both seed fields as decimal strings. Keep those strings when transforming a config. Convert only fields on which you intentionally want binary64 arithmetic. A script's calculations do not replace the experiment engine's real float32 PyTorch parameter, optimizer, autograd, or forensic records.

## Transform a real dataset

This custom node preserves metadata while changing targets:

```javascript
return {
  data: {
    ...inputs.data,
    targets: inputs.data.targets.map(value => value * params.factor),
    provenance: {
      ...inputs.data.provenance,
      transform: "scale_targets",
      factor: params.factor
    }
  }
};
```

Declare input `data: dataset`, output `data: dataset`, and parameter `factor: number`. Wire **Custom dataset** → this node → **Create experiment.dataset**. The host validates the resulting data and stores actual tensor-bound provenance; plugin-supplied provenance is user metadata, not a trusted assertion that the samples were verified elsewhere.

`plugin-dataset-training.workflow.json` is a complete example. Install `example.plugin.json` first. The script's 1 MiB input/output and heap limits still apply even though the host dataset row limit is 100,000.

## Declarative GUI panels

Panels offer portable controls without injecting browser code:

```json
{
  "id": "scale-controls",
  "title": "Scale controls",
  "description": "Reusable input and factor values.",
  "fields": [
    {"name":"value","type":"number","default":21},
    {"name":"factor","type":"number","default":2,"min":-100,"max":100}
  ]
}
```

The accepted panel fields are `id`, `title`, `description`, and `fields`. Each field uses the parameter declaration schema. In the installed plugin card:

1. Open the panel and edit its values.
2. **Add values as an Object node** copies them into an editable `core.object` node in the current workflow. Connect that object's output to a Script, Get property, or configuration input.
3. **Apply matching values to selected node** writes matching fields into the selected node's literal inputs/parameters. It refuses field names that node does not accept.

Changing a panel does not automatically train or run code. A disabled plugin's panel actions are disabled. HTML/CSS injection, arbitrary DOM widgets, and browser event-handler source are not panel features.

## Workflow templates and graph JSON

A plugin template is `{id,name,graph}`. Its local template ID must be unique. It appears under the globally namespaced catalog ID `plugin-id/template-id` while its plugin is enabled. Template installation validates its graph against the staged registry. Include all required types and complete connections.

A standalone graph has this shape:

```json
{
  "name": "Scale and inspect",
  "description": "A reusable pure-data workflow.",
  "nodes": [
    {"id":"number","type":"core.number","position":{"x":40,"y":100},"params":{"value":21}},
    {"id":"scale","type":"local.my-tools/scale","position":{"x":370,"y":100},"params":{"factor":2}},
    {"id":"preview","type":"core.preview","position":{"x":700,"y":100},"params":{}}
  ],
  "edges": [
    {"id":"number-scale","source":"number","source_port":"value","target":"scale","target_port":"value"},
    {"id":"scale-preview","source":"scale","source_port":"value","target":"preview","target_port":"value"}
  ],
  "view":{"x":0,"y":0,"zoom":1}
}
```

Graph documents may also include `id`, `revision`, `created_at`, and `updated_at`. The server assigns identity/revision when creating a workflow; the browser's workflow import creates a new identity. Node fields are `id`, `type`, optional `title`, `position:{x,y}`, and `params`. Input literals and parameter values are stored in `params`; the runtime resolves input literals into `inputs` before calling custom code.

An edge has exactly `id`, `source`, `source_port`, `target`, and `target_port`. Each source/target node must exist. Only one edge may target a given input. IDs must be unique. Execution requires declared ports, compatible types, required values, valid parameter bounds/options, and an acyclic graph.

Incomplete but structurally valid graph documents can be saved. Raw invalid node-field or manifest text is retained separately as UI drafts; it is not a new executable schema. Complete project saving preserves those drafts. Opening a project or importing a graph never executes it.

## Plugin lifecycle and reproducible queues

| Operation | Behavior |
| --- | --- |
| Install | Validate the manifest/templates, add an enabled plugin, and expose its nodes/templates. Duplicate installed IDs are a conflict. |
| Update | Keep the same plugin ID. Identical normalized content is a no-op; changed content requires a changed version string. |
| Disable | Remove types/templates from the active catalog. Saved workflows retain their node references for later re-enabling. |
| Enable | Make the stored definitions available again. |
| Remove | Remove the registry entry. Existing graphs retain missing-type nodes; future execution requires restoring/replacing them. |
| Download | Save the accepted manifest as portable JSON. Download draft saves the editor text, which may not yet validate. |

Each queued execution pins the graph revision and the actual node definitions/code it will use. It records plugin versions and SHA-256 manifest hashes. Later registry updates do not rewrite captured jobs. No host action is automatically retried. A repeated HTTP execution request with the same `idempotency_key` replays the captured job identity; a different graph/revision with that key conflicts.

The queue is serial. Cancellation preserves completed host effects, skips future nodes, and interrupts an active script process. It does not rewind a run or delete a previously created experiment. Training checks cancellation between steps. Reopened active jobs become interrupted and are not automatically resumed.

Export nodes retain exact ZIP bytes with byte length and SHA-256; they do not regenerate a newer ZIP when their link is clicked. These bytes are included in complete project serialization and validated on import. Clearing/pruning the owning completed job releases its retained artifacts.

## Local HTTP API

The server binds to loopback and checks local request origin/host boundaries. The API is for the local studio, not a deployed multiuser service. Mutation payloads are strict JSON; nonfinite numbers and duplicate object keys are rejected by the transport.

| Method and route | Result |
| --- | --- |
| `GET /api/graph/catalog` | Current types, templates, sandbox metadata, and limits. Custom code is omitted from the palette definitions. |
| `GET /api/graph/state` | Graph/plugin/job state and retained artifact descriptors. |
| `GET /api/graph/jobs/<id>` | Raw exact JSON attachment for one retained job; preserves server numeric tokens without browser reserialization. |
| `POST /api/graph/action` | One action; response contains `ok`, refreshed graph `state`, and action `result`. |
| `GET /api/graph/artifacts/<id>` | The stored immutable ZIP bytes for that artifact. |
| `GET /api/project` | Complete workspace/project download. |
| `POST /api/project` | Validate and replace the workspace from a complete project. Active workflows must be stopped first. |
| `GET /api/ui/drafts` | Portable unapplied editor buffers as `{drafts,revision,workspace_epoch}`. |
| `POST /api/action` with `workspace.drafts` | Save draft buffers; optional `expected_revision` enables a strict compare-and-swap check. |

Graph actions are `graph.create`, `graph.update`, `graph.select`, `graph.delete`, `graph.duplicate`, `graph.validate`, and `graph.execute`. Job actions are `job.cancel`, `job.delete`, and `job.clear`. Plugin actions are `plugin.install`, `plugin.update`, `plugin.enable`, `plugin.disable`, and `plugin.remove`.

### Bind a client to the current workspace identity

The main state, graph state, and draft-read responses expose `workspace_epoch`, a random hexadecimal identity for the currently loaded live workspace. It changes only after a successful project import. It is never restored from an imported file, so even reopening a file with the same saved workspace ID produces a new live epoch.

After the client's initial read, send that value as the optional **`X-Studio-Epoch`** header on workspace API reads and writes. This guards state/draft requests, mutations, and raw project/job/artifact downloads against acting on a workspace that another tab replaced. A stale epoch returns HTTP 409 with error type `StudioEpochError` before copying state or performing a mutation. Headerless clients retain backward compatibility.

Treat the epoch as a concurrency identity, not an authentication credential. Do not silently adopt a new epoch after a stale response and retry an old mutation. Retain the client's buffers and explicitly decide whether to reload the server project or export the local draft first. The browser pauses editing/autosaves on this conflict, keeps local buffers available, and offers explicit draft-download/reload choices. Its initial load and its own successful Open action are the deliberate points at which it captures the new identity.

The epoch guard is separate from graph `expected_revision` and draft `expected_revision`: the epoch identifies which loaded workspace a request belongs to; a revision protects edits within that workspace. The example PowerShell client captures and sends the epoch automatically when the server exposes it.

Examples of action bodies:

```json
{"action":"graph.create","template_id":"sandbox","name":"My playground"}
```

```json
{"action":"graph.execute","graph_id":"YOUR_GRAPH_ID","expected_revision":1,"idempotency_key":"YOUR_UNIQUE_REQUEST_ID"}
```

```json
{"action":"job.cancel","job_id":"YOUR_JOB_ID"}
```

```json
{"action":"plugin.install","manifest":{"schema_version":"spobs-plugin/1","id":"local.example","name":"Example","version":"1.0.0","nodes":[{"id":"constant","title":"Constant","inputs":[],"outputs":[{"name":"value","type":"number"}],"parameters":[],"code":"return {value:42};"}]}}
```

`graph.update` supplies the complete graph and `expected_revision`; stale revisions return a conflict instead of overwriting another editor. `job.delete` requires a completed job. `job.clear` clears completed jobs and leaves active work alone.

### Portable raw editor drafts and revision conflicts

Editor buffers use `schema_version: "spobs-ui-drafts/1"`. They are stored separately from applied experiment/graph state. A raw string may contain incomplete or invalid JSON; it is preserved for editing and is never implicitly parsed into executable state.

| Draft field | Meaning and bounds |
| --- | --- |
| `runs` | At most 100 experiment-buffer objects. Supported fields are `config`, `config_json`, `config_json_dirty`, `csv`, `notes`, `bits`, `bit_dirty`, and `optimizer_json`. |
| `workspace_notes` | Unapplied workspace-note text. |
| `plugin`, `plugin_dirty` | Raw manifest editor text and its unapplied flag. |
| `node_fields` | Raw field strings grouped by workflow ID; at most 100 workflow groups and 10,000 fields per group. |
| `panel_fields` | Raw add-on field strings grouped by stable panel ID; at most 512 panels and 100 fields per panel. |

The aggregate draft payload is limited to 16 MiB. Individual raw node/panel field strings are limited to 1,000,000 characters; other draft fields have their own text bounds. The endpoint returns the authoritative current schema and buffers, so an API client can read and round-trip a buffer set without guessing control IDs.

Read `/api/ui/drafts`, preserve its complete `drafts` object and the returned epoch, edit the intended fields, then submit:

```json
{
  "action": "workspace.drafts",
  "expected_revision": 7,
  "drafts": {
    "schema_version": "spobs-ui-drafts/1",
    "runs": {},
    "plugin": "{ unfinished manifest",
    "plugin_dirty": true,
    "node_fields": {},
    "panel_fields": {}
  }
}
```

Send this action to **`POST /api/action`**, rather than `/api/graph/action`, and include the captured `X-Studio-Epoch` header. The action replaces the draft set, so begin with the latest full object when you need to retain unrelated buffers. `expected_revision` is optional for existing clients; when supplied, it must be an exact integer. A stale value returns HTTP 409 before mutation. Success returns the new draft `revision` in the action result. Ordinary workspace state exposes only `ui_drafts_revision` for these buffers; obtain their raw content from the dedicated draft endpoint.

The browser uses this revision check and retains this tab's text on conflict. Its dialog offers **Keep this tab's edits** or **Load saved edits**. Those are explicit choices to replace the saved draft set or the current unapplied buffers. They do not apply experiment settings, perform optimizer steps, install a plugin, or execute a workflow.

For an executable PowerShell client that imports an optional plugin, creates/validates/queues a supplied workflow, polls its terminal result, and downloads any node ZIPs, see [run-workflow.ps1](examples/node_lab/run-workflow.ps1). Run it only when you intend that example's host actions to execute.

## Python integration

These are the real local functions used by the graph runtime:

```python
from single_parameter_observatory.studio_plugins import (
    ScriptSandboxError, execute_script, sandbox_capabilities, validate_plugin,
)

manifest = validate_plugin(manifest_dictionary)
limits = sandbox_capabilities()
try:
    result = execute_script(
        "return {value: inputs.x * params.factor};",
        {"x": 21}, {"factor": 2}, timeout_ms=1000, memory_mb=32,
    )
    print(result["outputs"]["value"])  # 42
except ScriptSandboxError as error:
    print(error)
```

The result contains `outputs`, `logs`, total `duration_ms`, `runtime`, and the initialization/execution timing fields described above. A `threading.Event` may be supplied as `cancel=`. `node_definitions(manifest)` produces namespaced catalog entries; `example_plugin()` returns the delivered SDK example. These APIs do not install a plugin into a workspace by themselves; use the graph registry actions for lifecycle/storage.

## Practical authoring checks

Keep one small input/output example for each custom node. Check missing optional fields, declared output names/types, finite numeric behavior, and the maximum data size you actually intend to process. A valid manifest proves schema validity, not correctness of its algorithm. A successful script proves that invocation completed; it does not establish a clinical, scientific, or model-quality conclusion beyond its recorded data.

Use [catalog.json](examples/node_lab/catalog.json) as a source snapshot for these examples, and query the running `/api/graph/catalog` when writing against a different installed version. Installed source, manifests, saved jobs, actual outputs, and user acceptance remain distinct pieces of evidence.
