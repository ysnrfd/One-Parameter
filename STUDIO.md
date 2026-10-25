# Parameter Studio

Parameter Studio 2.2.0 is a local visual coding workspace for multi-parameter
PyTorch and Hugging Face Transformers workflows. Compose architectures, edit
forward programs, prepare data, train, evaluate, generate, inspect actual
parameter tensors, and export safe weights or runnable Python projects.
The original scalar observatory remains available for its detailed numerical
history, interventions, and exact-state branches. Node lab provides the complete
workflow editor, and Plugins provides reusable JavaScript nodes and declarative
add-ons. Each execution system has distinct controls, observations, and limits.

## Start the studio

On this Windows checkout, double-click **Launch Studio.cmd**. It uses the
checkout's `.venv` and opens <http://127.0.0.1:8765/>. If the studio is already
running there with the same version, the launcher reopens it. If another
version owns the port, the launcher asks you to stop that server before retrying.
Keep its terminal window open while using the studio; Ctrl+C stops the server
and its owned training and workflow workers.

From the repository root:

```powershell
.venv\Scripts\python.exe -m single_parameter_observatory studio
```

From an installed package:

```text
spobs studio
spobs studio --port 8767 --storage my-studio --no-browser
spobs-studio --port 8767 --storage my-studio --open
```

Both command-line entry points default to persistent storage in `.spobs-studio`
relative to the working directory. The Windows launcher fixes that directory
beside itself. Both accept only loopback hosts. A busy port produces an error; select
another port instead of opening a second service on the same port.

The setup commands for a fresh Python environment are in the root README.
The browser interface adds no web framework or JavaScript package dependency.
It serves its HTML, CSS, and JavaScript from the installed Python package.

## Explore the workspace

Nine views are available. The first six share the selected scalar experiment
and transport bar. Node lab, Plugins, and Transformers use the full work area,
with workflow and execution controls:

| View | What you can do |
|---|---|
| **World** | Create experiments from all presets, drag cards, pan and zoom the canvas, fit the world, and follow branch connections. |
| **Timeline** | Observe actual parameter, loss, gradient, update, moment, and learning-rate series; compare experiments; scrub to an exact historical record. |
| **Microscope** | Inspect float32 bits, hex, recorded tensor stages, optimizer state, and the full raw record. Edit the live parameter, learning rate, or supported optimizer state explicitly. |
| **Data lab** | Inspect inputs, targets, predictions, and a float64 diagnostic loss landscape. Edit, import, or export two-column CSV and create a custom-data branch. |
| **Journal** | Keep workspace and experiment notes alongside the engine's activity events. |
| **Artifacts** | Download a portable project, export an experiment's numerical evidence, verify its record chain, and capture or restore checkpoints. |
| **Node lab** | Build a typed node graph, edit literals and scripts, connect ports, arrange and reuse selections, validate, queue, cancel, inspect outputs, and download execution artifacts. |
| **Plugins** | Create custom nodes, import/export or edit versioned manifests, install/update/enable/disable/remove plugins, use contributed templates, and turn add-on forms into workflow values. |
| **Transformers** | Build many-tensor architectures, open HF templates, add model/data/code/training blocks, observe actual model counts and losses, inspect tensor reports, and download safe checkpoints. |

Use **New experiment** to start another independent scalar run. Transformers
models are created and trained by workflow jobs, with their outputs in Node lab
and Transformers; they do not appear as scalar experiment cards. Each scalar run has its own
configuration, controller, parameter, optimizer, history, and playback speed.
Selecting a different experiment does not stop an experiment already playing.
While a selection request is pending, selected-experiment controls and downloads
wait for acknowledgement so they cannot act on the previous experiment.

The inspector exposes the complete configuration: optimizer and hyperparameters,
initial value, step budget, batching and accumulation, data generation, model
scales and tanh/linear choice, scheduler, reference track, validation, landscape,
and stability settings. The advanced JSON editor uses the same validated schema.
**Apply as new branch** creates a fresh configured experiment and retains the
parent. A configuration branch starts from its configured initial value.
Changing optimizer, model, or training settings retains imported samples when
the data generation fields are unchanged. Changing data generation fields
explicitly creates newly generated data; the original custom-data experiment
remains available.

**Branch experiment** instead copies the current numerical state and history.
Use that when you want two experiments to continue from exactly the same place.
A checkpoint captures that same kind of complete state; restoring a checkpoint
creates a new branch, preserving the current experiment.

Playback calls the existing `loss.backward()` and `optimizer.step()` engine.
Step executes one real update; Advance by executes an exact number up to the
remaining step budget. Pause waits for an in-progress update to finish. A run
finishes at its configured budget or reports its numerical abort/error. Reset
returns live parameter/optimizer/reference state to the beginning, reinitializes
the analyzer, and retains recorded history. It uses the remaining global step
budget; resetting a completed run does not extend that budget. Restart starts
fresh history and a fresh budget while retaining imported data. Capture a
checkpoint or download a project before restarting when you want to retain the
current observations.

## Make an explicit intervention

Microscope can set a finite float32 parameter using a decimal value or an
eight-digit IEEE-754 hex word. Bit controls edit that word. Learning-rate and
optimizer-state edits affect the real optimizer and its enabled float64
reference. Edits pause the experiment and create a `MANUAL_INTERVENTION` event
with before/after values; they do not manufacture a training-step observation.

Adam/AdamW support `step`, `exp_avg`, and `exp_avg_sq`, plus `max_exp_avg_sq`
when AMSGrad is enabled. Initializing an empty Adam state requires every
applicable field together. SGD supports `momentum_buffer`. Variances must be
nonnegative, optimizer step must be an integer, and values must fit the finite
storage range. Learning rate must be positive. Invalid edits are rejected with
an explanation. Checkpoint first when you want a convenient return point.

Historical scrubbing is observation. Interventions always change the live
state, and their controls identify that distinction. To continue an earlier
state, restore a checkpoint captured there.

The live optimizer metrics read the actual optimizer state, including manual
changes and resets. Before initialization they remain empty. A pinned historical
record displays its recorded optimizer memory instead. Loss, gradient, adaptive
update, decay, and rounding diagnostics identify their last recorded step;
changing a live parameter does not invent a replacement forward observation.
Entering decimal `-0` uses the exact `0x80000000` float32 word.

## Bring your own data

CSV contains exactly two numeric columns, optionally headed `x,target`,
`input,target`, or `x,y`:

```csv
x,target
-1,-0.8
0,0
1,0.8
```

Applying CSV creates a separate experiment with those actual samples stored as
float32 tensors. The parent is preserved. The data provenance records the CSV
hash and imported content. Custom targets have no asserted known optimum;
the diagnostic landscape is the full dataset's mean squared error evaluated in
float64. It is a separate diagnostic from mini-batch training loss.

## Build a visual workflow

Open **Node lab** with its sidebar button or key **7**. Start with **+ New**,
add a node from the searchable category library, or choose a template:

* **Sandbox playground** connects an Object containing `{"value":21}` to a
  JavaScript sandbox and Preview. Queue it to observe the actual output `42`.
* **Training laboratory** connects configuration, experiment creation, ten real
  optimizer steps, a checkpoint, and recorded observation.
* **Custom-data experiment** builds a typed dataset and configuration, trains
  the actual samples, and inspects the resulting experiment and dataset.
* Enabled plugins can contribute additional workflow templates.

Select a node to edit its title, literal inputs, parameters, and source code.
An input accepts either a literal of its declared type or one incoming wire.
Connected wires take priority over that input's literal. Each output may feed
several nodes. Run inputs offer a dropdown of existing experiments as well as
an explicit run-ID field. Configuration seed editors retain all decimal digits
of supported 64-bit seeds; quote large seeds when writing complete workflow
JSON or generic script objects yourself.

Drag either an input or output dot onto a compatible opposite port, click the
two opposite ports, or use **Connect ports with a form** in the destination
node's inspector. Drag a port onto empty canvas to open **Add a compatible node**.
Its searchable choices list only compatible node ports, including separate
choices when a node accepts several compatible inputs. Choose one to create
and connect the new block in a single undo step. The existing wire on an input
stays intact until the new producer is chosen. **Escape**, outside click or a
view/project change cancels the chooser without modifying the graph.

For tensor-expression ports, discovery additionally checks known dtype and
rank: integer token matrices and floating hidden states are different contracts.
Custom-operation outputs remain unknown and permissive until compiled. The
picker displays width hints as advisory; the compiler validates full dimensions
before model allocation. A rejected connection preserves an input's old wire.

Compatible ports glow while dragging and incompatible ports dim. Port colors
and labels identify types. On a focused port, **Alt+Down** opens the same
chooser; use arrows to navigate, Enter to choose and Escape to cancel.
Replacing an input connection removes its previous incoming wire. The editor
rejects incompatible connections and cycles; the runtime validates again before
execution. Workflows are directed acyclic graphs. Scripts can implement bounded
loops inside an individual node.

New palette additions find a nearby free position without moving existing
nodes. Imported/template positions and deliberate dragging retain their exact
layout. Drag headers to move nodes. Shift-click selects several; **Edit → Choose several
nodes** provides checkbox selection without a keyboard. Copy, paste, duplicate,
delete, and selection controls are available in the Edit menu. Copied internal
wires are remapped to the new node IDs. **Save selection as workflow** creates
a reusable workflow from a selected group and its internal wires. Connections
from outside that selection become unwired inputs to configure before execution.
Drag the background to pan, scroll or use the zoom buttons to zoom, and choose
**Fit** to frame the graph. **Inspector** and **Library** toolbar buttons show or
hide those panes at every screen width. Closing the inspector removes its
reserved desktop column; selecting a node or wire reopens it. The narrow-screen
inspector remains an overlay. Pane changes cancel unfinished connections and
recompute wire positions without changing the graph's saved pan/zoom.

Undo/redo restores graph edits. It does not reverse training or other completed
host actions. Use a numerical checkpoint or exact-state branch for those return
points. **Workflow → Edit complete JSON** exposes the complete graph structure;
structural validation protects the saved document. Incomplete graphs may be
saved and corrected later. The same menu imports and downloads standalone
workflow JSON, duplicates workflows, and deletes a workflow without deleting
experiments already created by it.

### Export a runnable Python project

Choose **Workflow → Download runnable Python project**. Valid workflow drafts
are saved first; invalid field text remains in the editor and prevents this
export. The downloaded `.py` file contains an inert compressed project snapshot,
including workflows, plugin declarations, scalar checkpoints, and retained
artifacts. Importing the file does not create a workspace or execute nodes.
Run it explicitly with the installed Studio version recorded in its header, for example:

```powershell
python .\workflow.py --timeout 300 --output .\job.json --artifacts .\artifacts
```

The program restores the snapshot, queues its selected workflow, writes exact
job JSON, and optionally saves that job's immutable artifacts with identity
prefixes. It closes the owned runtime on completion or failure. This is an
editable normal Python program outside the browser sandbox; the in-studio
JavaScript and tensor-forward blocks retain their separate restricted interfaces.
Downloading the script never runs it.

### Core built-in nodes

The current catalog has 61 built-in types: the following 21 core operations
and 40 HF/tensor operations described in [TRANSFORMERS_STUDIO.md](TRANSFORMERS_STUDIO.md).
Installed plugins add their own namespaced types.

| Node | Actual behavior |
|---|---|
| Number | A finite numeric literal. |
| Text | A string literal. |
| Boolean | A true/false literal. |
| JSON value | A strict JSON literal. |
| Array | An editable array literal or incoming array. |
| Object | An editable object literal or incoming object. |
| Math | Add, subtract, multiply, divide, power, min, max, or modulo; invalid/non-finite arithmetic fails explicitly. |
| Get property | Read an object key or array index, with an explicit fallback. |
| JavaScript sandbox | Run a bounded function body on JSON inputs and expose its returned outputs. |
| Preview | Observe the exact incoming JSON value. |
| Experiment configuration | Validate a partial configuration and produce a complete experiment configuration. |
| Custom dataset | Construct finite float32 input/target samples without training. |
| Create experiment | Create a real independent PyTorch experiment, optionally using an incoming dataset. |
| Train | Perform the requested real optimizer steps on an experiment. |
| Branch experiment | Copy exact current experiment state and history into a branch. |
| Capture checkpoint | Capture a real numerical checkpoint. |
| Inspect experiment | Read a live snapshot or an actual historical record and optionally its dataset. |
| Manual intervention | Explicitly apply finite parameter, hex, learning-rate, or supported optimizer-state changes. |
| Verify record chain | Verify the experiment's actual numerical record chain. |
| Experiment control | Play, pause, reset, or restart a real experiment. |
| Export experiment | Generate and retain an actual evidence ZIP with a hash-bound local download. |

### Execute and inspect

**Validate** reports missing inputs, disabled or unknown node types, invalid
literal values, incompatible ports, and cycles. **Queue workflow** captures an
immutable workflow revision and the node definitions/plugin versions used by
that job. Editing a graph or updating a plugin afterward affects future jobs.
Saving, importing, installing, and opening do not execute code.

The serial execution queue records each node's status, actual inputs, parameters,
named outputs, logs, duration, and failure. The execution panel lets you select
past jobs, inspect scalar/text/JSON outputs, view numeric sequences and tables,
and download the complete results JSON. Display previews are bounded: long JSON
is truncated after 16,000 characters, tables preview 100 rows, and charts sample
up to about 2,000 actual sequence points. Complete retained outputs are available
in the job download. Export-node results provide the registered ZIP download,
filename, byte count, and SHA-256.

HF-node results instead show compact visual cards: connected operations and
incoming ports, model/data/tokenizer descriptors, measured training-loss charts,
evaluation metrics, weight-change summaries, generated token strips and decoded
text. **Inspect model visually** opens the same observed job in Transformers
without queueing another execution. **Advanced · exact outputs** and per-node
execution logs are collapsed; their bounded text is generated only when opened.
The node inspector uses these visual summaries too. Complete result downloads
and trusted checkpoint links retain their original contents and identities.

The execution download uses the server's original JSON response bytes rather
than re-encoding browser-parsed numbers. That preserves exact stored large
integer literals. The graph transport, JSON editors, copy/paste, undo, workflow
imports and exports preserve integer tokens outside JavaScript’s safe numeric
range without converting them to rounded numbers. Ordinary decimal or exponent
values and numeric node controls use JavaScript binary64; typed integer controls
reject unsafe values. Custom JavaScript computations require large integer
identifiers to be passed as decimal strings. Negative zero is retained in typed
controls, JSON documents and exports.

**Cancel job** works while the graph has unapplied or invalid editor text.
Cancellation is checked between native training steps and node boundaries; the
script subprocess can also be terminated. Completed host effects remain real
and recorded after cancellation or a later failure. Graph undo is not a rollback
of those effects. A Play node starts independent experiment playback; pause that
experiment explicitly if you also want its playback to stop.

An immutable workflow snapshot captures graph and definition identity. An
existing run-ID input still refers to that experiment's actual state when its
node executes. Pause competing playback and branch/checkpoint exact numerical
state when a controlled comparison is required.

## Build Transformers and many-parameter models

Open **Transformers** with key **9**. **+ Build dense architecture** prepares an
editable dense network with configurable hidden widths, input/output features,
activation, normalization, optimizer, learning rate, step count, seed, and JSON
tensor data. The gallery creates copies of the runtime's executable model
workflows. The searchable block library adds individual HF or tensor blocks to
the active Node lab graph. Computation starts only with **Queue workflow**.

There are separate blocks for HF configuration/model construction, pinned
cached pretrained models and tokenizers, local tokenizer construction,
tokenization, tensor datasets, layer recipes, Linear, Embedding, RMSNorm,
LayerNorm, activation, attention, composition, Sequential, custom tensor
forward, optimizer, train, evaluate, forward, generate, pipeline, inspect,
checkpoint export, and checkpoint load. Layer recipes form the forward
architecture; AdamW or another optimizer connects to Train together with the
model and data.

Models use actual PyTorch tensors and HF implementations on CPU. Config-created
models start with new weights. Pretrained blocks require a valid Hub repository
ID, an exact pinned commit, and already cached safetensors/tokenizer assets;
the worker performs no downloads and does not load remote Python code.
Custom forward bodies use the documented tensor AST and named layer recipes.
JavaScript plugins can emit those recipes or prepare data; they do not execute
arbitrary Python imports inside the studio.

**Observed executions** displays actual model parameter counts, recorded losses,
weight-change hashes, bounded tensor-name/shape/dtype/element reports, complete
job downloads, and registered safe checkpoint downloads. Model handles belong
to one execution. To reuse weights in another execution, export a safe checkpoint
and use Load safe model. Scalar project checkpoints and model safetensors
checkpoints are different artifacts.

Read the [complete Transformers guide](TRANSFORMERS_STUDIO.md),
[HF source audit and support boundary](TRANSFORMERS_AUDIT.md), and
[runnable model workflows](examples/transformers/README.md) for concrete schemas,
examples, prerequisites, and supported operations. The
[historical 2.0.0 release report](TRANSFORMERS_REPORT.md) records its executed qualification;
the [native Llama guide](LLAMA_STUDIO.md) records the historical 2.1.0 addition;
the [editable component guide](LLAMA_BLOCKS.md) describes the 2.2.0 graph runtime;
a catalog entry by itself does not establish every possible model/configuration.

## Build Llama from editable component blocks

Choose **Transformers → + Build Llama from blocks**. Set the decoder count,
hidden and feed-forward widths, query/KV heads, vocabulary capacity, token
context, training sentences, seed, learning rate and training updates. The live
preview reflects an expanded SwiGLU branch, a fused SwiGLU block, or a custom
activation. **Create connected Llama workflow** creates editable blocks without
running them. Choose **Validate → Queue workflow** to execute the graph.

The two component templates expose embeddings, RMSNorm, position IDs, Q/K/V
projections, RoPE, grouped-query causal attention, output projection, residual
additions, SiLU, gated products, feed-forward projections, final norm and the
LM head as separately wired operations. These `tensor-expression` ports feed
**Compile visual causal model**, whose `model` output connects to the ordinary
training, inspection, checkpoint and generation nodes. The compiler registers
shared weights once and replays the connected computation on every forward.

Cards show operation colors, formulas and up to three primitive settings
directly on the canvas. Valid inline edits commit on each input event, including
before focus leaves the field; a later unchanged change event adds no duplicate
undo entry. The node inspector exposes all remaining parameters; custom
operation code is an optional restricted tensor expression. Invalid inline
fields retain their draft text immediately and block saving or execution until
corrected. Training and generation use the same compiled graph, and safe
checkpoints preserve both the connected recipe and weights. Read
[LLAMA_BLOCKS.md](LLAMA_BLOCKS.md) for port contracts, custom expressions and
the component runtime's limits.

## Build and inspect native Llama

The included **Llama decoder training and architecture** template and [portable Llama workflow](examples/transformers/llama.workflow.json) use the actual installed `LlamaForCausalLM`: token embeddings, two causal decoder layers, four query heads and two key/value heads, RoPE, pre-attention and post-attention RMSNorm, SwiGLU, residual additions, final RMSNorm and a vocabulary projection. The default model has **78,144 trainable parameters in 21 tensors**. Its local 32-token vocabulary and eight training sentences require no model download or credentials. It starts from random weights.

After execution, the Transformers view displays the inspected model configuration, decoder-layer selector, forward structure, actual module inventory and per-layer parameter shapes. Edit the connected config, tokenizer/data, optimizer or training node in Node lab, then execute to observe the edited model. The diagram explains the forward structure; the job records actual losses, weight hashes, generation outputs and safe checkpoint reload. Read [LLAMA_STUDIO.md](LLAMA_STUDIO.md) for the complete working example and [LLAMA_SOURCE_NOTES.md](LLAMA_SOURCE_NOTES.md) for the HF Viewer reference and installed implementation review.

## Author plugins and add-ons

Open **Plugins** with key **8**. **Create custom node** collects the plugin ID,
name, node ID/title, typed input/output declarations, parameters, and JavaScript
function body. It builds a complete editable manifest. **Install / update
plugin** validates and registers it explicitly. **Load sample manifest** provides
two working custom nodes and an add-on control panel. Import and download
ordinary `.plugin.json` manifests; the JSON editor also supports plugins made
outside the studio.

A minimal manifest is:

```json
{
  "schema_version": "spobs-plugin/1",
  "id": "local.my-tools",
  "name": "My tools",
  "version": "1.0.0",
  "description": "Reusable local transformations",
  "nodes": [{
    "id": "scale",
    "title": "Scale value",
    "category": "Custom",
    "inputs": [{"name": "value", "type": "number", "required": true, "default": 21}],
    "outputs": [{"name": "value", "type": "number"}],
    "parameters": [{"name": "factor", "type": "number", "default": 2}],
    "code": "console.log('scaling', inputs.value); return {value: inputs.value * params.factor};"
  }]
}
```

Registered node types use `plugin_id/node_id`, such as
`local.my-tools/scale`. Port types are `any`, `number`, `integer`, `string`,
`boolean`, `array`, `object`, `run`, `config`, and `dataset`. Input literals may
declare defaults. Parameters may declare defaults, descriptions, numeric bounds,
and typed option lists. Unknown manifest fields and invalid declarations are
rejected so misspellings do not silently produce inactive features.

Custom code receives `inputs`, `params`, and a bounded `console` with
`log/info/warn/error`. Return a plain strict-JSON object with exactly the declared
output names. Code is a synchronous JavaScript function body, not an HTML page,
Python module, shell command, or browser script. JavaScript numeric arithmetic
uses binary64; large integer inputs outside its exact safe range must be encoded
as text when their exact decimal identity matters. Real experiments retain their
separate float32/float64 numerical rules.

Changed installed plugin content requires a new version. Updating retains the
plugin ID; jobs already queued keep their captured definitions. Disable removes
its node types from the available library, while saved graphs retain those nodes
for later re-enabling. Removing a plugin likewise preserves workflow documents;
validation identifies the missing types until their dependency is installed.

Plugins may also declare `templates` containing complete workflow documents and
`panels` containing safe form-field declarations. A panel has `id`, `title`, an
optional `description`, and `fields` using the parameter schema. Its GUI can
create an editable Object node from its current values, or apply matching values
to a selected node. These are working declarative add-on controls. They do not
inject arbitrary HTML or gain browser/host privileges. Panel drafts survive
reload and registry view rebuilds; disable the plugin to disable its panel actions.

### Sandbox boundary

Scripts run in an isolated V8 subprocess through the installed `mini-racer`
runtime. They receive JSON and expose no filesystem, network, DOM, Python-import,
or direct host API. Native experiment operations remain explicit built-in nodes.
Per-node limits include 10–10,000 ms timeouts and 16–64 MiB V8 memory settings,
with defaults of 1,000 ms and 32 MiB. Cancellation, strict JSON transport,
bounded logs/output, and process cleanup are enforced separately. These are
execution/resource controls for the local extension format; the studio is not
a general operating-system container or arbitrary Python plugin host.
The execution timeout is separate from bounded worker startup overhead. Script
source is limited to 64 KiB, and the worker's JSON data frame to 1 MiB; the V8
memory setting is not a claim that the complete Python worker process consumes
only that much operating-system memory.

The node editor is an original implementation inspired by visual workflow tools.
Its graph and plugin formats are documented here. ComfyUI Python-node packages
and ComfyUI workflow files are not interchangeable with these manifests.

## Keep and reopen your work

The default launcher autosaves atomically to `.spobs-studio/autosave.json` at
roughly two-second intervals and on a clean shutdown. The interface reports
storage errors. Download a project with **Save** or **Artifacts → Complete
workspace** for a portable copy. **Open** restores it into the current workspace.
Imported experiments always open paused.

Projects use strict JSON (`spobs-studio/1`), including exact tensor bytes,
optimizer state, accumulated gradients, scheduler and float64 reference state,
dataset tensors, numerical history, checkpoints, notes, and canvas positions.
They also include saved workflows, installed plugin manifests and enablement,
immutable queued-job snapshots and observations, retained export artifacts,
and unapplied editor drafts. Code is stored as source text and does not execute
when a project opens. Reopened pending/running workflows become interrupted
history; they do not resume automatically.
State digests and the record chain are verified before replacement. Invalid
projects are rejected before the current workspace is replaced. The loader uses
an explicit type whitelist, with no pickle or executable Python.

Configuration JSON/form drafts, imported or edited CSV, notes, bit edits,
optimizer JSON, plugin source drafts, node-field text, and add-on panel fields
are retained separately from applied state. Drafts save after a short editing
debounce and are included by the project Save action. Invalid editor text can
remain a draft without executing or overwriting the valid numerical state.
Correct invalid typed node fields before validating/executing their workflow.
Concurrent graph revisions are checked: a conflicting editor retains its draft
and offers a new-workflow copy or a deliberate reload of the saved version.
The complete editor-draft set also uses revision checks across tabs. A conflict
pauses draft autosave and retains the local text. **Keep this tab's edits**
explicitly replaces the saved draft set with this tab's editors after fetching
the current revision. **Load saved edits** replaces this tab's unapplied buffers
with the saved set. Neither choice manufactures numerical history or silently
merges competing editor text.

If an autosave is corrupt, the studio quarantines that source file and reports
the recovery path. Preserve the quarantined file for inspection; an empty
recovery workspace does not prove the old project was recovered.

Project digests detect inconsistent files; they are not a signature proving
authorship. Preserve a downloaded copy before deliberately replacing a workspace
or deleting an experiment.

An **experiment ZIP** contains the numerical export formats, dataset and
provenance, and a checkpoint of the selected run. It is evidence for that run;
use the full project download to reopen the whole studio.

## Shortcuts

| Key | Action |
|---|---|
| Space | Play / pause selected experiment |
| S | One training step |
| F | Branch current state |
| 1–9 | Switch between studio views |
| Ctrl+S / Cmd+S | Download project |
| Ctrl+K / Cmd+K | Open command palette |

Typing inside a form or editor takes priority over training shortcuts.
Training shortcuts do not operate scalar experiments while Node lab, Plugins,
or Transformers is active. In Node lab, Ctrl/Cmd+Enter queues the workflow; Ctrl/Cmd+Z and
Ctrl/Cmd+Shift+Z undo/redo; Ctrl/Cmd+A/C/V/D select all/copy/paste/duplicate;
Delete removes the selection; Escape cancels a pending wire or clears selection.
The Edit menu provides the same selection operations without keyboard shortcuts.

## Scope and practical limits

The original forensic views explore one real trainable `torch.float32` scalar
per experiment, using the linear/tanh model, mean squared error, and Adam, AdamW,
or SGD. Transformers and tensor workflows additionally support many parameters
across many trainable tensors, editable layer architectures, supported HF model
tasks, and AdamW/Adam/SGD/RMSprop training. Their observations do not imply a
bit-by-bit forensic trace of every model weight. General JSON transformations
use isolated JavaScript; custom tensor forward code uses a restricted AST.
Arbitrary local Python scripting is available through the exported program,
which runs outside the in-studio worker interfaces.

The default model budget is 2,000,000 parameters, with 1,000,000 tensor elements,
2,000 steps per training operation, sequence length 512, batch size 128, and a
2,048 MiB worker process memory limit. Supported environment settings can adjust
several budgets before server startup. Inspect the current Transformers runtime
capabilities and guide for the effective values. These limits and actual host
resources bound execution; the studio does not promise unlimited memory or
complete compatibility with every HF class, plugin, model, or task.

The server permits up to 100 experiments, 1,000,000 steps per experiment,
100,000 data rows, 10 MB CSV text, and 64 MiB HTTP project requests. Real memory,
CPU time, and project size impose lower practical limits; recorded diagnostics
are extensive. Browser trajectories select up to 2,000 actual observations per
run and disclose sampling. Full records remain available through history lookup
and exports. No displayed trajectory is generated as a substitute for training.

The graph layer permits 100 workflows, 500 nodes and 2,000 edges per workflow,
64 installed plugins, and 16 running/queued jobs. Its graph registry is bounded
to 8,000,000 bytes and plugin registry to 4,000,000 bytes. Execution uses an
8,000,000-byte cumulative observation budget per job and a 24,000,000-byte
retained job-history budget, with at most 50 job entries. Retained evidence ZIPs
share an 8 MiB artifact budget. Editor drafts have a 16 MiB budget. Graph
execution quotas are exposed by the current catalog; project and editor budgets
are also enforced by the server. They are resource limits, not a promise
that all maxima fit simultaneously into a 64 MiB portable-project request or
available machine memory. Large observations may fail explicitly and retain
already completed effects; export important records before filling the workspace.

Exact continuation is verified within the same installed PyTorch runtime.
Changing Python/PyTorch versions can change optimizer behavior; a portable
project does not guarantee identical arithmetic across different runtimes.

The local server checks request origins and hosts, rejects non-finite JSON and
duplicate JSON keys, and serves only fixed application assets. It has no remote
account or hosted execution service. Do not expose the port through a tunnel
or reverse proxy as a multi-user service.

Current editable-component verification and its practical scope belong to
[LLAMA_BLOCKS.md](LLAMA_BLOCKS.md). The native Llama **2.1.0** qualification is
in [LLAMA_STUDIO.md](LLAMA_STUDIO.md). Historical 2.0.0 Transformers verification
is in [TRANSFORMERS_REPORT.md](TRANSFORMERS_REPORT.md). The source review and support
boundary are in [TRANSFORMERS_AUDIT.md](TRANSFORMERS_AUDIT.md). Historical
1.3.0 node/plugin results remain in [NODE_LAB_REPORT.md](NODE_LAB_REPORT.md),
1.2.0 studio results remain in [STUDIO_REPORT.md](STUDIO_REPORT.md), and
1.1.1 numerical repairs remain in [REPAIR_REPORT.md](REPAIR_REPORT.md).
The current portable source/evidence archive is
[`single_parameter_observatory_2.2.0.zip`](download/single_parameter_observatory_2.2.0.zip);
release identities belong to [RELEASE-2.2.0.json](download/RELEASE-2.2.0.json).

Workspace identity is checked in addition to draft and workflow revisions. If
another tab opens a different project, this tab retains its local buffers and
blocks stale requests. Download this tab’s drafts for recovery, or explicitly
reload the replacement project. Opening a project pauses workflow autosave
before the replacement commits, so delayed edits cannot overwrite its graphs.
The live optimizer learning-rate metric is **Next step LR**: schedulers may
prepare a new rate after the last recorded step. A pinned historical record
shows the learning rate actually used by that recorded step. Microscope bits
come from the stored float32 word, including signed zero, Infinity and NaN
payloads; only finite edited values can be applied as interventions.
