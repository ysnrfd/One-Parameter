# Node Lab: visual workflows for Parameter Studio

Node Lab turns an experiment into an editable graph: create values and data, transform them with code, create a real PyTorch experiment, train it, branch its state, inspect it, and export its evidence. The **Plugins** view adds reusable custom nodes, workflow templates, and declarative control panels.

The canvas is open to arrangement and composition. Execution has a specific scope: this studio observes exactly one trainable `torch.float32` scalar, and custom code computes JSON inside the JavaScript sandbox. Arbitrary Python modules, browser scripts, system commands, and unrestricted file/network access are not plugin capabilities.

The interaction was informed by visual node editors such as [ComfyUI](https://github.com/comfy-org/ComfyUI). Parameter Studio has its own graph schema and runtime. A ComfyUI workflow or Python custom-node package is not directly compatible.

## Open the studio

From the source workspace, double-click **Launch Studio.cmd**, or run this from PowerShell in the project directory:

```powershell
.\.venv\Scripts\python.exe -m single_parameter_observatory studio
```

For an installed distribution, the dedicated entry point is:

```powershell
spobs-studio --open --host 127.0.0.1 --port 8765 --storage .spobs-studio
```

Open `http://127.0.0.1:8765/`, then choose **Node lab** in the sidebar. The local server must remain running. Keep a separate storage directory when trying examples you do not want mixed into your main workspace.

## Start with a complete workflow

The **TEMPLATES** selector creates a new, independent workflow. Creating or opening it does not execute its nodes.

| Template | What happens when you queue it |
| --- | --- |
| **Training laboratory** | Validate configuration, create an experiment, perform ten real optimizer steps, capture a checkpoint, and inspect the run. |
| **Sandbox playground** | Pass `{ "value": 21 }` to JavaScript, multiply it by two, and preview `42`. No experiment is created. |
| **Custom-data experiment** | Build input/target samples, create an experiment with those actual samples, train twenty steps, and inspect its data. |

Choose **Validate** to check the current wiring and settings. Choose **Queue workflow** to compute. Expand **EXECUTION QUEUE & OBSERVED OUTPUTS** to follow each node and select earlier executions. A successful validation establishes structural/type validity; scripts and real numerical training can still fail during execution.

Portable copies of these templates and additional complete examples are in [examples/node_lab](examples/node_lab/README.md). For a useful first extension, import `example.plugin.json` in **Plugins**, choose **Install / update plugin**, and then import `plugin-scale.workflow.json` in Node Lab. Its final preview should be `42` when executed successfully.

## Build and edit a graph

1. Choose **+ New**, or start from a template.
2. Search the **NODE LIBRARY**, optionally filter by category, and click a node type to add it.
3. Select a node header to open its inspector. **Inputs · wire or literal** accepts an incoming connection or an explicit value. **Parameters** changes that node's settings.
4. Connect an output to an input by dragging between ports, clicking the two ports, or using **Connect ports with a form** in the inspector.
5. Validate, then queue the workflow. Inspect actual results before changing the next experiment.

An output may feed several downstream nodes. An input accepts one incoming wire. A wire takes precedence over that input's literal. **Clear literal** removes the explicit value so the catalog default, if any, applies. A required input without a wire, literal, or default prevents execution. An unset optional input is omitted from a custom node's `inputs` object.

Workflows are directed acyclic graphs. Use a `run` connection to order experiment actions: Create experiment → Train → Branch experiment → Manual intervention → Train. A loop is a validation error. A script can perform a finite loop within its own time and memory budget; the graph itself does not implement cyclic execution.

The supported port types are `any`, `number`, `integer`, `string`, `boolean`, `object`, `array`, `run`, `config`, and `dataset`. Exact matching types connect directly. `any` can connect across types, with the actual value checked during execution. Integers can feed numeric inputs; config/dataset objects can feed object inputs; objects can feed config/dataset inputs and are validated for that meaning. A string can identify a run. An incompatible actual output is an explicit failed node, not a silently coerced value.

### Canvas and selection controls

| Action | Control |
| --- | --- |
| Move nodes | Drag a node header; a selected group moves together. |
| Select a group | Shift-click headers, use the selection tool, or press Ctrl+A. |
| Pan / zoom | Drag the background / scroll; use Fit and the zoom buttons. |
| Copy / paste / duplicate selection | Ctrl+C / Ctrl+V / Ctrl+D. Internal wires are retained. |
| Delete selected nodes or wire | Delete or Backspace, or the canvas Delete button. |
| Undo / redo graph edits | Ctrl+Z / Ctrl+Shift+Z; Ctrl+Y also redoes. |
| Queue the current graph | Ctrl+Enter when focus is outside a text field or dialog. |
| Cancel a wire / clear selection | Escape. |
| Save a complete project | Ctrl+S. |
| Change studio view | Number keys 1–8 when not editing a field. |

The workflow menu also offers duplicate, complete JSON editing, workflow import/download, deletion, and the studio guide. A selected group can be saved as a separate reusable workflow; connections from outside the group become unwired inputs.

Undo changes the graph document. It does not undo an already completed optimizer step, experiment creation, intervention, or export. Use checkpoints and branches to preserve numerical alternatives.

## The 21 built-in node types

Inputs and parameters below use their exact schema names. The inspector supplies defaults where defined. A `run` value is an existing experiment ID, not a serialized Python object.

| Type / title | Inputs | Parameters | Outputs and behavior |
| --- | --- | --- | --- |
| `core.number` / Number | — | `value` | Numeric `value`. |
| `core.text` / Text | — | `value` | String `value`. |
| `core.boolean` / Boolean | — | `value` | Boolean `value`. |
| `core.json` / JSON value | — | `value` | Any strict JSON `value`. |
| `core.math` / Math | `a`, `b` | `operation` | Numeric `value`; add, subtract, multiply, divide, power, min, max, or modulo. Nonfinite results and division by zero fail explicitly. |
| `core.array` / Array | `items` | — | Array `value`. |
| `core.object` / Object | `value` | — | Object `value`. Useful for script inputs and grouped controls. |
| `core.get` / Get property | `object`, `key` | `default` | Property/index `value`; missing keys or out-of-range indices return the default. Array indices must be integers; negative indices are supported. |
| `core.script` / JavaScript sandbox | `inputs` object | `code`, `timeout_ms`, `memory_mb` | `outputs` is the complete returned object; `value` is its `value` property, or null if omitted. |
| `core.preview` / Preview | `value` | — | The actual incoming `value`, shown in execution results. |
| `core.config` / Experiment configuration | `value` object | — | Validated complete `config`; seed fields are decimal strings so wide integers survive browser transport. |
| `core.dataset` / Custom dataset | `inputs`, `targets` arrays | `name`, `batch_size` | Validated `dataset` with actual samples and metadata. This node does not train. |
| `core.experiment` / Create experiment | `config`, optional `dataset` | `name`, `preset` | Create a real independent experiment and return its `run` ID. |
| `core.train` / Train | `run` | `steps` | Execute up to the requested real steps and return `run` and `record`. Reaching the configured run budget can execute fewer steps; the log states the actual count. A numerical abort fails the node. |
| `core.branch` / Branch experiment | `run` | `name` | Return a new `run` with the source's captured parameter, optimizer, scheduler, data, and observations. |
| `core.checkpoint` / Capture checkpoint | `run` | `name` | Capture a real checkpoint and return `run` and checkpoint ID `checkpoint`. |
| `core.inspect` / Inspect experiment | `run` | `step`, `include_dataset` | Return `run`, `record`, and `dataset`. `step=-1` reads live state; a nonnegative step requests a historical record. Dataset output is `{}` when not requested. |
| `core.intervene` / Manual intervention | `run`, `changes` | — | Explicitly change supported paused state, then return `run` and live `record`. This is recorded as an intervention, not an optimizer step. |
| `core.verify` / Verify record chain | `run` | — | Return `run` and `integrity` from actual record-chain verification. Read the integrity status, not just the job's success label. |
| `core.control` / Experiment control | `run` | `operation` | `play`, `pause`, `reset`, or `restart`; return `run` and live `record` after the control operation. |
| `core.export` / Export evidence | `run` | — | Generate an immutable evidence ZIP and return `run` and an `artifact` descriptor with its byte length, SHA-256, and download URL. |

`core.control` with `play` starts continuous training. Its successful node result means the play command completed; it does not mean the run reached its final step. Use **Train** for a bounded, awaited number of steps, and pause a continuously playing run before state-changing operations that require it to be paused.

`reset` restores live training state while preserving the recorded history. `restart` starts the run's history again from its initial state while retaining its configuration and custom data. Export evidence or capture a checkpoint before deliberately restarting a trajectory.

Manual intervention accepts `parameter_value` (alias `parameter`), `parameter_hex`, `learning_rate` (alias `lr`), and `optimizer_state`. Do not supply both an alias and its canonical field. Float32 parameter changes must be finite; unsupported or invalid optimizer fields fail validation. For example:

```json
{"parameter_hex":"0x3f800000","learning_rate":0.01}
```

This writes the finite float32 value `1.0` and changes the learning rate. It does not invent a gradient or claim that the optimizer produced that change.

## Observe results and evidence

The execution panel records node inputs, explicit parameters, outputs, logs, errors, durations, and state. Numeric arrays have a chart preview, arrays of objects have a table preview, and JSON remains inspectable. Large previews are shortened for display; **Results JSON** downloads the retained complete job record.

Queued jobs capture the graph revision and the node definitions they will use. Plugin versions and manifest hashes are pinned. Editing, disabling, updating, or removing a plugin affects later queues; it does not replace the source code of a job already captured.

Executions are serial within the graph queue. Status progresses through `queued`, `running`, and a terminal state such as `succeeded`, `failed`, or `cancelled`. `cancelling` means the request has arrived and the active operation is still stopping. Nodes not reached after a failure or cancellation are skipped. On project reopening, formerly active jobs become `interrupted`; they are never resumed automatically.

**Cancel job** stops future nodes and interrupts an active script worker. Training cancellation is checked between completed steps. A completed experiment creation, training step, checkpoint, intervention, or export remains recorded. Cancellation is not a rollback transaction. Review the job's completed outputs and the corresponding experiments in World, Timeline, Microscope, and Artifacts before continuing.

An **Export evidence** node retains the ZIP bytes captured during that execution. A later change to the experiment does not alter that ZIP. Download it from the artifact link in results, or use its `download_url`. These node ZIPs share an 8 MiB quota. For a larger one-off export, use the studio's main **Artifacts** view. Retained node artifacts belong to their jobs; deleting/pruning those jobs releases their artifacts.

## Save, reopen, and keep unfinished work

There are distinct things to save:

| Item | How it is retained |
| --- | --- |
| Structurally valid workflow document | Saved locally after graph edits, with a revision check. It can still have unwired inputs, disabled node types, or other execution-validation errors. |
| Standalone workflow JSON | Workflow menu → Download workflow JSON. Contains that graph, not all experiments/plugins/job history. |
| Installed plugin | Stored in the plugin registry and in complete projects. Export its manifest from Plugins. |
| Unapplied editor text | UI drafts preserve raw node-field, add-on panel, and manifest text, including text that is currently invalid JSON. Such text is not executable state. |
| Complete portable project | Top-bar Save / Ctrl+S. Saves workflow documents, experiments, plugins, retained jobs/artifacts, and UI drafts together. |

If a node field contains invalid JSON, the editor keeps its text and reports the error. Execution and normal graph validation require correction. Complete project saving can retain the valid graph document plus that invalid raw field draft, so the unfinished text is available after reopening. **Download draft** in Plugins also saves the manifest editor text as-is; an invalid draft cannot be installed until corrected.

Two tabs editing the same workflow can produce a revision conflict. The local draft is retained. Choose **Keep draft as a new workflow**, or download the workflow before choosing **Reload saved workflow**. This avoids quietly overwriting another tab's work.

The saved editor-buffer set has its own revision check. If another tab saved newer unapplied text, a separate dialog offers **Keep this tab's edits** or **Load saved edits**. The first explicitly replaces the saved draft set with this tab's buffers; the second replaces this tab's unapplied buffers with the saved set. Applied experiments and numerical history are separate from this choice.

If another tab opens a different complete project, this tab also detects that its live workspace identity changed. Editing and autosaves pause while this tab's buffers are retained. Download the local draft if needed, or explicitly reload the current server project. The studio does not silently redirect old edits or downloads into the replacement workspace. A local-draft download preserves editor work; use the server's complete project Save for the full numerical workspace.

Opening a complete project replaces the current workspace after confirmation. Download a project copy first when you need both. Active graph work must be cancelled and fully stopped before project replacement. Imported experiments are paused and imported jobs do not auto-execute.

## Current resource boundaries

These are implementation limits, not a claim that every maximum-size combination is practical on every computer. `/api/graph/catalog` reports current values for the running installation.

| Resource | Limit |
| --- | --- |
| Workflows / nodes / connections | 100 workflows; 500 nodes and 2,000 wires per workflow. |
| Plugin registry | 64 plugins and 4,000,000 encoded bytes across installed plugins. |
| Workflow documents | 2,000,000 bytes per graph; 8,000,000 bytes across the graph registry. |
| Queue | 16 queued/running/cancelling jobs together; one graph execution worker. |
| Retained execution history | 50 jobs, 8,000,000 bytes per job, and 24,000,000 bytes across retained job observations. Completed jobs can be pruned as budgets fill. |
| Retained node ZIPs | 8 MiB in total, also the maximum individual node ZIP. |
| Training node | 1–10,000 requested steps; the experiment's own step budget still applies. |
| Dataset | Up to 100,000 input/target pairs; script byte/memory limits can impose a smaller practical transform size. |
| Script | 64 KiB source; 1 MiB combined inputs/parameters; 1 MiB outputs; 16–64 MiB V8 heap. |

Script execution defaults to 1,000 ms and may be configured from 10 to 10,000 ms. The parent separately bounds trusted process/V8 initialization to 30 seconds. Once the runtime reports readiness, the parent allows the evaluation budget plus 3 seconds for input/output transfer, then terminates an overrun. The V8 evaluation timeout remains unchanged. A freshly started worker is used every time. On Windows, its Job Object also limits committed process memory to 256 MiB and permits one worker process.

The scripting boundary is a separate process plus V8 isolation and resource limits. It is not an operating-system virtual machine. See the [Plugin SDK](PLUGIN_SDK.md#sandbox-contract) for the precise script contract and the official [MiniRacer architecture](https://bpcreech.com/PyMiniRacer/architecture/) for upstream security goals.

## Troubleshooting

| Symptom | Meaning and next action |
| --- | --- |
| Unknown or disabled node type | Install/enable the matching plugin, or replace that node. Its graph document is still editable. |
| Missing required input | Connect the port, enter a literal, or provide a declared default. |
| Incompatible port type / invalid output | Inspect the declared schema and actual value. A wire's visual compatibility cannot guarantee every possible runtime value. |
| Script timeout or memory error | Reduce the computation/output, or change that node's allowed budget within the advertised limits. The next invocation starts fresh. |
| Integer cannot be represented exactly | Pass a wide integer as a decimal string. Use JavaScript `BigInt(value).toString()` for exact integer work. |
| Numerical abort | Inspect the real experiment and its last record. Change the learning rate/configuration or branch from an earlier checkpoint. |
| Artifact quota full | Download needed ZIPs, then remove completed job history through the available history controls/API. Use main Artifacts for a larger immediate export. |
| Offline / Save failed | Keep the tab open and download a draft if needed. Restore the local server connection, then save again. |

For complete authoring schemas, plugin lifecycle rules, HTTP examples, and extension development, continue with [PLUGIN_SDK.md](PLUGIN_SDK.md).
