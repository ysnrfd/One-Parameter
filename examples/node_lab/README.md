# Runnable Node Lab examples

These are ordinary JSON inputs for the current Parameter Studio Node Lab and Plugins views. They do not execute when a project starts or a file is imported. **Queue workflow**, or the PowerShell client below, explicitly executes them.

The three built-in templates and `catalog.json` were generated from `GraphStudio(None).catalog()`. `example.plugin.json` was generated from `example_plugin()`. The remaining workflows compose those current node contracts. Generating or parsing these files is separate from executing their numerical/script paths; consult the release's verification report for actual execution evidence.

| File | Required plugin | Expected behavior when successfully executed |
| --- | --- | --- |
| [sandbox.workflow.json](sandbox.workflow.json) | None | Pure JavaScript transform; final preview `42`. Creates no experiment. |
| [training.workflow.json](training.workflow.json) | None | Create a real experiment, train ten steps, capture a checkpoint, and inspect it. |
| [custom-data.workflow.json](custom-data.workflow.json) | None | Create a real experiment from explicit samples, train twenty steps, and inspect actual data. |
| [example.plugin.json](example.plugin.json) | This is the plugin | Two custom transformations, a three-node scale template, and a declarative control panel. |
| [plugin-scale.workflow.json](plugin-scale.workflow.json) | `local.examples` | Number → plugin scale → preview; final value `42`. |
| [plugin-dataset-training.workflow.json](plugin-dataset-training.workflow.json) | `local.examples` | Scale three target samples from `[-2,0,2]` to `[-6,0,6]`, create/train a real run, inspect data, and capture an evidence ZIP. |
| [branch-and-export.workflow.json](branch-and-export.workflow.json) | None | Train a baseline, checkpoint, branch exact state, change LR to `0.01`, train five more steps, verify, and capture branch evidence. |
| [wide-integer.workflow.json](wide-integer.workflow.json) | None | Exact decimal-string/BigInt calculation; final preview string `"9223372036854775808"`. |
| [catalog.json](catalog.json) | None | Snapshot of 21 built-in node definitions, three built-in templates, sandbox metadata, and quotas. Query `/api/graph/catalog` for the running version. |
| [run-workflow.ps1](run-workflow.ps1) | Optional `-Plugin` argument | Local API client: optional plugin installation/update, workflow creation/validation/queueing, terminal-state polling, raw graph-state capture, and byte/hash-verified ZIP downloads. |

## Use the browser

1. Launch the studio and open **Node lab**.
2. For a plugin example, open **Plugins**, import `example.plugin.json`, review it, and choose **Install / update plugin**.
3. In Node Lab, use the workflow menu → **Import workflow JSON** and select the desired `.workflow.json`.
4. Choose **Validate**, then **Queue workflow**. Inspect **EXECUTION QUEUE & OBSERVED OUTPUTS**.
5. A workflow containing **Create experiment**, **Train**, **Branch**, **Manual intervention**, or **Export evidence** performs those actual actions. Its effects are visible in the rest of the studio.

The plugin also exposes its scale workflow through the template selector. Its installed control panel can create an Object node or apply matching fields to the selected node. These GUI actions edit the graph; execution remains explicit.

## Use PowerShell

Start the local server first. From the repository root:

```powershell
.\examples\node_lab\run-workflow.ps1 -Workflow .\examples\node_lab\sandbox.workflow.json
```

Execute the plugin/data/training example and retain its captured evidence:

```powershell
.\examples\node_lab\run-workflow.ps1 -Plugin .\examples\node_lab\example.plugin.json -Workflow .\examples\node_lab\plugin-dataset-training.workflow.json -OutputDirectory .\node-lab-results
```

The client creates a new workflow identity every time. When `-Plugin` is supplied, it explicitly installs or updates and enables that plugin. Existing normalized content at the same version is a no-op; changed content needs a different version string. It polls for completion and requests cancellation if its client wait expires. Completed host effects remain even after a failure or cancellation.

The client captures the live `workspace_epoch` on its initial read and sends `X-Studio-Epoch` on subsequent API calls when supported. If another tab replaces the project, a stale request fails instead of applying old edits or downloading unrelated evidence from the new workspace. Rerun deliberately after reviewing the newly loaded workspace; the client does not automatically adopt a new identity and retry.

Results are written into a directory named with the returned job ID. `graph-state.json` saves the raw server response, preserving its exact JSON number tokens rather than reserializing them through PowerShell's numeric parser. Downloaded node ZIPs are checked against their captured SHA-256 descriptors.

This client uses the graph API; it does not download a complete numerical workspace/project. Use the studio's top-bar **Save** when you need the full portable project, including experiments and unapplied UI drafts.

See [Node Lab guide](../../NODE_LAB.md) and [Plugin SDK](../../PLUGIN_SDK.md) for all controls, schema details, execution boundaries, and resource limits.
