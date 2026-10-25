# Offline Transformers workflows

Import one of these `.workflow.json` files through the Node Lab workflow menu, then choose **Validate** and **Queue workflow**. They require the studio's Transformers runtime. Open the **Transformers** workspace for the architecture wizard, template gallery, block library, and observed execution summaries.

The current collection has **16 Transformers workflows**. Fifteen create local tiny models from explicit recipes and data, without a Hub checkpoint, credential, or network acquisition. One requires installing the included custom plugin. The separate cached-pretrained example requires the previously cached public fixture described below. Every execution uses real PyTorch or HF operations. Random tiny language models demonstrate execution rather than useful language quality.

| Workflow | What it constructs and observes |
| --- | --- |
| [cached-pretrained.workflow.json](cached-pretrained.workflow.json) | Load a real cached safetensors checkpoint and fast tokenizer at a pinned Hub commit; tokenize texts, fine-tune with AdamW, execute a pipeline, and verify a portable checkpoint reload. Requires the public fixture cached separately. |
| [plugin-architecture.workflow.json](plugin-architecture.workflow.json) | Install the included architecture plugin, let its sandboxed JavaScript node generate layer recipes, then build, train, inspect, and export a real multi-parameter model. |
| [rmsnorm-adamw.workflow.json](rmsnorm-adamw.workflow.json) | Separate Linear, RMSNorm, GELU, and output Linear blocks compose into a real multi-parameter model. AdamW trains it; inspection, safe export, reload, and evaluation follow. |
| [llama.workflow.json](llama.workflow.json) | Native LlamaForCausalLM with two decoder layers, grouped-query attention, RoPE, RMSNorm, SwiGLU and 78,144 parameters: tokenize, measure loss, train with AdamW, verify actual modules, export/reload weights, evaluate and generate. No Meta pretrained weights or downloads. |
| [llama-blocks.workflow.json](llama-blocks.workflow.json) | A 36-node component workflow with independent embedding, RMSNorm, Q/K/V, padding-aware positions, RoPE, grouped-query causal attention, residuals, expanded SwiGLU and vocabulary head. Compile the connected forward graph, train, inspect, checkpoint/reload and generate. |
| [llama-blocks-custom.workflow.json](llama-blocks-custom.workflow.json) | The same connected decoder with an editable custom tensor gate activation. Its restricted expression participates in autograd and is preserved with the connected graph and checkpoint. |
| [tiny-gpt2.workflow.json](tiny-gpt2.workflow.json) | A tiny actual `AutoModelForCausalLM` trains two token sequences, generates bounded token IDs, exposes parameter state, and exports safetensors. |
| [custom-forward.workflow.json](custom-forward.workflow.json) | Editable tensor-forward code calls declared dense/RMSNorm modules and GELU, then trains, evaluates, and exports actual weights. |
| [text-pipeline.workflow.json](text-pipeline.workflow.json) | A real local WordLevel fast tokenizer and tiny random GPT-2 execute the actual HF text-generation pipeline on two prompts. |
| [distilbert-classification.workflow.json](distilbert-classification.workflow.json) | A tiny DistilBERT sequence classifier trains on labelled token rows, evaluates its model loss, and executes an HF text-classification pipeline. |
| [checkpoint-roundtrip.workflow.json](checkpoint-roundtrip.workflow.json) | Train with SGD, inspect stored state, export safetensors, rebuild and reload the model, inspect again, and evaluate. The two inspection state hashes should match. |
| [fill-mask.workflow.json](fill-mask.workflow.json) | A tiny DistilBERT masked-language head and an explicitly configured `[MASK]` token execute the actual fill-mask pipeline. |
| [token-classification.workflow.json](token-classification.workflow.json) | A tiny DistilBERT token-classification head executes the actual HF named-entity/token pipeline with fast-tokenizer offsets. |
| [feature-extraction.workflow.json](feature-extraction.workflow.json) | A tiny base DistilBERT exposes actual contextual hidden-state values through the HF feature-extraction pipeline. |
| [tokenize-train-generate.workflow.json](tokenize-train-generate.workflow.json) | Tokenize unequal-length texts with left padding and masked causal labels, train tiny GPT-2, run forward, and generate tokens. |
| [custom-attention.workflow.json](custom-attention.workflow.json) | A custom embedding → RMSNorm → causal self-attention → residual addition → projection architecture trains with AdamW and exposes real tensor parameters. |

The custom attention and dense examples use an interpreted tensor language with Python syntax. They do not execute arbitrary Python, import modules, or define new host APIs. Layer and optimizer nodes have separate roles: layers build the forward model; the optimizer binds the complete model at Train.

For component graphs, **Transformers → + Build Llama from blocks** creates a
new editable architecture from visual controls and a live diagram preview.
Expanded SwiGLU exposes each operation; fused SwiGLU uses one editable branch
block; custom activation inserts an interpreted tensor program. In Node lab,
drag an input or output into empty canvas to search compatible blocks and add
the chosen node/port in one undoable action. Colored cards show formulas and
inline settings, with all parameters available in the inspector. Creating or
editing these graphs performs no model execution until queued. Read the
[editable Llama component guide](../../LLAMA_BLOCKS.md) for the exact contracts.

For the plugin workflow, first import [architecture.plugin.json](architecture.plugin.json) in **Plugins** and choose **Install / update plugin**. It adds the `local.transformer.recipes/mlp_layers` node, a reusable training template, and a **Custom architecture controls** GUI panel. The default hidden width is 8; node parameters or the optional `controls` object can select 1–64. Import the workflow after installing its manifest. Other examples do not require that plugin.

Execution handles live for one job. Preserve weights with the safe checkpoint node; saving the graph alone preserves recipes and code, not unexported worker memory. The checkpoint contains model weights and a recipe, not optimizer moments or a complete Trainer resume state.

The cached fixture is `hf-internal-testing/tiny-random-gpt2` at commit `71034c5d8bde858ff824298bdedc65515b97d2b9`. It is a random testing model, not a useful pretrained language model. The source packet records the seven acquired file hashes and 487,753-byte total in `evidence/transformers_20261023/hf_audit_demo_model.json`; external Hub cache contents are not bundled. See the main guide for the explicit cache preparation command and offline contract.

The historical **2.0.0** collection of **13 workflows completed successfully as actual graph jobs** in the source runtime on 2026-10-23. The hash-bound per-node results are in `evidence/transformers_20261023/examples_execution.json`; that receipt identifies the exact thirteen workflow source files executed in that build. The separate static wiring/code-syntax receipt is `hf_audit_examples_static.json`. Browser interaction and installed-release behavior have their own qualification evidence; static validation is not an execution pass.

Studio **2.1.0** adds `llama.workflow.json`, making **14 Transformer workflows** and **21 total portable workflows** when the seven Node Lab examples are included. The Llama execution and release qualification are recorded in [LLAMA_STUDIO.md](../../LLAMA_STUDIO.md) and `evidence/llama_20261023/`; the earlier receipt above does not include the new example. The [Llama source notes](../../LLAMA_SOURCE_NOTES.md) identify the reviewed HF Viewer and installed HF implementation.

Studio **2.2.0** adds the two component workflows, making **16 Transformers
workflows** and **23 portable workflows** including the seven core examples.
The component runtime's current execution, browser and distribution evidence
belongs to [LLAMA_BLOCKS.md](../../LLAMA_BLOCKS.md); the historical 2.0.0 and
2.1.0 receipts above retain their original scope.

See the [Transformers studio guide](../../TRANSFORMERS_STUDIO.md), [upstream source audit](../../TRANSFORMERS_AUDIT.md), and [Node Lab guide](../../NODE_LAB.md) for full controls, limits, data contracts, persistence, and extension behavior.
