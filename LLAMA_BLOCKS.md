# Visual Llama component studio · 2.2.0

RMSNorm, rotary positions, attention, SwiGLU and the surrounding decoder are editable, connected operations. The studio compiles the connected tensor graph into a registered PyTorch model. Every forward call, training update and generated token replays that graph; the editor's connections determine the computation. Shared outgoing wires reuse the original operation and its weights.

## Build through controls and blocks

In **Transformers**, choose **Build Llama from blocks**. Set decoder layers, hidden/intermediate widths, query/KV heads, vocabulary/context, optimizer learning rate, updates and seed using the form. Choose expanded SwiGLU to expose gate, SiLU, product, up and down separately; the fused option keeps them in one editable block. The custom activation option inserts an editable tensor program. Creating a graph does not start training. Use **Validate**, then **Queue workflow** explicitly.

The **Llama built from editable component blocks** template is also in Node lab. Portable examples are [expanded Llama blocks](examples/transformers/llama-blocks.workflow.json) and [custom-gate Llama blocks](examples/transformers/llama-blocks-custom.workflow.json). The default example uses one decoder, width 32, intermediate 64, four query/two KV heads, 32 local vocabulary entries and 20 AdamW updates on eight educational sentences. It creates actual from-scratch weights; it does not contain pretrained Meta weights.

The new card controls edit common primitive parameters directly. Valid input reaches the graph immediately; invalid input stays as a marked draft and blocks queueing. Remaining settings and custom code are available in the selected-node inspector. Colors, icons and short operation formulas distinguish projections, normalization, position operations, attention, residuals and feed-forward branches. Fit provides an overview of large graphs; zoom and pan to edit a region. The Inspector and Library toolbar controls collapse either pane at every screen width to enlarge the canvas.

## Discover compatible connections

Drag from either an **output or input** socket onto empty canvas. A searchable picker shows compatible node types and the exact compatible port. Choosing an item adds the node and connection together. Escape or an outside click cancels. Existing compatible sockets are highlighted during a drag. An input's old wire remains until a replacement is selected.

For keyboard use, focus a port and press **Alt+ArrowDown**. Search, use arrow keys and Enter to choose, or Escape to cancel. A successful add-and-connect action is one undo step. Search filters port types and known tensor ranks/dtypes. Unknown custom outputs remain available. Width hints are advisory: the compiler validates the connected dimensions before model allocation. Generic `any` ports can still inspect or transform a recipe; object/model ports do not masquerade as tensor-expression ports.

## Every decoder branch is visible

| Block | Connected inputs | Editable behavior |
| --- | --- | --- |
| Token / mask input | None | Input field and integer/float/boolean dtype |
| Token embedding | Token IDs | Vocabulary table size and hidden width |
| Projection / LM head | Hidden tensor | Input/output widths and bias; separate Q, K, V, output, gate, up, down and head weights |
| RMSNorm operation | Hidden tensor | Normalized width, epsilon and actual trainable scale |
| Position IDs from mask | Padding mask | Padding-aware position construction |
| Rotary position / RoPE | Projection, optional position IDs | Query/KV head count, even head dimension and theta |
| Grouped-query causal attention | Q, K, V, optional padding mask | Head geometry, causal masking and attention dropout |
| SiLU activation | Gate tensor | Parameter-free activation |
| Gated product | Two tensors | Elementwise product with checked geometry |
| Residual addition | Two tensors | Explicit skip branch with checked geometry |
| SwiGLU block | Hidden tensor | Fused trainable gate/up/down widths and bias |
| Custom tensor operation | x, optional a/b | Tensor code and optional named trainable layer recipes |
| Compile visual causal model | Vocabulary logits | Context/vocabulary, special token IDs and initialization seed |

Every expression block outputs `x` of semantic type `tensor-expression`. The final compiler accepts `logits` and outputs a model for the existing optimizer, Train, Evaluate, Forward, Generate, text pipeline, Inspect and checkpoint blocks. AdamW belongs to the training path. RMSNorm, RoPE and attention belong to the model's forward path.

Each operation is recorded in the actual inspection report, including its compiled shape, connections and registered module path. The Transformers result view provides selectable operation cards and actual parameter observations. This custom visual model is identified as a studio tensor graph; the separate native HF Llama template continues to use `LlamaForCausalLM`.

Node-lab execution results use visual operation cards, tensor shape/dtype and model metric chips, actual training-loss charts, generated token strips and decoded text. **Inspect model visually** opens that same execution in Transformers. Exact outputs and worker logs remain available in collapsed, lazy-rendered advanced details, and the Results JSON download retains the complete observations.

## Custom computation stays part of the trainable graph

Ordinary architecture editing uses controls and wires. Use a **Custom tensor operation** only when its mathematical behavior needs code. It accepts up to three wired tensors through `inputs['x']`, `inputs['a']` and `inputs['b']`; batch mask and positions are also available as `inputs['attention_mask']` and `inputs['position_ids']`. Optional named layer recipes in `layers` register trainable weights once and are saved with the model.

A custom gate can start with:

```python
return silu(inputs['x'])
```

A custom RoPE block can use the same bounded rotary kernel:

```python
return rope(inputs['x'], num_heads=4, head_dim=8, theta=10000)
```

A custom attention block can connect Q to x, K to a and V to b:

```python
return causal_attention(inputs['x'], inputs['a'], inputs['b'], num_heads=4, num_key_value_heads=2, head_dim=8)
```

Custom blocks accept and return rank-two or rank-three tensors with integer, boolean or float dtype, preserving the batch and token dimensions. This also allows custom embedding lookup, position IDs and padding-mask transformations. The compiler infers and checks the returned width and dtype.

The restricted tensor language supports single-name assignments and a final return, named layer calls, arithmetic, indexing, reshape/transpose/permute, reductions, concatenation, standard activations, sqrt/rsqrt/pow/cos/sin/cumsum/clamp and the contextual rotary/attention helpers. It preserves autograd. Imports, attribute access, loops, reflection and arbitrary Python execution remain unavailable in this worker. A tensor program and its named layer recipes are portable graph data, including through custom typed plugin nodes; the compiler validates the resulting recipe before use.

## Training, generation and saved projects

The causal-model training path shifts labels by one token and masks padding. Generation replays the connected graph autoregressively, with deterministic greedy decoding by default and bounded optional sampling. The tensor-graph text pipeline uses the same autoregressive implementation with a Hugging Face tokenizer and labels that implementation in its result; the original HF model pipeline path remains available.

Checkpoint ZIPs contain `model.safetensors`, `recipe.json` and `manifest.json`. The recipe includes every connected operation and custom program. Loading reconstructs a distinct model and verifies tensor names, shapes and state identity. These checkpoints preserve architecture and weights; optimizer/RNG training resume is outside this format. Runnable Python project export captures the saved graph for explicit execution in the installed studio environment.

The default process and allocation budgets still apply: 2,000,000 model parameters, finite tensor sizes, at most 500 workflow nodes and a bounded worker lifetime. Shape inference and a meta-device model probe precede real weight allocation. Query heads must divide evenly by KV heads, and RoPE requires positive even head dimensions. A graph can deliberately change the architecture, but incompatible dimensions produce a clear failure instead of silently rewiring the model.

## Reproduce the qualification

Run the standalone qualification in a disposable workspace:

```powershell
.\.venv\Scripts\python.exe .\scripts\qualify_components.py --output .\evidence\components-local --run-python-export
```

It executes expanded, custom two-layer and fused variants, verifies measured learning, generation and exact checkpoint evaluation, and replays each exported Python project with the same final weight-state hash. It saves the workflows, exact jobs, checkpoints, portable projects and execution receipts. The runtime tests additionally compare logits, causal loss and every trainable gradient against a matching native HF Llama after copying the same weights, and test causality, padding, invalid geometry, custom code restrictions and allocation limits.

Current execution and browser evidence are under `evidence/blocks_20261023/`. Historical native Llama 2.1.0 evidence remains under `evidence/llama_20261023/`. See the [native Llama guide](LLAMA_STUDIO.md), [Transformers guide](TRANSFORMERS_STUDIO.md), [Node lab guide](NODE_LAB.md) and [source review](LLAMA_SOURCE_NOTES.md) for the surrounding contracts and provenance.
