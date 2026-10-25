# Llama implementation authority and HF Viewer source review

Review date: 2026-10-23. This document records the sources behind the Llama workbench. Runtime results and release qualification belong in the accompanying Llama implementation report; source inspection alone does not establish successful training or generation. The full source release includes audit evidence; the installed wheel includes these notes only. Evidence paths below are relative to the full source/evidence release archive root.

## What the supplied HF Viewer reference provides

The user's [HF Viewer Space repository](https://huggingface.co/spaces/embedl/hfviewer/tree/main) was resolved through the public Hugging Face API to commit [`26aa6d623298cd60ecbc54c527d3ff545793889b`](https://huggingface.co/spaces/embedl/hfviewer/tree/26aa6d623298cd60ecbc54c527d3ff545793889b). The API identifies a public **static Space** with 62 repository entries. All 16 published text files with JavaScript, HTML, CSS, JSON or Markdown extensions were downloaded at that exact commit: 6,679,225 bytes, with individual SHA-256 hashes in the source-distribution-relative manifest `evidence/llama_20261023/hfviewer-source-manifest.json`. Images and SVG presentation assets were inventoried without downloading them.

The README describes an architecture explorer with zoom, a granularity slider and component explanations. The public frontend implements those viewing and model-resolution interactions. This is useful design guidance for an architecture inspector. The available repository does not publish the backend that traces models or the complete embedded graph renderer. It does not expose a training implementation or a visual architecture editing contract.

The relevant reviewed paths are:

- [`index.html`](https://huggingface.co/spaces/embedl/hfviewer/blob/26aa6d623298cd60ecbc54c527d3ff545793889b/index.html), especially its external API configuration around line 182: `https://hfviewer.com/api`.
- [`space-shell.js`](https://huggingface.co/spaces/embedl/hfviewer/blob/26aa6d623298cd60ecbc54c527d3ff545793889b/space-shell.js), including external viewer, card, iframe lifecycle and origin checks.
- [`hf-shell-app-2f72b4506d16.js`](https://huggingface.co/spaces/embedl/hfviewer/blob/26aa6d623298cd60ecbc54c527d3ff545793889b/hf-shell-app-2f72b4506d16.js), especially API URL construction around lines 5744–5775; model-resolution endpoints around 6059–6068; graph manifest and granularity loading around 14347–14455; and cached graph resolution around 52335–52445.

Manual review focused on those selected paths; downloading every text file is not a claim that every frontend line was manually audited. The source inventory contains no LICENSE file. The studio's local graph implementation should therefore be authored independently; these files are retained as provenance evidence and should not be represented as incorporated open-source components.

## The actual Llama graph returned by HF Viewer

A read-only, anonymous request to the external viewer's cached resolver for `meta-llama/Llama-3.2-1B` returned a ready graph. No model weights, credentials or POST requests were used. Its response reports `graph_fidelity: "trace-backed"`, `trace_backed: true`, and `residual_flow_guaranteed: true`. Both `hf_revision` and `reviewed_source_revision` are empty. Those fidelity labels are the remote service's assertions, not independently verified model execution or revision identity.

The response's bundle manifest points to `Llama-3.2-1B.json` and `_hfviewer_granularity_layout.json`. The graph has this general data shape:

```json
{
  "nodes": [
    {
      "id": "model.embed_tokens",
      "layerType": "Embedding",
      "parentIds": ["input"],
      "childrenIds": ["path:model.layers.0.__block_input__"],
      "properties": {"configVocab": 128256}
    }
  ],
  "metadata": {
    "modelType": "llama",
    "architecture": "LlamaForCausalLM",
    "hfConfigSummary": {"hidden_size": 2048, "num_hidden_layers": 16}
  }
}
```

The returned graph contains 21 nodes. It shows one decoder-layer motif with a separate repetition count of 16 in granularity metadata. Its attention-output node explicitly has `syntheticAttentionMerge: true`. The configuration summary reports hidden size 2,048, MLP width 8,192, 32 query heads, eight key/value heads, head dimension 64, vocabulary size 128,256, and maximum positions 131,072. This is a compact explanatory view, rather than a list of every runtime tensor operation or an executed checkpoint.

The raw resolver, manifest, graph and granularity responses are saved under the source-distribution-relative directory `evidence/llama_20261023/`. The source-distribution-relative review summary `evidence/llama_20261023/hfviewer-review-summary.json` and bundle receipt `evidence/llama_20261023/hfviewer-llama-bundle-receipt.json` record their hashes and limits.

## The execution authority: installed Transformers 5.20.2

Actual Llama execution must use the installed Hugging Face implementation, rather than reconstruct a model from the external viewer's summarized graph. The installed `configuration_llama.py` and `modeling_llama.py` were read directly and copied into the evidence directory with their existing copyright and Apache-2.0 notices intact. The source-distribution-relative installed-source receipt `evidence/llama_20261023/hfviewer-installed-llama-source.json` records:

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| `configuration_llama.py` | 3,896 | `5e2e803d15956cdc402ab1c4a5151a899bd39ddfbecbb81181733657817aaeda` |
| `modeling_llama.py` | 20,255 | `affc6a4ab36b562a6b4c90b84c4e7939c5417a69ad7bd37e858be5583f189830` |

The official [Llama documentation](https://huggingface.co/docs/transformers/model_doc/llama) describes its pre-normalization, gated feed-forward computation and rotary positions. The exact installed source gives the following executable structure:

1. `LlamaModel.embed_tokens` embeds token IDs. Its decoder layers share position-dependent cosine/sine values computed by `LlamaRotaryEmbedding`.
2. Each decoder layer retains its input as a residual and applies `input_layernorm`, implemented by `LlamaRMSNorm`. RMSNorm accumulates variance in float32 and returns to the input dtype before applying its trainable weight.
3. `LlamaAttention` applies independent `q_proj`, `k_proj` and `v_proj` linear projections, reshapes into heads, applies RoPE to query/key tensors, and optionally updates its key/value cache.
4. Grouped-query attention repeats the key/value heads to match the query heads. Eager attention uses scaled query-key products, the causal/padding mask, softmax, attention dropout and a value-weighted sum. The concatenated result passes through `o_proj` and is added to the retained input.
5. The second branch applies `post_attention_layernorm`, followed by the gated MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`. The branch is added to its own retained residual.
6. After all decoder layers, the model applies its final RMSNorm. `LlamaForCausalLM.lm_head` projects hidden states to vocabulary logits and its causal language-model loss uses labels. Generation uses the library's `GenerationMixin`.

Useful exact source ranges in the saved installed file are RMSNorm lines 52–69, rotary positions 72–150, gated MLP 152–165, grouped-query expansion and eager attention 168–202, attention 205–270, decoder residual paths 273–313, base-model construction and forward 335–406, and causal LM 409 onward.

## Concrete requirements for the studio's Llama implementation

Use an actual `LlamaConfig` plus `LlamaForCausalLM`/`AutoModelForCausalLM.from_config` as the execution object. A small randomly initialized configuration can run locally without gated Meta weights; it remains the real Llama architecture and must be labeled as a tiny from-scratch model. It is not a pretrained Meta checkpoint or a general-purpose assistant. Any later pretrained loading requires the exact model revision and legally available local files.

Generate the local architecture view from the configuration and the instantiated model's actual named modules, named parameters and shapes. Show every configured decoder layer, including the two residual branches, RMSNorm, grouped-query attention/RoPE and gated MLP. Present architecture inspection separately from recorded forward/training results. A view may explain functional operations such as RoPE or residual addition that do not own parameters; those should be identified as operations rather than invented trainable modules.

The configuration editor should validate query-head divisibility by key/value heads, valid positive dimensions, an even rotary head dimension, context and vocabulary bounds, and the studio's resource budget before allocation. Installed 5.20.2 uses `rope_parameters` (including `rope_type` and `rope_theta`), while `head_dim` can be explicitly configured. The upstream strict config validates `hidden_size % num_attention_heads`; the studio needs additional user-facing grouped-query and RoPE validation to avoid later reshape failures.

Qualify the same model through real tokenization, causal loss, autograd, AdamW updates, evaluation, autoregressive generation and safetensors save/reload. Compare exact tensor state before/after training and logits before/after reload. These results provide execution evidence that an external architecture screenshot cannot supply. User edits to configuration must feed the execution object and diagram from the same saved recipe.

An external HF Viewer link can provide a convenient comparison for published model IDs. It should remain optional and must not be required for a local from-scratch workflow, since it depends on another service and cannot describe this studio's newly trained weights or custom saved model identity.
