"""Visual, replayable tensor architecture recipes. No tensor allocation here."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

EXPRESSION_TYPE = "tensor-expression"


def definitions() -> list[dict[str, Any]]:
    def port(name: str, required: bool = True) -> dict[str, Any]:
        return {"name": name, "type": EXPRESSION_TYPE, "required": required}

    def p(name: str, kind: str, default: Any, **extra: Any) -> dict[str, Any]:
        return {"name": name, "type": kind, "default": default, **extra}

    def op(identity: str, title: str, description: str, inputs: list[dict[str, Any]],
           parameters: list[dict[str, Any]], group: str = "Tensor flow") -> dict[str, Any]:
        return {"id": "hf." + identity, "title": title, "category": "Llama blocks · " + group,
                "description": description, "inputs": inputs, "outputs": [port("x")], "parameters": parameters}

    def width(name: str, default: int) -> dict[str, Any]:
        return p(name, "integer", default, min=1, max=262144)
    bias = p("bias", "boolean", False)
    return [
        op("tensor_input", "Token / mask input", "A symbolic model input. Its connected operations replay on every forward pass, training update and generated token.", [],
           [p("field", "string", "input_ids", options=["input_ids", "attention_mask", "position_ids"]),
            p("dtype", "string", "int64", options=["int64", "float32", "bool"])]),
        op("op_embedding", "Token embedding", "Editable trainable vocabulary table; converts token IDs into hidden vectors.", [port("x")],
           [width("num_embeddings", 32), width("embedding_dim", 32)], "Embedding"),
        op("op_linear", "Projection / LM head", "A separately registered trainable Linear projection. Use for Q, K, V, attention output, gate, up, down or vocabulary head.", [port("x")],
           [width("in_features", 32), width("out_features", 32), bias], "Projection"),
        op("op_rmsnorm", "RMSNorm operation", "Trainable RMS normalization in the forward graph, with float32 variance accumulation and editable epsilon.", [port("x")],
           [width("normalized_shape", 32), p("eps", "number", 1e-6, min=1e-12, max=1)], "Normalization"),
        op("op_positions", "Position IDs from mask", "Convert a binary padding mask to zero-based, padding-aware position IDs for rotary embeddings.", [port("x")], [], "Position"),
        op("op_rope", "Rotary position / RoPE", "Apply real split-half rotary position embeddings to a Q or K projection. Connect editable position IDs or use the batch mask's positions.", [port("x"), port("positions", False)],
           [width("num_heads", 4), width("head_dim", 8), p("theta", "number", 10000, min=1, max=1e12)], "Position"),
        op("op_attention", "Grouped-query causal attention", "Q/K/V tensor inputs, grouped KV heads, scaled scores, padding/causal mask, softmax and weighted values. Every connection participates in autograd.",
           [port("q"), port("k"), port("v"), port("mask", False)],
           [width("num_heads", 4), width("num_key_value_heads", 2), width("head_dim", 8),
            p("causal", "boolean", True), p("dropout", "number", 0, min=0, max=0.99)], "Attention"),
        op("op_silu", "SiLU activation", "The parameter-free activation in the expanded SwiGLU gate branch.", [port("x")], [], "Feed forward"),
        op("op_multiply", "Gated product", "Elementwise multiply two equally shaped tensors, for the SiLU gate \u00d7 up branch.", [port("a"), port("b")], [], "Feed forward"),
        op("op_add", "Residual addition", "Add two equally shaped tensor branches without hiding the skip connection.", [port("a"), port("b")], [], "Residual"),
        op("op_swiglu", "SwiGLU block", "Editable fused trainable gate/up/down projections. Use the expanded template to wire each projection, SiLU and product separately.", [port("x")],
           [width("hidden_size", 32), width("intermediate_size", 64), bias], "Feed forward"),
        op("op_custom", "Custom tensor operation", "A reusable forward block with up to three wired tensors and optional named trainable layers. Restricted tensor expressions preserve autograd; no imports or arbitrary Python.",
           [port("x"), port("a", False), port("b", False)],
           [p("code", "code", "return silu(inputs['x'])"), p("layers", "object", {})], "Custom"),
        {"id": "hf.graph_model", "title": "Compile visual causal model", "category": "Llama blocks · Model",
         "description": "Compile the connected tensor DAG into a trainable causal language model. Shared visual nodes register weights once; the saved recipe replays for training, generation and checkpoint reload.",
         "inputs": [port("logits")], "outputs": [{"name": "model", "type": "object", "required": True}],
         "parameters": [width("vocab_size", 32), p("max_positions", "integer", 64, min=2, max=512),
                        p("pad_token_id", "integer", 0, min=0), p("bos_token_id", "integer", 2, min=0),
                        p("eos_token_id", "integer", 3, min=0), p("seed", "integer", 42, min=0, max=2**31-1)]},
    ]


def llama_blocks_workflow(*, layers: int = 1, hidden: int = 32, intermediate: int = 64,
                          heads: int = 4, kv_heads: int = 2, steps: int = 20,
                          custom: bool = False, fused: bool = False) -> dict[str, Any]:
    """Complete editable decoder, training, inference and checkpoint example."""
    if any(type(value) is not int or value < 1 for value in (layers, hidden, intermediate, heads, kv_heads, steps)):
        raise ValueError("Llama dimensions and steps must be positive integers")
    if layers > 8 or hidden % heads or heads % kv_heads or (hidden // heads) % 2:
        raise ValueError("Use at most eight layers, hidden divisible by Q heads, Q divisible by KV heads and an even head dimension")
    base = json.loads((Path(__file__).parent / "examples/transformers/llama.workflow.json").read_text(encoding="utf-8"))
    original = {node["id"]: node for node in base["nodes"]}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    def n(identity: str, op: str, title: str, x: int, y: int, **params: Any) -> str:
        nodes.append({"id": identity, "type": "hf." + op, "title": title,
                      "position": {"x": x, "y": y}, "params": params})
        return identity

    def e(source: str, target: str, port: str = "x", output: str = "x") -> None:
        edges.append({"id": f"{source}-{output}-{target}-{port}", "source": source,
                      "source_port": output, "target": target, "target_port": port})

    n("tokens", "tensor_input", "Token IDs", 0, 30, field="input_ids", dtype="int64")
    n("mask", "tensor_input", "Padding mask", 0, 330, field="attention_mask", dtype="int64")
    n("positions", "op_positions", "Padding-aware position IDs", 340, 340)
    e("mask", "positions")
    current = n("embedding", "op_embedding", "Token embedding · 32 \u00d7 hidden", 340, 30,
                num_embeddings=32, embedding_dim=hidden)
    e("tokens", current)
    for index in range(layers):
        prefix, x = f"layer{index}", 700 + index * 3000
        residual = current
        norm = n(prefix + "-norm1", "op_rmsnorm", f"Decoder {index + 1} · attention RMSNorm", x, 30,
                 normalized_shape=hidden, eps=1e-6)
        e(current, norm)
        for name, count, y in (("q", heads, 0), ("k", kv_heads, 420), ("v", kv_heads, 840)):
            project = n(prefix + "-" + name, "op_linear", f"{name.upper()} projection", x + 330, y,
                        in_features=hidden, out_features=count * (hidden // heads), bias=False)
            e(norm, project)
            if name != "v":
                rope = n(prefix + "-rope-" + name, "op_rope", "RoPE · " + name.upper(), x + 660, y,
                         num_heads=count, head_dim=hidden // heads, theta=10000)
                e(project, rope)
                e("positions", rope, "positions")
        attention = n(prefix + "-attention", "op_attention", "Grouped-query causal attention", x + 990, 30,
                      num_heads=heads, num_key_value_heads=kv_heads, head_dim=hidden // heads, causal=True, dropout=0)
        e(prefix + "-rope-q", attention, "q")
        e(prefix + "-rope-k", attention, "k")
        e(prefix + "-v", attention, "v")
        e("mask", attention, "mask")
        out = n(prefix + "-out", "op_linear", "Attention output projection", x + 1320, 30,
                in_features=hidden, out_features=hidden, bias=False)
        e(attention, out)
        add = n(prefix + "-residual1", "op_add", "Attention residual", x + 1650, 30)
        e(residual, add, "a")
        e(out, add, "b")
        norm2 = n(prefix + "-norm2", "op_rmsnorm", "Feed-forward RMSNorm", x + 1980, 30,
                  normalized_shape=hidden, eps=1e-6)
        e(add, norm2)
        if fused:
            down = n(prefix + "-swiglu", "op_swiglu", "SwiGLU · editable fused branch", x + 2310, 330,
                     hidden_size=hidden, intermediate_size=intermediate, bias=False)
            e(norm2, down)
        else:
            gate = n(prefix + "-gate", "op_linear", "SwiGLU gate projection", x + 1980, 390,
                     in_features=hidden, out_features=intermediate, bias=False)
            up = n(prefix + "-up", "op_linear", "SwiGLU up projection", x + 1980, 810,
                   in_features=hidden, out_features=intermediate, bias=False)
            e(norm2, gate)
            e(norm2, up)
            activate = n(prefix + "-activation", "op_custom" if custom else "op_silu",
                         "Custom gate activation" if custom else "SiLU gate", x + 2310, 390,
                         **({"code": "return silu(inputs['x'])", "layers": {}} if custom else {}))
            e(gate, activate)
            product = n(prefix + "-product", "op_multiply", "Gate \u00d7 up", x + 2640, 550)
            e(activate, product, "a")
            e(up, product, "b")
            down = n(prefix + "-down", "op_linear", "SwiGLU down projection", x + 2640, 30,
                     in_features=intermediate, out_features=hidden, bias=False)
            e(product, down)
        current = n(prefix + "-residual2", "op_add", "Feed-forward residual", x + 2640, 1050)
        e(add, current, "a")
        e(down, current, "b")
    x = 700 + layers * 3000
    n("final-norm", "op_rmsnorm", "Final RMSNorm", x, 30, normalized_shape=hidden, eps=1e-6)
    e(current, "final-norm")
    n("lm-head", "op_linear", "Vocabulary logits / LM head", x + 330, 30,
      in_features=hidden, out_features=32, bias=False)
    e("final-norm", "lm-head")
    n("model", "graph_model", "Compile connected decoder graph", x + 660, 30,
      vocab_size=32, max_positions=64, pad_token_id=0, bos_token_id=2, eos_token_id=3, seed=42)
    e("lm-head", "model", "logits")
    # Retain the proven local vocabulary and eight-sentence data while making
    # every architecture operation an independent node above.
    for identity, kind, title, dx, dy, params in (
        ("tokenizer", "tokenizer", "Local editable vocabulary", 0, 1350, original["tokenizer"]["params"]),
        ("data", "tokenize", "Eight training sentences", 340, 1350, original["data"]["params"]),
        ("optimizer", "optimizer", "AdamW", x + 330, 1350, original["optimizer"]["params"]),
        ("initial-eval", "evaluate", "Initial causal loss", x + 990, 30, {"loss": "model"}),
        ("train", "train", "Train the connected architecture", x + 1320, 30,
         {"steps": steps, "batch_size": 8, "loss": "model", "clip_grad_norm": 1}),
        ("trained-eval", "evaluate", "Trained causal loss", x + 1650, 30, {"loss": "model"}),
        ("inspect", "inspect", "Actual connected module and weight inventory", x + 1980, 30, {}),
        ("checkpoint", "checkpoint", "Save graph recipe + trained weights", x + 2310, 30, {}),
        ("reload", "load", "Rebuild the same graph and reload", x + 2640, 30, {}),
        ("reload-eval", "evaluate", "Reloaded causal loss", x + 2970, 30, {"loss": "model"}),
        ("prompts", "tokenize", "Generation prompts", x + 2970, 390, original["prompts"]["params"]),
        ("generate", "generate", "Generate with the connected decoder", x + 3300, 30, {"max_new_tokens": 8, "do_sample": False}),
        ("pipeline", "pipeline", "Decode generated text", x + 3630, 30, original["pipeline"]["params"]),
    ):
        n(identity, kind, title, dx, dy, **copy.deepcopy(params))
    e("tokenizer", "data", "tokenizer", "tokenizer")
    e("model", "initial-eval", "model", "model")
    e("data", "initial-eval", "data", "data")
    e("initial-eval", "train", "model", "model")
    e("data", "train", "data", "data")
    e("optimizer", "train", "optimizer", "optimizer")
    for source, target in (("train", "trained-eval"), ("trained-eval", "inspect"),
                           ("inspect", "checkpoint"), ("reload", "reload-eval"),
                           ("reload-eval", "generate"), ("generate", "pipeline")):
        e(source, target, "model", "model")
    for target in ("trained-eval", "reload-eval"):
        e("data", target, "data", "data")
    e("checkpoint", "reload", "artifact", "artifact")
    e("tokenizer", "prompts", "tokenizer", "tokenizer")
    e("prompts", "generate", "data", "data")
    e("tokenizer", "pipeline", "tokenizer", "tokenizer")
    return {"name": "Llama built from editable component blocks" + (" · custom gate" if custom else ""),
            "description": "Every embedding, projection, RMSNorm, RoPE, attention, SwiGLU operation and residual is connected visually. The compiled graph replays for real training, generation and safetensors reload.",
            "nodes": nodes, "edges": edges, "view": {"x": 25, "y": 40, "zoom": 0.55}}
