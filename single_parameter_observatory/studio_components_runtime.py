"""Trusted tensor-DAG compiler used by the isolated Studio tensor worker.

Graph records are data, never Python modules or executable source. Every forward
replays the connected operations with the same registered, trainable parameters.
The sole user-language extension is the worker's statically checked TensorProgram.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any


INPUT_PORTS = {
    "hf.tensor_input": (), "hf.op_positions": ("x",), "hf.op_embedding": ("x",), "hf.op_linear": ("x",),
    "hf.op_rmsnorm": ("x",), "hf.op_rope": ("x",), "hf.op_attention": ("q", "k", "v"),
    "hf.op_silu": ("x",), "hf.op_multiply": ("a", "b"), "hf.op_add": ("a", "b"),
    "hf.op_swiglu": ("x",), "hf.op_custom": ("x",),
}
PARAMETERS = {
    "hf.tensor_input": {"field", "dtype"}, "hf.op_positions": set(), "hf.op_embedding": {"num_embeddings", "embedding_dim"},
    "hf.op_linear": {"in_features", "out_features", "bias"}, "hf.op_rmsnorm": {"normalized_shape", "eps"},
    "hf.op_rope": {"num_heads", "head_dim", "theta"},
    "hf.op_attention": {"num_heads", "num_key_value_heads", "head_dim", "causal", "dropout"},
    "hf.op_silu": set(), "hf.op_multiply": set(), "hf.op_add": set(),
    "hf.op_swiglu": {"hidden_size", "intermediate_size", "bias"}, "hf.op_custom": {"code", "layers"},
}
MAX_NODES = 256
OPTIONAL_PORTS = {"hf.op_rope": ("positions",), "hf.op_attention": ("mask",), "hf.op_custom": ("a", "b")}


def _integer(value: Any, name: str, minimum: int = 1, maximum: int = 262144) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum:g} and {maximum:g}")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def validate_expression(value: Any) -> dict[str, Any]:
    """Validate full topology, including untrusted imported checkpoint recipes."""
    if (type(value) is not dict or set(value) != {"kind", "nodes", "output"}
            or value.get("kind") != "tensor-expression"):
        raise ValueError("A connected tensor-expression is required")
    nodes = value.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
        raise ValueError(f"Tensor expression requires 1 to {MAX_NODES} operations")
    seen: set[str] = set()
    for node in nodes:
        if type(node) is not dict or set(node) != {"id", "op", "inputs", "params"}:
            raise ValueError("Tensor operation records require id, op, inputs and params")
        identifier = node["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier) or identifier in seen:
            raise ValueError("Tensor operation IDs must be unique public identifiers")
        operation = node["op"]
        if operation not in INPUT_PORTS:
            raise ValueError(f"Unsupported tensor operation: {operation}")
        if (type(node["inputs"]) is not dict or set(INPUT_PORTS[operation]) - set(node["inputs"])
                or set(node["inputs"]) - set(INPUT_PORTS[operation]) - set(OPTIONAL_PORTS.get(operation, ()))):
            raise ValueError(f"{operation} requires connected inputs {INPUT_PORTS[operation]}")
        if any(not isinstance(source, str) or source not in seen for source in node["inputs"].values()):
            raise ValueError("Tensor DAG inputs must reference preceding connected operations; cycles are not allowed")
        if type(node["params"]) is not dict or set(node["params"]) - PARAMETERS[operation]:
            raise ValueError(f"Unknown parameters for {operation}")
        seen.add(identifier)
    if not isinstance(value["output"], str) or value["output"] not in seen:
        raise ValueError("Tensor expression output must reference a connected operation")
    # Reject disconnected hidden instructions in hand-written checkpoint recipes.
    required = {value["output"]}
    for node in reversed(nodes):
        if node["id"] in required:
            required.update(node["inputs"].values())
    if required != seen:
        raise ValueError("Tensor expression contains disconnected operations")
    return value


def append_expression(operation: str, inputs: dict[str, Any], params: dict[str, Any], identifier: str) -> dict[str, Any]:
    """Combine shared branches by identity, preserving shared module parameters."""
    if operation not in INPUT_PORTS:
        raise ValueError("Unknown tensor expression operation")
    nodes: list[dict[str, Any]] = []
    existing: dict[str, dict[str, Any]] = {}
    connected: dict[str, str] = {}
    for port in (*INPUT_PORTS[operation], *(key for key in OPTIONAL_PORTS.get(operation, ()) if key in inputs)):
        expression = validate_expression(inputs.get(port))
        connected[port] = expression["output"]
        for node in expression["nodes"]:
            if node["id"] in existing:
                if existing[node["id"]] != node:
                    raise ValueError("Conflicting tensor operation definitions share the same ID")
                continue
            existing[node["id"]] = node
            nodes.append(copy.deepcopy(node))
    while identifier in existing:
        identifier += "n"
    nodes.append({"id": identifier, "op": operation, "inputs": connected, "params": copy.deepcopy(params)})
    return validate_expression({"kind": "tensor-expression", "nodes": nodes, "output": identifier})


def imported_expression(value: Any, known: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Namespace foreign IDs without changing recognized live branch identities.

    Plugin literals can use arbitrary valid IDs, including future worker counter
    IDs. Content plus original identity and canonical ancestry supplies a stable
    foreign namespace. Changed copies cannot overwrite a registered live node.
    """
    expression = validate_expression(value)
    remapped: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []
    for source in expression["nodes"]:
        identifier = source["id"]
        node = copy.deepcopy(source)
        node["inputs"] = {port: remapped[parent] for port, parent in source["inputs"].items()}
        if known.get(identifier) != node:
            raw = json.dumps([identifier, node["op"], node["inputs"], node["params"]], sort_keys=True,
                             ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
            identifier = "import_" + hashlib.sha256(raw).hexdigest()[:40]
            node["id"] = identifier
            if identifier in known and known[identifier] != node:
                raise ValueError("Imported tensor operation identity conflicts with a registered operation")
        remapped[source["id"]] = identifier
        nodes.append(node)
    return validate_expression({"kind": "tensor-expression", "nodes": nodes, "output": remapped[expression["output"]]})


def build_model(owner: Any, recipe: dict[str, Any], program_type: Any) -> Any:
    """Compile a bounded, trainable causal language model from visible operations."""
    torch, nn = owner.torch, owner.torch.nn
    expression = validate_expression(recipe.get("expression"))
    vocab = _integer(recipe.get("vocab_size", 32), "vocab_size", 4, 100000)
    context = _integer(recipe.get("max_positions", 128), "max_positions", 2, owner.limits["max_sequence_length"])
    specials: dict[str, int | None] = {}
    for field, default in (("pad_token_id", 0), ("bos_token_id", 2), ("eos_token_id", 3)):
        value = recipe.get(field, default)
        specials[field] = None if value is None else _integer(value, field, 0, vocab - 1)
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    specs: dict[str, dict[str, Any]] = {}
    tensor_dtypes = {"int64": torch.int64, "float32": torch.float32, "bool": torch.bool}

    def apply_rope(x: Any, num_heads: Any, head_dim: Any, theta: Any = 10000,
                   positions: Any = None) -> Any:
        heads = _integer(num_heads, "RoPE num_heads", 1, 256)
        dimension = _integer(head_dim, "RoPE head_dim", 2, 1024)
        theta = _number(theta, "RoPE theta", 1, 1e12)
        if dimension % 2 or x.dim() != 3 or x.shape[-1] != heads * dimension:
            raise ValueError("RoPE requires even head_dim and [batch, sequence, num_heads * head_dim]")
        batch, length = x.shape[:2]
        if positions is None or positions.shape != x.shape[:2]:
            raise ValueError("RoPE positions must match [batch, sequence]")
        if positions.dtype not in (torch.int64, torch.int32):
            raise ValueError("RoPE positions must contain integer position IDs")
        if positions.device.type != "meta" and ((positions < 0) | (positions >= context)).any().item():
            raise ValueError("RoPE positions must be within the context window")
        headed = x.reshape(batch, length, heads, dimension).transpose(1, 2)
        frequency = 1.0 / (theta ** (torch.arange(0, dimension, 2, device=x.device, dtype=torch.float32) / dimension))
        angle = positions.float().unsqueeze(-1) * frequency
        angle = torch.cat((angle, angle), -1).unsqueeze(1)
        rotated = torch.cat((-headed[..., dimension // 2:], headed[..., :dimension // 2]), -1)
        return (headed * angle.cos().to(x.dtype) + rotated * angle.sin().to(x.dtype)).transpose(1, 2).reshape(batch, length, -1)

    def apply_attention(q: Any, k: Any, v: Any, num_heads: Any, num_key_value_heads: Any, head_dim: Any,
                        causal: Any = True, dropout: Any = 0, mask: Any = None, training: bool = False) -> Any:
        heads = _integer(num_heads, "Attention num_heads", 1, 256)
        kv = _integer(num_key_value_heads, "Attention num_key_value_heads", 1, heads)
        dimension = _integer(head_dim, "Attention head_dim", 1, 1024)
        _boolean(causal, "causal")
        dropout = _number(dropout, "dropout", 0, 0.99)
        if heads % kv or q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError("Attention requires grouped-query head divisibility and rank-three Q/K/V")
        batch, length = q.shape[:2]
        if (q.shape[-1] != heads * dimension or k.shape[-1] != kv * dimension or v.shape[-1] != kv * dimension
                or k.shape[:2] != q.shape[:2] or v.shape[:2] != q.shape[:2]):
            raise ValueError("Attention Q/K/V shapes do not match the declared head geometry")
        if batch * heads * length * length > owner.limits["max_tensor_elements"]:
            raise ValueError("Attention scores exceed the configured tensor element budget")
        if mask is None:
            mask = torch.ones((batch, length), device=q.device, dtype=torch.bool)
        if mask.shape != q.shape[:2]:
            raise ValueError("Attention mask must match [batch, sequence]")
        if mask.device.type != "meta" and (not ((mask == 0) | (mask == 1)).all().item() or not (mask.sum(-1) > 0).all().item()):
            raise ValueError("Attention mask must be binary with an unmasked token per row")
        q = q.reshape(batch, length, heads, dimension).transpose(1, 2)
        k = k.reshape(batch, length, kv, dimension).transpose(1, 2).repeat_interleave(heads // kv, dim=1)
        v = v.reshape(batch, length, kv, dimension).transpose(1, 2).repeat_interleave(heads // kv, dim=1)
        allowed = mask.bool()[:, None, None, :].expand(batch, 1, length, length)
        if causal:
            allowed = allowed & torch.ones((length, length), device=q.device, dtype=torch.bool).tril()[None, None]
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                        dropout_p=dropout if training else 0.0, is_causal=False)
        return attended.transpose(1, 2).reshape(batch, length, -1) * mask.unsqueeze(-1)

    class LlamaRMSNorm(nn.Module):  # type: ignore[name-defined]
        def __init__(self, width: int, epsilon: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))
            self.variance_epsilon = epsilon

        def forward(self, x: Any) -> Any:
            dtype = x.dtype
            value = x.float()
            normalized = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            return self.weight * normalized.to(dtype)

    class SwiGLU(nn.Module):  # type: ignore[name-defined]
        def __init__(self, width: int, intermediate: int, bias: bool) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(width, intermediate, bias=bias)
            self.up_proj = nn.Linear(width, intermediate, bias=bias)
            self.down_proj = nn.Linear(intermediate, width, bias=bias)

        def forward(self, x: Any) -> Any:
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class CustomTensorOperation(nn.Module):  # type: ignore[name-defined]
        def __init__(self, program: Any, layers: dict[str, Any]) -> None:
            super().__init__()
            self.layers = nn.ModuleDict({key: owner._layer(layers[key]) for key in sorted(layers)})
            self.program = program

        def forward(self, x: Any, attention_mask: Any = None, position_ids: Any = None, **incoming: Any) -> Any:
            def rotary(value: Any, num_heads: Any, head_dim: Any, theta: Any = 10000, positions: Any = None) -> Any:
                return apply_rope(value, num_heads, head_dim, theta, position_ids if positions is None else positions)

            def attention(q: Any, k: Any, v: Any, num_heads: Any, num_key_value_heads: Any, head_dim: Any,
                          causal: Any = True, dropout: Any = 0, mask: Any = None) -> Any:
                return apply_attention(q, k, v, num_heads, num_key_value_heads, head_dim, causal, dropout,
                                       attention_mask if mask is None else mask, self.training)

            return self.program({"x": x, "attention_mask": attention_mask, "position_ids": position_ids, **incoming},
                                self.layers, torch, training=self.training,
                                extra_functions={"rope": rotary, "causal_attention": attention})

    def module_for(node: dict[str, Any]) -> Any:
        spec = specs[node["id"]]
        operation = node["op"]
        if operation == "hf.op_embedding":
            return nn.Embedding(spec["num_embeddings"], spec["embedding_dim"])
        if operation == "hf.op_linear":
            return nn.Linear(spec["in_features"], spec["out_features"], bias=spec["bias"])
        if operation == "hf.op_rmsnorm":
            return LlamaRMSNorm(spec["normalized_shape"], spec["eps"])
        if operation == "hf.op_swiglu":
            return SwiGLU(spec["hidden_size"], spec["intermediate_size"], spec["bias"])
        if operation == "hf.op_custom":
            return CustomTensorOperation(spec["program"], spec["layers"])
        return nn.Identity()

    # Symbolic B=2,T=3 probes keep shape checks independent of actual user data.
    # All user-sized modules are constructed on meta before any CPU allocation.
    for node in expression["nodes"]:
        operation, params, identifier = node["op"], node["params"], node["id"]
        incoming = {port: shapes[source] for port, source in node["inputs"].items()}
        incoming_types = {port: dtypes[source] for port, source in node["inputs"].items()}
        spec: dict[str, Any] = {}
        shape: tuple[int, ...]
        if operation == "hf.tensor_input":
            field = params.get("field", "input_ids")
            if not isinstance(field, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", field):
                raise ValueError("Tensor input field must be a public field name")
            dtype = params.get("dtype", "int64")
            if dtype not in {"int64", "float32", "bool"}:
                raise ValueError("Tensor input dtype must be int64, float32 or bool")
            spec, shape = {"field": field, "dtype": dtype}, (2, 3)
        else:
            dtype = "float32"
            primary = "x" if "x" in incoming else "q" if "q" in incoming else "a"
            shape = incoming[primary]
            if operation == "hf.op_positions":
                if len(shape) != 2 or incoming_types["x"] not in {"int64", "bool"}:
                    raise ValueError("Position IDs require a connected integer or boolean attention mask [batch, sequence]")
                dtype = "int64"
            elif operation == "hf.op_embedding":
                if incoming_types["x"] != "int64" or len(shape) != 2:
                    raise ValueError("Embedding expects integer token IDs with shape [batch, sequence]")
                count = _integer(params.get("num_embeddings", 32), "num_embeddings", 4, 100000)
                width = _integer(params.get("embedding_dim", 32), "embedding_dim")
                spec, shape = {"num_embeddings": count, "embedding_dim": width}, (*shape, width)
            else:
                tensor_ports = set(incoming) - set(OPTIONAL_PORTS.get(operation, ())) if operation != "hf.op_custom" else set(incoming)
                if operation != "hf.op_custom" and (any(incoming_types[key] != "float32" for key in tensor_ports) or any(len(incoming[key]) != 3 for key in tensor_ports)):
                    raise ValueError(f"{operation} expects floating tensors [batch, sequence, width]")
                if operation == "hf.op_linear":
                    width = _integer(params.get("in_features", 32), "in_features")
                    output = _integer(params.get("out_features", 32), "out_features")
                    if shape[-1] != width:
                        raise ValueError(f"Linear in_features {width} does not match connected width {shape[-1]}")
                    spec = {"in_features": width, "out_features": output, "bias": _boolean(params.get("bias", False), "bias")}
                    shape = (*shape[:-1], output)
                elif operation == "hf.op_rmsnorm":
                    width = _integer(params.get("normalized_shape", 32), "normalized_shape")
                    if shape[-1] != width:
                        raise ValueError(f"RMSNorm normalized_shape {width} does not match connected width {shape[-1]}")
                    spec = {"normalized_shape": width, "eps": _number(params.get("eps", 1e-6), "eps", 1e-12, 1)}
                elif operation in {"hf.op_add", "hf.op_multiply"}:
                    if incoming["a"] != incoming["b"]:
                        raise ValueError("Residual/add and multiply inputs must have identical shapes; broadcasting is not implicit")
                elif operation == "hf.op_rope":
                    heads = _integer(params.get("num_heads", 4), "num_heads", 1, 256)
                    dimension = _integer(params.get("head_dim", 8), "head_dim", 2, 1024)
                    if dimension % 2:
                        raise ValueError("RoPE head_dim must be even for paired rotary coordinates")
                    if shape[-1] != heads * dimension:
                        raise ValueError("RoPE connected width must equal num_heads * head_dim")
                    if "positions" in incoming and (incoming["positions"] != shape[:2] or incoming_types["positions"] != "int64"):
                        raise ValueError("RoPE positions input must contain integer [batch, sequence] position IDs")
                    spec = {"num_heads": heads, "head_dim": dimension, "theta": _number(params.get("theta", 10000), "theta", 1, 1e12)}
                elif operation == "hf.op_attention":
                    heads = _integer(params.get("num_heads", 4), "num_heads", 1, 256)
                    kv = _integer(params.get("num_key_value_heads", 2), "num_key_value_heads", 1, heads)
                    dimension = _integer(params.get("head_dim", 8), "head_dim", 1, 1024)
                    if heads % kv:
                        raise ValueError("Attention num_heads must be divisible by num_key_value_heads")
                    if incoming["q"][-1] != heads * dimension or any(incoming[key][-1] != kv * dimension for key in ("k", "v")):
                        raise ValueError("Attention Q/K/V widths must match declared query and key/value head geometry")
                    if len({value[:2] for value in incoming.values()}) != 1:
                        raise ValueError("Attention Q/K/V must share batch and sequence dimensions")
                    if "mask" in incoming and (incoming["mask"] != shape[:2] or incoming_types["mask"] not in {"int64", "bool"}):
                        raise ValueError("Attention mask input must be integer or boolean [batch, sequence]")
                    shape = incoming["q"]
                    spec = {"num_heads": heads, "num_key_value_heads": kv, "head_dim": dimension,
                            "causal": _boolean(params.get("causal", True), "causal"),
                            "dropout": _number(params.get("dropout", 0), "dropout", 0, 0.99)}
                elif operation == "hf.op_swiglu":
                    width = _integer(params.get("hidden_size", 32), "hidden_size")
                    intermediate = _integer(params.get("intermediate_size", 64), "intermediate_size")
                    if shape[-1] != width:
                        raise ValueError("SwiGLU hidden_size must match its connected tensor width")
                    spec = {"hidden_size": width, "intermediate_size": intermediate,
                            "bias": _boolean(params.get("bias", False), "bias")}
                elif operation == "hf.op_custom":
                    if any(len(value) not in (2, 3) for value in incoming.values()):
                        raise ValueError("Custom tensor inputs must preserve [batch, sequence] with an optional width axis")
                    layers = params.get("layers", {})
                    if (type(layers) is not dict or len(layers) > 128
                            or any(not isinstance(key, str) or not key.isidentifier() or key.startswith("_") for key in layers)):
                        raise ValueError("Custom operation layers require at most 128 public identifier names")
                    program = program_type(params.get("code", "return inputs['x']"), set(layers), owner.limits["max_tensor_elements"])
                    spec = {"program": program, "layers": layers}
                    specs[identifier] = spec
                    with torch.device("meta"):
                        probe = module_for(node)
                        for length in (3, 5):
                            probe_inputs = {key: torch.empty((2, length, *value[2:]), dtype=tensor_dtypes[incoming_types[key]])
                                            for key, value in incoming.items()}
                            result = probe(attention_mask=torch.ones((2, length), dtype=torch.int64),
                                           position_ids=torch.zeros((2, length), dtype=torch.int64), **probe_inputs)
                            if not torch.is_tensor(result) or result.dim() not in (2, 3) or tuple(result.shape[:2]) != (2, length):
                                raise ValueError("Custom tensor operation must return [batch, sequence] or [batch, sequence, width] and preserve batch/sequence")
                            output_dtype = next((key for key, value in tensor_dtypes.items() if value == result.dtype), None)
                            if output_dtype is None:
                                raise ValueError("Custom tensor outputs must have bool, int64 or float32 dtype")
                            if length == 3:
                                shape = tuple(result.shape)
                                dtype = output_dtype
                            elif tuple(result.shape[2:]) != shape[2:] or output_dtype != dtype:
                                raise ValueError("Custom tensor operation output rank, width and dtype must not depend on sequence length")
                elif operation != "hf.op_silu":
                    raise ValueError("Unsupported tensor graph operation")
        specs[identifier], shapes[identifier], dtypes[identifier] = spec, shape, dtype
        if math.prod(shape) > owner.limits["max_tensor_elements"]:
            raise ValueError("Tensor operation shape exceeds the configured element budget")
    output_shape = shapes[expression["output"]]
    if len(output_shape) != 3 or output_shape[-1] != vocab or dtypes[expression["output"]] != "float32":
        raise ValueError("Graph model logits must have shape [batch, sequence, vocab_size]")
    if not any(node["op"] == "hf.tensor_input" and specs[node["id"]]["field"] == "input_ids" for node in expression["nodes"]):
        raise ValueError("Causal graph models require a connected input_ids tensor input")

    class GraphConfiguration:
        model_type = "studio-tensor-graph"
        is_encoder_decoder = False
        vocab_size = vocab
        max_position_embeddings = context
        pad_token_id = specials["pad_token_id"]
        bos_token_id = specials["bos_token_id"]
        eos_token_id = specials["eos_token_id"]

        def to_dict(self) -> dict[str, Any]:
            return {"model_type": self.model_type, "vocab_size": vocab, "max_position_embeddings": context, **specials}

    class TensorGraphCausalLM(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.operations = nn.ModuleDict({node["id"]: module_for(node) for node in expression["nodes"]})
            self.config = GraphConfiguration()
            self.expression = copy.deepcopy(expression)
            self.operation_shapes = {key: ["batch", "sequence", *value[2:]] for key, value in shapes.items()}

        def get_input_embeddings(self) -> Any:
            for node in expression["nodes"]:
                if node["op"] == "hf.op_embedding":
                    source = node["inputs"]["x"]
                    if specs[source].get("field") == "input_ids":
                        return self.operations[node["id"]]
            return None

        def forward(self, **fields: Any) -> dict[str, Any]:
            ids = fields.get("input_ids")
            if ids is None or ids.dim() != 2 or ids.dtype not in (torch.int64, torch.int32):
                raise ValueError("Tensor graph input_ids must be integer [batch, sequence]")
            batch, length = ids.shape
            if not 1 <= length <= context or batch > owner.limits["max_batch_size"]:
                raise ValueError("Tensor graph input exceeds batch or context limits")
            if ((ids < 0) | (ids >= vocab)).any().item():
                raise ValueError("Token IDs must be within this graph model's input vocabulary")
            mask = fields.get("attention_mask", torch.ones_like(ids))
            if mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all().item() or not (mask.sum(-1) > 0).all().item():
                raise ValueError("Attention mask must match input_ids with at least one unmasked token per row")
            positions = fields.get("position_ids")
            if positions is None:
                positions = (mask.long().cumsum(-1) - 1).clamp(min=0)
            elif positions.shape != ids.shape or positions.dtype not in (torch.int32, torch.int64) or ((positions < 0) | (positions >= context)).any().item():
                raise ValueError("position_ids must be integer [batch, sequence] within the context window")
            values: dict[str, Any] = {}
            for node in expression["nodes"]:
                identifier, operation = node["id"], node["op"]
                spec = specs[identifier]
                expected = (batch, length, *shapes[identifier][2:])
                if math.prod(expected) > owner.limits["max_tensor_elements"]:
                    raise ValueError(f"{operation} output exceeds the configured tensor element budget")
                if operation == "hf.op_swiglu" and batch * length * spec["intermediate_size"] > owner.limits["max_tensor_elements"]:
                    raise ValueError("SwiGLU intermediate activation exceeds the tensor element budget")
                if operation == "hf.op_custom":
                    for module in self.operations[identifier].modules():
                        width = getattr(module, "out_features", getattr(module, "embedding_dim", None))
                        if width is not None and batch * length * width > owner.limits["max_tensor_elements"]:
                            raise ValueError("Custom layer activation exceeds the tensor element budget")
                incoming = {key: values[source] for key, source in node["inputs"].items()}
                if operation == "hf.tensor_input":
                    value = fields.get(spec["field"])
                    if value is None:
                        value = mask if spec["field"] == "attention_mask" else positions if spec["field"] == "position_ids" else None
                    if value is None or not torch.is_tensor(value) or value.shape != ids.shape:
                        raise ValueError(f"Tensor input {spec['field']} must match [batch, sequence]")
                    if spec["dtype"] == "int64":
                        if value.is_floating_point() and not torch.equal(value, value.round()):
                            raise ValueError("Integer tensor inputs require integral values")
                        value = value.long()
                    elif spec["dtype"] == "bool":
                        value = value.bool()
                    else:
                        value = value.float()
                elif operation == "hf.op_positions":
                    if not ((incoming["x"] == 0) | (incoming["x"] == 1)).all().item():
                        raise ValueError("Position-ID input must be a binary attention mask")
                    value = (incoming["x"].long().cumsum(-1) - 1).clamp(min=0)
                elif operation == "hf.op_custom":
                    value = self.operations[identifier](attention_mask=mask, position_ids=positions, **incoming)
                elif operation in {"hf.op_embedding", "hf.op_linear", "hf.op_rmsnorm", "hf.op_swiglu"}:
                    if operation == "hf.op_embedding" and ((incoming["x"] < 0) | (incoming["x"] >= spec["num_embeddings"])).any().item():
                        raise ValueError("Embedding token IDs exceed num_embeddings")
                    value = self.operations[identifier](incoming["x"])
                elif operation == "hf.op_silu":
                    value = torch.nn.functional.silu(incoming["x"])
                elif operation == "hf.op_add":
                    value = incoming["a"] + incoming["b"]
                elif operation == "hf.op_multiply":
                    value = incoming["a"] * incoming["b"]
                elif operation == "hf.op_rope":
                    explicit_positions = incoming.get("positions", positions)
                    if ((explicit_positions < 0) | (explicit_positions >= context)).any().item():
                        raise ValueError("Connected RoPE position IDs must be within the context window")
                    value = apply_rope(incoming["x"], **spec, positions=explicit_positions)
                elif operation == "hf.op_attention":
                    explicit_mask = incoming.get("mask", mask)
                    if not ((explicit_mask == 0) | (explicit_mask == 1)).all().item() or not (explicit_mask.sum(-1) > 0).all().item():
                        raise ValueError("Connected attention mask must be binary with an unmasked token per row")
                    value = apply_attention(incoming["q"], incoming["k"], incoming["v"], **spec,
                                            mask=explicit_mask, training=self.training)
                else:
                    raise ValueError("Unsupported tensor graph operation")
                if not torch.is_tensor(value) or tuple(value.shape) != expected:
                    raise ValueError(f"{operation} produced {getattr(value, 'shape', None)}; expected {expected}")
                if value.dtype != tensor_dtypes[dtypes[identifier]]:
                    raise ValueError(f"{operation} produced {value.dtype}; expected {dtypes[identifier]}")
                owner._bound_tensor(value)
                values[identifier] = value
            return {"logits": values[expression["output"]]}

        def generate(self, input_ids: Any, attention_mask: Any = None, max_new_tokens: int = 8,
                     do_sample: bool = False, temperature: float = 1.0, **kwargs: Any) -> Any:
            if kwargs:
                raise ValueError(f"Tensor graph generation does not accept fields {sorted(kwargs)}")
            maximum = _integer(max_new_tokens, "max_new_tokens", 1, 128)
            _boolean(do_sample, "do_sample")
            temperature = _number(temperature, "temperature", 1e-6, 100)
            if input_ids.shape[1] + maximum > context:
                raise ValueError("Generation input plus new tokens exceeds the graph context window")
            tokens = input_ids.clone()
            mask = torch.ones_like(tokens) if attention_mask is None else attention_mask.clone()
            # Also normalize right padding for callers outside the worker wrapper.
            for row_index, row in enumerate(tokens):
                valid = row[mask[row_index].bool()]
                if valid.numel() == 0:
                    raise ValueError("Generation requires at least one unmasked token per row")
                tokens[row_index].fill_(specials["pad_token_id"] if specials["pad_token_id"] is not None else 0)
                tokens[row_index, -len(valid):] = valid
                mask[row_index].zero_()
                mask[row_index, -len(valid):] = 1
            finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
            for _ in range(maximum):
                logits = self(input_ids=tokens, attention_mask=mask)["logits"][:, -1, :]
                next_token = torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(-1) if do_sample else logits.argmax(-1)
                if finished.any().item():
                    pad = specials["pad_token_id"] if specials["pad_token_id"] is not None else specials["eos_token_id"]
                    next_token = torch.where(finished, torch.full_like(next_token, pad if pad is not None else 0), next_token)
                active = ~finished
                tokens = torch.cat((tokens, next_token[:, None]), -1)
                mask = torch.cat((mask, active.long()[:, None]), -1)
                if specials["eos_token_id"] is not None:
                    finished = finished | (next_token == specials["eos_token_id"])
                    if finished.all().item():
                        break
            return tokens

    return TensorGraphCausalLM()


def causal_loss(owner: Any, model: Any, fields: dict[str, Any]) -> tuple[Any, Any]:
    """Next-token loss excludes padding and uses exactly the visible DAG logits."""
    torch = owner.torch
    result = model(**{key: value for key, value in fields.items() if key != "labels"})
    logits = result["logits"]
    labels = fields.get("labels")
    if labels is None or labels.shape != logits.shape[:-1] or labels.shape[1] < 2:
        raise ValueError("Causal graph training requires labels [batch, sequence] and at least two positions")
    if labels.is_floating_point() and not torch.equal(labels, labels.round()):
        raise ValueError("Causal graph labels must be integer token IDs or -100")
    labels = labels.long()
    if ((labels != -100) & ((labels < 0) | (labels >= logits.shape[-1]))).any().item():
        raise ValueError("Causal graph labels must be vocabulary token IDs or -100")
    targets = labels[:, 1:].clone()
    mask = fields.get("attention_mask")
    if mask is not None:
        targets[(mask[:, 1:] == 0) | (mask[:, :-1] == 0)] = -100
    if not (targets != -100).any().item():
        raise ValueError("Causal graph labels contain no unmasked next-token targets")
    loss = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
    return loss, logits
