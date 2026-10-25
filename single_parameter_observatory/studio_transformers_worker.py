"""Trusted CPU tensor-operation worker. User source is interpreted, never executed.

Launched with isolated CPython and explicit dependency paths. No plugin imports,
downloads, pickle, shell, user callbacks or arbitrary Python execution are used.
The process boundary supplies cancellation/resource isolation, not an OS VM.
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import uuid
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_LIMITS = {"max_parameters": 2_000_000, "max_tensor_elements": 1_000_000, "worker_memory_mb": 2048,
                  "max_steps": 2000, "max_sequence_length": 512, "max_batch_size": 128}
TASK_CLASSES = {"base": "AutoModel", "causal-lm": "AutoModelForCausalLM", "seq2seq-lm": "AutoModelForSeq2SeqLM",
                "sequence-classification": "AutoModelForSequenceClassification", "token-classification": "AutoModelForTokenClassification",
                "question-answering": "AutoModelForQuestionAnswering", "masked-lm": "AutoModelForMaskedLM",
                "image-classification": "AutoModelForImageClassification", "audio-classification": "AutoModelForAudioClassification"}


def components_runtime() -> Any:
    """Load only the trusted sibling compiler, including in isolated -I workers."""
    name = "_spobs_trusted_components_runtime"
    if name not in sys.modules:
        path = Path(__file__).with_name("studio_components_runtime.py")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Trusted tensor-component compiler is unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def number(value: Any, name: str, low: float = -1e30, high: float = 1e30) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low:g} and {high:g}")
    return float(value)


def strict_object(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict or any(not isinstance(key, str) or key.startswith("_") for key in value):
        raise ValueError(f"{name} must be an object with public string keys")
    return value


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def json_load(raw: bytes | str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON keys are not allowed")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"Nonfinite JSON value {value} is not allowed")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


class TensorProgram:
    """A small tensor expression language with Python syntax and static validation.

    There is deliberately no exec, eval, attribute access, loop, comprehension,
    import, reflection, module lookup, user callable or mutation of inputs.
    """

    FUNCTIONS = frozenset({"relu", "gelu", "silu", "tanh", "sigmoid", "softmax", "log_softmax", "matmul", "add", "multiply",
                           "reshape", "transpose", "permute", "mean", "sum", "concat", "stack", "flatten", "dropout", "normalize",
                           "sqrt", "rsqrt", "pow", "cos", "sin", "cumsum", "clamp", "rope", "causal_attention"})

    def __init__(self, code: Any, layer_names: set[str], maximum_elements: int) -> None:
        if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > 16384:
            raise ValueError("Custom forward code must be nonempty and at most 16 KiB")
        self.tree = ast.parse(code, mode="exec")
        self.maximum_elements = maximum_elements
        self.layer_names = layer_names
        if len(list(ast.walk(self.tree))) > 1000 or len(self.tree.body) > 128:
            raise ValueError("Custom forward program is too large")
        self.variables = {"inputs", "layers"}
        returned = False
        for statement in self.tree.body:
            if returned:
                raise ValueError("Statements after return are not allowed")
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name.startswith("_") or name in {"inputs", "layers"} | self.FUNCTIONS:
                    raise ValueError("Custom variables must have public names and cannot replace inputs, layers or functions")
                self._validate(statement.value)
                self.variables.add(name)
            elif isinstance(statement, ast.Return) and statement.value is not None:
                self._validate(statement.value)
                returned = True
            else:
                raise ValueError("Custom forward supports only single-name assignments and a final return")
        if not returned:
            raise ValueError("Custom forward must end with return")

    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float, bool, str, type(None)):
                raise ValueError("Unsupported custom-code literal")
            if isinstance(node.value, str) and len(node.value) > 128:
                raise ValueError("Custom string literals are limited to 128 characters")
            if type(node.value) in (int, float):
                number(node.value, "custom literal")
        elif isinstance(node, ast.Name):
            if node.id not in self.variables:
                raise ValueError(f"Unknown custom forward variable: {node.id}")
        elif isinstance(node, (ast.Tuple, ast.List)):
            if len(node.elts) > 32:
                raise ValueError("Custom lists/tuples are limited to 32 entries")
            for item in node.elts:
                self._validate(item)
        elif isinstance(node, ast.Dict):
            if len(node.keys) > 32 or any(not isinstance(key, ast.Constant) or not isinstance(key.value, str) for key in node.keys):
                raise ValueError("Custom dictionaries require at most 32 literal string keys")
            for item in node.values:
                self._validate(item)
        elif isinstance(node, ast.Subscript):
            self._validate(node.value)
            self._validate(node.slice)
        elif isinstance(node, ast.Slice):
            for slice_part in (node.lower, node.upper, node.step):
                if slice_part is not None:
                    self._validate(slice_part)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._validate(node.operand)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.MatMult)):
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self.FUNCTIONS:
                pass
            elif (isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "layers" and isinstance(node.func.slice, ast.Constant)
                  and node.func.slice.value in self.layer_names):
                pass
            else:
                raise ValueError("Only declared layer calls and documented tensor functions are allowed")
            if len(node.args) > 16 or any(keyword.arg is None or keyword.arg.startswith("_") for keyword in node.keywords):
                raise ValueError("Argument unpacking and private keywords are not allowed")
            for item in node.args:
                self._validate(item)
            for keyword in node.keywords:
                self._validate(keyword.value)
        else:
            raise ValueError(f"Unsupported custom forward syntax: {type(node).__name__}")

    def __call__(self, inputs: dict[str, Any], layers: Any, torch: Any, training: bool = True,
                 extra_functions: dict[str, Any] | None = None) -> Any:
        functional = torch.nn.functional
        functions = {"relu": functional.relu, "gelu": functional.gelu, "silu": functional.silu, "tanh": torch.tanh,
                     "sigmoid": torch.sigmoid, "softmax": functional.softmax, "log_softmax": functional.log_softmax,
                     "matmul": torch.matmul, "add": torch.add, "multiply": torch.mul, "reshape": torch.reshape,
                     "transpose": torch.transpose, "permute": torch.permute, "mean": torch.mean, "sum": torch.sum,
                     "concat": torch.cat, "stack": torch.stack, "flatten": torch.flatten,
                     "dropout": lambda x, p=0.5: functional.dropout(x, p=p, training=training),
                     "normalize": functional.normalize, "sqrt": torch.sqrt, "rsqrt": torch.rsqrt, "pow": torch.pow,
                     "cos": torch.cos, "sin": torch.sin, "cumsum": torch.cumsum, "clamp": torch.clamp,
                     **(extra_functions or {})}
        variables = {"inputs": inputs, "layers": layers}

        def bounded(result: Any) -> Any:
            if torch.is_tensor(result) and result.numel() > self.maximum_elements:
                raise ValueError("Custom forward tensor exceeds the tensor element budget")
            return result

        def expression(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return variables[node.id]
            if isinstance(node, ast.List):
                return [expression(item) for item in node.elts]
            if isinstance(node, ast.Tuple):
                return tuple(expression(item) for item in node.elts)
            if isinstance(node, ast.Dict):
                entries = {}
                for key, value in zip(node.keys, node.values, strict=True):
                    assert isinstance(key, ast.Constant)
                    entries[key.value] = expression(value)
                return entries
            if isinstance(node, ast.Slice):
                return slice(*(expression(item) if item is not None else None for item in (node.lower, node.upper, node.step)))
            if isinstance(node, ast.Subscript):
                container, key = expression(node.value), expression(node.slice)
                if not (torch.is_tensor(container) or isinstance(container, (Mapping, list, tuple, torch.nn.ModuleDict))):
                    raise ValueError("Only tensor, input-object and declared layer indexing is allowed")
                return bounded(container[key])
            if isinstance(node, ast.UnaryOp):
                item = expression(node.operand)
                return bounded(-item if isinstance(node.op, ast.USub) else +item)
            if isinstance(node, ast.BinOp):
                left, right = expression(node.left), expression(node.right)
                if not (torch.is_tensor(left) or torch.is_tensor(right)):
                    raise ValueError("Custom arithmetic must operate on tensors")
                if isinstance(node.op, ast.Add):
                    return bounded(left + right)
                if isinstance(node.op, ast.Sub):
                    return bounded(left - right)
                if isinstance(node.op, ast.Mult):
                    return bounded(left * right)
                if isinstance(node.op, ast.Div):
                    return bounded(left / right)
                return bounded(left @ right)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in functions:
                    raise ValueError(f"{node.func.id} is available only inside a connected custom tensor component")
                function = functions[node.func.id] if isinstance(node.func, ast.Name) else layers[node.func.slice.value]  # type: ignore[attr-defined]
                args = [expression(item) for item in node.args]
                kwargs = {str(keyword.arg): expression(keyword.value) for keyword in node.keywords}
                if any(key in {"out", "inplace"} for key in kwargs):
                    raise ValueError("Tensor out/inplace mutation is not allowed")
                return bounded(function(*args, **kwargs))
            raise ValueError("Unsupported tensor expression")

        for statement in self.tree.body:
            if isinstance(statement, ast.Assign):
                variables[statement.targets[0].id] = expression(statement.value)  # type: ignore[attr-defined]
            elif isinstance(statement, ast.Return):
                return expression(statement.value)  # type: ignore[arg-type]
        raise ValueError("Custom forward did not return a tensor")


class Runtime:
    """Trusted operations on real CPU models; JSON objects cannot resolve Python APIs."""

    def __init__(self, budgets: dict[str, int] | None = None) -> None:
        import torch  # noqa: PLC0415 -- initialize only inside a job worker or explicit test.

        self.torch = torch
        self.limits = {**DEFAULT_LIMITS, **(budgets or {})}
        self.models: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.datasets: dict[str, dict[str, Any]] = {}
        self.tokenizers: dict[str, Any] = {}
        self.local_tokenizers: set[str] = set()
        self.optimizers: dict[str, tuple[Any, str, dict[str, Any]]] = {}
        self.expression_counter = 0
        self.expression_nodes: dict[str, dict[str, Any]] = {}
        self.on_progress: Any = None
        torch.set_num_threads(1)

    def _bound_tensor(self, value: Any) -> None:
        if self.torch.is_tensor(value):
            if value.numel() > self.limits["max_tensor_elements"]:
                raise ValueError("Tensor exceeds the configured element budget")
            if value.is_floating_point() and not self.torch.isfinite(value).all().item():
                raise ValueError("Operation produced a nonfinite tensor")
        elif isinstance(value, Mapping):
            for item in value.values():
                self._bound_tensor(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._bound_tensor(item)

    def _handle(self, value: Any, registry: dict[str, Any], kind: str) -> str:
        if not isinstance(value, dict) or value.get("kind") != kind or value.get("handle") not in registry:
            raise ValueError(f"A live {kind} handle from this same workflow execution is required")
        return str(value["handle"])

    def _summary(self, value: Any) -> Any:
        torch = self.torch
        if torch.is_tensor(value):
            self._bound_tensor(value)
            flat = value.detach().cpu().reshape(-1)
            return {"shape": list(value.shape), "dtype": str(value.dtype), "elements": value.numel(),
                    "preview": flat[:64].tolist(), "truncated": value.numel() > 64}
        if isinstance(value, Mapping):
            return {str(key): self._summary(item) for key, item in value.items() if item is not None}
        if isinstance(value, (tuple, list)):
            return [self._summary(item) for item in value[:32]]
        if value is None or type(value) in (str, bool, int, float):
            return value
        return str(type(value).__name__)

    def _pipeline_json(self, value: Any, depth: int = 0) -> Any:
        """HF task pipelines may expose NumPy scalar scores, not Python floats."""
        import numpy as np  # noqa: PLC0415 -- normalize trusted pipeline outputs at the JSON boundary.

        if depth > 32:
            raise ValueError("Pipeline output nesting exceeds its budget")
        if isinstance(value, np.generic):
            return self._pipeline_json(value.item(), depth + 1)
        if isinstance(value, np.ndarray):
            if value.size > self.limits["max_tensor_elements"]:
                raise ValueError("Pipeline array exceeds the tensor element budget")
            return self._pipeline_json(value.tolist(), depth + 1)
        if isinstance(value, Mapping):
            return {str(key): self._pipeline_json(item, depth + 1) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            if len(value) > self.limits["max_tensor_elements"]:
                raise ValueError("Pipeline output list exceeds its element budget")
            return [self._pipeline_json(item, depth + 1) for item in value]
        if value is None or type(value) in (str, bool, int, float):
            if type(value) is float and not math.isfinite(value):
                raise ValueError("Pipeline output contains a nonfinite number")
            return value
        if self.torch.is_tensor(value):
            return self._summary(value)
        raise ValueError(f"Unsupported pipeline output value: {type(value).__name__}")

    def _hash(self, model: Any) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(json_bytes([name, list(tensor.shape), str(tensor.dtype)]))
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _model_descriptor(self, handle: str) -> dict[str, Any]:
        model, recipe = self.models[handle]
        return {"kind": "model", "handle": handle, "architecture": type(model).__name__, "family": recipe["family"],
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "parameter_tensors": len(list(model.parameters())), "device": "cpu"}

    def _architecture_report(self, model: Any) -> dict[str, Any]:
        """Observe the instantiated module tree, rather than infer it from recipes.

        Containers include descendant counts; direct counts avoid counting those
        descendants again when a viewer adds rows. Shared modules follow PyTorch's
        canonical named_modules traversal. Output remains bounded for deep models.
        """
        modules: list[dict[str, Any]] = []
        module_count = 0
        for name, module in model.named_modules():
            module_count += 1
            if len(modules) >= 256:
                continue
            parameters = list(module.parameters())
            children = [f"{name}.{child}" if name else child for child, _ in module.named_children()]
            modules.append({"name": name, "parent": name.rpartition(".")[0] if name else None,
                            "type": type(module).__name__, "depth": name.count(".") + 1 if name else 0,
                            "direct_parameter_count": sum(parameter.numel() for parameter in module.parameters(recurse=False)),
                            "total_parameter_count": sum(parameter.numel() for parameter in parameters),
                            "parameter_tensors": len(parameters),
                            "trainable_parameters": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
                            "children": children[:256], "children_truncated": len(children) > 256})
        result: dict[str, Any] = {"modules": modules, "module_count": module_count,
                                  "modules_truncated": module_count > len(modules)}
        config = getattr(model, "config", None)
        if config is not None:
            # Read the actual resolved HF configuration, including normalized
            # RoPE parameters; user recipes can omit architecture defaults.
            fields = {"model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                      "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
                      "hidden_act", "rms_norm_eps", "rope_theta", "rope_parameters", "tie_word_embeddings",
                      "attention_bias", "mlp_bias", "attention_dropout", "bos_token_id", "eos_token_id", "pad_token_id"}
            result["config"] = {key: value for key, value in config.to_dict().items() if key in fields}
        return result

    def _guard_model(self, model: Any) -> None:
        count = sum(parameter.numel() for parameter in model.parameters())
        if count > self.limits["max_parameters"]:
            raise ValueError(f"Model has {count:,} parameters; configured budget is {self.limits['max_parameters']:,}")
        buffers = sum(buffer.numel() for buffer in model.buffers())
        if buffers > self.limits["max_tensor_elements"] * 4:
            raise ValueError("Model buffers exceed the configured tensor budget")
        if count == 0:
            raise ValueError("A trainable model must contain at least one parameter")

    def _layer(self, recipe: Any) -> Any:
        value = strict_object(recipe, "Layer recipe")
        name = value.get("type")
        if not isinstance(name, str):
            raise ValueError("Layer recipe requires a type")
        torch = self.torch
        nn = torch.nn
        def widths(key: str) -> int:
            return integer(value.get(key), key, 1, 262144)
        if name == "Linear":
            allowed = {"type", "in_features", "out_features", "bias"}
            args = {"in_features": widths("in_features"), "out_features": widths("out_features"), "bias": value.get("bias", True)}
            if type(args["bias"]) is not bool:
                raise ValueError("bias must be boolean")
            factory = nn.Linear
        elif name == "Embedding":
            allowed = {"type", "num_embeddings", "embedding_dim"}
            args = {"num_embeddings": widths("num_embeddings"), "embedding_dim": widths("embedding_dim")}
            factory = nn.Embedding
        elif name in {"RMSNorm", "LayerNorm"}:
            allowed = {"type", "normalized_shape", "eps"}
            args = {"normalized_shape": widths("normalized_shape"), "eps": number(value.get("eps", 1e-6 if name == "RMSNorm" else 1e-5), "eps", 1e-12, 1)}
            factory = nn.RMSNorm if name == "RMSNorm" else nn.LayerNorm
        elif name in {"GELU", "ReLU", "SiLU", "Tanh", "Sigmoid", "Identity"}:
            allowed, args = {"type"}, {}
            factory = getattr(nn, name)
        elif name == "Dropout":
            allowed, args, factory = {"type", "p"}, {"p": number(value.get("p", 0.1), "dropout probability", 0, 0.99)}, nn.Dropout
        elif name == "Flatten":
            allowed = {"type", "start_dim", "end_dim"}
            args = {"start_dim": integer(value.get("start_dim", 1), "start_dim", -8, 8),
                    "end_dim": integer(value.get("end_dim", -1), "end_dim", -8, 8)}
            factory = nn.Flatten
        elif name == "SelfAttention":
            allowed = {"type", "embed_dim", "num_heads", "dropout", "causal"}
            embed, heads = widths("embed_dim"), widths("num_heads")
            if embed % heads:
                raise ValueError("Attention embed_dim must be divisible by num_heads")
            dropout = number(value.get("dropout", 0), "dropout", 0, 0.99)
            causal = value.get("causal", True)
            if type(causal) is not bool:
                raise ValueError("causal must be boolean")
            budget = self.limits["max_tensor_elements"]

            class SelfAttention(nn.Module):  # type: ignore[name-defined]  # Runtime-local trusted PyTorch type.
                def __init__(self) -> None:
                    super().__init__()
                    self.attention = nn.MultiheadAttention(embed, heads, dropout=dropout, batch_first=True)

                def forward(self, x: Any) -> Any:
                    if x.dim() != 3 or x.shape[-1] != embed:
                        raise ValueError("SelfAttention expects [batch, sequence, embed_dim]")
                    if x.shape[0] * heads * x.shape[1] ** 2 > budget:
                        raise ValueError("Attention scores exceed the configured tensor element budget")
                    mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), diagonal=1) if causal else None
                    return self.attention(x, x, x, attn_mask=mask, need_weights=False)[0]

            if set(value) - allowed:
                raise ValueError(f"Unknown {name} layer fields: {sorted(set(value) - allowed)}")
            return SelfAttention()
        else:
            raise ValueError(f"Unsupported layer type: {name}")
        if set(value) - allowed:
            raise ValueError(f"Unknown {name} layer fields: {sorted(set(value) - allowed)}")
        return factory(**args)

    def _build(self, recipe: dict[str, Any]) -> Any:
        torch = self.torch
        family = recipe.get("family")
        if family == "tensor-graph":
            return components_runtime().build_model(self, recipe, TensorProgram)
        if family == "sequential":
            layers = recipe.get("layers")
            if not isinstance(layers, list) or not 1 <= len(layers) <= 128:
                raise ValueError("Sequential requires 1 to 128 layer recipes")
            return torch.nn.Sequential(*(self._layer(item) for item in layers))
        if family == "custom":
            named_layers: dict[str, Any] = strict_object(recipe.get("layers"), "Named layers")
            if not 1 <= len(named_layers) <= 128 or any(not key.isidentifier() or key.startswith("_") for key in named_layers):
                raise ValueError("Custom models require 1 to 128 layers with public identifier names")
            program = TensorProgram(recipe.get("code"), set(named_layers), self.limits["max_tensor_elements"])

            class CustomTensorModel(torch.nn.Module):  # type: ignore[name-defined]  # Runtime-local trusted PyTorch type.
                def __init__(self, owner: Runtime) -> None:
                    super().__init__()
                    self.layers = torch.nn.ModuleDict({key: owner._layer(value) for key, value in named_layers.items()})

                def forward(self, **inputs: Any) -> Any:
                    return program(inputs, self.layers, torch, training=self.training)

            return CustomTensorModel(self)
        if family == "huggingface":
            import transformers  # noqa: PLC0415 -- optional runtime operation, never plugin code.

            config = recipe.get("config")
            if not isinstance(config, dict) or config.get("kind") != "hf-config":
                raise ValueError("HF model requires an HF configuration recipe")
            values = strict_object(config.get("values"), "HF configuration")
            for field in ("auto_map", "trust_remote_code", "torch_dtype", "dtype", "architectures", "name_or_path"):
                if field in values:
                    raise ValueError(f"HF configuration field {field} is managed by the trusted runtime")
            self._validate_config_tree(values)
            if config["model_type"] == "llama":
                self._validate_llama_values(values, transformers.LlamaConfig)
            cfg = transformers.AutoConfig.for_model(config["model_type"], **values)
            if cfg.model_type == "llama":
                self._validate_llama_config(cfg)
            task = recipe.get("task")
            if task not in TASK_CLASSES:
                raise ValueError("Unsupported AutoModel task")
            cfg._attn_implementation = "eager"
            return getattr(transformers, TASK_CLASSES[task]).from_config(cfg, trust_remote_code=False)
        raise ValueError("Unknown model recipe family")

    @staticmethod
    def _validate_llama_values(values: dict[str, Any], config_type: Any) -> None:
        # HF configuration construction itself derives head_dim by division and
        # validates RoPE, so reject explicit invalid values before that work too.
        for field in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim"):
            if field in values and values[field] is not None:
                integer(values[field], "Llama " + field, 1, 262144)
        hidden = integer(values.get("hidden_size", config_type.hidden_size), "Llama hidden_size", 1, 262144)
        heads = integer(values.get("num_attention_heads", config_type.num_attention_heads), "Llama num_attention_heads", 1, 262144)
        if hidden % heads:
            raise ValueError("Llama hidden_size must be divisible by num_attention_heads")
        head_dim = values.get("head_dim")
        head_dim = integer(hidden // heads if head_dim is None else head_dim, "Llama head_dim", 1, 262144)
        if head_dim % 2:
            raise ValueError("Llama head_dim must be even for paired rotary position embeddings")

    @staticmethod
    def _validate_llama_config(config: Any) -> None:
        """Reject invalid native attention geometry before allocating any model."""
        integer(config.hidden_size, "Llama hidden_size", 1, 262144)
        integer(config.intermediate_size, "Llama intermediate_size", 1, 262144)
        heads = integer(config.num_attention_heads, "Llama num_attention_heads", 1, 262144)
        kv_heads = integer(config.num_key_value_heads, "Llama num_key_value_heads", 1, heads)
        head_dim = integer(config.head_dim, "Llama head_dim", 1, 262144)
        if heads % kv_heads:
            raise ValueError("Llama num_attention_heads must be divisible by num_key_value_heads for grouped-query attention")
        if head_dim % 2:
            raise ValueError("Llama head_dim must be even for paired rotary position embeddings")

    def _validate_config_tree(self, value: Any, depth: int = 0) -> None:
        if depth > 12:
            raise ValueError("HF config nesting is too deep")
        if type(value) is dict:
            if len(value) > 128 or any(not isinstance(key, str) or key.startswith("_") for key in value):
                raise ValueError("HF config must have bounded public fields")
            for key, item in value.items():
                if key in {"n_layer", "n_layers", "num_layers", "num_hidden_layers", "num_decoder_layers", "encoder_layers", "decoder_layers", "depth"}:
                    integer(item, key, 1, 128)
                self._validate_config_tree(item, depth + 1)
        elif type(value) is list:
            if len(value) > 256:
                raise ValueError("HF config lists are limited to 256 entries")
            for item in value:
                self._validate_config_tree(item, depth + 1)
        elif type(value) is int:
            integer(value, "HF config integer", -100000, 262144)
        elif type(value) is float:
            number(value, "HF config number", -1e6, 1e6)
        elif type(value) is str:
            if len(value) > 256:
                raise ValueError("HF config strings are limited to 256 characters")
        elif value is not None and type(value) is not bool:
            raise ValueError("HF configuration must be strict JSON")

    def _new_model(self, recipe: dict[str, Any], seed: Any = 42) -> dict[str, Any]:
        torch = self.torch
        integer(seed, "seed", 0, 2**31 - 1)
        with torch.device("meta"):
            probe = self._build(recipe)
        self._guard_model(probe)
        del probe
        torch.manual_seed(seed)
        model = self._build(recipe).cpu()
        self._guard_model(model)
        handle = uuid.uuid4().hex
        self.models[handle] = (model, {**recipe, "seed": seed})
        return self._model_descriptor(handle)

    def _new_data(self, fields: Any) -> dict[str, Any]:
        fields = strict_object(fields, "Dataset fields")
        if not 1 <= len(fields) <= 32:
            raise ValueError("A tensor dataset requires 1 to 32 named fields")
        tensors: dict[str, Any] = {}
        total = 0
        for key, value in fields.items():
            if not key.isidentifier() or key.startswith("_"):
                raise ValueError("Dataset fields must have public identifier names")
            count = 0

            def inspect(item: Any, depth: int = 0, preceding: int = total) -> None:
                nonlocal count
                if depth > 6:
                    raise ValueError("Tensor rank is limited to six")
                if isinstance(item, list):
                    if not item:
                        raise ValueError("Tensor dimensions must be nonempty")
                    for child in item:
                        inspect(child, depth + 1)
                elif type(item) in (int, float, bool):
                    if type(item) is not bool:
                        number(item, "Tensor value", -(2**53), 2**53)
                    count += 1
                    if count + preceding > self.limits["max_tensor_elements"]:
                        raise ValueError("Dataset exceeds the configured tensor element budget")
                else:
                    raise ValueError("Dataset tensors must be rectangular finite numeric arrays")

            inspect(value)
            integer_fields = {"input_ids", "attention_mask", "token_type_ids", "position_ids", "decoder_input_ids", "decoder_attention_mask"}
            dtype = self.torch.long if key in integer_fields else None
            if key in {"x", "pixel_values", "input_values", "inputs_embeds", "decoder_inputs_embeds"}:
                dtype = self.torch.float32
            tensor = self.torch.tensor(value, dtype=dtype)
            if key in integer_fields:
                source = self.torch.tensor(value, dtype=self.torch.float64)
                if not self.torch.equal(source, source.round()):
                    raise ValueError(f"{key} must contain exact integers; fractional IDs/masks are rejected")
                if key.endswith("attention_mask") and not ((tensor == 0) | (tensor == 1)).all().item():
                    raise ValueError("Attention masks must contain only zero or one")
            if tensor.dim() < 1 or tensor.shape[0] > 100000:
                raise ValueError("Dataset must have a bounded sample dimension")
            if key == "input_ids" and (tensor.dim() != 2 or tensor.shape[1] > self.limits["max_sequence_length"]):
                raise ValueError("input_ids must be a [samples, sequence] tensor within the sequence budget")
            if tensors and tensor.shape[0] != next(iter(tensors.values())).shape[0]:
                raise ValueError("All dataset fields must have the same sample count")
            self._bound_tensor(tensor)
            tensors[key] = tensor
            total += count
        handle = uuid.uuid4().hex
        self.datasets[handle] = tensors
        return {"kind": "data", "handle": handle, "samples": next(iter(tensors.values())).shape[0],
                "elements": total, "fields": {key: self._summary(value) for key, value in tensors.items()}}

    def _forward(self, model: Any, recipe: dict[str, Any], fields: dict[str, Any], with_labels: bool = False) -> Any:
        values = {key: value for key, value in fields.items() if with_labels or key not in {"labels", "start_positions", "end_positions"}}
        self._guard_batch(values)
        if recipe["family"] in {"huggingface", "tensor-graph"}:
            self._guard_hf_inputs(model, recipe, values)
        if recipe["family"] == "sequential":
            if "x" not in values:
                raise ValueError("Sequential models require an x tensor field")
            x = values["x"]
            if isinstance(model[0], self.torch.nn.Embedding):
                if x.is_floating_point() and not self.torch.equal(x, x.round()):
                    raise ValueError("An embedding input must contain integer token IDs")
                x = x.long()
            return model(x)
        return model(**values)

    def _guard_hf_inputs(self, model: Any, recipe: dict[str, Any], fields: dict[str, Any], new_tokens: int = 0) -> None:
        ids = fields.get("input_ids")
        if ids is None:
            return
        if ids.dim() != 2:
            raise ValueError("HF input_ids must have shape [batch, sequence]")
        config = model.config
        embedding = model.get_input_embeddings()
        vocab = getattr(embedding, "num_embeddings", getattr(config, "vocab_size", None))
        if vocab is not None and ((ids < 0) | (ids >= vocab)).any().item():
            raise ValueError("Token IDs must be within this model's input vocabulary")
        window = getattr(config, "max_position_embeddings", getattr(config, "n_positions", None))
        length = ids.shape[1] + new_tokens
        if length > self.limits["max_sequence_length"] or (window is not None and length > window):
            raise ValueError("Input sequence plus generated tokens exceeds the model or configured context window")
        if recipe.get("task") in {"causal-lm", "seq2seq-lm", "masked-lm"} and vocab is not None:
            if ids.shape[0] * length * vocab > self.limits["max_tensor_elements"]:
                raise ValueError("Language-model logits exceed the configured tensor element budget")
        mask = fields.get("attention_mask")
        if mask is not None:
            if mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all().item() or not (mask.sum(dim=-1) > 0).all().item():
                raise ValueError("Attention masks must match input_ids and contain at least one unmasked token per sample")

    def _guard_batch(self, fields: dict[str, Any]) -> None:
        if not fields:
            raise ValueError("A model input tensor is required")
        first = next(iter(fields.values()))
        if first.shape[0] > self.limits["max_batch_size"]:
            raise ValueError("Forward batch exceeds the configured maximum batch size")
        # Eager attention may allocate [B,H,T,T] even for small parameter models.
        # Cap a conservative bound before entering the architecture implementation.
        ids = fields.get("input_ids")
        if ids is not None and ids.shape[0] * ids.shape[1] ** 2 * 32 > self.limits["max_tensor_elements"]:
            raise ValueError("Token batch attention scores exceed the configured tensor budget")

    def _loss(self, model: Any, recipe: dict[str, Any], fields: dict[str, Any], name: str) -> tuple[Any, Any]:
        torch = self.torch
        if name == "model":
            if recipe["family"] == "tensor-graph":
                self._guard_batch(fields)
                self._guard_hf_inputs(model, recipe, fields)
                return components_runtime().causal_loss(self, model, fields)
            if recipe["family"] != "huggingface":
                raise ValueError("The model loss mode requires an HF model that implements a loss")
            result = self._forward(model, recipe, fields, with_labels=True)
            loss = getattr(result, "loss", None)
            if loss is None:
                raise ValueError("HF model returned no loss; supply the labels required by this task")
            return loss, getattr(result, "logits", None)
        result = self._forward(model, recipe, fields)
        logits: Any = result.get("logits") if isinstance(result, Mapping) else result
        if not torch.is_tensor(logits):
            raise ValueError("Custom loss requires a tensor output or HF logits")
        if "labels" not in fields:
            raise ValueError("Training/evaluation requires a labels tensor")
        labels = fields["labels"]
        if name == "mse":
            if list(labels.shape) != list(logits.shape):
                raise ValueError("MSE labels must have exactly the output shape; broadcasting is not implicit")
            loss = torch.nn.functional.mse_loss(logits, labels.to(logits.dtype))
        elif name == "cross_entropy":
            if labels.is_floating_point() and not torch.equal(labels, labels.round()):
                raise ValueError("Cross-entropy labels must be integer class indices")
            if logits.shape[:-1] != labels.shape:
                raise ValueError("Cross-entropy labels must match every output dimension except the final class dimension")
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.long().reshape(-1))
        else:
            raise ValueError("Loss must be mse, cross_entropy or model")
        return loss, logits

    def _optimizer(self, handle: str, descriptor: Any) -> Any:
        descriptor = strict_object(descriptor, "Optimizer")
        if descriptor.get("kind") != "optimizer" or set(descriptor) - {"kind", "name", "lr", "weight_decay", "momentum"}:
            raise ValueError("An optimizer recipe is required")
        name = descriptor.get("name")
        if name not in {"AdamW", "Adam", "SGD", "RMSprop"}:
            raise ValueError("Unsupported optimizer")
        lr = number(descriptor.get("lr", 0.01), "learning rate", 1e-12, 10)
        weight_decay = number(descriptor.get("weight_decay", 0), "weight decay", 0, 100)
        momentum = number(descriptor.get("momentum", 0), "momentum", 0, 0.999999)
        key = hashlib.sha256(json_bytes([handle, descriptor])).hexdigest()
        if key not in self.optimizers:
            kwargs = {"lr": lr, "weight_decay": weight_decay}
            if name in {"SGD", "RMSprop"}:
                kwargs["momentum"] = momentum
            optimizer = getattr(self.torch.optim, name)(self.models[handle][0].parameters(), **kwargs)
            self.optimizers[key] = (optimizer, handle, descriptor)
        return self.optimizers[key][0]

    def _export(self, handle: str) -> dict[str, Any]:
        from safetensors.torch import save  # noqa: PLC0415 -- safe tensor format, no executable deserialization.

        model, recipe = self.models[handle]
        tensors = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
        weights = save(tensors)
        if len(weights) > 8 * 1024 * 1024:
            raise ValueError("Checkpoint exceeds the graph artifact's 8 MiB uncompressed weight budget")
        recipe_data = json_bytes({"schema_version": "spobs-tensor-model/1", "recipe": recipe})
        manifest = json_bytes({"schema_version": "spobs-model-artifact/1", "weight_sha256": hashlib.sha256(weights).hexdigest(),
                               "recipe_sha256": hashlib.sha256(recipe_data).hexdigest(), "weight_bytes": len(weights),
                               "parameter_count": self._model_descriptor(handle)["parameter_count"], "state_sha256": self._hash(model)})
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("model.safetensors", weights)
            archive.writestr("recipe.json", recipe_data)
            archive.writestr("manifest.json", manifest)
        return {"model": self._model_descriptor(handle), "artifact": {"filename": "transformers-model.zip",
                "data_base64": base64.b64encode(stream.getvalue()).decode("ascii")}}

    def _load(self, encoded: Any) -> dict[str, Any]:
        from safetensors.torch import load  # noqa: PLC0415 -- never pickle or Python model code.

        if not isinstance(encoded, str) or len(encoded) > 12 * 1024 * 1024:
            raise ValueError("Checkpoint must be a bounded registered ZIP artifact")
        raw = base64.b64decode(encoded, validate=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) != 3 or {member.filename for member in members} != {"model.safetensors", "recipe.json", "manifest.json"}:
                raise ValueError("Model checkpoint must contain exactly safetensors, recipe and manifest")
            if any(member.flag_bits & 1 or member.file_size > 8 * 1024 * 1024 for member in members):
                raise ValueError("Encrypted or oversized model checkpoint members are rejected")
            if sum(member.file_size for member in members) > 9 * 1024 * 1024:
                raise ValueError("Checkpoint decompression exceeds its budget")
            weights, recipe_raw, manifest_raw = (archive.read(name) for name in ("model.safetensors", "recipe.json", "manifest.json"))
        manifest, envelope = json_load(manifest_raw), json_load(recipe_raw)
        if (manifest.get("schema_version") != "spobs-model-artifact/1" or envelope.get("schema_version") != "spobs-tensor-model/1"
                or manifest.get("weight_sha256") != hashlib.sha256(weights).hexdigest()
                or manifest.get("recipe_sha256") != hashlib.sha256(recipe_raw).hexdigest() or manifest.get("weight_bytes") != len(weights)):
            raise ValueError("Checkpoint manifest hashes or schema do not match")
        recipe = strict_object(envelope.get("recipe"), "Checkpoint model recipe")
        descriptor = self._new_model(recipe, recipe.get("seed", 42))
        handle = descriptor["handle"]
        try:
            model = self.models[handle][0]
            tensors = load(weights)
            current = model.state_dict()
            if set(tensors) != set(current) or any(tensors[key].shape != current[key].shape or tensors[key].dtype != current[key].dtype for key in current):
                raise ValueError("Checkpoint tensor names, shapes or dtypes do not match the declared architecture")
            pointers: dict[int, list[str]] = {}
            for key, value in current.items():
                pointers.setdefault(value.data_ptr(), []).append(key)
            for aliases in pointers.values():
                if any(not self.torch.equal(tensors[aliases[0]], tensors[key]) for key in aliases[1:]):
                    raise ValueError("Checkpoint contains conflicting values for tied weights")
            for tensor in tensors.values():
                self._bound_tensor(tensor)
            model.load_state_dict(tensors, strict=True)
            if self._hash(model) != manifest.get("state_sha256"):
                raise ValueError("Loaded model state does not match the checkpoint state hash")
            return {"model": descriptor}
        except BaseException:
            self.models.pop(handle, None)
            raise

    def execute(self, operation: str, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        if operation == "hf.tensor_input" or operation.startswith("hf.op_"):
            compiler = components_runtime()
            if operation not in compiler.INPUT_PORTS:
                raise ValueError("Unknown tensor expression operation")
            normalized = {port: compiler.imported_expression(value, self.expression_nodes)
                          for port, value in inputs.items()
                          if port in (*compiler.INPUT_PORTS[operation], *compiler.OPTIONAL_PORTS.get(operation, ()))}
            self.expression_counter += 1
            expression = compiler.append_expression(operation, normalized, params, f"op{self.expression_counter:04d}")
            self.expression_nodes.update({node["id"]: copy.deepcopy(node) for node in expression["nodes"]})
            return {"x": expression}
        if operation == "hf.graph_model":
            expression = components_runtime().imported_expression(inputs.get("logits"), self.expression_nodes)
            fields = {key: value for key, value in params.items() if key != "seed"}
            if set(fields) - {"vocab_size", "max_positions", "pad_token_id", "bos_token_id", "eos_token_id"}:
                raise ValueError("Unknown graph model parameter")
            return {"model": self._new_model({"family": "tensor-graph", "task": "causal-lm", "expression": expression, **fields}, params.get("seed", 42))}
        recipes = {"hf.linear": "Linear", "hf.embedding": "Embedding", "hf.rmsnorm": "RMSNorm", "hf.layernorm": "LayerNorm",
                   "hf.attention": "SelfAttention"}
        if operation in recipes or operation in {"hf.layer", "hf.activation"}:
            if operation == "hf.layer":
                value = inputs["recipe"]
            elif operation == "hf.activation":
                value = {"type": params.get("name", "GELU")}
            else:
                value = {"type": recipes[operation], **params}
            with torch.device("meta"):
                probe = self._layer(value)
            if sum(parameter.numel() for parameter in probe.parameters()) > self.limits["max_parameters"]:
                raise ValueError("Layer exceeds the configured parameter budget")
            return {"layer": value}
        if operation == "hf.compose":
            layers = []
            for value in (inputs.get("a", {}), inputs.get("b", {})):
                if value == {}:
                    continue
                if isinstance(value, list):
                    layers.extend(value)
                else:
                    layers.append(value)
            if len(layers) > 128:
                raise ValueError("Composed layer list exceeds 128 layers")
            for layer in layers:
                with torch.device("meta"):
                    self._layer(layer)
            return {"layers": layers}
        if operation == "hf.stack":
            layers = inputs.get("layers", [])
            if not isinstance(layers, list) or len(layers) >= 128:
                raise ValueError("Layer list must contain fewer than 128 layers")
            with torch.device("meta"):
                self._layer(inputs["layer"])
            return {"layers": [*layers, inputs["layer"]]}
        if operation == "hf.optimizer":
            name = params.get("name", "AdamW")
            if name not in {"AdamW", "Adam", "SGD", "RMSprop"}:
                raise ValueError("Unsupported optimizer")
            descriptor = {"kind": "optimizer", "name": name, "lr": number(params.get("lr", 0.01), "learning rate", 1e-12, 10),
                          "weight_decay": number(params.get("weight_decay", 0), "weight decay", 0, 100),
                          "momentum": number(params.get("momentum", 0), "momentum", 0, 0.999999)}
            return {"optimizer": descriptor}
        if operation == "hf.config":
            model_type = params.get("model_type", "gpt2")
            if not isinstance(model_type, str) or not model_type or len(model_type) > 100:
                raise ValueError("model_type must be an installed HF model type")
            values = strict_object(inputs.get("values", {}), "HF config values")
            self._validate_config_tree(values)
            return {"config": {"kind": "hf-config", "model_type": model_type, "values": values}}
        if operation == "hf.model":
            return {"model": self._new_model({"family": "huggingface", "config": inputs["config"], "task": params.get("task", "causal-lm")}, params.get("seed", 42))}
        if operation in {"hf.pretrained", "hf.auto_tokenizer"}:
            import transformers  # noqa: PLC0415 -- trusted installed HF code, cache-only loading.

            repo_id, revision = params.get("repo_id"), params.get("revision")
            if not isinstance(repo_id, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9_-][A-Za-z0-9_.-]{0,95}", repo_id):
                raise ValueError("repo_id must be a public namespace/repository ID; filesystem paths are not allowed")
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError("revision must be a pinned 40-character Hub commit hash, already present in the local cache")
            kwargs = {"revision": revision.lower(), "local_files_only": True, "trust_remote_code": False}
            if operation == "hf.auto_tokenizer":
                wrapped = transformers.AutoTokenizer.from_pretrained(repo_id, use_fast=True, **kwargs)
                if not wrapped.is_fast:
                    raise ValueError("Cached tokenizer must have a safe fast tokenizer representation")
                use_eos_as_pad = params.get("use_eos_as_pad", True)
                if type(use_eos_as_pad) is not bool:
                    raise ValueError("use_eos_as_pad must be boolean")
                pad_token = params.get("pad_token", "")
                if not isinstance(pad_token, str) or len(pad_token) > 256:
                    raise ValueError("pad_token must be an existing vocabulary token of at most 256 characters")
                if pad_token:
                    if pad_token not in wrapped.get_vocab():
                        raise ValueError("pad_token must already exist in the cached vocabulary; implicit model resizing is not allowed")
                    wrapped.pad_token = pad_token
                elif wrapped.pad_token_id is None and use_eos_as_pad and wrapped.eos_token_id is not None:
                    wrapped.pad_token = wrapped.eos_token
                wrapped.padding_side = "left"
                wrapped.model_max_length = min(self.limits["max_sequence_length"], wrapped.model_max_length)
                handle = uuid.uuid4().hex
                self.tokenizers[handle] = wrapped
                return {"tokenizer": {"kind": "tokenizer", "handle": handle, "vocab_size": len(wrapped), "padding_side": "left",
                                      "implementation": type(wrapped).__name__, "repo_id": repo_id, "revision": revision.lower(),
                                      "pad_token_id": wrapped.pad_token_id, "eos_token_id": wrapped.eos_token_id}}
            task = params.get("task", "causal-lm")
            if task not in TASK_CLASSES:
                raise ValueError("Unsupported AutoModel task")
            cfg = transformers.AutoConfig.from_pretrained(repo_id, **kwargs)
            if getattr(cfg, "auto_map", None):
                raise ValueError("Cached pretrained model requires remote code and is not admitted")
            values = json_load(cfg.to_json_string(use_diff=False))
            values = {key: value for key, value in values.items() if not key.startswith("_") and key not in
                      {"model_type", "architectures", "auto_map", "transformers_version", "dtype", "torch_dtype", "name_or_path"}}
            recipe = {"family": "huggingface", "config": {"kind": "hf-config", "model_type": cfg.model_type, "values": values},
                      "task": task, "seed": 42, "source": {"repo_id": repo_id, "revision": revision.lower()}}
            with torch.device("meta"):
                probe = self._build(recipe)
            self._guard_model(probe)
            del probe
            cfg._attn_implementation = "eager"
            model = getattr(transformers, TASK_CLASSES[task]).from_pretrained(repo_id, config=cfg, use_safetensors=True,
                                                                           dtype=torch.float32, **kwargs)
            model.cpu().eval()
            self._guard_model(model)
            for parameter in model.parameters():
                self._bound_tensor(parameter)
            handle = uuid.uuid4().hex
            self.models[handle] = (model, recipe)
            return {"model": {**self._model_descriptor(handle), "source": recipe["source"]}}
        if operation == "hf.sequential":
            return {"model": self._new_model({"family": "sequential", "layers": inputs["layers"]}, params.get("seed", 42))}
        if operation == "hf.custom":
            return {"model": self._new_model({"family": "custom", "layers": inputs["layers"], "code": params.get("code")}, params.get("seed", 42))}
        if operation == "hf.data":
            return {"data": self._new_data(inputs["fields"])}
        if operation == "hf.tokenizer":
            from tokenizers import Tokenizer, models, normalizers, pre_tokenizers  # noqa: PLC0415
            from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

            vocab = strict_object(inputs["vocab"], "Vocabulary")
            if not 4 <= len(vocab) <= 100000 or set(vocab.values()) != set(range(len(vocab))) or any(type(index) is not int for index in vocab.values()):
                raise ValueError("Vocabulary IDs must be distinct contiguous integers starting at zero (4 to 100000 tokens)")
            for token in ("[PAD]", "[UNK]", "[BOS]", "[EOS]"):
                if token not in vocab:
                    raise ValueError(f"Vocabulary requires {token}")
            tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
            if params.get("lowercase", True):
                tokenizer.normalizer = normalizers.Lowercase()
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                                              bos_token="[BOS]", eos_token="[EOS]", padding_side="left", model_max_length=self.limits["max_sequence_length"],
                                              **({"mask_token": "[MASK]"} if "[MASK]" in vocab else {}))
            handle = uuid.uuid4().hex
            self.tokenizers[handle] = wrapped
            self.local_tokenizers.add(handle)
            return {"tokenizer": {"kind": "tokenizer", "handle": handle, "vocab_size": len(vocab), "padding_side": "left", "implementation": "PreTrainedTokenizerFast/WordLevel"}}
        if operation == "hf.tokenize":
            handle = self._handle(inputs["tokenizer"], self.tokenizers, "tokenizer")
            texts = self._texts(inputs["texts"])
            maximum = integer(params.get("max_length", 32), "max_length", 1, self.limits["max_sequence_length"])
            fields = dict(self.tokenizers[handle](texts, padding=True, truncation=True, max_length=maximum))
            if params.get("labels", "causal") == "causal":
                fields["labels"] = [[token if mask else -100 for token, mask in zip(ids, masks, strict=True)]
                                    for ids, masks in zip(fields["input_ids"], fields["attention_mask"], strict=True)]
            elif params.get("labels", "none") != "none":
                raise ValueError("Tokenization labels must be none or causal")
            return {"data": self._new_data(fields)}
        if operation == "hf.load":
            return self._load(inputs["data_base64"])
        model_handle = self._handle(inputs.get("model"), self.models, "model")
        model, recipe = self.models[model_handle]
        if operation == "hf.inspect":
            report = {**self._model_descriptor(model_handle), "state_sha256": self._hash(model), "recipe": recipe,
                      **self._architecture_report(model),
                      "parameters": [{"name": name, "shape": list(parameter.shape), "elements": parameter.numel(),
                                      "dtype": str(parameter.dtype), "trainable": parameter.requires_grad,
                                      "finite": bool(torch.isfinite(parameter).all())} for name, parameter in model.named_parameters()][:512]}
            if recipe["family"] == "tensor-graph":
                report["tensor_graph"] = {"expression": model.expression, "operation_shapes": model.operation_shapes,
                                          "operation_count": len(model.expression["nodes"]), "execution": "Connected operations replay on every forward with registered shared parameters"}
            return {"model": self._model_descriptor(model_handle), "report": report}
        if operation == "hf.checkpoint":
            return self._export(model_handle)
        if operation == "hf.pipeline":
            from transformers import pipeline  # noqa: PLC0415 -- task registry remains trusted installed HF code.

            tokenizer_handle = self._handle(inputs["tokenizer"], self.tokenizers, "tokenizer")
            texts = self._texts(inputs["texts"])
            task = params.get("task", "text-generation")
            if task not in {"text-generation", "text-classification", "token-classification", "fill-mask", "feature-extraction"}:
                raise ValueError("Unsupported pipeline task")
            tokenized = self.tokenizers[tokenizer_handle](texts, padding=True, return_tensors="pt")
            self._guard_batch(dict(tokenized))
            requested = integer(params.get("max_new_tokens", 8), "max_new_tokens", 1, 128) if task == "text-generation" else 0
            self._guard_hf_inputs(model, recipe, dict(tokenized), requested)
            if recipe["family"] == "tensor-graph":
                if task != "text-generation":
                    raise ValueError("Connected causal graph pipelines support text-generation")
                model.eval()
                with torch.no_grad():
                    sequences = model.generate(input_ids=tokenized["input_ids"], attention_mask=tokenized.get("attention_mask"),
                                               max_new_tokens=requested, do_sample=False)
                records = [[{"generated_token_ids": row.tolist(),
                             "generated_text": self.tokenizers[tokenizer_handle].decode(row.tolist(), skip_special_tokens=True)}]
                           for row in sequences]
                return {"model": self._model_descriptor(model_handle), "result": {"task": task, "outputs": records,
                        "implementation": "Studio connected tensor DAG autoregressive generation with HF tokenizer"}}
            runner = pipeline(task=task, model=model, tokenizer=self.tokenizers[tokenizer_handle], device="cpu", trust_remote_code=False)
            model.eval()
            kwargs = {}
            if task == "text-generation":
                kwargs = {"max_new_tokens": integer(params.get("max_new_tokens", 8), "max_new_tokens", 1, 128), "do_sample": False}
                if tokenizer_handle in self.local_tokenizers:
                    # HF's text pipeline concatenates the original prompt with
                    # a separately decoded suffix. WordLevel tokens do not carry
                    # leading spaces, so that joins words at the prompt boundary.
                    # Keep actual pipeline generation and decode its complete IDs.
                    kwargs["return_tensors"] = True
            with torch.no_grad():
                result = runner(texts, **kwargs)
            if task == "text-generation" and tokenizer_handle in self.local_tokenizers:
                tokenizer = self.tokenizers[tokenizer_handle]
                result = [[{**record, "generated_text": tokenizer.decode(record["generated_token_ids"], skip_special_tokens=True)}
                           for record in rows] for rows in result]
            return {"model": self._model_descriptor(model_handle), "result": {"task": task, "outputs": self._pipeline_json(result)}}
        data_handle = self._handle(inputs.get("data"), self.datasets, "data")
        fields = self.datasets[data_handle]
        if operation == "hf.forward":
            model.eval()
            with torch.no_grad():
                result = self._forward(model, recipe, fields)
                self._bound_tensor(result)
            return {"model": self._model_descriptor(model_handle), "result": {"outputs": self._summary(result)}}
        if operation == "hf.generate":
            if recipe["family"] not in {"huggingface", "tensor-graph"} or not callable(getattr(model, "generate", None)):
                raise ValueError("Generation requires an HF or connected causal graph model")
            self._guard_batch(fields)
            maximum = integer(params.get("max_new_tokens", 8), "max_new_tokens", 1, 128)
            tokens = fields.get("input_ids")
            if tokens is None or tokens.shape[1] + maximum > self.limits["max_sequence_length"]:
                raise ValueError("Generation input plus new tokens exceeds the sequence budget")
            # Explicit non-generative fields never enter generate's kwargs.
            generate_fields = {key: value for key, value in fields.items() if key in {"input_ids", "attention_mask", "token_type_ids", "decoder_input_ids", "decoder_attention_mask"}}
            self._guard_hf_inputs(model, recipe, generate_fields, maximum)
            mask = generate_fields.get("attention_mask")
            if mask is not None and not getattr(model.config, "is_encoder_decoder", False):
                # Decoder-only generation reads the last position. Normalize
                # manually supplied right-padded rows as well as tokenizer data.
                pad = getattr(model.config, "pad_token_id", None)
                if pad is None:
                    eos = getattr(model.config, "eos_token_id", None)
                    pad = eos if type(eos) is int else None
                if pad is None and (mask == 0).any().item():
                    raise ValueError("Padded generation requires a model pad_token_id or a single EOS token to pad")
                for key in ("input_ids", "token_type_ids"):
                    if key not in generate_fields:
                        continue
                    value = generate_fields[key]
                    normalized = torch.full_like(value, pad if key == "input_ids" else 0)
                    for index, row in enumerate(value):
                        valid = row[mask[index].bool()]
                        normalized[index, -len(valid):] = valid
                    generate_fields[key] = normalized
                normalized_mask = torch.zeros_like(mask)
                for index, row in enumerate(mask):
                    normalized_mask[index, -int(row.sum()):] = 1
                generate_fields["attention_mask"] = normalized_mask
            model.eval()
            generation_kwargs: dict[str, Any] = {"max_new_tokens": maximum, "do_sample": bool(params.get("do_sample", False))}
            if generation_kwargs["do_sample"]:
                generation_kwargs["temperature"] = number(params.get("temperature", 1), "temperature", 1e-6, 100)
            with torch.no_grad():
                result = model.generate(**generate_fields, **generation_kwargs)
            return {"model": self._model_descriptor(model_handle), "result": {"sequences": result.tolist(), "shape": list(result.shape),
                    "max_new_tokens": maximum, "input_ids": generate_fields["input_ids"].tolist(),
                    "attention_mask": generate_fields["attention_mask"].tolist() if "attention_mask" in generate_fields else None}}
        if operation == "hf.evaluate":
            model.eval()
            before = self._hash(model)
            with torch.no_grad():
                loss, logits = self._loss(model, recipe, fields, params.get("loss", "mse"))
                self._bound_tensor(loss)
            metrics = {"loss": float(loss), "samples": next(iter(fields.values())).shape[0], "state_sha256": before,
                       "parameter_changed": before != self._hash(model)}
            if params.get("loss") == "cross_entropy" and logits is not None:
                metrics["accuracy"] = float((logits.argmax(dim=-1) == fields["labels"]).float().mean())
            return {"model": self._model_descriptor(model_handle), "metrics": metrics}
        if operation == "hf.train":
            steps = integer(params.get("steps", 20), "steps", 1, self.limits["max_steps"])
            batch_size = integer(params.get("batch_size", 3), "batch_size", 1, self.limits["max_batch_size"])
            clip = number(params.get("clip_grad_norm", 1), "clip_grad_norm", 0, 1000000)
            optimizer = self._optimizer(model_handle, inputs["optimizer"])
            before = self._hash(model)
            losses, gradients = [], []
            samples = next(iter(fields.values())).shape[0]
            model.train()
            for step in range(steps):
                index = torch.arange(step * batch_size, (step + 1) * batch_size) % samples
                batch = {key: value[index] for key, value in fields.items()}
                optimizer.zero_grad(set_to_none=True)
                loss, _ = self._loss(model, recipe, batch, params.get("loss", "mse"))
                self._bound_tensor(loss)
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self._bound_tensor(parameter.grad)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else float("inf"), error_if_nonfinite=True)
                optimizer.step()
                for parameter in model.parameters():
                    self._bound_tensor(parameter)
                losses.append(float(loss.detach()))
                gradients.append(float(norm))
                if self.on_progress is not None and (step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 100) == 0):
                    self.on_progress({"step": step + 1, "steps": steps, "loss": losses[-1], "gradient_norm": gradients[-1]})
            after = self._hash(model)
            metrics = {"steps": steps, "batch_size": batch_size, "samples": samples, "loss": losses[-1], "loss_history": losses,
                       "gradient_norms": gradients, "parameter_count": self._model_descriptor(model_handle)["parameter_count"],
                       "parameter_changed": before != after, "state_sha256_before": before, "state_sha256_after": after,
                       "optimizer": inputs["optimizer"], "optimizer_state_tensors": sum(torch.is_tensor(value) for state in optimizer.state.values() for value in state.values())}
            return {"model": self._model_descriptor(model_handle), "metrics": metrics}
        raise ValueError(f"Unknown trusted tensor operation: {operation}")

    def _texts(self, value: Any) -> list[str]:
        if not isinstance(value, list) or not 1 <= len(value) <= self.limits["max_batch_size"]:
            raise ValueError("Texts must contain a nonempty bounded list of strings")
        if any(not isinstance(text, str) or not text.strip() or len(text) > 16384 for text in value):
            raise ValueError("Each text must be nonempty and at most 16384 characters")
        return value


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    sys.path.insert(0, sys.argv[1])
    budgets = json_load(sys.argv[2])
    if os.name != "nt":
        import resource  # noqa: PLC0415 -- worker-only resource protection.

        cap = budgets["worker_memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_DATA, (cap, cap))  # type: ignore[attr-defined]  # POSIX-only, absent in Windows stubs.
    protocol = sys.stdout
    # Trusted libraries may print progress/status; never let them corrupt JSON
    # framing. A parent reader drains the bounded stderr tail continuously.
    sys.stdout = sys.stderr

    def send(value: dict[str, Any]) -> None:
        protocol.write(json_bytes(value).decode("utf-8") + "\n")
        protocol.flush()

    try:
        runtime = Runtime(budgets)
        runtime.on_progress = lambda progress: send({"progress": progress})
        send({"ready": True})
    except BaseException as exc:
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"[:2000]})
        return 1
    while True:
        line = sys.stdin.buffer.readline(12 * 1024 * 1024 + 1)
        if not line:
            return 0
        if len(line) > 12 * 1024 * 1024:
            send({"ok": False, "error": "Worker request exceeds protocol budget"})
            return 1
        try:
            request = json_load(line)
            outputs = runtime.execute(request["operation"], request["inputs"], request["params"])
            send({"ok": True, "outputs": outputs})
        except BaseException as exc:
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]})


if __name__ == "__main__":
    raise SystemExit(main())
