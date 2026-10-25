"""Declarative Transformers / tensor workflows and their disposable worker client.

No user Python is executed.  The worker accepts a small trusted operation
catalog, keeps tensors local to one graph job, and returns only JSON handles.
Editable tensor code is parsed by the restricted AST interpreter in the worker.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import studio_components


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(low, min(high, value))


def limits() -> dict[str, int]:
    return {"max_parameters": _env_int("SPOBS_TRANSFORMERS_MAX_PARAMETERS", 2_000_000, 1, 100_000_000),
            "max_tensor_elements": _env_int("SPOBS_TRANSFORMERS_MAX_ELEMENTS", 1_000_000, 1, 10_000_000),
            "worker_memory_mb": _env_int("SPOBS_TRANSFORMERS_MEMORY_MB", 2048, 768, 8192),
            "max_steps": _env_int("SPOBS_TRANSFORMERS_MAX_STEPS", 2000, 1, 100000),
            "operation_timeout_seconds": _env_int("SPOBS_TRANSFORMERS_TIMEOUT", 300, 5, 3600),
            "startup_timeout_seconds": 180, "max_sequence_length": 512, "max_batch_size": 128}


TASKS = ["base", "causal-lm", "seq2seq-lm", "sequence-classification", "token-classification",
         "question-answering", "masked-lm", "image-classification", "audio-classification"]


def capabilities() -> dict[str, Any]:
    available = importlib.util.find_spec("transformers") is not None
    try:
        version = importlib.metadata.version("transformers") if available else None
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"available": available, "version": version, "device": "cpu", "execution": "job-scoped CPU worker",
            "model_tasks": TASKS, "supported_operations": [item["id"] for item in definitions()], "limits": limits(),
            "network": False, "cached_pretrained": True, "pretrained_revision": "pinned 40-character Hub commit; cached files only",
            "trust_remote_code": False,
            "arbitrary_python": False, "custom_code": "restricted tensor-forward Python AST",
            "checkpoint_format": "ZIP: model.safetensors + recipe.json + manifest.json",
            "handle_lifetime": "one workflow execution; export/load a checkpoint to reuse weights",
            "description": "Actual Hugging Face AutoConfig/AutoModel, PyTorch modules and optimizers. Offline models from config or pinned cached safetensors weights."}


def definitions() -> list[dict[str, Any]]:
    def port(name: str, kind: str = "object", default: Any = None, required: bool = True) -> dict[str, Any]:
        out = {"name": name, "type": kind, "required": required}
        if default is not None:
            out["default"] = default
        return out

    def param(name: str, kind: str, default: Any, **kw: Any) -> dict[str, Any]:
        return {"name": name, "type": kind, "default": default, **kw}

    def node(identity: str, title: str, category: str, description: str, inputs: list[dict[str, Any]],
             outputs: list[dict[str, Any]], parameters: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {"id": "hf." + identity, "title": title, "category": category, "description": description,
                "inputs": inputs, "outputs": outputs, "parameters": parameters or []}

    seed = param("seed", "integer", 42, min=0, max=2**31 - 1)
    model = port("model")
    data = port("data")
    default_layers = [{"type": "Linear", "in_features": 1, "out_features": 8, "bias": True},
                      {"type": "RMSNorm", "normalized_shape": 8, "eps": 1e-6}, {"type": "GELU"},
                      {"type": "Linear", "in_features": 8, "out_features": 1, "bias": True}]
    return [
        node("config", "HF model configuration", "Transformers · Architecture", "Create an offline AutoConfig recipe. No model download or remote code.",
             [port("values", default={"vocab_size": 16, "n_positions": 32, "n_ctx": 32, "n_embd": 16,
                                      "n_layer": 1, "n_head": 2, "bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0})],
             [port("config")], [param("model_type", "string", "gpt2")]),
        node("model", "HF AutoModel", "Transformers · Architecture", "Instantiate an actual supported HF model from config after a meta-device parameter budget check.",
             [port("config")], [model], [param("task", "string", "causal-lm", options=TASKS), seed]),
        node("pretrained", "Cached HF pretrained model", "Transformers · Architecture", "Load actual cached safetensors weights by public Hub repository and pinned commit. No downloads, arbitrary paths, remote code or pickle.",
             [], [model], [param("repo_id", "string", "hf-internal-testing/tiny-random-gpt2"), param("revision", "string", ""),
                           param("task", "string", "causal-lm", options=TASKS)]),
        node("auto_tokenizer", "Cached HF AutoTokenizer", "Transformers · Data", "Load a matching cached fast tokenizer by repository and pinned Hub commit. Never imports remote tokenizer code.",
             [], [port("tokenizer")], [param("repo_id", "string", "hf-internal-testing/tiny-random-gpt2"), param("revision", "string", ""),
                                      param("pad_token", "string", ""), param("use_eos_as_pad", "boolean", True)]),
        node("tokenizer", "Local HF fast tokenizer", "Transformers · Data", "Build a real offline WordLevel PreTrainedTokenizerFast from an editable vocabulary.",
             [port("vocab", default={"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "hello": 4, "world": 5})],
             [port("tokenizer")], [param("lowercase", "boolean", True)]),
        node("tokenize", "Tokenize text", "Transformers · Data", "Tokenize and pad text; causal labels mask padding with -100.",
             [port("tokenizer"), port("texts", "array", ["hello world", "hello"])], [data],
             [param("max_length", "integer", 32, min=1, max=512), param("labels", "string", "causal", options=["none", "causal"])]),
        node("data", "Tensor dataset", "Transformers · Data", "Finite named tensor batches: x/labels for custom models or input_ids/attention_mask/labels for HF.",
             [port("fields", default={"x": [[-1], [0], [1]], "labels": [[-2], [0], [2]]})], [data]),
        node("layer", "Layer recipe", "Transformers · Architecture", "A declarative PyTorch layer. Connect recipes into Compose layers / Sequential / custom forward.",
             [port("recipe", default={"type": "Linear", "in_features": 1, "out_features": 8})], [port("layer")]),
        node("linear", "Linear", "Transformers · Architecture", "Trainable dense projection with arbitrary bounded input/output width.", [], [port("layer")],
             [param("in_features", "integer", 1, min=1), param("out_features", "integer", 8, min=1), param("bias", "boolean", True)]),
        node("embedding", "Embedding", "Transformers · Architecture", "A trainable token embedding table.", [], [port("layer")],
             [param("num_embeddings", "integer", 16, min=1), param("embedding_dim", "integer", 8, min=1)]),
        node("rmsnorm", "RMSNorm", "Transformers · Architecture", "Actual torch.nn.RMSNorm. Normalization connects to layer outputs; optimizers bind the complete model.", [], [port("layer")],
             [param("normalized_shape", "integer", 8, min=1), param("eps", "number", 1e-6)]),
        node("layernorm", "LayerNorm", "Transformers · Architecture", "Actual torch.nn.LayerNorm.", [], [port("layer")],
             [param("normalized_shape", "integer", 8, min=1), param("eps", "number", 1e-5)]),
        node("activation", "Activation", "Transformers · Architecture", "A stateless tensor activation.", [], [port("layer")],
             [param("name", "string", "GELU", options=["GELU", "ReLU", "SiLU", "Tanh", "Sigmoid", "Identity"])]),
        node("attention", "Self attention", "Transformers · Architecture", "Trainable batch-first multi-head self attention with optional causal masking.", [], [port("layer")],
             [param("embed_dim", "integer", 8, min=1), param("num_heads", "integer", 2, min=1),
              param("dropout", "number", 0), param("causal", "boolean", True)]),
        node("compose", "Compose layers", "Transformers · Architecture", "Join two layer recipes or composed layer lists. Connect the layers output to Sequential.",
             [port("a", "any", {}), port("b", "any", {})], [port("layers", "array")]),
        node("stack", "Append layer", "Transformers · Architecture", "Append an individual layer to a layer list.",
             [port("layers", "array", []), port("layer")], [port("layers", "array")]),
        node("sequential", "Sequential model", "Transformers · Architecture", "Instantiate a real multi-parameter PyTorch architecture from ordered layer recipes.",
             [port("layers", "array", default_layers)], [model], [seed]),
        node("custom", "Custom tensor forward", "Transformers · Code", "Editable restricted Python tensor expressions and named trainable layers. No imports, attributes, loops, eval or exec.",
             [port("layers", default={"input": {"type": "Linear", "in_features": 1, "out_features": 8},
                                      "output": {"type": "Linear", "in_features": 8, "out_features": 1}})], [model],
             [param("code", "code", "x = layers['input'](inputs['x'])\nx = relu(x)\nreturn layers['output'](x)"), seed]),
        node("optimizer", "Optimizer", "Transformers · Training", "Optimizer recipe. Connect to Train alongside a model; AdamW updates the model's complete parameter set.",
             [], [port("optimizer")], [param("name", "string", "AdamW", options=["AdamW", "Adam", "SGD", "RMSprop"]),
                                      param("lr", "number", 0.01), param("weight_decay", "number", 0), param("momentum", "number", 0)]),
        node("train", "Train multi-parameter model", "Transformers · Training", "Actual autograd/optimizer updates, finite loss checks, gradient clipping, measured parameter hashes and loss history.",
             [model, data, port("optimizer")], [model, port("metrics")],
             [param("steps", "integer", 20, min=1), param("batch_size", "integer", 3, min=1, max=128),
              param("loss", "string", "mse", options=["mse", "cross_entropy", "model"]), param("clip_grad_norm", "number", 1)]),
        node("forward", "Forward pass", "Transformers · Inference", "Run a real model forward; tensors remain in the job worker and expose shape/dtype/value previews.",
             [model, data], [model, port("result")]),
        node("evaluate", "Evaluate model", "Transformers · Inference", "Evaluate a finite loss and optional classification accuracy without changing weights.",
             [model, data], [model, port("metrics")], [param("loss", "string", "mse", options=["mse", "cross_entropy", "model"])]),
        node("generate", "HF generate", "Transformers · Inference", "Real bounded HF autoregressive generation with attention masks and deterministic default sampling.",
             [model, data], [model, port("result")], [param("max_new_tokens", "integer", 8, min=1, max=128),
                                                     param("do_sample", "boolean", False), param("temperature", "number", 1)]),
        node("pipeline", "HF pipeline", "Transformers · Inference", "Actual Transformers pipeline with a job-local model/tokenizer, offline inference and bounded inputs.",
             [model, port("tokenizer"), port("texts", "array", ["hello world"])], [model, port("result")],
             [param("task", "string", "text-generation", options=["text-generation", "text-classification", "token-classification", "fill-mask", "feature-extraction"]),
              param("max_new_tokens", "integer", 8, min=1, max=128)]),
        node("inspect", "Inspect model", "Transformers · Observe", "Real parameter names, shapes, trainable counts, architecture, finite checks and weight hash.",
             [model], [model, port("report")]),
        node("checkpoint", "Export safe model", "Transformers · Artifacts", "Immutable ZIP with safetensors weights and a declarative architecture recipe. No pickle or executable model code.",
             [model], [model, port("artifact")]),
        node("load", "Load safe model", "Transformers · Artifacts", "Load only a registered graph artifact, verify safe ZIP/recipe/safetensors and rebuild within model budgets.",
             [port("artifact")], [model]),
        *studio_components.definitions(),
    ]


def templates() -> list[dict[str, Any]]:
    def n(identity: str, kind: str, x: int, y: int, **params: Any) -> dict[str, Any]:
        return {"id": identity, "type": "hf." + kind, "position": {"x": x, "y": y}, "params": params}

    def e(source: str, output: str, target: str, input_name: str) -> dict[str, Any]:
        return {"id": f"{source}:{output}-{target}:{input_name}", "source": source, "source_port": output,
                "target": target, "target_port": input_name}

    mlp = {"name": "Multi-parameter RMSNorm laboratory", "description": "Dense → RMSNorm → GELU → dense; real AdamW training and safe checkpoint.",
           "nodes": [n("model", "sequential", 30, 30), n("data", "data", 30, 330), n("optimizer", "optimizer", 360, 330),
                     n("train", "train", 700, 30), n("inspect", "inspect", 1030, 30), n("checkpoint", "checkpoint", 1360, 30)],
           "edges": [e("model", "model", "train", "model"), e("data", "data", "train", "data"),
                     e("optimizer", "optimizer", "train", "optimizer"), e("train", "model", "inspect", "model"),
                     e("inspect", "model", "checkpoint", "model")], "view": {"x": 0, "y": 0, "zoom": 0.7}}
    hf = {"name": "Tiny offline GPT-2 training and generation", "description": "Actual HF AutoModelForCausalLM, token batches, AdamW, generation and safetensors export. No download.",
          "nodes": [n("config", "config", 20, 20), n("model", "model", 350, 20),
                    n("data", "data", 20, 330, fields={"input_ids": [[2, 4, 5, 3], [2, 5, 4, 3]],
                                                      "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
                                                      "labels": [[2, 4, 5, 3], [2, 5, 4, 3]]}),
                    n("optimizer", "optimizer", 350, 330, lr=0.001), n("train", "train", 700, 20, steps=3, batch_size=2, loss="model"),
                    n("generate", "generate", 1030, 20, max_new_tokens=4), n("checkpoint", "checkpoint", 1360, 20)],
          "edges": [e("config", "config", "model", "config"), e("model", "model", "train", "model"),
                    e("data", "data", "train", "data"), e("optimizer", "optimizer", "train", "optimizer"),
                    e("train", "model", "generate", "model"), e("data", "data", "generate", "data"),
                    e("generate", "model", "checkpoint", "model")], "view": {"x": 0, "y": 0, "zoom": 0.7}}
    custom = {"name": "Editable tensor forward", "description": "Named trainable layers with an editable restricted Python tensor-forward program, autograd and AdamW.",
              "nodes": [n("model", "custom", 20, 20), n("data", "data", 20, 330), n("optimizer", "optimizer", 350, 330),
                        n("train", "train", 700, 20, steps=10), n("evaluate", "evaluate", 1030, 20)],
              "edges": [e("model", "model", "train", "model"), e("data", "data", "train", "data"),
                        e("optimizer", "optimizer", "train", "optimizer"), e("train", "model", "evaluate", "model"),
                        e("data", "data", "evaluate", "data")], "view": {"x": 0, "y": 0, "zoom": 0.8}}
    llama = json.loads((Path(__file__).parent / "examples" / "transformers" / "llama.workflow.json").read_text(encoding="utf-8"))
    return [{"id": identity, "category": "Transformers", "graph": graph, "name": graph["name"], "description": graph["description"]}
            for identity, graph in [("transformers-mlp", mlp), ("transformers-causal-lm", hf), ("transformers-custom-forward", custom),
                                   ("transformers-llama", llama), ("transformers-llama-blocks", studio_components.llama_blocks_workflow()),
                                   ("transformers-llama-blocks-custom", studio_components.llama_blocks_workflow(custom=True))]]


class TransformersWorkerError(ValueError):
    """A trusted runtime operation failed or its worker was terminated."""


class TransformersWorker:
    """One bounded worker per graph execution; never survives project/job close."""

    def __init__(self, cancel: threading.Event | None = None, on_progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        if not capabilities()["available"]:
            raise TransformersWorkerError("Install the Transformers runtime dependency before executing HF nodes")
        self.cancel = cancel or threading.Event()
        self._on_progress = on_progress
        self._responses: queue.Queue[bytes | BaseException] = queue.Queue()
        self._stderr = bytearray()
        self._closed = False
        self._job: Any = None
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.origin is None:
            raise TransformersWorkerError("PyTorch is required")
        site_packages = str(Path(spec.origin).parent.parent)
        worker = str(Path(__file__).with_name("studio_transformers_worker.py"))
        executable = getattr(sys, "_base_executable", None) or sys.executable
        env = {key: value for key, value in os.environ.items() if key.upper() in
               {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "HOME", "USERPROFILE", "LOCALAPPDATA",
                "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"}}
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                    "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.process = subprocess.Popen([executable, "-I", "-S", "-u", "-B", worker, site_packages,
                                         json.dumps(limits(), separators=(",", ":"))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._reader = threading.Thread(target=self._read, daemon=True, name="transformers-protocol")
        self._error_reader = threading.Thread(target=self._read_errors, daemon=True, name="transformers-stderr")
        self._reader.start()
        self._error_reader.start()
        try:
            self._bound_process(limits()["worker_memory_mb"])
            ready = self._receive(limits()["startup_timeout_seconds"])
            if not ready.get("ready"):
                raise TransformersWorkerError(str(ready.get("error", "Worker did not initialize")))
        except BaseException:
            self.close()
            raise

    def _bound_process(self, memory_mb: int) -> None:
        if os.name != "nt":
            # Linux workers apply RLIMIT_DATA themselves before importing Torch.
            return
        import ctypes  # noqa: PLC0415 -- platform-specific process resource bound.
        from ctypes import wintypes  # noqa: PLC0415

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ["ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount"]]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "Could not create Transformers process job")
        cap = Extended()
        cap.BasicLimitInformation.LimitFlags = 0x0100 | 0x2000 | 0x0008
        cap.BasicLimitInformation.ActiveProcessLimit = 1
        cap.ProcessMemoryLimit = memory_mb * 1024 * 1024
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(cap), ctypes.sizeof(cap)) or not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(int(self.process._handle))):  # type: ignore[attr-defined]
            kernel.CloseHandle(job)
            raise OSError(ctypes.get_last_error(), "Could not enforce Transformers worker resource limits")
        self._job = (kernel, job)

    def _read(self) -> None:
        try:
            assert self.process.stdout is not None
            while True:
                line = self.process.stdout.readline(16 * 1024 * 1024 + 1)
                if not line:
                    raise TransformersWorkerError("Transformers worker exited before completing the operation")
                if len(line) > 16 * 1024 * 1024:
                    raise TransformersWorkerError("Transformers worker response exceeds the protocol budget")
                self._responses.put(line)
        except BaseException as exc:
            self._responses.put(exc)

    def _read_errors(self) -> None:
        assert self.process.stderr is not None
        while True:
            chunk = self.process.stderr.read(1024)
            if not chunk:
                return
            self._stderr.extend(chunk)
            del self._stderr[:-8192]

    def _receive(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            if self.cancel.is_set():
                self.close()
                raise TransformersWorkerError("Transformers execution cancelled; worker terminated and reaped")
            if time.monotonic() >= deadline:
                self.close()
                raise TransformersWorkerError(f"Transformers worker exceeded the {timeout:g}-second operation deadline")
            try:
                value = self._responses.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if isinstance(value, BaseException):
                diagnostic = self._stderr.decode("utf-8", "replace")[-1200:]
                raise TransformersWorkerError(f"{value}. {diagnostic}".strip()) from value
            try:
                decoded = json.loads(value)
            except (ValueError, UnicodeError) as exc:
                raise TransformersWorkerError("Invalid worker protocol response") from exc
            if not isinstance(decoded, dict):
                raise TransformersWorkerError("Invalid worker response type")
            if "progress" in decoded:
                if self._on_progress is not None:
                    self._on_progress(decoded["progress"])
                continue
            return decoded

    def request(self, operation: str, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise TransformersWorkerError("Transformers worker is closed")
        payload = json.dumps({"operation": operation, "inputs": inputs, "params": params}, allow_nan=False).encode("utf-8") + b"\n"
        if len(payload) > 12 * 1024 * 1024:
            raise TransformersWorkerError("Transformers request exceeds 12 MiB")
        assert self.process.stdin is not None
        # Writing occurs on a bounded helper so cancellation can reap a worker
        # that exits or stops reading while a large request fills the pipe.
        errors: list[BaseException] = []

        def send() -> None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(payload)
                self.process.stdin.flush()
            except BaseException as exc:
                errors.append(exc)

        sender = threading.Thread(target=send, daemon=True, name="transformers-send")
        sender.start()
        try:
            response = self._receive(limits()["operation_timeout_seconds"])
            if not response.get("ok"):
                raise TransformersWorkerError(str(response.get("error", "Transformers operation failed")))
            return response["outputs"]
        finally:
            if errors and self.process.poll() is not None:
                self.close()
            sender.join(timeout=1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._job is not None:
            kernel, handle = self._job
            kernel.CloseHandle(handle)
            self._job = None
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self._reader.join(timeout=2)
        self._error_reader.join(timeout=2)
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None:
                pipe.close()

    def __enter__(self) -> TransformersWorker:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
