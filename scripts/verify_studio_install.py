"""Exercise a wheel-installed studio and write concrete HTTP/artifact evidence.

Run with the installed environment's Python and ``-I`` so checkout imports
cannot conceal a missing package asset or entry module.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

import torch

import single_parameter_observatory as spobs
from single_parameter_observatory.studio_server import create_server
from single_parameter_observatory.studio_plugins import example_plugin


def node(identity: str, kind: str, **params: object) -> dict:
    return {"id": identity, "type": kind, "position": {"x": 0, "y": 0}, "params": params}


def edge(source: str, source_port: str, target: str, target_port: str) -> dict:
    return {"id": f"{source}-{source_port}-{target}-{target_port}", "source": source,
            "source_port": source_port, "target": target, "target_port": target_port}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distributions", type=Path,
                        help="also compare built wheel/sdist bytes with this checkout")
    args = parser.parse_args()
    module_path = Path(spobs.__file__).resolve()
    if not module_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected installed module beneath {sys.prefix}, got {module_path}")
    server = create_server(port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def request(path: str, payload: dict | None = None, *, headers: dict | None = None) -> tuple[bytes, dict]:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request_headers = {"Content-Type": "application/json"} if body else {}
        request_headers.update(headers or {})
        req = urllib.request.Request(server.url + path.lstrip("/"), data=body,
                                     headers=request_headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 200:
                raise AssertionError(response.status)
            return response.read(), dict(response.headers)

    def action(name: str, **payload: object) -> dict:
        content, _ = request("api/action", {"action": name, **payload})
        return json.loads(content)

    def graph_action(name: str, **payload: object) -> dict:
        content, _ = request("api/graph/action", {"action": name, **payload})
        return json.loads(content)

    def create_graph(name: str, nodes: list[dict], edges: list[dict] | None = None) -> str:
        return graph_action("graph.create", graph={"name": name, "nodes": nodes, "edges": edges or []})["result"]["graph_id"]

    def wait_job(job_id: str, *, timeout: float = 60) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            job = json.loads(request(f"api/graph/jobs/{job_id}")[0])
            if job["status"] not in ("queued", "running", "cancelling"):
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Installed node job {job_id} did not finish")
            threading.Event().wait(0.05)

    def execute(graph_id: str, token: str, *, timeout: float = 60) -> dict:
        queued = graph_action("graph.execute", graph_id=graph_id, idempotency_key=token)
        job = wait_job(queued["result"]["job_id"], timeout=timeout)
        if job["status"] != "succeeded":
            raise AssertionError(f"Installed workflow failed: {job['error']}")
        return job

    def verify_zip_bytes(archive_bytes: bytes, artifact: dict) -> list[str]:
        if (hashlib.sha256(archive_bytes).hexdigest() != artifact["sha256"]
                or len(archive_bytes) != artifact["byte_length"] or artifact["mime"] != "application/zip"):
            raise AssertionError("The immutable graph evidence download differs from its receipt")
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            if archive.testzip() is not None:
                raise AssertionError("Export ZIP CRC error")
            members = sorted(archive.namelist())
            if len(members) != len(set(members)):
                raise AssertionError("Export ZIP has duplicate member names")
            if set(members) == {"model.safetensors", "recipe.json", "manifest.json"}:
                weights = archive.read("model.safetensors")
                recipe_bytes = archive.read("recipe.json")
                recipe = json.loads(recipe_bytes)
                manifest = json.loads(archive.read("manifest.json"))
                if (manifest.get("schema_version") != "spobs-model-artifact/1"
                        or recipe.get("schema_version") != "spobs-tensor-model/1"
                        or manifest.get("weight_sha256") != hashlib.sha256(weights).hexdigest()
                        or manifest.get("recipe_sha256") != hashlib.sha256(recipe_bytes).hexdigest()
                        or manifest.get("weight_bytes") != len(weights)
                        or type(manifest.get("parameter_count")) is not int
                        or manifest["parameter_count"] < 2
                        or len(manifest.get("state_sha256", "")) != 64):
                    raise AssertionError("Safe model checkpoint schema, identity or parameter count differs")
                if len(weights) < 8:
                    raise AssertionError("Safe model weights are not a safetensors document")
                header_length = int.from_bytes(weights[:8], "little")
                if not 2 <= header_length <= len(weights) - 8:
                    raise AssertionError("Safe model weights have an invalid safetensors header length")
                tensors = json.loads(weights[8:8 + header_length])
                if not any(name != "__metadata__" for name in tensors):
                    raise AssertionError("Safe model checkpoint contains no tensor state")
            else:
                if not any(name.endswith("trajectory.csv") for name in members):
                    raise AssertionError("Numerical evidence export missing")
                if "checkpoint.json" not in members or "dataset.json" not in members:
                    raise AssertionError("Restorable numerical state or actual dataset evidence missing")
        return members

    def verify_artifact(artifact: dict) -> tuple[bytes, list[str]]:
        archive_bytes, archive_headers = request(artifact["download_url"])
        if (archive_headers.get("Content-Type") != artifact["mime"]
                or "attachment" not in archive_headers.get("Content-Disposition", "")):
            raise AssertionError("The immutable graph evidence download has incorrect MIME/attachment headers")
        members = verify_zip_bytes(archive_bytes, artifact)
        return archive_bytes, members

    def verify_transformer_outputs(job: dict) -> dict:
        """Check real observations, not only a successful graph status label."""
        observed = {}
        graph_nodes = {item["id"]: item for item in job["graph"]["nodes"]}
        for identity, record in job["nodes"].items():
            if record["status"] != "succeeded":
                raise AssertionError(f"Installed Transformer node did not succeed: {identity}")
            kind = graph_nodes[identity]["type"]
            outputs = record["outputs"]
            if kind == "hf.train":
                metrics = outputs["metrics"]
                steps = graph_nodes[identity]["params"].get("steps", 20)
                if (metrics["steps"] != steps or len(metrics["loss_history"]) != steps
                        or len(metrics["gradient_norms"]) != steps or not metrics["parameter_changed"]
                        or metrics["state_sha256_before"] == metrics["state_sha256_after"]
                        or metrics["parameter_count"] < 2
                        or any(not math.isfinite(value) for value in metrics["loss_history"] + metrics["gradient_norms"])):
                    raise AssertionError(f"Installed training did not make measured finite parameter updates: {identity}")
                if metrics["optimizer"]["name"] in {"Adam", "AdamW"} and metrics["optimizer_state_tensors"] <= 0:
                    raise AssertionError("Adam-family training did not create actual optimizer tensor state")
                observed[identity] = {"type": kind, "steps": steps, "parameter_count": metrics["parameter_count"],
                                      "state_sha256_before": metrics["state_sha256_before"],
                                      "state_sha256_after": metrics["state_sha256_after"],
                                      "loss_first": metrics["loss_history"][0], "loss_last": metrics["loss_history"][-1]}
            elif kind == "hf.evaluate":
                metrics = outputs["metrics"]
                if not math.isfinite(metrics["loss"]) or metrics["parameter_changed"]:
                    raise AssertionError("Evaluation changed parameters or produced nonfinite loss")
            elif kind == "hf.inspect":
                report = outputs["report"]
                if (report["parameter_count"] < 2 or report["parameter_tensors"] < 2
                        or not report["parameters"] or any(not item["finite"] for item in report["parameters"])
                        or len(report["state_sha256"]) != 64):
                    raise AssertionError("Inspection did not observe complete finite multi-parameter model metadata")
            elif kind == "hf.generate":
                result = outputs["result"]
                if not result["sequences"] or result["shape"][0] != len(result["sequences"]):
                    raise AssertionError("HF generate did not return actual token sequences")
            elif kind == "hf.pipeline":
                if not outputs["result"]["outputs"]:
                    raise AssertionError("The installed HF pipeline returned no actual outputs")
            elif kind == "hf.pretrained":
                source = outputs["model"]["source"]
                if (source["repo_id"] != "hf-internal-testing/tiny-random-gpt2"
                        or source["revision"] != "71034c5d8bde858ff824298bdedc65515b97d2b9"
                        or outputs["model"]["parameter_count"] != 111968):
                    raise AssertionError("The cached HF fixture's pinned source/model identity differs")
            elif kind == "hf.load":
                incoming = next(item for item in job["graph"]["edges"]
                                if item["target"] == identity and item["target_port"] == "artifact")
                original = job["nodes"][incoming["source"]]["outputs"]["model"]
                restored = outputs["model"]
                if original["handle"] == restored["handle"] or original["parameter_count"] != restored["parameter_count"]:
                    raise AssertionError("Safe checkpoint load did not construct a distinct matching real model")
        if "before" in job["nodes"] and "after" in job["nodes"]:
            before = job["nodes"]["before"]["outputs"]["report"]
            after = job["nodes"]["after"]["outputs"]["report"]
            if before["state_sha256"] != after["state_sha256"]:
                raise AssertionError("Safe checkpoint round trip changed actual model-state bytes")
            observed["checkpoint_round_trip_state_sha256"] = after["state_sha256"]
        return observed

    def verify_llama_outputs(job: dict) -> dict:
        """Bind the packaged Llama example to its actual modules and trained state."""
        nodes = job["nodes"]
        initial = nodes["initial-inspect"]["outputs"]["report"]
        trained = nodes["trained-inspect"]["outputs"]["report"]
        restored = nodes["reloaded-inspect"]["outputs"]["report"]
        expected_config = {"model_type": "llama", "vocab_size": 32, "hidden_size": 64,
                           "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
                           "num_key_value_heads": 2, "head_dim": 16, "hidden_act": "silu"}
        for report in (initial, trained, restored):
            if (report["architecture"] != "LlamaForCausalLM" or report["parameter_count"] != 78144
                    or report["parameter_tensors"] != 21 or report["trainable_parameters"] != 78144
                    or report["modules_truncated"] or report["module_count"] != 33
                    or len(report["modules"]) != report["module_count"]
                    or len(report["parameters"]) != report["parameter_tensors"]
                    or any(report["config"].get(key) != value for key, value in expected_config.items())
                    or report["config"]["rope_parameters"] != {"rope_type": "default", "rope_theta": 10000.0}
                    or sum(item["direct_parameter_count"] for item in report["modules"]) != 78144
                    or sum(item["elements"] for item in report["parameters"]) != 78144):
                raise AssertionError("The installed Llama architecture/configuration or complete parameter inventory differs")
            modules = {item["name"]: item for item in report["modules"]}
            parameters = {item["name"]: item for item in report["parameters"]}
            for index in range(2):
                prefix = f"model.layers.{index}"
                expected_modules = {prefix: "LlamaDecoderLayer", f"{prefix}.self_attn": "LlamaAttention",
                                    f"{prefix}.mlp": "LlamaMLP", f"{prefix}.input_layernorm": "LlamaRMSNorm",
                                    f"{prefix}.post_attention_layernorm": "LlamaRMSNorm"}
                expected_shapes = {"self_attn.q_proj.weight": [64, 64], "self_attn.k_proj.weight": [32, 64],
                                   "self_attn.v_proj.weight": [32, 64], "self_attn.o_proj.weight": [64, 64],
                                   "mlp.gate_proj.weight": [128, 64], "mlp.up_proj.weight": [128, 64],
                                   "mlp.down_proj.weight": [64, 128]}
                if (any(modules.get(name, {}).get("type") != kind for name, kind in expected_modules.items())
                        or any(parameters.get(f"{prefix}.{name}", {}).get("shape") != shape
                               for name, shape in expected_shapes.items())):
                    raise AssertionError("Llama decoder, grouped-query projections or SwiGLU shapes differ")
            if (modules.get("model.rotary_emb", {}).get("type") != "LlamaRotaryEmbedding"
                    or modules.get("model.norm", {}).get("type") != "LlamaRMSNorm"
                    or parameters.get("model.embed_tokens.weight", {}).get("shape") != [32, 64]
                    or parameters.get("lm_head.weight", {}).get("shape") != [32, 64]):
                raise AssertionError("Native Llama rotary positions, final normalization or vocabulary projections are missing")
        loss_before = nodes["initial-eval"]["outputs"]["metrics"]["loss"]
        loss_after = nodes["trained-eval"]["outputs"]["metrics"]["loss"]
        loss_restored = nodes["reloaded-eval"]["outputs"]["metrics"]["loss"]
        training = nodes["train"]["outputs"]["metrics"]
        custom_checks = nodes["architecture-checks"]["outputs"]["value"]
        if (not all(math.isfinite(value) for value in (loss_before, loss_after, loss_restored))
                or loss_after >= loss_before or loss_after != loss_restored
                or initial["state_sha256"] == trained["state_sha256"]
                or trained["state_sha256"] != restored["state_sha256"]
                or initial["state_sha256"] != training["state_sha256_before"]
                or trained["state_sha256"] != training["state_sha256_after"]
                or trained["handle"] == restored["handle"] or training["steps"] != 40
                or training["optimizer"]["name"] != "AdamW" or not custom_checks.get("verified")
                or custom_checks.get("parameters") != 78144 or custom_checks.get("tensors") != 21
                or nodes["forward"]["outputs"]["result"]["outputs"]["logits"]["shape"] != [3, 2, 32]
                or len(nodes["generate"]["outputs"]["result"]["sequences"]) != 3
                or len(nodes["pipeline"]["outputs"]["result"]["outputs"]) != 3):
            raise AssertionError("Installed Llama training, custom verification, inference or exact checkpoint reload failed")
        return {"architecture": trained["architecture"], "parameter_count": 78144, "parameter_tensors": 21,
                "module_count": 33, "decoder_layers": 2, "query_heads": 4, "key_value_heads": 2,
                "loss_before": loss_before, "loss_after": loss_after, "loss_reloaded": loss_restored,
                "state_sha256_before": initial["state_sha256"], "state_sha256_after": trained["state_sha256"],
                "reloaded_state_equal": True, "reloaded_loss_equal": True, "custom_script_verified": True,
                "inference_logit_shape": [3, 2, 32], "generated_sequences": 3, "decoded_pipeline_outputs": 3}

    def require_stale_epoch(path: str, epoch: str, payload: dict | None = None) -> None:
        try:
            request(path, payload, headers={"X-Studio-Epoch": epoch})
        except urllib.error.HTTPError as exc:
            rejection = json.loads(exc.read())
            if exc.code != 409 or rejection.get("type") != "StudioEpochError":
                raise AssertionError(f"Unexpected stale-workspace rejection: {exc.code}: {rejection}") from exc
        else:
            raise AssertionError(f"Stale workspace identity was accepted for {path}")

    try:
        assets = {}
        for path in ("", "studio.css", "studio.js", "graph.css", "graph.js", "graph_core.js", "transformers.js", "transformers.css"):
            content, headers = request(path)
            if not content or "Content-Security-Policy" not in headers:
                raise AssertionError(f"Missing asset or security headers: {path}")
            assets[path or "index.html"] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        state = json.loads(request("api/state")[0])
        initial_epoch = state["workspace_epoch"]
        if json.loads(request("api/graph/state", headers={"X-Studio-Epoch": initial_epoch})[0])["workspace_epoch"] != initial_epoch:
            raise AssertionError("Main and graph snapshots disagree on the live workspace identity")
        root_id = state["workspace"]["active_run_id"]
        state = action("run.step", run_id=root_id, count=4)["state"]
        sealed_root_record = json.loads(request(f"api/runs/{root_id}/record?step=3")[0])
        if state["runs"][0]["step"] != 4 or not sealed_root_record["record_hash"]:
            raise AssertionError("Actual optimizer observations missing")
        child_id = action("run.dataset", run_id=root_id, csv="x,target\n-1,-0.8\n0,0\n1,0.8\n")['result']['run_id']
        configured = action("run.configure", run_id=child_id, config={"lr": 0.03, "steps": 20})
        configured_id = configured["result"]["run_id"]
        if configured["result"].get("dataset_mode") != "retained_custom":
            raise AssertionError("Configuration discarded custom samples")
        data = json.loads(request(f"api/runs/{configured_id}/dataset")[0])
        if data["inputs"] != [-1.0, 0.0, 1.0] or data["provenance"]["kind"] != "csv":
            raise AssertionError("Custom data not retained")
        action("run.step", run_id=configured_id, count=2)
        catalog = json.loads(request("api/graph/catalog")[0])
        core_types = {item["id"] for item in catalog["types"] if item["id"].startswith("core.")}
        expected_types = {f"core.{name}" for name in (
            "number", "text", "boolean", "json", "math", "array", "object", "get", "script", "preview", "config",
            "dataset", "experiment", "train", "branch", "checkpoint", "inspect", "intervene", "verify", "control", "export",
        )}
        if core_types != expected_types or not catalog["sandbox"]["available"]:
            raise AssertionError("The installed node catalog or sandbox dependency is incomplete")
        transformer_types = {item["id"] for item in catalog["types"] if item["id"].startswith("hf.")}
        expected_transformer_types = {f"hf.{name}" for name in (
            "config", "model", "pretrained", "auto_tokenizer", "tokenizer", "tokenize", "data", "layer", "linear",
            "embedding", "rmsnorm", "layernorm", "activation", "attention", "compose", "stack", "sequential",
            "custom", "optimizer", "train", "forward", "evaluate", "generate", "pipeline", "inspect", "checkpoint", "load",
            "tensor_input", "op_embedding", "op_linear", "op_rmsnorm", "op_positions", "op_rope",
            "op_attention", "op_silu", "op_multiply", "op_add", "op_swiglu", "op_custom", "graph_model",
        )}
        if (transformer_types != expected_transformer_types or not catalog["transformers"]["available"]
                or catalog["transformers"]["version"] != "5.20.2"):
            raise AssertionError(f"The installed Transformers catalog/runtime differs: {catalog.get('transformers')}")
        script_id = create_graph("Installed sandbox", [node(
            "script", "core.script", inputs={"value": 21},
            code="console.log('installed V8 result'); return {value: inputs.value * 2};",
        )])
        script_job = execute(script_id, "installed-script")
        if script_job["nodes"]["script"]["outputs"]["value"] != 42 or not script_job["nodes"]["script"]["logs"]:
            raise AssertionError("The installed sandbox did not produce actual typed outputs and logs")

        plugin = example_plugin()
        graph_action("plugin.install", manifest=plugin)
        installed_catalog = json.loads(request("api/graph/catalog")[0])
        template_id = f"{plugin['id']}/{plugin['templates'][0]['id']}"
        if template_id not in {item["id"] for item in installed_catalog["templates"]}:
            raise AssertionError("Installed plugin template is absent from the catalog")
        plugin_id = graph_action("graph.create", template_id=template_id)["result"]["graph_id"]
        graph_state = json.loads(request("api/graph/state")[0])
        plugin_graph = next(item for item in graph_state["graphs"] if item["id"] == plugin_id)
        next(item for item in plugin_graph["nodes"] if item["type"] == "core.number")["params"]["value"] = 3
        next(item for item in plugin_graph["nodes"] if item["type"] == f"{plugin['id']}/scale")["params"]["factor"] = 4
        graph_action("graph.update", graph=plugin_graph, expected_revision=plugin_graph["revision"])
        plugin_job = execute(plugin_id, "installed-plugin-template")
        preview_id = next(item["id"] for item in plugin_job["graph"]["nodes"] if item["type"] == "core.preview")
        if plugin_job["nodes"][preview_id]["outputs"]["value"] != 12:
            raise AssertionError("The installed plugin/template did not execute its actual script")

        example_directory = module_path.parent / "examples" / "node_lab"
        example_files = sorted(example_directory.glob("*.workflow.json"))
        expected_examples = {f"{name}.workflow.json" for name in (
            "branch-and-export", "custom-data", "plugin-dataset-training", "plugin-scale",
            "sandbox", "training", "wide-integer",
        )}
        if {file.name for file in example_files} != expected_examples:
            raise AssertionError("The installed package does not contain all seven portable workflow examples")
        example_results = []
        for file in example_files:
            source_bytes = file.read_bytes()
            graph_id = graph_action("graph.create", graph=json.loads(source_bytes))["result"]["graph_id"]
            validation = graph_action("graph.validate", graph_id=graph_id)["result"]
            if not validation["valid"]:
                raise AssertionError(f"Packaged example {file.name} is invalid: {validation['errors']}")
            job = execute(graph_id, f"installed-example-{file.stem}")
            node_outputs = {identity: record["outputs"] for identity, record in job["nodes"].items()}
            artifacts = []
            for outputs in node_outputs.values():
                if "artifact" in outputs:
                    descriptor = outputs["artifact"]
                    _content, artifact_members = verify_artifact(descriptor)
                    artifacts.append({**descriptor, "verified_zip_members": artifact_members})
            expected_preview = {"sandbox.workflow.json": 42, "plugin-scale.workflow.json": 42,
                                "wide-integer.workflow.json": "9223372036854775808"}.get(file.name)
            if expected_preview is not None and node_outputs["preview"]["value"] != expected_preview:
                raise AssertionError(f"Packaged example {file.name} produced an incorrect actual result")
            if file.name == "plugin-dataset-training.workflow.json":
                if node_outputs["inspect"]["dataset"]["targets"] != [-6.0, 0.0, 6.0]:
                    raise AssertionError("The packaged custom-node example discarded transformed training targets")
                if node_outputs["inspect"]["record"]["step"] != 20:
                    raise AssertionError("The packaged custom-node example did not perform its twenty optimizer steps")
            example_results.append({"file": file.name, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                                    "graph_id": graph_id, "job_id": job["id"], "status": job["status"],
                                    "snapshot_sha256": job["snapshot_sha256"], "outputs": node_outputs,
                                    "verified_artifacts": artifacts})

        transformer_directory = module_path.parent / "examples" / "transformers"
        transformer_plugins = sorted(transformer_directory.glob("*.plugin.json"))
        if {file.name for file in transformer_plugins} != {"architecture.plugin.json"}:
            raise AssertionError("The packaged custom Transformer architecture plugin is missing")
        transformer_plugin_results = []
        for file in transformer_plugins:
            source_bytes = file.read_bytes()
            manifest = json.loads(source_bytes)
            graph_action("plugin.install", manifest=manifest)
            current_catalog = json.loads(request("api/graph/catalog")[0])
            if any(f"{manifest['id']}/{item['id']}" not in {item["id"] for item in current_catalog["types"]}
                   for item in manifest["nodes"]):
                raise AssertionError("Installed architecture plugin nodes are absent from the live catalog")
            transformer_plugin_results.append({"file": file.name, "id": manifest["id"],
                                               "source_sha256": hashlib.sha256(source_bytes).hexdigest()})
        transformer_files = sorted(transformer_directory.glob("*.workflow.json"))
        expected_transformer_examples = {f"{name}.workflow.json" for name in (
            "cached-pretrained", "checkpoint-roundtrip", "custom-attention", "custom-forward",
            "distilbert-classification", "feature-extraction", "fill-mask", "llama", "llama-blocks", "llama-blocks-custom", "plugin-architecture",
            "rmsnorm-adamw", "text-pipeline", "tiny-gpt2", "token-classification", "tokenize-train-generate",
        )}
        if {file.name for file in transformer_files} != expected_transformer_examples:
            raise AssertionError("The installed package does not contain all sixteen Transformer workflows")
        transformer_results = []
        pipeline_tasks = set()
        tiny_gpt2_graph_id = None
        llama_graph_id = None
        llama_result = None
        for file in transformer_files:
            source_bytes = file.read_bytes()
            graph_id = graph_action("graph.create", graph=json.loads(source_bytes))["result"]["graph_id"]
            validation = graph_action("graph.validate", graph_id=graph_id)["result"]
            if not validation["valid"]:
                raise AssertionError(f"Packaged Transformer example {file.name} is invalid: {validation['errors']}")
            job = execute(graph_id, f"installed-transformer-example-{file.stem}", timeout=120)
            observations = verify_transformer_outputs(job)
            node_outputs = {identity: record["outputs"] for identity, record in job["nodes"].items()}
            artifacts = []
            for outputs in node_outputs.values():
                if "artifact" in outputs:
                    descriptor = outputs["artifact"]
                    _content, artifact_members = verify_artifact(descriptor)
                    artifacts.append({**descriptor, "verified_zip_members": artifact_members})
            for item in job["graph"]["nodes"]:
                if item["type"] == "hf.pipeline":
                    pipeline_tasks.add(item["params"].get("task", "text-generation"))
            if file.name == "tiny-gpt2.workflow.json":
                tiny_gpt2_graph_id = graph_id
            if file.name == "llama.workflow.json":
                llama_graph_id = graph_id
                llama_result = verify_llama_outputs(job)
            if file.name == "plugin-architecture.workflow.json":
                layers = node_outputs["recipes"]["layers"]
                if len(layers) != 4 or layers[1]["type"] != "RMSNorm" or not observations.get("train"):
                    raise AssertionError("The custom JavaScript architecture did not feed real model training")
            transformer_results.append({"file": file.name, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                                        "graph_id": graph_id, "job_id": job["id"], "status": job["status"],
                                        "snapshot_sha256": job["snapshot_sha256"], "outputs": node_outputs,
                                        "observed_training": observations, "verified_artifacts": artifacts})
        if pipeline_tasks != {"text-generation", "text-classification", "token-classification", "fill-mask", "feature-extraction"}:
            raise AssertionError("The installed example suite did not execute all five exposed pipeline tasks")
        if tiny_gpt2_graph_id is None:
            raise AssertionError("The installed tiny GPT-2 graph needed for Python export is absent")
        if llama_graph_id is None or llama_result is None:
            raise AssertionError("The installed native Llama graph and verified architecture are absent")

        python_export_results = {}
        for export_label, selected_graph_id, expected_training_steps in (
            ("tiny_gpt2", tiny_gpt2_graph_id, 3), ("llama", llama_graph_id, 40),
        ):
            python_source, python_headers = request(f"api/graph/python/{selected_graph_id}")
            if (not python_source or not python_headers.get("Content-Type", "").startswith("text/x-python")
                    or "attachment" not in python_headers.get("Content-Disposition", "")):
                raise AssertionError("Runnable Python project export has incorrect bytes or attachment headers")
            compile(python_source, f"installed_{export_label}.py", "exec")
            with tempfile.TemporaryDirectory(prefix="spobs-installed-python-export-") as export_directory:
                temporary = Path(export_directory)
                exported_script = temporary / "generated.py"
                exported_job = temporary / "job.json"
                exported_artifacts = temporary / "artifacts"
                exported_script.write_bytes(python_source)
                completed = subprocess.run(
                    [sys.executable, "-I", str(exported_script), "--timeout", "120", "--output", str(exported_job),
                     "--artifacts", str(exported_artifacts)],
                    cwd=temporary, capture_output=True, timeout=150, check=False,
                )
                if completed.returncode != 0:
                    raise AssertionError(
                        f"Installed exported HF Python project failed ({completed.returncode}): "
                        f"{completed.stderr[-4000:].decode('utf-8', errors='replace')}"
                    )
                job_bytes = exported_job.read_bytes()
                if completed.stdout.replace(b"\r\n", b"\n") != job_bytes.replace(b"\r\n", b"\n"):
                    raise AssertionError("Exported Python stdout and saved exact job JSON differ")
                python_job = json.loads(job_bytes)
                if python_job["status"] != "succeeded" or python_job["graph_id"] != selected_graph_id:
                    raise AssertionError("The exported Python program did not execute the captured selected HF workflow")
                python_observations = verify_transformer_outputs(python_job)
                if not python_observations.get("train") or python_observations["train"]["steps"] != expected_training_steps:
                    raise AssertionError("Exported Python did not perform the selected HF graph's actual optimizer steps")
                exported_llama_observations = verify_llama_outputs(python_job) if export_label == "llama" else None
                python_artifact_results = []
                expected_artifact_names = set()
                for python_node_observation in python_job["nodes"].values():
                    artifact = python_node_observation["outputs"].get("artifact")
                    if artifact is None:
                        continue
                    name = artifact["id"] + "-" + artifact["filename"]
                    expected_artifact_names.add(name)
                    content = (exported_artifacts / name).read_bytes()
                    artifact_members = verify_zip_bytes(content, artifact)
                    python_artifact_results.append({**artifact, "saved_filename": name,
                                                    "verified_zip_members": artifact_members})
                if not expected_artifact_names or {file.name for file in exported_artifacts.iterdir()} != expected_artifact_names:
                    raise AssertionError("Exported Python saved missing, extra or non-owned checkpoint artifacts")
                python_export_result = {"source_sha256": hashlib.sha256(python_source).hexdigest(),
                                        "source_bytes": len(python_source), "installed_python": sys.executable,
                                        "isolated_python": True, "returncode": completed.returncode,
                                        "job_id": python_job["id"], "graph_id": selected_graph_id,
                                        "job_sha256": hashlib.sha256(job_bytes).hexdigest(), "job_bytes": len(job_bytes),
                                        "stdout_equal_to_saved_job": True, "observed_training": python_observations,
                                        "verified_artifacts": python_artifact_results,
                                        "llama_architecture": exported_llama_observations}
            python_export_results[export_label] = python_export_result
        python_export_result = python_export_results["tiny_gpt2"]
        llama_python_export_result = python_export_results["llama"]

        small = {"steps": 10, "dataset": {"samples": 6, "batch_size": 2}, "validation": {"enabled": False},
                 "landscape": {"enabled": False}, "run_fp64_reference_track": False}
        training_id = create_graph("Installed real training", [
            node("config", "core.config", value=small), node("data", "core.dataset", batch_size=2),
            node("experiment", "core.experiment"), node("train", "core.train", steps=2),
            node("checkpoint", "core.checkpoint"), node("inspect", "core.inspect", include_dataset=True),
        ], [edge("config", "config", "experiment", "config"), edge("data", "dataset", "experiment", "dataset"),
            edge("experiment", "run", "train", "run"), edge("train", "run", "checkpoint", "run"),
            edge("checkpoint", "run", "inspect", "run")])
        training_job = execute(training_id, "installed-training")
        graph_run_id = training_job["nodes"]["experiment"]["outputs"]["run"]
        graph_record = json.loads(request(f"api/runs/{graph_run_id}/record?step=1")[0])
        if training_job["nodes"]["inspect"]["outputs"]["record"]["step"] != 2:
            raise AssertionError("The installed graph did not execute actual training steps")
        if training_job["nodes"]["inspect"]["outputs"]["dataset"]["targets"] != [-2.0, 0.0, 2.0]:
            raise AssertionError("The graph training node discarded custom input/target samples")

        wide_value = 2**60 + 1
        wide_id = create_graph("Exact JSON evidence", [node("value", "core.json", value=wide_value)])
        wide_job = execute(wide_id, "installed-wide-integer")
        wide_bytes, wide_headers = request(f"api/graph/jobs/{wide_job['id']}")
        if (f'"value":{wide_value}'.encode() not in wide_bytes
                or json.loads(wide_bytes)["nodes"]["value"]["outputs"]["value"] != wide_value
                or "attachment" not in wide_headers.get("Content-Disposition", "")):
            raise AssertionError("Raw job evidence lost integer precision or its download headers")

        cancel_id = create_graph("Installed cancellation", [node("busy", "core.script", code="while (true) {}", timeout_ms=10000)])
        cancel_job_id = graph_action("graph.execute", graph_id=cancel_id)["result"]["job_id"]
        deadline = time.monotonic() + 20
        while True:
            pending = json.loads(request(f"api/graph/jobs/{cancel_job_id}")[0])
            if pending["status"] == "running":
                break
            if pending["status"] not in ("queued", "running") or time.monotonic() >= deadline:
                raise AssertionError(f"Cancellation fixture never ran: {pending['status']}")
            threading.Event().wait(0.02)
        graph_action("job.cancel", job_id=cancel_job_id)
        cancelled_job = wait_job(cancel_job_id, timeout=20)
        if cancelled_job["status"] != "cancelled":
            raise AssertionError("Cancellation reported success/failure instead of cancellation")

        export_id = create_graph("Installed exact evidence", [node("export", "core.export", run=configured_id)])
        export_job = execute(export_id, "installed-evidence")
        artifact = export_job["nodes"]["export"]["outputs"]["artifact"]
        archive_bytes, members = verify_artifact(artifact)

        drafts = json.loads(request("api/ui/drafts")[0])
        drafts["drafts"].update(plugin="{ retained uninstalled manifest draft", plugin_dirty=True,
                                 panel_fields={f"{plugin['id']}/example_controls": {"factor": "unfinished number"}})
        action("workspace.drafts", drafts=drafts["drafts"], expected_revision=drafts["revision"])
        project_bytes, _ = request("api/project")
        project = json.loads(project_bytes)
        opened = json.loads(request("api/project", project, headers={"X-Studio-Epoch": initial_epoch})[0])
        reopened_epoch = opened["state"]["workspace_epoch"]
        if reopened_epoch == initial_epoch:
            raise AssertionError("Opening a project did not replace the live workspace identity")
        for path in ("api/state", "api/project", f"api/graph/jobs/{wide_job['id']}", artifact["download_url"]):
            require_stale_epoch(path, initial_epoch)
        require_stale_epoch("api/graph/action", initial_epoch, {"action": "graph.select", "graph_id": wide_id})
        for path in ("api/state", "api/graph/state", "api/ui/drafts"):
            if json.loads(request(path, headers={"X-Studio-Epoch": reopened_epoch})[0])["workspace_epoch"] != reopened_epoch:
                raise AssertionError(f"Reopened workspace identity disagrees in {path}")
        restored = json.loads(request(f"api/runs/{root_id}/record?step=3")[0])
        if restored != sealed_root_record:
            raise AssertionError("Project round trip changed a sealed observation")
        graph_restored = json.loads(request(f"api/runs/{graph_run_id}/record?step=1")[0])
        if graph_restored != graph_record:
            raise AssertionError("Project round trip changed the graph-produced numerical observations")
        if request(artifact["download_url"])[0] != archive_bytes:
            raise AssertionError("Project round trip changed immutable ZIP bytes")
        if json.loads(request("api/ui/drafts")[0])["drafts"] != drafts["drafts"]:
            raise AssertionError("Project round trip applied or discarded unapplied drafts")
        if json.loads(request(f"api/graph/jobs/{cancel_job_id}")[0])["status"] != "cancelled":
            raise AssertionError("Opening a project resumed a cancelled job")
        if json.loads(request(f"api/graph/jobs/{wide_job['id']}")[0])["nodes"]["value"]["outputs"]["value"] != wide_value:
            raise AssertionError("Opening a project rounded raw job evidence")
        report = {
            "status": "passed", "version": spobs.__version__, "python": platform.python_version(),
            "torch": torch.__version__, "module_path": str(module_path), "sys_prefix": sys.prefix,
            "assets": assets, "real_root_steps": 4, "real_custom_steps": 2,
            "custom_data_retained_after_configure": True, "project_record_round_trip_equal": True,
            "project_bytes": len(project_bytes), "project_sha256": hashlib.sha256(project_bytes).hexdigest(),
            "zip_bytes": len(archive_bytes), "zip_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "zip_members": members,
            "node_lab": {"core_node_types": sorted(core_types), "core_node_count": len(core_types),
                         "sandbox": catalog["sandbox"], "actual_script_value": 42, "actual_plugin_template_value": 12,
                         "plugin_id": plugin["id"], "plugin_template": template_id, "actual_graph_training_steps": 2,
                         "actual_graph_record_hash": graph_record["record_hash"], "cancelled_job": cancel_job_id,
                         "cancelled_status": cancelled_job["status"], "exact_job_integer": str(wide_value),
                         "raw_job_sha256": hashlib.sha256(wide_bytes).hexdigest(),
                         "artifact": artifact, "artifact_project_round_trip_byte_equal": True,
                         "unapplied_drafts_round_trip_equal": True, "job_snapshots_reopened_without_execution": True,
                         "workspace_epoch_rotated_and_stale_requests_rejected": True,
                         "executed_packaged_examples": example_results},
            "transformers": {"node_types": sorted(transformer_types), "node_count": len(transformer_types),
                             "runtime": catalog["transformers"], "installed_plugins": transformer_plugin_results,
                             "pipeline_tasks_executed": sorted(pipeline_tasks),
                             "executed_packaged_examples": transformer_results,
                             "runnable_python_project": python_export_result,
                             "llama_architecture": llama_result,
                             "runnable_llama_python_project": llama_python_export_result},
        }
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        if worker.is_alive():
            raise AssertionError("HTTP server failed to stop")
    if args.distributions is not None:
        checkout = Path(__file__).resolve().parents[1]
        package = checkout / "single_parameter_observatory"
        ignored_parts = {"__pycache__", "run_checks_export", "runs", "evidence"}
        sources = sorted(file for file in package.rglob("*") if file.is_file()
                         and not ignored_parts.intersection(file.relative_to(package).parts)
                         and (file.suffix in (".py", ".md", ".html", ".css", ".js")
                              or file.name in ("py.typed", "LICENSE", "requirements.txt")
                              or file.relative_to(package).parts[0] == "studio_web"
                              or (file.relative_to(package).parts[:2] in {("examples", "node_lab"), ("examples", "transformers")}
                                  and file.suffix in (".json", ".ps1"))))
        guides = ("README.md", "STUDIO.md", "STUDIO_REPORT.md", "NODE_LAB.md", "PLUGIN_SDK.md", "NODE_LAB_REPORT.md",
                  "TRANSFORMERS_AUDIT.md", "TRANSFORMERS_STUDIO.md", "TRANSFORMERS_REPORT.md",
                  "LLAMA_STUDIO.md", "LLAMA_SOURCE_NOTES.md", "LLAMA_BLOCKS.md")
        identical_guides = ("NODE_LAB.md", "PLUGIN_SDK.md", "NODE_LAB_REPORT.md",
                            "TRANSFORMERS_AUDIT.md", "TRANSFORMERS_STUDIO.md", "TRANSFORMERS_REPORT.md",
                            "LLAMA_STUDIO.md", "LLAMA_SOURCE_NOTES.md", "LLAMA_BLOCKS.md")
        for name in guides:
            source_guide, package_guide = checkout / name, package / name
            if not source_guide.is_file() or not package_guide.is_file():
                raise AssertionError(f"Root or packaged guide is missing: {name}")
            if name in identical_guides and source_guide.read_bytes() != package_guide.read_bytes():
                raise AssertionError(f"Root and packaged release guides differ: {name}")
            if name in ("README.md", "STUDIO.md"):
                for guide in (source_guide, package_guide):
                    if spobs.__version__ not in guide.read_text(encoding="utf-8"):
                        raise AssertionError(f"Guide does not describe the current {spobs.__version__} release: {guide}")
        examples = checkout / "examples"
        root_examples = []
        example_groups = {}
        for group in ("node_lab", "transformers"):
            example_group_directory = examples / group
            files = sorted(file for file in example_group_directory.rglob("*") if file.is_file()
                           and file.suffix in (".json", ".ps1", ".py", ".md"))
            if not files or not any(file.suffix == ".json" for file in files):
                raise AssertionError(f"Portable workflow examples are missing: {group}")
            if group == "node_lab" and not any(file.suffix == ".ps1" for file in files):
                raise AssertionError("Portable PowerShell examples are missing")
            for file in files:
                packaged = package / "examples" / file.relative_to(examples)
                if not packaged.is_file() or packaged.read_bytes() != file.read_bytes():
                    raise AssertionError(f"Packaged example differs from root example: {file.relative_to(examples)}")
            example_groups[group] = [file.relative_to(example_group_directory).as_posix() for file in files]
            root_examples.extend(files)
        wheel = args.distributions / f"single_parameter_observatory-{spobs.__version__}-py3-none-any.whl"
        sdist = args.distributions / f"single_parameter_observatory-{spobs.__version__}.tar.gz"
        with zipfile.ZipFile(wheel) as archive:
            for file in sources:
                name = file.relative_to(checkout).as_posix()
                if archive.read(name) != file.read_bytes():
                    raise AssertionError(f"Wheel differs from current source: {name}")
                installed = module_path.parent / file.relative_to(package)
                if not installed.is_file() or installed.read_bytes() != file.read_bytes():
                    raise AssertionError(f"Installed package differs from current source: {name}")
        root_sources = [checkout / name for name in (
            *guides, "Launch Studio.cmd",
            "MANIFEST.in", "pyproject.toml", "LICENSE", "requirements.txt",
        )]
        with tarfile.open(sdist) as source_archive:
            prefix = source_archive.getnames()[0].split("/")[0]
            for file in sources + root_sources + root_examples:
                name = file.relative_to(checkout).as_posix()
                member = source_archive.extractfile(f"{prefix}/{name}")
                if member is None or member.read() != file.read_bytes():
                    raise AssertionError(f"Source distribution differs from source: {name}")
        report["wheel_source_files_equal"] = len(sources)
        report["installed_source_files_equal"] = len(sources)
        report["sdist_source_files_equal"] = len(sources) + len(root_sources) + len(root_examples)
        report["portable_examples_equal"] = example_groups["node_lab"]
        report["transformers_examples_equal"] = example_groups["transformers"]
        report["guides_equal"] = list(identical_guides)
        report["guides_present"] = list(guides)
        report["current_version_guidance"] = ["README.md", "STUDIO.md"]
        report["distributions"] = [{"name": file.name, "bytes": file.stat().st_size,
                                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
                                   for file in (wheel, sdist)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
