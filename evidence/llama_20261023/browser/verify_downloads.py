"""Independently inspect downloaded Llama evidence without importing a model."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile
import zlib


HERE = Path(__file__).resolve().parent


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def identity(path: Path) -> dict:
    content = path.read_bytes()
    return {"file": path.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def verify_job(path: Path) -> tuple[dict, dict]:
    job = json.loads(path.read_text(encoding="utf-8"))
    assert job["status"] == "succeeded"
    assert len(job["nodes"]) == len(job["graph"]["nodes"]) == 19
    assert all(node["status"] == "succeeded" for node in job["nodes"].values())
    outputs = {name: record["outputs"] for name, record in job["nodes"].items()}
    before = outputs["initial-inspect"]["report"]
    trained = outputs["trained-inspect"]["report"]
    restored = outputs["reloaded-inspect"]["report"]
    for report in (before, trained, restored):
        assert report["architecture"] == "LlamaForCausalLM"
        assert report["parameter_count"] == report["trainable_parameters"] == 78144
        assert report["parameter_tensors"] == len(report["parameters"]) == 21
        assert report["module_count"] == len(report["modules"]) == 33 and not report["modules_truncated"]
        assert sum(row["direct_parameter_count"] for row in report["modules"]) == 78144
        assert sum(row["elements"] for row in report["parameters"]) == 78144
        assert all(row["finite"] for row in report["parameters"])
        config = report["config"]
        expected = {"model_type": "llama", "vocab_size": 32, "hidden_size": 64, "intermediate_size": 128,
                    "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "head_dim": 16, "max_position_embeddings": 128, "hidden_act": "silu"}
        assert all(config[name] == value for name, value in expected.items())
        assert config["rope_parameters"] == {"rope_type": "default", "rope_theta": 10000.0}
        modules = {row["name"]: row for row in report["modules"]}
        parameters = {row["name"]: row for row in report["parameters"]}
        assert modules["model.rotary_emb"]["type"] == "LlamaRotaryEmbedding"
        assert modules["model.norm"]["type"] == "LlamaRMSNorm"
        for index in range(2):
            prefix = f"model.layers.{index}"
            assert modules[prefix]["type"] == "LlamaDecoderLayer"
            for suffix, kind in {"self_attn": "LlamaAttention", "mlp": "LlamaMLP",
                                 "input_layernorm": "LlamaRMSNorm", "post_attention_layernorm": "LlamaRMSNorm"}.items():
                assert modules[f"{prefix}.{suffix}"]["type"] == kind
            for name, shape in {"self_attn.q_proj": [64, 64], "self_attn.k_proj": [32, 64],
                                "self_attn.v_proj": [32, 64], "self_attn.o_proj": [64, 64],
                                "mlp.gate_proj": [128, 64], "mlp.up_proj": [128, 64],
                                "mlp.down_proj": [64, 128]}.items():
                assert parameters[f"{prefix}.{name}.weight"]["shape"] == shape
    training = outputs["train"]["metrics"]
    assert training["steps"] == len(training["loss_history"]) == 40
    assert training["optimizer"]["name"] == "AdamW" and training["optimizer_state_tensors"] == 63
    assert training["parameter_changed"]
    assert before["state_sha256"] == training["state_sha256_before"]
    assert trained["state_sha256"] == training["state_sha256_after"] == restored["state_sha256"]
    assert before["state_sha256"] != trained["state_sha256"]
    assert trained["handle"] != restored["handle"]
    initial_loss = outputs["initial-eval"]["metrics"]["loss"]
    trained_loss = outputs["trained-eval"]["metrics"]["loss"]
    restored_loss = outputs["reloaded-eval"]["metrics"]["loss"]
    assert all(math.isfinite(value) for value in (initial_loss, trained_loss, restored_loss))
    assert initial_loss > trained_loss == restored_loss
    assert outputs["architecture-checks"]["value"]["verified"]
    assert outputs["forward"]["result"]["outputs"]["logits"]["shape"] == [3, 2, 32]
    graph_nodes = {node["id"]: node for node in job["graph"]["nodes"]}
    vocabulary = graph_nodes["tokenizer"]["params"]["vocab"]
    by_id = {value: key for key, value in vocabulary.items()}
    special = {vocabulary[name] for name in ("[PAD]", "[UNK]", "[BOS]", "[EOS]")}
    generation = outputs["generate"]["result"]
    pipeline = outputs["pipeline"]["result"]
    assert generation["shape"] == [3, 10] and pipeline["task"] == "text-generation"
    assert len(generation["sequences"]) == len(pipeline["outputs"]) == 3
    decoded = []
    for sequence, group in zip(generation["sequences"], pipeline["outputs"], strict=True):
        assert len(group) == 1 and group[0]["generated_token_ids"] == sequence
        assert len(sequence) == 10 and all(type(token) is int and token in by_id for token in sequence)
        text = " ".join(by_id[token] for token in sequence if token not in special)
        assert group[0]["generated_text"] == text
        decoded.append(text)
    return job, {**identity(path), "job_id": job["id"], "graph_id": job["graph_id"],
                 "nodes_succeeded": 19, "architecture": trained["architecture"],
                 "parameter_count": 78144, "parameter_tensors": 21, "module_count": 33,
                 "steps": 40, "optimizer_state_tensors": 63,
                 "initial_loss": initial_loss, "trained_loss": trained_loss, "restored_loss": restored_loss,
                 "state_sha256_before": before["state_sha256"], "state_sha256_after": trained["state_sha256"],
                 "reload_hash_equal": True, "reload_loss_equal": True,
                 "pipeline_ids_equal_generate_ids": True, "full_wordlevel_decode_equal": True,
                 "decoded_generation": decoded}


def verify_checkpoint(path: Path, job: dict) -> dict:
    descriptor = job["nodes"]["checkpoint"]["outputs"]["artifact"]
    receipt = identity(path)
    assert receipt["bytes"] == descriptor["byte_length"] and receipt["sha256"] == descriptor["sha256"]
    assert descriptor["mime"] == "application/zip" and descriptor["job_id"] == job["id"]
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 3
        assert set(archive.namelist()) == {"model.safetensors", "recipe.json", "manifest.json"}
        weights, recipe_data, manifest_data = (archive.read(name) for name in
                                               ("model.safetensors", "recipe.json", "manifest.json"))
    manifest = json.loads(manifest_data)
    recipe = json.loads(recipe_data)
    assert manifest["schema_version"] == "spobs-model-artifact/1"
    assert recipe["schema_version"] == "spobs-tensor-model/1"
    assert recipe["recipe"]["config"]["model_type"] == "llama"
    assert manifest["weight_sha256"] == hashlib.sha256(weights).hexdigest()
    assert manifest["recipe_sha256"] == hashlib.sha256(recipe_data).hexdigest()
    assert manifest["weight_bytes"] == len(weights) and manifest["parameter_count"] == 78144
    header_length = int.from_bytes(weights[:8], "little")
    assert 2 <= header_length <= len(weights) - 8
    header = json.loads(weights[8:8 + header_length])
    payload = weights[8 + header_length:]
    parameters = {row["name"]: row for row in job["nodes"]["trained-inspect"]["outputs"]["report"]["parameters"]}
    assert set(header) == set(parameters) and len(header) == 21
    digest = hashlib.sha256()
    intervals = []
    for name, tensor in sorted(header.items()):
        assert tensor["dtype"] == "F32" and tensor["shape"] == parameters[name]["shape"]
        start, end = tensor["data_offsets"]
        assert 0 <= start < end <= len(payload)
        assert end - start == math.prod(tensor["shape"]) * 4
        raw = payload[start:end]
        assert all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", raw))
        digest.update(json_bytes([name, tensor["shape"], "torch.float32"]))
        digest.update(raw)
        intervals.append((start, end))
    intervals.sort()
    assert intervals[0][0] == 0 and intervals[-1][1] == len(payload)
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:], strict=False))
    state_digest = digest.hexdigest()
    assert state_digest == manifest["state_sha256"]
    assert state_digest == job["nodes"]["trained-inspect"]["outputs"]["report"]["state_sha256"]
    return {**receipt, "descriptor_byte_identity": True, "zip_crc_passed": True,
            "members": ["manifest.json", "model.safetensors", "recipe.json"],
            "safetensors_header_valid": True, "tensor_count": len(header), "all_float32_values_finite": True,
            "tensor_names_shapes_equal_inspection": True, "independently_reconstructed_state_sha256": state_digest,
            "manifest": manifest}


def verify_program(path: Path, browser_job: dict) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {}
    imports = {"__future__", "argparse", "base64", "hashlib", "json", "pathlib", "zlib"}
    for node in tree.body:
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        elif isinstance(node, ast.Import):
            assert all(alias.name in imports for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module in imports
        elif isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            constants[node.targets[0].id] = value
        elif isinstance(node, ast.FunctionDef):
            assert not node.decorator_list and not node.args.defaults and not node.args.kw_defaults
            assert node.name in {"project_data", "main"}
        else:
            assert isinstance(node, ast.If)
            expected = ast.parse('if __name__ == "__main__":\n    raise SystemExit(main())').body[0]
            assert ast.dump(node) == ast.dump(expected)
    assert set(constants) == {"GRAPH_ID", "PROJECT_SHA256", "PROJECT_DATA"}
    raw = zlib.decompress(base64.b64decode(constants["PROJECT_DATA"], validate=True))
    assert hashlib.sha256(raw).hexdigest() == constants["PROJECT_SHA256"]
    project = json.loads(raw)
    graph = next(item for item in project["extensions"]["graphs"] if item["id"] == constants["GRAPH_ID"])
    assert constants["GRAPH_ID"] == browser_job["graph_id"] and graph == browser_job["graph"]
    assert project["application_version"] == "2.1.0"
    return {**identity(path), "ast_inert_on_import": True, "program_not_imported_or_executed_by_this_audit": True,
            "embedded_project_sha256": constants["PROJECT_SHA256"], "embedded_project_bytes": len(raw),
            "embedded_project_hash_valid": True, "application_version": project["application_version"],
            "graph_id": constants["GRAPH_ID"], "embedded_graph_equal_browser_job_graph": True}


def main() -> None:
    job, job_receipt = verify_job(HERE / "llama-browser.job.json")
    checkpoint_receipt = verify_checkpoint(HERE / "llama-browser-model.zip", job)
    program_receipt = verify_program(HERE / "llama-browser-workflow.py", job)
    result = {"status": "passed", "audit_scope": "Independent static/download inspection; no model imported or run",
              "browser_job": job_receipt, "browser_checkpoint": checkpoint_receipt, "browser_python_program": program_receipt}
    python_path = HERE / "llama-browser-python.job.json"
    if python_path.exists():
        python_job, python_receipt = verify_job(python_path)
        assert python_job["graph_id"] == job["graph_id"] and python_job["graph"] == job["graph"]
        assert python_receipt["state_sha256_after"] == job_receipt["state_sha256_after"]
        stdout = HERE / "llama-browser-python.stdout.json"
        assert stdout.read_bytes().replace(b"\r\n", b"\n") == python_path.read_bytes().replace(b"\r\n", b"\n")
        descriptor = python_job["nodes"]["checkpoint"]["outputs"]["artifact"]
        artifact_path = HERE / "python-artifacts" / (descriptor["id"] + "-" + descriptor["filename"])
        python_checkpoint = verify_checkpoint(artifact_path, python_job)
        assert python_checkpoint["manifest"]["weight_sha256"] == checkpoint_receipt["manifest"]["weight_sha256"]
        result["browser_export_python_execution"] = {"job": python_receipt, "checkpoint": python_checkpoint,
                                                      "graph_equal": True, "stdout_equal_saved_job": True,
                                                      "final_weight_state_equal_browser": True,
                                                      "safetensors_bytes_equal_browser": True}
    else:
        result["browser_export_python_execution"] = {"status": "pending root execution"}
    (HERE / "download-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
