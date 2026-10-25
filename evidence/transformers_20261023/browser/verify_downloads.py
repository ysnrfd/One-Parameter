"""Independently inspect saved browser downloads using only Python's stdlib.

No studio modules, models, workers, or downloaded Python code are executed.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
import zipfile
import zlib


BASE = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw: bytes | str):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"Nonfinite JSON numeric token: {value}")
        return parsed

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)


def identity(path: Path):
    raw = path.read_bytes()
    return {"path": path.relative_to(BASE).as_posix(), "bytes": len(raw), "sha256": digest(raw)}


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode_maps(value):
    """Decode the declared JSON map wrapper for comparison; never deserialize tensors."""
    if isinstance(value, dict):
        if set(value) == {"$map"}:
            return {key: decode_maps(item) for key, item in value["$map"]}
        return {key: decode_maps(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_maps(item) for item in value]
    return value


def safe_zip(raw: bytes):
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = archive.namelist()
    assert len(names) == len(set(names)), "Duplicate ZIP names"
    assert archive.testzip() is None, "ZIP CRC failure"
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        assert not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename
        assert not item.flag_bits & 1, "Encrypted ZIP member"
    return archive, sorted(names)


def verify_job(path: Path):
    raw = path.read_bytes()
    job = strict_json(raw)
    assert job["status"] == "succeeded" and job["error"] is None
    nodes = {item["id"]: item for item in job["graph"]["nodes"]}
    assert len(nodes) == len(job["graph"]["nodes"]) == len(job["nodes"])
    assert set(nodes) == set(job["nodes"]) == set(job["order"])
    assert len(job["order"]) == len(set(job["order"]))
    captured = {key: job[key] for key in ("graph", "definitions", "plugin_versions", "order")}
    assert digest(canonical(captured)) == job["snapshot_sha256"], "Immutable execution snapshot differs"
    training, artifacts = [], []
    for key, observation in job["nodes"].items():
        assert observation["status"] == "succeeded" and observation["error"] is None
        assert isinstance(observation["inputs"], dict) and isinstance(observation["outputs"], dict)
        if nodes[key]["type"] == "hf.train":
            metrics = observation["outputs"]["metrics"]
            assert metrics["parameter_changed"] and metrics["state_sha256_before"] != metrics["state_sha256_after"]
            assert metrics["steps"] == len(metrics["loss_history"]) == len(metrics["gradient_norms"])
            assert metrics["steps"] == nodes[key]["params"].get("steps", 20)
            assert all(math.isfinite(value) for value in metrics["loss_history"] + metrics["gradient_norms"])
            training.append({"node_id": key, "steps": metrics["steps"], "parameter_count": metrics["parameter_count"],
                             "optimizer": metrics["optimizer"]["name"], "parameter_changed": True,
                             "loss_first": metrics["loss_history"][0], "loss_last": metrics["loss_history"][-1],
                             "state_sha256_before": metrics["state_sha256_before"],
                             "state_sha256_after": metrics["state_sha256_after"]})
        artifact = observation["outputs"].get("artifact")
        if artifact is not None:
            assert artifact["job_id"] == job["id"] and artifact["node_id"] == key
            assert artifact["mime"] == "application/zip" and artifact["byte_length"] > 0
            assert len(artifact["sha256"]) == 64 and all(char in "0123456789abcdef" for char in artifact["sha256"])
            assert artifact["download_url"] == f"/api/graph/artifacts/{artifact['id']}"
            artifacts.append(artifact)
    return job, {**identity(path), "job_id": job["id"], "status": job["status"], "node_count": len(nodes),
                 "all_nodes_succeeded": True, "immutable_snapshot_sha256_verified": job["snapshot_sha256"],
                 "training": training, "artifact_descriptors": artifacts}


def verify_model_zip(path: Path, descriptor, expected_state_hash: str):
    raw = path.read_bytes()
    assert digest(raw) == descriptor["sha256"] and len(raw) == descriptor["byte_length"]
    archive, members = safe_zip(raw)
    try:
        assert set(members) == {"model.safetensors", "recipe.json", "manifest.json"}
        weights, recipe_raw = archive.read("model.safetensors"), archive.read("recipe.json")
        manifest, recipe = strict_json(archive.read("manifest.json")), strict_json(recipe_raw)
    finally:
        archive.close()
    assert manifest["schema_version"] == "spobs-model-artifact/1"
    assert recipe["schema_version"] == "spobs-tensor-model/1"
    assert manifest["weight_bytes"] == len(weights) and manifest["weight_sha256"] == digest(weights)
    assert manifest["recipe_sha256"] == digest(recipe_raw)
    header_length = int.from_bytes(weights[:8], "little")
    assert 2 <= header_length <= len(weights) - 8
    header = strict_json(weights[8:8 + header_length])
    payload = weights[8 + header_length:]
    type_map = {"F32": (4, "torch.float32", "f"), "F64": (8, "torch.float64", "d"),
                "F16": (2, "torch.float16", "e"), "I64": (8, "torch.int64", "q"),
                "I32": (4, "torch.int32", "i"), "I16": (2, "torch.int16", "h"),
                "I8": (1, "torch.int8", "b"), "U8": (1, "torch.uint8", "B"), "BOOL": (1, "torch.bool", "?")}
    state_digest, tensor_records, occupied = hashlib.sha256(), [], []
    for name in sorted(key for key in header if key != "__metadata__"):
        entry = header[name]
        width, torch_dtype, format_code = type_map[entry["dtype"]]
        shape = entry["shape"]
        assert all(type(size) is int and size >= 0 for size in shape)
        count = math.prod(shape)
        start, end = entry["data_offsets"]
        assert 0 <= start <= end <= len(payload) and end - start == count * width
        data = payload[start:end]
        assert all(not (left < end and start < right) for left, right in occupied)
        occupied.append((start, end))
        if entry["dtype"].startswith("F"):
            assert all(math.isfinite(value[0]) for value in struct.iter_unpack("<" + format_code, data))
        state_digest.update(canonical([name, shape, torch_dtype]))
        state_digest.update(data)
        tensor_records.append({"name": name, "shape": shape, "dtype": entry["dtype"], "elements": count,
                               "data_sha256": digest(data)})
    assert state_digest.hexdigest() == manifest["state_sha256"] == expected_state_hash
    return {**identity(path), "descriptor_byte_length_and_sha256_equal": True, "zip_crc_valid": True,
            "members": members, "manifest": manifest, "safetensors_header_and_ranges_valid": True,
            "all_float_values_finite": True, "independent_tensor_state_sha256": state_digest.hexdigest(),
            "tensor_count": len(tensor_records), "stored_elements": sum(item["elements"] for item in tensor_records),
            "tensor_records": tensor_records}


def main():
    project_path = BASE / "downloaded-project.json"
    project = strict_json(project_path.read_bytes())
    assert project["schema_version"] == "spobs-studio/1" and project["application_version"] == "2.0.0"
    run_ids = [item["id"] for item in project["runs"]]
    assert len(run_ids) == len(set(run_ids)) and project["workspace"]["active_run_id"] in run_ids
    for item in project["runs"]:
        assert digest(canonical(item["engine"]["state"])) == item["engine"]["state_hash"]
    project_result = {**identity(project_path), "schema_version": project["schema_version"],
                      "application_version": project["application_version"], "run_count": len(run_ids),
                      "unique_run_ids_and_existing_active_run": True,
                      "graph_count": len(project["extensions"]["graphs"]), "jobs": len(project["extensions"]["jobs"])}

    experiment_path = BASE / "downloaded-experiment.zip"
    archive, members = safe_zip(experiment_path.read_bytes())
    try:
        assert {"checkpoint.json", "dataset.json", "trajectory.csv", "trajectory.json", "summary.json", "provenance.json"} <= set(members)
        checkpoint, dataset = strict_json(archive.read("checkpoint.json")), strict_json(archive.read("dataset.json"))
        summary = strict_json(archive.read("summary.json"))
        for name in members:
            if name.endswith(".json"):
                strict_json(archive.read(name))
        member_hashes = {name: {"bytes": archive.getinfo(name).file_size, "sha256": digest(archive.read(name))} for name in members}
    finally:
        archive.close()
    assert digest(canonical(checkpoint["state"])) == checkpoint["state_hash"]
    matching, memoized_cache_only = [], []
    for item in project["runs"]:
        if item["engine"] == checkpoint:
            matching.append(item["id"])
            continue
        left, right = dict(item["engine"]), dict(checkpoint)
        left_state, right_state = decode_maps(left.pop("state")), decode_maps(right.pop("state"))
        left.pop("state_hash")
        right.pop("state_hash")
        left_state["misc"].pop("_provenance_cache", None)
        right_state["misc"].pop("_provenance_cache", None)
        if left == right and left_state == right_state:
            memoized_cache_only.append(item["id"])
    assert matching or memoized_cache_only, "The numerical export differs beyond a memoized provenance cache"
    assert summary["chain_verified"] is True and summary["record_count"] > 0
    assert len(dataset["inputs"]) == len(dataset["targets"]) > 0
    experiment_result = {**identity(experiment_path), "zip_crc_valid": True, "members": members,
                         "member_hashes": member_hashes, "strict_json_members_valid": True,
                         "checkpoint_exact_matches_project_run_ids": matching,
                         "checkpoint_matches_except_memoized_provenance_cache_run_ids": memoized_cache_only,
                         "project_and_export_state_hashes_independently_valid": True,
                         "cross_snapshot_limit": "Project and ZIP downloads were captured separately; a derived _provenance_cache field changed, while model, gradient, optimizer, scheduler, data, records, events, config, and interventions match",
                         "record_count": summary["record_count"],
                         "saved_chain_verified": summary["chain_verified"], "samples": len(dataset["inputs"]),
                         "limit": "chain_verified is the saved application's observation; this stdlib inspection does not reconstruct optimizer steps"}

    rms_job, rms_result = verify_job(BASE / "rmsnorm-job.json")
    assert rms_result["node_count"] == 6
    rms_training = rms_result["training"][0]
    assert rms_training["parameter_count"] == 33 and rms_training["optimizer"] == "AdamW" and rms_training["steps"] == 20
    rms_descriptor = rms_job["nodes"]["checkpoint"]["outputs"]["artifact"]
    rms_model = verify_model_zip(BASE / "rmsnorm-model.zip", rms_descriptor, rms_training["state_sha256_after"])

    all_job, all_result = verify_job(BASE / "all-nodes.job.json")
    fixture = strict_json((BASE / "all-nodes.workflow.json").read_bytes())
    assert all_result["node_count"] == 135 and len(all_job["graph"]["edges"]) == 124
    assert all_job["graph"]["nodes"] == fixture["nodes"] and all_job["graph"]["edges"] == fixture["edges"]
    core = {"core." + item for item in "number text boolean json math array object get script preview config dataset experiment train branch checkpoint inspect intervene verify control export".split()}
    hf = {"hf." + item for item in "config model pretrained auto_tokenizer tokenizer tokenize data layer linear embedding rmsnorm layernorm activation attention compose stack sequential custom optimizer train forward evaluate generate pipeline inspect checkpoint load".split()}
    expected = core | hf
    observed = {item["type"] for item in all_job["graph"]["nodes"]}
    assert len(expected) == 48 and expected <= observed and observed - expected == {"local.transformer.recipes/mlp_layers"}
    pipeline_tasks, load_checks, round_trips = set(), [], []
    graph_nodes = {item["id"]: item for item in all_job["graph"]["nodes"]}
    for key, item in graph_nodes.items():
        record = all_job["nodes"][key]
        if item["type"] == "hf.pipeline":
            task = item["params"]["task"]
            assert record["outputs"]["result"]["task"] == task
            pipeline_tasks.add(task)
        if item["type"] == "hf.load":
            incoming = next(edge for edge in all_job["graph"]["edges"] if edge["target"] == key and edge["target_port"] == "artifact")
            before = all_job["nodes"][incoming["source"]]["outputs"]["model"]
            after = record["outputs"]["model"]
            assert before["handle"] != after["handle"] and before["parameter_count"] == after["parameter_count"]
            load_checks.append({"node_id": key, "new_model_handle": True, "parameter_count_equal": True})
    for prefix in ("qa01:", "qa02:"):
        before = all_job["nodes"][prefix + "before"]["outputs"]["report"]["state_sha256"]
        after = all_job["nodes"][prefix + "after"]["outputs"]["report"]["state_sha256"]
        assert before == after
        round_trips.append({"prefix": prefix, "before_and_after_state_sha256_equal": before})
    assert pipeline_tasks == {"text-generation", "text-classification", "token-classification", "fill-mask", "feature-extraction"}
    all_result.update({"fixture_nodes_and_edges_exact_equal": True, "builtin_types_expected": sorted(expected),
                       "builtin_types_executed": len(expected), "additional_plugin_types_executed": sorted(observed - expected),
                       "pipeline_tasks_executed": sorted(pipeline_tasks), "checkpoint_loads": load_checks,
                       "observed_checkpoint_state_round_trips": round_trips,
                       "total_multi_parameter_optimizer_steps": sum(item["steps"] for item in all_result["training"]),
                       "artifact_byte_verification_limit": "All descriptors are bound to this job/node; only separately downloaded ZIPs in this report have actual bytes inspected"})

    python_path = BASE / "browser-exported-workflow.py"
    python_raw = python_path.read_bytes()
    tree = ast.parse(python_raw.decode("utf-8"))
    constants = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            if statement.targets[0].id in {"GRAPH_ID", "PROJECT_SHA256", "PROJECT_DATA"}:
                constants[statement.targets[0].id] = ast.literal_eval(statement.value)
    compressed = base64.b64decode(constants["PROJECT_DATA"], validate=True)
    decompressor = zlib.decompressobj()
    embedded_raw = decompressor.decompress(compressed, 64 * 1024 * 1024 + 1)
    assert decompressor.eof and not decompressor.unused_data and len(embedded_raw) <= 64 * 1024 * 1024
    assert digest(embedded_raw) == constants["PROJECT_SHA256"]
    embedded = strict_json(embedded_raw)
    captured_graph = next(item for item in embedded["extensions"]["graphs"] if item["id"] == constants["GRAPH_ID"])
    python_job, python_result = verify_job(BASE / "python-export.job.json")
    assert python_job["graph_id"] == constants["GRAPH_ID"] and python_result["node_count"] == 6
    assert captured_graph["nodes"] == python_job["graph"]["nodes"] and captured_graph["edges"] == python_job["graph"]["edges"]
    python_training = python_result["training"][0]
    assert python_training["optimizer"] == "RMSprop" and python_training["parameter_count"] == 81 and python_training["steps"] == 4
    console = (BASE / "python-export-console.txt").read_bytes()
    job_raw = (BASE / "python-export.job.json").read_bytes()
    assert console.replace(b"\r\n", b"\n").strip() == job_raw.replace(b"\r\n", b"\n").strip()
    artifact_checks = []
    for descriptor in python_result["artifact_descriptors"]:
        artifact_path = BASE / "python-export-artifacts" / (descriptor["id"] + "-" + descriptor["filename"])
        artifact_checks.append(verify_model_zip(artifact_path, descriptor, python_training["state_sha256_after"]))
    assert artifact_checks and len(artifact_checks) == len(list((BASE / "python-export-artifacts").glob("*.zip")))
    python_result.update({"downloaded_python": identity(python_path), "embedded_project_sha256_verified": constants["PROJECT_SHA256"],
                          "selected_captured_graph_nodes_and_edges_equal": True, "stdout_and_job_file_equal": True,
                          "independently_verified_checkpoint_files": artifact_checks,
                          "inspection_did_not_execute_downloaded_code": True})
    report = {"schema_version": "spobs-browser-download-verification/1", "status": "passed",
              "scope": "Independent stdlib inspection of saved browser downloads and separately observed exported-Python results; no model execution or browser interaction performed by this verifier",
              "downloaded_project": project_result, "downloaded_experiment": experiment_result,
              "rmsnorm_job": rms_result, "rmsnorm_model_zip": rms_model,
              "all_nodes_browser_job": all_result, "downloaded_python_execution": python_result,
              "coverage_limit": "Saved evidence proves the named executions and file identities. It does not prove every possible architecture, parameter setting, UI gesture, responsive layout, failure mode or future extension."}
    (BASE / "download-verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "all_nodes": all_result["node_count"],
                      "builtins": all_result["builtin_types_executed"], "pipelines": len(pipeline_tasks),
                      "downloaded_rmsnorm_parameters": rms_training["parameter_count"],
                      "exported_python_nodes": python_result["node_count"],
                      "exported_python_parameters": python_training["parameter_count"],
                      "exported_python_optimizer": python_training["optimizer"]}, indent=2))


if __name__ == "__main__":
    main()
