"""Package final disposable Llama component examples without reading user storage.

Uses only the standard library. Verifies saved evidence; never executes a model,
exported program, plugin, or user tensor program.
"""
from __future__ import annotations

import ast
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any
import zipfile
import zlib


VERSION = "2.2.0"
VARIANTS = ("expanded", "custom-two-layer", "fused")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode()


def strict_json(data: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value: str) -> Any:
        raise ValueError("Non-finite JSON constant: " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def safe_paths(names: list[str]) -> None:
    assert len(names) == len(set(names)) == len({name.casefold() for name in names})
    for name in names:
        path = PurePosixPath(name)
        assert name and not path.is_absolute() and ".." not in path.parts
        assert "\\" not in name and ":" not in name and not name.endswith("/")
        assert path.as_posix() == name


def inspect_model(data: bytes, report: dict[str, Any], verified: dict[str, Any]) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        safe_paths(archive.namelist())
        assert set(archive.namelist()) == {"manifest.json", "recipe.json", "model.safetensors"}
        assert archive.testzip() is None
        manifest = strict_json(archive.read("manifest.json"))
        recipe_data = archive.read("recipe.json")
        recipe = strict_json(recipe_data)
        weights = archive.read("model.safetensors")
    assert manifest["schema_version"] == "spobs-model-artifact/1"
    assert recipe["schema_version"] == "spobs-tensor-model/1"
    assert recipe["recipe"]["family"] == "tensor-graph"
    assert recipe["recipe"] == report["recipe"]
    assert manifest["recipe_sha256"] == digest(recipe_data)
    assert manifest["weight_sha256"] == digest(weights)
    assert manifest["weight_bytes"] == len(weights)
    assert manifest["parameter_count"] == verified["parameters"] == report["parameter_count"]
    header_length = int.from_bytes(weights[:8], "little")
    assert 2 <= header_length <= len(weights) - 8
    header = strict_json(weights[8:8 + header_length])
    payload = weights[8 + header_length:]
    parameters = {row["name"]: row for row in report["parameters"]}
    assert set(header) == set(parameters) and len(header) == verified["parameter_tensors"]
    state_hash = hashlib.sha256()
    intervals = []
    elements = 0
    for name, tensor in sorted(header.items()):
        assert tensor["dtype"] == "F32" and tensor["shape"] == parameters[name]["shape"]
        start, end = tensor["data_offsets"]
        count = math.prod(tensor["shape"])
        assert 0 <= start < end <= len(payload) and end - start == count * 4
        raw = payload[start:end]
        assert all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", raw))
        state_hash.update(json_bytes([name, tensor["shape"], "torch.float32"]))
        state_hash.update(raw)
        intervals.append((start, end))
        elements += count
    intervals.sort()
    assert intervals[0][0] == 0 and intervals[-1][1] == len(payload)
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:], strict=False))
    assert elements == verified["parameters"]
    assert state_hash.hexdigest() == manifest["state_sha256"] == verified["state_sha256"] == report["state_sha256"]
    return {"sha256": digest(data), "bytes": len(data), "state_sha256": state_hash.hexdigest(),
            "recipe_sha256": digest(recipe_data), "parameter_tensors": len(header), "parameters": elements,
            "finite_float32_values": True, "complete_safetensors_payload": True, "zip_crc_passed": True}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    qualification = Path(__file__).parent / "final-qualification"
    sources: dict[Path, str] = {}

    def read(path: Path) -> bytes:
        content = path.read_bytes()
        sources[path] = digest(content)
        return content

    verification_data = read(qualification / "verification.json")
    verification = strict_json(verification_data)
    assert verification["status"] == "passed" and verification["version"] == VERSION
    assert set(verification["variants"]) == set(VARIANTS)
    files: dict[str, bytes] = {"verification.json": verification_data}
    proofs: dict[str, Any] = {}
    for name in VARIANTS:
        directory = qualification / name
        data = {filename: read(directory / filename) for filename in
                ("workflow.json", "project.json", "model.zip", "workflow.py", "job.json")}
        verified = verification["variants"][name]
        job = strict_json(data["job.json"])
        assert verified["status"] == "passed" and verified["version"] == VERSION
        assert job["status"] == "succeeded" and job["id"] == verified["job_id"]
        assert job["snapshot_sha256"] == verified["snapshot_sha256"]
        assert len(job["nodes"]) == len(job["graph"]["nodes"]) == verified["nodes_succeeded"]
        assert all(node["status"] == "succeeded" for node in job["nodes"].values())
        assert verified["trained_loss"] < verified["initial_loss"]
        assert verified["trained_loss"] == verified["reloaded_loss"]
        replay = verified["python_replay"]
        assert replay["status"] == "passed" and replay["version"] == VERSION
        assert replay["state_sha256"] == verified["state_sha256"]
        assert replay["snapshot_sha256"] == verified["snapshot_sha256"]
        workflow = strict_json(data["workflow.json"])
        assert workflow == {key: job["graph"][key] for key in workflow}
        report = job["nodes"]["inspect"]["outputs"]["report"]
        model_proof = inspect_model(data["model.zip"], report, verified)
        constants = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse(data["workflow.py"]).body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        embedded_data = zlib.decompress(base64.b64decode(constants["PROJECT_DATA"], validate=True))
        assert digest(embedded_data) == constants["PROJECT_SHA256"]
        project = strict_json(embedded_data)
        source_project = strict_json(data["project.json"])
        normalized_project, normalized_source = copy.deepcopy(project), copy.deepcopy(source_project)
        embedded_time = normalized_project.pop("saved_at")
        source_time = normalized_source.pop("saved_at")
        assert normalized_project == normalized_source
        assert project["application_version"] == VERSION
        extension = project["extensions"]
        assert len(extension["graphs"]) == 2 and len(extension["jobs"]) == 1
        assert extension["jobs"][0] == job and not extension["plugins"]
        assert extension["active_graph_id"] == constants["GRAPH_ID"] == job["graph_id"]
        assert next(graph for graph in extension["graphs"] if graph["id"] == constants["GRAPH_ID"]) == job["graph"]
        assert len(extension["artifacts"]) == 1
        artifact = extension["artifacts"][0]
        assert artifact["job_id"] == job["id"] and artifact["node_id"] == "checkpoint"
        assert artifact["sha256"] == digest(data["model.zip"]) and artifact["byte_length"] == len(data["model.zip"])
        assert base64.b64decode(artifact["data_base64"], validate=True) == data["model.zip"]
        descriptor = job["nodes"]["checkpoint"]["outputs"]["artifact"]
        assert all(artifact[key] == value for key, value in descriptor.items())
        # The only scalar experiment is the untouched disposable workspace's
        # starter. No user experiment history or browser drafts enter this bundle.
        assert len(project["runs"]) == 1 and not project["workspace"]["notes"]
        run = project["runs"][0]
        state = dict(run["engine"]["state"]["$map"])
        assert state["records"] == [] and dict(state["misc"]["$map"])["step_index"] == 0
        assert not run["checkpoints"] and not run["notes"] and not run["engine"]["interventions"]
        assert project["ui_drafts"] == {"schema_version": "spobs-ui-drafts/1", "runs": {},
                                         "plugin": "", "plugin_dirty": False}
        # Deliver the exact Python-embedded snapshot, rather than the later
        # pretty export whose sole content difference is its saved_at timestamp.
        data["project.json"] = embedded_data
        files.update({name + "/" + filename: content for filename, content in data.items()})
        proofs[name] = {"job_id": job["id"], "nodes_succeeded": len(job["nodes"]), "model": model_proof,
                        "project_sha256": digest(embedded_data), "python_embedded_project_byte_equal": True,
                        "source_project_sha256": sources[directory / "project.json"],
                        "source_project_equivalence": "Complete structural equality after removing only saved_at",
                        "source_project_saved_at": source_time, "embedded_project_saved_at": embedded_time,
                        "logical_project_sha256_without_saved_at": digest(json_bytes(normalized_project)),
                        "registered_model_artifact_byte_equal": True, "fresh_workspace_without_user_history": True,
                        "qualified_python_replay_same_state_sha256": True}

    guide_source = read(root / "LLAMA_BLOCKS.md")
    guide = guide_source.decode("utf-8")
    adaptations = []
    for old, new in (
        ("[native Llama guide](LLAMA_STUDIO.md)", "native Llama guide in the full Studio distribution"),
        ("[Transformers guide](TRANSFORMERS_STUDIO.md)", "Transformers guide in the full Studio distribution"),
        ("[Node lab guide](NODE_LAB.md)", "Node lab guide in the full Studio distribution"),
    ):
        assert guide.count(old) == 1
        guide = guide.replace(old, new)
        adaptations.append({"from": old, "to": new, "occurrences": 1})
    files["LLAMA_BLOCKS.md"] = guide.encode("utf-8")
    files["LLAMA_SOURCE_NOTES.md"] = read(root / "LLAMA_SOURCE_NOTES.md")
    for example in ("llama-blocks.workflow.json", "llama-blocks-custom.workflow.json"):
        relative = "examples/transformers/" + example
        files[relative] = read(root / relative)
    table = "\n".join(f"| [{name}]({name}/workflow.json) | {verification['variants'][name]['nodes_succeeded']} | "
                      f"{verification['variants'][name]['parameters']:,} | {verification['variants'][name]['steps']} | "
                      f"{verification['variants'][name]['initial_loss']:.7f} → {verification['variants'][name]['trained_loss']:.7f} |"
                      for name in VARIANTS)
    files["README.md"] = ("""# Runnable visual Llama component laboratory · Studio 2.2.0

These three examples are complete, editable tensor graphs compiled into real
PyTorch causal models. Their trained safe checkpoints, completed disposable
projects, actual job receipts and runnable Python exports are included. Every
RMSNorm, rotary operation, attention branch and residual participates in execution.
They are small from-scratch educational models, not pretrained Meta weights.

| Workflow | Nodes | Parameters | AdamW updates | Evaluated corpus loss before → after |
| --- | ---: | ---: | ---: | ---: |
""" + table + """

## Open and edit

1. Start Parameter Studio **2.2.0** in its installed environment.
2. To ADD an editable workflow to your current workspace, choose Node lab →
   Workflow → Import workflow JSON, then select a variant's `workflow.json`.
   Validate, then Queue workflow to initialize and train a fresh model.
3. To inspect a COMPLETED example, save your current workspace first. Use the
   studio's Open project command and select the variant's `project.json`.
   **Open project replaces the current workspace.** Each included project is a
   fresh disposable qualification workspace with its own registered checkpoint;
   it contains no user workspace history. Use Inspect model visually on its job.
4. Edit scalar controls, drag input/output sockets for compatible blocks, reconnect
   operations, and edit a Custom tensor operation when custom mathematics is needed.
   Changed dimensions and topology are validated before the model is allocated.

## Run the Python export

Python exports require the installed **Studio 2.2.0** package and its tensor-runtime
dependencies. Enter one variant directory and run with that environment's Python:

```powershell
python workflow.py --timeout 300 --output replay-job.json --artifacts replay-artifacts
```

This explicitly replays the saved workflow and trains a fresh model. Merely
extracting this ZIP or opening the Python file executes nothing. `project.json`
is byte-for-byte the decompressed project embedded in that directory's Python
export, including its completed job and registered checkpoint.

Each `model.zip` contains only `model.safetensors`, `recipe.json`, and
`manifest.json`. It preserves the connected architecture and trained weights;
optimizer/RNG training resume is outside this format. `job.json` records the
completed source execution. `verification.json` records checkpoint equality and
the successful Python replays with identical final model-state hashes.

Read [the visual component guide](LLAMA_BLOCKS.md) for the controls, custom
tensor language, generation and budgets, and [source notes](LLAMA_SOURCE_NOTES.md)
for the native Llama/HF Viewer reference review. The guide's standalone repository
qualification command and additional guides belong to the full Studio source
distribution. Its two linked example JSON files are included at their original
paths; the three qualified variants above include their own exact workflows.

`MANIFEST.json` hashes every other member and records packaging provenance.
The package builder rechecked ZIP CRCs, every byte hash, exact embedded projects,
registered model bytes, recipe family, all finite float32 weights, complete tensor
payload coverage and model-state hashes without rerunning training.
""").encode("utf-8")
    safe_paths(list(files))
    for name, content in files.items():
        if name.endswith(".json"):
            strict_json(content)
        if name.endswith(".md"):
            for target in re.findall(r"\]\(([^)]+)\)", content.decode("utf-8")):
                if "://" not in target and not target.startswith("#"):
                    assert target.split("#", 1)[0] in files, (name, target)
    identities = {name: {"bytes": len(content), "sha256": digest(content)} for name, content in sorted(files.items())}
    provenance = {"source_guide_sha256": digest(guide_source), "delivered_guide_sha256": digest(files["LLAMA_BLOCKS.md"]),
                  "guide_adaptations": adaptations, "qualification_sha256": digest(verification_data)}
    manifest = {"schema_version": "spobs-component-laboratory/1", "version": VERSION,
                "files": identities, "provenance": provenance, "variants": proofs}
    files["MANIFEST.json"] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 24, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, content, compresslevel=9)
    output_data = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(output_data)) as archive:
        safe_paths(archive.namelist())
        assert set(archive.namelist()) == set(files) and archive.testzip() is None
        for name, content in files.items():
            assert archive.read(name) == content
    assert all(digest(path.read_bytes()) == expected for path, expected in sources.items()), "A packaged source changed during assembly"
    output = root / "download" / "llama-blocks-studio-2.2.0.zip"
    output.write_bytes(output_data)
    result = {"status": "passed", "version": VERSION, "path": output.relative_to(root).as_posix(),
              "bytes": len(output_data), "sha256": digest(output_data), "verified_files": len(identities),
              "archive_members": len(files), "manifest_sha256": digest(files["MANIFEST.json"]),
              "zip_crc_passed": True, "safe_unique_paths": True, "all_member_bytes_verified": True,
              "packaging_helper_sha256": digest(Path(__file__).read_bytes()), "provenance": provenance,
              "variants": proofs, "source_files": {path.relative_to(root).as_posix(): value for path, value in sources.items()}}
    receipt_path = root / "download" / "component-bundle-2.2.0.json"
    receipt_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"variants", "source_files"}}, indent=2))


if __name__ == "__main__":
    main()
