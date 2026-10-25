"""Execute the packaged Llama graph and verify actual training and safe reload.

All model execution occurs through the studio's existing graph tools in a fresh
workspace. This script never opens or replaces the user's saved studio project.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import single_parameter_observatory as spobs
from single_parameter_observatory.studio_codegen import python_project
from single_parameter_observatory.studio_session import StudioWorkspace


def verify_job(job: dict) -> dict:
    """Check outcomes independently of the workflow's custom verification node."""
    if job["status"] != "succeeded" or any(n["status"] != "succeeded" for n in job["nodes"].values()):
        raise AssertionError(f"Llama workflow did not succeed: {job.get('error')}")
    outputs = {identity: value["outputs"] for identity, value in job["nodes"].items()}
    before = outputs["initial-eval"]["metrics"]
    after = outputs["trained-eval"]["metrics"]
    restored = outputs["reloaded-eval"]["metrics"]
    train = outputs["train"]["metrics"]
    report = outputs["trained-inspect"]["report"]
    loaded = outputs["reloaded-inspect"]["report"]
    assert report["architecture"] == loaded["architecture"] == "LlamaForCausalLM"
    assert report["parameter_count"] == 78_144 and report["parameter_tensors"] == 21
    assert report["state_sha256"] == loaded["state_sha256"] == train["state_sha256_after"]
    assert report["handle"] != loaded["handle"], "Reload must reconstruct a distinct live model"
    assert train["parameter_changed"] and train["state_sha256_before"] != train["state_sha256_after"]
    assert train["steps"] == 40 and len(train["loss_history"]) == 40
    assert all(math.isfinite(value) for value in [*train["loss_history"], *train["gradient_norms"]])
    assert after["loss"] < before["loss"] * 0.8, "Expected measured learning on this fixed tiny corpus"
    assert restored["loss"] == after["loss"], "Safe reload must preserve evaluation loss exactly"
    assert not any(item["parameter_changed"] for item in [before, after, restored])
    assert outputs["architecture-checks"]["value"]["verified"] is True
    assert report["module_count"] == len(report["modules"]) and not report["modules_truncated"]
    layers = [m for m in report["modules"] if m["type"] == "LlamaDecoderLayer"]
    assert len(layers) == 2
    assert sum(m["direct_parameter_count"] for m in report["modules"]) == report["parameter_count"]
    shapes = {p["name"]: p["shape"] for p in report["parameters"]}
    for index in range(2):
        prefix = f"model.layers.{index}."
        assert shapes[prefix + "self_attn.q_proj.weight"] == [64, 64]
        assert shapes[prefix + "self_attn.k_proj.weight"] == [32, 64]
        assert shapes[prefix + "self_attn.v_proj.weight"] == [32, 64]
        assert shapes[prefix + "mlp.gate_proj.weight"] == [128, 64]
        assert shapes[prefix + "mlp.up_proj.weight"] == [128, 64]
        assert shapes[prefix + "mlp.down_proj.weight"] == [64, 128]
    generated = outputs["generate"]["result"]
    assert generated["shape"][0] == 3 and 2 < generated["shape"][1] <= 10
    assert len(generated["sequences"]) == 3
    assert all(0 <= token < 32 for row in generated["sequences"] for token in row)
    pipeline = outputs["pipeline"]["result"]
    assert pipeline["task"] == "text-generation" and len(pipeline["outputs"]) == 3
    assert all(isinstance(row[0]["generated_text"], str) for row in pipeline["outputs"])
    return {"status": "passed", "architecture": report["architecture"], "parameters": report["parameter_count"],
            "parameter_tensors": report["parameter_tensors"], "module_count": report["module_count"],
            "decoder_layers": len(layers), "config": report["config"], "steps": train["steps"],
            "optimizer_state_tensors": train["optimizer_state_tensors"], "initial_evaluation_loss": before["loss"],
            "trained_evaluation_loss": after["loss"], "reloaded_evaluation_loss": restored["loss"],
            "weight_sha256_before": train["state_sha256_before"], "weight_sha256_after": report["state_sha256"],
            "reload_hash_equal": True, "reload_loss_equal": True, "generation": generated,
            "decoded_generation": pipeline["outputs"], "nodes_succeeded": len(job["nodes"]),
            "job_id": job["id"], "snapshot_sha256": job["snapshot_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-python-export", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    workflow = Path(spobs.__file__).parent / "examples" / "transformers" / "llama.workflow.json"
    source = workflow.read_bytes()
    with closing(StudioWorkspace()) as workspace:
        created = workspace.graph_action("graph.create", {"graph": json.loads(source)})
        graph_id = created["graph_id"]
        validation = workspace.graph_action("graph.validate", {"graph_id": graph_id})
        if not validation["valid"]:
            raise AssertionError(validation["errors"])
        started = workspace.graph_action("graph.execute", {"graph_id": graph_id})
        job = workspace.graphs.wait_job(started["job_id"], timeout=180)
        (directory / "llama.job.json").write_text(json.dumps(job, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        receipt = verify_job(job)
        artifacts = []
        for artifact in workspace.graphs.snapshot()["artifacts"]:
            if artifact["job_id"] != job["id"]:
                continue
            content, filename, _ = workspace.graphs.artifact(artifact["id"])
            assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                assert archive.testzip() is None
                assert set(archive.namelist()) == {"recipe.json", "manifest.json", "model.safetensors"}
            target = directory / ("llama-" + filename)
            target.write_bytes(content)
            artifacts.append({**artifact, "saved_path": str(target)})
        project = workspace.export_project()
        program = directory / "llama-workflow.py"
        program.write_bytes(python_project(project, graph_id))
        receipt.update(version=spobs.__version__, module_path=spobs.__file__,
                       workflow_sha256=hashlib.sha256(source).hexdigest(), artifacts=artifacts,
                       exported_python_sha256=hashlib.sha256(program.read_bytes()).hexdigest())
    if args.run_python_export:
        result = subprocess.run([sys.executable, "-I", str(program), "--timeout", "180", "--output",
                                 str(directory / "llama-python.job.json"), "--artifacts",
                                 str(directory / "python-artifacts")], capture_output=True, timeout=210, check=False)
        (directory / "llama-python.stdout.json").write_bytes(result.stdout)
        (directory / "llama-python.stderr.txt").write_bytes(result.stderr)
        if result.returncode != 0:
            raise AssertionError(f"Exported Python failed: {result.returncode}; inspect saved stderr")
        exported_job = json.loads((directory / "llama-python.job.json").read_bytes())
        assert json.loads(result.stdout) == exported_job
        exported = verify_job(exported_job)
        assert exported["weight_sha256_after"] == receipt["weight_sha256_after"]
        receipt["python_export_execution"] = exported
        receipt["python_export_reproduced_weight_hash"] = True
    (directory / "llama-verification.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",
                                                     encoding="utf-8")
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
