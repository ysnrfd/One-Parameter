"""Qualify visual Llama DAG variants, actual training, generation and Python replay."""
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
from typing import Any

import single_parameter_observatory as spobs
from single_parameter_observatory.studio_codegen import python_project
from single_parameter_observatory.studio_components import llama_blocks_workflow
from single_parameter_observatory.studio_session import StudioWorkspace


def verify_job(job: dict[str, Any]) -> dict[str, Any]:
    assert job["status"] == "succeeded", job.get("error")
    assert all(value["status"] == "succeeded" for value in job["nodes"].values())
    outputs = {identity: value["outputs"] for identity, value in job["nodes"].items()}
    initial, trained, reloaded = [outputs[key]["metrics"] for key in ("initial-eval", "trained-eval", "reload-eval")]
    metrics, report = outputs["train"]["metrics"], outputs["inspect"]["report"]
    assert metrics["parameter_changed"]
    assert metrics["state_sha256_before"] != metrics["state_sha256_after"] == report["state_sha256"]
    assert trained["loss"] < initial["loss"] * 0.85, "Expected learning on the included corpus"
    assert trained["loss"] == reloaded["loss"], "Reconstructed graph must reproduce exact evaluation loss"
    assert all(math.isfinite(value) for value in [*metrics["loss_history"], *metrics["gradient_norms"]])
    assert len(metrics["loss_history"]) == metrics["steps"]
    compiled = outputs["model"]["model"]
    assert compiled["family"] == "tensor-graph"
    assert outputs["checkpoint"]["model"]["handle"] != outputs["reload"]["model"]["handle"]
    assert outputs["checkpoint"]["model"]["parameter_count"] == outputs["reload"]["model"]["parameter_count"]
    generated = outputs["generate"]["result"]
    assert len(generated["sequences"]) == 3
    assert all(2 < len(row) <= 10 and all(type(token) is int and 0 <= token < 32 for token in row)
               for row in generated["sequences"])
    pipeline = outputs["pipeline"]["result"]
    assert pipeline["task"] == "text-generation" and len(pipeline["outputs"]) == 3
    assert all(isinstance(row[0]["generated_text"], str) and row[0]["generated_text"] for row in pipeline["outputs"])
    return {"status": "passed", "version": spobs.__version__, "job_id": job["id"],
            "nodes_succeeded": len(job["nodes"]), "snapshot_sha256": job["snapshot_sha256"],
            "architecture": report["architecture"], "parameters": report["parameter_count"],
            "parameter_tensors": report["parameter_tensors"], "modules": report["module_count"],
            "steps": metrics["steps"], "initial_loss": initial["loss"], "trained_loss": trained["loss"],
            "reloaded_loss": reloaded["loss"], "state_sha256": report["state_sha256"],
            "generated": generated, "decoded": pipeline["outputs"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-python-export", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict[str, Any]] = {"expanded": {}, "custom-two-layer": {"layers": 2, "hidden": 24, "intermediate": 48, "custom": True, "steps": 25},
                "fused": {"fused": True}}
    results = {}
    for name, options in variants.items():
        target = directory / name
        target.mkdir(exist_ok=True)
        graph = llama_blocks_workflow(**options)
        (target / "workflow.json").write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        with closing(StudioWorkspace()) as workspace:
            before = workspace.snapshot()["runs"]
            graph_id = workspace.graph_action("graph.create", {"graph": graph})["graph_id"]
            validation = workspace.graph_action("graph.validate", {"graph_id": graph_id})
            assert validation["valid"], validation["errors"]
            job_id = workspace.graph_action("graph.execute", {"graph_id": graph_id})["job_id"]
            job = workspace.graphs.wait_job(job_id, 180)
            (target / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
            result = verify_job(job)
            assert workspace.snapshot()["runs"] == before, "Component training must preserve scalar experiments"
            artifact = job["nodes"]["checkpoint"]["outputs"]["artifact"]
            raw, _, _ = workspace.graphs.artifact(artifact["id"])
            assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                assert archive.testzip() is None
                assert set(archive.namelist()) == {"model.safetensors", "recipe.json", "manifest.json"}
                assert json.loads(archive.read("recipe.json"))["recipe"]["family"] == "tensor-graph"
            (target / "model.zip").write_bytes(raw)
            program = target / "workflow.py"
            program.write_bytes(python_project(workspace.export_project(), graph_id))
            (target / "project.json").write_text(json.dumps(workspace.export_project(), indent=2) + "\n", encoding="utf-8")
        if args.run_python_export:
            process = subprocess.run([sys.executable, "-I", str(program), "--timeout", "180", "--output",
                                      str(target / "python-job.json"), "--artifacts", str(target / "python-artifacts")],
                                     capture_output=True, timeout=210, check=False)
            (target / "python-stdout.json").write_bytes(process.stdout)
            (target / "python-stderr.txt").write_bytes(process.stderr)
            assert process.returncode == 0, process.stderr[-4000:]
            exported = json.loads((target / "python-job.json").read_bytes())
            assert json.loads(process.stdout) == exported
            replay = verify_job(exported)
            assert replay["state_sha256"] == result["state_sha256"], "Same saved graph must replay deterministically"
            result["python_replay"] = replay
        results[name] = result
        print(f"{name}: passed; {result['parameters']} parameters; loss {result['initial_loss']:.6f} -> {result['trained_loss']:.6f}", flush=True)
    (directory / "verification.json").write_text(json.dumps({"status": "passed", "version": spobs.__version__, "variants": results}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
