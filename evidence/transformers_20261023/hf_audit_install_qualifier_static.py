"""Replay qualifier assertions against frozen evidence without importing Torch.

This does not run the installed release, any model, or the HTTP qualification.
"""

import ast
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile


root = Path(__file__).resolve().parents[2]
source = root / "scripts/verify_studio_install.py"
module = ast.parse(source.read_text(encoding="utf-8"))
main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
names = {"verify_transformer_outputs", "verify_zip_bytes"}
functions = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(functions) == len(names)
namespace = {"hashlib": hashlib, "io": io, "json": json, "math": math, "zipfile": zipfile}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
evidence = json.loads((root / "evidence/transformers_20261023/examples_execution.json").read_bytes())
assert evidence["complete"] and evidence["passed"] == 13 and evidence["failed"] == 0
records = []
for example in evidence["examples"]:
    workflow = root / example["file"]
    raw = workflow.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == example["sha256"]
    graph = json.loads(raw)
    observations = namespace["verify_transformer_outputs"]({"graph": graph, "nodes": example["nodes"]})
    records.append({"file": example["file"], "source_sha256": example["sha256"],
                    "observed_training_assertions": len(observations), "status": "assertions_passed"})

header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, separators=(",", ":")).encode()
header += b" " * (-len(header) % 8)
weights = struct.pack("<Q", len(header)) + header + struct.pack("<ff", 1.0, 2.0)
recipe = json.dumps({"schema_version": "spobs-tensor-model/1", "recipe": {}}, sort_keys=True).encode()
manifest = json.dumps({"schema_version": "spobs-model-artifact/1", "weight_sha256": hashlib.sha256(weights).hexdigest(),
                       "recipe_sha256": hashlib.sha256(recipe).hexdigest(), "weight_bytes": len(weights),
                       "parameter_count": 2, "state_sha256": "a" * 64}).encode()
zip_checks = []
for kind, files in (("safe_model", {"model.safetensors": weights, "recipe.json": recipe, "manifest.json": manifest}),
                    ("scalar_evidence", {"trajectory.csv": b"step,value\n0,1\n", "checkpoint.json": b"{}", "dataset.json": b"{}"})):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    raw = stream.getvalue()
    descriptor = {"byte_length": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "mime": "application/zip"}
    members = namespace["verify_zip_bytes"](raw, descriptor)
    assert set(members) == set(files)
    try:
        namespace["verify_zip_bytes"](raw, {**descriptor, "sha256": "0" * 64})
    except AssertionError:
        pass
    else:
        raise AssertionError("Qualifier accepted a corrupted artifact identity")
    zip_checks.append({"kind": kind, "valid_container_accepted": True, "bad_hash_rejected": True})
report = {"scope": "Static assertion replay against thirteen prior actual source-runtime records plus authored ZIP containers; no installed runtime execution",
          "qualifier_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "workflow_records": records,
          "zip_assertion_checks": zip_checks, "status": "passed"}
destination = Path(__file__).with_name("hf_audit_install_qualifier_static.json")
destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "passed", "workflow_records": len(records), "zip_assertion_checks": len(zip_checks),
                  "scope": report["scope"]}, indent=2))
