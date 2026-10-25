"""Assemble the qualified portable Llama example; never reads the user's project."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import zipfile
import zlib


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = Path(__file__).parent
    receipt_bytes = (evidence / "llama-verification.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["status"] == "passed" and receipt["version"] == "2.1.0"
    program = (evidence / "llama-workflow.py").read_bytes()
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(program).body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    project = zlib.decompress(base64.b64decode(constants["PROJECT_DATA"], validate=True))
    assert hashlib.sha256(project).hexdigest() == constants["PROJECT_SHA256"]
    project_value = json.loads(project)
    assert any(g["id"] == constants["GRAPH_ID"] for g in project_value["extensions"]["graphs"])
    model = (evidence / "llama-transformers-model.zip").read_bytes()
    assert hashlib.sha256(model).hexdigest() == receipt["artifacts"][0]["sha256"]
    readme = """# Runnable Llama architecture laboratory — Studio 2.1.0

This is an actual HF LlamaForCausalLM, initialized and trained locally: 78,144
parameters, 21 tensors, two decoder layers, four query/two KV heads, RoPE,
RMSNorm and SwiGLU. The checkpoint is the qualified 40-step AdamW result.
It is a small from-scratch example, not pretrained Meta weights.

1. Start the installed Parameter Studio 2.1.0 runtime.
2. To add the graph to your current workspace, use Node lab → Workflow → Import
   workflow JSON and select llama.workflow.json. Validate and Queue workflow.
3. To inspect the completed example with its registered weights and job history,
   save your current workspace first, then Open llama-studio.project.json.
   Open replaces the current workspace. Choose Transformers to inspect Llama.
4. To repeat the workflow from Python, run with the studio environment:
   python llama-workflow.py --timeout 180 --output job.json --artifacts artifacts
5. The safe-model ZIP contains model.safetensors, recipe.json and manifest.json.
   The project JSON retains the corresponding registered checkpoint for Load.

LLAMA_STUDIO.md explains editing, execution, inspection and limits. The verification
receipt and job JSON record actual learning and exact checkpoint/replay equality.
MANIFEST.json records every member's byte count and SHA-256.
"""
    # The compact lab is a flat project, while the full release keeps its repo
    # layout. Adapt only those guide links; retain the full guide in the release.
    guide_source = (root / "LLAMA_STUDIO.md").read_bytes()
    guide = guide_source.decode().replace("(examples/transformers/llama.workflow.json)", "(llama.workflow.json)")
    guide = guide.replace("[Transformers guide](TRANSFORMERS_STUDIO.md)", "Transformers guide included with the full studio")
    guide = guide.replace("[Node lab guide](NODE_LAB.md)", "Node lab guide included with the full studio")
    files = {
        "README.md": readme.encode(), "LLAMA_STUDIO.md": guide.encode(),
        "LLAMA_SOURCE_NOTES.md": (root / "LLAMA_SOURCE_NOTES.md").read_bytes(),
        "llama.workflow.json": (root / "examples/transformers/llama.workflow.json").read_bytes(),
        "llama-workflow.py": program, "llama-studio.project.json": project,
        "llama-trained-model.zip": model, "llama-verification.json": receipt_bytes,
        "llama.job.json": (evidence / "llama.job.json").read_bytes(),
    }
    for name in ("hfviewer-source-manifest.json", "hfviewer-review-summary.json",
                 "hfviewer-llama-bundle-receipt.json", "hfviewer-installed-llama-source.json"):
        files["evidence/llama_20261023/" + name] = (evidence / name).read_bytes()
    manifest = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()}
    output = root / "download" / "llama-studio-2.1.0.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("MANIFEST.json", json.dumps({"version": "2.1.0", "files": manifest}, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(manifest) | {"MANIFEST.json"}
        for name, expected in manifest.items():
            data = archive.read(name)
            assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"]
    result = {"status": "passed", "path": output.relative_to(root).as_posix(), "bytes": output.stat().st_size,
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "verified_files": len(files),
              "model_state_sha256": receipt["weight_sha256_after"], "project_sha256": constants["PROJECT_SHA256"],
              "source_guide_sha256": hashlib.sha256(guide_source).hexdigest(),
              "guide_changes": "Only compact-bundle relative links adapted; full release guide retained separately"}
    (root / "download" / "llama-bundle-2.1.0.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
