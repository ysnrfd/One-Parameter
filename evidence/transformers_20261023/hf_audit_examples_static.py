"""Check authored example wiring/code syntax without initializing Torch or HF."""

import ast
import hashlib
import json
import runpy
from pathlib import Path


root = Path(__file__).resolve().parents[2]
catalog = runpy.run_path(str(root / "single_parameter_observatory/studio_transformers.py"))["definitions"]()
definitions = {item["id"]: item for item in catalog}
program = runpy.run_path(str(root / "single_parameter_observatory/studio_transformers_worker.py"))["TensorProgram"]
plugin_helpers = runpy.run_path(str(root / "single_parameter_observatory/studio_plugins.py"))
plugin_records = []
for path in sorted((root / "examples/transformers").glob("*.plugin.json")):
    raw = path.read_bytes()
    manifest = plugin_helpers["validate_plugin"](json.loads(raw))
    definitions.update({item["id"]: item for item in plugin_helpers["node_definitions"](manifest)})
    plugin_records.append({"path": path.relative_to(root).as_posix(), "id": manifest["id"],
                           "sha256": hashlib.sha256(raw).hexdigest(), "status": "manifest_valid"})
records = []
for path in sorted((root / "examples/transformers").glob("*.workflow.json")):
    raw = path.read_bytes()
    graph = json.loads(raw)
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert len(nodes) == len(graph["nodes"]), path
    assert len({edge["id"] for edge in graph["edges"]}) == len(graph["edges"]), path
    incoming = set()
    successors = {identity: set() for identity in nodes}
    for edge in graph["edges"]:
        source, target = nodes[edge["source"]], nodes[edge["target"]]
        assert source is not target, (path, edge)
        assert edge["source_port"] in {port["name"] for port in definitions[source["type"]]["outputs"]}, (path, edge)
        assert edge["target_port"] in {port["name"] for port in definitions[target["type"]]["inputs"]}, (path, edge)
        input_key = (target["id"], edge["target_port"])
        assert input_key not in incoming, (path, edge)
        incoming.add(input_key)
        successors[source["id"]].add(target["id"])
    seen, visiting = set(), set()

    def visit(identity):
        assert identity not in visiting, (path, "cycle", identity)
        if identity in seen:
            return
        visiting.add(identity)
        for child in successors[identity]:
            visit(child)
        visiting.remove(identity)
        seen.add(identity)

    for identity, node in nodes.items():
        visit(identity)
        definition = definitions[node["type"]]
        allowed = {item["name"] for item in definition["inputs"] + definition["parameters"]}
        assert set(node["params"]) <= allowed, (path, node["id"], set(node["params"]) - allowed)
        for port in definition["inputs"]:
            assert not port.get("required", True) or "default" in port or port["name"] in node["params"] or (identity, port["name"]) in incoming, (path, node["id"], port)
        if node["type"] == "hf.custom":
            code = node["params"]["code"]
            ast.parse(code)
            program(code, set(node["params"]["layers"]), 1_000_000)
    records.append({"path": path.relative_to(root).as_posix(), "nodes": len(nodes), "edges": len(graph["edges"]),
                    "sha256": hashlib.sha256(raw).hexdigest(), "status": "static_valid"})
report = {"scope": "Actual current catalog node/port/required-input DAG wiring and restricted code syntax; no tensor/model execution",
          "workflows": len(records), "records": records, "plugins": plugin_records}
destination = Path(__file__).with_name("hf_audit_examples_static.json")
destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
