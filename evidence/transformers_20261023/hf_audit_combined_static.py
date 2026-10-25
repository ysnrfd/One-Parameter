"""Validate the combined browser QA fixture without executing any graph node."""

import hashlib
import json
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from single_parameter_observatory.studio_graph import BUILTINS, GraphStudio
from single_parameter_observatory.studio_plugins import node_definitions, validate_plugin


path = root / "evidence/transformers_20261023/browser/all-nodes.workflow.json"
raw = path.read_bytes()
graph = GraphStudio._graph(json.loads(raw))
plugin_path = root / "examples/transformers/architecture.plugin.json"
plugin_raw = plugin_path.read_bytes()
plugin = validate_plugin(json.loads(plugin_raw))
definitions = {item["id"]: item for item in BUILTINS + node_definitions(plugin)}
validation = GraphStudio._validate(graph, definitions)
builtins = {item["id"] for item in BUILTINS}
included = {node["type"] for node in graph["nodes"]}
report = {"scope": "Actual current graph schema, typed ports, defaults, constraints and DAG validation; NO NODE EXECUTION",
          "workflow": path.relative_to(root).as_posix(), "workflow_sha256": hashlib.sha256(raw).hexdigest(),
          "workflow_bytes": len(raw), "node_count": len(graph["nodes"]), "edge_count": len(graph["edges"]),
          "builtin_types_expected": len(builtins), "builtin_types_covered": len(builtins & included),
          "builtin_types_missing": sorted(builtins - included), "all_types_included": sorted(included),
          "plugin_prerequisite": {"id": plugin["id"], "file": plugin_path.relative_to(root).as_posix(),
                                  "sha256": hashlib.sha256(plugin_raw).hexdigest()},
          "cached_model_prerequisite": {"repo_id": "hf-internal-testing/tiny-random-gpt2",
                                        "revision": "71034c5d8bde858ff824298bdedc65515b97d2b9"},
          "validation": validation, "execution_status": "not_executed_by_static_audit",
          "coverage_limit": "One queued graph can exercise every included node type; it does not cover every parameter combination, optional architecture, failure mode, pointer gesture or UI layout."}
report_path = path.with_name("all-nodes.static-validation.json")
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key not in {"validation", "all_types_included"}}, indent=2))
print(json.dumps({"validation_valid": validation["valid"], "errors": validation["errors"]}, indent=2))
if not validation["valid"] or builtins - included:
    raise SystemExit(1)
