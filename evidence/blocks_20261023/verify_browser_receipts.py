"""Read live saved snapshots; verify receipts from actual browser actions."""

import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def project(port):
    with urlopen(f"http://127.0.0.1:{port}/api/project", timeout=30) as response:
        return json.load(response)


def main():
    live, qa = project(8765), project(8767)
    original = json.loads((ROOT / "before_project.json").read_text(encoding="utf-8"))
    pre_upgrade = json.loads((ROOT / "pre_upgrade_project.json").read_text(encoding="utf-8"))
    (ROOT / "final-user-project.json").write_text(json.dumps(live, indent=2) + "\n", encoding="utf-8")
    checks = {"original_runs_unchanged": live["runs"] == original["runs"]}
    for field in ("graphs", "plugins", "jobs", "artifacts"):
        current = {item["id"]: item for item in live["extensions"][field]}
        for item in original["extensions"][field]:
            if field == "graphs":
                content_keys = set(item) - {"view", "revision", "updated_at"}
                checks[f"original_graph_content_{item['id']}"] = all(current[item["id"]].get(k) == item[k] for k in content_keys)
            else:
                checks[f"original_{field}_{item['id']}"] = current.get(item["id"]) == item
        for item in pre_upgrade["extensions"][field]:
            checks[f"pre_upgrade_{field}_{item['id']}"] = current.get(item["id"]) == item
    all_jobs = live["extensions"]["jobs"] + qa["extensions"]["jobs"]
    selected = {
        "custom": "1c3ac89641ec4d2ab6be6729d8a83b08",
        "fused": "8b35f19011c14f7c8c84db3dbd6f2f15",
        "port-built": "8cdbfca3e8734985a6dd982b1e8f186d",
        "expanded": "b581c9a648584f339b08e1c344ac592c",
        "plugin": "2bf3a9e002e44015b92efcc1cf1725af",
    }
    coverage = set()
    receipts = {}
    for name, identifier in selected.items():
        job = next(item for item in all_jobs if item["id"] == identifier)
        checks[f"{name}_job_succeeded"] = job["status"] == "succeeded"
        checks[f"{name}_every_node_succeeded"] = all(n["status"] == "succeeded" for n in job["nodes"].values())
        coverage.update(node["type"] for node in job["graph"]["nodes"])
        receipts[name] = {"job_id": identifier, "nodes": len(job["graph"]["nodes"]),
                          "edges": len(job["graph"]["edges"]), "status": job["status"]}
        (ROOT / "browser" / f"final-{name}.job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        if name == "port-built":
            norm = next(node for node in job["graph"]["nodes"] if node["type"] == "hf.op_rmsnorm")
            checks["inline_eps_committed_before_queue"] = norm["params"]["eps"] == 1e-5
    required = {"hf.tensor_input", "hf.op_embedding", "hf.op_linear", "hf.op_rmsnorm",
                "hf.op_positions", "hf.op_rope", "hf.op_attention", "hf.op_silu",
                "hf.op_multiply", "hf.op_add", "hf.op_swiglu", "hf.op_custom", "hf.graph_model"}
    checks["all_13_new_node_types_executed_in_browser"] = required <= coverage
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "browser_jobs": receipts, "new_component_types": sorted(required),
              "preserved_original_run_count": len(original["runs"]),
              "live_graph_count": len(live["extensions"]["graphs"]),
              "note": "Read-only snapshots. Original graph contents are exact; one legacy graph viewport was panned before upgrade. All pre-upgrade graphs, jobs and artifacts remain byte-structurally identical. Additional user edits and graphs are retained. Browser interactions and visual observations are documented separately."}
    (ROOT / "browser-verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
