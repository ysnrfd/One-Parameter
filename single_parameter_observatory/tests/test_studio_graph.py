"""Real graph execution, immutable queue snapshots, extensions and safe reopening."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import threading
import time
import unittest
import zipfile
from unittest.mock import patch

from single_parameter_observatory.studio_graph import GraphConflictError, GraphStudio, _edge, _node
from single_parameter_observatory.studio_plugins import example_plugin
from single_parameter_observatory.studio_session import StudioWorkspace


class TestGraphStudio(unittest.TestCase):
    def setUp(self):
        self.workspace = StudioWorkspace()
        self.addCleanup(self.workspace.close)
        self.graphs = self.workspace.graphs if hasattr(self.workspace, "graphs") else GraphStudio(self.workspace)
        self.addCleanup(self.graphs.close)

    def graph(self, nodes, edges=None, **extra):
        result = self.graphs.dispatch("graph.create", {"graph": {"name": "Test workflow", "nodes": nodes, "edges": edges or [], **extra}})
        return result["graph_id"]

    def execute(self, graph_id, **extra):
        result = self.graphs.dispatch("graph.execute", {"graph_id": graph_id, **extra})
        return self.graphs.wait_job(result["job_id"], 60)

    def saved_graph(self, graph_id):
        return next(item for item in self.graphs.snapshot()["graphs"] if item["id"] == graph_id)

    def update(self, graph):
        return self.graphs.dispatch("graph.update", {"graph": graph, "expected_revision": graph["revision"]})

    def test_starter_is_an_idle_real_training_template(self):
        snapshot = self.graphs.snapshot()
        self.assertEqual(len(snapshot["graphs"]), 1)
        self.assertEqual(snapshot["jobs"], [])
        self.assertEqual(self.workspace.snapshot()["runs"][0]["step"], 0)
        self.assertTrue(self.graphs.validate(snapshot["graphs"][0])["valid"])
        self.assertEqual(sum(item["id"].startswith("core.") for item in self.graphs.catalog()["types"]), 21)

    def test_all_value_and_transform_nodes_execute_exact_outputs(self):
        nodes = [_node("a", "core.number", 0, 0, value=5), _node("b", "core.number", 0, 0, value=7),
                 _node("math", "core.math", 0, 0, operation="multiply"), _node("preview", "core.preview", 0, 0),
                 _node("text", "core.text", 0, 0, value="<script>literal</script>"),
                 _node("boolean", "core.boolean", 0, 0, value=False), _node("json", "core.json", 0, 0, value=None),
                 _node("array", "core.array", 0, 0, items=[1, "two", False]),
                 _node("object", "core.object", 0, 0, value={"key": 17}),
                 _node("get", "core.get", 0, 0, key="key")]
        edges = [_edge("a", "value", "math", "a"), _edge("b", "value", "math", "b"),
                 _edge("math", "value", "preview", "value"), _edge("object", "value", "get", "object")]
        job = self.execute(self.graph(nodes, edges))
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertEqual(job["nodes"]["preview"]["outputs"], {"value": 35})
        self.assertEqual(job["nodes"]["get"]["outputs"], {"value": 17})
        self.assertEqual(job["nodes"]["text"]["outputs"]["value"], "<script>literal</script>")
        self.assertEqual(job["nodes"]["boolean"]["outputs"]["value"], False)
        self.assertIsNone(job["nodes"]["json"]["outputs"]["value"])
        self.assertEqual(job["nodes"]["array"]["outputs"]["value"], [1, "two", False])
        self.assertIsNotNone(job["nodes"]["math"]["duration_ms"])

    def test_every_math_operation_and_missing_get_default(self):
        expected = {"add": 10, "subtract": 6, "multiply": 16, "divide": 4, "power": 64, "min": 2, "max": 8, "modulo": 0}
        nodes = [_node(key, "core.math", 0, 0, a=8, b=2, operation=key) for key in expected]
        nodes.append(_node("get", "core.get", 0, 0, object=[1, 2], key=99, default="absent"))
        job = self.execute(self.graph(nodes))
        self.assertEqual(job["status"], "succeeded", job["error"])
        for key, value in expected.items():
            self.assertEqual(job["nodes"][key]["outputs"]["value"], value)
        self.assertEqual(job["nodes"]["get"]["outputs"]["value"], "absent")

    def test_real_training_pipeline_checkpoint_hash_chain_and_history_step_zero(self):
        graph_id = self.graphs.dispatch("graph.create", {"template_id": "training"})["graph_id"]
        graph = self.saved_graph(graph_id)
        graph["nodes"].append(_node("history", "core.inspect", 0, 0, step=0))
        graph["nodes"].append(_node("verify", "core.verify", 0, 0))
        graph["edges"].extend([_edge("checkpoint", "run", "history", "run"), _edge("checkpoint", "run", "verify", "run")])
        self.update(graph)
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "succeeded", job["error"])
        run_id = job["nodes"]["train"]["outputs"]["run"]
        self.assertEqual(job["nodes"]["inspect"]["outputs"]["record"]["step"], 10)
        self.assertEqual(job["nodes"]["history"]["outputs"]["record"], self.workspace.record(run_id, 0))
        self.assertEqual(job["nodes"]["verify"]["outputs"]["integrity"]["status"], "verified")
        self.assertEqual(len(self.workspace._runs[run_id].engine.recorder.records), 10)
        self.assertEqual(len(self.workspace._runs[run_id].checkpoints), 1)
        self.assertEqual(job["nodes"]["train"]["inputs"], {"run": run_id})

    def test_custom_dataset_pipeline_uses_actual_arrays_without_extra_base_run(self):
        before = len(self.workspace.snapshot()["runs"])
        graph_id = self.graphs.dispatch("graph.create", {"template_id": "custom-data"})["graph_id"]
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "succeeded", job["error"])
        run_id = job["nodes"]["train"]["outputs"]["run"]
        self.assertEqual(len(self.workspace.snapshot()["runs"]), before + 1)
        data = self.workspace.dataset(run_id)
        self.assertEqual(data["inputs"], [-1, 0, 1])
        self.assertEqual(data["targets"], [-2, 0, 2])
        self.assertEqual(len(self.workspace._runs[run_id].engine.recorder.records), 20)

    def test_branch_and_manual_intervention_retain_source_state(self):
        run_id = self.workspace.snapshot()["runs"][0]["id"]
        self.workspace.dispatch("run.step", {"run_id": run_id, "count": 3})
        source_value = self.workspace._runs[run_id].engine.parameter.item()
        nodes = [_node("branch", "core.branch", 0, 0, run=run_id, name="Graph branch"),
                 _node("intervene", "core.intervene", 0, 0, changes={"parameter": 1.25, "lr": 0.005}),
                 _node("train", "core.train", 0, 0, steps=2)]
        job = self.execute(self.graph(nodes, [_edge("branch", "run", "intervene", "run"), _edge("intervene", "run", "train", "run")]))
        self.assertEqual(job["status"], "succeeded", job["error"])
        branch_id = job["nodes"]["branch"]["outputs"]["run"]
        self.assertNotEqual(branch_id, run_id)
        self.assertEqual(self.workspace._runs[run_id].engine.parameter.item(), source_value)
        self.assertEqual(len(self.workspace._runs[run_id].engine.recorder.records), 3)
        self.assertEqual(len(self.workspace._runs[branch_id].engine.recorder.records), 5)
        self.assertEqual(self.workspace._runs[branch_id].interventions[0]["parameter_after"], 1.25)

    def test_incomplete_drafts_save_and_reopen_but_cannot_execute(self):
        graph_id = self.graph([_node("preview", "core.preview", 0, 0)])
        result = self.graphs.dispatch("graph.validate", {"graph_id": graph_id})
        self.assertFalse(result["valid"])
        self.assertIn("Missing required", result["errors"][0]["message"])
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.execute(graph_id)
        state = self.graphs.export_state()
        self.graphs.import_state(json.loads(json.dumps(state)))
        self.assertEqual(self.saved_graph(graph_id)["nodes"], state["graphs"][-1]["nodes"])
        graph = self.saved_graph(graph_id)
        graph["nodes"][0]["params"]["value"] = {"now": "connected literal"}
        self.update(graph)
        self.assertEqual(self.execute(graph_id)["status"], "succeeded")

    def test_duplicate_update_select_delete_and_stale_compare_and_swap(self):
        first = self.graph([_node("value", "core.number", 0, 0)])
        stale = self.saved_graph(first)
        graph = copy.deepcopy(stale)
        graph["nodes"][0]["params"]["value"] = 9
        self.update(graph)
        with self.assertRaises(GraphConflictError):
            self.update(stale)
        self.assertEqual(self.saved_graph(first)["nodes"][0]["params"]["value"], 9)
        with self.assertRaises(GraphConflictError):
            self.graphs.dispatch("graph.execute", {"graph_id": first, "expected_revision": stale["revision"]})
        second = self.graphs.dispatch("graph.duplicate", {"graph_id": first, "name": "Copy"})["graph_id"]
        self.assertEqual(self.saved_graph(second)["nodes"], self.saved_graph(first)["nodes"])
        self.graphs.dispatch("graph.select", {"graph_id": first})
        self.assertEqual(self.graphs.snapshot()["active_graph_id"], first)
        self.graphs.dispatch("graph.delete", {"graph_id": first})
        self.assertNotIn(first, {item["id"] for item in self.graphs.snapshot()["graphs"]})

    def test_structurally_invalid_update_is_atomic(self):
        graph_id = self.graph([_node("n", "core.number", 0, 0)])
        original = self.saved_graph(graph_id)
        variants = []
        duplicate = copy.deepcopy(original)
        duplicate["nodes"].append(copy.deepcopy(duplicate["nodes"][0]))
        variants.append(duplicate)
        dangling = copy.deepcopy(original)
        dangling["edges"].append(_edge("n", "value", "absent", "value"))
        variants.append(dangling)
        nonfinite = copy.deepcopy(original)
        nonfinite["nodes"][0]["position"]["x"] = float("nan")
        variants.append(nonfinite)
        malformed = copy.deepcopy(original)
        malformed["nodes"][0]["params"] = []
        variants.append(malformed)
        for candidate in variants:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    self.update(candidate)
                self.assertEqual(self.saved_graph(graph_id), original)

    def test_reserved_dot_path_identities_reject_import_atomically(self):
        graph_id = self.graph([_node("node.v1:ok", "core.number", 0, 0)])
        baseline = self.graphs.export_state()
        graph = next(item for item in baseline["graphs"] if item["id"] == graph_id)
        graph["id"] = "workflow.v1:ok"
        baseline["active_graph_id"] = graph["id"]
        self.graphs.import_state(baseline)
        original = self.graphs.export_state()
        for reserved in (".", ".."):
            for field in ("graph", "node"):
                candidate = copy.deepcopy(original)
                changed = next(item for item in candidate["graphs"] if item["id"] == "workflow.v1:ok")
                if field == "graph":
                    changed["id"] = reserved
                    candidate["active_graph_id"] = reserved
                else:
                    changed["nodes"][0]["id"] = reserved
                with self.subTest(identity=reserved, field=field), self.assertRaisesRegex(ValueError, "reserved path segment"):
                    self.graphs.import_state(candidate)
                self.assertEqual(self.graphs.export_state(), original)

    def test_full_typed_validation_detects_cycles_missing_ports_types_and_options(self):
        nodes = [_node("a", "core.math", 0, 0), _node("b", "core.math", 0, 0)]
        graph_id = self.graph(nodes, [_edge("a", "value", "b", "a"), _edge("b", "value", "a", "a")])
        validation = self.graphs.dispatch("graph.validate", {"graph_id": graph_id})
        self.assertTrue(any("cycle" in error["message"] for error in validation["errors"]))
        with self.assertRaises(ValueError):
            self.execute(graph_id)
        bad_type = self.graph([_node("text", "core.text", 0, 0), _node("math", "core.math", 0, 0)], [_edge("text", "value", "math", "a")])
        self.assertFalse(self.graphs.dispatch("graph.validate", {"graph_id": bad_type})["valid"])
        unknown_port = self.saved_graph(bad_type)
        unknown_port["edges"][0]["source_port"] = "missing"
        self.update(unknown_port)
        self.assertIn("unknown", self.graphs.dispatch("graph.validate", {"graph_id": bad_type})["errors"][0]["message"].lower())
        option = self.graph([_node("math", "core.math", 0, 0, operation="execute-python")])
        self.assertFalse(self.graphs.dispatch("graph.validate", {"graph_id": option})["valid"])

    def test_duplicate_input_edge_and_edge_id_are_rejected(self):
        nodes = [_node("a", "core.number", 0, 0), _node("b", "core.number", 0, 0), _node("math", "core.math", 0, 0)]
        with self.assertRaisesRegex(ValueError, "one incoming"):
            self.graph(nodes, [_edge("a", "value", "math", "a"), _edge("b", "value", "math", "a")])
        edge = _edge("a", "value", "math", "a")
        with self.assertRaisesRegex(ValueError, "duplicate edge"):
            self.graph(nodes, [edge, copy.deepcopy(edge)])

    def test_runtime_any_output_is_checked_at_typed_destination(self):
        graph_id = self.graph([_node("json", "core.json", 0, 0, value="not a number"), _node("math", "core.math", 0, 0)],
                              [_edge("json", "value", "math", "a")])
        self.assertTrue(self.graphs.dispatch("graph.validate", {"graph_id": graph_id})["valid"])
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["nodes"]["math"]["status"], "failed")
        self.assertIn("requires type number", job["error"])

    def test_failed_node_retains_previous_real_host_effect_and_skips_dependents(self):
        nodes = [_node("run", "core.experiment", 0, 0), _node("math", "core.math", 0, 0, a=1, b=0, operation="divide"),
                 _node("preview", "core.preview", 0, 0)]
        before = len(self.workspace.snapshot()["runs"])
        job = self.execute(self.graph(nodes, [_edge("math", "value", "preview", "value")]))
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["nodes"]["run"]["status"], "succeeded")
        self.assertEqual(job["nodes"]["preview"]["status"], "skipped")
        self.assertEqual(len(self.workspace.snapshot()["runs"]), before + 1)
        # The same worker remains usable after an arithmetic error.
        good = self.graph([_node("n", "core.number", 0, 0)])
        self.assertEqual(self.execute(good)["status"], "succeeded")

    def test_idempotent_execute_does_not_duplicate_host_effects(self):
        graph_id = self.graph([_node("run", "core.experiment", 0, 0)])
        before = len(self.workspace.snapshot()["runs"])
        first = self.graphs.dispatch("graph.execute", {"graph_id": graph_id, "idempotency_key": "one-request"})
        second = self.graphs.dispatch("graph.execute", {"graph_id": graph_id, "idempotency_key": "one-request"})
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertTrue(second["replayed"])
        self.assertEqual(self.graphs.wait_job(first["job_id"])["status"], "succeeded")
        self.assertEqual(len(self.workspace.snapshot()["runs"]), before + 1)
        graph = self.saved_graph(graph_id)
        graph["name"] = "Changed revision"
        self.update(graph)
        with self.assertRaises(GraphConflictError):
            self.graphs.dispatch("graph.execute", {"graph_id": graph_id, "idempotency_key": "one-request"})

    def test_script_node_runs_real_v8_and_records_logs(self):
        graph_id = self.graph([_node("script", "core.script", 0, 0, inputs={"value": 21},
                                    code="console.log('received', inputs.value); return {value: inputs.value * 2, extra: [true, null]};")])
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "succeeded", job["error"])
        record = job["nodes"]["script"]
        self.assertEqual(record["outputs"], {"value": 42, "outputs": {"value": 42, "extra": [True, None]}})
        self.assertTrue(any("received" in item for item in record["logs"]))

    def test_plugin_lifecycle_and_dataset_transform_are_real(self):
        plugin = example_plugin()
        result = self.graphs.dispatch("plugin.install", {"manifest": plugin})
        self.assertEqual(result["plugin_id"], plugin["id"])
        self.assertIn("local.examples/scale", {item["id"] for item in self.graphs.catalog()["types"]})
        nodes = [_node("data", "core.dataset", 0, 0), _node("scale", "local.examples/scale_targets", 0, 0, factor=3),
                 _node("run", "core.experiment", 0, 0), _node("train", "core.train", 0, 0, steps=3)]
        edges = [_edge("data", "dataset", "scale", "data"), _edge("scale", "data", "run", "dataset"), _edge("run", "run", "train", "run")]
        graph_id = self.graph(nodes, edges)
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "succeeded", job["error"])
        run_id = job["nodes"]["run"]["outputs"]["run"]
        self.assertEqual(self.workspace.dataset(run_id)["targets"], [-6, 0, 6])
        self.assertEqual(self.workspace.dataset(run_id)["provenance"]["user_provenance"]["transform"], "scale_targets")
        self.assertEqual(job["plugin_versions"][plugin["id"]]["version"], "1.0.0")
        self.graphs.dispatch("plugin.disable", {"plugin_id": plugin["id"]})
        self.assertFalse(self.graphs.dispatch("graph.validate", {"graph_id": graph_id})["valid"])
        self.graphs.dispatch("plugin.enable", {"plugin_id": plugin["id"]})
        self.assertTrue(self.graphs.dispatch("graph.validate", {"graph_id": graph_id})["valid"])
        self.graphs.dispatch("plugin.remove", {"plugin_id": plugin["id"]})
        self.assertEqual(self.saved_graph(graph_id)["nodes"], nodes)
        self.graphs.import_state(self.graphs.export_state())
        self.assertFalse(self.graphs.dispatch("graph.validate", {"graph_id": graph_id})["valid"])

    def test_bad_manifest_and_same_version_update_are_atomic(self):
        plugin = example_plugin()
        self.graphs.dispatch("plugin.install", {"manifest": plugin})
        with self.assertRaises(GraphConflictError):
            self.graphs.dispatch("plugin.install", {"manifest": plugin})
        changed = copy.deepcopy(plugin)
        changed["nodes"][0]["code"] = "return {value: 999};"
        with self.assertRaises(GraphConflictError):
            self.graphs.dispatch("plugin.update", {"plugin_id": plugin["id"], "manifest": changed})
        self.assertEqual(self.graphs.snapshot()["plugins"][0]["manifest"], plugin)
        changed["version"] = "1.1.0"
        self.graphs.dispatch("plugin.update", {"plugin_id": plugin["id"], "manifest": changed})
        self.assertEqual(self.graphs.snapshot()["plugins"][0]["manifest"]["version"], "1.1.0")
        bad = copy.deepcopy(changed)
        bad["nodes"][0]["code"] = 4
        with self.assertRaises(ValueError):
            self.graphs.dispatch("plugin.update", {"plugin_id": plugin["id"], "manifest": bad})
        self.assertEqual(self.graphs.snapshot()["plugins"][0]["manifest"]["version"], "1.1.0")

    def test_plugin_output_types_and_declared_ports_are_enforced(self):
        plugin = example_plugin()
        plugin["nodes"][0]["code"] = "return {value: 'bad'};"
        self.graphs.dispatch("plugin.install", {"manifest": plugin})
        graph_id = self.graph([_node("bad", "local.examples/scale", 0, 0, value=2)])
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("requires type number", job["error"])
        plugin["version"] = "1.1.0"
        plugin["nodes"][0]["code"] = "return {value: 2, undeclared: 1};"
        self.graphs.dispatch("plugin.update", {"plugin_id": plugin["id"], "manifest": plugin})
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("declared output", job["error"])

    def test_queue_snapshots_graph_and_plugin_before_later_edits(self):
        plugin = example_plugin()
        self.graphs.dispatch("plugin.install", {"manifest": plugin})
        gate = threading.Event()
        entered = threading.Event()
        original = self.graphs._execute_node

        def blocked(kind, inputs, params, definition, cancel):
            if kind == "core.number" and not entered.is_set():
                entered.set()
                gate.wait(5)
            return original(kind, inputs, params, definition, cancel)

        with patch.object(self.graphs, "_execute_node", side_effect=blocked):
            blocker_id = self.graph([_node("gate", "core.number", 0, 0)])
            blocker = self.graphs.dispatch("graph.execute", {"graph_id": blocker_id})["job_id"]
            self.assertTrue(entered.wait(5))
            graph_id = self.graph([_node("scale", "local.examples/scale", 0, 0, value=3, factor=2)])
            target = self.graphs.dispatch("graph.execute", {"graph_id": graph_id})["job_id"]
            edited = self.saved_graph(graph_id)
            edited["nodes"][0]["params"]["factor"] = 100
            self.update(edited)
            plugin["version"] = "2.0.0"
            plugin["nodes"][0]["code"] = "return {value: 999};"
            self.graphs.dispatch("plugin.update", {"plugin_id": plugin["id"], "manifest": plugin})
            self.graphs.dispatch("plugin.disable", {"plugin_id": plugin["id"]})
            gate.set()
            self.assertEqual(self.graphs.wait_job(blocker, timeout=60)["status"], "succeeded")
            job = self.graphs.wait_job(target, timeout=60)
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertEqual(job["nodes"]["scale"]["outputs"]["value"], 6)
        self.assertEqual(job["plugin_versions"][plugin["id"]]["version"], "1.0.0")

    def test_cancel_queued_job_never_performs_host_effect(self):
        gate = threading.Event()
        entered = threading.Event()
        original = self.graphs._execute_node

        def blocked(kind, inputs, params, definition, cancel):
            entered.set()
            gate.wait(5)
            return original(kind, inputs, params, definition, cancel)

        before = len(self.workspace.snapshot()["runs"])
        with patch.object(self.graphs, "_execute_node", side_effect=blocked):
            blocker = self.graphs.dispatch("graph.execute", {"graph_id": self.graph([_node("n", "core.number", 0, 0)])})["job_id"]
            self.assertTrue(entered.wait(5))
            queued = self.graphs.dispatch("graph.execute", {"graph_id": self.graph([_node("run", "core.experiment", 0, 0)])})["job_id"]
            self.graphs.dispatch("job.cancel", {"job_id": queued})
            self.assertEqual(self.graphs.wait_job(queued)["status"], "cancelled")
            gate.set()
            self.graphs.wait_job(blocker, timeout=60)
        self.assertEqual(len(self.workspace.snapshot()["runs"]), before)

    def test_cancel_running_training_stops_between_steps_and_refuses_import_while_busy(self):
        run_id = self.workspace.dispatch("run.create", {"config": {"steps": 1000, "dataset": {"samples": 8, "batch_size": 4}}})["run_id"]
        graph_id = self.graph([_node("train", "core.train", 0, 0, run=run_id, steps=1000)])
        stepped = threading.Event()
        release = threading.Event()
        original = self.workspace.dispatch

        def slowed(action, payload):
            result = original(action, payload)
            if action == "run.step":
                stepped.set()
                release.wait(5)
            return result

        with patch.object(self.workspace, "dispatch", side_effect=slowed):
            job_id = self.graphs.dispatch("graph.execute", {"graph_id": graph_id})["job_id"]
            self.assertTrue(stepped.wait(5))
            with self.assertRaises(GraphConflictError):
                self.graphs.import_state(self.graphs.export_state())
            self.graphs.dispatch("job.cancel", {"job_id": job_id})
            release.set()
            job = self.graphs.wait_job(job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(len(self.workspace._runs[run_id].engine.recorder.records), 1)
        self.assertNotEqual(job["nodes"]["train"]["status"], "succeeded")

    def test_cancel_actual_infinite_script_terminates_worker_and_queue_recovers(self):
        graph_id = self.graph([_node("script", "core.script", 0, 0, code="while (true) {}", timeout_ms=10000)])
        job_id = self.graphs.dispatch("graph.execute", {"graph_id": graph_id})["job_id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = next(item for item in self.graphs.snapshot()["jobs"] if item["id"] == job_id)
            if job["nodes"]["script"]["status"] == "running":
                break
            time.sleep(0.01)
        started = time.monotonic()
        self.graphs.dispatch("job.cancel", {"job_id": job_id})
        job = self.graphs.wait_job(job_id, 5)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(self.execute(self.graph([_node("n", "core.number", 0, 0)]))["status"], "succeeded")

    def test_project_reopens_terminal_jobs_and_never_resumes_queued_work(self):
        graph_id = self.graph([_node("n", "core.number", 0, 0, value=4)])
        job = self.execute(graph_id, idempotency_key="saved-request")
        state = self.graphs.export_state()
        reopened = GraphStudio(self.workspace)
        self.addCleanup(reopened.close)
        reopened.import_state(json.loads(json.dumps(state, allow_nan=False)))
        self.assertEqual(reopened.snapshot()["jobs"][0]["nodes"], job["nodes"])
        replay = reopened.dispatch("graph.execute", {"graph_id": graph_id, "idempotency_key": "saved-request"})
        self.assertEqual(replay["job_id"], job["id"])
        state["jobs"][0]["status"] = "queued"
        state["jobs"][0]["nodes"]["n"]["status"] = "pending"
        reopened.import_state(state)
        self.assertEqual(reopened.snapshot()["jobs"][0]["status"], "interrupted")
        self.assertIsNone(reopened._worker)

    def test_invalid_project_import_cannot_replace_existing_graphs_plugins_or_jobs(self):
        graph_id = self.graph([_node("n", "core.number", 0, 0)])
        self.execute(graph_id)
        self.graphs.dispatch("plugin.install", {"manifest": example_plugin()})
        before = self.graphs.export_state()
        bad = copy.deepcopy(before)
        bad["plugins"][0]["manifest"]["nodes"][0]["outputs"][0]["type"] = "python-module"
        with self.assertRaises(ValueError):
            self.graphs.import_state(bad)
        self.assertEqual(self.graphs.export_state(), before)

        bad = copy.deepcopy(before)
        bad["jobs"][0]["order"] = ["missing"]
        with self.assertRaises(ValueError):
            self.graphs.import_state(bad)
        self.assertEqual(self.graphs.export_state(), before)

    def test_snapshot_hash_detects_changed_pinned_code_and_incomplete_template_import_is_structural(self):
        self.graphs.dispatch("plugin.install", {"manifest": example_plugin()})
        graph_id = self.graph([_node("scale", "local.examples/scale", 0, 0, value=2)])
        self.execute(graph_id)
        before = self.graphs.export_state()
        bad = copy.deepcopy(before)
        bad["jobs"][0]["definitions"]["local.examples/scale"]["code"] = "return {value: 999};"
        with self.assertRaisesRegex(ValueError, "snapshot hash"):
            self.graphs.import_state(bad)
        self.assertEqual(self.graphs.export_state(), before)
        bad = copy.deepcopy(before)
        bad["plugins"][0]["manifest"]["templates"][0]["graph"]["nodes"].append(
            copy.deepcopy(bad["plugins"][0]["manifest"]["templates"][0]["graph"]["nodes"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate node"):
            self.graphs.import_state(bad)
        self.assertEqual(self.graphs.export_state(), before)

    def test_plugin_templates_are_available_and_execute_after_install(self):
        plugin = example_plugin()
        self.graphs.dispatch("plugin.install", {"manifest": plugin})
        template = next(item for item in self.graphs.catalog()["templates"] if item["id"].startswith(plugin["id"] + "/"))
        graph_id = self.graphs.dispatch("graph.create", {"template_id": template["id"]})["graph_id"]
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertEqual(len(job["nodes"]), 3)
        self.graphs.dispatch("plugin.disable", {"plugin_id": plugin["id"]})
        self.assertNotIn(template["id"], {item["id"] for item in self.graphs.catalog()["templates"]})

    def test_parameter_defaults_do_not_duplicate_literal_inputs_into_script_params(self):
        plugin = example_plugin()
        plugin["nodes"][0]["code"] = "return {value: Object.keys(params).length === 1 && params.factor === 2 ? inputs.value * 2 : -1};"
        self.graphs.dispatch("plugin.install", {"manifest": plugin})
        graph_id = self.graph([_node("scale", "local.examples/scale", 0, 0, value=3)])
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertEqual(job["nodes"]["scale"]["outputs"]["value"], 6)
        self.assertEqual(job["nodes"]["scale"]["parameters"], {"factor": 2})

    def test_resource_budget_fails_explicitly_before_downstream_host_effect(self):
        run_id = self.workspace.snapshot()["runs"][0]["id"]
        parameter = self.workspace._runs[run_id].engine.parameter.item()
        nodes = [_node("value", "core.object", 0, 0, value={"parameter_value": 1.25}),
                 _node("intervene", "core.intervene", 0, 0, run=run_id)]
        graph_id = self.graph(nodes, [_edge("value", "value", "intervene", "changes")])
        before = len(self.workspace.snapshot()["runs"])
        original = self.graphs._execute_node
        with patch("single_parameter_observatory.studio_graph.MAX_JOB_BYTES", 8_000_000):
            def lower_budget(kind, inputs, params, definition, cancel):
                import single_parameter_observatory.studio_graph as graph_module
                graph_module.MAX_JOB_BYTES = self.graphs._size(self.graphs.snapshot()["jobs"][-1]) + 5
                return original(kind, inputs, params, definition, cancel)
            with patch.object(self.graphs, "_execute_node", side_effect=lower_budget):
                job = self.execute(graph_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("byte", job["error"])
        self.assertEqual(job["nodes"]["intervene"]["status"], "skipped")
        self.assertEqual(self.workspace._runs[run_id].engine.parameter.item(), parameter)
        self.assertEqual(len(self.workspace.snapshot()["runs"]), before)

    def test_nonfinite_math_invalid_dataset_and_typed_config_are_explicit_failures(self):
        graph_id = self.graph([_node("math", "core.math", 0, 0, a=1e308, b=1e308, operation="multiply")])
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("finite", job["error"])
        graph_id = self.graph([_node("data", "core.dataset", 0, 0, inputs=[1], targets=[1, 2], batch_size=1)])
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("equal-length", job["error"])
        graph_id = self.graph([_node("run", "core.experiment", 0, 0, config={"unknown_field": 1})])
        self.assertFalse(self.graphs.dispatch("graph.validate", {"graph_id": graph_id})["valid"])

    def test_snapshot_and_exports_are_deep_copies(self):
        graph_id = self.graph([_node("n", "core.number", 0, 0, value=2)])
        self.execute(graph_id)
        snapshot = self.graphs.snapshot()
        snapshot["graphs"][-1]["nodes"][0]["params"]["value"] = 999
        snapshot["jobs"][0]["nodes"]["n"]["outputs"]["value"] = 999
        self.assertEqual(self.saved_graph(graph_id)["nodes"][0]["params"]["value"], 2)
        self.assertEqual(self.graphs.snapshot()["jobs"][0]["nodes"]["n"]["outputs"]["value"], 2)

    def test_old_project_default_and_closed_studio(self):
        state = GraphStudio.validate_state(None)
        self.assertEqual(state["schema_version"], "spobs-graphs/1")
        self.assertEqual(len(state["graphs"]), 1)
        self.graphs.close()
        with self.assertRaises(GraphConflictError):
            self.graphs.dispatch("graph.create", {})

    def test_job_clear_keeps_active_jobs_and_removes_replay_keys(self):
        graph_id = self.graph([_node("n", "core.number", 0, 0)])
        self.execute(graph_id, idempotency_key="clear-me")
        self.graphs.dispatch("job.clear", {})
        self.assertEqual(self.graphs.snapshot()["jobs"], [])
        self.assertFalse(self.graphs.dispatch("graph.execute", {"graph_id": graph_id, "idempotency_key": "clear-me"})["replayed"])

    def test_control_node_play_pause_reset_restart_are_real_and_keep_dataset(self):
        run_id = self.workspace.dispatch("run.create", {"config": {"steps": 100, "dataset": {"samples": 3, "batch_size": 3}},
                                                        "dataset": {"inputs": [-1, 0, 1], "targets": [-3, 0, 3]}})["run_id"]
        self.workspace.dispatch("run.step", {"run_id": run_id, "count": 2})
        initial = self.workspace._runs[run_id].engine.config.initial_value
        for operation in ("play", "pause", "reset", "restart"):
            job = self.execute(self.graph([_node("control", "core.control", 0, 0, run=run_id, operation=operation)]))
            self.assertEqual(job["status"], "succeeded", job["error"])
            if operation == "pause":
                self.assertEqual(self.workspace._runs[run_id].controller.state, "paused")
            if operation in ("reset", "restart"):
                self.assertEqual(self.workspace._runs[run_id].engine.parameter.item(), initial)
            if operation == "restart":
                self.assertEqual(len(self.workspace._runs[run_id].engine.recorder.records), 0)
            self.assertEqual(self.workspace.dataset(run_id)["targets"], [-3, 0, 3])
        bad = self.graph([_node("control", "core.control", 0, 0, run=run_id, operation="delete-all-files")])
        self.assertFalse(self.graphs.dispatch("graph.validate", {"graph_id": bad})["valid"])

    def test_export_node_retains_exact_zip_across_later_training_and_project_reopen(self):
        run_id = self.workspace.dispatch("run.create", {"config": {"steps": 10, "dataset": {"samples": 4, "batch_size": 2}}})["run_id"]
        self.workspace.dispatch("run.step", {"run_id": run_id, "count": 2})
        graph_id = self.graph([_node("export", "core.export", 0, 0, run=run_id)])
        job = self.execute(graph_id)
        self.assertEqual(job["status"], "succeeded", job["error"])
        descriptor = job["nodes"]["export"]["outputs"]["artifact"]
        data, filename, mime = self.graphs.artifact(descriptor["id"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), descriptor["sha256"])
        self.assertEqual(len(data), descriptor["byte_length"])
        self.assertEqual(filename, descriptor["filename"])
        self.assertEqual(mime, "application/zip")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("checkpoint.json", archive.namelist())
            self.assertIn("trajectory.csv", archive.namelist())
        self.workspace.dispatch("run.step", {"run_id": run_id, "count": 2})
        self.assertEqual(self.graphs.artifact(descriptor["id"])[0], data)
        self.assertNotIn("data_base64", self.graphs.snapshot()["artifacts"][0])
        saved = json.loads(json.dumps(self.graphs.export_state()))
        self.graphs.import_state(saved)
        self.assertEqual(self.graphs.artifact(descriptor["id"])[0], data)
        bad = copy.deepcopy(saved)
        bad["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.graphs.import_state(bad)
        self.assertEqual(self.graphs.artifact(descriptor["id"])[0], data)
        self.graphs.dispatch("job.delete", {"job_id": job["id"]})
        self.assertEqual(self.graphs.snapshot()["artifacts"], [])
        with self.assertRaises(KeyError):
            self.graphs.artifact(descriptor["id"])

    def test_saved_job_status_fields_and_duplicate_replay_keys_cannot_claim_false_success(self):
        graph_id = self.graph([_node("n", "core.number", 0, 0)])
        self.execute(graph_id, idempotency_key="distinct-request")
        original = self.graphs.export_state()
        for mutation in ("field", "status", "key"):
            bad = copy.deepcopy(original)
            if mutation == "field":
                del bad["jobs"][0]["nodes"]["n"]["outputs"]
            elif mutation == "status":
                bad["jobs"][0]["nodes"]["n"]["status"] = "failed"
            else:
                copied = copy.deepcopy(bad["jobs"][0])
                copied["id"] = "second-job"
                bad["jobs"].append(copied)
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.graphs.import_state(bad)
                self.assertEqual(self.graphs.export_state(), original)

    def test_history_retains_declared_count_and_rejects_boolean_execution_revisions(self):
        graph_id = self.graph([_node("n", "core.number", 0, 0)])
        for invalid in (True, 1.0, "1"):
            with self.subTest(revision=invalid):
                with self.assertRaises(ValueError):
                    self.graphs.dispatch("graph.execute", {"graph_id": graph_id, "expected_revision": invalid})
        with patch("single_parameter_observatory.studio_graph.MAX_JOBS", 2):
            identities = [self.execute(graph_id)["id"] for _ in range(3)]
            self.assertEqual([job["id"] for job in self.graphs.snapshot()["jobs"]], identities[-2:])

    def test_job_observation_download_getter_is_an_exact_independent_copy(self):
        value = 2**60 + 1
        result = self.execute(self.graph([_node("value", "core.json", 0, 0, value=value)]))
        snapshot = self.graphs.job(result["id"])
        self.assertEqual(snapshot["nodes"]["value"]["outputs"]["value"], value)
        snapshot["nodes"]["value"]["outputs"]["value"] = 0
        self.assertEqual(self.graphs.job(result["id"])["nodes"]["value"]["outputs"]["value"], value)
        with self.assertRaises(KeyError):
            self.graphs.job("missing-job")


if __name__ == "__main__":
    unittest.main()
