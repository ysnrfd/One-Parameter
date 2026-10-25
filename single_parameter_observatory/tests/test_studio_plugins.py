"""Real adversarial V8 execution and declarative extension validation."""

from __future__ import annotations

import copy
import math
import os
import subprocess
import tempfile
import threading
import time
import unittest
from typing import Any
from pathlib import Path
from unittest.mock import patch

try:
    from . import _bootstrap
except ImportError:
    import _bootstrap  # type: ignore[import-not-found, no-redef]

from single_parameter_observatory.studio_plugins import (
    MAX_CODE_BYTES, MAX_JSON_BYTES, MAX_WORKER_REQUEST, WORKER_EXECUTION_GRACE_MS,
    WORKER_STARTUP_GRACE_MS, ScriptSandboxError, example_plugin,
    execute_script, node_definitions, sandbox_capabilities, validate_plugin,
)


TENSOR_EXPRESSION: dict[str, Any] = {"kind": "tensor-expression", "nodes": [
    {"id": "op0001", "op": "hf.tensor_input", "inputs": {},
     "params": {"field": "input_ids", "dtype": "int64"}},
], "output": "op0001"}


class TestPluginManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = example_plugin()

    def test_example_and_namespaced_independent_catalog(self) -> None:
        result = validate_plugin(self.manifest)
        self.assertEqual(result, self.manifest)
        self.assertEqual(result["nodes"][0]["inputs"][0]["required"], True)
        definitions = node_definitions(result)
        self.assertEqual(definitions[0]["type"], "local.examples/scale")
        self.assertEqual(definitions[0]["id"], "local.examples/scale")
        self.assertEqual(definitions[0]["node_id"], "scale")
        definitions[0]["parameters"][0]["default"] = 99
        self.assertEqual(result["nodes"][0]["parameters"][0]["default"], 2)

    def test_all_port_types_and_rich_parameter_constraints(self) -> None:
        self.manifest["nodes"][0]["inputs"] = [
            {"name": kind.replace("-", "_"), "type": kind, "required": False} for kind in
            ("any", "number", "integer", "string", "boolean", "object", "array", "run", "config", "dataset", "tensor-expression")
        ]
        self.manifest["nodes"][0]["parameters"] = [
            {"name": "factor", "type": "number", "default": 2.5, "min": 0, "max": 5,
             "options": [1, 2.5, 5], "description": "A precise factor"},
            {"name": "flags", "type": "object", "default": {"enabled": True}},
        ]
        self.assertEqual(validate_plugin(self.manifest)["nodes"][0]["parameters"][0]["default"], 2.5)

    def test_tensor_expression_input_parameter_defaults_and_options_are_validated_independently(self) -> None:
        alternative = copy.deepcopy(TENSOR_EXPRESSION)
        alternative["nodes"][0]["params"]["field"] = "attention_mask"
        self.manifest["nodes"][0]["inputs"] = [
            {"name": "expression", "type": "tensor-expression", "default": TENSOR_EXPRESSION},
        ]
        self.manifest["nodes"][0]["parameters"] = [
            {"name": "fallback", "type": "tensor-expression", "default": TENSOR_EXPRESSION,
             "options": [TENSOR_EXPRESSION, alternative]},
        ]
        result = validate_plugin(self.manifest)
        fields = result["nodes"][0]
        self.assertEqual(fields["inputs"][0]["default"], TENSOR_EXPRESSION)
        self.assertEqual(fields["parameters"][0]["default"], TENSOR_EXPRESSION)
        self.assertEqual(fields["parameters"][0]["options"], [TENSOR_EXPRESSION, alternative])
        definitions = node_definitions(result)
        definitions[0]["inputs"][0]["default"]["nodes"][0]["params"]["field"] = "mutated"
        self.assertEqual(fields["inputs"][0]["default"]["nodes"][0]["params"]["field"], "input_ids")

    def test_invalid_tensor_expression_defaults_and_options_raise_sandbox_error(self) -> None:
        invalid_values: list[Any] = [None, True, 3, "tensor-expression", [], {}, {"kind": "object"}, {"kind": None}]
        for value in invalid_values:
            for location in ("input-default", "parameter-default", "parameter-option", "panel-default"):
                candidate = copy.deepcopy(self.manifest)
                field: dict[str, Any] = {"name": "expression", "type": "tensor-expression"}
                if location.endswith("option"):
                    field["options"] = [TENSOR_EXPRESSION, value]
                else:
                    field["default"] = value
                if location == "input-default":
                    candidate["nodes"][0]["inputs"] = [field]
                elif location == "panel-default":
                    candidate["panels"] = [{"id": "settings", "title": "Settings", "fields": [field]}]
                else:
                    candidate["nodes"][0]["parameters"] = [field]
                with self.subTest(value=value, location=location), self.assertRaisesRegex(ScriptSandboxError, "must match type tensor-expression"):
                    validate_plugin(candidate)

    def test_real_plugin_literal_and_live_expression_passthrough_train_reload_and_generate(self) -> None:
        from single_parameter_observatory.studio_components import llama_blocks_workflow
        from single_parameter_observatory.studio_graph import GraphStudio

        manifest = validate_plugin({
            "schema_version": "spobs-plugin/1", "id": "local.tensor_bridge", "name": "Tensor expression bridge",
            "version": "1.0.0", "nodes": [{
                "id": "pass", "title": "Pass tensor expression",
                "inputs": [{"name": "expression", "type": "tensor-expression", "default": TENSOR_EXPRESSION}],
                "outputs": [{"name": "x", "type": "tensor-expression"}],
                "parameters": [{"name": "fallback", "type": "tensor-expression", "default": TENSOR_EXPRESSION,
                                "options": [TENSOR_EXPRESSION]}],
                "code": "return {x: inputs.expression};",
            }],
        })
        graph = llama_blocks_workflow(steps=2)
        tokens = next(node for node in graph["nodes"] if node["id"] == "tokens")
        tokens.update(type="local.tensor_bridge/pass", params={})
        graph["nodes"].append({"id": "live_passthrough", "type": "local.tensor_bridge/pass",
                               "position": {"x": 900, "y": 0}, "params": {}})
        q_edge = next(edge for edge in graph["edges"] if edge["source"] == "layer0-norm1" and edge["target"] == "layer0-q")
        q_edge.update(target="live_passthrough", target_port="expression")
        graph["edges"].append({"id": "bridge-to-q", "source": "live_passthrough", "source_port": "x",
                               "target": "layer0-q", "target_port": "x"})
        studio = GraphStudio(None)
        try:
            studio.dispatch("plugin.install", {"manifest": manifest})
            graph_id = studio.dispatch("graph.create", {"graph": graph})["graph_id"]
            validation = studio.dispatch("graph.validate", {"graph_id": graph_id})
            self.assertTrue(validation["valid"], validation)
            job_id = studio.dispatch("graph.execute", {"graph_id": graph_id})["job_id"]
            job = studio.wait_job(job_id, 120)
            self.assertEqual(job["status"], "succeeded", job.get("error"))
            outputs = {identity: node["outputs"] for identity, node in job["nodes"].items()}
            self.assertEqual(outputs["tokens"]["x"], TENSOR_EXPRESSION)
            self.assertEqual(outputs["mask"]["x"]["output"], "op0001")
            # The literal op0001 and live mask op0001 are different sources.
            imported = next(node for node in outputs["embedding"]["x"]["nodes"] if node["op"] == "hf.tensor_input")
            self.assertTrue(imported["id"].startswith("import_"))
            self.assertEqual(imported["params"]["field"], "input_ids")
            # A genuine live expression crossing V8 retains shared identity.
            self.assertEqual(outputs["live_passthrough"]["x"], outputs["layer0-norm1"]["x"])
            report = outputs["inspect"]["report"]
            norm_id = outputs["layer0-norm1"]["x"]["output"]
            self.assertEqual(sum(node["id"] == norm_id for node in report["tensor_graph"]["expression"]["nodes"]), 1)
            self.assertEqual(sum(parameter["name"] == f"operations.{norm_id}.weight" for parameter in report["parameters"]), 1)
            self.assertTrue(outputs["train"]["metrics"]["parameter_changed"])
            self.assertEqual(outputs["trained-eval"]["metrics"]["loss"], outputs["reload-eval"]["metrics"]["loss"])
            self.assertEqual(len(outputs["generate"]["result"]["sequences"]), 3)
            self.assertEqual(len(outputs["pipeline"]["result"]["outputs"]), 3)
        finally:
            studio.close()

    def test_declarative_template_and_panel_are_independent(self) -> None:
        self.manifest["templates"] = [{"id": "flow", "name": "Example flow", "graph": {"nodes": [], "edges": []}}]
        self.manifest["panels"] = [{"id": "settings", "title": "Experiment settings", "fields": [
            {"name": "rate", "type": "number", "default": 0.1, "min": 0, "max": 1},
        ]}]
        result = validate_plugin(self.manifest)
        self.manifest["templates"][0]["graph"]["nodes"].append("changed")
        self.assertEqual(result["templates"][0]["graph"]["nodes"], [])
        self.assertEqual(result["panels"][0]["fields"][0]["name"], "rate")

    def test_delivered_example_template_executes_actual_custom_node(self) -> None:
        from single_parameter_observatory.studio_graph import GraphStudio

        studio = GraphStudio(None)
        try:
            studio.dispatch("plugin.install", {"manifest": self.manifest})
            graph = studio.dispatch("graph.create", {"template_id": "local.examples/scale_workflow"})
            validation = studio.dispatch("graph.validate", {"graph_id": graph["graph_id"]})
            self.assertTrue(validation["valid"], validation)
            queued = studio.dispatch("graph.execute", {"graph_id": graph["graph_id"]})
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                job = next(item for item in studio.snapshot()["jobs"] if item["id"] == queued["job_id"])
                if job["status"] not in ("queued", "running", "cancelling"):
                    break
                time.sleep(0.01)
            self.assertEqual(job["status"], "succeeded", job.get("error"))
            self.assertEqual(job["nodes"]["preview"]["outputs"], {"value": 42})
        finally:
            studio.close()

    def test_unknown_code_capabilities_rejected(self) -> None:
        mutations = [
            lambda p: p.update(python="import os"),
            lambda p: p.update(permissions=["filesystem"]),
            lambda p: p["nodes"][0].update(python="import os"),
            lambda p: p["nodes"][0].update(javascript="alert(1)"),
            lambda p: p.update(panels=[{"id": "x", "title": "X", "html": "<script>alert(1)</script>"}]),
            lambda p: p.update(panels=[{"id": "x", "title": "X", "fields": [{"name": "x", "type": "number", "onclick": "run()"}]}]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                candidate = copy.deepcopy(self.manifest)
                mutate(candidate)
                with self.assertRaises(ScriptSandboxError):
                    validate_plugin(candidate)

    def test_invalid_schema_identifiers_and_duplicates(self) -> None:
        mutations = [
            lambda p: p.update(schema_version="wrong"), lambda p: p.update(id="../path"),
            lambda p: p.update(id="évil"), lambda p: p.update(id="constructor"),
            lambda p: p["nodes"].append(copy.deepcopy(p["nodes"][0])),
            lambda p: p["nodes"][0]["inputs"].append(copy.deepcopy(p["nodes"][0]["inputs"][0])),
            lambda p: p["nodes"][0]["inputs"][0].update(name="__proto__"),
            lambda p: p["nodes"][0]["inputs"][0].update(type="file"),
            lambda p: p["nodes"][0]["inputs"][0].update(required=1),
            lambda p: p["nodes"][0].update(code="é" * MAX_CODE_BYTES),
            lambda p: p.update(nodes=[], templates=[], panels=[]),
        ]
        for mutate in mutations:
            candidate = copy.deepcopy(self.manifest)
            mutate(candidate)
            with self.subTest(candidate=str(candidate)[:150]), self.assertRaises(ScriptSandboxError):
                validate_plugin(candidate)

    def test_bad_defaults_bounds_and_options_rejected(self) -> None:
        bad = [
            {"name": "x", "type": "number", "default": True},
            {"name": "x", "type": "integer", "default": 1.5},
            {"name": "x", "type": "number", "default": float("nan")},
            {"name": "x", "type": "number", "min": 10 ** 1000},
            {"name": "x", "type": "number", "min": 5, "max": 1},
            {"name": "x", "type": "number", "default": 6, "min": 1, "max": 5},
            {"name": "x", "type": "integer", "min": 1.5},
            {"name": "x", "type": "string", "min": 1},
            {"name": "x", "type": "string", "default": "c", "options": ["a", "b"]},
            {"name": "x", "type": "string", "options": []},
            {"name": "x", "type": "dataset", "default": {"inputs": [1], "targets": []}},
            {"name": "x", "type": "dataset", "default": {"inputs": ["x"], "targets": [1]}},
        ]
        for parameter in bad:
            candidate = copy.deepcopy(self.manifest)
            candidate["nodes"][0]["parameters"] = [parameter]
            with self.subTest(parameter=parameter), self.assertRaises(ScriptSandboxError):
                validate_plugin(candidate)


class TestScriptSandbox(unittest.TestCase):
    def test_real_javascript_transform_named_outputs_and_logs(self) -> None:
        result = execute_script(
            "console.log('factor', params.factor); console.warn('observed'); "
            "return {sum: inputs.values.reduce((s,x)=>s+x,0)*params.factor, nested: {ok: true}};",
            {"values": [1, 2, 3]}, {"factor": 4},
        )
        self.assertEqual(result["outputs"], {"sum": 24, "nested": {"ok": True}})
        self.assertEqual(result["logs"], ["log: factor 4", "warn: observed"])
        self.assertIn("V8", result["runtime"])
        self.assertGreater(result["duration_ms"], 0)

    def test_actual_example_plugin_node_runs(self) -> None:
        node = example_plugin()["nodes"][1]
        result = execute_script(node["code"], {"data": {"inputs": [1, 2], "targets": [3, 4],
                                                      "name": "Exact source", "batch_size": 2,
                                                      "provenance": {"source": "test"}}}, {"factor": 2})
        self.assertEqual(result["outputs"]["data"]["targets"], [6, 8])
        self.assertEqual(result["outputs"]["data"]["inputs"], [1, 2])
        self.assertEqual(result["outputs"]["data"]["name"], "Exact source")
        self.assertEqual(result["outputs"]["data"]["batch_size"], 2)
        self.assertEqual(result["outputs"]["data"]["provenance"]["source"], "test")

    def test_no_host_io_or_callbacks_are_exposed(self) -> None:
        result = execute_script(
            "return {process: typeof process, require: typeof require, fetch: typeof fetch, "
            "XMLHttpRequest: typeof XMLHttpRequest, document: typeof document, Python: typeof Python, "
            "Deno: typeof Deno, Bun: typeof Bun, read: typeof read, "
            "timers: typeof setTimeout, wasm: typeof WebAssembly, buffer: typeof ArrayBuffer};", {}, {},
        )
        self.assertTrue(all(value == "undefined" for value in result["outputs"].values()), result)
        for code in ("return {data: require('fs').readFileSync('secret')};",
                     "return {data: process.env};", "return {data: fetch('https://example.com')};"):
            with self.subTest(code=code), self.assertRaises(ScriptSandboxError):
                execute_script(code, {}, {})

    def test_each_execution_has_fresh_global_state(self) -> None:
        execute_script("globalThis.saved = 123; return {ok: true};", {}, {})
        self.assertEqual(execute_script("return {saved: typeof saved};", {}, {})["outputs"], {"saved": "undefined"})

    def test_cpu_exhaustion_is_bounded_and_healthy_worker_recovers(self) -> None:
        started = time.monotonic()
        with self.assertRaisesRegex(ScriptSandboxError, "timeout|deadline|terminated"):
            execute_script("while (true) {}", {}, {}, timeout_ms=50)
        self.assertLess(time.monotonic() - started, (WORKER_STARTUP_GRACE_MS + WORKER_EXECUTION_GRACE_MS) / 1000 + 6)
        self.assertEqual(execute_script("return {ok: 1};", {}, {})["outputs"], {"ok": 1})

    def test_memory_exhaustion_is_bounded_and_healthy_worker_recovers(self) -> None:
        started = time.monotonic()
        with self.assertRaisesRegex(ScriptSandboxError, "memory|resource|terminated|deadline"):
            execute_script("const x=[]; while(true) x.push(new Array(20000).fill(Math.random()));", {}, {},
                           timeout_ms=2000, memory_mb=16)
        self.assertLess(time.monotonic() - started, (WORKER_STARTUP_GRACE_MS + WORKER_EXECUTION_GRACE_MS) / 1000 + 8)
        self.assertEqual(execute_script("return {ok: 2};", {}, {})["outputs"], {"ok": 2})

    def test_cancel_mid_execution_kills_and_reaps_actual_worker(self) -> None:
        cancel = threading.Event()
        created: list[subprocess.Popen[bytes]] = []
        original = subprocess.Popen

        def capture(*args, **kwargs):
            process = original(*args, **kwargs)
            created.append(process)
            return process

        timer = threading.Timer(0.5, cancel.set)
        timer.start()
        try:
            started = time.monotonic()
            with patch("single_parameter_observatory.studio_plugins.subprocess.Popen", side_effect=capture):
                with self.assertRaisesRegex(ScriptSandboxError, "cancelled"):
                    execute_script("while (true) {}", {}, {}, timeout_ms=10000, cancel=cancel)
            self.assertLess(time.monotonic() - started, 4)
            self.assertEqual(len(created), 1)
            self.assertIsNotNone(created[0].poll(), "Cancelled worker must be reaped")
            for pipe in (created[0].stdin, created[0].stdout, created[0].stderr):
                assert pipe is not None
                self.assertTrue(pipe.closed)
        finally:
            timer.cancel()
            timer.join()

    def test_cancel_before_execution_starts_no_process(self) -> None:
        cancel = threading.Event()
        cancel.set()
        with patch("single_parameter_observatory.studio_plugins.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ScriptSandboxError, "cancelled"):
                execute_script("return {};", {}, {}, cancel=cancel)
            popen.assert_not_called()

    def test_cancel_during_blocked_large_stdin_write_kills_and_reaps_worker(self) -> None:
        cancel, entered_communicate = threading.Event(), threading.Event()
        created: list[subprocess.Popen[bytes]] = []
        original = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "blocked_input.py"
            fixture.write_text(
                "import sys, time\nprint('{\"ready\":true,\"initialization_ms\":0}', flush=True)\ntime.sleep(60)\n",
                encoding="utf-8",
            )

            def capture(*args, **kwargs):
                command = list(args[0])
                command[4] = str(fixture)
                process = original(command, **kwargs)
                original_communicate = process.communicate

                def communicate(*comm_args, **comm_kwargs):
                    if comm_kwargs.get("input") is not None:
                        entered_communicate.set()
                    return original_communicate(*comm_args, **comm_kwargs)

                # Observe the actual pipe entry while retaining Popen behavior.
                process.communicate = communicate  # type: ignore[method-assign]
                created.append(process)
                return process

            def cancel_after_write_starts() -> None:
                entered_communicate.wait(WORKER_STARTUP_GRACE_MS / 1000 + 2)
                time.sleep(0.1)
                cancel.set()

            canceller = threading.Thread(target=cancel_after_write_starts, daemon=True)
            canceller.start()
            try:
                with patch("single_parameter_observatory.studio_plugins.subprocess.Popen", side_effect=capture):
                    with self.assertRaisesRegex(ScriptSandboxError, "cancelled"):
                        execute_script("return {};", {"large": "x" * 500000}, {}, cancel=cancel)
                self.assertTrue(entered_communicate.is_set(), "Regression must reach the synchronously blocked write")
                self.assertEqual(len(created), 1)
                self.assertIsNotNone(created[0].poll(), "The actual blocked worker must be killed and reaped")
            finally:
                entered_communicate.set()
                cancel.set()
                canceller.join(timeout=2)

    def test_startup_stderr_is_drained_before_ready_and_diagnostics_are_bounded(self) -> None:
        created: list[subprocess.Popen[bytes]] = []
        original = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "startup_diagnostics.py"
            fixture.write_text(
                "import sys\n"
                "sys.stderr.buffer.write(b'x' * 200000 + b'\\nSTARTUP-TAIL-MARKER\\n')\n"
                "sys.stderr.flush()\n"
                "print('{\"ready\":true,\"initialization_ms\":0}', flush=True)\n"
                "sys.stdin.buffer.read()\n"
                "print('{\"ok\":false,\"error\":\"diagnostic fixture completed\"}', flush=True)\n",
                encoding="utf-8",
            )

            def capture(*args, **kwargs):
                command = list(args[0])
                command[4] = str(fixture)
                process = original(command, **kwargs)
                created.append(process)
                return process

            with patch("single_parameter_observatory.studio_plugins.subprocess.Popen", side_effect=capture):
                with self.assertRaisesRegex(ScriptSandboxError, "diagnostic fixture completed") as raised:
                    execute_script("return {};", {}, {})
            diagnostic = str(raised.exception).split("; worker stderr: ", 1)[1]
            self.assertIn("STARTUP-TAIL-MARKER", diagnostic)
            self.assertLessEqual(len(diagnostic.encode("utf-8")), 1000)
            self.assertEqual(len(created), 1)
            self.assertIsNotNone(created[0].poll(), "The one-process Job worker must be reaped")
            for pipe in (created[0].stdin, created[0].stdout, created[0].stderr):
                assert pipe is not None
                self.assertTrue(pipe.closed)

    def test_output_must_be_strict_finite_plain_json_object(self) -> None:
        bad = ["return undefined;", "return null;", "return [1,2];", "return {bad: undefined};",
               "return {bad: NaN};", "return {bad: Infinity};", "return {bad: ()=>1};",
               "return {bad: 1n};", "return {bad: new Date()};", "return new Promise(()=>{});",
               "const a={}; a.self=a; return a;", "return {bad: [1,,3]};",
               "return {get bad(){return 1;}};", "return {bad: Symbol('x')};",
               "const a={}; Object.defineProperty(a,'hidden',{value:1}); return a;",
               "const a={}; a[Symbol('x')]=1; return a;",
               "return {bad: {toJSON(){return 1}}};"]
        for code in bad:
            with self.subTest(code=code), self.assertRaises(ScriptSandboxError):
                execute_script(code, {}, {})

    def test_syntax_error_and_script_error_surface(self) -> None:
        for code in ("return {", "throw new Error('specific fault');"):
            with self.subTest(code=code), self.assertRaises(ScriptSandboxError):
                execute_script(code, {}, {})

    def test_function_body_cannot_escape_into_trusted_wrapper(self) -> None:
        with self.assertRaises(ScriptSandboxError):
            execute_script("}); return '{\"outputs\":{\"fake\":true}}'; //", {}, {})
        result = execute_script("return {closure: typeof payload, host: Function('return typeof process')()};", {}, {})
        self.assertEqual(result["outputs"], {"closure": "undefined", "host": "undefined"})

    def test_input_mutations_cannot_modify_parent_and_prototype_pollution_fails(self) -> None:
        inputs = {"values": [1, 2]}
        execute_script("inputs.values.push(3); return {data: inputs.values};", inputs, {})
        self.assertEqual(inputs, {"values": [1, 2]})
        with self.assertRaises(ScriptSandboxError):
            execute_script("Object.prototype.toJSON=()=>({fake:true}); return {real:true};", {}, {})
        result = execute_script("return JSON.parse('{\"__proto__\":{\"polluted\":true}}');", {}, {})
        self.assertEqual(result["outputs"], {"__proto__": {"polluted": True}})

    def test_logs_and_result_sizes_are_bounded(self) -> None:
        result = execute_script("for(let i=0;i<10000;i++) console.log('x'.repeat(3000)); return {ok:true};", {}, {})
        self.assertLessEqual(len(result["logs"]), 100)
        self.assertLessEqual(sum(map(len, result["logs"])), 16384)
        with self.assertRaisesRegex(ScriptSandboxError, "1 MiB"):
            execute_script("return {large: 'x'.repeat(1100000)};", {}, {})

    def test_invalid_input_limits_rejected_before_worker_launch(self) -> None:
        circular: dict[str, Any] = {}
        circular["self"] = circular
        cases: list[tuple[str, Any, Any, dict[str, Any]]] = [
            ("", {}, {}, {}), ("x" * (MAX_CODE_BYTES + 1), {}, {}, {}),
            ("return {bad:'\ud800'};", {}, {}, {}),
            ("return {};", [], {}, {}), ("return {};", {}, [], {}),
            ("return {};", {"bad": float("inf")}, {}, {}),
            ("return {};", {1: "bad key"}, {}, {}), ("return {};", {"tuple": (1, 2)}, {}, {}),
            ("return {};", circular, {}, {}), ("return {};", {"large": "x" * MAX_JSON_BYTES}, {}, {}),
            ("return {};", {}, {}, {"timeout_ms": True}), ("return {};", {}, {}, {"timeout_ms": 0}),
            ("return {};", {}, {}, {"memory_mb": 15}), ("return {};", {}, {}, {"memory_mb": 65}),
        ]
        for code, inputs, params, options in cases:
            with self.subTest(options=options, code=code[:20]), self.assertRaises(ScriptSandboxError):
                execute_script(code, inputs, params, **options)

    def test_nonexact_integer_inputs_rejected_and_decimal_bigints_preserved(self) -> None:
        for value in (2 ** 53 + 1, -(2 ** 53 + 1), 10 ** 1000):
            with self.subTest(value=str(value)[:30]), self.assertRaisesRegex(ScriptSandboxError, "decimal string"):
                execute_script("return {value: inputs.value};", {"value": value}, {})
        wide = "9223372036854775807"
        result = execute_script("return {value: (BigInt(inputs.value) + 1n).toString()};", {"value": wide}, {})
        self.assertEqual(result["outputs"], {"value": "9223372036854775808"})
        result = execute_script("return {value: inputs.value === Math.pow(2,60)};", {"value": 2 ** 60}, {})
        self.assertEqual(result["outputs"], {"value": True})
        result = execute_script("return {value: inputs.value};", {"value": 2 ** 60}, {})
        self.assertEqual(result["outputs"], {"value": 2 ** 60})

    def test_negative_zero_input_and_computation_keep_their_sign(self) -> None:
        result = execute_script(
            "return {echo: inputs.value, computed: -1 * 0, nested: [-0], positive: 0};",
            {"value": -0.0}, {},
        )
        for value in (result["outputs"]["echo"], result["outputs"]["computed"], result["outputs"]["nested"][0]):
            self.assertEqual(value, 0)
            self.assertLess(math.copysign(1, value), 0)
        self.assertGreater(math.copysign(1, result["outputs"]["positive"]), 0)

    def test_capabilities_report_enforced_limits_and_boundaries(self) -> None:
        capabilities = sandbox_capabilities()
        self.assertTrue(capabilities["available"])
        self.assertEqual(capabilities["timeout_ms"], {"min": 10, "max": 10000, "default": 1000})
        self.assertEqual(capabilities["memory_mb"], {"min": 16, "max": 64, "default": 32})
        self.assertEqual(capabilities["wall_grace_ms"], WORKER_STARTUP_GRACE_MS)
        self.assertEqual(capabilities["startup_timeout_ms"], 30000)
        self.assertEqual(capabilities["execution_wall_grace_ms"], 3000)
        self.assertEqual(capabilities["worker_request_limit_bytes"], MAX_WORKER_REQUEST)
        self.assertFalse(capabilities["host_io"])
        self.assertFalse(capabilities["os_virtual_machine"])

    def test_resource_job_failure_kills_reaps_process_and_is_visible(self) -> None:
        created: list[subprocess.Popen[bytes]] = []
        original = subprocess.Popen

        def capture(*args, **kwargs):
            process = original(*args, **kwargs)
            created.append(process)
            return process

        with patch("single_parameter_observatory.studio_plugins.subprocess.Popen", side_effect=capture):
            with patch("single_parameter_observatory.studio_plugins._WindowsJob", side_effect=OSError("resource job denied")):
                with self.assertRaisesRegex(ScriptSandboxError, "resource job denied"):
                    execute_script("return {ok:true};", {}, {})
        self.assertEqual(len(created), 1)
        self.assertIsNotNone(created[0].poll())

    def test_binary64_math_unicode_and_null_prototype_values(self) -> None:
        result = execute_script(
            "const value=Object.create(null); value.label='سلام 🧪'; value.number=Math.sqrt(9); "
            "return {value, text: inputs.text};", {"text": "café"}, {},
        )
        self.assertEqual(result["outputs"], {"value": {"label": "سلام 🧪", "number": 3}, "text": "café"})

    def test_large_dataset_within_json_budget_can_be_transformed(self) -> None:
        # Both arrays together exceed 150000 values but remain under the real
        # byte budget and the studio's 100000-row per-dataset boundary.  Select
        # the permitted ten-second evaluation budget for this high-cardinality
        # workload; the JSON size limit does not promise a two-second execution.
        result = execute_script(
            "return {data: {inputs: inputs.data.inputs, targets: inputs.data.targets.map(x => x + 1)}};",
            {"data": {"inputs": [0] * 80000, "targets": [0] * 80000}}, {}, timeout_ms=10000,
        )
        self.assertEqual(len(result["outputs"]["data"]["targets"]), 80000)
        self.assertEqual(result["outputs"]["data"]["targets"][0], 1)
        self.assertEqual(result["outputs"]["data"]["targets"][-1], 1)

    def test_untrusted_environment_and_current_directory_are_not_inherited(self) -> None:
        created = []
        original = subprocess.Popen

        def capture(*args, **kwargs):
            created.append((args, kwargs))
            return original(*args, **kwargs)

        with patch.dict(os.environ, {"PYTHONPATH": "malicious", "OPENAI_API_KEY": "secret", "NODE_OPTIONS": "--bad"}):
            with patch("single_parameter_observatory.studio_plugins.subprocess.Popen", side_effect=capture):
                execute_script("return {ok:true};", {}, {})
        args, kwargs = created[0]
        command = args[0]
        self.assertIn("-I", command)
        self.assertIn("-S", command)
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
        self.assertNotIn("NODE_OPTIONS", kwargs["env"])
        self.assertNotEqual(os.path.realpath(kwargs["cwd"]), os.path.realpath(os.getcwd()))
        self.assertFalse(os.path.exists(kwargs["cwd"]), "Worker directory must be removed after completion")


if __name__ == "__main__":
    unittest.main()
