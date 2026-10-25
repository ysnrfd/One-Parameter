"""Real tensor-DAG composition, native Llama parity, training and safe reload."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from typing import Any

from single_parameter_observatory.studio_transformers import TransformersWorker
from single_parameter_observatory.studio_transformers_worker import Runtime


def llama_graph(runtime: Any, *, custom_attention: bool = False, custom_norm: bool = False,
                custom_rope: bool = False, custom_embedding: bool = False, custom_positions: bool = False,
                custom_mask: bool = False, fused: bool = False, layers: int = 1):
    expressions, identifiers = {}, {}

    def op(name, operation, inputs=None, **params):
        value = runtime.execute(operation, inputs or {}, params)["x"]
        expressions[name], identifiers[name] = value, value["output"]
        return value

    tokens = op("tokens", "hf.tensor_input", field="input_ids", dtype="int64")
    mask = op("mask", "hf.tensor_input", field="attention_mask", dtype="int64")
    if custom_mask:
        mask = op("mask_custom", "hf.op_custom", {"x": mask}, code="return clamp(inputs['x'], min=0, max=1)", layers={})
    if custom_positions:
        positions = op("positions", "hf.op_custom", {"x": mask}, code="return clamp(cumsum(inputs['x'], dim=-1) - 1, min=0)", layers={})
    else:
        positions = op("positions", "hf.op_positions", {"x": mask})
    if custom_embedding:
        x = op("embed", "hf.op_custom", {"x": tokens}, code="return layers['table'](inputs['x'])",
               layers={"table": {"type": "Embedding", "num_embeddings": 16, "embedding_dim": 16}})
    else:
        x = op("embed", "hf.op_embedding", {"x": tokens}, num_embeddings=16, embedding_dim=16)
    for index in range(layers):
        prefix = f"l{index}_"
        residual = x
        if custom_norm:
            x = op(prefix + "norm1", "hf.op_custom", {"x": x}, code="return layers['norm'](inputs['x'])",
                   layers={"norm": {"type": "RMSNorm", "normalized_shape": 16, "eps": 1e-6}})
        else:
            x = op(prefix + "norm1", "hf.op_rmsnorm", {"x": x}, normalized_shape=16, eps=1e-6)
        q = op(prefix + "q", "hf.op_linear", {"x": x}, in_features=16, out_features=16, bias=False)
        k = op(prefix + "k", "hf.op_linear", {"x": x}, in_features=16, out_features=8, bias=False)
        v = op(prefix + "v", "hf.op_linear", {"x": x}, in_features=16, out_features=8, bias=False)
        if custom_rope:
            q = op(prefix + "ropeq", "hf.op_custom", {"x": q}, code="return rope(inputs['x'], 2, 8)", layers={})
            k = op(prefix + "ropek", "hf.op_custom", {"x": k}, code="return rope(inputs['x'], 1, 8)", layers={})
        else:
            q = op(prefix + "ropeq", "hf.op_rope", {"x": q, "positions": positions}, num_heads=2, head_dim=8, theta=10000)
            k = op(prefix + "ropek", "hf.op_rope", {"x": k, "positions": positions}, num_heads=1, head_dim=8, theta=10000)
        if custom_attention:
            x = op(prefix + "attention", "hf.op_custom", {"x": q, "a": k, "b": v},
                   code="return causal_attention(inputs['x'], inputs['a'], inputs['b'], 2, 1, 8)", layers={})
        else:
            x = op(prefix + "attention", "hf.op_attention", {"q": q, "k": k, "v": v, "mask": mask},
                   num_heads=2, num_key_value_heads=1, head_dim=8, causal=True, dropout=0)
        x = op(prefix + "o", "hf.op_linear", {"x": x}, in_features=16, out_features=16, bias=False)
        x = op(prefix + "residual1", "hf.op_add", {"a": residual, "b": x})
        residual = x
        x = op(prefix + "norm2", "hf.op_rmsnorm", {"x": x}, normalized_shape=16)
        if fused:
            x = op(prefix + "swiglu", "hf.op_swiglu", {"x": x}, hidden_size=16, intermediate_size=32, bias=False)
        else:
            gate = op(prefix + "gate", "hf.op_linear", {"x": x}, in_features=16, out_features=32, bias=False)
            up = op(prefix + "up", "hf.op_linear", {"x": x}, in_features=16, out_features=32, bias=False)
            gate = op(prefix + "silu", "hf.op_silu", {"x": gate})
            x = op(prefix + "multiply", "hf.op_multiply", {"a": gate, "b": up})
            x = op(prefix + "down", "hf.op_linear", {"x": x}, in_features=32, out_features=16, bias=False)
        x = op(prefix + "residual2", "hf.op_add", {"a": residual, "b": x})
    x = op("final_norm", "hf.op_rmsnorm", {"x": x}, normalized_shape=16)
    x = op("lm_head", "hf.op_linear", {"x": x}, in_features=16, out_features=16, bias=False)
    model = runtime.execute("hf.graph_model", {"logits": x}, {"vocab_size": 16, "max_positions": 32,
                            "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 3, "seed": 42})
    return model, identifiers, expressions


class TestStudioComponents(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.r = self.runtime.execute
        self.torch = self.runtime.torch

    def live(self, descriptor):
        return self.runtime.models[descriptor["model"]["handle"]][0]

    def data(self):
        return self.r("hf.data", {"fields": {"input_ids": [[2, 4, 5, 6, 3], [2, 4, 7, 6, 3]],
                       "attention_mask": [[1] * 5, [1] * 5], "labels": [[2, 4, 5, 6, 3], [2, 4, 7, 6, 3]]}}, {})

    def test_registered_dag_matches_native_llama_logits_and_gradients(self):
        import transformers

        descriptor, ids, _ = llama_graph(self.runtime)
        model = self.live(descriptor)
        cfg = transformers.LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                 num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32, rms_norm_eps=1e-6,
                 pad_token_id=0, bos_token_id=2, eos_token_id=3, attention_bias=False, mlp_bias=False, tie_word_embeddings=False)
        cfg._attn_implementation = "eager"
        native = transformers.LlamaForCausalLM(cfg)
        mapping = {"embed": native.model.embed_tokens, "l0_norm1": native.model.layers[0].input_layernorm,
                   "l0_q": native.model.layers[0].self_attn.q_proj, "l0_k": native.model.layers[0].self_attn.k_proj,
                   "l0_v": native.model.layers[0].self_attn.v_proj, "l0_o": native.model.layers[0].self_attn.o_proj,
                   "l0_norm2": native.model.layers[0].post_attention_layernorm,
                   "l0_gate": native.model.layers[0].mlp.gate_proj, "l0_up": native.model.layers[0].mlp.up_proj,
                   "l0_down": native.model.layers[0].mlp.down_proj, "final_norm": native.model.norm, "lm_head": native.lm_head}
        for name, module in mapping.items():
            model.operations[ids[name]].load_state_dict(module.state_dict())
        tokens = self.torch.tensor([[0, 2, 4, 5, 6], [2, 4, 7, 6, 3]])
        mask = (tokens != 0).long()
        model.eval(), native.eval()
        expected = native(input_ids=tokens, attention_mask=mask).logits
        actual = model(input_ids=tokens, attention_mask=mask)["logits"]
        self.torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], rtol=2e-5, atol=2e-6)
        actual[mask.bool()].sum().backward()
        expected[mask.bool()].sum().backward()
        for name, module in mapping.items():
            for graph_parameter, native_parameter in zip(model.operations[ids[name]].parameters(), module.parameters(), strict=True):
                self.assertIsNotNone(graph_parameter.grad)
                self.torch.testing.assert_close(graph_parameter.grad, native_parameter.grad, rtol=1e-4, atol=1e-5)
        report = self.r("hf.inspect", descriptor, {})["report"]
        self.assertEqual(report["parameter_count"], sum(p.numel() for p in native.parameters()))
        self.assertEqual(report["tensor_graph"]["operation_count"], len(report["recipe"]["expression"]["nodes"]))
        fields = {"input_ids": tokens[1:].tolist(), "labels": tokens[1:].tolist()}
        data = self.r("hf.data", {"fields": fields}, {})
        graph_loss = self.r("hf.evaluate", {**descriptor, **data}, {"loss": "model"})["metrics"]["loss"]
        native_loss = float(native(input_ids=tokens[1:], labels=tokens[1:]).loss.detach())
        self.assertAlmostEqual(graph_loss, native_loss, places=6)

    def test_training_generation_checkpoint_and_deterministic_replay(self):
        model, _, _ = llama_graph(self.runtime, layers=2)
        data = self.data()
        before = self.r("hf.evaluate", {**model, **data}, {"loss": "model"})["metrics"]["loss"]
        optimizer = self.r("hf.optimizer", {}, {"name": "AdamW", "lr": 0.015})
        trained = self.r("hf.train", {**model, **data, **optimizer}, {"steps": 35, "batch_size": 2, "loss": "model"})
        metrics = trained["metrics"]
        after = self.r("hf.evaluate", {**model, **data}, {"loss": "model"})["metrics"]["loss"]
        self.assertTrue(metrics["parameter_changed"])
        self.assertLess(after, before * 0.3)
        self.assertTrue(all(value > 0 for value in metrics["gradient_norms"]))
        prompt = self.r("hf.data", {"fields": {"input_ids": [[2, 4]], "attention_mask": [[1, 1]]}}, {})
        generated = self.r("hf.generate", {**model, **prompt}, {"max_new_tokens": 4})["result"]
        exported = self.r("hf.checkpoint", model, {})
        loaded = self.r("hf.load", {"data_base64": exported["artifact"]["data_base64"]}, {})
        self.assertEqual(self.r("hf.inspect", loaded, {})["report"]["state_sha256"], metrics["state_sha256_after"])
        self.assertEqual(self.r("hf.generate", {**loaded, **prompt}, {"max_new_tokens": 4})["result"], generated)
        self.assertEqual(self.r("hf.evaluate", {**loaded, **data}, {"loss": "model"})["metrics"]["loss"], after)
        fresh = Runtime()
        replay, _, _ = llama_graph(fresh, layers=2)
        initial = Runtime()
        original, _, _ = llama_graph(initial, layers=2)
        self.assertEqual(fresh.execute("hf.inspect", replay, {})["report"]["state_sha256"],
                         initial.execute("hf.inspect", original, {})["report"]["state_sha256"])

    def test_causality_padding_positions_and_shifted_loss(self):
        descriptor, _, _ = llama_graph(self.runtime)
        model = self.live(descriptor).eval()
        original = self.torch.tensor([[2, 4, 5, 6]])
        changed = self.torch.tensor([[2, 4, 7, 8]])
        a, b = model(input_ids=original)["logits"], model(input_ids=changed)["logits"]
        self.assertTrue(self.torch.equal(a[:, :2], b[:, :2]))
        padded = self.torch.tensor([[0, 0, 2, 4, 5, 6]])
        p = model(input_ids=padded, attention_mask=(padded != 0).long())["logits"]
        self.torch.testing.assert_close(a, p[:, 2:], rtol=2e-5, atol=2e-6)
        unpadded = self.r("hf.data", {"fields": {"input_ids": original.tolist(), "labels": original.tolist()}}, {})
        padded_data = self.r("hf.data", {"fields": {"input_ids": padded.tolist(), "labels": padded.tolist(),
                              "attention_mask": (padded != 0).long().tolist()}}, {})
        a_loss = self.r("hf.evaluate", {**descriptor, **unpadded}, {"loss": "model"})["metrics"]["loss"]
        b_loss = self.r("hf.evaluate", {**descriptor, **padded_data}, {"loss": "model"})["metrics"]["loss"]
        self.assertAlmostEqual(a_loss, b_loss, places=6)

    def test_custom_norm_rope_attention_and_fused_swiglu_really_execute(self):
        for choices in ({"custom_norm": True}, {"custom_rope": True}, {"custom_attention": True}, {"fused": True},
                        {"custom_norm": True, "custom_rope": True, "custom_attention": True, "fused": True}):
            with self.subTest(choices=choices):
                runtime = Runtime()
                descriptor, _ids, _ = llama_graph(runtime, **choices)
                data = runtime.execute("hf.data", {"fields": {"input_ids": [[2, 4, 5, 3]], "labels": [[2, 4, 5, 3]]}}, {})
                optimizer = runtime.execute("hf.optimizer", {}, {"name": "AdamW", "lr": 0.01})
                trained = runtime.execute("hf.train", {**descriptor, **data, **optimizer}, {"steps": 3, "batch_size": 1, "loss": "model"})
                self.assertTrue(trained["metrics"]["parameter_changed"])
                self.assertTrue(all(parameter.grad is not None for parameter in runtime.models[descriptor["model"]["handle"]][0].parameters()))
                exported = runtime.execute("hf.checkpoint", descriptor, {})
                loaded = runtime.execute("hf.load", {"data_base64": exported["artifact"]["data_base64"]}, {})
                self.assertEqual(runtime.execute("hf.inspect", loaded, {})["report"]["state_sha256"], trained["metrics"]["state_sha256_after"])

    def test_invalid_geometry_custom_syntax_and_disconnected_recipe_rejected(self):
        _, _, expressions = llama_graph(self.runtime)
        for operation, inputs, params, message in (
            ("hf.op_linear", {"x": expressions["embed"]}, {"in_features": 12, "out_features": 16}, "in_features"),
            ("hf.op_rope", {"x": expressions["embed"]}, {"num_heads": 2, "head_dim": 7}, "even"),
            ("hf.op_attention", {"q": expressions["l0_q"], "k": expressions["l0_k"], "v": expressions["l0_v"]},
             {"num_heads": 3, "num_key_value_heads": 2, "head_dim": 8}, "divisible"),
            ("hf.op_add", {"a": expressions["embed"], "b": expressions["l0_up"]}, {}, "identical"),
            ("hf.op_custom", {"x": expressions["embed"]}, {"code": "import os\nreturn inputs['x']"}, "only"),
            ("hf.op_custom", {"x": expressions["embed"]}, {"code": "return inputs['x'].__class__"}, "Unsupported"),
        ):
            with self.subTest(operation=operation, params=params):
                expression = self.r(operation, inputs, params)["x"]
                head = self.r("hf.op_linear", {"x": expression}, {"in_features": 16, "out_features": 16})["x"]
                with self.assertRaisesRegex((ValueError, SyntaxError), message):
                    self.r("hf.graph_model", {"logits": head}, {"vocab_size": 16, "max_positions": 32})
        disconnected = copy.deepcopy(expressions["lm_head"])
        disconnected["nodes"].append({"id": "hidden", "op": "hf.tensor_input", "inputs": {}, "params": {}})
        with self.assertRaisesRegex(ValueError, "disconnected"):
            self.r("hf.graph_model", {"logits": disconnected}, {"vocab_size": 16})

    def test_budget_guard_runs_before_cpu_weight_allocation(self):
        runtime = Runtime({"max_parameters": 100})
        with self.assertRaisesRegex(ValueError, "parameters"):
            llama_graph(runtime)
        self.assertEqual(runtime.models, {})

    def test_graph_text_generation_pipeline_decodes_complete_sequences(self):
        descriptor, _, _ = llama_graph(self.runtime)
        vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, **{f"w{index}": index for index in range(4, 16)}}
        tokenizer = self.r("hf.tokenizer", {"vocab": vocab}, {})
        result = self.r("hf.pipeline", {**descriptor, **tokenizer, "texts": ["w4 w5", "w6"]},
                        {"task": "text-generation", "max_new_tokens": 3})["result"]
        self.assertEqual(result["task"], "text-generation")
        native_tokenizer = self.runtime.tokenizers[tokenizer["tokenizer"]["handle"]]
        for records in result["outputs"]:
            record = records[0]
            self.assertEqual(record["generated_text"], native_tokenizer.decode(record["generated_token_ids"], skip_special_tokens=True))

    def test_custom_embedding_position_and_mask_blocks_support_actual_integer_axes_and_reload(self):
        descriptor, ids, _ = llama_graph(self.runtime, custom_embedding=True, custom_positions=True, custom_mask=True)
        model = self.live(descriptor)
        observed = {}
        for key in ("embed", "positions", "mask_custom"):
            model.operations[ids[key]].register_forward_hook(lambda module, arguments, output, key=key: observed.update({key: output.detach().clone()}))
        padded = self.torch.tensor([[0, 0, 2, 4, 5], [0, 2, 4, 7, 3]])
        mask = (padded != 0).long()
        logits = model(input_ids=padded, attention_mask=mask)["logits"]
        self.assertEqual(list(logits.shape), [2, 5, 16])
        self.assertEqual(observed["embed"].dtype, self.torch.float32)
        self.assertEqual(list(observed["embed"].shape), [2, 5, 16])
        self.assertEqual(observed["positions"].dtype, self.torch.int64)
        self.assertEqual(observed["positions"].tolist(), [[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]])
        self.assertTrue(self.torch.equal(observed["mask_custom"], mask))
        data = self.r("hf.data", {"fields": {"input_ids": padded.tolist(), "attention_mask": mask.tolist(), "labels": padded.tolist()}}, {})
        before = self.r("hf.evaluate", {**descriptor, **data}, {"loss": "model"})["metrics"]["loss"]
        optimizer = self.r("hf.optimizer", {}, {"name": "AdamW", "lr": 0.015})
        trained = self.r("hf.train", {**descriptor, **data, **optimizer}, {"steps": 12, "batch_size": 2, "loss": "model"})
        after = self.r("hf.evaluate", {**descriptor, **data}, {"loss": "model"})["metrics"]["loss"]
        self.assertLess(after, before)
        self.assertIsNotNone(model.operations[ids["embed"]].layers["table"].weight.grad)
        exported = self.r("hf.checkpoint", descriptor, {})
        loaded = self.r("hf.load", {"data_base64": exported["artifact"]["data_base64"]}, {})
        self.assertEqual(self.r("hf.inspect", loaded, {})["report"]["state_sha256"], trained["metrics"]["state_sha256_after"])
        self.assertEqual(self.r("hf.generate", {**descriptor, **data}, {"max_new_tokens": 2})["result"],
                         self.r("hf.generate", {**loaded, **data}, {"max_new_tokens": 2})["result"])

    def test_actual_isolated_worker_loads_trusted_sibling_and_replays_custom_model(self):
        with TransformersWorker() as worker:
            descriptor, _, _ = llama_graph(SimpleNamespace(execute=worker.request), custom_rope=True, custom_attention=True)
            data = worker.request("hf.data", {"fields": {"input_ids": [[2, 4, 5, 3]], "labels": [[2, 4, 5, 3]]}}, {})
            optimizer = worker.request("hf.optimizer", {}, {"name": "AdamW", "lr": 0.01})
            trained = worker.request("hf.train", {**descriptor, **data, **optimizer}, {"steps": 2, "batch_size": 1, "loss": "model"})
            self.assertTrue(trained["metrics"]["parameter_changed"])
            artifact = worker.request("hf.checkpoint", descriptor, {})["artifact"]
            loaded = worker.request("hf.load", {"data_base64": artifact["data_base64"]}, {})
            self.assertEqual(worker.request("hf.inspect", loaded, {})["report"]["state_sha256"], trained["metrics"]["state_sha256_after"])

    def test_imported_expression_namespace_preserves_live_identity_and_rejects_shadowing(self):
        live = self.r("hf.tensor_input", {}, {"field": "input_ids", "dtype": "int64"})["x"]
        live_id = live["output"]
        tampered = copy.deepcopy(live)
        tampered["nodes"][0]["params"]["field"] = "attention_mask"
        token_embedding = self.r("hf.op_embedding", {"x": live}, {"num_embeddings": 16, "embedding_dim": 16})["x"]
        forged_embedding = self.r("hf.op_embedding", {"x": tampered}, {"num_embeddings": 16, "embedding_dim": 16})["x"]
        forged_input = forged_embedding["nodes"][0]
        self.assertTrue(forged_input["id"].startswith("import_"))
        self.assertNotEqual(forged_input["id"], live_id)
        self.assertEqual(self.runtime.expression_nodes[live_id]["params"]["field"], "input_ids")
        joined = self.r("hf.op_add", {"a": token_embedding, "b": forged_embedding}, {})["x"]
        self.assertEqual(len([node for node in joined["nodes"] if node["op"] == "hf.tensor_input"]), 2)
        shared = self.r("hf.op_add", {"a": token_embedding, "b": copy.deepcopy(token_embedding)}, {})["x"]
        self.assertEqual(len([node for node in shared["nodes"] if node["op"] == "hf.op_embedding"]), 1)
        self.assertEqual(shared["nodes"][0]["id"], live_id)
        # A literal can predate a worker counter ID without colliding later.
        future = copy.deepcopy(live)
        future["nodes"][0]["id"] = future["output"] = "op0008"
        imported = self.r("hf.op_embedding", {"x": future}, {"num_embeddings": 16, "embedding_dim": 16})["x"]
        self.assertTrue(imported["nodes"][0]["id"].startswith("import_"))
        normalized_id = imported["nodes"][0]["id"]
        second = self.r("hf.op_embedding", {"x": future}, {"num_embeddings": 16, "embedding_dim": 16})["x"]
        self.assertEqual(second["nodes"][0]["id"], normalized_id)


if __name__ == "__main__":
    unittest.main()
