"""Actual multi-parameter training, safe artifacts and restricted forward code."""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import socket
import threading
import time
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from single_parameter_observatory.studio_graph import GraphStudio
from single_parameter_observatory.studio_session import StudioWorkspace
from single_parameter_observatory.studio_transformers import TransformersWorker, TransformersWorkerError, capabilities, definitions, templates
from single_parameter_observatory.studio_transformers_worker import Runtime, TensorProgram


MLP = [{"type": "Linear", "in_features": 1, "out_features": 8}, {"type": "RMSNorm", "normalized_shape": 8},
       {"type": "GELU"}, {"type": "Linear", "in_features": 8, "out_features": 1}]
GPT2 = {"vocab_size": 16, "n_positions": 32, "n_ctx": 32, "n_embd": 16, "n_layer": 1, "n_head": 2,
        "bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0}
VOCAB = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "hello": 4, "world": 5}
LLAMA = {"vocab_size": 32, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
         "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 128,
         "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "attention_bias": False,
         "mlp_bias": False, "tie_word_embeddings": False, "bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0}


class TestTensorRuntime(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.r = self.runtime.execute

    def model(self):
        return self.r("hf.sequential", {"layers": MLP}, {"seed": 42})

    def data(self):
        return self.r("hf.data", {"fields": {"x": [[-1], [0], [1]], "labels": [[-2], [0], [2]]}}, {})

    def optimizer(self, name="AdamW"):
        return self.r("hf.optimizer", {}, {"name": name, "lr": 0.01})

    def hf_model(self, task="causal-lm", model_type="gpt2", values=None):
        config = self.r("hf.config", {"values": values or GPT2}, {"model_type": model_type})
        return self.r("hf.model", config, {"task": task})

    def test_actual_multiple_parameters_and_adamw_updates(self):
        model, data = self.model(), self.data()
        self.assertEqual(model["model"]["parameter_count"], 33)
        trained = self.r("hf.train", {**model, **data, **self.optimizer()}, {"steps": 12, "batch_size": 3, "loss": "mse"})
        metrics = trained["metrics"]
        self.assertTrue(metrics["parameter_changed"])
        self.assertNotEqual(metrics["state_sha256_before"], metrics["state_sha256_after"])
        self.assertEqual(len(metrics["loss_history"]), 12)
        self.assertLess(metrics["loss_history"][-1], metrics["loss_history"][0])
        self.assertEqual(metrics["optimizer_state_tensors"], 15)
        self.assertEqual(self.r("hf.inspect", model, {})["report"]["parameter_count"], 33)

    def test_optimizer_moments_continue_between_train_nodes(self):
        inputs = {**self.model(), **self.data(), **self.optimizer()}
        self.r("hf.train", inputs, {"steps": 2})
        optimizer = next(iter(self.runtime.optimizers.values()))[0]
        previous_steps = [float(state["step"]) for state in optimizer.state.values()]
        self.r("hf.train", inputs, {"steps": 3})
        self.assertEqual(len(self.runtime.optimizers), 1)
        self.assertTrue(all(float(state["step"]) == old + 3 for state, old in zip(optimizer.state.values(), previous_steps, strict=True)))

    def test_all_optimizer_choices_execute_real_updates(self):
        for name in ("AdamW", "Adam", "SGD", "RMSprop"):
            with self.subTest(name=name):
                result = self.r("hf.train", {**self.model(), **self.data(), **self.optimizer(name)}, {"steps": 2})
                self.assertTrue(result["metrics"]["parameter_changed"])

    def test_evaluation_preserves_weights_and_forward_exposes_actual_shape(self):
        inputs = {**self.model(), **self.data()}
        evaluated = self.r("hf.evaluate", inputs, {"loss": "mse"})
        self.assertFalse(evaluated["metrics"]["parameter_changed"])
        self.assertEqual(self.r("hf.forward", inputs, {})["result"]["outputs"]["shape"], [3, 1])
        self.assertEqual(evaluated["metrics"]["state_sha256"], self.r("hf.inspect", inputs, {})["report"]["state_sha256"])

    def test_safe_checkpoint_roundtrip_preserves_exact_tensor_bytes(self):
        model = self.model()
        trained = self.r("hf.train", {**model, **self.data(), **self.optimizer()}, {"steps": 3})
        exported = self.r("hf.checkpoint", model, {})
        raw = base64.b64decode(exported["artifact"]["data_base64"])
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertEqual(set(archive.namelist()), {"model.safetensors", "recipe.json", "manifest.json"})
        loaded = self.r("hf.load", exported["artifact"], {})
        self.assertNotEqual(model["model"]["handle"], loaded["model"]["handle"])
        self.assertEqual(self.r("hf.inspect", loaded, {})["report"]["state_sha256"], trained["metrics"]["state_sha256_after"])

    def test_checkpoint_rejects_zip_traversal_duplicate_members_and_hash_tamper(self):
        exported = self.r("hf.checkpoint", self.model(), {})["artifact"]
        raw = base64.b64decode(exported["data_base64"])
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        for mutation in ("traversal", "duplicate", "tamper"):
            with self.subTest(mutation=mutation):
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, "w") as archive:
                    for name, data in contents.items():
                        archive.writestr(name, data + b" " if mutation == "tamper" and name == "recipe.json" else data)
                    if mutation == "traversal":
                        archive.writestr("../escaped.py", b"print('never')")
                    elif mutation == "duplicate":
                        archive.writestr("extra", b"unexpected")
                with self.assertRaises(ValueError):
                    self.r("hf.load", {"data_base64": base64.b64encode(stream.getvalue()).decode("ascii")}, {})

    def test_parameter_budget_checked_on_meta_before_cpu_allocation(self):
        with self.assertRaisesRegex(ValueError, "parameters.*budget"):
            self.r("hf.sequential", {"layers": [{"type": "Linear", "in_features": 10000, "out_features": 10000}]}, {})
        self.assertEqual(self.runtime.models, {})

    def test_dataset_finite_shape_and_element_budgets(self):
        invalid = [{"x": [[1], [2, 3]]}, {"x": [[1]], "labels": [[1], [2]]}, {"x": [[float("inf")]]},
                   {"x": [[1e100]]}, {"x": []}, {"x": [["bad"]]}]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises((ValueError, TypeError)):
                self.r("hf.data", {"fields": fields}, {})
        bounded = Runtime({"max_tensor_elements": 2})
        with self.assertRaisesRegex(ValueError, "element budget"):
            bounded.execute("hf.data", {"fields": {"x": [[1, 2, 3]]}}, {})

    def test_loss_rejects_implicit_broadcasting_and_fractional_classes(self):
        model = self.model()
        data = self.r("hf.data", {"fields": {"x": [[1], [2]], "labels": [1, 2]}}, {})
        with self.assertRaisesRegex(ValueError, "exactly the output shape"):
            self.r("hf.evaluate", {**model, **data}, {"loss": "mse"})
        classifier = self.r("hf.sequential", {"layers": [{"type": "Linear", "in_features": 1, "out_features": 2}]}, {})
        data = self.r("hf.data", {"fields": {"x": [[1], [2]], "labels": [0.5, 1]}}, {})
        with self.assertRaisesRegex(ValueError, "integer class"):
            self.r("hf.evaluate", {**classifier, **data}, {"loss": "cross_entropy"})

    def test_cross_entropy_classification_and_accuracy(self):
        model = self.r("hf.sequential", {"layers": [{"type": "Linear", "in_features": 1, "out_features": 2}]}, {})
        data = self.r("hf.data", {"fields": {"x": [[-1], [1]], "labels": [0, 1]}}, {})
        self.r("hf.train", {**model, **data, **self.optimizer()}, {"loss": "cross_entropy", "steps": 4})
        metrics = self.r("hf.evaluate", {**model, **data}, {"loss": "cross_entropy"})["metrics"]
        self.assertIn(metrics["accuracy"], (0, 0.5, 1))
        self.assertFalse(metrics["parameter_changed"])

    def test_custom_forward_residual_architecture_updates_all_layers(self):
        layers = {"input": {"type": "Linear", "in_features": 1, "out_features": 4},
                  "hidden": {"type": "Linear", "in_features": 4, "out_features": 4},
                  "output": {"type": "Linear", "in_features": 4, "out_features": 1}}
        code = "x = layers['input'](inputs['x'])\ny = relu(layers['hidden'](x))\nreturn layers['output'](x + y)"
        model = self.r("hf.custom", {"layers": layers}, {"code": code})
        result = self.r("hf.train", {**model, **self.data(), **self.optimizer()}, {"steps": 4})
        self.assertTrue(result["metrics"]["parameter_changed"])
        self.assertEqual(model["model"]["parameter_tensors"], 6)

    def test_custom_functional_dropout_respects_eval_mode(self):
        model = self.r("hf.custom", {"layers": {"input": {"type": "Linear", "in_features": 1, "out_features": 4}}},
                       {"code": "return dropout(layers['input'](inputs['x']), p=0.9)"})
        inputs = {**model, **self.data()}
        first = self.r("hf.forward", inputs, {})["result"]
        second = self.r("hf.forward", inputs, {})["result"]
        self.assertEqual(first, second)
        actual = self.runtime.models[model["model"]["handle"]][0]
        data = self.runtime.datasets[inputs["data"]["handle"]]
        expected = actual.layers["input"](data["x"]).reshape(-1).tolist()
        self.assertEqual(first["outputs"]["preview"], expected)

    def test_custom_ast_rejects_code_execution_reflection_and_mutation(self):
        invalid = ["import os\nreturn inputs['x']", "return __import__('os').system('echo bad')", "return inputs['x'].__class__",
                   "return eval('1')", "return exec('x=1')", "return [x for x in inputs]", "while True:\n x = 1\nreturn inputs['x']",
                   "inputs = {}\nreturn inputs", "layers = {}\nreturn layers", "x[0] = 1\nreturn inputs['x']",
                   "return layers['unknown'](inputs['x'])", "return (lambda: 1)()", "return globals()", "return inputs['x']\nx = 1"]
        for code in invalid:
            with self.subTest(code=code), self.assertRaises((ValueError, SyntaxError)):
                TensorProgram(code, {"input"}, 100)

    def test_layer_blocks_and_composition_have_real_matching_shapes(self):
        recipes = []
        cases: list[tuple[str, dict[str, Any]]] = [("hf.linear", {"in_features": 1, "out_features": 8}), ("hf.rmsnorm", {"normalized_shape": 8}),
                                                ("hf.activation", {"name": "GELU"}), ("hf.layernorm", {"normalized_shape": 8}),
                                                ("hf.linear", {"in_features": 8, "out_features": 1})]
        for name, params in cases:
            recipes.append(self.r(name, {}, params)["layer"])
        pair = self.r("hf.compose", {"a": recipes[0], "b": recipes[1]}, {})
        for layer in recipes[2:]:
            pair = self.r("hf.stack", {**pair, "layer": layer}, {})
        model = self.r("hf.sequential", pair, {})
        self.assertEqual(self.r("hf.forward", {**model, **self.data()}, {})["result"]["outputs"]["shape"], [3, 1])

    def test_embedding_causal_attention_and_custom_matmul(self):
        layers = {"embed": {"type": "Embedding", "num_embeddings": 16, "embedding_dim": 8},
                  "attention": {"type": "SelfAttention", "embed_dim": 8, "num_heads": 2, "causal": True},
                  "norm": {"type": "RMSNorm", "normalized_shape": 8}, "head": {"type": "Linear", "in_features": 8, "out_features": 4}}
        code = "x = layers['embed'](inputs['input_ids'])\ny = layers['attention'](x)\nreturn layers['head'](layers['norm'](x + y))"
        model = self.r("hf.custom", {"layers": layers}, {"code": code})
        data = self.r("hf.data", {"fields": {"input_ids": [[1, 2, 3]], "labels": [[[0, 0, 0, 0]] * 3]}}, {})
        result = self.r("hf.train", {**model, **data, **self.optimizer()}, {"steps": 2, "batch_size": 1})
        self.assertTrue(result["metrics"]["parameter_changed"])
        bad = {"type": "SelfAttention", "embed_dim": 7, "num_heads": 2}
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.r("hf.layer", {"recipe": bad}, {})

    def test_actual_hf_causal_lm_training_generation_and_tied_checkpoint(self):
        model = self.hf_model()
        self.assertEqual(model["model"]["parameter_count"], 4080)
        data = self.r("hf.data", {"fields": {"input_ids": [[2, 4, 5, 3]], "attention_mask": [[1, 1, 1, 1]], "labels": [[2, 4, 5, 3]]}}, {})
        metrics = self.r("hf.train", {**model, **data, **self.optimizer()}, {"steps": 2, "loss": "model", "batch_size": 1})["metrics"]
        self.assertTrue(metrics["parameter_changed"])
        generated = self.r("hf.generate", {**model, **data}, {"max_new_tokens": 3})["result"]
        self.assertEqual(generated["sequences"][0][:4], [2, 4, 5, 3])
        self.assertLessEqual(len(generated["sequences"][0]), 7)
        restored = self.r("hf.load", self.r("hf.checkpoint", model, {})["artifact"], {})
        self.assertEqual(self.r("hf.inspect", restored, {})["report"]["state_sha256"], metrics["state_sha256_after"])

    def test_native_llama_architecture_has_gqa_rope_swiglu_rmsnorm_and_causal_logits(self):
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Llama from config must remain offline")):
            model = self.hf_model(model_type="llama", values=LLAMA)
            self.assertEqual(model["model"]["architecture"], "LlamaForCausalLM")
            self.assertEqual(model["model"]["parameter_count"], 78144)
            self.assertEqual(model["model"]["parameter_tensors"], 21)
            actual = self.runtime.models[model["model"]["handle"]][0]
            torch = self.runtime.torch
            actual.eval()
            self.assertEqual(len(actual.model.layers), 2)
            layer = actual.model.layers[0]
            self.assertEqual(type(layer.self_attn).__name__, "LlamaAttention")
            self.assertEqual(tuple(layer.self_attn.q_proj.weight.shape), (64, 64))
            self.assertEqual(tuple(layer.self_attn.k_proj.weight.shape), (32, 64))
            self.assertEqual(tuple(layer.self_attn.v_proj.weight.shape), (32, 64))
            self.assertEqual(actual.config.num_attention_heads // actual.config.num_key_value_heads, 2)
            self.assertEqual(type(layer.input_layernorm).__name__, "LlamaRMSNorm")
            self.assertEqual(type(layer.post_attention_layernorm).__name__, "LlamaRMSNorm")
            self.assertEqual(type(actual.model.norm).__name__, "LlamaRMSNorm")
            self.assertEqual(type(actual.model.rotary_emb).__name__, "LlamaRotaryEmbedding")
            expected_frequencies = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
            self.assertTrue(torch.equal(actual.model.rotary_emb.inv_freq, expected_frequencies))
            hidden = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64) / 32
            with torch.no_grad():
                expected_norm = layer.input_layernorm.weight * hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
                self.assertTrue(torch.allclose(layer.input_layernorm(hidden), expected_norm))
                expected_mlp = layer.mlp.down_proj(torch.nn.functional.silu(layer.mlp.gate_proj(hidden)) * layer.mlp.up_proj(hidden))
                self.assertTrue(torch.equal(layer.mlp(hidden), expected_mlp))
                prefix = torch.tensor([[2, 4, 5, 6]])
                changed_future = torch.tensor([[2, 4, 5, 7]])
                first = actual(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=False).logits
                second = actual(input_ids=changed_future, attention_mask=torch.ones_like(prefix), use_cache=False).logits
            self.assertEqual(list(first.shape), [1, 4, 32])
            self.assertTrue(torch.equal(first[:, :3], second[:, :3]), "Future tokens must not alter prefix logits")
            self.assertFalse(torch.equal(first[:, 3], second[:, 3]))
            report = self.r("hf.inspect", model, {})["report"]
            modules = {item["name"]: item for item in report["modules"]}
            self.assertEqual(report["module_count"], len(list(actual.named_modules())))
            self.assertFalse(report["modules_truncated"])
            self.assertEqual(modules[""]["total_parameter_count"], 78144)
            self.assertEqual(modules["model.layers.0.self_attn.k_proj"]["direct_parameter_count"], 2048)
            self.assertEqual(modules["model.layers.0.self_attn"]["parent"], "model.layers.0")
            self.assertIn("model.layers.0.self_attn.q_proj", modules["model.layers.0.self_attn"]["children"])
            self.assertEqual(report["config"]["rope_parameters"], {"rope_type": "default", "rope_theta": 10000.0})
            self.assertEqual(report["config"]["head_dim"], 16)

    def test_module_inspection_bounds_output_and_reports_exact_total_without_changing_weights(self):
        model = self.r("hf.sequential", {"layers": [{"type": "SelfAttention", "embed_dim": 2, "num_heads": 1}] * 128}, {})
        before = self.runtime._hash(self.runtime.models[model["model"]["handle"]][0])
        report = self.r("hf.inspect", model, {})["report"]
        self.assertEqual(len(report["modules"]), 256)
        self.assertEqual(report["module_count"], 385)
        self.assertTrue(report["modules_truncated"])
        self.assertEqual(report["state_sha256"], before)
        self.assertEqual(report["modules"][0]["total_parameter_count"], model["model"]["parameter_count"])
        self.assertEqual(sum(item["direct_parameter_count"] for item in report["modules"][:4]), 24)

    def test_llama_rejects_invalid_gqa_and_rotary_geometry_before_model_allocation(self):
        for changed, message in (({"num_key_value_heads": 3}, "divisible"),
                                 ({"num_key_value_heads": 0}, "num_key_value_heads"),
                                 ({"num_attention_heads": 0}, "num_attention_heads"),
                                 ({"head_dim": 15}, "even"),
                                 ({"hidden_size": 60}, "even"),
                                 ({"hidden_size": 62}, "divisible"),
                                 ({"intermediate_size": 0}, "intermediate_size")):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, message):
                self.hf_model(model_type="llama", values={**LLAMA, **changed})
        self.assertEqual(self.runtime.models, {})

    def test_actual_hf_fast_tokenizer_padding_mask_and_causal_labels(self):
        tokenizer = self.r("hf.tokenizer", {"vocab": VOCAB}, {})
        self.assertEqual(tokenizer["tokenizer"]["padding_side"], "left")
        data = self.r("hf.tokenize", {**tokenizer, "texts": ["HELLO world", "hello"]}, {"labels": "causal"})
        fields = self.runtime.datasets[data["data"]["handle"]]
        self.assertEqual(fields["input_ids"].tolist(), [[4, 5], [0, 4]])
        self.assertEqual(fields["attention_mask"].tolist(), [[1, 1], [0, 1]])
        self.assertEqual(fields["labels"].tolist(), [[4, 5], [-100, 4]])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.r("hf.tokenize", {**tokenizer, "texts": []}, {})

    def test_actual_hf_text_generation_pipeline(self):
        model = self.hf_model()
        tokenizer = self.r("hf.tokenizer", {"vocab": VOCAB}, {})
        result = self.r("hf.pipeline", {**model, **tokenizer, "texts": ["hello world"]}, {"task": "text-generation", "max_new_tokens": 2})
        self.assertTrue(result["result"]["outputs"])
        self.assertIn("generated_text", result["result"]["outputs"][0][0])

    def test_local_wordlevel_generation_decodes_full_token_sequence_without_merging_prompt_words(self):
        for model_type, values in (("gpt2", GPT2), ("llama", LLAMA)):
            with self.subTest(model_type=model_type):
                model = self.hf_model(model_type=model_type, values=values)
                tokenizer = self.r("hf.tokenizer", {"vocab": VOCAB}, {})
                actual_model = self.runtime.models[model["model"]["handle"]][0]
                actual_tokenizer = self.runtime.tokenizers[tokenizer["tokenizer"]["handle"]]
                vocab_size = values["vocab_size"]
                assert isinstance(vocab_size, int)
                # A real HF logits processor makes native generation deterministic;
                # generated words cross the prompt boundary in both batch examples.
                actual_model.generation_config.suppress_tokens = [index for index in range(vocab_size) if index != VOCAB["world"]]
                outputs = self.r("hf.pipeline", {**model, **tokenizer, "texts": ["HELLO world", "hello"]},
                                 {"task": "text-generation", "max_new_tokens": 2})["result"]["outputs"]
                self.assertEqual(len(outputs), 2)
                for records, expected_ids, expected_text in zip(outputs, ([4, 5, 5, 5], [4, 5, 5]),
                                                                ("hello world world world", "hello world world"), strict=True):
                    self.assertEqual(len(records), 1)
                    record = records[0]
                    self.assertEqual(record["generated_token_ids"], expected_ids)
                    self.assertEqual(record["generated_text"], actual_tokenizer.decode(expected_ids, skip_special_tokens=True))
                    self.assertEqual(record["generated_text"], expected_text)

    def test_pipeline_eval_is_repeatable_after_model_training(self):
        model = self.hf_model(task="sequence-classification", model_type="distilbert", values={
            "vocab_size": 16, "max_position_embeddings": 32, "n_layers": 1, "n_heads": 2,
            "dim": 8, "hidden_dim": 16, "num_labels": 2, "dropout": 0.8, "seq_classif_dropout": 0.8})
        tokenizer = self.r("hf.tokenizer", {"vocab": VOCAB}, {})
        actual_model = self.runtime.models[model["model"]["handle"]][0]
        actual_model.train()
        inputs = {**model, **tokenizer, "texts": ["hello world"]}
        first = self.r("hf.pipeline", inputs, {"task": "text-classification"})
        second = self.r("hf.pipeline", inputs, {"task": "text-classification"})
        self.assertEqual(first, second)
        self.assertFalse(actual_model.training)

    def test_fill_mask_uses_real_mask_token(self):
        model = self.hf_model(task="masked-lm", model_type="distilbert", values={
            "vocab_size": 16, "max_position_embeddings": 32, "n_layers": 1, "n_heads": 2, "dim": 8, "hidden_dim": 16})
        tokenizer = self.r("hf.tokenizer", {"vocab": {**VOCAB, "[MASK]": 6}}, {"lowercase": False})
        result = self.r("hf.pipeline", {**model, **tokenizer, "texts": ["hello [MASK]"]}, {"task": "fill-mask"})
        self.assertIn("score", result["result"]["outputs"][0])

    def test_token_classification_pipeline_normalizes_numpy_scores_to_json(self):
        model = self.hf_model(task="token-classification", model_type="distilbert", values={
            "vocab_size": 16, "max_position_embeddings": 32, "n_layers": 1, "n_heads": 2,
            "dim": 8, "hidden_dim": 16, "num_labels": 3})
        tokenizer = self.r("hf.tokenizer", {"vocab": VOCAB}, {})
        result = self.r("hf.pipeline", {**model, **tokenizer, "texts": ["hello world"]}, {"task": "token-classification"})
        self.assertTrue(result["result"]["outputs"])
        encoded = json.dumps(result, allow_nan=False)
        self.assertIn("score", encoded)

    def test_generation_normalizes_manual_right_padding_and_checks_context(self):
        model = self.hf_model()
        data = self.r("hf.data", {"fields": {"input_ids": [[2, 4, 0], [2, 4, 5]], "attention_mask": [[1, 1, 0], [1, 1, 1]]}}, {})
        result = self.r("hf.generate", {**model, **data}, {"max_new_tokens": 1})["result"]
        self.assertEqual(result["input_ids"], [[0, 2, 4], [2, 4, 5]])
        self.assertEqual(result["attention_mask"], [[0, 1, 1], [1, 1, 1]])
        with self.assertRaisesRegex(ValueError, "context window"):
            self.r("hf.generate", {**model, **data}, {"max_new_tokens": 32})

    def test_token_fields_reject_fractional_ids_nonbinary_and_empty_masks(self):
        for fields in ({"input_ids": [[1.5, 2]]}, {"input_ids": [[1, 2]], "attention_mask": [[1, 2]]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.r("hf.data", {"fields": fields}, {})
        data = self.r("hf.data", {"fields": {"input_ids": [[0, 0]], "attention_mask": [[0, 0]]}}, {})
        with self.assertRaisesRegex(ValueError, "unmasked token"):
            self.r("hf.generate", {**self.hf_model(), **data}, {"max_new_tokens": 1})

    def test_sequential_embedding_accepts_integral_x_and_rejects_fractional_ids(self):
        model = self.r("hf.sequential", {"layers": [{"type": "Embedding", "num_embeddings": 8, "embedding_dim": 4},
                                                    {"type": "Linear", "in_features": 4, "out_features": 2}]}, {})
        data = self.r("hf.data", {"fields": {"x": [[1, 2, 3]]}}, {})
        self.assertEqual(self.r("hf.forward", {**model, **data}, {})["result"]["outputs"]["shape"], [1, 3, 2])
        data = self.r("hf.data", {"fields": {"x": [[1.5, 2, 3]]}}, {})
        with self.assertRaisesRegex(ValueError, "integer token"):
            self.r("hf.forward", {**model, **data}, {})

    def test_pretrained_rejects_paths_and_floating_revision(self):
        for params in ({"repo_id": "C:/models", "revision": "a" * 40}, {"repo_id": "../models", "revision": "a" * 40},
                       {"repo_id": "org/model", "revision": "main"}, {"repo_id": "org/model", "revision": "a" * 39}):
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, "repo_id|revision"):
                self.r("hf.pretrained", {}, params)

    def test_cached_pretrained_real_weights_training_and_portable_checkpoint_without_network(self):
        from huggingface_hub import try_to_load_from_cache
        from safetensors.torch import load_file

        repo_id, revision = "hf-internal-testing/tiny-random-gpt2", "71034c5d8bde858ff824298bdedc65515b97d2b9"
        cached = try_to_load_from_cache(repo_id, "model.safetensors", revision=revision)
        if not isinstance(cached, str):
            self.skipTest("Optional pinned pretrained demonstration is not in the local Hub cache; no download attempted")
        params = {"repo_id": repo_id, "revision": revision}
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Cache-only nodes must not use network")):
            model = self.r("hf.pretrained", {}, params)
            self.assertEqual(model["model"]["parameter_count"], 111968)
            self.assertEqual(model["model"]["source"], params)
            actual = self.runtime.models[model["model"]["handle"]][0]
            self.assertTrue(self.runtime.torch.equal(actual.transformer.wte.weight, load_file(cached)["transformer.wte.weight"]))
            tokenizer = self.r("hf.auto_tokenizer", {}, params)
            self.assertEqual(tokenizer["tokenizer"]["pad_token_id"], 0)
            native_output = self.r("hf.pipeline", {**model, **tokenizer, "texts": ["hello"]},
                                   {"task": "text-generation", "max_new_tokens": 1})["result"]["outputs"][0][0]
            self.assertIn("generated_text", native_output)
            self.assertNotIn("generated_token_ids", native_output, "Cached tokenizers retain native HF pipeline postprocessing")
            data = self.r("hf.tokenize", {**tokenizer, "texts": ["hello world", "hello"]}, {"labels": "causal"})
            metrics = self.r("hf.train", {**model, **data, **self.optimizer()}, {"steps": 2, "batch_size": 2, "loss": "model"})["metrics"]
            self.assertTrue(metrics["parameter_changed"])
            exported = self.r("hf.checkpoint", model, {})
            restored = self.r("hf.load", exported["artifact"], {})
            self.assertEqual(self.r("hf.inspect", restored, {})["report"]["state_sha256"], metrics["state_sha256_after"])
            recipe = self.runtime.models[restored["model"]["handle"]][1]
            self.assertEqual(recipe["family"], "huggingface")
            self.assertEqual(recipe["source"], params)

    def test_hf_config_remote_code_fields_rejected(self):
        for key in ("auto_map", "trust_remote_code", "torch_dtype", "architectures", "_name_or_path"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                values = {**GPT2, key: {"AutoModel": "untrusted.Escape"} if key == "auto_map" else "bad"}
                self.hf_model(values=values)

    def test_handles_cannot_cross_worker_or_be_forged(self):
        model = self.model()
        other = Runtime()
        with self.assertRaisesRegex(ValueError, "same workflow"):
            other.execute("hf.inspect", model, {})
        with self.assertRaisesRegex(ValueError, "same workflow"):
            self.r("hf.inspect", {"model": {"kind": "model", "handle": "not-real"}}, {})

    def test_hash_supports_scalar_tensor_state(self):
        model = self.runtime.torch.nn.ParameterDict({"scalar": self.runtime.torch.nn.Parameter(self.runtime.torch.tensor(1.25))})
        self.assertEqual(len(self.runtime._hash(model)), 64)

    def test_invalid_training_budgets_and_optimizer_values_are_explicit(self):
        inputs = {**self.model(), **self.data(), **self.optimizer()}
        for params in ({"steps": 0}, {"steps": 2001}, {"batch_size": 129}, {"clip_grad_norm": -1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.r("hf.train", inputs, params)
        with self.assertRaisesRegex(ValueError, "learning rate"):
            self.r("hf.optimizer", {}, {"lr": -0.1})


class TestTransformersIntegration(unittest.TestCase):
    def test_real_llama_graph_trains_generates_and_reloads_identical_safe_weights(self):
        workspace = StudioWorkspace()
        self.addCleanup(workspace.close)
        before = workspace.snapshot()["runs"][0]["step"]
        specifications = [("config", "config", {"model_type": "llama", "values": LLAMA}),
                          ("model", "model", {"task": "causal-lm", "seed": 42}),
                          ("data", "data", {"fields": {"input_ids": [[2, 4, 5, 3], [2, 5, 4, 3]],
                                                       "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
                                                       "labels": [[2, 4, 5, 3], [2, 5, 4, 3]]}}),
                          ("optimizer", "optimizer", {"name": "AdamW", "lr": 0.003}),
                          ("train", "train", {"steps": 4, "batch_size": 2, "loss": "model"}),
                          ("inspect", "inspect", {}), ("generate", "generate", {"max_new_tokens": 3}),
                          ("checkpoint", "checkpoint", {}), ("load", "load", {}), ("restored", "inspect", {}),
                          ("regenerate", "generate", {"max_new_tokens": 3})]
        wires = [("config", "config", "model", "config"), ("model", "model", "train", "model"),
                 ("data", "data", "train", "data"), ("optimizer", "optimizer", "train", "optimizer"),
                 ("train", "model", "inspect", "model"), ("inspect", "model", "generate", "model"),
                 ("data", "data", "generate", "data"), ("generate", "model", "checkpoint", "model"),
                 ("checkpoint", "artifact", "load", "artifact"), ("load", "model", "restored", "model"),
                 ("restored", "model", "regenerate", "model"), ("data", "data", "regenerate", "data")]
        graph = {"name": "Native Llama offline integration", "nodes": [
            {"id": identity, "type": "hf." + kind, "position": {"x": index * 320, "y": 20}, "params": params}
            for index, (identity, kind, params) in enumerate(specifications)], "edges": [
            {"id": str(index), "source": source, "source_port": output, "target": target, "target_port": input_name}
            for index, (source, output, target, input_name) in enumerate(wires)]}
        identity = workspace.graphs.dispatch("graph.create", {"graph": graph})["graph_id"]
        job_id = workspace.graphs.dispatch("graph.execute", {"graph_id": identity})["job_id"]
        job = workspace.graphs.wait_job(job_id, 180)
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertEqual(len(job["nodes"]), 11)
        metrics = job["nodes"]["train"]["outputs"]["metrics"]
        self.assertTrue(metrics["parameter_changed"])
        self.assertLess(metrics["loss_history"][-1], metrics["loss_history"][0])
        self.assertEqual(metrics["optimizer_state_tensors"], 63)
        original = job["nodes"]["inspect"]["outputs"]["report"]
        restored = job["nodes"]["restored"]["outputs"]["report"]
        self.assertEqual(original["architecture"], "LlamaForCausalLM")
        self.assertEqual(original["parameter_count"], 78144)
        self.assertEqual(original["state_sha256"], restored["state_sha256"])
        self.assertEqual(original["config"], restored["config"])
        self.assertEqual(original["modules"], restored["modules"])
        self.assertEqual(job["nodes"]["generate"]["outputs"]["result"], job["nodes"]["regenerate"]["outputs"]["result"])
        descriptor = job["nodes"]["checkpoint"]["outputs"]["artifact"]
        raw, _, mime = workspace.graphs.artifact(descriptor["id"])
        self.assertEqual(mime, "application/zip")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertEqual(set(archive.namelist()), {"model.safetensors", "recipe.json", "manifest.json"})
            self.assertEqual(json.loads(archive.read("recipe.json"))["recipe"]["config"]["model_type"], "llama")
        self.assertEqual(workspace.snapshot()["runs"][0]["step"], before)

    def test_catalog_templates_and_examples_validate_without_execution(self):
        workspace = StudioWorkspace()
        self.addCleanup(workspace.close)
        catalog = workspace.graphs.catalog()
        self.assertTrue(catalog["transformers"]["available"])
        self.assertGreater(catalog["transformers"]["limits"]["max_parameters"], 1)
        node_ids = {node["id"] for node in definitions()}
        self.assertTrue({"hf.op_rmsnorm", "hf.op_rope", "hf.op_attention", "hf.op_swiglu",
                         "hf.op_custom", "hf.graph_model"}.issubset(node_ids))
        self.assertEqual(len(node_ids), len(definitions()))
        for template in templates():
            self.assertTrue(workspace.graphs.validate(template["graph"])["valid"], template["id"])
        folder = Path(__file__).resolve().parents[2] / "examples" / "transformers"
        for file in folder.glob("*.plugin.json"):
            workspace.graphs.dispatch("plugin.install", {"manifest": json.loads(file.read_text(encoding="utf-8"))})
        for file in folder.glob("*.workflow.json"):
            self.assertTrue(workspace.graphs.validate(json.loads(file.read_text(encoding="utf-8")))["valid"], file.name)
        self.assertEqual(workspace.graphs.snapshot()["jobs"], [])

    def test_real_graph_training_artifact_reload_and_no_scalar_workspace_effect(self):
        workspace = StudioWorkspace()
        self.addCleanup(workspace.close)
        before = workspace.snapshot()["runs"][0]["step"]
        graph = copy.deepcopy(templates()[0]["graph"])
        graph["nodes"] += [{"id": "load", "type": "hf.load", "position": {"x": 1600, "y": 20}, "params": {}},
                           {"id": "restored", "type": "hf.inspect", "position": {"x": 1900, "y": 20}, "params": {}}]
        graph["edges"] += [{"id": "checkpoint-load", "source": "checkpoint", "source_port": "artifact", "target": "load", "target_port": "artifact"},
                           {"id": "load-restored", "source": "load", "source_port": "model", "target": "restored", "target_port": "model"}]
        identity = workspace.graphs.dispatch("graph.create", {"graph": graph})["graph_id"]
        job_id = workspace.graphs.dispatch("graph.execute", {"graph_id": identity})["job_id"]
        job = workspace.graphs.wait_job(job_id, 180)
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertTrue(job["nodes"]["train"]["outputs"]["metrics"]["parameter_changed"])
        self.assertEqual(job["nodes"]["inspect"]["outputs"]["report"]["state_sha256"], job["nodes"]["restored"]["outputs"]["report"]["state_sha256"])
        descriptor = job["nodes"]["checkpoint"]["outputs"]["artifact"]
        raw, _, mime = workspace.graphs.artifact(descriptor["id"])
        self.assertEqual(mime, "application/zip")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertIn("model.safetensors", archive.namelist())
        self.assertEqual(workspace.snapshot()["runs"][0]["step"], before)
        saved = workspace.graphs.export_state()
        validated = GraphStudio.validate_state(saved)
        self.assertEqual(validated["artifacts"][0]["sha256"], descriptor["sha256"])

    def test_worker_cancellation_kills_and_reaps_actual_native_training(self):
        cancel = threading.Event()
        with patch.dict(os.environ, {"SPOBS_TRANSFORMERS_MAX_STEPS": "100000"}):
            worker = TransformersWorker(cancel)
            self.addCleanup(worker.close)
            model = worker.request("hf.sequential", {"layers": MLP}, {})
            data = worker.request("hf.data", {"fields": {"x": [[-1], [0], [1]], "labels": [[-2], [0], [2]]}}, {})
            optimizer = worker.request("hf.optimizer", {}, {})
            timer = threading.Timer(0.2, cancel.set)
            timer.start()
            self.addCleanup(timer.cancel)
            started = time.monotonic()
            with self.assertRaisesRegex(TransformersWorkerError, "cancelled"):
                worker.request("hf.train", {**model, **data, **optimizer}, {"steps": 100000})
            self.assertLess(time.monotonic() - started, 10)
            self.assertIsNotNone(worker.process.poll())
            self.assertFalse(worker._reader.is_alive())
            self.assertFalse(worker._error_reader.is_alive())

    def test_worker_streams_actual_training_progress_before_final_metrics(self):
        observed: list[dict[str, Any]] = []
        with TransformersWorker(on_progress=observed.append) as worker:
            model = worker.request("hf.sequential", {"layers": MLP}, {})
            data = worker.request("hf.data", {"fields": {"x": [[-1], [0], [1]], "labels": [[-2], [0], [2]]}}, {})
            optimizer = worker.request("hf.optimizer", {}, {})
            result = worker.request("hf.train", {**model, **data, **optimizer}, {"steps": 3})
            self.assertEqual([item["step"] for item in observed], [1, 2, 3])
            self.assertEqual([item["loss"] for item in observed], result["metrics"]["loss_history"])
            self.assertEqual([item["gradient_norm"] for item in observed], result["metrics"]["gradient_norms"])
        self.assertIsNotNone(worker.process.poll())


if __name__ == "__main__":
    unittest.main()
