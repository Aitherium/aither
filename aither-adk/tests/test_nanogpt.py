"""Tests for NanoGPT — zero-dependency character-level transformer."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from adk.nanogpt import NanoGPT, Value, softmax, rmsnorm, linear


# ─── Autograd Engine ──────────────────────────────────────────────────────────

class TestValue:
    def test_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_mul(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_pow(self):
        a = Value(3.0)
        c = a ** 2
        assert abs(c.data - 9.0) < 1e-6

    def test_relu(self):
        assert Value(5.0).relu().data == 5.0
        assert Value(-3.0).relu().data == 0.0

    def test_log(self):
        v = Value(1.0).log()
        assert abs(v.data - 0.0) < 1e-6

    def test_exp(self):
        v = Value(0.0).exp()
        assert abs(v.data - 1.0) < 1e-6

    def test_neg(self):
        v = -Value(5.0)
        assert v.data == -5.0

    def test_sub(self):
        c = Value(5.0) - Value(3.0)
        assert c.data == 2.0

    def test_div(self):
        c = Value(6.0) / Value(2.0)
        assert abs(c.data - 3.0) < 1e-6

    def test_backward_simple(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        assert abs(a.grad - 3.0) < 1e-6  # dc/da = b
        assert abs(b.grad - 2.0) < 1e-6  # dc/db = a

    def test_backward_chain(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        d = c + Value(1.0)
        d.backward()
        assert abs(a.grad - 3.0) < 1e-6

    def test_radd(self):
        c = 5 + Value(3.0)
        assert c.data == 8.0

    def test_rmul(self):
        c = 5 * Value(3.0)
        assert c.data == 15.0

    def test_rsub(self):
        c = 5 - Value(3.0)
        assert c.data == 2.0


# ─── Math Helpers ─────────────────────────────────────────────────────────────

class TestMathHelpers:
    def test_softmax_sums_to_one(self):
        logits = [Value(1.0), Value(2.0), Value(3.0)]
        probs = softmax(logits)
        total = sum(p.data for p in probs)
        assert abs(total - 1.0) < 1e-6

    def test_softmax_max_is_largest(self):
        logits = [Value(1.0), Value(5.0), Value(2.0)]
        probs = softmax(logits)
        assert probs[1].data > probs[0].data
        assert probs[1].data > probs[2].data

    def test_rmsnorm_nonzero(self):
        x = [Value(1.0), Value(2.0), Value(3.0)]
        normed = rmsnorm(x)
        assert all(isinstance(v, Value) for v in normed)
        assert any(v.data != 0.0 for v in normed)

    def test_linear(self):
        x = [Value(1.0), Value(2.0)]
        w = [[Value(1.0), Value(0.0)], [Value(0.0), Value(1.0)]]
        out = linear(x, w)
        assert abs(out[0].data - 1.0) < 1e-6
        assert abs(out[1].data - 2.0) < 1e-6


# ─── NanoGPT Model ────────────────────────────────────────────────────────────

class TestNanoGPTModel:
    def test_init_defaults(self):
        model = NanoGPT()
        assert model.n_layer == 1
        assert model.n_embd == 16
        assert model.block_size == 16
        assert model.is_trained is False

    def test_init_model(self):
        model = NanoGPT()
        model.init_model(vocab_size=10)
        assert model.vocab_size == 10
        assert len(model.params) > 0
        assert 'wte' in model.state_dict
        assert 'lm_head' in model.state_dict

    def test_forward(self):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        model.init_model(vocab_size=5)
        keys = [[] for _ in range(model.n_layer)]
        values = [[] for _ in range(model.n_layer)]
        logits = model.forward(0, 0, keys, values)
        assert len(logits) == 5  # vocab_size

    def test_status(self):
        model = NanoGPT()
        status = model.status()
        assert status["is_trained"] is False
        assert status["n_params"] == 0
        assert "architecture" in status


# ─── Training ─────────────────────────────────────────────────────────────────

class TestNanoGPTTraining:
    @pytest.mark.asyncio
    async def test_train_basic(self):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        await model.train(["hello", "world"], num_steps=5)
        assert model.is_trained is True
        assert model.training_status == "completed"
        assert model.current_loss > 0

    @pytest.mark.asyncio
    async def test_train_empty_docs(self):
        model = NanoGPT()
        await model.train([], num_steps=5)
        assert model.training_status == "failed"
        assert model.is_trained is False

    @pytest.mark.asyncio
    async def test_train_progress(self):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        steps_seen = []
        def callback(step, total, loss):
            steps_seen.append(step)
        await model.train(["test data"], num_steps=20, update_callback=callback)
        assert len(steps_seen) > 0

    @pytest.mark.asyncio
    async def test_train_builds_vocab(self):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        await model.train(["abc", "def"], num_steps=3)
        assert len(model.uchars) >= 6  # a,b,c,d,e,f


# ─── Evaluation ───────────────────────────────────────────────────────────────

class TestNanoGPTEvaluation:
    @pytest.mark.asyncio
    async def test_evaluate_trained(self):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        await model.train(["hello world"], num_steps=10)
        loss = model.evaluate("hello")
        assert loss > 0

    def test_evaluate_untrained(self):
        model = NanoGPT()
        loss = model.evaluate("test")
        assert loss == 0.0

    def test_evaluate_empty(self):
        model = NanoGPT()
        model.is_trained = True
        loss = model.evaluate("")
        assert loss == 0.0


# ─── Generation ───────────────────────────────────────────────────────────────

class TestNanoGPTGeneration:
    @pytest.mark.asyncio
    async def test_generate_trained(self):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        await model.train(["hello world test"], num_steps=20)
        samples = await model.generate(num_samples=3)
        assert len(samples) == 3
        assert all(isinstance(s, str) for s in samples)

    @pytest.mark.asyncio
    async def test_generate_untrained(self):
        model = NanoGPT()
        samples = await model.generate(num_samples=3)
        assert samples == []


# ─── Hypernetwork (LoRA) ──────────────────────────────────────────────────────

class TestNanoGPTHypernetwork:
    @pytest.mark.asyncio
    async def test_train_hypernetwork(self):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        await model.train(["base training data"], num_steps=10)
        await model.train_hypernetwork("doc1", "specialized content", num_steps=5)
        assert "doc1" in model.hyper_adapters
        assert model.training_status == "completed"

    @pytest.mark.asyncio
    async def test_train_hypernetwork_no_base(self):
        model = NanoGPT()
        await model.train_hypernetwork("doc1", "content", num_steps=5)
        assert model.training_status == "failed_no_base_model"

    @pytest.mark.asyncio
    async def test_generate_with_adapter(self):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        await model.train(["base data here"], num_steps=10)
        await model.train_hypernetwork("doc1", "specialized", num_steps=5)
        samples = await model.generate(num_samples=2, doc_id="doc1")
        assert len(samples) == 2


# ─── Persistence ──────────────────────────────────────────────────────────────

class TestNanoGPTPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        await model.train(["hello world"], num_steps=5)
        save_path = tmp_path / "model.json"
        model.save(save_path)
        assert save_path.exists()

        model2 = NanoGPT()
        model2.load(save_path)
        assert model2.is_trained is True
        assert model2.vocab_size == model.vocab_size
        assert len(model2.params) == len(model.params)

    @pytest.mark.asyncio
    async def test_save_with_adapters(self, tmp_path):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        await model.train(["data here"], num_steps=5)
        await model.train_hypernetwork("doc1", "specialized", num_steps=3)
        save_path = tmp_path / "model_lora.json"
        model.save(save_path)

        model2 = NanoGPT()
        model2.load(save_path)
        assert "doc1" in model2.hyper_adapters

    @pytest.mark.asyncio
    async def test_evaluate_after_load(self, tmp_path):
        model = NanoGPT(n_layer=1, n_embd=8, block_size=8, n_head=2)
        await model.train(["test text"], num_steps=5)
        original_loss = model.evaluate("test")

        save_path = tmp_path / "model.json"
        model.save(save_path)
        model2 = NanoGPT()
        model2.load(save_path)
        loaded_loss = model2.evaluate("test")
        assert abs(original_loss - loaded_loss) < 0.01


# ─── Export ────────────────────────────────────────────────────────────────────

class TestExport:
    def test_nanogpt_in_adk(self):
        import adk
        assert hasattr(adk, "NanoGPT")
