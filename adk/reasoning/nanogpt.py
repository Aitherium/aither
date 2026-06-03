"""NanoGPT — Zero-dependency character-level transformer.

Pure Python autograd engine + transformer with LoRA hypernetwork adapters.
Based on Karpathy's microGPT, extended with:
- Async training (runs in worker thread)
- LoRA adapters for document-specific memory
- Anomaly detection via loss evaluation
- Text generation with temperature sampling

Usage:
    from adk.nanogpt import NanoGPT

    model = NanoGPT()
    await model.train(["hello world", "foo bar baz"], num_steps=200)
    loss = model.evaluate("hello")
    samples = await model.generate(num_samples=3)

    # LoRA adapter for a specific document
    await model.train_hypernetwork("doc1", "important content here")
    samples = await model.generate(doc_id="doc1")
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("adk.nanogpt")


# ─────────────────────────────────────────────────────────────────────────────
# Autograd engine (from karpathy/micrograd)
# ─────────────────────────────────────────────────────────────────────────────

class Value:
    """Scalar-valued autograd node."""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data + 1e-8), (self,), (1/(self.data + 1e-8),))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo: list[Value] = []
        visited: set[int] = set()
        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def linear(x: list, w: list) -> list:
    """Matrix-vector multiply: w @ x."""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits: list) -> list:
    max_val = max(val.data if isinstance(val, Value) else val for val in logits)
    exps = [
        (val - max_val).exp() if isinstance(val, Value) else math.exp(val - max_val)
        for val in logits
    ]
    total = sum(e.data if isinstance(e, Value) else e for e in exps)
    return [e / total for e in exps]


def rmsnorm(x: list) -> list:
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# ─────────────────────────────────────────────────────────────────────────────
# NanoGPT Model
# ─────────────────────────────────────────────────────────────────────────────

class NanoGPT:
    """Character-level transformer. Zero external dependencies.

    Architecture: 1-layer transformer with multi-head attention, RMSNorm,
    MLP with ReLU, and optional LoRA hypernetwork adapters.
    """

    def __init__(
        self,
        n_layer: int = 1,
        n_embd: int = 16,
        block_size: int = 16,
        n_head: int = 4,
    ):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.vocab_size = 27  # default, overridden by init_model
        self.uchars: list[str] = []
        self.BOS = 26
        self.state_dict: dict[str, list] = {}
        self.params: list[Value] = []
        self.is_trained = False
        self.training_status = "idle"
        self.training_progress = 0.0
        self.current_loss = 0.0

        # LoRA hypernetwork state
        self.hyper_adapters: dict[str, dict] = {}
        self.active_doc_id: str | None = None
        self.rank = 2

    def init_model(self, vocab_size: int):
        """Initialize model weights for the given vocabulary size."""
        self.vocab_size = vocab_size
        matrix = lambda nout, nin, std=0.08: [
            [Value(random.gauss(0, std)) for _ in range(nin)]
            for _ in range(nout)
        ]
        self.state_dict = {
            'wte': matrix(vocab_size, self.n_embd),
            'wpe': matrix(self.block_size, self.n_embd),
            'lm_head': matrix(vocab_size, self.n_embd),
        }
        for i in range(self.n_layer):
            self.state_dict[f'layer{i}.attn_wq'] = matrix(self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.attn_wk'] = matrix(self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.attn_wv'] = matrix(self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.attn_wo'] = matrix(self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * self.n_embd, self.n_embd)
            self.state_dict[f'layer{i}.mlp_fc2'] = matrix(self.n_embd, 4 * self.n_embd)
        self.params = [p for mat in self.state_dict.values() for row in mat for p in row]

    def forward(self, token_id: int, pos_id: int, keys: list, values: list) -> list:
        """Forward pass. Returns logits over vocabulary."""
        tok_emb = self.state_dict['wte'][token_id]
        pos_emb = self.state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(self.n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, self.state_dict[f'layer{li}.attn_wq'])
            k = linear(x, self.state_dict[f'layer{li}.attn_wk'])
            v = linear(x, self.state_dict[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)

            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs + self.head_dim]
                k_h = [ki[hs:hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs + self.head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / (self.head_dim**0.5)
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)

            x = linear(x_attn, self.state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]

            x_residual = x
            x = rmsnorm(x)
            x_fc1 = linear(x, self.state_dict[f'layer{li}.mlp_fc1'])
            x_fc1 = [xi.relu() for xi in x_fc1]
            x_fc2 = linear(x_fc1, self.state_dict[f'layer{li}.mlp_fc2'])

            # LoRA adapter injection
            if self.active_doc_id and self.active_doc_id in self.hyper_adapters:
                lora_a = self.hyper_adapters[self.active_doc_id][f'layer{li}.lora_a']
                lora_b = self.hyper_adapters[self.active_doc_id][f'layer{li}.lora_b']
                x_lora_a = linear(x_fc1, lora_a)
                x_lora_b = linear(x_lora_a, lora_b)
                x_fc2 = [a + b for a, b in zip(x_fc2, x_lora_b)]

            x = [a + b for a, b in zip(x_fc2, x_residual)]

        return linear(x, self.state_dict['lm_head'])

    # ─── Training ──────────────────────────────────────────────────────

    def _train_sync(
        self,
        docs: list[str],
        num_steps: int,
        update_callback: Callable | None,
    ):
        """Synchronous training loop (runs in worker thread)."""
        lr, beta1, beta2, eps = 0.01, 0.85, 0.99, 1e-8
        m = [0.0] * len(self.params)
        v = [0.0] * len(self.params)

        for step in range(num_steps):
            doc = docs[step % len(docs)]
            tokens = (
                [self.BOS]
                + [self.uchars.index(ch) if ch in self.uchars else self.BOS for ch in doc]
                + [self.BOS]
            )
            n = min(self.block_size, len(tokens) - 1)
            if n <= 0:
                continue

            keys = [[] for _ in range(self.n_layer)]
            values = [[] for _ in range(self.n_layer)]
            losses = []
            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                logits = self.forward(token_id, pos_id, keys, values)
                probs = softmax(logits)
                losses.append(-probs[target_id].log())

            loss = (1 / n) * sum(losses)
            loss.backward()

            lr_t = lr * (1 - step / num_steps)
            for i, p in enumerate(self.params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
                m_hat = m[i] / (1 - beta1 ** (step + 1))
                v_hat = v[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
                p.grad = 0

            self.current_loss = loss.data
            self.training_progress = (step + 1) / num_steps

            if update_callback and ((step + 1) % 10 == 0 or step == 0):
                update_callback(step, num_steps, loss.data)

    async def train(
        self,
        docs: list[str],
        num_steps: int = 500,
        update_callback: Callable | None = None,
    ):
        """Train the model on a list of text documents.

        Args:
            docs: Training documents (character-level tokenization)
            num_steps: Number of training steps
            update_callback: Optional fn(step, total, loss) called every 10 steps
        """
        if not docs:
            self.training_status = "failed"
            return

        self.training_status = "training"
        self.is_trained = False

        if not self.uchars:
            self.uchars = sorted(set(''.join(docs)))
            self.BOS = len(self.uchars)
            self.init_model(len(self.uchars) + 1)

        await asyncio.to_thread(self._train_sync, docs, num_steps, update_callback)

        self.is_trained = True
        self.training_status = "completed"
        logger.info("NanoGPT training complete: %d steps, loss=%.4f", num_steps, self.current_loss)

    # ─── Hypernetwork (LoRA) ───────────────────────────────────────────

    def _train_hypernetwork_sync(self, doc_id: str, doc_content: str, num_steps: int):
        matrix = lambda nout, nin, std=0.08: [
            [Value(random.gauss(0, std)) for _ in range(nin)]
            for _ in range(nout)
        ]

        if doc_id not in self.hyper_adapters:
            adapters = {}
            for i in range(self.n_layer):
                adapters[f'layer{i}.lora_a'] = matrix(self.rank, 4 * self.n_embd, std=0.01)
                adapters[f'layer{i}.lora_b'] = matrix(self.n_embd, self.rank, std=0.0)
            self.hyper_adapters[doc_id] = adapters

        adapters = self.hyper_adapters[doc_id]
        adapter_params = [p for mat in adapters.values() for row in mat for p in row]
        self.active_doc_id = doc_id

        tokens = (
            [self.BOS]
            + [self.uchars.index(ch) if ch in self.uchars else self.BOS for ch in doc_content]
            + [self.BOS]
        )
        n = min(self.block_size, len(tokens) - 1)
        if n <= 0:
            return

        for step in range(num_steps):
            keys = [[] for _ in range(self.n_layer)]
            values = [[] for _ in range(self.n_layer)]
            losses = []
            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                logits = self.forward(token_id, pos_id, keys, values)
                probs = softmax(logits)
                losses.append(-probs[target_id].log())

            loss = (1 / n) * sum(losses)
            loss.backward()

            for p in adapter_params:
                p.data -= 0.05 * p.grad
                p.grad = 0

    async def train_hypernetwork(self, doc_id: str, doc_content: str, num_steps: int = 100):
        """Compile a document into LoRA adapters (durable memory).

        The base model must be trained first.
        """
        if not self.is_trained:
            self.training_status = "failed_no_base_model"
            return

        self.training_status = "training_hypernetwork"
        await asyncio.to_thread(self._train_hypernetwork_sync, doc_id, doc_content, num_steps)
        self.training_status = "completed"
        logger.info("Hypernetwork adapter trained for doc '%s'", doc_id)

    # ─── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, text: str) -> float:
        """Evaluate loss on text. High loss = anomaly/unfamiliar content."""
        if not self.is_trained or not text:
            return 0.0

        tokens = (
            [self.BOS]
            + [self.uchars.index(ch) if ch in self.uchars else self.BOS for ch in text]
            + [self.BOS]
        )
        n = min(self.block_size, len(tokens) - 1)
        if n <= 0:
            return 0.0

        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = self.forward(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t.data if isinstance(loss_t, Value) else loss_t)

        return sum(losses) / n

    # ─── Generation ────────────────────────────────────────────────────

    def _generate_sync(
        self,
        num_samples: int,
        temperature: float,
        doc_id: str | None,
    ) -> list[str]:
        previous = self.active_doc_id
        if doc_id and doc_id in self.hyper_adapters:
            self.active_doc_id = doc_id

        results = []
        for _ in range(num_samples):
            keys = [[] for _ in range(self.n_layer)]
            values = [[] for _ in range(self.n_layer)]
            token_id = self.BOS
            sample = []
            for pos_id in range(self.block_size):
                logits = self.forward(token_id, pos_id, keys, values)
                log_data = [l.data for l in logits]
                probs = softmax([l / temperature for l in log_data])
                probs_data = [p.data if isinstance(p, Value) else p for p in probs]

                if any(math.isnan(p) for p in probs_data) or sum(probs_data) <= 0:
                    token_id = random.choice(range(self.vocab_size - 1))
                else:
                    token_id = random.choices(range(self.vocab_size), weights=probs_data)[0]

                if token_id == self.BOS:
                    break
                sample.append(self.uchars[token_id])

            results.append(''.join(sample))

        self.active_doc_id = previous
        return results

    async def generate(
        self,
        num_samples: int = 5,
        temperature: float = 0.5,
        doc_id: str | None = None,
    ) -> list[str]:
        """Generate text samples.

        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature (lower = more deterministic)
            doc_id: Optional LoRA adapter ID for document-specific generation
        """
        if not self.is_trained:
            return []
        return await asyncio.to_thread(self._generate_sync, num_samples, temperature, doc_id)

    # ─── Persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save model weights to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "uchars": self.uchars,
            "BOS": self.BOS,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "n_head": self.n_head,
            "state_dict": {
                k: [[p.data for p in row] for row in mat]
                for k, mat in self.state_dict.items()
            },
            "hyper_adapters": {
                doc_id: {
                    k: [[p.data for p in row] for row in mat]
                    for k, mat in adapters.items()
                }
                for doc_id, adapters in self.hyper_adapters.items()
            },
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: str | Path):
        """Load model weights from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.uchars = data["uchars"]
        self.BOS = data["BOS"]
        self.vocab_size = data["vocab_size"]
        self.n_layer = data.get("n_layer", 1)
        self.n_embd = data.get("n_embd", 16)
        self.block_size = data.get("block_size", 16)
        self.n_head = data.get("n_head", 4)
        self.head_dim = self.n_embd // self.n_head

        self.state_dict = {
            k: [[Value(p) for p in row] for row in mat]
            for k, mat in data["state_dict"].items()
        }
        self.params = [p for mat in self.state_dict.values() for row in mat for p in row]

        if "hyper_adapters" in data:
            self.hyper_adapters = {
                doc_id: {
                    k: [[Value(p) for p in row] for row in mat]
                    for k, mat in adapters.items()
                }
                for doc_id, adapters in data["hyper_adapters"].items()
            }

        self.is_trained = True
        self.training_status = "loaded"

    # ─── Status ────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "is_trained": self.is_trained,
            "training_status": self.training_status,
            "training_progress": self.training_progress,
            "current_loss": self.current_loss,
            "vocab_size": self.vocab_size,
            "n_params": len(self.params),
            "n_adapters": len(self.hyper_adapters),
            "architecture": {
                "n_layer": self.n_layer,
                "n_embd": self.n_embd,
                "block_size": self.block_size,
                "n_head": self.n_head,
            },
        }
