"""Semantic query encoder: MiniLM-L6-v2 via ONNX Runtime (no torch).

Produces the SAME 384-dim vectors as SentenceTransformer('all-MiniLM-L6-v2')
(mean-pooled last hidden state, L2-normalized), so the existing Qdrant
collection stays fully compatible — no re-index needed.

Assets (downloaded once into artifacts/semantic/):
    tokenizer.json  — HF fast-tokenizer file
    model.onnx      — exported MiniLM encoder

Parity with sentence-transformers is asserted by tests/unit/test_semantic_parity.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SEMANTIC_DIR = Path(os.environ.get("SEMANTIC_ASSETS_DIR", "artifacts/semantic"))


class OnnxEncoder:
    """Minimal ONNX MiniLM text encoder (mean pooling + L2 norm)."""

    def __init__(self, model_dir: Path = SEMANTIC_DIR) -> None:
        from tokenizers import Tokenizer
        import onnxruntime as ort

        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # The exported tokenizer.json ships with padding/truncation enabled; disable
        # both so we control masking explicitly (pad tokens must NOT be attended).
        self.tok.no_padding()
        self.tok.no_truncation()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # keep RSS low on free tier
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.max_len = 128

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        enc = self.tok.encode_batch(texts, add_special_tokens=True)
        max_len = min(self.max_len, max((len(e.ids) for e in enc), default=1))
        input_ids = np.zeros((len(enc), max_len), dtype=np.int64)
        attention = np.zeros((len(enc), max_len), dtype=np.int64)
        token_type = np.zeros((len(enc), max_len), dtype=np.int64)
        for i, e in enumerate(enc):
            ids = e.ids[:max_len]                      # pure ids (padding disabled)
            type_ids = e.type_ids[:max_len]
            real = len(ids)
            input_ids[i, :real] = ids
            token_type[i, :real] = type_ids
            attention[i, :real] = 1                    # only REAL tokens attended
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": token_type,
        }
        # Only feed inputs the model actually declares.
        feeds = {k: v for k, v in feeds.items()
                 if k in {inp.name for inp in self.session.get_inputs()}}
        out = self.session.run(None, feeds)[0]  # (B, T, H) last_hidden_state
        mask = attention[:, :, None].astype(np.float32)
        summed = (out * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        emb = summed / counts
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)  # sentence-transformers normalize
        return emb.astype(np.float32)


def get_encoder() -> OnnxEncoder | None:
    """Lazily build the encoder; returns None if assets are absent.
    Not cached: a None result must be retryable after the assets are
    downloaded by api.main._get_embedder (which caches on state.embedder)."""
    if not (SEMANTIC_DIR / "model.onnx").exists():
        return None
    return OnnxEncoder()
