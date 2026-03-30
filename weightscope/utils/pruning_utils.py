"""
WeightScope - Pruning Utilities
================================
Shared code for the Vocabulary Pruning and Transformer Pruning plugins.

Contents
--------
Architecture detection
    ``detect_architecture(config)``  →  ``ArchitectureProfile``

    Reads ``config.json`` and returns a dataclass describing which tensor
    names carry embeddings, attention heads, and FFN neurons for that model
    family.  Supports:

    * Llama-style decoder-only  (Llama, Mistral, Qwen2/3, Gemma, Phi-3, MiniCPM,
                                  Falcon, OpenLM, Yi, Deepseek, …)
    * BERT/RoBERTa encoder-only  (BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, …)
    * GPT-2 style               (GPT-2, GPT-Neo, CodeGen, …)
    * GPT-NeoX / Pythia         (fused QKV tensor)
    * Phi-2 / Phi-1.5           (dense projection naming)
    * OPT                       (decoder with slightly different prefix)
    * T5 / BART encoder-decoder  (partial — embedding only)
    * Unknown                   (falls back to pattern scanning)

Safetensors I/O
    ``load_shard_tensors(shard_paths)``     →  ``{name: np.ndarray}``
    ``save_model_dir(tensors, out_dir, shard_size_gb)``

Output directory naming
    ``make_output_name(model_id, suffix)``  →  e.g. ``amd--AMD-Llama-135m-vocab_pr``

Config / tokenizer helpers
    ``copy_model_support_files(src_dir, dst_dir, new_vocab_size=None)``

---

Copyright (C) 2026 Bryan K Reinhart & BeySoft

This file is part of WeightScope.

WeightScope is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

WeightScope is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public
License along with WeightScope. If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


# ─── Architecture profile ────────────────────────────────────────────────────

@dataclass
class ArchitectureProfile:
    """
    Describes the tensor naming convention for one model family.

    All patterns use ``{layer}`` as a placeholder for the layer index.
    """
    family: str                 # human-readable name, e.g. "llama"

    # Embedding / LM head
    embed_weight: str           # e.g. "model.embed_tokens.weight"
    lm_head_weight: str         # e.g. "lm_head.weight"
    tie_word_embeddings: bool   # True → lm_head shares embed weights

    # Attention — patterns; {layer} is replaced by the layer index
    q_proj_pat:   str           # query projection
    k_proj_pat:   str           # key projection
    v_proj_pat:   str           # value projection
    o_proj_pat:   str           # output projection
    fused_qkv_pat: str          # non-empty when Q/K/V are fused into one tensor

    # FFN — patterns
    ffn_up_pat:   str           # up / intermediate projection (expand)
    ffn_gate_pat: str           # gate projection (SwiGLU only; empty if absent)
    ffn_down_pat: str           # down projection (contract)

    # Config keys used for head/FFN dimension info
    n_layers_key:       str = "num_hidden_layers"
    n_heads_key:        str = "num_attention_heads"
    n_kv_heads_key:     str = "num_key_value_heads"   # GQA; fall back to n_heads
    hidden_size_key:    str = "hidden_size"
    intermediate_key:   str = "intermediate_size"
    vocab_size_key:     str = "vocab_size"

    # Extra tensors that must be row-sliced together with embed (e.g. token_type)
    extra_embed_tensors: List[str] = field(default_factory=list)


# ─── Architecture definitions ────────────────────────────────────────────────

_LLAMA_PROFILE = ArchitectureProfile(
    family             = "llama",
    embed_weight       = "model.embed_tokens.weight",
    lm_head_weight     = "lm_head.weight",
    tie_word_embeddings = False,
    q_proj_pat         = "model.layers.{layer}.self_attn.q_proj.weight",
    k_proj_pat         = "model.layers.{layer}.self_attn.k_proj.weight",
    v_proj_pat         = "model.layers.{layer}.self_attn.v_proj.weight",
    o_proj_pat         = "model.layers.{layer}.self_attn.o_proj.weight",
    fused_qkv_pat      = "",
    ffn_up_pat         = "model.layers.{layer}.mlp.up_proj.weight",
    ffn_gate_pat       = "model.layers.{layer}.mlp.gate_proj.weight",
    ffn_down_pat       = "model.layers.{layer}.mlp.down_proj.weight",
)

_BERT_PROFILE = ArchitectureProfile(
    family             = "bert",
    embed_weight       = "bert.embeddings.word_embeddings.weight",
    lm_head_weight     = "cls.predictions.decoder.weight",
    tie_word_embeddings = True,
    q_proj_pat         = "bert.encoder.layer.{layer}.attention.self.query.weight",
    k_proj_pat         = "bert.encoder.layer.{layer}.attention.self.key.weight",
    v_proj_pat         = "bert.encoder.layer.{layer}.attention.self.value.weight",
    o_proj_pat         = "bert.encoder.layer.{layer}.attention.output.dense.weight",
    fused_qkv_pat      = "",
    ffn_up_pat         = "bert.encoder.layer.{layer}.intermediate.dense.weight",
    ffn_gate_pat       = "",
    ffn_down_pat       = "bert.encoder.layer.{layer}.output.dense.weight",
    n_heads_key        = "num_attention_heads",
    intermediate_key   = "intermediate_size",
    extra_embed_tensors = [
        "bert.embeddings.position_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight",
    ],
)

_ROBERTA_PROFILE = ArchitectureProfile(
    family             = "roberta",
    embed_weight       = "roberta.embeddings.word_embeddings.weight",
    lm_head_weight     = "lm_head.decoder.weight",
    tie_word_embeddings = True,
    q_proj_pat         = "roberta.encoder.layer.{layer}.attention.self.query.weight",
    k_proj_pat         = "roberta.encoder.layer.{layer}.attention.self.key.weight",
    v_proj_pat         = "roberta.encoder.layer.{layer}.attention.self.value.weight",
    o_proj_pat         = "roberta.encoder.layer.{layer}.attention.output.dense.weight",
    fused_qkv_pat      = "",
    ffn_up_pat         = "roberta.encoder.layer.{layer}.intermediate.dense.weight",
    ffn_gate_pat       = "",
    ffn_down_pat       = "roberta.encoder.layer.{layer}.output.dense.weight",
)

_GPT2_PROFILE = ArchitectureProfile(
    family             = "gpt2",
    embed_weight       = "transformer.wte.weight",
    lm_head_weight     = "lm_head.weight",
    tie_word_embeddings = True,
    q_proj_pat         = "",
    k_proj_pat         = "",
    v_proj_pat         = "",
    o_proj_pat         = "transformer.h.{layer}.attn.c_proj.weight",
    fused_qkv_pat      = "transformer.h.{layer}.attn.c_attn.weight",
    ffn_up_pat         = "transformer.h.{layer}.mlp.c_fc.weight",
    ffn_gate_pat       = "",
    ffn_down_pat       = "transformer.h.{layer}.mlp.c_proj.weight",
    n_layers_key       = "n_layer",
    n_heads_key        = "n_head",
    hidden_size_key    = "n_embd",
    intermediate_key   = "n_inner",
)

_NEOX_PROFILE = ArchitectureProfile(
    family             = "gpt_neox",
    embed_weight       = "gpt_neox.embed_in.weight",
    lm_head_weight     = "embed_out.weight",
    tie_word_embeddings = False,
    q_proj_pat         = "",
    k_proj_pat         = "",
    v_proj_pat         = "",
    o_proj_pat         = "gpt_neox.layers.{layer}.attention.dense.weight",
    fused_qkv_pat      = "gpt_neox.layers.{layer}.attention.query_key_value.weight",
    ffn_up_pat         = "gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight",
    ffn_gate_pat       = "",
    ffn_down_pat       = "gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight",
    n_layers_key       = "num_hidden_layers",
    n_heads_key        = "num_attention_heads",
)

_OPT_PROFILE = ArchitectureProfile(
    family             = "opt",
    embed_weight       = "model.decoder.embed_tokens.weight",
    lm_head_weight     = "lm_head.weight",
    tie_word_embeddings = True,
    q_proj_pat         = "model.decoder.layers.{layer}.self_attn.q_proj.weight",
    k_proj_pat         = "model.decoder.layers.{layer}.self_attn.k_proj.weight",
    v_proj_pat         = "model.decoder.layers.{layer}.self_attn.v_proj.weight",
    o_proj_pat         = "model.decoder.layers.{layer}.self_attn.out_proj.weight",
    fused_qkv_pat      = "",
    ffn_up_pat         = "model.decoder.layers.{layer}.fc1.weight",
    ffn_gate_pat       = "",
    ffn_down_pat       = "model.decoder.layers.{layer}.fc2.weight",
)

_PHI_PROFILE = ArchitectureProfile(
    family             = "phi",
    embed_weight       = "model.embed_tokens.weight",
    lm_head_weight     = "lm_head.weight",
    tie_word_embeddings = False,
    q_proj_pat         = "model.layers.{layer}.self_attn.q_proj.weight",
    k_proj_pat         = "model.layers.{layer}.self_attn.k_proj.weight",
    v_proj_pat         = "model.layers.{layer}.self_attn.v_proj.weight",
    o_proj_pat         = "model.layers.{layer}.self_attn.dense.weight",
    fused_qkv_pat      = "",
    ffn_up_pat         = "model.layers.{layer}.mlp.fc1.weight",
    ffn_gate_pat       = "",
    ffn_down_pat       = "model.layers.{layer}.mlp.fc2.weight",
)

_DISTILBERT_PROFILE = ArchitectureProfile(
    family             = "distilbert",
    embed_weight       = "distilbert.embeddings.word_embeddings.weight",
    lm_head_weight     = "vocab_projector.weight",
    tie_word_embeddings = True,
    q_proj_pat         = "distilbert.transformer.layer.{layer}.attention.q_lin.weight",
    k_proj_pat         = "distilbert.transformer.layer.{layer}.attention.k_lin.weight",
    v_proj_pat         = "distilbert.transformer.layer.{layer}.attention.v_lin.weight",
    o_proj_pat         = "distilbert.transformer.layer.{layer}.attention.out_lin.weight",
    fused_qkv_pat      = "",
    ffn_up_pat         = "distilbert.transformer.layer.{layer}.ffn.lin1.weight",
    ffn_gate_pat       = "",
    ffn_down_pat       = "distilbert.transformer.layer.{layer}.ffn.lin2.weight",
    n_layers_key       = "n_layers",
    n_heads_key        = "n_heads",
    hidden_size_key    = "dim",
    intermediate_key   = "hidden_dim",
)

# Map model_type → profile (covers all aliases)
_MODEL_TYPE_MAP: Dict[str, ArchitectureProfile] = {
    # Llama-style (largest family)
    "llama":       _LLAMA_PROFILE,
    "mistral":     _LLAMA_PROFILE,
    "mixtral":     _LLAMA_PROFILE,
    "qwen2":       _LLAMA_PROFILE,
    "qwen3":       _LLAMA_PROFILE,
    "gemma":       _LLAMA_PROFILE,
    "gemma2":      _LLAMA_PROFILE,
    "gemma3":      _LLAMA_PROFILE,
    "minicpm":     _LLAMA_PROFILE,
    "falcon":      _LLAMA_PROFILE,
    "yi":          _LLAMA_PROFILE,
    "deepseek":    _LLAMA_PROFILE,
    "deepseek_v2": _LLAMA_PROFILE,
    "internlm2":   _LLAMA_PROFILE,
    "phi3":        _LLAMA_PROFILE,
    "phi-3":       _LLAMA_PROFILE,
    # BERT family
    "bert":        _BERT_PROFILE,
    "roberta":     _ROBERTA_PROFILE,
    "deberta":     _BERT_PROFILE,
    "deberta-v2":  _BERT_PROFILE,
    "albert":      _BERT_PROFILE,
    "camembert":   _ROBERTA_PROFILE,
    "xlm-roberta": _ROBERTA_PROFILE,
    # GPT-2
    "gpt2":        _GPT2_PROFILE,
    "gpt_neo":     _GPT2_PROFILE,
    # GPT-NeoX / Pythia
    "gpt_neox":    _NEOX_PROFILE,
    # OPT
    "opt":         _OPT_PROFILE,
    # Phi
    "phi":         _PHI_PROFILE,
    "phi-msft":    _PHI_PROFILE,
    # DistilBERT
    "distilbert":  _DISTILBERT_PROFILE,
}


def detect_architecture(config: dict) -> ArchitectureProfile:
    """
    Return the ``ArchitectureProfile`` for a model given its parsed config.json.

    Falls back to the Llama profile (the most common decoder-only naming) when
    the ``model_type`` field is absent or unrecognized.
    """
    model_type = config.get("model_type", "").lower().replace("_", "-")
    # Sort keys by descending length so more-specific keys (e.g. "gpt-neox")
    # are checked before shorter prefix matches (e.g. "gpt-neo" or "gpt2").
    for key in sorted(_MODEL_TYPE_MAP, key=len, reverse=True):
        if model_type.startswith(key.replace("_", "-")):
            return _MODEL_TYPE_MAP[key]
    return _LLAMA_PROFILE   # safe default for unknown decoder-only models


# ─── Safetensors I/O helpers ─────────────────────────────────────────────────

def _parse_st_header(path: Path) -> Tuple[Dict, int]:
    """Read the JSON header from a safetensors file without loading tensors."""
    with open(path, "rb") as fh:
        hdr_len  = struct.unpack("<Q", fh.read(8))[0]
        hdr_json = json.loads(fh.read(hdr_len).rstrip(b" \x00"))
    return {k: v for k, v in hdr_json.items() if k != "__metadata__"}, 8 + hdr_len


def iter_tensors(
    shard_paths: List[Path],
) -> Iterator[Tuple[str, np.ndarray, str]]:
    """
    Yield ``(tensor_name, float32_array, shard_filename)`` for every
    supported tensor across all shards, without holding more than one
    tensor in RAM at a time.

    BF16 tensors are converted to float32 using the bit-shift trick.
    All other float types are cast to float32.
    Integer tensors are cast to float32 (needed for embedding manipulation).
    """
    from safetensors import safe_open

    _NP_COMPAT = {"F32", "F16", "I8", "U8", "I16", "U16", "I32", "U32"}

    for shard_path in shard_paths:
        hdr, data_start = _parse_st_header(shard_path)

        # Group keys by whether they need the raw-byte path
        np_keys  = [k for k, v in hdr.items() if v["dtype"] in _NP_COMPAT]
        raw_keys = [k for k, v in hdr.items() if v["dtype"] in ("BF16", "F8_E4M3", "F8_E5M2")]

        if np_keys:
            with safe_open(str(shard_path), framework="np") as sf:
                for name in np_keys:
                    t = sf.get_tensor(name).astype(np.float32)
                    yield name, t, shard_path.name

        if raw_keys:
            with open(shard_path, "rb") as fh:
                fh.seek(data_start)
                data_region = fh.read()
            for name in raw_keys:
                info = hdr[name]
                s, e = info["data_offsets"]
                raw  = data_region[s:e]
                if info["dtype"] == "BF16":
                    u16 = np.frombuffer(raw, dtype=np.uint16)
                    t   = (u16.astype(np.uint32) << 16).view(np.float32).copy()
                else:
                    t   = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                yield name, t, shard_path.name


def load_all_tensors(shard_paths: List[Path]) -> Dict[str, np.ndarray]:
    """
    Load all tensors from all shards into a single dict of float32 arrays.
    For large models this can use significant RAM; prefer ``iter_tensors``
    when only a subset of tensors is needed.
    """
    return {name: arr for name, arr, _ in iter_tensors(shard_paths)}


def save_model_dir(
    tensors: Dict[str, np.ndarray],
    out_dir: Path,
    dtype_str: str = "F32",
    shard_size_gb: float = 4.0,
) -> None:
    """
    Save *tensors* as one or more safetensors shards in *out_dir*.

    Parameters
    ----------
    tensors      : dict of {name: float32 ndarray}
    out_dir      : destination directory (created if absent)
    dtype_str    : storage dtype; "F32" or "BF16"
    shard_size_gb: maximum bytes per shard (approximate)
    """
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert if BF16 storage requested
    def _to_storage(arr: np.ndarray) -> np.ndarray:
        if dtype_str == "BF16":
            # float32 → BF16 via view trick
            f32 = arr.astype(np.float32)
            u32 = f32.view(np.uint32)
            return (u32 >> 16).astype(np.uint16)
        return arr.astype(np.float32)

    shard_bytes = int(shard_size_gb * 1024 ** 3)
    names       = list(tensors.keys())

    # Partition into shards
    shards:    List[Dict[str, np.ndarray]] = [{}]
    cur_bytes: int = 0

    for name in names:
        arr   = _to_storage(tensors[name])
        nbytes = arr.nbytes
        if cur_bytes + nbytes > shard_bytes and cur_bytes > 0:
            shards.append({})
            cur_bytes = 0
        shards[-1][name] = arr
        cur_bytes += nbytes

    if len(shards) == 1:
        save_file(shards[0], str(out_dir / "model.safetensors"))
        _write_index(tensors, ["model.safetensors"], out_dir)
    else:
        n = len(shards)
        filenames = []
        for idx, shard in enumerate(shards, 1):
            fname = f"model-{idx:05d}-of-{n:05d}.safetensors"
            save_file(shard, str(out_dir / fname))
            filenames.append(fname)
        _write_index(tensors, filenames, out_dir)


def _write_index(
    tensors:   Dict[str, np.ndarray],
    filenames: List[str],
    out_dir:   Path,
) -> None:
    """Write model.safetensors.index.json for sharded models."""
    if len(filenames) == 1:
        return  # single-file models don't need an index

    # Rebuild shard → tensor mapping
    # We need to know which filename each tensor ended up in.
    # Re-read the shard headers to get the mapping.
    weight_map = {}
    total_size = 0
    for fname in filenames:
        hdr, _ = _parse_st_header(out_dir / fname)
        for name, info in hdr.items():
            weight_map[name] = fname
            s, e = info["data_offsets"]
            total_size += e - s

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(out_dir / "model.safetensors.index.json", "w") as fh:
        json.dump(index, fh, indent=2)


# ─── Output directory naming ─────────────────────────────────────────────────

def make_output_name(model_id: str, suffix: str) -> str:
    """
    Build an output directory name from a model ID and a pruning suffix.

    Examples
    --------
    >>> make_output_name("amd/AMD-Llama-135m", "vocab_pr")
    'amd--AMD-Llama-135m-vocab_pr'
    >>> make_output_name("bert-base-uncased", "trans_pr")
    'bert-base-uncased-trans_pr'
    """
    safe = model_id.replace("/", "--").rstrip("-")
    return f"{safe}-{suffix}"


# ─── Model directory resolution ──────────────────────────────────────────────

def resolve_model_dir(metadata) -> "Optional[Path]":
    """
    Reliably locate the model directory (the folder containing config.json).

    shard_paths in WeightScope metadata can point into the HuggingFace hub
    blob cache (``blobs/sha256abc…``) rather than the snapshot directory that
    holds ``config.json``.  This function walks UP the path hierarchy until
    it finds a directory containing ``config.json`` (up to 8 levels).

    Returns None if metadata is absent or no config.json ancestor is found.
    """
    if metadata is None:
        return None

    shard_paths = metadata.get("shard_paths", [])
    if not shard_paths:
        fp = metadata.get("file_path", "")
        shard_paths = [fp] if fp else []
    if not shard_paths:
        return None

    start = Path(shard_paths[0])
    candidate = start if start.is_dir() else start.parent

    for _ in range(8):
        if (candidate / "config.json").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:   # reached filesystem root
            break
        candidate = parent

    # Fallback: direct parent even if config.json was not found
    return start.parent if not start.is_dir() else start


# ─── Config / tokenizer support file helpers ─────────────────────────────────

_SUPPORT_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",          # SentencePiece models
    "special_tokens_map.json",
    "vocab.json",               # GPT-2 / RoBERTa
    "merges.txt",               # GPT-2 / RoBERTa BPE merges
    "vocab.txt",                # BERT WordPiece
    "added_tokens.json",
    "generation_config.json",
]


def copy_support_files(src_dir: Path, dst_dir: Path) -> List[str]:
    """
    Copy tokenizer and config files from *src_dir* to *dst_dir*.
    Returns a list of filenames that were copied.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for fname in _SUPPORT_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
            copied.append(fname)
    return copied


def update_config(dst_dir: Path, updates: Dict) -> None:
    """
    Merge *updates* into the ``config.json`` in *dst_dir*.
    Creates a minimal config.json if none exists.
    """
    cfg_path = dst_dir / "config.json"
    cfg: Dict = {}
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    cfg.update(updates)
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)


def load_config(model_dir: Path) -> Dict:
    """Load and return config.json from *model_dir*, or {} if absent."""
    p = model_dir / "config.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return {}


# ─── Tokenizer helpers ────────────────────────────────────────────────────────

def try_load_tokenizer(model_dir: Path):
    """
    Attempt to load a HuggingFace tokenizer from *model_dir*.
    Returns the tokenizer object or None if transformers is not installed.
    """
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    except Exception:
        return None


def tokenize_corpus(
    tokenizer,
    corpus_lines: List[str],
    batch_size: int = 512,
) -> np.ndarray:
    """
    Tokenize *corpus_lines* with *tokenizer* and return a sorted unique
    array of token IDs that appear in the corpus.

    Parameters
    ----------
    tokenizer   : HuggingFace tokenizer
    corpus_lines: list of text strings
    batch_size  : lines to encode per batch

    Returns
    -------
    np.ndarray of dtype int64, sorted unique token IDs
    """
    seen: set = set()
    for i in range(0, len(corpus_lines), batch_size):
        batch = corpus_lines[i : i + batch_size]
        enc   = tokenizer(batch, add_special_tokens=True,
                          truncation=False, padding=False)
        for ids in enc["input_ids"]:
            seen.update(ids)
    return np.array(sorted(seen), dtype=np.int64)


def load_corpus_lines(
    source: str,
    max_lines: int = 200_000,
) -> List[str]:
    """
    Load text lines from *source*.

    *source* can be:
    - A local file path (plain text, one document per line)
    - A HuggingFace dataset name, e.g. ``"wikitext"`` or ``"wikitext:wikitext-2-raw-v1"``
    - A URL to a plain-text file

    Returns up to *max_lines* non-empty stripped lines.
    """
    lines: List[str] = []

    # Local file
    p = Path(source)
    if p.exists() and p.is_file():
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if line:
                    lines.append(line)
                if len(lines) >= max_lines:
                    break
        return lines

    # HuggingFace dataset  (format: "dataset_name" or "dataset_name:config")
    if not source.startswith("http"):
        parts   = source.split(":", 1)
        ds_name = parts[0]
        ds_cfg  = parts[1] if len(parts) > 1 else None
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_name, ds_cfg, split="train", streaming=True,
                              trust_remote_code=False)
            for sample in ds:
                text = sample.get("text", "") or sample.get("content", "") or ""
                text = text.strip()
                if text:
                    lines.append(text)
                if len(lines) >= max_lines:
                    break
            return lines
        except Exception:
            pass  # fall through to URL attempt

    # URL
    try:
        import urllib.request
        with urllib.request.urlopen(source, timeout=30) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    lines.append(line)
                if len(lines) >= max_lines:
                    break
    except Exception:
        pass

    return lines
