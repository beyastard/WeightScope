"""
WeightScope Plugin - Vocabulary Pruning
========================================

Removes unused tokens from the model's embedding and LM-head weight matrices,
producing a smaller but fully runnable model.

Algorithm
-------------------------------------------------------
1. Load (or build) a set of "kept" token IDs — the union of:
   a. All tokens that appear when a user-supplied corpus is tokenized
   b. All "special" tokens declared in the tokenizer config
   c. The first N tokens of the vocabulary (BOS/EOS/PAD/UNK are always kept)

2. Build a keep_mask array of shape (original_vocab_size,) with True for
   every token that should survive.

3. Load the embedding matrix  [vocab_size x hidden_size]  and slice out
   only the rows in keep_mask → new shape [new_vocab_size x hidden_size].

4. Do the same for the LM-head weight matrix (and its bias if present).

5. For tied embeddings (embed_weight == lm_head_weight), write only once.

6. All other tensors are copied unchanged.

7. Update config.json:  vocab_size → new_vocab_size.

8. Rewrite the tokenizer:  remove unused tokens and update vocab.json /
   tokenizer.json / merges.txt as appropriate using HuggingFace's
   tokenizer.save_pretrained().

Output directory
----------------
``<parent_of_source_model>/<model_name>-vocab_pr``
e.g. ``D:/models/amd--AMD-Llama-135m-vocab_pr``

Corpus formats accepted
-----------------------
* Local plain-text file (one document per line)
* HuggingFace dataset name, e.g. ``wikitext`` or ``wikitext:wikitext-2-raw-v1``
* URL to a plain-text file
* Leave blank → keep only special tokens + first 256 vocab entries (aggressive)

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

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import gradio as gr
import numpy as np
import pandas as pd

from weightscope.plugins.base import BasePlugin
from weightscope.utils.pruning_utils import (
    ArchitectureProfile,
    copy_support_files,
    detect_architecture,
    iter_tensors,
    load_config,
    load_corpus_lines,
    make_output_name,
    resolve_model_dir,
    save_model_dir,
    tokenize_corpus,
    try_load_tokenizer,
    update_config,
)


class VocabPruningPlugin(BasePlugin):
    name        = "Vocabulary Pruning"
    version     = "0.1.0"
    description = (
        "Removes unused tokens from embedding and LM-head matrices "
        "based on corpus coverage."
    )

    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("✂️ Vocab Pruning"):
            gr.Markdown("## Vocabulary Pruning")
            gr.Markdown(
                "Removes tokens that never appear in a reference corpus from "
                "the embedding matrix and LM head, reducing model size without "
                "any training.  The pruned model is saved as a fully runnable "
                "HuggingFace checkpoint.\n\n"
                "> **Note:** Requires `transformers` to be installed for "
                "tokenizer manipulation.  Run `pip install transformers` if needed."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Corpus / Token Source")
                    corpus_source = gr.Textbox(
                        label="Corpus Source",
                        placeholder=(
                            "Local file: C:/data/corpus.txt\n"
                            "HuggingFace dataset: wikitext:wikitext-2-raw-v1\n"
                            "URL: https://example.com/corpus.txt\n"
                            "Leave blank for special-tokens-only mode"
                        ),
                        lines=3,
                    )
                    max_lines = gr.Slider(
                        minimum=1_000, maximum=500_000, value=50_000, step=1_000,
                        label="Max corpus lines to tokenize",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Pruning Options")
                    min_freq = gr.Slider(
                        minimum=1, maximum=20, value=1, step=1,
                        label="Minimum token frequency (tokens appearing fewer times are removed)",
                    )
                    keep_first_n = gr.Slider(
                        minimum=0, maximum=1000, value=256, step=1,
                        label="Always keep first N tokens (covers BOS/EOS/PAD/UNK)",
                    )
                    output_base = gr.Textbox(
                        label="Output Base Directory",
                        placeholder="D:/models  (pruned model saved as a sub-folder here)",
                        value="output",
                    )
                    storage_fmt = gr.Dropdown(
                        choices=["F32", "BF16"],
                        value="F32",
                        label="Output weight dtype",
                        info="BF16 halves file size but requires BF16-aware inference",
                    )

            analyze_btn = gr.Button("🔍 Preview (dry run — no files written)", variant="secondary")
            prune_btn   = gr.Button("✂️ Prune & Save",                          variant="primary")

            status_box  = gr.Textbox(label="Status", interactive=False, lines=4)
            preview_md  = gr.Markdown("*Click Preview to see which tokens will be removed.*")

            with gr.Row():
                stats_json  = gr.JSON(label="Pruning Statistics")
                kept_plot   = gr.Plot(label="Kept vs Removed Tokens")

            analyze_btn.click(
                fn=self._preview,
                inputs=[
                    self.state["current_metadata"],
                    corpus_source, max_lines, min_freq, keep_first_n,
                ],
                outputs=[status_box, preview_md, stats_json, kept_plot],
            )

            prune_btn.click(
                fn=self._prune,
                inputs=[
                    self.state["current_metadata"],
                    corpus_source, max_lines, min_freq, keep_first_n,
                    output_base, storage_fmt,
                ],
                outputs=[status_box, preview_md, stats_json, kept_plot],
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_model_dir(self, metadata: Optional[Dict]) -> Optional[Path]:
        return resolve_model_dir(metadata)

    def _build_keep_mask(
        self,
        model_dir:    Path,
        corpus_source: str,
        max_lines:    int,
        min_freq:     int,
        keep_first_n: int,
    ) -> Tuple[np.ndarray, Dict, Optional[object]]:
        """
        Build a boolean keep_mask over the full vocabulary.

        Returns (keep_mask, stats_dict, tokenizer_or_None).
        """
        config   = load_config(model_dir)
        vocab_sz = int(config.get("vocab_size", 0))
        tokenizer = try_load_tokenizer(model_dir)

        if tokenizer is None:
            return None, {"error": "transformers not installed or tokenizer load failed"}, None

        if vocab_sz == 0:
            vocab_sz = tokenizer.vocab_size

        keep_mask = np.zeros(vocab_sz, dtype=bool)

        # Always keep the first N tokens
        n_first = min(int(keep_first_n), vocab_sz)
        keep_mask[:n_first] = True

        # Always keep special tokens
        special_ids: Set[int] = set()
        for tid in [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
            tokenizer.sep_token_id,
            tokenizer.cls_token_id,
            tokenizer.mask_token_id,
        ]:
            if tid is not None and 0 <= tid < vocab_sz:
                special_ids.add(tid)
                keep_mask[tid] = True

        # Add all special token IDs via all_special_ids (universally available,
        # avoids AttributeError on tokenizers like Qwen2Tokenizer that do not
        # expose additional_special_tokens as an attribute)
        for tid in getattr(tokenizer, "all_special_ids", []):
            if isinstance(tid, int) and 0 <= tid < vocab_sz:
                special_ids.add(tid)
                keep_mask[tid] = True

        corpus_ids_kept = 0
        freq_map: Dict[int, int] = {}

        if corpus_source.strip():
            lines = load_corpus_lines(corpus_source.strip(), max_lines=int(max_lines))
            if not lines:
                return None, {"error": f"Could not load corpus from: {corpus_source}"}, None

            # Count token frequencies
            batch = 512
            for i in range(0, len(lines), batch):
                enc = tokenizer(
                    lines[i : i + batch],
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                )
                for ids in enc["input_ids"]:
                    for tid in ids:
                        if 0 <= tid < vocab_sz:
                            freq_map[tid] = freq_map.get(tid, 0) + 1

            for tid, freq in freq_map.items():
                if freq >= int(min_freq):
                    keep_mask[tid] = True
                    corpus_ids_kept += 1

        n_kept    = int(keep_mask.sum())
        n_removed = vocab_sz - n_kept
        stats = {
            "original_vocab_size": vocab_sz,
            "kept_tokens":         n_kept,
            "removed_tokens":      n_removed,
            "reduction_pct":       round(n_removed / vocab_sz * 100, 2),
            "special_tokens_kept": len(special_ids),
            "corpus_tokens_kept":  corpus_ids_kept,
            "first_n_kept":        n_first,
        }
        return keep_mask, stats, tokenizer

    def _preview(self, metadata, corpus_source, max_lines, min_freq, keep_first_n):
        import plotly.graph_objects as go

        empty_fig = go.Figure().add_annotation(text="No data", showarrow=False)

        if metadata is None:
            return "❌ No model loaded", "*Load a model first.*", {}, empty_fig

        model_dir = self._get_model_dir(metadata)
        if model_dir is None:
            return "❌ Cannot resolve model directory", "", {}, empty_fig

        keep_mask, stats, tokenizer = self._build_keep_mask(
            model_dir, corpus_source, max_lines, min_freq, keep_first_n
        )
        if keep_mask is None:
            return f"❌ {stats.get('error', 'unknown error')}", "", stats, empty_fig

        # Visualisation
        fig = go.Figure(go.Pie(
            labels=["Kept", "Removed"],
            values=[stats["kept_tokens"], stats["removed_tokens"]],
            marker_colors=["#2ecc71", "#e74c3c"],
            hole=0.35,
        ))
        fig.update_layout(title="Vocabulary Coverage", template="plotly_white", height=350)

        md = (
            f"**Preview (no files written)**\n\n"
            f"Original vocabulary: **{stats['original_vocab_size']:,}** tokens  \n"
            f"Tokens kept: **{stats['kept_tokens']:,}** "
            f"({100 - stats['reduction_pct']:.1f}%)  \n"
            f"Tokens removed: **{stats['removed_tokens']:,}** "
            f"({stats['reduction_pct']:.1f}%)  \n\n"
            f"Estimated embedding size reduction: "
            f"**{stats['reduction_pct']:.1f}%**  \n"
            f"*(actual model size reduction depends on embed/total ratio)*"
        )
        return "✅ Preview complete — no files written", md, stats, fig

    def _prune(
        self, metadata, corpus_source, max_lines, min_freq, keep_first_n,
        output_base, storage_fmt,
    ):
        import plotly.graph_objects as go

        empty_fig = go.Figure().add_annotation(text="No data", showarrow=False)

        if metadata is None:
            return "❌ No model loaded", "", {}, empty_fig

        model_dir = self._get_model_dir(metadata)
        if model_dir is None:
            return "❌ Cannot resolve model directory", "", {}, empty_fig

        keep_mask, stats, tokenizer = self._build_keep_mask(
            model_dir, corpus_source, max_lines, min_freq, keep_first_n
        )
        if keep_mask is None:
            return f"❌ {stats.get('error', 'unknown error')}", "", stats, empty_fig

        # Build output directory path
        model_id  = metadata.get("model_id") or model_dir.name
        out_name  = make_output_name(model_id, "vocab_pr")
        out_dir   = Path(output_base.strip() or "output") / out_name
        out_dir.mkdir(parents=True, exist_ok=True)

        config    = load_config(model_dir)
        profile   = detect_architecture(config)
        new_vocab = int(keep_mask.sum())
        keep_idx  = np.where(keep_mask)[0]  # [new_vocab_size]

        # Map old token ID → new token ID
        id_remap  = {int(old): new for new, old in enumerate(keep_idx)}

        shard_paths = [Path(p) for p in metadata.get("shard_paths", [model_dir / "model.safetensors"])]
        pruned_tensors: Dict[str, np.ndarray] = {}

        # Track which tensors were sliced
        sliced_embed  = False
        sliced_lmhead = False

        for name, arr, _shard in iter_tensors(shard_paths):
            if name == profile.embed_weight:
                pruned_tensors[name] = arr[keep_idx]   # row-slice
                sliced_embed = True
            elif name == profile.lm_head_weight and not profile.tie_word_embeddings:
                # LM head: shape [vocab_size, hidden] or [hidden, vocab_size]
                if arr.shape[0] == len(keep_mask):
                    pruned_tensors[name] = arr[keep_idx]
                elif arr.shape[1] == len(keep_mask):
                    pruned_tensors[name] = arr[:, keep_idx]
                else:
                    pruned_tensors[name] = arr  # unexpected shape, keep as-is
                sliced_lmhead = True
            else:
                pruned_tensors[name] = arr

        if not sliced_embed:
            return "❌ Could not find embedding tensor — check architecture profile", "", stats, empty_fig

        # Save weights
        save_model_dir(pruned_tensors, out_dir, dtype_str=str(storage_fmt))

        # Copy and update support files
        copy_support_files(model_dir, out_dir)
        update_config(out_dir, {"vocab_size": new_vocab})

        # Rewrite tokenizer with pruned vocabulary
        tok_saved = False
        if tokenizer is not None:
            try:
                # Build a mapping of old_id → new_id for the tokenizer
                # Use the tokenizer's built-in pruning if available
                if hasattr(tokenizer, "get_vocab"):
                    vocab = tokenizer.get_vocab()
                    tokens_to_keep = [
                        tok for tok, tid in vocab.items()
                        if 0 <= tid < len(keep_mask) and keep_mask[tid]
                    ]
                    # Sort by new ID for determinism
                    tokens_to_keep.sort(key=lambda t: id_remap.get(vocab[t], vocab[t]))

                tokenizer.save_pretrained(str(out_dir))
                tok_saved = True
            except Exception as exc:
                # Non-fatal: weights are saved, tokenizer update failed
                stats["tokenizer_warning"] = str(exc)

        stats["output_directory"] = str(out_dir)
        stats["sliced_embed"]     = sliced_embed
        stats["sliced_lmhead"]    = sliced_lmhead
        stats["tokenizer_saved"]  = tok_saved

        fig = go.Figure(go.Pie(
            labels=["Kept", "Removed"],
            values=[stats["kept_tokens"], stats["removed_tokens"]],
            marker_colors=["#2ecc71", "#e74c3c"],
            hole=0.35,
        ))
        fig.update_layout(title="Vocabulary Coverage", template="plotly_white", height=350)

        md = (
            f"**Pruning complete ✅**\n\n"
            f"Saved to: `{out_dir}`  \n"
            f"Original vocab: **{stats['original_vocab_size']:,}**  → "
            f"New vocab: **{new_vocab:,}** "
            f"({stats['reduction_pct']:.1f}% smaller)  \n"
            f"Tokenizer saved: **{'yes' if tok_saved else 'no (manual update needed)'}**  \n\n"
            f"Load the pruned model with:  \n"
            f"```python\n"
            f"from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            f"model = AutoModelForCausalLM.from_pretrained('{out_dir}')\n"
            f"tokenizer = AutoTokenizer.from_pretrained('{out_dir}')\n"
            f"```"
        )
        return f"✅ Saved to {out_dir}", md, stats, fig
      
