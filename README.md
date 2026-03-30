# 🔍 WeightScope

**SafeTensors Model Analyzer** — a research tool for inspecting, visualizing,
and simulating the weight distributions of machine-learning models stored in
the `.safetensors` format, with a growing plugin ecosystem for compression
analysis, model comparison, and experimental weight pruning.

> **Version 0.2.2** · AGPL-3.0 · Copyright © 2026 Bryan K Reinhart & BeySoft

[![Tests](https://img.shields.io/badge/tests-49%20passed-green)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)]()

Note: this `README.md` may be slightly out of date.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Supported Formats](#supported-formats)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Project Structure](#project-structure)
8. [Memory Architecture](#memory-architecture)
9. [Sharded Model Support](#sharded-model-support)
10. [UI Tabs Reference](#ui-tabs-reference)
11. [Configuration](#configuration)
12. [Plugin System](#plugin-system)
13. [Plugin Reference](#plugin-reference)
14. [Cache Management](#cache-management)
15. [Running the Tests](#running-the-tests)
16. [Contributing](#contributing)
17. [License](#license)

---

## Overview

WeightScope loads `.safetensors` model files — locally or directly from the
HuggingFace Hub — and analyzes every weight value at the bit-pattern level.
Instead of sampling or approximating, it counts **every unique floating-point
or integer pattern** that appears in the model, producing an exact frequency
table that drives all downstream analysis.

Analysis is **streaming and memory-bounded**: tensors are processed one at a
time and frequency counts are accumulated in a temporary DuckDB database.
Models of any size can be analyzed on modest hardware — the peak RAM footprint
is determined by the largest single tensor, not the total model size.  Both
single-file and **sharded** models are supported.

A **plugin architecture** allows extending WeightScope with new analysis tabs
without modifying core code. Seven plugins are included, covering extended
statistics, compression estimation, model comparison, and experimental pruning.

Typical use cases:

- Understand the sparsity and distribution shape of a model before fine-tuning
  or deployment.
- Estimate how much a model can be compressed with quantization, pruning, or
  lookup-table encoding.
- Compare weight distributions between two model checkpoints.
- Simulate the information loss introduced by clipping, normalization, or
  bit-width reduction.
- Identify unusual distribution patterns (e.g. multi-modal distributions caused
  by weight-sharing or codebook quantization schemes).

---

## Features

| Tab | What it does |
|-----|-------------|
| 📂 **Load Model** | Load single-file or sharded models locally or from HuggingFace Hub; native OS folder picker via the Browse button |
| 📊 **Overview** | Parameter count, unique pattern count, shard count, tensor inventory, dtype summary |
| 📈 **Distribution** | Interactive histogram (log/linear) and scatter plot with singleton/outlier filters |
| 🔎 **Query** | Filter weights by value range and occurrence count; 8 built-in presets + Custom mode |
| 🗜️ **Compression** | Uniform quantization simulation (4–16 bit, incl. INT4/INT8) and low-count removal impact |
| ✂️ **Pruning** | Live sparsity analysis — see exactly how many parameters fall below any threshold ε |
| ✂️ **Clip & Normalize** | Simulate clipping outliers and normalizing to [−1, 1]; reports MSE, MAE, SNR, bits saved |
| ⚖️ **Compare** | Side-by-side distribution overlay of two exported analyzes |
| 💾 **Export** | Save the weight frequency table (Parquet/CSV/JSON) and plots (PNG/SVG/HTML) |
| 🔌 **Plugins** | Seven bundled plugins add additional analysis, compression, and pruning capabilities |

---

## Supported Formats

WeightScope uses a **hybrid read strategy** to handle all dtype variants.
Numpy-compatible dtypes are loaded via `safe_open`; BF16 and FP8 variants
are read as raw bytes and converted to float32 manually (avoiding the
`TypeError: data type 'bfloat16' not understood` that `safe_open` raises).

| dtype | Storage | Read path | Notes |
|-------|---------|-----------|-------|
| `float32` | 4 bytes | Bit-exact pattern counting via uint32 view |
| `float16` | 2 bytes | Upcast to float32; bit-exact uint32 key |
| `bfloat16` | 2 bytes | Raw bytes → uint16 → uint32 left-shift 16; no numpy dtype involvement |
| `float8_e4m3fn` | 1 byte | Upcast to float32 |
| `float8_e5m2` | 1 byte | Upcast to float32 |
| `int8` | 1 byte | Cast to float32 |
| `uint8` | 1 byte | Cast to float32 |
| `int4` (packed) | ½ byte | Two 4-bit signed values per byte; nibble-unpacked with sign extension |

**Supported model families (non-exhaustive):** Llama, Mistral, Qwen, Phi,
Gemma, Falcon, MiniCPM, BERT, RoBERTa, Whisper, CLIP, Stable Diffusion, FLUX,
ControlNet, and any other model saved in `.safetensors` format — both
single-file and sharded.

---

## Requirements

- Python 3.10 or later (only 3.13 tested)
- See `requirements.txt` for the full dependency list

**Core runtime:**

```
safetensors>=0.4.0
numpy>=2.0.0
pandas>=2.0.0
pyarrow>=14.0.0
duckdb>=0.10.0
gradio>=4.0.0
plotly>=5.18.0
kaleido==0.2.1
huggingface_hub>=0.20.0
psutil>=5.9.0
```

**Optional — required only by specific plugins:**

```
transformers>=4.40.0    # Vocabulary Pruning and Transformer Pruning plugins
datasets>=2.0.0         # Vocabulary Pruning plugin (HuggingFace dataset corpus source)
```

> **kaleido version note:** Pin exactly to `kaleido==0.2.1`. Version 1.0 and
> later require Google Chrome to be installed. The 0.2.1 release works
> headlessly on Windows, macOS, and Linux without any browser dependency.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/beyastard/WeightScope.git
chdir WeightScope              # Linux / macOS
cd WeightScope                 # Windows

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate.bat     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: install plugin dependencies for pruning
pip install transformers datasets
```

No additional build steps are required.

---

## Quick Start

```bash
python app.py
```

Then open **http://127.0.0.1:7860** in your browser (set to automatically open with your default browser).

### Load a local model

1. Select **Local** in the Load Model tab.
2. Click **📁 Browse…** to open the OS native folder picker (or type the path
   directly). Navigate to any drive or directory — there are no restrictions.
3. Click **🚀 Load & Analyze**.

WeightScope auto-detects the layout — single file or sharded — and processes
all shards as a single unified analysis. Results are cached in `.save_state/`
keyed by SHA-256 hash; subsequent loads of the same model are nearly instant.

```
# Single-file layout
my-model/
└── model.safetensors

# Sharded layout (HF standard)
my-model/
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
└── model-00003-of-00003.safetensors
```

The first load analyzes the file(s) and caches results in `.save_state/`.
For sharded models the cache key is a SHA-256 chain across **all** shard
files in order, so any change to any shard invalidates the cache.
Subsequent loads of the same model are virtually instantaneous.

### Load from HuggingFace

1. Select **HuggingFace** in the Load Model tab.
2. Enter a model ID such as `amd/AMD-Llama-135m` or
   `Qwen/Qwen2.5-0.5B-Instruct`.
3. Click **🚀 Load & Analyze**.

An internet connection is required.  WeightScope lists all `.safetensors`
files in the repository and downloads every shard before analysis begins.
Downloaded files are cached locally in `models/` for future runs.

### Environment variable overrides

| Variable | Default | Effect |
|----------|---------|--------|
| `WEIGHTSCOPE_HOST` | `127.0.0.1` | Server bind address (`0.0.0.0` to expose on LAN) |
| `WEIGHTSCOPE_PORT` | `7860` | Server port |
| `WEIGHTSCOPE_CACHE_DIR` | `.save_state` | Analysis cache location |
| `WEIGHTSCOPE_MODELS_DIR` | `models` | HuggingFace download cache |
| `WEIGHTSCOPE_OUTPUT_DIR` | `output` | Default export directory |
| `WEIGHTSCOPE_PLUGINS_DIR` | `plugins` | Plugin discovery root |
| `WEIGHTSCOPE_CHUNK_SIZE` | `4000000` | DuckDB in-memory buffer (entries) |
| `WEIGHTSCOPE_TEMP_DIR` | OS temp dir | DuckDB working directory |
| `WEIGHTSCOPE_BROWSE_ROOT` | Home directory | Root shown in folder browser |

```bash
# Expose on LAN, point folder browser at model drive, use scratch disk for DuckDB
WEIGHTSCOPE_HOST=0.0.0.0 \
WEIGHTSCOPE_BROWSE_ROOT=D:/Models \
WEIGHTSCOPE_TEMP_DIR=/mnt/scratch \
python app.py
```

---

## Project Structure

```
WeightScope/
├── app.py                              ← Entry point (~28 lines)
├── requirements.txt
├── README.md
│
├── weightscope/                        ← Core package
│   ├── __init__.py                     ← APP_NAME, APP_VERSION
│   ├── config.py                       ← All constants & env-var overrides
│   │
│   ├── core/                           ← Business logic (no UI dependencies)
│   │   ├── loader.py                   ← ModelLoader: shard discovery, local + HF
│   │   ├── analyzer.py                 ← WeightAnalyzer: streaming analysis
│   │   └── cache.py                    ← SessionCache: persistence + export
│   │
│   ├── utils/
│   │   ├── helpers.py                  ← General utilities
│   │   └── pruning_utils.py            ← Shared code for pruning plugins
│   │
│   ├── ui/
│   │   ├── app_builder.py              ← build_app(): assembles tabs + events
│   │   ├── plotting.py                 ← Plotly figure functions + save_figure()
│   │   └── tabs/                       ← One file per Gradio tab (9 tabs)
│   │
│   └── plugins/
│       ├── __init__.py                 ← PluginRegistry + auto-discovery
│       └── base.py                     ← BasePlugin ABC
│
├── plugins/                            ← User-installable plugin packages
│   ├── example_plugin/                 ← Extended Stats (reference implementation)
│   ├── entropy_analysis/               ← Shannon entropy & information density
│   ├── layer_breakdown/                ← Per-tensor statistics
│   ├── lookup_table_estimator/         ← LUT compression sizing
│   ├── model_fingerprint/              ← Distribution fingerprint & similarity
│   ├── outlier_tensor_report/          ← Anomalous tensor detection
│   ├── vocab_pruning/                  ← ⚠️ EXPERIMENTAL vocabulary pruning
│   └── transformer_pruning/            ← ⚠️ EXPERIMENTAL transformer pruning
│
├── tests/
│   └── test_analyzer.py                ← 49 unit tests
│
├── docs/
├── examples/
├── models/                             ← HuggingFace download cache (git-ignored)
└── output/                             ← Default export destination (git-ignored)
```

## Memory Architecture

WeightScope is designed to analyze models larger than available RAM without
swapping or crashing.  The streaming engine works as follows:

1. **Header parse** — the safetensors JSON header of each shard is read once
   (a few KB) to determine tensor names, dtypes, shapes, and byte offsets.
   No tensor data is loaded yet.

2. **Tensor streaming** — tensors are read one at a time.  Each tensor is
   converted to a flat `uint32` key array and passed to `_StreamingCounter`.

3. **Per-tensor uniqueness** — `np.unique` runs on the single tensor's keys
   (bounded by that tensor's size), producing a compact `(key, count)` pair
   set which is bulk-inserted into DuckDB.  The tensor is then released.

4. **DuckDB aggregation** — after all shards are processed, a single
   `GROUP BY key, SUM(count) ORDER BY key` query in C++ merges every shard's
   partial counts into the final frequency table.  DuckDB spills to disk
   automatically if needed.

5. **DataFrame** — the sorted result is read back in 100,000-row batches,
   float32 values are recovered from the uint32 keys via bit-reinterpretation,
   and the final `pd.DataFrame` is returned.

**Peak RAM** is approximately the size of the largest single tensor plus a
small DuckDB working buffer — not the full model size.  The `ANALYSIS_CHUNK_SIZE`
and `ANALYSIS_TEMP_DIR` settings let you trade RAM usage against disk I/O.

| `ANALYSIS_CHUNK_SIZE` | Approx. RAM for buffer | Notes |
|----------------------|------------------------|-------|
| 1,000,000 | ~8 MB | Safest; more DuckDB flush operations |
| 4,000,000 | ~32 MB | Default; good balance |
| 16,000,000 | ~128 MB | Faster on 32 GB+ systems |

---

## Sharded Model Support

| Priority | Pattern | Example |
|----------|---------|---------|
| 1 | `model.safetensors` | Single-file model |
| 2 | `model-NNNNN-of-MMMMM.safetensors` | HF standard sharding |
| 3 | Any `*.safetensors` | Non-standard naming (sorted alphabetically) |

The cache key for sharded models is a SHA-256 chain across all shard files in shard order. Any shard change invalidates the cache automatically.

**Important:** delete `.save_state/<model-name>/` if you analyzed a model with
an older version of WeightScope, as the cached result will be served without
re-analysis until the hash changes.

---

## UI Tabs Reference

### 📂 Load Model

- **Source** radio — toggle between *Local* and *HuggingFace*.
- **📁 Browse…** button — opens the OS native folder picker (Windows Explorer, macOS Finder, GTK on Linux). Navigate to any drive or network path freely. Requires `tkinter` (ships with standard Python on all platforms). If unavailable, the path textbox remains fully editable by hand.
- **Memory Estimate** JSON panel — `estimated_gb`, `available_gb`, `safe_to_load`, `warning_level` (safe / caution / warning / critical).

### 📊 Overview

Parameter count (exact, from shard headers), unique bit patterns, tensor count, shard count, dtypes found, analysis timestamp, composite file hash. Skipped tensors (unsupported dtypes) are listed.

### 📈 Distribution

- **Histogram** — frequency distribution with adjustable value range sliders and log/linear toggle.
- **Scatter** — value vs. occurrence count (log-y), filterable by singletons (count=1) and statistical outliers (outside 3×IQR). Stratified sampling preserves rare values.

### 🔎 Query

Filter the frequency table by value range and count. Eight built-in presets:

| Preset | Value Range | Count |
|--------|-------------|-------|
| 🌱 Pruning Candidates | \|v\| < 1×10⁻⁴ | any |
| 🔍 Singletons | any | = 1 |
| 📉 Rare Values | any | 1–10 |
| 📈 High-Frequency | any | > 10,000 |
| 🎯 Near Zero | \|v\| < 1×10⁻³ | any |
| ⚡ Extreme Values | \|v\| > 5 | any |
| 🗑️ Low-Count | any | ≤ 4 |
| Custom | user-defined | user-defined |

### 🗜️ Compression

**Quantization simulation** — uniform linear quantization from 4 to 16 bits. Reports MSE, MAE, max error, step size, and level count.

**Low-count removal** — impact of zeroing values that appear fewer than N times.

### ✂️ Pruning

Live sparsity analysis (threshold ε range: 1×10⁻⁶ to 1×10⁻²). Shows prunable parameter count, sparsity %, unique candidates, and the top 50 near-zero values.

### ✂️ Clip & Normalize

Simulate clipping to ±T then normalizing to [−1, 1]. Three threshold modes: Absolute, Standard Deviations (σ), or Percentile. Reports MSE, MAE, SNR (dB), clipped %, bits saved, unique value reduction.

### ⚖️ Compare

Upload two Parquet or CSV files exported from the Export tab to overlay their distributions in a single histogram.

### 💾 Export

**Data export** — frequency table as Parquet (recommended), CSV, or JSON.

**Plot export** — saves histogram and scatter plots at the current filter settings. Set the desired value range and filter options, then click **🖼️ Save Plots**. Formats: PNG, SVG, HTML. PNG and SVG require `kaleido==0.2.1`.

---

## Configuration

```python
# weightscope/config.py  (key settings)

SAVE_STATE_DIR    = Path(".save_state")
MODELS_DIR        = Path("models")
OUTPUT_DIR        = Path("output")
PLUGINS_DIR       = Path("plugins")

DEFAULT_PRUNING_THRESHOLD = 1e-4
MAX_UNIQUE_FOR_PLOT       = 100_000
MEMORY_SAFETY_THRESHOLD   = 0.90

ANALYSIS_CHUNK_SIZE = 4_000_000
ANALYSIS_TEMP_DIR   = Path(tempfile.gettempdir())
```

---

## Plugin System

### Auto-discovery

At startup, `weightscope/plugins/__init__.py` walks `plugins/`. Any subdirectory containing `plugin.py` is imported. Any class subclassing `BasePlugin` is instantiated, registered, and mounted as a new Gradio tab. No changes to core code are required.

### Writing a plugin

```python
# plugins/my_analysis/__init__.py  (empty)
# plugins/my_analysis/plugin.py

import gradio as gr
from weightscope.plugins.base import BasePlugin

class MyPlugin(BasePlugin):
    name        = "My Analysis"
    version     = "0.1.0"
    description = "One-line description."

    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("🔧 My Analysis"):
            out = gr.JSON()
            gr.Button("Run").click(
                fn=self._run,
                inputs=[self.state["current_df"]],
                outputs=[out],
            )

    def _run(self, df):
        if df is None:
            return {"error": "No model loaded"}
        return {"rows": len(df)}
```

Restart WeightScope — the tab appears automatically.

### Shared state

| `self.state` key | Contents |
|---|---|
| `current_df` | Frequency DataFrame: `value` (float32), `count` (int64), `bit_pattern` (str) |
| `current_metadata` | Metadata dict: `total_parameters`, `unique_patterns`, `shard_count`, `dtypes_found`, `file_hash`, `tensors`, … |
| `current_model_id` | String model identifier |

---

## Plugin Reference

WeightScope ships with eight plugins. All are located in the `plugins/`
directory and can be individually disabled by renaming or removing their
folder.

---

### 📐 Extended Stats
**Folder:** `example_plugin` · **Tab:** 📐 Extended Stats

The reference plugin implementation. Computes weighted descriptive statistics
from the global frequency table: mean, standard deviation, skewness, excess
kurtosis, weighted percentiles P1 through P99, IQR, and sparsity at threshold
1×10⁻⁴. Displays results in a sortable table alongside a percentile bar chart.
Intended primarily as a fully-worked example for plugin authors.

---

### 🔢 Entropy & Information Density
**Folder:** `entropy_analysis` · **Tab:** 🔢 Entropy

Computes Shannon entropy H = −Σ p(v) log₂ p(v) from the weight distribution,
where p(v) is the proportion of parameters taking value v. Reports:

- **Shannon entropy** — how much information each weight carries on average
- **Maximum possible entropy** — log₂(unique values), achieved only if all values are equally frequent
- **Relative entropy** — H / H_max (0 = single repeated value; 1 = maximally uniform)
- **Theoretical minimum bits per weight** — the Shannon lower bound; actual storage cannot beat this
- **Redundancy** — actual bits per weight (from storage dtype) minus theoretical minimum
- **Compression ceiling** — the maximum lossless compression ratio theoretically achievable
- **Perplexity** — the effective number of distinct values if the distribution were uniform
- **Gini coefficient** — inequality measure (high = a few values dominate most parameters)

A probability-mass pie chart shows how much of the distribution is concentrated
in the top 10 values, values 11–100, and the remainder. A Top-N bar chart
plots the N most frequent values by probability. Useful for quickly comparing
how compressible a model is without loading it into a training framework.

---

### 🧅 Layer Breakdown
**Folder:** `layer_breakdown` · **Tab:** 🧅 Layer Breakdown

Re-reads each tensor individually (from disk, not from the cached frequency
table) to compute per-layer statistics: mean, standard deviation, min, max,
sparsity at the chosen threshold ε, and unique value count. Results are
displayed in a sortable table. Two bar charts show sparsity and unique value
counts across the top 40 layers by sparsity.

**Use before pruning:** identifying which layers are already sparse (high
sparsity %) indicates which are the safest targets for the Transformer Pruning
plugin, and which layers contain embedding or other large tensors that dominate
the model's size.

*Note: this plugin re-reads tensors from disk. On large models (>7B parameters)
it may take 30–120 seconds to complete.*

---

### 📦 LUT Compression Estimator
**Folder:** `lookup_table_estimator` · **Tab:** 📦 LUT Compression

When a model has very few unique weight values (e.g. MiniCPM4 with 6,820
unique patterns across 433M parameters), it can be stored as an index array
pointing into a small lookup table rather than storing every value in full.
This plugin estimates the compressed file size for each feasible index width:

- Calculates the minimum index bit-width needed to address all unique values
- Computes total storage (index array + lookup table) for 8, 12, 13, 14, and 16-bit indices
- Shows compression ratio vs raw file size and the space saving percentage
- Identifies the break-even point: at how many unique values does each index width stop saving space

A bar chart compares storage sizes across formats. A line chart shows
compression ratio as a function of index width. Results include the LUT table
overhead (unique_count × 4 bytes, storing values as float32).

---

### 🔏 Model Fingerprint & Similarity
**Folder:** `model_fingerprint` · **Tab:** 🔏 Fingerprint

Exports a compact, portable fingerprint of a model's weight distribution as a
256-bucket histogram vector, then computes similarity scores between two
fingerprints.

**Export** — click *💾 Export Fingerprint* to save a JSON file containing
the bucket edges, histogram counts, and normalized probabilities alongside
model metadata. The file is self-contained and can be shared without the
model weights.

**Compare** — upload two fingerprint JSON files to compute:

| Metric | Meaning |
|--------|---------|
| Cosine Similarity | 1.0 = identical distributions; interpreted as "virtually identical / very similar / similar / moderately different / quite different" |
| L1 Distance | Sum of absolute differences in probability mass per bucket |
| L2 Distance | Euclidean distance between probability vectors |
| KL(A‖B) | Information lost when approximating model A with model B |
| KL(B‖A) | Information lost when approximating model B with model A |

Practical uses: verifying a quantized model is close to its float32 source;
detecting distribution drift between a base model and a fine-tune; building a
library of model fingerprints for nearest-neighbour search.

---

### 🚨 Outlier Tensor Report
**Folder:** `outlier_tensor_report` · **Tab:** 🚨 Outlier Report

Scans all tensor metadata from the analysis cache and flags structurally
anomalous tensors without re-reading any model files. Flags raised:

| Flag | Condition |
|------|-----------|
| 🔵 LARGE_TENSOR | Tensor accounts for > N % of total parameters (configurable; default 10 %) |
| 🟡 SINGLETON_HEAVY | Tensor has ≤ 64 parameters (bias terms, layer-norm scalars) |
| 🟢 NORMAL | No flags raised |

Results are displayed in a sortable table with parameter count, shape, dtype,
and percentage of total model parameters. Two charts show the flag distribution
as a pie chart and the tensor size distribution as a log-scale histogram.
Normal tensors are hidden by default but can be shown via checkbox.

Because this plugin uses only cached metadata, results are available instantly
with no disk I/O after the initial analysis.

---

### ✂️ Vocabulary Pruning
**Folder:** `vocab_pruning` · **Tab:** ✂️ Vocab Pruning

> ⚠️ **Experimental.** This plugin modifies and writes model weight files.
> Always keep the original model files intact. Test the pruned model thoroughly
> before using it in any production or research context. Results may vary by
> model architecture, corpus, and pruning ratio.

Removes tokens that do not appear in a reference corpus from the embedding
matrix and LM-head weight matrix, producing a smaller model with an identical
architecture but a reduced vocabulary. The pruned model is saved as a complete
HuggingFace checkpoint (weights + tokenizer + config) ready for inference
testing.

**Algorithm**:

1. Tokenize the reference corpus with the model's own tokenizer
2. Build a keep-set: corpus tokens ∪ all special tokens ∪ first-N vocabulary entries
3. Row-slice the embedding matrix `[vocab_size × hidden]` → `[new_vocab × hidden]`
4. Apply the same slice to the LM-head weight matrix
5. Rewrite the tokenizer via `save_pretrained()` to reflect the new vocabulary
6. Update `vocab_size` in `config.json`

**Corpus sources accepted:**
- Local plain-text file (one document per line)
- HuggingFace dataset name, e.g. `wikitext:wikitext-2-raw-v1` or `roneneldan/TinyStories`
- URL to a plain-text file
- Leave blank → keep only special tokens and first-N entries (most aggressive)

**Controls:**
- *Minimum token frequency* — tokens appearing fewer than N times in the corpus are removed
- *Always keep first N tokens* — protects BOS, EOS, PAD, UNK regardless of corpus coverage
- *Output dtype* — F32 (full precision) or BF16 (half file size, requires BF16-capable inference)

**Output directory naming:** `<output_base>/<model_name>-vocab_pr`
Example: `D:/models/Qwen--Qwen2.5-0.5B-Instruct-vocab_pr`

**Requirements:** `transformers` must be installed (`pip install transformers`).
For HuggingFace dataset corpus sources, also `pip install datasets`.

---

### 🔧 Transformer Pruning
**Folder:** `transformer_pruning` · **Tab:** 🔧 Transformer Pruning

> ⚠️ **Experimental.** This plugin modifies and writes model weight files using
> training-free magnitude-based importance scoring, which is an approximation
> of full activation-based pruning. Quality degrades faster at aggressive
> ratios (> 30 %) than gradient-based methods. The pruned model should be
> tested with inference benchmarks before use. Fine-tuning after pruning is
> strongly recommended for any ratio above 20 %. Always keep the original
> model files intact.

Removes the least-important attention heads and/or FFN neurons from every
transformer layer using weight-magnitude scoring. No forward pass, calibration
data, PyTorch, or GPU is required. The pruned model is saved as a complete
HuggingFace checkpoint ready for inference testing.

**Importance scoring** — for each attention head, the score is the sum of L2
norms of its corresponding slices in the Q, K, V, and O weight matrices.
For each FFN neuron, the score is the L2 norm of its row in the up-projection
(and gate-projection for SwiGLU models). Heads and neurons with the lowest
scores are removed.

This is a training-free approximation; full tools like
[TextPruner](https://github.com/airaria/TextPruner) use activation-based
scoring on calibration data for higher-quality results at aggressive ratios.
Magnitude-based pruning is most reliable at modest ratios (≤ 20–30 %) and
is useful for exploring the pruning landscape before committing to a
calibration-data-based approach.

**Supported architectures:**

| Family | Models |
|--------|--------|
| Llama-style | Llama, Mistral, Mixtral, Qwen2/3, Gemma, MiniCPM, Phi-3, Falcon, Yi, Deepseek, InternLM2 |
| BERT / RoBERTa | BERT, RoBERTa, DeBERTa, ALBERT, CamemBERT, XLM-RoBERTa |
| GPT-2 style | GPT-2, GPT-Neo, CodeGen |
| GPT-NeoX | Pythia, GPT-NeoX |
| OPT | OPT |
| Phi | Phi, Phi-2 |
| DistilBERT | DistilBERT |

GQA (Grouped Query Attention) models — where `num_key_value_heads` <
`num_attention_heads` — are handled correctly. KV head masks are derived from
query head masks so a KV head is only removed when all query heads mapped to
it are also pruned.

**Pruning modes:**

| Mode | Behaviour | Config update |
|------|-----------|---------------|
| **Soft (zero masking)** | Sets pruned head/neuron weights to zero | None — drop-in replacement |
| **Hard (structural)** | Removes pruned rows and columns physically | `num_attention_heads`, `num_key_value_heads`, `intermediate_size` updated |

Hard pruning produces a genuinely smaller model that loads and infers faster.
Soft pruning is safer as a first step — the weights are zeroed but the model
structure is unchanged and can be inspected or further fine-tuned.

**Controls:**
- *Prune attention heads / Prune FFN neurons* — enable either or both
- *Head / FFN pruning ratio* — fraction to remove per layer (0.10 = 10 %)
- *Uniform ratio per layer* — checked: each layer loses the same fraction; unchecked: global ranking removes the weakest heads across all layers regardless of which layer they are in
- *Hard pruning* — structural removal vs zero masking
- *Output dtype* — F32 or BF16

**Output directory naming:** `<output_base>/<model_name>-trans_pr`
Example: `D:/models/amd--AMD-Llama-135m-trans_pr`

**Preview (dry run):** click *🔍 Preview* to see exactly how many heads and
neurons would be removed and to view the head importance score chart for layer
0 before writing any files.

---

## Cache Management

```
.save_state/
└── <sanitized-model-id>/
    ├── analysis_state.parquet   ← frequency DataFrame
    └── metadata.json            ← metadata + composite file hash
```

```bash
# Clear one model's cache
rm -rf .save_state/<model-directory-name>/

# Clear everything
rm -rf .save_state/
```

Temporary DuckDB files (`ws_*.duckdb`) in `ANALYSIS_TEMP_DIR` are deleted
automatically on completion. Any orphaned files left by an interrupted analysis
can be deleted manually.

---

## Running the Tests

```bash
# All tests
python -m pytest tests/ -v

# A specific class
python -m pytest tests/test_analyzer.py::TestQuantizationSimulation -v

# With coverage (requires pytest-cov)
python -m pytest tests/ --cov=weightscope --cov-report=term-missing
```

The test suite (49 tests) covers the core analyzer, streaming engine, sharded
model analysis, session cache, utility helpers, and plugin system.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Add or update tests in `tests/` for any changed behaviour.
3. Ensure all tests pass: `python -m pytest tests/ -v`
4. Open a pull request with a clear description of the change.

**Adding a new tab** — create a file in `weightscope/ui/tabs/`, follow the
pattern of any existing tab, then wire it in `weightscope/ui/app_builder.py`.

**Adding dtype support** — add to `SUPPORTED_DTYPES` in `config.py`, then
add a conversion branch in `_np_tensor_to_keys()` or `_raw_bytes_to_keys()`
in `weightscope/core/analyzer.py`.

**Adding a plugin** — see [Plugin System](#plugin-system) above. No core
code changes needed.

---

## License

WeightScope is free software released under the
**GNU Affero General Public License v3.0** (AGPL-3.0).

You may use, modify, and distribute it under the terms of that license. If
you run a modified version as a network service, you must make the modified
source code available to users of that service.

See the `LICENSE` file or <https://www.gnu.org/licenses/agpl-3.0.html> for
the full text.
