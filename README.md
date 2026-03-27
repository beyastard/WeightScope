# 🔍 WeightScope

**SafeTensors Model Analyzer** — a research tool for inspecting, visualizing,
and simulating the weight distributions of machine-learning models stored in
the `.safetensors` format.

> **Version 0.2.1** · AGPL-3.0 · Copyright © 2026 Bryan K Reinhart & BeySoft

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
13. [Cache Management](#cache-management)
14. [Running the Tests](#running-the-tests)
15. [Contributing](#contributing)
16. [License](#license)

---

## Overview

WeightScope loads `.safetensors` model files — locally or directly from the
HuggingFace Hub — and analyses every weight value at the bit-pattern level.
Instead of sampling or approximating, it counts **every unique floating-point
or integer pattern** that appears in the model, producing an exact frequency
table that drives all downstream analysis.

Analysis is **streaming and memory-bounded**: tensors are processed one at a
time and frequency counts are accumulated in a temporary DuckDB database.
Models of any size can be analysed on modest hardware — the peak RAM footprint
is determined by the largest single tensor, not the total model size.  Both
single-file and **sharded** models are supported.

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
| 📂 **Load Model** | Load single-file or sharded models locally or from HuggingFace Hub, with per-shard memory-safety checks |
| 📊 **Overview** | Parameter count, unique pattern count, shard count, tensor inventory, dtype summary |
| 📈 **Distribution** | Interactive histogram (log/linear) and scatter plot with singleton/outlier filters |
| 🔎 **Query** | Filter weights by value range and occurrence count; 8 built-in presets + Custom mode |
| 🗜️ **Compression** | Uniform quantization simulation (4–16 bit, incl. INT4/INT8) and low-count removal impact |
| ✂️ **Pruning** | Live sparsity analysis — see exactly how many parameters fall below any threshold ε |
| ✂️ **Clip & Normalize** | Simulate clipping outliers and normalizing to [−1, 1]; reports MSE, MAE, SNR, bits saved |
| ⚖️ **Compare** | Side-by-side distribution overlay of two exported analyses |
| 💾 **Export** | Save the weight frequency table as Parquet, CSV, or JSON |
| 🔌 **Plugins** | Drop-in tab extensions — no core code changes needed |

---

## Supported Formats

WeightScope uses a **hybrid read strategy** to handle all dtype variants.
Numpy-compatible dtypes are loaded via `safe_open`; BF16 and FP8 variants —
which cause `safe_open` to raise `TypeError: data type 'bfloat16' not
understood` — are read as raw bytes and converted to float32 manually.
Tensors with unrecognised dtypes are skipped and reported in the session
metadata.

| dtype | Storage | Read path | Notes |
|-------|---------|-----------|-------|
| `float32` | 4 bytes | `safe_open` | Bit-exact pattern counting via uint32 view |
| `float16` | 2 bytes | `safe_open` | Upcast to float32; bit-exact uint32 key |
| `bfloat16` | 2 bytes | Raw bytes | uint16 → uint32 left-shift 16; no numpy involvement |
| `float8_e4m3fn` | 1 byte | Raw bytes | Upcast to float32 |
| `float8_e5m2` | 1 byte | Raw bytes | Upcast to float32 |
| `int8` | 1 byte | `safe_open` | Cast to float32 |
| `uint8` | 1 byte | `safe_open` | Cast to float32 |
| `int4` (packed) | ½ byte | `safe_open` | Two 4-bit signed values per byte; nibble-unpacked with sign extension |

**Supported model families (non-exhaustive):** Llama, Mistral, Qwen, Phi,
Gemma, Falcon, MiniCPM, BERT, RoBERTa, Whisper, CLIP, Stable Diffusion, FLUX,
ControlNet, and any other model saved in `.safetensors` format — both
single-file and sharded.

---

## Requirements

- Python 3.10 or later (only 3.13 tested)
- See `requirements.txt` for the full dependency list

Core runtime dependencies:

```
safetensors>=0.4.0
numpy>=2.0.0
pandas>=2.0.0
pyarrow>=14.0.0
duckdb>=1.5.0
gradio>=4.0.0
plotly>=5.18.0
kaleido>=0.2.1
huggingface_hub>=0.20.0
psutil>=5.9.0
```

> **Note:** `duckdb` replaces the earlier SQLite backend.  It performs the
> cross-shard frequency aggregation (`GROUP BY key, SUM(count)`) in C++ and
> reduces analysis time on large models from tens of minutes to under two
> minutes on typical hardware.

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
2. Enter the full path to the model directory.
3. Click **🚀 Load & Analyse**.

WeightScope auto-detects the layout — single file or sharded — and processes
all shards as a single unified analysis.

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
   `black-forest-labs/FLUX.2-klein-base-4B`.
3. Click **🚀 Load & Analyse**.

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
| `WEIGHTSCOPE_CHUNK_SIZE` | `4000000` | In-memory buffer size before DuckDB flush |
| `WEIGHTSCOPE_TEMP_DIR` | OS temp dir | DuckDB working directory during analysis |

```bash
# Expose on LAN, use /mnt/scratch for DuckDB temp files
WEIGHTSCOPE_HOST=0.0.0.0 WEIGHTSCOPE_TEMP_DIR=/mnt/scratch python app.py
```

---

## Project Structure

```
WeightScope/
├── app.py                              ← Entry point (~28 lines)
├── requirements.txt
├── MIGRATION.md                        ← v0.1 → v0.2 import-path guide
├── README.md
│
├── weightscope/                        ← Core package
│   ├── __init__.py                     ← APP_NAME, APP_VERSION
│   ├── config.py                       ← All constants & env-var overrides
│   │
│   ├── core/                           ← Business logic (no UI dependencies)
│   │   ├── __init__.py
│   │   ├── loader.py                   ← ModelLoader: shard discovery, local + HF loading
│   │   ├── analyzer.py                 ← WeightAnalyzer: streaming analysis + simulations
│   │   └── cache.py                    ← SessionCache: disk persistence + export
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py                  ← sanitize_model_name, compute_file_hash,
│   │                                      get_available_ram_gb, format_number, …
│   │
│   ├── ui/                             ← Gradio UI (depends on core, not vice-versa)
│   │   ├── __init__.py
│   │   ├── app_builder.py              ← build_app(): assembles tabs + wires events
│   │   ├── plotting.py                 ← create_histogram, create_scatter, …
│   │   └── tabs/                       ← One file per Gradio tab
│   │       ├── __init__.py
│   │       ├── load_model.py           ← Shard-aware loader UI + combined hash cache key
│   │       ├── overview.py
│   │       ├── distribution.py
│   │       ├── query.py
│   │       ├── compression.py
│   │       ├── pruning.py
│   │       ├── clip_normalize.py
│   │       ├── compare.py
│   │       └── export.py
│   │
│   └── plugins/                        ← Plugin registry & base class
│       ├── __init__.py                 ← PluginRegistry, auto-discovery engine
│       └── base.py                     ← BasePlugin ABC
│
├── plugins/                            ← User-installable plugin packages
│   └── example_plugin/
│       ├── __init__.py
│       └── plugin.py                   ← WeightStatisticsPlugin (extended stats)
│
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py                ← 49 unit tests
│
├── docs/
│   ├── compression_guide.md
│   ├── features.md
│   └── pruning_guide.md
│
├── examples/
│   └── amd-llama-135m-analysis.md
│
├── models/                             ← HuggingFace download cache (git-ignored)
└── output/                             ← Default export destination (git-ignored)
```

### Dependency direction

```
ui/tabs/* → ui/app_builder.py → ui/plotting.py
                              → core/*
                              → plugins/*
core/* → utils/*
       → config.py
plugins/* → core/*   (optional — plugins may use the analyzer directly)
```

The `core/` package has **no dependency on Gradio** and can be imported and
used in scripts or notebooks independently.

---

## Memory Architecture

WeightScope is designed to analyse models larger than available RAM without
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

WeightScope handles all common shard layouts automatically.

### Detection order (local models)

| Priority | Pattern | Example |
|----------|---------|---------|
| 1 | `model.safetensors` | Single-file model |
| 2 | `model-NNNNN-of-MMMMM.safetensors` | HF standard sharding |
| 3 | Any `*.safetensors` | Non-standard naming (sorted alphabetically) |

Shards in priority 2 are sorted numerically by their shard index so processing
order is always correct regardless of filesystem ordering.

### HuggingFace remote models

`load_remote_model()` calls `list_repo_files()` to discover all shard
filenames before downloading anything, applies the same priority ordering, and
downloads every shard sequentially.  Already-cached shards are not
re-downloaded.

### Cache key for sharded models

The cache key for a sharded model is a SHA-256 digest computed by chaining
the individual SHA-256 hash of each shard file **in shard order**.  If any
shard changes, the composite hash changes and the cache is invalidated.

### Important: clearing a stale cache

If you analysed a model with an older version of WeightScope and then upgrade,
the cached result will be served from `.save_state/` without re-analysis.
Delete the relevant subdirectory (or the entire `.save_state/` folder) to
force a fresh analysis:

```bash
# Clear one model's cache
rm -rf .save_state/my-model-name/

# Clear everything
rm -rf .save_state/
```

---

## UI Tabs Reference

### 📂 Load Model

Handles both local and remote model loading with pre-flight memory estimation.

- **Source** — toggle between *Local* (filesystem path) and *HuggingFace*
  (model ID string such as `meta-llama/Llama-3.2-1B`).
- **Shard detection** — auto-detected; the status message reports the number
  of shards found, e.g. `✅ Loaded: Llama-3.2-1B (1,235,814,400 params, 2 shards)`.
- **Memory Estimate** — JSON panel showing `estimated_gb`, `available_gb`,
  `safe_to_load`, and a `warning_level` of `safe / caution / warning / critical`.
  The estimate is conservative (0.5 bytes × param count) since the streaming
  engine never holds the full model in RAM.

### 📊 Overview

Displays a summary table after a model is loaded:

- Total parameters (exact, counted from shard headers — no config.json needed)
- Unique bit patterns
- Tensor count and shard count
- Dtypes found across all shards
- Analysis timestamp and composite file hash prefix
- Any skipped tensors (unsupported dtypes) are listed

### 📈 Distribution

Two linked plots driven by the loaded frequency table:

- **Histogram** — frequency distribution of weight values, with adjustable
  min/max range sliders and a log/linear count toggle.  Range `[-20, 20]`
  covers virtually all models; narrow it to zoom into the main mass.
- **Scatter** — value vs. occurrence count (log-y axis), with filter checkboxes
  for *Singletons* (count = 1) and *Statistical Outliers* (outside 3×IQR).
  Uses stratified sampling to ensure rare values are never dropped when the
  dataset is large.

### 🔎 Query

Filter the weight frequency table by value range and occurrence count.

**Built-in presets:**

| Preset | Value Range | Count Range |
|--------|-------------|-------------|
| 🌱 Pruning Candidates | \|v\| < 1×10⁻⁴ | any |
| 🔍 Singletons | any | = 1 |
| 📉 Rare Values | any | 1–10 |
| 📈 High-Frequency | any | > 10,000 |
| 🎯 Near Zero | \|v\| < 1×10⁻³ | any |
| ⚡ Extreme Values | \|v\| > 5 | any |
| 🗑️ Low-Count | any | ≤ 4 |
| Custom | user-defined | user-defined |

A free-text search field additionally filters by bit pattern string or value
substring.  Results are capped at 100 rows, sorted by descending count.

### 🗜️ Compression

Two independent simulations:

**Quantization simulation** — models uniform linear quantization to any target
bit-width from 4 (INT4) to 16 (INT16/FP16).  Reports MSE, MAE, max error,
step size, and the resulting number of quantization levels.  Useful for
estimating how much precision is lost when quantizing a BF16 or FP32 model.

**Low-count removal simulation** — shows the impact of zeroing out every
weight value that appears fewer than N times (threshold controlled by a
slider).  Reports removed parameter count, unique-value reduction %, and
estimated compression gain.

### ✂️ Pruning

Interactive sparsity analysis driven by a threshold slider ε (range 1×10⁻⁶
to 1×10⁻²).  The summary updates live as the slider moves and shows:

- **Prunable parameters** — total occurrences of values with |v| ≤ ε
- **Sparsity %** — prunable / total × 100
- **Unique candidates** — number of distinct near-zero values
- A table of the top 50 candidates sorted by descending frequency

### ✂️ Clip & Normalize

Simulates two sequential operations applied to the entire weight distribution:

1. **Clip** — truncate all weights to [−T, +T] where T is the chosen threshold.
2. **Normalize** — linearly rescale the clipped range to [−1, 1].

Three threshold modes:

| Mode | Meaning |
|------|---------|
| Absolute | T entered directly |
| Standard Deviations (σ) | T = N × weighted standard deviation |
| Percentile | T = value at the N-th weighted percentile |

Output metrics: MSE, MAE, SNR (dB), clipped parameter %, theoretical bits
saved, unique value reduction %, and a dynamic range comparison bar chart.

### ⚖️ Compare

Upload two Parquet (or CSV) files previously exported from the Export tab to
overlay their weight distributions in a single histogram.  Useful for
comparing a base model against a fine-tuned checkpoint, two quantization
configurations, or different shard subsets.

### 💾 Export

Export the in-memory frequency table for the currently loaded model.

| Format | Use case |
|--------|---------|
| **Parquet** | Recommended — compact, typed, fast to reload in pandas/polars |
| **CSV** | Human-readable, compatible with Excel and most BI tools |
| **JSON** | Interoperability with JavaScript or REST APIs |

The output file is named `<model-id>_weights.<ext>` and written to the
configured output directory (default `./output`).

---

## Configuration

All constants live in `weightscope/config.py` and can be overridden by
environment variables before launch.

```python
# weightscope/config.py  (selected settings)

SAVE_STATE_DIR    = Path(".save_state")   # analysis cache
MODELS_DIR        = Path("models")        # HuggingFace download cache
OUTPUT_DIR        = Path("output")        # default export directory
PLUGINS_DIR       = Path("plugins")       # plugin discovery root

DEFAULT_PRUNING_THRESHOLD = 1e-4
MAX_UNIQUE_FOR_PLOT       = 100_000       # plot downsampling cap
MEMORY_SAFETY_THRESHOLD   = 0.90         # fraction of available RAM

ANALYSIS_CHUNK_SIZE = 4_000_000          # in-memory buffer (entries) before DuckDB flush
ANALYSIS_TEMP_DIR   = Path(tempfile.gettempdir())  # DuckDB working directory
```

Adding support for a new dtype requires two steps:

1. Add an entry to `SUPPORTED_DTYPES` in `config.py`.
2. Add a conversion branch in `_np_tensor_to_keys()` or `_raw_bytes_to_keys()`
   in `weightscope/core/analyzer.py` (use the raw-bytes path for dtypes numpy
   does not natively understand).

---

## Plugin System

WeightScope supports drop-in plugins that add new Gradio tabs without touching
any core code.

### How auto-discovery works

At startup, `weightscope/plugins/__init__.py` walks the `plugins/` directory.
Any sub-directory that contains a `plugin.py` file is imported.  Any class
inside that file which subclasses `BasePlugin` is instantiated and registered
with the `PluginRegistry` singleton.  Its `mount()` method is then called
inside the open `gr.Blocks` context, appending a new tab to the UI.

### Writing a plugin

**Step 1** — create a directory inside `plugins/`:

```
plugins/
└── my_analysis/
    ├── __init__.py      ← can be empty
    └── plugin.py
```

**Step 2** — implement `BasePlugin` in `plugin.py`:

```python
import gradio as gr
from weightscope.plugins.base import BasePlugin

class MyAnalysisPlugin(BasePlugin):
    name        = "My Analysis"
    version     = "0.1.0"
    description = "Adds a custom analysis tab."

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
        return {"rows": len(df), "mean": float(df["value"].mean())}
```

**Step 3** — restart WeightScope.  Your tab appears automatically.

### Shared state

`inject_state()` is called before `mount()` and populates `self.state`:

| Key | Contents |
|-----|---------|
| `current_df` | The weight frequency DataFrame: columns `value` (float32), `count` (int64), `bit_pattern` (str) |
| `current_metadata` | Model metadata dict: `total_parameters`, `unique_patterns`, `tensor_count`, `shard_count`, `dtypes_found`, `file_hash`, `tensors`, … |
| `current_model_id` | String model identifier (directory name or HF model ID) |

### Using core modules in a plugin

```python
from weightscope.core.analyzer import WeightAnalyzer
from weightscope.core.loader   import find_safetensors_shards
from weightscope.utils         import format_number
```

### Bundled example plugin

`plugins/example_plugin/plugin.py` implements `WeightStatisticsPlugin`, which
adds an **Extended Stats** tab with weighted percentiles (P1–P99), skewness,
excess kurtosis, IQR, and a percentile bar chart.  It serves as a
fully-worked reference implementation.

---

## Cache Management

Analysis results are persisted in `.save_state/` so re-loading the same model
is instant.

```
.save_state/
└── <sanitized-model-id>/
    ├── analysis_state.parquet   ← frequency DataFrame
    └── metadata.json            ← metadata + composite file hash
```

The cache is validated on every load by comparing the stored hash against the
hash of the current file(s).  If any shard has changed (or if the cache was
written by an older version of WeightScope), the hash will not match and a
fresh analysis runs automatically.

```bash
# Manually clear a single model's cache
rm -rf .save_state/<model-directory-name>/

# Clear the entire cache
rm -rf .save_state/
```

During analysis, a temporary DuckDB file is created in `ANALYSIS_TEMP_DIR` and
deleted automatically on completion.  If analysis is interrupted, any orphaned
`ws_*.duckdb` files in that directory can be deleted safely.

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

The test suite (49 tests) covers:

- `WeightAnalyzer` — initialization, all simulation methods, INT4 unpacking,
  BF16 conversion, error handling when `df` is `None`
- `_StreamingCounter` — DuckDB feed/finalise/cleanup cycle
- Sharded model analysis — cross-shard frequency count correctness
- `SessionCache` — save/load roundtrip, hash validation, invalidation,
  listing, all three export formats
- Utility helpers — `sanitize_model_name`, `format_number`
- Plugin system — `BasePlugin` ABC enforcement, `PluginRegistry` validation,
  auto-discovery of the example plugin

---

## Contributing

1. Fork the repository and create a feature branch.
2. Add or update tests in `tests/` for any changed behaviour.
3. Ensure the full test suite passes: `python -m pytest tests/ -v`
4. Open a pull request with a clear description of the change.

**Adding a new analysis tab** — create a file in `weightscope/ui/tabs/`,
follow the pattern of any existing tab (return components + callbacks from a
`create_*_tab()` function), then wire it in `weightscope/ui/app_builder.py`.

**Adding dtype support** — add an entry to `SUPPORTED_DTYPES` in `config.py`,
then add a conversion branch in `_np_tensor_to_keys()` or
`_raw_bytes_to_keys()` in `weightscope/core/analyzer.py`.

**Adding a plugin** — see the [Plugin System](#plugin-system) section above.
Plugins require no changes to the core codebase.

---

## License

WeightScope is free software released under the
**GNU Affero General Public License v3.0** (AGPL-3.0).

You may use, modify, and distribute it under the terms of that license.  If
you run a modified version as a network service, you must make the modified
source code available to users of that service.

See the `LICENSE` file or <https://www.gnu.org/licenses/agpl-3.0.html> for
the full text.
