# WeightScope Features

Comprehensive documentation of all WeightScope features and capabilities.

---

## 📊 Weight Distribution Analysis

### Overview
Analyze the statistical distribution of weight values in any `.safetensors` model.

### Features
- **Bit-Exact Counting**: Count exact occurrences of every unique weight value using bit-pattern matching
- **Histogram Visualization**: Interactive histograms with log/linear scale toggle
- **Scatter Plots**: Value vs. frequency scatter plots with outlier detection
- **Range Filtering**: Filter distributions by value range for focused analysis

### Use Cases
- Identify weight value clustering patterns
- Detect anomalous weight distributions
- Compare distributions across model layers
- Validate weight initialization schemes

---

## 🗜️ Compression Analysis

### Overview
Simulate various compression techniques to estimate potential size reductions.

### Features

#### Quantization Simulation
- **Bit-Width Reduction**: Simulate 4-bit to 16-bit quantization
- **Error Metrics**: MSE, MAE, and max error calculation
- **Error Metric Selection**: Choose between MSE or MAE optimization

#### Index-Based Compression
- **Lookup Table Detection**: Automatically detect if unique values ≤ 65,535 (lossless 16-bit indexing possible)
- **Memory Savings Estimate**: Calculate exact MB savings for each compression method

#### Fixed-Point Analysis
- **Range Detection**: Identify models suitable for fixed-point representation
- **Precision Loss Estimation**: Calculate quantization error for Q-format fixed-point

### Use Cases
- Pre-deployment compression planning
- Hardware compatibility assessment (edge devices, mobile, etc.)
- Research on quantization-aware training requirements

---

## ✂️ Pruning Analysis

### Overview
Identify weight pruning candidates using multiple strategies.

### Features

#### Magnitude-Based Pruning
- **Threshold Slider**: Adjust pruning threshold (ε) from 1e-6 to 1e-2
- **Sparsity Calculation**: Automatic sparsity percentage calculation
- **Candidate Export**: Export pruning masks for fine-tuning

#### Frequency-Based Pruning (Novel)
- **Low-Count Removal**: Identify weights with rare value occurrences (count ≤ N)
- **Uniqueness Reduction**: Measure impact on unique value count
- **Compression Gain**: Estimate compression improvement from reduced uniqueness

### Use Cases
- Model size reduction research
- Studying weight importance by magnitude vs. frequency
- Preparing models for sparse inference engines

---

## ✂️ Clip & Normalize

### Overview
Simulate clipping extreme weight values and normalizing to a standard range.

### Features

#### Threshold Modes
- **Absolute**: Clip at fixed value (e.g., |w| < 5.0)
- **Standard Deviations (σ)**: Clip at N standard deviations from mean
- **Percentile**: Clip at N-th percentile of weight distribution

#### Metrics
- **MSE/MAE**: Reconstruction error after clip + normalize + de-normalize
- **SNR (dB)**: Signal-to-noise ratio for quality assessment
- **Bits Saved**: Theoretical bit reduction from smaller dynamic range
- **Clipped %**: Fraction of parameters affected by clipping

### Use Cases
- Dynamic range optimization for quantization
- Outlier robustness studies
- Preparing models for symmetric quantization (INT8, FP8)

---

## 🔎 Query System

### Overview
Powerful filtering and search capabilities for weight exploration.

### Features

#### Query Presets
| Preset | Description |
|--------|-------------|
| 🌱 Pruning Candidates | \|v\| < 1e-4 |
| 🔍 Singletons | count = 1 |
| 📉 Rare Values | count 1-10 |
| 📈 High-Frequency | count > 10,000 |
| 🎯 Near Zero | \|v\| < 1e-3 |
| ⚡ Extreme Values | \|v\| > 5 |
| 🗑️ Low-Count Removal | count ≤ 4 |
| Custom | Manual filter configuration |

#### Search Capabilities
- **Value Range**: Filter by minimum/maximum weight value
- **Count Range**: Filter by occurrence frequency
- **Text Search**: Search by bit pattern (e.g., `0x3F800000`) or value substring

### Use Cases
- Finding specific weight values for analysis
- Identifying pruning candidates quickly
- Exporting subsets for downstream processing

---

## ⚖️ Model Comparison

### Overview
Compare weight distributions across two different models.

### Features
- **Overlay Histograms**: Visual comparison of two distributions
- **Statistical Metrics**: Compare unique counts, ranges, sparsity
- **Export Comparison**: Save comparison data for reports

### Use Cases
- Base model vs. fine-tuned model comparison
- Pre-pruning vs. post-pruning analysis
- Different architecture comparisons

---

## 💾 Export System

### Overview
Export analysis data in multiple formats for downstream processing.

### Supported Formats
| Format | Best For |
|--------|----------|
| **Parquet** | Large datasets, fast I/O, Python/R analysis |
| **CSV** | Excel, Google Sheets, universal compatibility |
| **JSON** | Web applications, JavaScript processing |

### Export Options
- Full weight distribution data
- Current query filter results
- Plot images (PNG, SVG, HTML)

### Use Cases
- Offline analysis in Jupyter notebooks
- Sharing results with collaborators
- Archiving analysis for publications

---

## 🔐 Session Caching

### Overview
Automatic caching of analysis results for instant re-loading.

### Features
- **Hash-Based Validation**: SHA256 hash of model file ensures cache validity
- **Automatic Detection**: Re-loading same model skips re-analysis
- **Organized Storage**: `.save_state/{model_name}/` structure

### Cache Contents
- `analysis_state.parquet`: Full weight distribution DataFrame
- `metadata.json`: Model info, tensor count, analysis timestamp
- `plots/`: Cached plot images (future feature)

### Use Cases
- Iterative analysis without re-processing
- Sharing cached analysis with team members
- Resuming analysis after application restart

---

## 🛡️ Memory Safety

### Overview
Protects against loading models that exceed available RAM.

### Features
- **Pre-Load Estimation**: Calculate RAM requirements before loading
- **Warning Levels**: Safe, Caution, Warning, Critical
- **Automatic Blocking**: Prevents loading if >90% RAM would be used

### Memory Calculation
Estimated RAM = (parameters × 4 bytes × 1.5 overhead)
- 4 bytes: F32 weight size
- 1.5×: Overhead for NumPy arrays + pandas DataFrame + indices

### Use Cases
- Preventing system crashes on large models
- Planning hardware requirements for analysis
- Safe shared-environment usage

---

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 32 GB+ |
| **Storage** | 2 GB free | 10 GB+ SSD |
| **Python** | 3.10 | 3.13+ |
| **Internet** | Optional | Required for HF download |

### Model Size Guidelines
| Model Size | RAM Required* | Status |
|------------|--------------|--------|
| ≤ 1B params | 4-8 GB | ✅ Safe |
| 1-7B params | 8-32 GB | ⚠️ Caution |
| 7-13B params | 32-64 GB | ⚠️⚠️ High Risk |
| ≥ 70B params | 140-280 GB | ❌ Block |
##### \*note that these values have not been thoroughly tested/verified
---

## 🤝 Contributing

WeightScope is open for contributions! See `CONTRIBUTING.md` for guidelines.

### Feature Requests
- Per-layer analysis breakdown
- Batch model processing
- Plugin system for custom analysis
- Interactive weight editing simulation

### Bug Reports
Please include:
- Model name and size
- Error message and traceback
- System specifications (RAM, Python version)
