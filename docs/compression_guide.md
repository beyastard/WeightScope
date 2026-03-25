# Compression Research Guide

A comprehensive guide to using WeightScope for model compression research.

---

## 🎯 Overview

Model compression is essential for deploying neural networks on resource-constrained devices. WeightScope provides simulation tools to estimate compression potential **before** applying actual transformations.

---

## 📊 Understanding Weight Distributions

### Why Distribution Matters

Weight value distributions directly impact compression efficiency:

| Distribution Property | Compression Impact |
|----------------------|-------------------|
| **Unique Value Count** | Determines lookup table feasibility |
| **Dynamic Range** | Affects quantization bit requirements |
| **Sparsity** | Enables sparse format compression |
| **Outlier Presence** | Wastes bits in fixed-point representation |

### Analyzing Your Model

1. **Load your model** in WeightScope
2. **Check Overview tab** for unique value count
3. **Review Distribution tab** for range and clustering
4. **Use Query tab** to identify compression candidates

---

## 🗜️ Compression Techniques

### 1. Index-Based Compression (Lossless)

**When to Use**: Unique values ≤ 65,535

**How It Works**:
Original: Store each weight as 32-bit float
Compressed: Store 16-bit index + 32-bit lookup table
Example:
- 134M parameters, 50K unique values
- Original: 134M × 4 bytes = 536 MB
- Compressed: 134M × 2 bytes + 50K × 4 bytes = 268 MB + 0.2 MB
- Savings: ~50% (268 MB)

**WeightScope Analysis**:
1. Check **Compression tab** → "Unique Values"
2. If ≤ 65,535 → Index compression is feasible
3. Review "Memory Saved MB" estimate

**Research Questions**:
- Does fine-tuning increase or decrease unique value count?
- Which layers have the most unique values?
- Can we merge similar values to reach the 65K threshold?

---

### 2. Quantization (Lossy)

**When to Use**: Any model (trade accuracy for size)

**How It Works**:
F32 (32-bit) → INT8 (8-bit) = 75% size reduction
F32 (32-bit) → INT4 (4-bit) = 87.5% size reduction

**WeightScope Analysis**:
1. Go to **Compression tab**
2. Adjust "Quantization Bits" slider (4-16)
3. Review MSE, MAE, and max error
4. Compare error metrics across bit widths

**Expected Results** (Typical LLM):
| Bits | Size Reduction | MSE | Perplexity Impact |
|------|---------------|-----|-------------------|
| 16-bit | 50% | ~1e-8 | Negligible |
| 8-bit | 75% | ~1e-4 | -0.5% to -2% |
| 4-bit | 87.5% | ~1e-2 | -2% to -10% |

**Research Questions**:
- What's the minimum bit-width for acceptable accuracy?
- Do certain layers require higher precision?
- How does quantization interact with pruning?

---

### 3. Clip + Normalize + Quantize (Hybrid)

**When to Use**: Models with extreme outliers

**How It Works**:
Step 1: Clip outliers (e.g., |w| > 3σ)
Step 2: Normalize to [-1, 1]
Step 3: Quantize to INT8/FP8
Benefit: Smaller range → finer quantization steps → lower error

**WeightScope Analysis**:
1. Go to **Clip & Normalize tab**
2. Try threshold modes: Absolute, σ, Percentile
3. Review "Bits Saved" and "SNR (dB)"
4. Combine with quantization simulation

**Example Workflow**:
Original range: [-15.8, +15.8] → 31.6 range
After clip ±3σ: [-5.2, +5.2] → 10.4 range
After normalize: [-1, +1] → 2.0 range
Bits saved: log2(31.6 / 2.0) ≈ 4 bits
Compression gain: 4/32 = 12.5% additional savings

**Research Questions**:
- What's the optimal clipping threshold per architecture?
- Does clipping require fine-tuning to recover accuracy?
- Can we learn per-layer clipping thresholds?

---

### 4. Frequency-Based Pruning + Compression

**When to Use**: Models with long-tail value distributions

**How It Works**:
Step 1: Identify rare values (count ≤ 4)
Step 2: Replace with nearest common value
Step 3: Apply index or quantization compression
Benefit: Fewer unique values → better compression

**WeightScope Analysis**:
1. Go to **Compression tab**
2. Use "Low-Count Removal Test" (count ≤ 4)
3. Review "Unique Reduction %" and "Compression Gain"
4. Export pruned weights for fine-tuning

**Expected Results** (AMD-Llama-135m example):
Original: 71.3M unique values
After removing count≤4: ~45M unique values
Reduction: 37% fewer unique values
Impact: <1% of total parameters affected

**Research Questions**:
- Do rare values encode important edge-case knowledge?
- How many fine-tuning steps to recover accuracy?
- Is frequency pruning complementary to magnitude pruning?

---

## 🧪 Compression Research Workflow

### Phase 1: Baseline Analysis
1. Load model in WeightScope
2. Record: unique values, range, sparsity
3. Export baseline distribution (Parquet)

### Phase 2: Simulation
1. Test index compression feasibility
2. Simulate quantization at 4/8/16 bits
3. Test clip+normalize at various thresholds
4. Simulate low-count removal
5. Record all metrics (MSE, bits saved, etc.)

### Phase 3: Decision
1. Choose compression strategy based on:
   - Target size reduction
   - Acceptable accuracy loss
   - Hardware constraints
2. Document expected metrics

### Phase 4: Implementation
1. Apply chosen compression to model
2. Fine-tune if necessary (10-100 epochs)
3. Evaluate on validation set
4. Compare actual vs. predicted metrics

### Phase 5: Publication
1. Export WeightScope analysis data
2. Include distribution plots in paper
3. Document compression pipeline
4. Share cached analysis for reproducibility

---

## 📈 Compression Metrics Explained

### MSE (Mean Squared Error)
MSE = Σ(original - compressed)² / N
Interpretation:
- < 1e-6: Excellent (negligible impact)
- 1e-6 to 1e-4: Good (minor impact)
- 1e-4 to 1e-2: Moderate (fine-tuning recommended)
- 1e-2: Poor (significant accuracy loss expected)

### SNR (Signal-to-Noise Ratio, dB)
SNR = 10 × log10(signal_power / noise_power)
Interpretation:
- 60 dB: Excellent
- 40-60 dB: Good
- 20-40 dB: Moderate
- < 20 dB: Poor

### Bits Saved
Bits Saved = log2(original_range / compressed_range)
Example:
- Original range: 31.6 ([-15.8, +15.8])
- Compressed range: 2.0 ([-1, +1])
- Bits saved: log2(31.6/2.0) ≈ 3.98 bits
- Compression gain: 3.98/32 ≈ 12.4%

---

## 🔬 Advanced Research Topics

### 1. Per-Layer Compression Analysis
- Different layers may have different optimal compression
- Attention layers often more sensitive than MLP
- Embedding layers may benefit from separate treatment

### 2. Mixed-Precision Compression
- Keep sensitive layers at F16/F32
- Compress robust layers to INT8/INT4
- Use WeightScope to identify layer sensitivity

### 3. Compression + Fine-Tuning Co-Design
- Jointly optimize compression and fine-tuning
- Use WeightScope metrics as loss function components
- Explore compression-aware training objectives

### 4. Hardware-Aware Compression
- Target specific hardware (mobile GPU, TPU, NPU)
- Consider memory bandwidth, compute patterns
- Use WeightScope to estimate hardware-specific gains

---

## 📚 References

1. Jacob, B. et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR.
2. Frantar, E. et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR.
3. Dettmers, T. et al. (2022. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." EMNLP.
4. Lin, J. et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression." arXiv.

---

## 🤝 Contributing Research

Have you used WeightScope for compression research? Share your findings:
- Open a GitHub issue with your results
- Submit a pull request to add to this guide
- Cite WeightScope in your publications

---
