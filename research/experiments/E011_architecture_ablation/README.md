# Experiment E011: Architecture Ablation (Standard vs ALBERT vs MobileBERT vs Hybrid)

### Objective
Isolate architectural design patterns (cross-layer weight sharing, intra-block linear bottlenecks, gated depthwise 1D convolutions) to determine optimal structural efficiency for mobile System-1 edge agents.

### Candidate Architectures Evaluated
1. **Standard Tiny Transformer**: Post-LN/Pre-LN standard Transformer Encoder stack with Multi-Head Self Attention (MHA) and 2-layer MLP feed-forward networks.
2. **ALBERT-style**: Factorized embedding matrix ($V \to E \to D$) combined with full cross-layer weight sharing across all transformer passes.
3. **MobileBERT-style**: Intra-block linear projection bottlenecks ($d_{model} \to d_{bottleneck} \to d_{model}$) compressing attention rank while preserving representation dimension.
4. **MobiMIND Hybrid**: Multi-Head Attention paired with Gated Depthwise 1D Convolutional Feed-Forward Blocks (SiLU gating + depthwise 3x1 kernel).

### Cross-Architecture Comparison (Across 10M, 25M, and 50M Tiers)

#### Latency Efficiency (P50 on Snapdragon 6 Gen 4 CPU)
| Architecture | 10M Tier P50 | 25M Tier P50 | 50M Tier P50 | Average Latency Rank |
| :--- | :--- | :--- | :--- | :--- |
| **MobiMIND Hybrid** | **9.15 ms** | **22.40 ms** | **44.90 ms** | **1 (Fastest)** |
| **MobileBERT** | 10.45 ms | 26.80 ms | 53.10 ms | 2 |
| **Standard Transformer** | 11.21 ms | 28.27 ms | 56.81 ms | 3 |
| **ALBERT** | 13.85 ms | 34.20 ms | 69.50 ms | 4 (Slowest) |

#### Memory Footprint (Peak RSS on Device)
| Architecture | 10M Tier RSS | 25M Tier RSS | 50M Tier RSS | Memory Rank |
| :--- | :--- | :--- | :--- | :--- |
| **ALBERT** | **62.0 MB** | **115.0 MB** | **210.0 MB** | **1 (Smallest)** |
| **MobiMIND Hybrid** | 68.0 MB | 124.0 MB | 228.0 MB | 2 |
| **MobileBERT** | 72.0 MB | 132.0 MB | 242.0 MB | 3 |
| **Standard Transformer** | 78.5 MB | 145.0 MB | 260.0 MB | 4 (Largest) |

### Key Architectural Invariants & Discoveries
1. **The ALBERT Compute Trap**:
   - While ALBERT achieves lower resident memory (115 MB vs 145 MB at 25M), its latency is **21–23% worse** than the Standard Transformer (34.20 ms vs 28.27 ms). Cross-layer parameter sharing conserves static parameters but executes the exact same FLOP sequence sequentially through a single shared weight buffer, suffering from repeated cache re-loads and higher recursive depth without compute savings.
2. **MobileBERT Linear Bottlenecks**:
   - Compressing attention dimensionality into bottleneck sub-spaces ($d_{bottleneck} = d_{model} / 4$) provides a consistent **5–6% latency improvement** over Standard Transformers without sacrificing semantic fidelity (95.9% vs 95.7% accuracy at 25M).
3. **MobiMIND Hybrid Superiority**:
   - Gated depthwise 1D convolutions in the FFN layer replace heavy dense projection matrices with local sequence mixing. This delivers the highest throughput across all tiers: **18–21% faster** than the Standard Transformer and **34–35% faster** than ALBERT, while maintaining superior accuracy (96.3% at 25M, 99.9% at 50M).
