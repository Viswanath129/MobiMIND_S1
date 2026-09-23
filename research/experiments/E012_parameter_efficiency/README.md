# Experiment E012: Parameter & Computational Efficiency (Pareto Frontier)

### Objective
Map the empirical Pareto frontier across Accuracy vs. Latency vs. Memory vs. Energy for on-device mobile System-1 deployment on Snapdragon 6 Gen 4.

### Pareto Frontier Matrix

```text
Accuracy (%)
   100 |                                                 ● Hybrid-50M (44.9ms)
       |                                          ▲ MobBERT-50M (53.1ms)
    95 |                       ● Hybrid-25M (22.4ms)
       |                   ▲ MobBERT-25M (26.8ms)
       |                   ■ StdTrans-25M (28.3ms)
    90 |       ● Hybrid-10M (9.1ms)
       |   ▲ MobBERT-10M (10.5ms)
       |   ■ StdTrans-10M (11.2ms)
    85 |   ◆ ALBERT-10M (13.9ms)
       +───────────────────────────────────────────────────────────── Latency (ms)
       0      10        20        30        40        50        60
```

### Quantitative Metrics Table

| Model Identifier | Parameters | Accuracy (%) | Macro F1 | ECE | Latency P50 (ms) | Peak RSS (MB) | Energy (mJ) | Pareto Optimal? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MobiMind-S1-v0 (Control)** | 2,093 | 100.0%* | 1.00 | 0.030 | **0.0013** | **34.8** | **0.02** | **Yes (Control)** |
| **mobimind_hybrid_10m** | 9,731,600 | 92.4% | 0.92 | 0.025 | **9.15** | 68.0 | **1.10** | **Yes** |
| mobilebert_10m | 11,055,184 | 90.1% | 0.90 | 0.042 | 10.45 | 72.0 | 1.25 | No (Dominated) |
| standard_transformer_10m | 10,511,888 | 89.7% | 0.90 | 0.044 | 11.21 | 78.5 | 1.35 | No (Dominated) |
| albert_10m | 10,786,064 | 88.2% | 0.88 | 0.028 | 13.85 | **62.0** | 1.66 | Yes (Memory Pareto) |
| **mobimind_hybrid_25m** | 24,536,016 | 96.3% | 0.96 | 0.031 | **22.40** | 124.0 | **2.69** | **Yes (Best Tradeoff)** |
| mobilebert_25m | 24,864,856 | 95.9% | 0.96 | 0.031 | 26.80 | 132.0 | 3.22 | No (Dominated) |
| standard_transformer_25m | 25,213,456 | 95.7% | 0.96 | 0.029 | 28.27 | 145.0 | 3.39 | No (Dominated) |
| albert_25m | 24,849,936 | 92.4% | 0.92 | 0.031 | 34.20 | **115.0** | 4.10 | Yes (Memory Pareto) |
| **mobimind_hybrid_50m** | 55,992,976 | 99.9% | 1.00 | 0.044 | **44.90** | 228.0 | **5.39** | **Yes (Max Accuracy)** |
| mobilebert_50m | 49,742,816 | 98.2% | 0.98 | 0.028 | 53.10 | 242.0 | 6.37 | No (Dominated) |
| standard_transformer_50m | 50,399,248 | 96.9% | 0.97 | 0.041 | 56.81 | 260.0 | 6.82 | No (Dominated) |
| albert_50m | 46,886,928 | 95.0% | 0.95 | 0.037 | 69.50 | **210.0** | 8.34 | Yes (Memory Pareto) |

### Pareto Frontier Definition for v0.3
The empirical Pareto frontier consists of:
1. **Control Point (`MobiMind-S1-v0`)**: Micro-kernel control for deterministic fast paths.
2. **`mobimind_hybrid_10m`**: Ultra-low latency edge model (9.15 ms P50, 1.10 mJ, 92.4% accuracy).
3. **`mobimind_hybrid_25m`**: **The Recommended Production Configuration**. Balances near-saturation accuracy (96.3%) with sub-30ms latency (22.40 ms P50) and 124 MB RAM.
4. **`mobimind_hybrid_50m`**: Maximum semantic fidelity configuration (99.9% accuracy, 44.90 ms P50).
5. **`albert_25m`**: Alternative memory-constrained Pareto point when resident RAM is capped strictly at <120 MB.
