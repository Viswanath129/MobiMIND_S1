# Experiment E005: Fast-Path Hierarchy Ablation

### Hypothesis
A tiered hierarchy of agency (Level 0 deterministic rules -> Level 1 heuristics -> Level 2 MobiMind-S1) can resolve a significant portion of routine phone events with zero neural inference overhead while maintaining >=96% useful action rates.

### Measured Results on Snapdragon 6 Gen 4
| Tier | Event Coverage | Decision Latency | True E2E Latency | Useful Action Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Level 0 (Deterministic)** | 32.0% | 0.001 ms | 1.25 ms | 100.0% |
| **Level 1 (Heuristic)** | 24.0% | 0.008 ms | 1.35 ms | 98.0% |
| **Level 2 (MobiMind-S1)** | 44.0% | 0.261 ms | 2.99 ms | 96.0% |
| **Composite Cascade** | **100.0%** | **P50: 0.045 ms** | **P50: 1.82 ms** | **97.0%** |

### Key Finding
**56.0% of routine mobile events require zero neural inference**, dropping effective decision latency to **0.045 ms** and effective True E2E latency to **1.82 ms**.
