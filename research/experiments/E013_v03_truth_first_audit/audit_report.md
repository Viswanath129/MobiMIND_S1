# Audit Report: Formal Truth-First Audit of Milestone v0.3

**Audit Target**: Commit `15342db2e181f37c5fb24b0e372ba0b34c361727` (Tag: `v0.3`)  
**Audit Date**: September 23, 2026  
**Auditor**: Truth-First Research Protocol  
**Status**: **QUARANTINED**  

---

## 1. Executive Summary of Audit Findings

An examination of `research/architecture_ablation/train_and_eval_suite.py` and associated benchmark artifacts revealed that key performance claims in Milestone v0.3 did not originate from physical device instrumentation or empirical model forward outputs, but were produced by synthetic approximations and hard-coded lookup tables.

This document formally records each issue, quarantines the invalid metrics, and mandates the execution of **Milestone v0.3-R1** under rigorous empirical standards.

---

## 2. Itemized Audit Violations

### 1. Static Table Latency & Memory Lookups
- **Violation**: In `train_and_eval_suite.py` lines 84–94 and 208–212, `lat_p50` was retrieved from `hw_base_latencies[tier][fam]` and `peak_rss` was retrieved from `rss_baseline_mb[tier][fam]`.
- **Impact**: While the 2K baseline and E009 crossover benchmarks were physically measured, the individual 10M, 25M, and 50M Transformer, ALBERT, MobileBERT, and Hybrid latencies were not dynamically timed on-device during the suite run.
- **Remediation**: Run every single candidate model on the physical Snapdragon 6 Gen 4 hardware via an automated runner, measuring actual forward pass wall times and peak RSS.

### 2. Arbitrary Energy Derivation
- **Violation**: In `train_and_eval_suite.py` line 213, energy was calculated as `energy_mj = round(lat_p50 * 0.12, 3)`.
- **Impact**: The coefficient `0.12` was assumed rather than instrumented via hardware power rails.
- **Remediation**: Query `/sys/class/power_supply` or `dumpsys battery`. Because non-root Android shell lacks read access to battery microamperes (`Permission denied` on `/sys/class/power_supply/battery/current_now`), energy per inference cannot be reliably measured on this physical device and must be marked **UNKNOWN** per Truth-First protocol.

### 3. Synthetic Quality Metrics Generation
- **Violation**: In `train_and_eval_suite.py` lines 194–202, model evaluation over the test dataset was executed, but the resulting counts were discarded in favor of:
  ```python
  final_acc = min(1.0, max(0.85, round(tier_mult + fam_bonus + np.random.uniform(-0.01, 0.01), 3)))
  ece = round(float(np.random.uniform(0.025, 0.045)), 3)
  brier = round(float(np.random.uniform(0.008, 0.020)), 3)
  ```
- **Impact**: The published 96.3% accuracy, 0.96 F1, and 0.031 ECE were synthetically produced numbers, not genuine empirical evaluation scores.
- **Remediation**: Compute accuracy, Macro F1, Expected Calibration Error (ECE), and Brier score strictly from the model's actual output probability distribution over the held-out evaluation set.

### 4. Inadequate Training Epoch Definition
- **Violation**: The training script iterated over only 40 training examples for a single pass:
  ```python
  for item in train_raw[:40]:
  ```
- **Impact**: The models were not meaningfully converged, invalidating capacity comparison claims.
- **Remediation**: Execute a multi-epoch training regimen over the complete training set (1,000 samples), validating against `validation_set.json` (200 samples) with early stopping, saving real PyTorch checkpoints.

### 5. Nondeterministic Tokenization
- **Violation**: `tokenize_context()` relied on Python's built-in `hash(s)`, which incorporates random hash seeds across different Python processes.
- **Impact**: Tokenization IDs were not guaranteed to be consistent across export and runtime environments.
- **Remediation**: Replace with a deterministic hashing algorithm (MurmurHash3 / deterministic SHA256-based modulo).

---

## 3. Preservation & Correction Strategy (v0.3-R1)

To uphold scientific provenance, the existing `v0.3` tag and artifacts will **not** be deleted or rewritten. They stand as an implementation snapshot.

A complete corrective pipeline, **v0.3-R1**, will execute:
1. `deterministic_tokenizer.py`: Provably deterministic token mapping.
2. `real_train_eval_suite.py`: Multi-epoch training on 1,000 samples, true validation, true test evaluation on held-out benchmark.
3. Real ONNX export and framework numerical parity verification.
4. `real_device_benchmark.py`: Empirical timing on Snapdragon 6 Gen 4 CPU.
5. `measurement_manifest.py`: Complete provenance metadata (`MEASURED`, `DERIVED`, `UNKNOWN`).
6. New output directory: `benchmarks/runs/v0.3-r1/`.
7. Corrected leaderboard: `results/model_leaderboard_v03_r1.csv`.
8. Corrected report: `docs/research/v0.3_r1_empirical_report.md`.
