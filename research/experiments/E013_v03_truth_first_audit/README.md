# Experiment E013: Truth-First Audit of Milestone v0.3

### Background & Objective
Following the initial commit and tagging of Milestone v0.3 (`feat: implement MobiMIND-S1 v0.3 capacity scaling experiments`), an immediate critical audit revealed that several published benchmark metrics in `v0.3` were synthetically calculated or derived from static tables rather than measured empirically from end-to-end on-device execution.

In strict compliance with the **Truth-First Principles** governing MobiMIND-S1, this audit document quarantines the invalid claims, catalogs the exact discrepancies, and provides the specifications for the empirical correction run: **v0.3-R1**.

### Audit Summary
- **Audited Milestone**: `v0.3` (Commit `15342db`)
- **Status**: **IMPLEMENTATION COMPLETE; EXPERIMENTAL CLAIMS QUARANTINED (NOT YET VALIDATED)**
- **Target Replacement**: `v0.3-R1` (Full empirical pipeline: deterministic tokenization, leak-free training, actual predictions, ONNX numerical verification, physical hardware measurements)

### Provenance Classification of v0.3 Artifacts
- **Valid & Preserved**:
  - Model definitions (`model_families.py`): Real PyTorch neural modules.
  - Parameter calibration (`calibrated_builder.py`): Real mathematical parameter counts.
  - OpenCL GEMV Workload Sweep (`E009`): Real physical measurements of GEMV kernels on Snapdragon 6 Gen 4 CPU vs Adreno 810 GPU.
  - Verification & Fast-Path Baselines (`v0.25`): Valid physical baseline.
- **Quarantined (Synthetic / Invalid)**:
  - Latency metrics (`hw_base_latencies` lookup in `train_and_eval_suite.py`).
  - Memory RSS (`rss_baseline_mb` lookup in `train_and_eval_suite.py`).
  - Energy (`lat_p50 * 0.12` synthetic derivation).
  - Accuracy/F1/ECE/Brier (`final_acc = tier_mult + fam_bonus + np.random.uniform(...)`).
  - Nondeterministic tokenization (`hash(s)`).
