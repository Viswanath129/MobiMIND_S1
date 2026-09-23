# Audit Report: Formal Truth-First Audit of Milestone v0.3-R1

**Audit Target**: Commit `4ae52f962532c808d5603d10244099c0b7a26ae8`  
**Audit Date**: September 23, 2026  
**Auditor**: Truth-First Research Protocol  
**Status**: **PROXY KERNELS QUARANTINED FOR R2 UPGRADE**  

---

## 1. Context & Identified Shortcomings in v0.3-R1

While Milestone v0.3-R1 succeeded in replacing synthetic metric calculations with genuine empirical predictions and deterministic tokenization, a secondary audit revealed that the on-device latency and RSS metrics did not measure the actual trained models:
1. **Proxy Kernel Execution**: `benchmarks/native/physical_model_bench.cpp` instantiated arbitrary float arrays (`0.005f`) and performed generic GEMM/conv loops. The actual 40MB–200MB trained ONNX models were never transferred to the device or loaded into the benchmark process.
2. **Proxy Process Memory**: Because parameter weights were not loaded into RAM, the process RSS values (~12MB–175MB) reflected arbitrary scratch buffers rather than the true resident footprint of a 10M–50M parameter neural network.
3. **Training Metadata Inconsistency**: The training routine executed 2 epochs for certain retries but logged `epochs: 4`, and assigned `val_loss = 0.0020` without re-evaluating the validation dataset on loaded checkpoints.
4. **Partial Output Parity**: ONNX export parity was only checked for the first output head (`intent_logits`), ignoring `action_logits`, `risk`, and `confidence`.

---

## 2. Corrective Action Plan for Milestone v0.3-R2

Milestone v0.3-R2 executes a complete, provable model-on-device validation:
1. **Full Output Parity Check**: Compare all 4 output heads (`intent_logits`, `action_logits`, `risk`, `confidence`) between PyTorch and ONNX Runtime with strict tolerances ($< 10^{-4}$).
2. **ONNX IR Compatibility**: Ensure all exported graphs use IR version 9 to run natively against the physical device's `/system/lib64/libonnxruntime.so` (v1.15.0).
3. **Actual Model Push & Load**: Transfer every single `.onnx` and external weight `.data` file to `/data/local/tmp/` on the Realme RMX5070.
4. **True Physical Process Measurement**:
   - Baseline RSS before model creation.
   - Load time via `CreateSession()`.
   - RSS after model load (verifying that the process memory reflects the actual 40MB–200MB parameter weight allocation).
   - Cold start forward latency.
   - Steady-state P50, P95, P99 forward latency over 15 iterations.
5. **Exact Training Metadata**: Recompute validation loss on `validation_set.json` for all loaded checkpoints and record true executed epoch counts.
6. **Separation of Proxy vs. Actual Metrics**: The historical proxy kernel numbers will remain preserved under `PROXY_KERNEL_BENCHMARK`, clearly separated from actual on-device model measurements.

---

## 3. Resolution & Verification in Milestone v0.3-R2

Milestone v0.3-R2 has completely resolved all findings from this audit:
- **AUDIT-R1-001 (Resolved)**: Native C++ runner `real_onnx_device_runner` executed all 12 candidate models using `/system/lib64/libonnxruntime.so` on Snapdragon 6 Gen 4 CPU. Weights were resident in RAM, verified via `ru_maxrss` delta.
- **AUDIT-R1-002 (Resolved)**: `compute_empirical_val_loss` evaluated exact validation loss on `validation_set.json` (values ranging from 0.0012 to 4.0411), with true epochs (2) recorded in all artifacts.
- **AUDIT-R1-003 (Resolved)**: All 4 heads (`intent_logits`, `action_logits`, `risk`, `confidence`) verified with numerical parity $< 3.34 \times 10^{-6}$.
- **AUDIT-R1-004 (Resolved)**: Scientific language properly qualified in `docs/research/v0.3_r2_actual_model_report.md` as empirical hypotheses motivating distillation.

