# Experiment E014: Actual Model-on-Device Validation (v0.3-R2)

### Background & Objective
In Milestone v0.3-R1, synthetic metric generators were removed and actual predictions evaluated. However, an architectural audit revealed that the on-device timing binary (`physical_model_bench.cpp`) executed hand-written NEON GEMM/conv proxy kernels with synthetic static float arrays rather than executing the actual compiled ONNX candidate models.

**Experiment E014 executes Milestone v0.3-R2**:
1. Every trained PyTorch candidate model is exported to an ONNX graph with verified numerical parity across **ALL 4 outputs** (`intent_logits`, `action_logits`, `risk`, `confidence`).
2. The actual `.onnx` and external weight `.data` files are transferred to the physical Realme RMX5070 (`Snapdragon 6 Gen 4` / `SM6650`).
3. An on-device C++ executable (`real_onnx_device_runner`) linked against the phone's native `/system/lib64/libonnxruntime.so` (version 1.15.0) loads each model into process memory, verifies memory delta/RSS, and measures cold load, cold start, and steady-state P50/P95/P99 latency across 15 iterations.
4. Historical proxy kernel results are preserved and clearly labeled as `PROXY_KERNEL_BENCHMARK`.

### Hardware & Runtime Environment
- **Device**: Realme RMX5070 (`SM6650`, 4× Cortex-A720 + 4× Cortex-A520, Android 16 / API 36)
- **Runtime**: `/system/lib64/libonnxruntime.so` (v1.15.0)
- **Execution Provider**: `CPUExecutionProvider` (4 CPU threads)
- **ONNX IR Compatibility**: IR version 9, Opset 17/18
