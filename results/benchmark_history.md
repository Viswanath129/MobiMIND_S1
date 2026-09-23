# Benchmark History & Experimental Records

This document preserves the immutable history of physical hardware runs executed on the Snapdragon 6 Gen 4 platform.

### Run Summary
1. `benchmarks/runs/v0.1/frozen_baseline.json`: First control pipeline validation using 2K parameter model.
2. `benchmarks/runs/v0.2/optimized_verification.json`: Real physical device latency breakdown comparing Direct API, Event-driven, UI Node, and Legacy Polling strategies.
3. `benchmarks/runs/v0.3/*_comparison.json`: Initial v0.3 comparison runs (QUARANTINED: Contains synthetic metrics and table lookups).
4. `benchmarks/runs/v0.3-r1/10m_empirical_comparison.json`: v0.3-R1 10M tier empirical hardware measurements on Snapdragon 6 Gen 4 CPU (Proxy Kernel).
5. `benchmarks/runs/v0.3-r1/25m_empirical_comparison.json`: v0.3-R1 25M tier empirical hardware measurements on Snapdragon 6 Gen 4 CPU (Proxy Kernel).
6. `benchmarks/runs/v0.3-r1/50m_empirical_comparison.json`: v0.3-R1 50M tier empirical hardware measurements on Snapdragon 6 Gen 4 CPU (Proxy Kernel).
7. `benchmarks/runs/v0.3-r2/device_runtime_manifest.json`: Native `/system/lib64/libonnxruntime.so` v1.15.0 hardware manifest on Snapdragon 6 Gen 4 (Realme RMX5070).
8. `benchmarks/runs/v0.3-r2/10m_actual_comparison.json`: v0.3-R2 10M tier actual ONNX model inference and peak RSS on device.
9. `benchmarks/runs/v0.3-r2/25m_actual_comparison.json`: v0.3-R2 25M tier actual ONNX model inference and peak RSS on device.
10. `benchmarks/runs/v0.3-r2/50m_actual_comparison.json`: v0.3-R2 50M tier actual ONNX model inference and peak RSS on device.

