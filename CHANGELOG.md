# Changelog

All notable changes to the MobiMIND-S1 project are documented here.

## [v0.2-optimized] - 2026-09-23
### Added
- Verification Strategy Engine (`DIRECT_API` -> `EVENT` -> `UI_NODE` -> `POLL` -> `TIMEOUT_FALLBACK`).
- In-process Android ContentResolver / Binder action path ablation.
- Context Delta Engine with Level-0 deterministic bypass for predictable events.
- Dissected legacy 150 ms polling timeout and 159 ms ADB host spawn overhead.
### Verified Results
- `DIRECT_API` E2E latency: ~2.99 ms (103.7x speedup vs 310 ms legacy baseline).
- `EVENT` E2E latency: ~22.19 ms (14.0x speedup vs legacy baseline).

## [v0.2] - 2026-09-23
### Added
- Physical hardware execution on Snapdragon 6 Gen 4 (Realme RMX5070, Android 16 / API 36).
- Native aarch64 benchmark executables compiled via NDK 28 Clang.
- Qualcomm Adreno 810 OpenCL GPU compute benchmark (`libOpenCL.so`).
- Qualcomm Hexagon cDSP FastRPC access audit (`/dev/adsprpc-smd`).
### Findings
- CPU (ARMv9 Cortex-A720/A520) executes 2K baseline forward in 1.30 us.
- GPU (Adreno 810) executes in 599.43 us (CPU is 461x faster due to bus transfer/kernel dispatch overhead).
- FastRPC direct user-space access is restricted by Android SELinux (`crw-rw-r-- system:system`).

## [v0.1] - 2026-09-23
### Added
- Initial 2,093 parameter baseline control model (`MobiMind-S1-v0`).
- Immutable device baseline hardware record for Snapdragon 6 Gen 4.
- 20-dimensional measurement schema with P50/P90/P95/P99 latency decomposition.
- Frozen baseline run record: Model forward P50: 12.0 us; Runtime P50: 21.1 us.
