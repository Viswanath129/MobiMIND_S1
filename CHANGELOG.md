# Changelog

All notable changes to the MobiMIND-S1 project are documented here.

## [v0.25] - 2026-09-23
### Added
- **E005**: Fast-path hierarchy ablation (Level 0 deterministic rules bypass 32% of events with 0 ms model compute; Level 1 heuristics resolve 24%; composite cascade yields 0.045 ms decision latency).
- **E006**: Context Delta Engine (reduces context latency from 42.15 ms to 0.18 ms; 234.1x faster with 99.1% memory reduction).
- **E007**: 10-Action verification matrix (Direct API 2.3-2.9ms, Event-driven 10-29ms, UI Node 50ms).
- **E008**: True physical-device E2E latency trace (`T_total = 2.81 ms`).
- **E009**: Complete physical hardware crossover curve sweeping 2K to 50M parameters on Snapdragon 6 Gen 4 (ARMv9 CPU vs Adreno 810 GPU).
### Verified Hardware Findings
- CPU outpaces GPU across the entire 2K-50M single-batch sweep (CPU: 56.8 ms for 50M vs GPU: 1,385.9 ms).
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
