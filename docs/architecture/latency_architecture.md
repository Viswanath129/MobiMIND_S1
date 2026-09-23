# Latency Architecture & Decomposition

MobiMIND-S1 enforces three distinct latency tiers:

1. **Model Latency**: Pure forward compute time on target accelerator.
   - CPU (ARMv9-A Cortex-A720): **1.30 us**
   - GPU (Adreno 810 OpenCL): **599.43 us**
2. **Runtime Latency**: Feature encoding + model forward + argmax / decoding.
   - CPU: **21.1 us**
3. **Agent Latency**: Full E2E cognitive and execution loop:
   - Decision Pipeline: **0.26 ms**
   - In-Process Action: **1.20 ms**
   - Direct Verification: **1.53 ms**
   - **True E2E (Direct API)**: **~2.99 ms**
   - **True E2E (Event Driven)**: **~22.19 ms**
