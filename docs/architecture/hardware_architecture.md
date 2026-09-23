# Hardware Architecture & Acceleration

Target Device: **Realme RMX5070 / Snapdragon 6 Gen 4 (SM6650)**
- **CPU**: 8-Core ARMv9-A with `i8mm`, `bf16`, `asimddp` SIMD acceleration.
- **GPU**: Qualcomm Adreno (TM) 810 with OpenCL 3.0 and Vulkan 1.3 compute.
- **Hexagon cDSP**: FastRPC subsystem `/dev/adsprpc-smd`.
  - *Finding*: Direct user-space access is restricted by Android SELinux. Production path requires vendor HAL service (`vendor.qti.hardware.dsp@1.0.so`).
- **Accelerator Hierarchy**:
  ```kotlin
  enum class Accelerator { CPU, GPU, HEXAGON }
  ```
