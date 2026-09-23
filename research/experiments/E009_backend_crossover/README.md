# Experiment E009: Hardware Backend Workload Crossover

### Empirical Sweep on Snapdragon 6 Gen 4 (Physical Device)
Executed directly via native aarch64 NDK binaries (`mobimind_crossover_bench`) on ARMv9-A CPU and Qualcomm Adreno (TM) 810 GPU (OpenCL):

| Workload | Parameters | CPU P50 Latency | GPU P50 Latency | Speedup Winner | CPU vs GPU Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2K** | 2,048 | **0.0016 ms (1.56 us)** | 1.0206 ms | **CPU** | CPU 654.2x faster |
| **10K** | 10,000 | **0.0786 ms (78.6 us)** | 1.1640 ms | **CPU** | CPU 14.8x faster |
| **100K** | 100,000 | **0.4235 ms (423 us)** | 1.7481 ms | **CPU** | CPU 4.1x faster |
| **500K** | 500,000 | **0.8981 ms (898 us)** | 6.3352 ms | **CPU** | CPU 7.1x faster |
| **1M** | 1,000,000 | **1.3927 ms** | 7.6579 ms | **CPU** | CPU 5.5x faster |
| **5M** | 5,000,000 | **5.5637 ms** | 32.6452 ms | **CPU** | CPU 5.9x faster |
| **10M** | 10,000,000 | **11.2142 ms** | 73.5914 ms | **CPU** | CPU 6.6x faster |
| **25M** | 25,000,000 | **28.2694 ms** | 383.4257 ms | **CPU** | CPU 13.6x faster |
| **50M** | 50,000,000 | **56.8082 ms** | 1385.9127 ms | **CPU** | CPU 24.4x faster |

### Architectural Insight
For single-batch interactive System-1 inference (`batch=1`), **ARMv9 CPU outpaces Adreno GPU across the entire 2K to 50M parameter range**. Mobile GPU execution suffers from bus dispatch, buffer serialization, and command synchronization overheads that outweigh computational concurrency for vector-matrix kernels.
