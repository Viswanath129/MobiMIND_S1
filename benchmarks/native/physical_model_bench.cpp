#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <sys/resource.h>
#include <unistd.h>
#include <arm_neon.h>
#include <omp.h>

static inline double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

long get_peak_rss_kb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;
    }
    return 0;
}

struct TimingStats {
    double p50_ms;
    double p95_ms;
    double p99_ms;
    double min_ms;
    double max_ms;
    double mean_ms;
};

TimingStats compute_stats(std::vector<double>& times_ms) {
    std::sort(times_ms.begin(), times_ms.end());
    size_t n = times_ms.size();
    double sum = 0.0;
    for (double t : times_ms) sum += t;
    TimingStats s;
    s.p50_ms = times_ms[n * 50 / 100];
    s.p95_ms = times_ms[n * 95 / 100];
    s.p99_ms = times_ms[n * 99 / 100];
    s.min_ms = times_ms[0];
    s.max_ms = times_ms[n - 1];
    s.mean_ms = sum / n;
    return s;
}

// Optimized GEMM with NEON and OpenMP: C = A * B (A: M x K, B: K x N, C: M x N)
void matmul_neon(const float* A, const float* B, float* C, int M, int K, int N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 4) {
            float32x4_t c_vec = vdupq_n_f32(0.0f);
            for (int k = 0; k < K; ++k) {
                float32x4_t a_val = vdupq_n_f32(A[m * K + k]);
                float32x4_t b_vec = vld1q_f32(&B[k * N + n]);
                c_vec = vmlaq_f32(c_vec, a_val, b_vec);
            }
            vst1q_f32(&C[m * N + n], c_vec);
        }
    }
}

// Depthwise 1D Conv (sequence: S x D, kernel: 3 x D)
void depthwise_conv1d_neon(const float* in, const float* weight, float* out, int S, int D) {
    #pragma omp parallel for schedule(static)
    for (int d = 0; d < D; d += 4) {
        for (int s = 0; s < S; ++s) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int k = 0; k < 3; ++k) {
                int in_idx = s + k - 1;
                if (in_idx >= 0 && in_idx < S) {
                    float32x4_t in_vec = vld1q_f32(&in[in_idx * D + d]);
                    float32x4_t w_vec = vld1q_f32(&weight[k * D + d]);
                    acc = vmlaq_f32(acc, in_vec, w_vec);
                }
            }
            vst1q_f32(&out[s * D + d], acc);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s <arch> <tier> <d_model> <layers_or_passes> <dim_ff_or_bn>\n", argv[0]);
        return 1;
    }

    const char* arch = argv[1];
    const char* tier = argv[2];
    int d_model = atoi(argv[3]);
    int layers = atoi(argv[4]);
    int aux_dim = atoi(argv[5]);

    const int S = 16;
    const int WARMUP_ITERS = 5;
    const int BENCH_ITERS = 30;

    // Use 4 Performance Cores (Cortex-A720)
    omp_set_num_threads(4);

    std::vector<float> input(S * d_model, 0.01f);
    std::vector<float> hidden(S * d_model, 0.0f);
    std::vector<float> qkv_weight(d_model * (3 * d_model), 0.005f);
    std::vector<float> qkv_out(S * (3 * d_model), 0.0f);
    std::vector<float> attn_proj_weight(d_model * d_model, 0.005f);

    // Warmup & Cold start timing
    double t_cold_start_ms = 0.0;
    {
        double t0 = get_time_ns();
        matmul_neon(input.data(), qkv_weight.data(), qkv_out.data(), S, d_model, 3 * d_model);
        matmul_neon(qkv_out.data(), attn_proj_weight.data(), hidden.data(), S, d_model, d_model);
        double t1 = get_time_ns();
        t_cold_start_ms = (t1 - t0) / 1e6;
    }

    std::vector<double> forward_times_ms;
    forward_times_ms.reserve(BENCH_ITERS);

    if (strcmp(arch, "standard_transformer") == 0) {
        int dim_ff = aux_dim;
        std::vector<float> ff1_weight(d_model * dim_ff, 0.005f);
        std::vector<float> ff1_out(S * dim_ff, 0.0f);
        std::vector<float> ff2_weight(dim_ff * d_model, 0.005f);

        for (int iter = 0; iter < BENCH_ITERS; ++iter) {
            double t0 = get_time_ns();
            for (int l = 0; l < layers; ++l) {
                matmul_neon(input.data(), qkv_weight.data(), qkv_out.data(), S, d_model, 3 * d_model);
                matmul_neon(qkv_out.data(), attn_proj_weight.data(), hidden.data(), S, d_model, d_model);
                matmul_neon(hidden.data(), ff1_weight.data(), ff1_out.data(), S, d_model, dim_ff);
                matmul_neon(ff1_out.data(), ff2_weight.data(), hidden.data(), S, dim_ff, d_model);
            }
            double t1 = get_time_ns();
            forward_times_ms.push_back((t1 - t0) / 1e6);
        }
    } else if (strcmp(arch, "albert") == 0) {
        int dim_ff = aux_dim;
        std::vector<float> ff1_weight(d_model * dim_ff, 0.005f);
        std::vector<float> ff1_out(S * dim_ff, 0.0f);
        std::vector<float> ff2_weight(dim_ff * d_model, 0.005f);

        for (int iter = 0; iter < BENCH_ITERS; ++iter) {
            double t0 = get_time_ns();
            for (int p = 0; p < layers; ++p) {
                matmul_neon(input.data(), qkv_weight.data(), qkv_out.data(), S, d_model, 3 * d_model);
                matmul_neon(qkv_out.data(), attn_proj_weight.data(), hidden.data(), S, d_model, d_model);
                matmul_neon(hidden.data(), ff1_weight.data(), ff1_out.data(), S, d_model, dim_ff);
                matmul_neon(ff1_out.data(), ff2_weight.data(), hidden.data(), S, dim_ff, d_model);
            }
            double t1 = get_time_ns();
            forward_times_ms.push_back((t1 - t0) / 1e6);
        }
    } else if (strcmp(arch, "mobilebert") == 0) {
        int bottleneck_dim = aux_dim;
        std::vector<float> in_proj(d_model * bottleneck_dim, 0.005f);
        std::vector<float> in_proj_out(S * bottleneck_dim, 0.0f);
        std::vector<float> bn_qkv_weight(bottleneck_dim * (3 * bottleneck_dim), 0.005f);
        std::vector<float> bn_qkv_out(S * (3 * bottleneck_dim), 0.0f);
        std::vector<float> out_proj(bottleneck_dim * d_model, 0.005f);
        std::vector<float> ff1_weight(d_model * (2 * d_model), 0.005f);
        std::vector<float> ff1_out(S * (2 * d_model), 0.0f);
        std::vector<float> ff2_weight((2 * d_model) * d_model, 0.005f);

        for (int iter = 0; iter < BENCH_ITERS; ++iter) {
            double t0 = get_time_ns();
            for (int l = 0; l < layers; ++l) {
                matmul_neon(input.data(), in_proj.data(), in_proj_out.data(), S, d_model, bottleneck_dim);
                matmul_neon(in_proj_out.data(), bn_qkv_weight.data(), bn_qkv_out.data(), S, bottleneck_dim, 3 * bottleneck_dim);
                matmul_neon(bn_qkv_out.data(), out_proj.data(), hidden.data(), S, bottleneck_dim, d_model);
                matmul_neon(hidden.data(), ff1_weight.data(), ff1_out.data(), S, d_model, 2 * d_model);
                matmul_neon(ff1_out.data(), ff2_weight.data(), hidden.data(), S, 2 * d_model, d_model);
            }
            double t1 = get_time_ns();
            forward_times_ms.push_back((t1 - t0) / 1e6);
        }
    } else if (strcmp(arch, "mobimind_hybrid") == 0) {
        std::vector<float> conv_w(3 * d_model, 0.005f);
        std::vector<float> conv_out(S * d_model, 0.0f);
        std::vector<float> gate_weight(d_model * (2 * d_model), 0.005f);
        std::vector<float> gate_out(S * (2 * d_model), 0.0f);
        std::vector<float> proj_weight(d_model * d_model, 0.005f);

        for (int iter = 0; iter < BENCH_ITERS; ++iter) {
            double t0 = get_time_ns();
            for (int l = 0; l < layers; ++l) {
                matmul_neon(input.data(), qkv_weight.data(), qkv_out.data(), S, d_model, 3 * d_model);
                matmul_neon(qkv_out.data(), attn_proj_weight.data(), hidden.data(), S, d_model, d_model);
                depthwise_conv1d_neon(hidden.data(), conv_w.data(), conv_out.data(), S, d_model);
                matmul_neon(conv_out.data(), gate_weight.data(), gate_out.data(), S, d_model, 2 * d_model);
                matmul_neon(conv_out.data(), proj_weight.data(), hidden.data(), S, d_model, d_model);
            }
            double t1 = get_time_ns();
            forward_times_ms.push_back((t1 - t0) / 1e6);
        }
    }

    TimingStats stats = compute_stats(forward_times_ms);
    long peak_rss_kb = get_peak_rss_kb();

    printf("{\n");
    printf("  \"arch\": \"%s\",\n", arch);
    printf("  \"tier\": \"%s\",\n", tier);
    printf("  \"iterations\": %d,\n", BENCH_ITERS);
    printf("  \"model_p50_ms\": %.3f,\n", stats.p50_ms);
    printf("  \"model_p95_ms\": %.3f,\n", stats.p95_ms);
    printf("  \"model_p99_ms\": %.3f,\n", stats.p99_ms);
    printf("  \"model_min_ms\": %.3f,\n", stats.min_ms);
    printf("  \"model_max_ms\": %.3f,\n", stats.max_ms);
    printf("  \"model_mean_ms\": %.3f,\n", stats.mean_ms);
    printf("  \"cold_start_ms\": %.3f,\n", t_cold_start_ms);
    printf("  \"peak_rss_kb\": %ld,\n", peak_rss_kb);
    printf("  \"peak_rss_mb\": %.2f\n", peak_rss_kb / 1024.0);
    printf("}\n");

    return 0;
}
