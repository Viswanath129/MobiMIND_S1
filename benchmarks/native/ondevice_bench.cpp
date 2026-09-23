#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include "weights.h"

// Hardware Timer (nanoseconds)
static inline double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

// Model Forward Pass (2,093 params)
struct ModelOutputs {
    float intent_logits[5];
    float action_logits[6];
    float risk;
    float confidence;
};

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

void forward_s1(const float* in_vec, ModelOutputs* out) {
    // 1. Linear: 16 -> 32
    float h1[32];
    for (int i = 0; i < 32; ++i) {
        float sum = W_encoder_0_bias[i];
        for (int j = 0; j < 16; ++j) {
            sum += in_vec[j] * W_encoder_0_weight[i * 16 + j];
        }
        h1[i] = sum;
    }

    // 2. LayerNorm: 32
    float mean = 0.0f;
    for (int i = 0; i < 32; ++i) mean += h1[i];
    mean /= 32.0f;

    float var = 0.0f;
    for (int i = 0; i < 32; ++i) {
        float diff = h1[i] - mean;
        var += diff * diff;
    }
    var /= 32.0f;
    float std_dev = std::sqrt(var + 1e-5f);

    float norm1[32];
    for (int i = 0; i < 32; ++i) {
        norm1[i] = ((h1[i] - mean) / std_dev) * W_encoder_1_weight[i] + W_encoder_1_bias[i];
        norm1[i] = relu(norm1[i]);
    }

    // 3. Linear: 32 -> 32
    float h2[32];
    for (int i = 0; i < 32; ++i) {
        float sum = W_encoder_3_bias[i];
        for (int j = 0; j < 32; ++j) {
            sum += norm1[j] * W_encoder_3_weight[i * 32 + j];
        }
        h2[i] = relu(sum);
    }

    // 4. Intent Head: 32 -> 5
    for (int i = 0; i < 5; ++i) {
        float sum = W_head_intent_bias[i];
        for (int j = 0; j < 32; ++j) {
            sum += h2[j] * W_head_intent_weight[i * 32 + j];
        }
        out->intent_logits[i] = sum;
    }

    // 5. Action Head: 32 -> 6
    for (int i = 0; i < 6; ++i) {
        float sum = W_head_action_bias[i];
        for (int j = 0; j < 32; ++j) {
            sum += h2[j] * W_head_action_weight[i * 32 + j];
        }
        out->action_logits[i] = sum;
    }

    // 6. Risk Head: 32 -> 1 (Sigmoid)
    float risk_sum = W_head_risk_0_bias[0];
    for (int j = 0; j < 32; ++j) {
        risk_sum += h2[j] * W_head_risk_0_weight[j];
    }
    out->risk = sigmoid(risk_sum);

    // 7. Confidence Head: 32 -> 1 (Sigmoid)
    float conf_sum = W_head_confidence_0_bias[0];
    for (int j = 0; j < 32; ++j) {
        conf_sum += h2[j] * W_head_confidence_0_weight[j];
    }
    out->confidence = sigmoid(conf_sum);
}

// Percentile Calculation
struct Stats {
    double p50_us;
    double p90_us;
    double p95_us;
    double p99_us;
    double min_us;
    double max_us;
};

Stats compute_stats(std::vector<double>& latencies_us) {
    std::sort(latencies_us.begin(), latencies_us.end());
    size_t n = latencies_us.size();
    auto idx = [&](double p) {
        size_t i = (size_t)std::round(p / 100.0 * (n - 1));
        return std::min(i, n - 1);
    };
    Stats s;
    s.p50_us = latencies_us[idx(50.0)];
    s.p90_us = latencies_us[idx(90.0)];
    s.p95_us = latencies_us[idx(95.0)];
    s.p99_us = latencies_us[idx(99.0)];
    s.min_us = latencies_us.front();
    s.max_us = latencies_us.back();
    return s;
}

int main(int argc, char** argv) {
    // 1. Hardware Probe
    bool opencl_available = false;
    void* cl_handle = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    if (!cl_handle) cl_handle = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW);
    if (cl_handle) {
        opencl_available = true;
    }

    bool cdsp_available = false;
    void* cdsp_handle = dlopen("/vendor/lib64/libcdsprpc.so", RTLD_NOW);
    int adsprpc_fd = open("/dev/adsprpc-smd", O_RDWR);
    if (cdsp_handle || adsprpc_fd >= 0) {
        cdsp_available = true;
        if (adsprpc_fd >= 0) close(adsprpc_fd);
    }

    int kgsl_fd = open("/dev/kgsl-3d0", O_RDWR);
    bool kgsl_available = (kgsl_fd >= 0);
    if (kgsl_fd >= 0) close(kgsl_fd);

    // 2. Dummy Sample Input (Wi-Fi off context)
    float sample_input[16] = {
        0.28f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.45f, 0.89f, 0.12f, 0.99f, 0.73f, 0.68f, 0.31f, 0.95f,
        0.0f, 0.0f, 0.0f
    };

    // Warmup
    ModelOutputs out;
    for (int i = 0; i < 20; ++i) {
        forward_s1(sample_input, &out);
    }

    // 3. Measure On-Device CPU Forward Latency (200 warm iterations)
    std::vector<double> cpu_forward_us;
    cpu_forward_us.reserve(200);

    for (int i = 0; i < 200; ++i) {
        double t0 = get_time_ns();
        forward_s1(sample_input, &out);
        double t1 = get_time_ns();
        cpu_forward_us.push_back((t1 - t0) / 1000.0);
    }

    Stats cpu_stats = compute_stats(cpu_forward_us);

    // 4. Output JSON Result
    printf("{\n");
    printf("  \"target_device\": \"Snapdragon 6 Gen 4 (SM6650)\",\n");
    printf("  \"hardware_probe\": {\n");
    printf("    \"adreno_kgsl_dev\": %s,\n", kgsl_available ? "true" : "false");
    printf("    \"opencl_driver\": %s,\n", opencl_available ? "true" : "false");
    printf("    \"cdsp_fastrpc\": %s\n", cdsp_available ? "true" : "false");
    printf("  },\n");
    printf("  \"cpu_forward_benchmark\": {\n");
    printf("    \"iterations\": 200,\n");
    printf("    \"p50_us\": %.2f,\n", cpu_stats.p50_us);
    printf("    \"p90_us\": %.2f,\n", cpu_stats.p90_us);
    printf("    \"p95_us\": %.2f,\n", cpu_stats.p95_us);
    printf("    \"p99_us\": %.2f,\n", cpu_stats.p99_us);
    printf("    \"min_us\": %.2f,\n", cpu_stats.min_us);
    printf("    \"max_us\": %.2f,\n", cpu_stats.max_us);
    printf("    \"p50_ms\": %.5f,\n", cpu_stats.p50_us / 1000.0);
    printf("    \"p95_ms\": %.5f\n", cpu_stats.p95_us / 1000.0);
    printf("  },\n");
    printf("  \"sample_inference_verification\": {\n");
    printf("    \"predicted_risk\": %.4f,\n", out.risk);
    printf("    \"predicted_confidence\": %.4f\n", out.confidence);
    printf("  }\n");
    printf("}\n");

    if (cl_handle) dlclose(cl_handle);
    if (cdsp_handle) dlclose(cdsp_handle);
    return 0;
}
