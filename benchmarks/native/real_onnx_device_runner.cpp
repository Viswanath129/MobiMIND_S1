#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <sys/resource.h>
#include <unistd.h>
#include "onnxruntime_c_api.h"

static inline double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

long get_current_rss_kb() {
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

#define CHECK_ORT_STATUS(api, status) \
    if (status != NULL) { \
        const char* msg = api->GetErrorMessage(status); \
        fprintf(stderr, "ORT Error: %s\n", msg); \
        api->ReleaseStatus(status); \
        return 1; \
    }

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path.onnx> [iterations=20]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int num_iters = (argc >= 3) ? atoi(argv[2]) : 20;
    if (num_iters < 5) num_iters = 5;

    long rss_before_load_kb = get_current_rss_kb();

    // 1. Initialize ONNX Runtime C API
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to get OrtApi for version %d\n", ORT_API_VERSION);
        return 1;
    }

    OrtEnv* env = NULL;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mobimind_device_runner", &env);
    CHECK_ORT_STATUS(g_ort, status);

    OrtSessionOptions* session_options = NULL;
    status = g_ort->CreateSessionOptions(&session_options);
    CHECK_ORT_STATUS(g_ort, status);

    // Configure 4 CPU threads (Snapdragon 6 Gen 4 Cortex-A720 cluster)
    g_ort->SetIntraOpNumThreads(session_options, 4);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);

    // 2. Measure Model Load Time
    double t_load_start = get_time_ns();
    OrtSession* session = NULL;
    status = g_ort->CreateSession(env, model_path, session_options, &session);
    CHECK_ORT_STATUS(g_ort, status);
    double t_load_end = get_time_ns();
    double load_time_ms = (t_load_end - t_load_start) / 1e6;

    long rss_after_load_kb = get_current_rss_kb();

    // 3. Prepare Input Tensor [1, 16] INT64
    const int64_t input_shape[2] = {1, 16};
    int64_t input_data[16] = {1, 713, 1297, 128, 1297, 1489, 2041, 4233, 1297, 0, 0, 0, 0, 0, 0, 0};

    OrtMemoryInfo* memory_info = NULL;
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    CHECK_ORT_STATUS(g_ort, status);

    OrtValue* input_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        input_data, 16 * sizeof(int64_t),
        input_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_tensor
    );
    CHECK_ORT_STATUS(g_ort, status);

    const char* input_names[] = {"input_ids"};
    const char* output_names[] = {"intent_logits", "action_logits", "risk", "confidence"};
    const size_t num_outputs = 4;
    OrtValue* output_tensors[4] = {NULL, NULL, NULL, NULL};

    // 4. Measure Warmup / Cold Start Run
    double t_cold_start = get_time_ns();
    status = g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, num_outputs, output_tensors);
    CHECK_ORT_STATUS(g_ort, status);
    double t_cold_end = get_time_ns();
    double cold_start_ms = (t_cold_end - t_cold_start) / 1e6;

    for (size_t i = 0; i < num_outputs; ++i) {
        if (output_tensors[i]) {
            g_ort->ReleaseValue(output_tensors[i]);
            output_tensors[i] = NULL;
        }
    }

    // 5. Measure Benchmark Iterations
    std::vector<double> inference_times_ms;
    inference_times_ms.reserve(num_iters);

    for (int iter = 0; iter < num_iters; ++iter) {
        double t0 = get_time_ns();
        status = g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, num_outputs, output_tensors);
        double t1 = get_time_ns();
        CHECK_ORT_STATUS(g_ort, status);
        inference_times_ms.push_back((t1 - t0) / 1e6);

        for (size_t i = 0; i < num_outputs; ++i) {
            if (output_tensors[i]) {
                g_ort->ReleaseValue(output_tensors[i]);
                output_tensors[i] = NULL;
            }
        }
    }

    TimingStats stats = compute_stats(inference_times_ms);
    long peak_rss_kb = get_current_rss_kb();

    // 6. Output Rigorous JSON
    printf("{\n");
    printf("  \"model_file\": \"%s\",\n", model_path);
    printf("  \"runtime\": \"ONNXRuntime-C-API\",\n");
    printf("  \"backend\": \"CPUExecutionProvider\",\n");
    printf("  \"threads\": 4,\n");
    printf("  \"iterations\": %d,\n", num_iters);
    printf("  \"load_time_ms\": %.3f,\n", load_time_ms);
    printf("  \"cold_start_ms\": %.3f,\n", cold_start_ms);
    printf("  \"model_p50_ms\": %.3f,\n", stats.p50_ms);
    printf("  \"model_p95_ms\": %.3f,\n", stats.p95_ms);
    printf("  \"model_p99_ms\": %.3f,\n", stats.p99_ms);
    printf("  \"model_min_ms\": %.3f,\n", stats.min_ms);
    printf("  \"model_max_ms\": %.3f,\n", stats.max_ms);
    printf("  \"model_mean_ms\": %.3f,\n", stats.mean_ms);
    printf("  \"rss_before_load_kb\": %ld,\n", rss_before_load_kb);
    printf("  \"rss_after_load_kb\": %ld,\n", rss_after_load_kb);
    printf("  \"peak_rss_kb\": %ld,\n", peak_rss_kb);
    printf("  \"rss_delta_mb\": %.2f,\n", (peak_rss_kb - rss_before_load_kb) / 1024.0);
    printf("  \"peak_rss_mb\": %.2f\n", peak_rss_kb / 1024.0);
    printf("}\n");

    // Clean up
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    return 0;
}
