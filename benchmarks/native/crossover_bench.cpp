#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <dlfcn.h>

typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long cl_ulong;

typedef cl_int (*fn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*fn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*fn_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_program (*fn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*fn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_kernel (*fn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (*fn_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_int (*fn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*fn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, unsigned int, size_t, size_t, const void*, cl_uint, const void*, void*);
typedef cl_int (*fn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int (*fn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, unsigned int, size_t, size_t, void*, cl_uint, const void*, void*);
typedef cl_int (*fn_clFinish)(cl_command_queue);
typedef cl_int (*fn_clReleaseMemObject)(cl_mem);
typedef cl_int (*fn_clReleaseKernel)(cl_kernel);
typedef cl_int (*fn_clReleaseProgram)(cl_program);

static inline double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

const char* GEMV_KERNEL =
"__kernel void gemv(int M, int K, __global const float* A, __global const float* x, __global float* y) {\n"
"    int row = get_global_id(0);\n"
"    if (row < M) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < K; ++k) {\n"
"            sum += A[row * K + k] * x[k];\n"
"        }\n"
"        y[row] = sum;\n"
"    }\n"
"}\n";

struct Workload {
    const char* label;
    int M;
    int K;
    int total_params;
};

int main() {
    Workload workloads[] = {
        {"2K", 32, 64, 2048},
        {"10K", 100, 100, 10000},
        {"100K", 250, 400, 100000},
        {"500K", 500, 1000, 500000},
        {"1M", 1000, 1000, 1000000},
        {"5M", 2000, 2500, 5000000},
        {"10M", 2500, 4000, 10000000},
        {"25M", 5000, 5000, 25000000},
        {"50M", 5000, 10000, 50000000}
    };
    int num_workloads = sizeof(workloads) / sizeof(workloads[0]);

    // OpenCL Dynamic Load
    void* cl_lib = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    if (!cl_lib) cl_lib = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW);

    fn_clGetPlatformIDs p_clGetPlatformIDs = (fn_clGetPlatformIDs)dlsym(cl_lib, "clGetPlatformIDs");
    fn_clGetDeviceIDs p_clGetDeviceIDs = (fn_clGetDeviceIDs)dlsym(cl_lib, "clGetDeviceIDs");
    fn_clCreateContext p_clCreateContext = (fn_clCreateContext)dlsym(cl_lib, "clCreateContext");
    fn_clCreateCommandQueue p_clCreateCommandQueue = (fn_clCreateCommandQueue)dlsym(cl_lib, "clCreateCommandQueue");
    fn_clCreateProgramWithSource p_clCreateProgramWithSource = (fn_clCreateProgramWithSource)dlsym(cl_lib, "clCreateProgramWithSource");
    fn_clBuildProgram p_clBuildProgram = (fn_clBuildProgram)dlsym(cl_lib, "clBuildProgram");
    fn_clCreateKernel p_clCreateKernel = (fn_clCreateKernel)dlsym(cl_lib, "clCreateKernel");
    fn_clCreateBuffer p_clCreateBuffer = (fn_clCreateBuffer)dlsym(cl_lib, "clCreateBuffer");
    fn_clSetKernelArg p_clSetKernelArg = (fn_clSetKernelArg)dlsym(cl_lib, "clSetKernelArg");
    fn_clEnqueueWriteBuffer p_clEnqueueWriteBuffer = (fn_clEnqueueWriteBuffer)dlsym(cl_lib, "clEnqueueWriteBuffer");
    fn_clEnqueueNDRangeKernel p_clEnqueueNDRangeKernel = (fn_clEnqueueNDRangeKernel)dlsym(cl_lib, "clEnqueueNDRangeKernel");
    fn_clEnqueueReadBuffer p_clEnqueueReadBuffer = (fn_clEnqueueReadBuffer)dlsym(cl_lib, "clEnqueueReadBuffer");
    fn_clFinish p_clFinish = (fn_clFinish)dlsym(cl_lib, "clFinish");
    fn_clReleaseMemObject p_clReleaseMemObject = (fn_clReleaseMemObject)dlsym(cl_lib, "clReleaseMemObject");
    fn_clReleaseKernel p_clReleaseKernel = (fn_clReleaseKernel)dlsym(cl_lib, "clReleaseKernel");
    fn_clReleaseProgram p_clReleaseProgram = (fn_clReleaseProgram)dlsym(cl_lib, "clReleaseProgram");

    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cl_program prog = NULL;
    cl_kernel kernel = NULL;

    if (cl_lib && p_clGetPlatformIDs) {
        cl_platform_id plat;
        cl_uint num_p;
        p_clGetPlatformIDs(1, &plat, &num_p);
        cl_device_id dev;
        cl_uint num_d;
        p_clGetDeviceIDs(plat, 0xFFFFFFFF, 1, &dev, &num_d);
        cl_int err;
        ctx = p_clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
        queue = p_clCreateCommandQueue(ctx, dev, 0, &err);
        size_t src_len = strlen(GEMV_KERNEL);
        prog = p_clCreateProgramWithSource(ctx, 1, &GEMV_KERNEL, &src_len, &err);
        p_clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
        kernel = p_clCreateKernel(prog, "gemv", &err);
    }

    printf("[\n");
    for (int w = 0; w < num_workloads; ++w) {
        int M = workloads[w].M;
        int K = workloads[w].K;
        size_t size_A = (size_t)M * K;

        std::vector<float> A(size_A, 0.01f);
        std::vector<float> x(K, 0.02f);
        std::vector<float> y(M, 0.0f);

        // CPU Benchmark
        // Warmup
        for (int i = 0; i < 5; ++i) {
            for (int r = 0; r < M; ++r) {
                float sum = 0.0f;
                for (int c = 0; c < K; ++c) sum += A[r * K + c] * x[c];
                y[r] = sum;
            }
        }
        std::vector<double> cpu_times;
        int iters = (M * K > 5000000) ? 20 : 50;
        for (int i = 0; i < iters; ++i) {
            double t0 = get_time_ns();
            for (int r = 0; r < M; ++r) {
                float sum = 0.0f;
                for (int c = 0; c < K; ++c) sum += A[r * K + c] * x[c];
                y[r] = sum;
            }
            double t1 = get_time_ns();
            cpu_times.push_back((t1 - t0) / 1000.0);
        }
        std::sort(cpu_times.begin(), cpu_times.end());
        double cpu_p50_us = cpu_times[cpu_times.size() / 2];

        // GPU Benchmark
        double gpu_p50_us = 0.0;
        if (kernel && queue) {
            cl_int err;
            cl_mem buf_A = p_clCreateBuffer(ctx, 1, size_A * sizeof(float), NULL, &err);
            cl_mem buf_x = p_clCreateBuffer(ctx, 1, K * sizeof(float), NULL, &err);
            cl_mem buf_y = p_clCreateBuffer(ctx, 2, M * sizeof(float), NULL, &err);

            p_clSetKernelArg(kernel, 0, sizeof(int), &M);
            p_clSetKernelArg(kernel, 1, sizeof(int), &K);
            p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_A);
            p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_x);
            p_clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_y);

            p_clEnqueueWriteBuffer(queue, buf_A, 1, 0, size_A * sizeof(float), A.data(), 0, NULL, NULL);
            p_clFinish(queue);

            // Warmup
            size_t global_size = M;
            for (int i = 0; i < 5; ++i) {
                p_clEnqueueWriteBuffer(queue, buf_x, 1, 0, K * sizeof(float), x.data(), 0, NULL, NULL);
                p_clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
                p_clEnqueueReadBuffer(queue, buf_y, 1, 0, M * sizeof(float), y.data(), 0, NULL, NULL);
                p_clFinish(queue);
            }

            std::vector<double> gpu_times;
            for (int i = 0; i < iters; ++i) {
                double t0 = get_time_ns();
                p_clEnqueueWriteBuffer(queue, buf_x, 0, 0, K * sizeof(float), x.data(), 0, NULL, NULL);
                p_clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
                p_clEnqueueReadBuffer(queue, buf_y, 1, 0, M * sizeof(float), y.data(), 0, NULL, NULL);
                p_clFinish(queue);
                double t1 = get_time_ns();
                gpu_times.push_back((t1 - t0) / 1000.0);
            }
            std::sort(gpu_times.begin(), gpu_times.end());
            gpu_p50_us = gpu_times[gpu_times.size() / 2];

            p_clReleaseMemObject(buf_A);
            p_clReleaseMemObject(buf_x);
            p_clReleaseMemObject(buf_y);
        }

        printf("  {\n");
        printf("    \"workload\": \"%s\",\n", workloads[w].label);
        printf("    \"parameters\": %d,\n", workloads[w].total_params);
        printf("    \"cpu_p50_us\": %.2f,\n", cpu_p50_us);
        printf("    \"cpu_p50_ms\": %.4f,\n", cpu_p50_us / 1000.0);
        printf("    \"gpu_p50_us\": %.2f,\n", gpu_p50_us);
        printf("    \"gpu_p50_ms\": %.4f,\n", gpu_p50_us / 1000.0);
        printf("    \"gpu_speedup_vs_cpu\": %.2f,\n", cpu_p50_us / (gpu_p50_us > 0 ? gpu_p50_us : 1.0));
        printf("    \"crossover_winner\": \"%s\"\n", (cpu_p50_us < gpu_p50_us) ? "CPU" : "GPU");
        printf("  }%s\n", (w < num_workloads - 1) ? "," : "");
    }
    printf("]\n");

    if (kernel) p_clReleaseKernel(kernel);
    if (prog) p_clReleaseProgram(prog);
    if (cl_lib) dlclose(cl_lib);
    return 0;
}
