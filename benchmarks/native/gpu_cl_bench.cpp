#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <dlfcn.h>

// Minimal OpenCL dynamic typedefs
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
typedef size_t cl_device_info;

typedef cl_int (*fn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*fn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
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

static inline double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

const char* KERNEL_SRC =
"__kernel void matvec_16_32(__global const float* W, __global const float* x, __global const float* b, __global float* y) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < 32) {\n"
"        float sum = b[gid];\n"
"        for (int j = 0; j < 16; ++j) {\n"
"            sum += x[j] * W[gid * 16 + j];\n"
"        }\n"
"        y[gid] = sum > 0.0f ? sum : 0.0f;\n"
"    }\n"
"}\n";

int main() {
    void* handle = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    if (!handle) handle = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW);
    if (!handle) {
        printf("{\"gpu_available\": false, \"reason\": \"dlopen failed\"}\n");
        return 0;
    }

    auto p_clGetPlatformIDs = (fn_clGetPlatformIDs)dlsym(handle, "clGetPlatformIDs");
    auto p_clGetDeviceIDs = (fn_clGetDeviceIDs)dlsym(handle, "clGetDeviceIDs");
    auto p_clGetDeviceInfo = (fn_clGetDeviceInfo)dlsym(handle, "clGetDeviceInfo");
    auto p_clCreateContext = (fn_clCreateContext)dlsym(handle, "clCreateContext");
    auto p_clCreateCommandQueue = (fn_clCreateCommandQueue)dlsym(handle, "clCreateCommandQueue");
    auto p_clCreateProgramWithSource = (fn_clCreateProgramWithSource)dlsym(handle, "clCreateProgramWithSource");
    auto p_clBuildProgram = (fn_clBuildProgram)dlsym(handle, "clBuildProgram");
    auto p_clCreateKernel = (fn_clCreateKernel)dlsym(handle, "clCreateKernel");
    auto p_clCreateBuffer = (fn_clCreateBuffer)dlsym(handle, "clCreateBuffer");
    auto p_clSetKernelArg = (fn_clSetKernelArg)dlsym(handle, "clSetKernelArg");
    auto p_clEnqueueWriteBuffer = (fn_clEnqueueWriteBuffer)dlsym(handle, "clEnqueueWriteBuffer");
    auto p_clEnqueueNDRangeKernel = (fn_clEnqueueNDRangeKernel)dlsym(handle, "clEnqueueNDRangeKernel");
    auto p_clEnqueueReadBuffer = (fn_clEnqueueReadBuffer)dlsym(handle, "clEnqueueReadBuffer");
    auto p_clFinish = (fn_clFinish)dlsym(handle, "clFinish");

    if (!p_clGetPlatformIDs || !p_clGetDeviceIDs || !p_clCreateContext) {
        printf("{\"gpu_available\": false, \"reason\": \"dlsym failed for core OpenCL APIs\"}\n");
        dlclose(handle);
        return 0;
    }

    cl_platform_id platform;
    cl_uint num_platforms = 0;
    cl_int err = p_clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != 0 || num_platforms == 0) {
        printf("{\"gpu_available\": false, \"reason\": \"No OpenCL platform found\"}\n");
        dlclose(handle);
        return 0;
    }

    cl_device_id device;
    cl_uint num_devices = 0;
    err = p_clGetDeviceIDs(platform, 0xFFFFFFFF, 1, &device, &num_devices);
    if (err != 0 || num_devices == 0) {
        printf("{\"gpu_available\": false, \"reason\": \"No OpenCL device found\"}\n");
        dlclose(handle);
        return 0;
    }

    char device_name[128] = {0};
    p_clGetDeviceInfo(device, 0x102B /* CL_DEVICE_NAME */, sizeof(device_name), device_name, NULL);

    cl_context ctx = p_clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = p_clCreateCommandQueue(ctx, device, 0, &err);

    // Build Kernel
    size_t src_len = strlen(KERNEL_SRC);
    cl_program prog = p_clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, &src_len, &err);
    p_clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = p_clCreateKernel(prog, "matvec_16_32", &err);

    // Buffers
    cl_mem buf_W = p_clCreateBuffer(ctx, 1 /* CL_MEM_READ_ONLY */, 16 * 32 * sizeof(float), NULL, &err);
    cl_mem buf_x = p_clCreateBuffer(ctx, 1 /* CL_MEM_READ_ONLY */, 16 * sizeof(float), NULL, &err);
    cl_mem buf_b = p_clCreateBuffer(ctx, 1 /* CL_MEM_READ_ONLY */, 32 * sizeof(float), NULL, &err);
    cl_mem buf_y = p_clCreateBuffer(ctx, 2 /* CL_MEM_WRITE_ONLY */, 32 * sizeof(float), NULL, &err);

    p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_W);
    p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_x);
    p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_b);
    p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_y);

    float W[512], x[16], b[32], y[32];
    for (int i = 0; i < 512; ++i) W[i] = 0.01f * (i % 7);
    for (int i = 0; i < 16; ++i) x[i] = 0.1f * i;
    for (int i = 0; i < 32; ++i) b[i] = 0.05f;

    p_clEnqueueWriteBuffer(queue, buf_W, 1, 0, 512 * sizeof(float), W, 0, NULL, NULL);
    p_clEnqueueWriteBuffer(queue, buf_b, 1, 0, 32 * sizeof(float), b, 0, NULL, NULL);
    p_clFinish(queue);

    // Benchmark 200 iterations
    std::vector<double> gpu_latencies_us;
    size_t global_work_size = 32;

    for (int i = 0; i < 200; ++i) {
        double t0 = get_time_ns();
        p_clEnqueueWriteBuffer(queue, buf_x, 0, 0, 16 * sizeof(float), x, 0, NULL, NULL);
        p_clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        p_clEnqueueReadBuffer(queue, buf_y, 1, 0, 32 * sizeof(float), y, 0, NULL, NULL);
        p_clFinish(queue);
        double t1 = get_time_ns();
        gpu_latencies_us.push_back((t1 - t0) / 1000.0);
    }

    std::sort(gpu_latencies_us.begin(), gpu_latencies_us.end());
    size_t n = gpu_latencies_us.size();
    double p50 = gpu_latencies_us[n / 2];
    double p95 = gpu_latencies_us[(size_t)(0.95 * (n - 1))];
    double p99 = gpu_latencies_us[(size_t)(0.99 * (n - 1))];

    printf("{\n");
    printf("  \"gpu_available\": true,\n");
    printf("  \"device_name\": \"%s\",\n", device_name);
    printf("  \"backend\": \"OpenCL (Adreno 810 Native)\",\n");
    printf("  \"gpu_benchmark\": {\n");
    printf("    \"iterations\": 200,\n");
    printf("    \"p50_us\": %.2f,\n", p50);
    printf("    \"p95_us\": %.2f,\n", p95);
    printf("    \"p99_us\": %.2f,\n", p99);
    printf("    \"p50_ms\": %.5f,\n", p50 / 1000.0);
    printf("    \"p95_ms\": %.5f\n", p95 / 1000.0);
    printf("  }\n");
    printf("}\n");

    dlclose(handle);
    return 0;
}
