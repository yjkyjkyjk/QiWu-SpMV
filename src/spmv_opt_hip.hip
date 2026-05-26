#include "spmv_benchmark.h"
#include <hip/hip_runtime.h>
#include <string>
using namespace std;

void SpMV_Benchmark::spmv_preprocess_hip() {
    // Preprocessing function - users can add their own preprocessing operations here
    std::cout << "Executing preprocessing steps..." << std::endl;
    
    // Example preprocessing operations:
    // 1. Reorder matrix rows for better cache efficiency
    // 2. Pre-calculate some intermediate results
    // 3. Initialize data structures required for parallel computation
}

__global__ void my_hip_kernel(const SpMVValue* values, const int* col_idx, const int* row_ptr,
                              const SpMVValue* x, SpMVValue* y, int n) {
    // HIP kernel 代码
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        SpMVValue sum = 0.0;
        int start = row_ptr[idx];
        int end = row_ptr[idx + 1];
        for (int j = start; j < end; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[idx] = sum;
    }
}

void SpMV_Benchmark::spmv_optimized_hip() {
    // Optimized SpMV function for HIP
#if defined(HIP_ENABLED) && HIP_ENABLED
    int block_size = 256;
    int grid_size = (nrows + block_size - 1) / block_size;
    hipLaunchKernelGGL(my_hip_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       d_values, d_col_idx, d_row_ptr, d_x, d_y, nrows);

    // Check for kernel launch errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP kernel launch error: " << hipGetErrorString(err) << std::endl;
    }

    // Wait for kernel to complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP sync failed: %s\n", hipGetErrorString(err));
    }
#else
    std::cerr << "HIP not available, Exit." << std::endl;
    exit(1);
#endif
}
