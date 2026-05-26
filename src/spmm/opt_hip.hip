#include "spmm_benchmark.h"
#include <hip/hip_runtime.h>

#if defined(HIP_ENABLED) && HIP_ENABLED
__global__ void hip_spmm_kernel(const BenchmarkValue* values, const int* col_idx, const int* row_ptr,
                                const BenchmarkValue* x, BenchmarkValue* y, int n, int dense_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = n * dense_cols;
    if (idx >= total_outputs) {
        return;
    }

    int row = idx / dense_cols;
    int rhs_col = idx % dense_cols;
    BenchmarkValue sum = 0.0;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    for (int j = start; j < end; ++j) {
        sum += values[j] * x[col_idx[j] * dense_cols + rhs_col];
    }
    y[idx] = sum;
}
#endif

void SpMM_Benchmark::spmm_optimized_hip() {
#if defined(HIP_ENABLED) && HIP_ENABLED
    int block_size = 256;
    int total_outputs = nrows * dense_cols;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    hipLaunchKernelGGL(hip_spmm_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       d_values, d_col_idx, d_row_ptr, d_x, d_y, nrows, dense_cols);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP SpMM kernel launch error: " << hipGetErrorString(err) << std::endl;
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP sync failed: %s\n", hipGetErrorString(err));
    }
#else
    std::cerr << "HIP not available, Exit." << std::endl;
    exit(1);
#endif
}
