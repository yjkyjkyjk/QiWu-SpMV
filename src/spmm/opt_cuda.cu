#include "spmm_benchmark.h"

#if defined(CUDA_ENABLED) && CUDA_ENABLED
#include <cuda_runtime.h>

__global__ void csr_spmm_kernel(
    const int n,
    const int dense_cols,
    const BenchmarkValue* values,
    const int* col_indices,
    const int* row_pointers,
    const BenchmarkValue* x,
    BenchmarkValue* y
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = n * dense_cols;
    if (idx >= total_outputs) {
        return;
    }

    const int row = idx / dense_cols;
    const int rhs_col = idx % dense_cols;
    BenchmarkValue sum = 0.0;
    for (int j = row_pointers[row]; j < row_pointers[row + 1]; ++j) {
        const int col = col_indices[j];
        sum += values[j] * x[col * dense_cols + rhs_col];
    }
    y[idx] = sum;
}
#endif

void SpMM_Benchmark::spmm_optimized_cuda() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    int block_size = 256;
    int total_outputs = nrows * dense_cols;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    csr_spmm_kernel<<<grid_size, block_size>>>(nrows, dense_cols, d_values, d_col_idx, d_row_ptr, d_x, d_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA SpMM kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
#else
    std::cerr << "CUDA not available, Exit." << std::endl;
    exit(1);
#endif
}
