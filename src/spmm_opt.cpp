#include "spmm_benchmark.h"

void SpMM_Benchmark::spmm_optimized() {
    // Dense RHS and output are row-major matrices:
    // x is [ncols][dense_cols], y is [nrows][dense_cols].
    std::fill(y.begin(), y.end(), static_cast<SpMVValue>(0));

    for (int row = 0; row < nrows; ++row) {
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            const int col = col_idx[j];
            const SpMVValue val = values[j];
            for (int k = 0; k < dense_cols; ++k) {
                y[row * dense_cols + k] += val * x[col * dense_cols + k];
            }
        }
    }
}
