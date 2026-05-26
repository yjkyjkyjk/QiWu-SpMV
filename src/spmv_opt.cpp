#include "spmv_benchmark.h"
#include <string>
#include <iomanip>
#include <map>
#include <utility>
using namespace std;

void SpMV_Benchmark::spmv_preprocess() {
    // Preprocessing function - users can add their own preprocessing operations here
    std::cout << "Executing preprocessing steps..." << std::endl;
    
    // Example preprocessing operations:
    // 1. Reorder matrix rows for better cache efficiency
    // 2. Pre-calculate some intermediate results
    // 3. Initialize data structures required for parallel computation
}

void SpMV_Benchmark::spmv_optimized() {
    // Initialize output vector to zero
    fill(y.begin(), y.end(), static_cast<SpMVValue>(0));

    ///////////////////////////////////////////////////////////////////////////
    // TODO: Students should implement their optimized SpMV kernel here.
    // This is a simple serial implementation as a starting point.
    // You can optimize this code using techniques such as:
    // 1. Loop unrolling
    // 2. SIMD instructions (AVX, AVX2, AVX-512)
    // 3. Parallelization (OpenMP, pthreads, std::thread)
    // 4. Cache-aware optimizations (blocking, reordering)
    // 5. Preprocessing (matrix reordering, format conversion)
    //
    // Current implementation: basic CSR serial SpMV
    ///////////////////////////////////////////////////////////////////////////

    // Simple serial CSR SpMV
    for (int i = 0; i < nrows; ++i) {
        SpMVValue sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}
