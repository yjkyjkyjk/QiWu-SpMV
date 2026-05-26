#ifndef SPARSE_BENCHMARK_BASE_H
#define SPARSE_BENCHMARK_BASE_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(BENCHMARK_USE_FLOAT) && BENCHMARK_USE_FLOAT
using BenchmarkValue = float;
#else
using BenchmarkValue = double;
#endif

inline const char* benchmark_precision_name() {
#if defined(BENCHMARK_USE_FLOAT) && BENCHMARK_USE_FLOAT
    return "float";
#else
    return "double";
#endif
}

class SparseBenchmarkBase {
protected:
    int nrows = 0;
    int ncols = 0;
    int nnz = 0;
    std::vector<BenchmarkValue> values;
    std::vector<int> col_idx;
    std::vector<int> row_ptr;
    std::vector<BenchmarkValue> x;
    std::vector<BenchmarkValue> y;
    std::vector<BenchmarkValue> reference_y;
    std::string report_filename;
    std::string input_filename;
    std::string kernel_name;

#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    BenchmarkValue *d_values = nullptr;
    int *d_col_idx = nullptr;
    int *d_row_ptr = nullptr;
    BenchmarkValue *d_x = nullptr;
    BenchmarkValue *d_y = nullptr;
#endif

    SparseBenchmarkBase(int nrows, int ncols, int nnz);
    explicit SparseBenchmarkBase(const std::string& mtx_file);
    virtual ~SparseBenchmarkBase();

    void load_matrix_from_mtx(const std::string& filename);
    void initialize_matrix();
    void allocate_memory();
    void free_memory();
    double sparsity() const;
    int average_nnz_per_row() const;

private:
    void set_input_filename(const std::string& path);
    void convert_coo_to_csr(const std::vector<int>& coo_rows,
                            const std::vector<int>& coo_cols,
                            const std::vector<BenchmarkValue>& coo_vals,
                            int nrows,
                            int nnz);

public:
    void set_kernel_name(const std::string& kernelname);
};

#endif // SPARSE_BENCHMARK_BASE_H
