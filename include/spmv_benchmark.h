#ifndef SPMV_BENCHMARK_H
#define SPMV_BENCHMARK_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <map>
#include <utility>
#include <limits>
#include <string>

#if defined(SPMV_USE_FLOAT) && SPMV_USE_FLOAT
using SpMVValue = float;
#else
using SpMVValue = double;
#endif

inline const char* spmv_precision_name() {
#if defined(SPMV_USE_FLOAT) && SPMV_USE_FLOAT
    return "float";
#else
    return "double";
#endif
}

class SpMV_Benchmark {
protected:
    int nrows;                  // Number of rows
    int ncols;                  // Number of columns
    int nnz;                    // Total number of non-zero elements
    std::vector<SpMVValue> values; // Matrix values
    std::vector<int> col_idx;   // Column indices
    std::vector<int> row_ptr;   // Row pointers
    std::vector<SpMVValue> x;      // Input vector
    std::vector<SpMVValue> y;      // Output vector
    std::vector<SpMVValue> reference_y; // Reference result
    std::string report_filename;
    std::string input_filename;
    std::string kernel_name;

#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    // Device pointers for CUDA/HIP
    SpMVValue *d_values = nullptr;
    int *d_col_idx = nullptr;
    int *d_row_ptr = nullptr;
    SpMVValue *d_x = nullptr;
    SpMVValue *d_y = nullptr;
#endif

    void generate_report_filename();
    void convert_coo_to_csr(const std::vector<int>& coo_rows, 
                           const std::vector<int>& coo_cols, 
                           const std::vector<SpMVValue>& coo_vals,
                           int nrows, int nnz);

public:
    SpMV_Benchmark(int nrows, int ncols, int nnz);
    SpMV_Benchmark(const std::string& mtx_file);
    ~SpMV_Benchmark();

    void load_matrix_from_mtx(const std::string& filename);
    void initialize_matrix();
    void initialize_vectors();
    void allocate_memory();
    void free_memory();
    void run_spmv_kernel();
    void spmv_serial();
    void spmv_preprocess();
    void spmv_preprocess_cuda();
    void spmv_preprocess_hip();
    void spmv_optimized();
    void spmv_optimized_cuda();
    void spmv_optimized_hip();
    void warm_up_cache(int iterations = 10);
    std::pair<double, double> benchmark_spmv(int iterations = 10);
    bool validate_correctness();
    double calculate_performance(double spmv_time_us);
    void print_matrix_info();
    void set_kernel_name(const std::string& kernelname);
    void set_report_file(const std::string& reportfile);
    void write_report(std::pair<double, double> timing_results, double perf_gflops, bool correct);
};

#endif // SPMV_BENCHMARK_H
