#ifndef SPMV_BENCHMARK_H
#define SPMV_BENCHMARK_H

#include "sparse_benchmark_base.h"

class SpMV_Benchmark : public SparseBenchmarkBase {
private:
    void generate_report_filename();

public:
    SpMV_Benchmark(int nrows, int ncols, int nnz);
    SpMV_Benchmark(const std::string& mtx_file);

    void initialize_vectors();
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
    void set_report_file(const std::string& reportfile);
    void write_report(std::pair<double, double> timing_results, double perf_gflops, bool correct);
};

#endif // SPMV_BENCHMARK_H
