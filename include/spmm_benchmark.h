#ifndef SPMM_BENCHMARK_H
#define SPMM_BENCHMARK_H

#include "spmv_benchmark.h"

class SpMM_Benchmark : public SpMV_Benchmark {
private:
    int dense_cols = 8;

    void generate_spmm_report_filename();
    void initialize_dense_matrices();
    void run_spmm_kernel();
    void spmm_serial();

public:
    explicit SpMM_Benchmark(const std::string& mtx_file, int dense_cols = 8);

    void spmm_preprocess();
    void spmm_optimized();
    void spmm_optimized_cuda();
    void spmm_optimized_hip();
    void warm_up_cache(int iterations = 10);
    std::pair<double, double> benchmark_spmm(int iterations = 10);
    bool validate_correctness();
    double calculate_performance(double spmm_time_us);
    void print_matrix_info();
    void set_report_file(const std::string& reportfile);
    void write_report(std::pair<double, double> timing_results, double perf_gflops, bool correct);
};

#endif // SPMM_BENCHMARK_H
