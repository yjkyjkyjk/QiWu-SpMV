#include "spmv_benchmark.h"
#include "benchmark_device_interface.h"

void SpMV_Benchmark::generate_report_filename() {
    std::time_t now = std::time(nullptr);
    std::tm* timeinfo = std::localtime(&now);
    std::ostringstream oss;
    oss << "spmv-benchmark-"
        << std::setfill('0')
        << std::setw(4) << (timeinfo->tm_year + 1900)
        << std::setw(2) << (timeinfo->tm_mon + 1)
        << std::setw(2) << timeinfo->tm_mday
        << std::setw(2) << timeinfo->tm_hour
        << std::setw(2) << timeinfo->tm_min
        << std::setw(2) << timeinfo->tm_sec
        << ".txt";
    report_filename = oss.str();
}

SpMV_Benchmark::SpMV_Benchmark(int nrows, int ncols, int nnz)
    : SparseBenchmarkBase(nrows, ncols, nnz) {
    initialize_vectors();
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    allocate_memory();
#endif
}

SpMV_Benchmark::SpMV_Benchmark(const std::string& mtx_file)
    : SparseBenchmarkBase(mtx_file) {
    initialize_vectors();
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    allocate_memory();
#endif
}

void SpMV_Benchmark::initialize_vectors() {
    x.resize(ncols);
    y.resize(nrows);
    reference_y.resize(nrows);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < ncols; ++i) {
        x[i] = static_cast<BenchmarkValue>(dis(gen));
    }

    for (int i = 0; i < nrows; ++i) {
        y[i] = static_cast<BenchmarkValue>(0);
        reference_y[i] = static_cast<BenchmarkValue>(0);
    }
}

void SpMV_Benchmark::spmv_serial() {
    for (int i = 0; i < nrows; ++i) {
        BenchmarkValue sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        reference_y[i] = sum;
    }
}

void SpMV_Benchmark::run_spmv_kernel() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    spmv_optimized_cuda();
#elif defined(HIP_ENABLED) && HIP_ENABLED
    spmv_optimized_hip();
#else
    spmv_optimized();
#endif
}

void SpMV_Benchmark::warm_up_cache(int iterations) {
    std::cout << "Warming up cache with " << iterations << " iterations..." << std::endl;
    for (int i = 0; i < iterations; ++i) {
        run_spmv_kernel();
    }
    std::cout << "Cache warm-up completed." << std::endl;
}

std::pair<double, double> SpMV_Benchmark::benchmark_spmv(int iterations) {
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    spmv_preprocess();
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count();

    warm_up_cache(10);

#if defined(CUDA_ENABLED) && CUDA_ENABLED
    synchronize_cuda();
    set_cuda_sync_enabled(0);
#endif

    auto spmv_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        run_spmv_kernel();
    }

#if defined(CUDA_ENABLED) && CUDA_ENABLED
    synchronize_cuda();
    set_cuda_sync_enabled(1);
#endif

    auto spmv_end = std::chrono::high_resolution_clock::now();
    auto spmv_duration = std::chrono::duration_cast<std::chrono::microseconds>(spmv_end - spmv_start).count();

    return std::make_pair(
        static_cast<double>(preprocess_duration),
        static_cast<double>(spmv_duration) / iterations
    );
}

bool SpMV_Benchmark::validate_correctness() {
    spmv_serial();

#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    memset_device(d_y, 0, y.size() * sizeof(BenchmarkValue));
#endif
    run_spmv_kernel();

#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    copy_device_to_host(y.data(), d_y, y.size() * sizeof(BenchmarkValue));
    free_memory();
#endif

    double diff_norm = 0.0;
    double ref_norm = 0.0;
    for (int i = 0; i < nrows; ++i) {
        double diff = y[i] - reference_y[i];
        diff_norm += diff * diff;
        ref_norm += reference_y[i] * reference_y[i];
    }

    double relative_error = ref_norm > 0.0 ? sqrt(diff_norm) / sqrt(ref_norm) : sqrt(diff_norm);
    const double machine_epsilon = std::numeric_limits<BenchmarkValue>::epsilon();
    const double tolerance = 1e6 * machine_epsilon;

    std::cout << "Reference vector norm: " << sqrt(ref_norm) << std::endl;
    std::cout << "Difference vector norm: " << sqrt(diff_norm) << std::endl;
    std::cout << "Relative error: " << relative_error << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;

    bool passed = relative_error <= tolerance;
    std::cout << "Correctness validation: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

double SpMV_Benchmark::calculate_performance(double spmv_time_us) {
    if (spmv_time_us <= 0.0) {
        return 0.0;
    }
    long long total_ops = static_cast<long long>(nnz) * 2;
    double time_seconds = spmv_time_us / 1e6;
    return (total_ops / time_seconds) / 1e9;
}

void SpMV_Benchmark::print_matrix_info() {
    std::cout << "Matrix Name: " << input_filename << std::endl;
    std::cout << "Matrix Size: " << nrows << "x" << ncols << std::endl;
    std::cout << "Non-zeros per row: " << average_nnz_per_row() << std::endl;
    std::cout << "Total non-zeros: " << nnz << std::endl;
    std::cout << "Sparsity: " << sparsity() << std::endl;
}

void SpMV_Benchmark::set_report_file(const std::string& reportfile) {
    if (reportfile.empty()) {
        generate_report_filename();
    }
    else {
        report_filename = reportfile;
    }
}

void SpMV_Benchmark::write_report(std::pair<double, double> timing_results, double perf_gflops, bool correct) {
    std::ofstream report_file(report_filename, std::ios::app);
    if (!report_file.is_open()) {
        std::cerr << "Error: Cannot create report file " << report_filename << std::endl;
        return;
    }

    report_file << "SpMV Benchmark Report\n";
    report_file << "=====================\n";
    std::time_t now = std::time(nullptr);
    report_file << "Date: " << std::asctime(std::localtime(&now)) << "\n";
    report_file << "Matrix Information:\n";
    report_file << "  Name: " << input_filename << "\n";
    report_file << "  Size: " << nrows << "x" << ncols << "\n";
    report_file << "  Non-zeros per row: " << average_nnz_per_row() << "\n";
    report_file << "  Total non-zeros: " << nnz << "\n";
    report_file << "  Sparsity: " << sparsity() << "\n\n";

    report_file << "Benchmark Results:\n";
    report_file << "  Operator: SpMV\n";
    report_file << "  Kernel: " << kernel_name << "\n";
    report_file << "  Precision: " << benchmark_precision_name() << "\n";
    report_file << "  Preprocessing time: " << timing_results.first << " microseconds\n";
    report_file << "  Average SpMV execution time: " << timing_results.second << " microseconds\n";
    report_file << "  Performance: " << perf_gflops << " GFLOPS\n\n";

    long long total_ops = static_cast<long long>(nnz) * 2;
    report_file << "Additional Metrics:\n";
    report_file << "  Total operations: " << total_ops << " (multiply-adds)\n";
    report_file << "  Memory accessed (approx): "
                << ((ncols + nrows) * sizeof(BenchmarkValue) + nnz * (sizeof(BenchmarkValue) + sizeof(int)))
                << " bytes\n\n";

    report_file << "Correctness Validation:\n";
    report_file << "  Result: " << (correct ? "PASSED" : "FAILED") << "\n\n";

    report_file.close();
    std::cout << "Report saved to: " << report_filename << std::endl;
}
