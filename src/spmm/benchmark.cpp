#include "spmm_benchmark.h"
#include "benchmark_device_interface.h"

void SpMM_Benchmark::generate_spmm_report_filename() {
    std::time_t now = std::time(nullptr);
    std::tm* timeinfo = std::localtime(&now);
    std::ostringstream oss;
    oss << "spmm-benchmark-"
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

SpMM_Benchmark::SpMM_Benchmark(const std::string& mtx_file, int dense_cols)
    : SparseBenchmarkBase(mtx_file), dense_cols(std::max(1, dense_cols)) {
    initialize_dense_matrices();
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    allocate_memory();
#endif
}

void SpMM_Benchmark::initialize_dense_matrices() {
    const int x_size = ncols * dense_cols;
    const int y_size = nrows * dense_cols;
    x.resize(x_size);
    y.resize(y_size);
    reference_y.resize(y_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < x_size; ++i) {
        x[i] = static_cast<BenchmarkValue>(dis(gen));
    }

    for (int i = 0; i < y_size; ++i) {
        y[i] = static_cast<BenchmarkValue>(0);
        reference_y[i] = static_cast<BenchmarkValue>(0);
    }
}

void SpMM_Benchmark::spmm_serial() {
    std::fill(reference_y.begin(), reference_y.end(), static_cast<BenchmarkValue>(0));

    for (int row = 0; row < nrows; ++row) {
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            const int col = col_idx[j];
            const BenchmarkValue val = values[j];
            for (int k = 0; k < dense_cols; ++k) {
                reference_y[row * dense_cols + k] += val * x[col * dense_cols + k];
            }
        }
    }
}

void SpMM_Benchmark::run_spmm_kernel() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    spmm_optimized_cuda();
#elif defined(HIP_ENABLED) && HIP_ENABLED
    spmm_optimized_hip();
#else
    spmm_optimized();
#endif
}

void SpMM_Benchmark::spmm_preprocess() {
    std::cout << "Executing SpMM preprocessing steps..." << std::endl;
}

void SpMM_Benchmark::warm_up_cache(int iterations) {
    std::cout << "Warming up cache with " << iterations << " SpMM iterations..." << std::endl;
    for (int i = 0; i < iterations; ++i) {
        run_spmm_kernel();
    }
    std::cout << "Cache warm-up completed." << std::endl;
}

std::pair<double, double> SpMM_Benchmark::benchmark_spmm(int iterations) {
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    spmm_preprocess();
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count();

    warm_up_cache(10);

#if defined(CUDA_ENABLED) && CUDA_ENABLED
    synchronize_cuda();
    set_cuda_sync_enabled(0);
#endif

    auto spmm_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        run_spmm_kernel();
    }

#if defined(CUDA_ENABLED) && CUDA_ENABLED
    synchronize_cuda();
    set_cuda_sync_enabled(1);
#endif

    auto spmm_end = std::chrono::high_resolution_clock::now();
    auto spmm_duration = std::chrono::duration_cast<std::chrono::microseconds>(spmm_end - spmm_start).count();

    return std::make_pair(
        static_cast<double>(preprocess_duration),
        static_cast<double>(spmm_duration) / iterations
    );
}

bool SpMM_Benchmark::validate_correctness() {
    spmm_serial();

#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    memset_device(d_y, 0, y.size() * sizeof(BenchmarkValue));
#endif
    run_spmm_kernel();

#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    copy_device_to_host(y.data(), d_y, y.size() * sizeof(BenchmarkValue));
    free_memory();
#endif

    double diff_norm = 0.0;
    double ref_norm = 0.0;
    for (int i = 0; i < nrows * dense_cols; ++i) {
        double diff = y[i] - reference_y[i];
        diff_norm += diff * diff;
        ref_norm += reference_y[i] * reference_y[i];
    }

    double relative_error = ref_norm > 0.0 ? std::sqrt(diff_norm) / std::sqrt(ref_norm) : std::sqrt(diff_norm);
    const double machine_epsilon = std::numeric_limits<BenchmarkValue>::epsilon();
    const double tolerance = 1e6 * machine_epsilon;

    std::cout << "Reference matrix norm: " << std::sqrt(ref_norm) << std::endl;
    std::cout << "Difference matrix norm: " << std::sqrt(diff_norm) << std::endl;
    std::cout << "Relative error: " << relative_error << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;

    bool passed = relative_error <= tolerance;
    std::cout << "Correctness validation: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

double SpMM_Benchmark::calculate_performance(double spmm_time_us) {
    if (spmm_time_us <= 0.0) {
        return 0.0;
    }
    long long total_ops = static_cast<long long>(nnz) * dense_cols * 2;
    double time_seconds = spmm_time_us / 1e6;
    return (total_ops / time_seconds) / 1e9;
}

void SpMM_Benchmark::print_matrix_info() {
    std::cout << "Matrix Name: " << input_filename << std::endl;
    std::cout << "Matrix Size: " << nrows << "x" << ncols << std::endl;
    std::cout << "Operator: SpMM" << std::endl;
    std::cout << "Dense RHS columns: " << dense_cols << " (row-major)" << std::endl;
    std::cout << "Non-zeros per row: " << average_nnz_per_row() << std::endl;
    std::cout << "Total non-zeros: " << nnz << std::endl;
    std::cout << "Sparsity: " << sparsity() << std::endl;
}

void SpMM_Benchmark::set_report_file(const std::string& reportfile) {
    if (reportfile.empty()) {
        generate_spmm_report_filename();
    }
    else {
        report_filename = reportfile;
    }
}

void SpMM_Benchmark::write_report(std::pair<double, double> timing_results, double perf_gflops, bool correct) {
    std::ofstream report_file(report_filename, std::ios::app);
    if (!report_file.is_open()) {
        std::cerr << "Error: Cannot create report file " << report_filename << std::endl;
        return;
    }

    report_file << "SpMM Benchmark Report\n";
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
    report_file << "  Operator: SpMM\n";
    report_file << "  Kernel: " << kernel_name << "\n";
    report_file << "  Precision: " << benchmark_precision_name() << "\n";
    report_file << "  Dense RHS columns: " << dense_cols << "\n";
    report_file << "  Dense RHS layout: row-major\n";
    report_file << "  Preprocessing time: " << timing_results.first << " microseconds\n";
    report_file << "  Average SpMM execution time: " << timing_results.second << " microseconds\n";
    report_file << "  Performance: " << perf_gflops << " GFLOPS\n\n";

    long long total_ops = static_cast<long long>(nnz) * dense_cols * 2;
    report_file << "Additional Metrics:\n";
    report_file << "  Total operations: " << total_ops << " (multiply-adds)\n";
    report_file << "  Memory accessed (approx): "
                << (((ncols + nrows) * dense_cols) * sizeof(BenchmarkValue) + nnz * (sizeof(BenchmarkValue) + sizeof(int)))
                << " bytes\n\n";

    report_file << "Correctness Validation:\n";
    report_file << "  Result: " << (correct ? "PASSED" : "FAILED") << "\n\n";

    report_file.close();
    std::cout << "Report saved to: " << report_filename << std::endl;
}
