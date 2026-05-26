#include "spmv_benchmark.h"
#include "benchmark_device_interface.h"
#include <string>

using namespace std;

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

SpMV_Benchmark::SpMV_Benchmark(int nrows, int ncols, int nnz) : nrows(nrows), ncols(ncols), nnz(nnz) {
    initialize_matrix();
    initialize_vectors();
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    allocate_memory();
#endif
}

SpMV_Benchmark::SpMV_Benchmark(const std::string& mtx_file) {
    size_t lastSlashPos = mtx_file.find_last_of("/\\");

    if (mtx_file.empty()) {
        input_filename = "";
    }
    else if (lastSlashPos == string::npos) {
        input_filename = mtx_file;
    }
    else if (lastSlashPos == mtx_file.length() - 1) {
        input_filename = "";
    }
    else {
        input_filename = mtx_file.substr(lastSlashPos + 1);
    }

    // generate_report_filename();
    load_matrix_from_mtx(mtx_file);
    initialize_vectors();
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    allocate_memory();
#endif
}

SpMV_Benchmark::~SpMV_Benchmark() {
    free_memory();
}

void SpMV_Benchmark::load_matrix_from_mtx(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    std::string object, format, field, symmetry;
    int num_rows = 0, num_cols = 0, total_nnz = 0;
    bool is_matrix = false, is_vector = false;
    bool is_coordinate = false, is_array = false;
    bool is_real = false, is_complex = false, is_integer = false, is_pattern = false;
    bool is_general = false, is_symmetric = false, is_skew_symmetric = false, is_hermitian = false;

    // Read header line
    while (std::getline(file, line)) {
        if (line.substr(0, 2) == "%%") {
            // Parse Matrix Market header
            std::istringstream iss(line);
            std::string token;
            iss >> token; // %%MatrixMarket
            iss >> object; // matrix or vector
            iss >> format; // coordinate or array
            iss >> field; // real, integer, complex, pattern
            iss >> symmetry; // general, symmetric, skew-symmetric, Hermitian

            // Set flags
            is_matrix = (object == "matrix");
            is_vector = (object == "vector");
            is_coordinate = (format == "coordinate");
            is_array = (format == "array");
            is_real = (field == "real");
            is_integer = (field == "integer");
            is_complex = (field == "complex");
            is_pattern = (field == "pattern");
            is_general = (symmetry == "general");
            is_symmetric = (symmetry == "symmetric");
            is_skew_symmetric = (symmetry == "skew-symmetric");
            is_hermitian = (symmetry == "Hermitian" || symmetry == "hermitian");

            break;
        }
    }

    // Skip comments
    while (std::getline(file, line)) {
        if (line[0] != '%') {
            // This is the size line
            std::istringstream iss(line);
            if (is_coordinate) {
                iss >> num_rows >> num_cols >> total_nnz;
            }
            else if (is_array) {
                iss >> num_rows >> num_cols;
                total_nnz = num_rows * num_cols;
            }
            nrows = num_rows;
            ncols = num_cols;
            nnz = total_nnz;
            break;
        }
    }

    // Prepare data structures
    std::vector<int> coo_rows, coo_cols;
    std::vector<BenchmarkValue> coo_vals;

    if (is_vector) {
        // Vector object
        std::cerr << "Vector object not yet supported." << std::endl;
        exit(1);
    }
    if (is_array) {
        // Array format (dense matrix)
        std::cerr << "Array format not yet supported." << std::endl;
        exit(1);
    }
    if (is_complex) {
        // Warning: complex storage not fully implemented
        std::cerr << "Warning: Complex field detected. Imaginary part will be ignored." << std::endl;
    }

    if (is_matrix && is_coordinate) {
        // Read coordinate format data
        for (int i = 0; i < total_nnz; ++i) {
            if (!std::getline(file, line)) break;
            if (line.empty()) continue;

            std::istringstream iss(line);
            int row, col;
            double real_val = 0.0, imag_val = 0.0;

            // Determine field type and read values accordingly
            if (is_real) {
                iss >> row >> col >> real_val;
            }
            else if (is_integer) {
                int int_val;
                iss >> row >> col >> int_val;
                real_val = static_cast<double>(int_val);
            }
            else if (is_pattern) {
                iss >> row >> col;
                real_val = 1.0; // Default value for pattern
            }
            else if (is_complex) {
                iss >> row >> col >> real_val >> imag_val;
            }

            // Convert to 0-based indexing
            row--;
            col--;

            // Nested if statements for symmetry handling
            if (is_general){
                // General matrix
                coo_rows.push_back(row);
                coo_cols.push_back(col);
                coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
            }
            else if (is_symmetric) {
                // Only lower triangle (including diagonal) is stored
                if (row >= col) {
                    coo_rows.push_back(row);
                    coo_cols.push_back(col);
                    coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
                    // Add symmetric element if not on diagonal
                    if (row != col) {
                        coo_rows.push_back(col);
                        coo_cols.push_back(row);
                        coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
                    }
                }
                else {
                    // Error: symmetric matrix should only store lower triangle
                    std::cerr << "Error: symmetric matrix contains upper triangle entry ("
                              << row+1 << ", " << col+1 << "). Only lower triangle should be stored." << std::endl;
                    exit(1);
                }
            }
            else if (is_skew_symmetric) {
                // Only lower triangle (including diagonal) is stored
                // Diagonal must be zero
                if (row >= col) {
                    if (row == col) {
                        // Diagonal: force zero
                        real_val = 0.0;
                    }
                    coo_rows.push_back(row);
                    coo_cols.push_back(col);
                    coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
                    // Add skew-symmetric element if not on diagonal
                    if (row != col) {
                        coo_rows.push_back(col);
                        coo_cols.push_back(row);
                        coo_vals.push_back(static_cast<BenchmarkValue>(-real_val));
                    }
                }
                else {
                    // Error: skew-symmetric matrix should only store lower triangle
                    std::cerr << "Error: skew-symmetric matrix contains upper triangle entry ("
                              << row+1 << ", " << col+1 << "). Only lower triangle should be stored." << std::endl;
                    exit(1);
                }
            }
            else if (is_hermitian) {
                // Only lower triangle (including diagonal) is stored
                // Diagonal must be real
                if (row >= col) {
                    if (row == col) {
                        // Diagonal: ensure real (ignore imag_val)
                        imag_val = 0.0;
                    }
                    coo_rows.push_back(row);
                    coo_cols.push_back(col);
                    coo_vals.push_back(static_cast<BenchmarkValue>(real_val)); // Store real part only
                    // Add Hermitian conjugate element if not on diagonal
                    if (row != col) {
                        coo_rows.push_back(col);
                        coo_cols.push_back(row);
                        // For Hermitian: A(j,i) = conj(A(i,j))
                        // Since we only store real part, use real_val
                        coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
                    }
                }
                else {
                    // Error: Hermitian matrix should only store lower triangle
                    std::cerr << "Error: Hermitian matrix contains upper triangle entry ("
                              << row+1 << ", " << col+1 << "). Only lower triangle should be stored." << std::endl;
                    exit(1);
                }
            }
        }
    }

    // Convert COO to CSR format
    int expanded_nnz = coo_rows.size();
    convert_coo_to_csr(coo_rows, coo_cols, coo_vals, nrows, expanded_nnz);

    file.close();
}

void SpMV_Benchmark::convert_coo_to_csr(const std::vector<int>& coo_rows,
                                       const std::vector<int>& coo_cols,
                                       const std::vector<BenchmarkValue>& coo_vals,
                                       int nrows, int nnz) {
    this->nrows = nrows;
    this->nnz = nnz;
    
    // Initialize CSR structures
    values.resize(nnz);
    col_idx.resize(nnz);
    row_ptr.resize(nrows + 1, 0);
    
    // Count number of elements in each row
    for (int i = 0; i < nnz; i++) {
        row_ptr[coo_rows[i] + 1]++;
    }
    
    // Compute prefix sum to get row pointers
    for (int i = 1; i <= nrows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }
    
    // Fill CSR arrays. Keep an independent cursor per row so entries in the
    // same row do not overwrite row_ptr[row].
    std::vector<int> next_pos = row_ptr;
    for (int i = 0; i < nnz; i++) {
        int row = coo_rows[i];
        int pos = next_pos[row]++;
        values[pos] = coo_vals[i];
        col_idx[pos] = coo_cols[i];
    }
    
}

void SpMV_Benchmark::allocate_memory() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    // Use backend-agnostic device helpers instead of calling CUDA/HIP directly.
    d_values = static_cast<BenchmarkValue*>(allocate_device_memory(values.size() * sizeof(BenchmarkValue)));
    d_col_idx = static_cast<int*>(allocate_device_memory(col_idx.size() * sizeof(int)));
    d_row_ptr = static_cast<int*>(allocate_device_memory(row_ptr.size() * sizeof(int)));
    d_x = static_cast<BenchmarkValue*>(allocate_device_memory(x.size() * sizeof(BenchmarkValue)));
    d_y = static_cast<BenchmarkValue*>(allocate_device_memory(y.size() * sizeof(BenchmarkValue)));

    copy_host_to_device(d_values, values.data(), values.size() * sizeof(BenchmarkValue));
    copy_host_to_device(d_col_idx, col_idx.data(), col_idx.size() * sizeof(int));
    copy_host_to_device(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int));
    copy_host_to_device(d_x, x.data(), x.size() * sizeof(BenchmarkValue));
    memset_device(d_y, 0, y.size() * sizeof(BenchmarkValue));
#endif
}

void SpMV_Benchmark::free_memory() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    // Free device memory
    if (d_values) {
        free_device_memory(d_values);
        d_values = nullptr;
    }
    if (d_col_idx) {
        free_device_memory(d_col_idx);
        d_col_idx = nullptr;
    }
    if (d_row_ptr) {
        free_device_memory(d_row_ptr);
        d_row_ptr = nullptr;
    }
    if (d_x) {
        free_device_memory(d_x);
        d_x = nullptr;
    }
    if (d_y) {
        free_device_memory(d_y);
        d_y = nullptr;
    }
#endif
}

void SpMV_Benchmark::initialize_matrix() {
    // Create sparse matrix - CSR format
    row_ptr.resize(nrows + 1);
    int total_nnz = nnz;
    values.resize(total_nnz);
    col_idx.resize(total_nnz);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> val_dis(-1.0, 1.0);
    std::uniform_int_distribution<> col_dis(0, ncols - 1);

    int nnz_count = 0;
    int avg_nnz_per_row = (nrows > 0) ? nnz / nrows : 0;
    int remainder = (nrows > 0) ? nnz % nrows : 0;

    for (int i = 0; i < nrows; ++i) {
        row_ptr[i] = nnz_count;

        // Calculate number of non-zeros for this row
        int nnz_this_row = avg_nnz_per_row;
        if (i < remainder) {
            nnz_this_row++;
        }

        // Generate unique column indices for each row
        std::vector<int> temp_cols;
        for (int j = 0; j < nnz_this_row; ++j) {
            int col;
            do {
                col = col_dis(gen);
            } while (std::find(temp_cols.begin(), temp_cols.end(), col) != temp_cols.end());

            temp_cols.push_back(col);
            col_idx[nnz_count] = col;
            values[nnz_count] = static_cast<BenchmarkValue>(val_dis(gen));
            nnz_count++;
        }
    }
    row_ptr[nrows] = nnz_count;
}

void SpMV_Benchmark::initialize_vectors() {
    x.resize(ncols);
    y.resize(nrows);
    reference_y.resize(nrows);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // Initialize input vector x (size ncols)
    for (int i = 0; i < ncols; ++i) {
        x[i] = static_cast<BenchmarkValue>(dis(gen));
    }

    // Initialize output vector y and reference vector (size nrows)
    for (int i = 0; i < nrows; ++i) {
        y[i] = static_cast<BenchmarkValue>(0);
        reference_y[i] = static_cast<BenchmarkValue>(0);
    }
}

void SpMV_Benchmark::spmv_serial() {
    // Single-threaded CSR format SpMV calculation (for reference)
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
    // Timing preprocessing stage
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    spmv_preprocess();
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count();

    // Warm up cache before timing
    warm_up_cache(10);

#if defined(CUDA_ENABLED) && CUDA_ENABLED
    synchronize_cuda();
    set_cuda_sync_enabled(0);
#endif

    // Timing SpMV execution stage
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

    // Return preprocessing time and SpMV execution time (average per iteration)
    return std::make_pair(
        static_cast<double>(preprocess_duration),
        static_cast<double>(spmv_duration) / iterations
    );
}

bool SpMV_Benchmark::validate_correctness() {
    // Using single-threaded result as reference to verify current result's correctness
    spmv_serial();
    
    // Re-execute optimized version to get result to be validated
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    memset_device(d_y, 0, y.size() * sizeof(BenchmarkValue));
#endif
    run_spmv_kernel();

#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
    // Copy result back to host
    copy_device_to_host(y.data(), d_y, y.size() * sizeof(BenchmarkValue));
    free_memory();
#endif
    // Calculate L2 norm relative error
    double diff_norm = 0.0;  // Square of difference vector L2 norm
    double ref_norm = 0.0;   // Square of reference vector L2 norm
    for (int i = 0; i < nrows; ++i) {
        double diff = y[i] - reference_y[i];
        diff_norm += diff * diff;
        ref_norm += reference_y[i] * reference_y[i];
    }
    
    double relative_error = ref_norm > 0.0 ? sqrt(diff_norm) / sqrt(ref_norm) : sqrt(diff_norm);
    
    // Double precision error requirement: relative error should be less than a multiple of machine precision
    const double machine_epsilon = std::numeric_limits<BenchmarkValue>::epsilon();
    const double tolerance = 1e6 * machine_epsilon; // Allow million times machine precision error
    
    std::cout << "Reference vector norm: " << sqrt(ref_norm) << std::endl;
    std::cout << "Difference vector norm: " << sqrt(diff_norm) << std::endl;
    std::cout << "Relative error: " << relative_error << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;
    
    bool passed = relative_error <= tolerance;
    std::cout << "Correctness validation: " << (passed ? "PASSED" : "FAILED") << std::endl;
    
    return passed;
}

double SpMV_Benchmark::calculate_performance(double spmv_time_us) {
    // Calculate performance metric: GFLOPS
    long long total_ops = static_cast<long long>(nnz) * 2;  // One multiplication and one addition per non-zero element
    double time_seconds = spmv_time_us / 1e6;
    double gflops = (total_ops / time_seconds) / 1e9;
    return gflops;
}

void SpMV_Benchmark::print_matrix_info() {
    int total_nnz = nnz;
    int avg_nnz_per_row = (nrows > 0) ? nnz / nrows : 0;
    std::cout << "Matrix Name: " << input_filename << std::endl;
    std::cout << "Matrix Size: " << nrows << "x" << ncols << std::endl;
    std::cout << "Non-zeros per row: " << avg_nnz_per_row << std::endl;
    std::cout << "Total non-zeros: " << total_nnz << std::endl;
    double sparsity = 0.0;
    if (nrows > 0 && ncols > 0) {
        sparsity = 1.0 - (double)total_nnz / (double)nrows / (double)ncols;
    }
    std::cout << "Sparsity: " << sparsity << std::endl;
}

void SpMV_Benchmark::set_kernel_name(const std::string& kernelname) {
    kernel_name = kernelname;
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
    std::ofstream report_file(report_filename, ios::app);
    if (!report_file.is_open()) {
        std::cerr << "Error: Cannot create report file " << report_filename << std::endl;
        return;
    }

    int total_nnz = nnz;
    int avg_nnz_per_row = (nrows > 0) ? nnz / nrows : 0;

    report_file << "SpMV Benchmark Report\n";
    report_file << "=====================\n";
    std::time_t now = std::time(nullptr);
    report_file << "Date: " << std::asctime(std::localtime(&now)) << "\n";
    report_file << "Matrix Information:\n";
    report_file << "  Name: " << input_filename << "\n";
    report_file << "  Size: " << nrows << "x" << ncols << "\n";
    report_file << "  Non-zeros per row: " << avg_nnz_per_row << "\n";
    report_file << "  Total non-zeros: " << total_nnz << "\n";
    double sparsity = 0.0;
    if (nrows > 0 && ncols > 0) {
        sparsity = 1.0 - (double)total_nnz / (double)nrows / (double)ncols;
    }
    report_file << "  Sparsity: " << sparsity << "\n\n";
    
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
    report_file << "  Memory accessed (approx): " <<
               ((ncols + nrows) * sizeof(BenchmarkValue) + nnz * (sizeof(BenchmarkValue) + sizeof(int))) << " bytes\n\n";
    
    report_file << "Correctness Validation:\n";
    report_file << "  Result: " << (correct ? "PASSED" : "FAILED") << "\n\n";
    
    report_file.close();
    std::cout << "Report saved to: " << report_filename << std::endl;
}
