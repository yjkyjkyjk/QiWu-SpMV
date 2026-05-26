#include "sparse_benchmark_base.h"
#include "benchmark_device_interface.h"

using namespace std;

SparseBenchmarkBase::SparseBenchmarkBase(int nrows, int ncols, int nnz)
    : nrows(nrows), ncols(ncols), nnz(nnz) {
    initialize_matrix();
}

SparseBenchmarkBase::SparseBenchmarkBase(const std::string& mtx_file) {
    set_input_filename(mtx_file);
    load_matrix_from_mtx(mtx_file);
}

SparseBenchmarkBase::~SparseBenchmarkBase() {
    free_memory();
}

void SparseBenchmarkBase::set_input_filename(const std::string& path) {
    size_t lastSlashPos = path.find_last_of("/\\");

    if (path.empty()) {
        input_filename = "";
    }
    else if (lastSlashPos == string::npos) {
        input_filename = path;
    }
    else if (lastSlashPos == path.length() - 1) {
        input_filename = "";
    }
    else {
        input_filename = path.substr(lastSlashPos + 1);
    }
}

void SparseBenchmarkBase::load_matrix_from_mtx(const std::string& filename) {
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

    while (std::getline(file, line)) {
        if (line.substr(0, 2) == "%%") {
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            iss >> object;
            iss >> format;
            iss >> field;
            iss >> symmetry;

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

    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '%') {
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

    std::vector<int> coo_rows, coo_cols;
    std::vector<BenchmarkValue> coo_vals;

    if (is_vector) {
        std::cerr << "Vector object not yet supported." << std::endl;
        exit(1);
    }
    if (is_array) {
        std::cerr << "Array format not yet supported." << std::endl;
        exit(1);
    }
    if (is_complex) {
        std::cerr << "Warning: Complex field detected. Imaginary part will be ignored." << std::endl;
    }

    if (is_matrix && is_coordinate) {
        for (int i = 0; i < total_nnz; ++i) {
            if (!std::getline(file, line)) break;
            if (line.empty()) continue;

            std::istringstream iss(line);
            int row, col;
            double real_val = 0.0, imag_val = 0.0;

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
                real_val = 1.0;
            }
            else if (is_complex) {
                iss >> row >> col >> real_val >> imag_val;
            }

            row--;
            col--;

            if (is_general) {
                coo_rows.push_back(row);
                coo_cols.push_back(col);
                coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
            }
            else if (is_symmetric) {
                if (row >= col) {
                    coo_rows.push_back(row);
                    coo_cols.push_back(col);
                    coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
                    if (row != col) {
                        coo_rows.push_back(col);
                        coo_cols.push_back(row);
                        coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
                    }
                }
                else {
                    std::cerr << "Error: symmetric matrix contains upper triangle entry ("
                              << row + 1 << ", " << col + 1
                              << "). Only lower triangle should be stored." << std::endl;
                    exit(1);
                }
            }
            else if (is_skew_symmetric) {
                if (row >= col) {
                    if (row == col) {
                        real_val = 0.0;
                    }
                    coo_rows.push_back(row);
                    coo_cols.push_back(col);
                    coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
                    if (row != col) {
                        coo_rows.push_back(col);
                        coo_cols.push_back(row);
                        coo_vals.push_back(static_cast<BenchmarkValue>(-real_val));
                    }
                }
                else {
                    std::cerr << "Error: skew-symmetric matrix contains upper triangle entry ("
                              << row + 1 << ", " << col + 1
                              << "). Only lower triangle should be stored." << std::endl;
                    exit(1);
                }
            }
            else if (is_hermitian) {
                if (row >= col) {
                    if (row == col) {
                        imag_val = 0.0;
                    }
                    coo_rows.push_back(row);
                    coo_cols.push_back(col);
                    coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
                    if (row != col) {
                        coo_rows.push_back(col);
                        coo_cols.push_back(row);
                        coo_vals.push_back(static_cast<BenchmarkValue>(real_val));
                    }
                }
                else {
                    std::cerr << "Error: Hermitian matrix contains upper triangle entry ("
                              << row + 1 << ", " << col + 1
                              << "). Only lower triangle should be stored." << std::endl;
                    exit(1);
                }
            }
        }
    }

    int expanded_nnz = static_cast<int>(coo_rows.size());
    convert_coo_to_csr(coo_rows, coo_cols, coo_vals, nrows, expanded_nnz);
}

void SparseBenchmarkBase::convert_coo_to_csr(const std::vector<int>& coo_rows,
                                             const std::vector<int>& coo_cols,
                                             const std::vector<BenchmarkValue>& coo_vals,
                                             int nrows,
                                             int nnz) {
    this->nrows = nrows;
    this->nnz = nnz;

    values.resize(nnz);
    col_idx.resize(nnz);
    row_ptr.resize(nrows + 1, 0);

    for (int i = 0; i < nnz; i++) {
        row_ptr[coo_rows[i] + 1]++;
    }

    for (int i = 1; i <= nrows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }

    std::vector<int> next_pos = row_ptr;
    for (int i = 0; i < nnz; i++) {
        int row = coo_rows[i];
        int pos = next_pos[row]++;
        values[pos] = coo_vals[i];
        col_idx[pos] = coo_cols[i];
    }
}

void SparseBenchmarkBase::initialize_matrix() {
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

        int nnz_this_row = avg_nnz_per_row;
        if (i < remainder) {
            nnz_this_row++;
        }

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

void SparseBenchmarkBase::allocate_memory() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
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

void SparseBenchmarkBase::free_memory() {
#if defined(CUDA_ENABLED) && CUDA_ENABLED || defined(HIP_ENABLED) && HIP_ENABLED
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

double SparseBenchmarkBase::sparsity() const {
    if (nrows <= 0 || ncols <= 0) {
        return 0.0;
    }
    return 1.0 - static_cast<double>(nnz) / static_cast<double>(nrows) / static_cast<double>(ncols);
}

int SparseBenchmarkBase::average_nnz_per_row() const {
    return (nrows > 0) ? nnz / nrows : 0;
}

void SparseBenchmarkBase::set_kernel_name(const std::string& kernelname) {
    kernel_name = kernelname;
}
