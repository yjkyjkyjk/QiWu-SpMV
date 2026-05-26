# SpMV and SpMM Benchmark Toolkit

This project benchmarks sparse matrix kernels on Matrix Market (`.mtx`) inputs. It supports:

- CSR SpMV: `y = A * x`
- CSR SpMM: `C = A * B`
- CPU-only builds, plus CUDA/HIP builds when the toolchains are available
- `double` precision by default and optional `float` precision
- Automated multi-matrix runs and CSV report extraction

## Requirements

- CMake 3.18 or newer
- A C++17 compiler with OpenMP
- Optional: CUDA `nvcc`
- Optional: HIP `hipcc`
- Python 3.6 or newer for automation scripts

## Build

```bash
mkdir build
cd build
cmake ..
make all
```

By default the benchmark uses `double` values. Build a float-precision binary with:

```bash
cmake .. -DSPARSE_VALUE_TYPE=float
make all
```

Valid `SPARSE_VALUE_TYPE` values are `double` and `float`.

The executable targets are named after the full sparse benchmark suite:

- `sparseBenchmark_cpu`
- `sparseBenchmark_cuda`
- `sparseBenchmark_hip`
- `sparseBenchmark_unified`

## Single-Matrix Usage

Run SpMV:

```bash
./build/sparseBenchmark_cpu path/to/matrix.mtx my_kernel report.txt spmv
```

Run SpMM:

```bash
./build/sparseBenchmark_cpu path/to/matrix.mtx my_kernel report.txt spmm
```

SpMM uses a row-major dense right-hand-side matrix by default:

- Shape: `ncols(A) x 8`
- Layout: row-major

Override the dense RHS column count with the optional sixth argument:

```bash
./build/sparseBenchmark_cpu path/to/matrix.mtx my_kernel report.txt spmm 16
```

The executable arguments are:

```text
<matrix.mtx> <kernel_name> <report_file> [spmv|spmm] [spmm_cols]
```

`spmv` remains the default operator for backward compatibility. `spmm_cols` defaults to `8` and is only used by SpMM.

## Automated Runs

Run one executable over every `.mtx` file in a directory:

```bash
python auto-sparsebenchmark.py path_to_mtx_folder kernel_name sparseBenchmark_cpu
```

Run SpMM over every matrix using the default 8 RHS columns:

```bash
python auto-sparsebenchmark.py path_to_mtx_folder kernel_name sparseBenchmark_cpu --op spmm
```

Run SpMM with another RHS width:

```bash
python auto-sparsebenchmark.py path_to_mtx_folder kernel_name sparseBenchmark_cpu --op spmm --spmm-cols 16
```

The script looks for executables under `build/`.

## Parse Reports

Convert a text report to CSV:

```bash
python data-process.py spmv-test-YYYYMMDDHHMMSS
python data-process.py spmm-test-YYYYMMDDHHMMSS
```

When the output CSV is not specified, the parser names it as:

```text
<kernel>_<operator>_results.csv
```

## Report and CSV Fields

The report parser extracts:

- Matrix metadata: name, rows, columns, nonzeros per row, total nonzeros, sparsity
- Kernel metadata: kernel name, operator (`SpMV` or `SpMM`), precision, dense RHS columns, dense RHS layout
- Timings: preprocessing time, average execution time
- Performance: GFLOPS, total operations, approximate memory accessed
- Derived metrics: execution time in ms, memory bandwidth, compute intensity
- Validation result and test date

For SpMV, total operations are `2 * nnz`.

For SpMM, total operations are `2 * nnz * spmm_cols`.

Approximate memory access includes CSR values/column indices plus dense input/output vectors or matrices.

## GPU Architecture Notes

To support different NVIDIA GPU architectures, edit `CUDA_ARCHITECTURES` in `CMakeLists.txt`.

Common values:

- `70`: V100
- `75`: T4 / RTX 20 series
- `80`: A100
- `86`: RTX 30 series

Example for Pascal and newer GPUs:

```cmake
CUDA_ARCHITECTURES "60;61;70;75;80;86"
```

## Custom Kernels

The source tree separates shared code, the CLI entry point, and each operator:

```text
src/
  app/       benchmark CLI entry point
  common/    backend device-memory helpers
  spmv/      SpMV benchmark and SpMV kernel implementations
  spmm/      SpMM benchmark and SpMM kernel implementations
```

Start from the operator-specific files:

- SpMV CPU: `src/spmv/opt.cpp`
- SpMV CUDA: `src/spmv/opt_cuda.cu`
- SpMV HIP: `src/spmv/opt_hip.hip`
- SpMM CPU: `src/spmm/opt.cpp`
- SpMM CUDA: `src/spmm/opt_cuda.cu`
- SpMM HIP: `src/spmm/opt_hip.hip`

The benchmark stores SpMM dense matrices in row-major order:

```text
x[col * dense_cols + k]
y[row * dense_cols + k]
```

Use `BenchmarkValue` for scalar values so your implementation works with both `double` and `float` builds.
