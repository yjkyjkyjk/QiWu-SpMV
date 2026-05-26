import subprocess
import os
import sys
from datetime import datetime
import argparse

def run_cpp_benchmark(folder_path, user_string, executable_name, operator_name="spmv", dense_cols=8):
    # 1. Check path valid
    if not os.path.isdir(folder_path):
        print(f"Error: path '{folder_path}' is not a valid directory", file=sys.stderr)
        return False

    # 2. Get all .mtx files
    mtx_files = [f for f in os.listdir(folder_path) if f.endswith('.mtx')]
    if not mtx_files:
        print("No .mtx files found in the specified directory.", file=sys.stderr)
        return False

    # 3. Construct full executable path
    cpp_executable = os.path.join("build", executable_name)
    if not os.path.isfile(cpp_executable):
        print(f"Error: executable '{cpp_executable}' not found", file=sys.stderr)
        return False

    # Generate report name with timestamp
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    report_name = f"{operator_name}-test-{current_time}"

    print(f"Running benchmark using: {cpp_executable}")
    print(f"Output report prefix: {report_name}")
    print("-" * 50)

    success_count = 0
    for mtx in mtx_files:
        mtx_path = os.path.join(folder_path, mtx)
        command = [cpp_executable, mtx_path, user_string, report_name, operator_name, str(dense_cols)]
        print(f"Processing: {mtx} ...")
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f" ❌ Failed on {mtx}: {e}", file=sys.stderr)
            # Uncomment below to see stderr output
            # print(f"   stderr: {e.stderr}", file=sys.stderr)

    print("-" * 50)
    print(f"✅ Completed: {success_count}/{len(mtx_files)} matrices processed.")
    return True

def main():
    parser = argparse.ArgumentParser(
        prog='auto-spmvbenchmark.py',
        description='Automatically run SpMV or SpMM benchmark on all .mtx files in a directory.',
        epilog='Example: python auto-spmvbenchmark.py ./matrices_path unroll_cpu spmvBenchmark_cpu --op spmm'
    )

    parser.add_argument(
        'mtx_dir',
        help='Path to directory containing .mtx matrix files'
    )
    parser.add_argument(
        'kernel_name',
        help='Name for the SpMV implementation'
    )
    parser.add_argument(
        'executable',
        nargs='?',
        default='spmvBenchmark',
        help='Name of the compiled executable (default: spmvBenchmark). '
             'Common options: spmvBenchmark_cpu, spmvBenchmark_cuda, spmvBenchmark_unified'
    )
    parser.add_argument(
        '--op',
        choices=['spmv', 'spmm'],
        default='spmv',
        help='Operator to benchmark (default: spmv). SpMM uses a row-major dense RHS.'
    )
    parser.add_argument(
        '--spmm-cols',
        type=int,
        default=8,
        help='Dense RHS column count for SpMM (default: 8). Ignored by SpMV.'
    )

    args = parser.parse_args()

    success = run_cpp_benchmark(args.mtx_dir, args.kernel_name, args.executable, args.op, max(1, args.spmm_cols))
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
