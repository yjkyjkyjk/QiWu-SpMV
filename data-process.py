"""
SpMV/SpMM Benchmark Report Parser
Extracts performance metrics from benchmark reports and saves to CSV format.
Author: [Your Name]
Date: [Current Date]
"""

import re
import csv
import sys
import os
from collections import defaultdict

def parse_spmv_report(file_path, output_csv=None):
    """
    Parse SpMV benchmark report file and extract data to CSV.
    
    Args:
        file_path (str): Path to input report file
        output_csv (str): Path to output CSV file (default: auto-generated)
    
    Returns:
        tuple: (output_file_path, number_of_reports) or (None, -1) on error
    """
    
    # Regular expression patterns for extracting data
    patterns = {
        'name': r'Name:\s*(.+)',
        'size': r'Size:\s*(\d+)x(\d+)',
        'nnz_per_row': r'Non-zeros per row:\s*([\d.]+)',
        'total_nnz': r'Total non-zeros:\s*([\d.]+)',
        'sparsity': r'Sparsity:\s*([\d.]+)',
        'kernel': r'Kernel:\s*(.+)',
        'operator': r'Operator:\s*(SpMV|SpMM)',
        'precision': r'Precision:\s*(.+)',
        'dense_cols': r'Dense RHS columns:\s*(\d+)',
        'dense_layout': r'Dense RHS layout:\s*(.+)',
        'preprocess_time': r'Preprocessing time:\s*([\d.]+)\s*microseconds',
        'execution_time': r'Average (?:SpMV|SpMM) execution time:\s*([\d.]+)\s*microseconds',
        'performance': r'Performance:\s*([\d.]+)\s*GFLOPS',
        'operations': r'Total operations:\s*([\d.]+)\s*\(multiply-adds\)',
        'memory_accessed': r'Memory accessed \(approx\):\s*([\d.]+)\s*bytes',
        'validation': r'Result:\s*(PASSED|FAILED)',
        'date': r'Date:\s*(.+)'
    }
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return None, -1
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        return None, -1
    
    # Split content into individual reports
    reports = re.split(r'(?:SpMV|SpMM) Benchmark Report\s*=+', content)
    
    # Remove empty elements
    reports = [r.strip() for r in reports if r.strip()]
    
    if not reports:
        print("WARNING: No SpMV benchmark reports found in file")
        return None, 0
    
    print(f"INFO: Found {len(reports)} benchmark reports")
    
    # Data container for all extracted metrics
    data_rows = []
    kernel_counts = defaultdict(int)
    
    # Process each report
    for i, report in enumerate(reports, 1):
        # Initialize data dictionary for current report
        row_data = {
            'matrix_name': '',
            'size_rows': 0,
            'size_cols': 0,
            'nnz_per_row': 0.0,
            'total_nnz': 0,
            'sparsity': 0.0,
            'kernel': 'default',
            'operator': 'SpMV',
            'precision': '',
            'dense_cols': 1,
            'dense_layout': '',
            'preprocess_time_us': 0.0,
            'execution_time_us': 0.0,
            'performance_gflops': 0.0,
            'total_operations': 0,
            'memory_accessed_bytes': 0,
            'validation_result': '',
            'test_date': ''
        }
        
        # Extract matrix name
        name_match = re.search(patterns['name'], report)
        if name_match:
            row_data['matrix_name'] = name_match.group(1)
        
        # Extract matrix dimensions
        size_match = re.search(patterns['size'], report)
        if size_match:
            row_data['size_rows'] = int(size_match.group(1))
            row_data['size_cols'] = int(size_match.group(2))
        
        # Extract non-zeros per row
        nnz_per_row_match = re.search(patterns['nnz_per_row'], report)
        if nnz_per_row_match:
            row_data['nnz_per_row'] = float(nnz_per_row_match.group(1))
        
        # Extract total non-zeros
        total_nnz_match = re.search(patterns['total_nnz'], report)
        if total_nnz_match:
            row_data['total_nnz'] = int(float(total_nnz_match.group(1)))
        
        # Extract sparsity
        sparsity_match = re.search(patterns['sparsity'], report)
        if sparsity_match:
            row_data['sparsity'] = float(sparsity_match.group(1))
        
        # Extract kernel information
        operator_match = re.search(patterns['operator'], report)
        if operator_match:
            row_data['operator'] = operator_match.group(1).strip()

        kernel_match = re.search(patterns['kernel'], report)
        if kernel_match:
            kernel_name = kernel_match.group(1).strip()
            row_data['kernel'] = kernel_name
            kernel_counts[kernel_name] += 1
        
        # Extract precision and dense RHS metadata
        precision_match = re.search(patterns['precision'], report)
        if precision_match:
            row_data['precision'] = precision_match.group(1).strip()

        dense_cols_match = re.search(patterns['dense_cols'], report)
        if dense_cols_match:
            row_data['dense_cols'] = int(dense_cols_match.group(1))

        dense_layout_match = re.search(patterns['dense_layout'], report)
        if dense_layout_match:
            row_data['dense_layout'] = dense_layout_match.group(1).strip()

        # Extract preprocessing time
        preprocess_match = re.search(patterns['preprocess_time'], report)
        if preprocess_match:
            row_data['preprocess_time_us'] = float(preprocess_match.group(1))
        
        # Extract execution time
        execution_match = re.search(patterns['execution_time'], report)
        if execution_match:
            row_data['execution_time_us'] = float(execution_match.group(1))
        
        # Extract performance (GFLOPS)
        performance_match = re.search(patterns['performance'], report)
        if performance_match:
            row_data['performance_gflops'] = float(performance_match.group(1))
        
        # Extract total operations
        operations_match = re.search(patterns['operations'], report)
        if operations_match:
            row_data['total_operations'] = int(float(operations_match.group(1)))
        
        # Extract memory accessed
        memory_match = re.search(patterns['memory_accessed'], report)
        if memory_match:
            row_data['memory_accessed_bytes'] = int(float(memory_match.group(1)))
        
        # Extract validation result
        validation_match = re.search(patterns['validation'], report)
        if validation_match:
            row_data['validation_result'] = validation_match.group(1)
        
        # Extract test date
        date_match = re.search(patterns['date'], report)
        if date_match:
            row_data['test_date'] = date_match.group(1)
        
        # Calculate derived metrics
        row_data['execution_time_ms'] = row_data['execution_time_us'] / 1000.0
        
        # Memory bandwidth calculation
        if row_data['execution_time_us'] > 0:
            memory_gb = row_data['memory_accessed_bytes'] / (1024**3)
            time_s = row_data['execution_time_us'] / 1_000_000
            row_data['memory_bandwidth_gbs'] = memory_gb / time_s if time_s > 0 else 0
        else:
            row_data['memory_bandwidth_gbs'] = 0
        
        # Computational intensity calculation
        if row_data['memory_accessed_bytes'] > 0:
            row_data['compute_intensity'] = row_data['total_operations'] / row_data['memory_accessed_bytes']
        else:
            row_data['compute_intensity'] = 0
        
        # Add to data collection
        data_rows.append(row_data)
        
        # Progress indicator
        if i % 10 == 0 or i == len(reports):
            print(f"INFO: Parsed {i}/{len(reports)}: {row_data['matrix_name']} (Kernel: {row_data['kernel']})")
    
    if not data_rows:
        print("WARNING: No valid data extracted")
        return None, 0
    
    # Determine output filename based on kernel information
    if output_csv is None:
        # Generate output filename based on kernels used
        if len(kernel_counts) == 1:
            # Single kernel: use format "{kernel}_spmv_results.csv"
            kernel_name = list(kernel_counts.keys())[0]
            operator_name = data_rows[0]['operator'].lower()
            output_csv = f"{kernel_name}_{operator_name}_results.csv"
        else:
            # Multiple kernels: create combined file
            kernel_str = "_".join(sorted(kernel_counts.keys()))
            # Limit length of filename
            if len(kernel_str) > 50:
                kernel_str = f"{len(kernel_counts)}_kernels"
            output_csv = f"combined_{kernel_str}_spmv_results.csv"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write data to CSV file
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            # Define column order for CSV
            fieldnames = [
                'matrix_name',
                'size_rows',
                'size_cols',
                'nnz_per_row',
                'total_nnz',
                'sparsity',
                'kernel',
                'operator',
                'precision',
                'dense_cols',
                'dense_layout',
                'preprocess_time_us',
                'execution_time_us',
                'execution_time_ms',
                'performance_gflops',
                'total_operations',
                'memory_accessed_bytes',
                'memory_bandwidth_gbs',
                'compute_intensity',
                'validation_result',
                'test_date'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_rows)
        
        print(f"SUCCESS: Data saved to {output_csv}")
        print(f"INFO: Processed {len(data_rows)} matrices")
        
        # Print summary statistics
        print_summary_statistics(data_rows, kernel_counts)
        
    except Exception as e:
        print(f"ERROR: Failed to write CSV file: {e}")
        return None, -1
    
    return output_csv, len(data_rows)


def print_summary_statistics(data_rows, kernel_counts):
    """Print summary statistics of parsed benchmark data."""
    
    if not data_rows:
        return
    
    # Calculate overall statistics
    passed_count = sum(1 for row in data_rows if row['validation_result'] == 'PASSED')
    failed_count = sum(1 for row in data_rows if row['validation_result'] == 'FAILED')
    
    # Collect all performance and execution time values
    perf_values = [row['performance_gflops'] for row in data_rows if row['performance_gflops'] > 0]
    exec_times = [row['execution_time_ms'] for row in data_rows if row['execution_time_ms'] > 0]
    
    if perf_values:
        min_perf = min(perf_values)
        max_perf = max(perf_values)
        avg_perf = sum(perf_values) / len(perf_values)
    else:
        min_perf = max_perf = avg_perf = 0
    
    if exec_times:
        min_time = min(exec_times)
        max_time = max(exec_times)
        avg_time = sum(exec_times) / len(exec_times)
    else:
        min_time = max_time = avg_time = 0
    
    # Print statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total matrices processed: {len(data_rows)}")
    print(f"Validation passed: {passed_count} ({passed_count/len(data_rows)*100:.1f}%)")
    print(f"Validation failed: {failed_count} ({failed_count/len(data_rows)*100:.1f}%)")
    print(f"Performance range: {min_perf:.4f} - {max_perf:.4f} GFLOPS")
    print(f"Average performance: {avg_perf:.4f} GFLOPS")
    print(f"Execution time range: {min_time:.2f} - {max_time:.2f} ms")
    print(f"Average execution time: {avg_time:.2f} ms")
    
    # Print kernel distribution
    print(f"\nKERNEL DISTRIBUTION:")
    print("-"*60)
    for kernel, count in sorted(kernel_counts.items()):
        percentage = (count / len(data_rows)) * 100
        print(f"  {kernel}: {count} matrices ({percentage:.1f}%)")
    
    # Print kernel-specific statistics
    print(f"\nKERNEL-SPECIFIC STATISTICS:")
    print("-"*60)
    
    # Group data by kernel
    kernel_data = defaultdict(list)
    for row in data_rows:
        kernel_data[row['kernel']].append(row)
    
    for kernel, rows in kernel_data.items():
        kernel_perf = [row['performance_gflops'] for row in rows if row['performance_gflops'] > 0]
        kernel_time = [row['execution_time_ms'] for row in rows if row['execution_time_ms'] > 0]
        kernel_passed = sum(1 for row in rows if row['validation_result'] == 'PASSED')
        kernel_failed = sum(1 for row in rows if row['validation_result'] == 'FAILED')
        
        if kernel_perf:
            kernel_min_perf = min(kernel_perf)
            kernel_max_perf = max(kernel_perf)
            kernel_avg_perf = sum(kernel_perf) / len(kernel_perf)
        else:
            kernel_min_perf = kernel_max_perf = kernel_avg_perf = 0
        
        if kernel_time:
            kernel_min_time = min(kernel_time)
            kernel_max_time = max(kernel_time)
            kernel_avg_time = sum(kernel_time) / len(kernel_time)
        else:
            kernel_min_time = kernel_max_time = kernel_avg_time = 0
        
        print(f"  Kernel: {kernel}")
        print(f"    Matrices: {len(rows)} (Passed: {kernel_passed}, Failed: {kernel_failed})")
        if kernel_perf:
            print(f"    Performance: {kernel_min_perf:.4f} - {kernel_max_perf:.4f} GFLOPS")
            print(f"    Avg performance: {kernel_avg_perf:.4f} GFLOPS")
        if kernel_time:
            print(f"    Execution time: {kernel_min_time:.2f} - {kernel_max_time:.2f} ms")
            print(f"    Avg execution time: {kernel_avg_time:.2f} ms")
        print()
    
    print("="*60)


def main():
    """Main function for command-line execution."""
    
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("USAGE: python parse_spmv_report.py <input_file> [output_csv]")
        print("\nExamples:")
        print("  python parse_spmv_report.py spmv-test-20260124112352.txt")
        print("    Output: default_spmv_results.csv")
        print("  python parse_spmv_report.py spmv-test-20260124112352.txt custom_results.csv")
        print("    Output: custom_results.csv")
        print("  python parse_spmv_report.py spmv-test-20260124112352.txt csr_spmv_results.csv")
        print("    Output: csr_spmv_results.csv")
        return 1
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Parse the report
    output_path, count = parse_spmv_report(input_file, output_file)
    
    if output_path and count > 0:
        print(f"\nSUCCESS: Parsed {count} benchmark reports.")
        print(f"Output file: {output_path}")
        return 0
    else:
        print("\nERROR: No data parsed. Check input file format.")
        return 1


if __name__ == "__main__":
    # Execute main function when run as script
    sys.exit(main())
