#include "spmv_benchmark.h"
#include "spmm_benchmark.h"
#include <algorithm>

int main(int argc, char* argv[]) {
    
    std::string kernelname = "default_spmv";
    std::string reportfile = "";
    std::string operator_name = "spmv";
    int dense_cols = 8;
    
    std::cout << "SpMV/SpMM Benchmark Test" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Precision: " << spmv_precision_name() << std::endl;

    std::string filename;

    if (argc > 3) {
        // 从命令行参数读取.mtx文件
        filename = argv[1];
        kernelname = argv[2];
        reportfile = argv[3];
        if (argc > 4) {
            operator_name = argv[4];
        }
        if (argc > 5) {
            dense_cols = std::max(1, std::stoi(argv[5]));
        }
        std::cout << "Loading matrix from file: " << filename << std::endl;
    } 
    else {
        // 参数不全
        std::cout << "Missing Inputs!! Please  Ensure 2 Inputs (path/to/matrix, YourKernelName). " << std::endl; 
        exit(0);
    }

    if (operator_name == "spmm" || operator_name == "SpMM") {
        SpMM_Benchmark benchmark(filename, dense_cols);
        benchmark.set_kernel_name(kernelname);
        benchmark.set_report_file(reportfile);
        benchmark.print_matrix_info();

        std::cout << "\nRunning SpMM benchmark of " << kernelname << " with preprocessing..." << std::endl;
        auto [preprocess_time, spmm_avg_time] = benchmark.benchmark_spmm(10);
        double perf_gflops = benchmark.calculate_performance(spmm_avg_time);

        std::cout << "Preprocessing time: " << preprocess_time << " microseconds" << std::endl;
        std::cout << "Average SpMM execution time: " << spmm_avg_time << " microseconds" << std::endl;
        std::cout << "Performance: " << perf_gflops << " GFLOPS" << std::endl;

        std::cout << "\nValidating correctness..." << std::endl;
        bool correct = benchmark.validate_correctness();
        benchmark.write_report(std::make_pair(preprocess_time, spmm_avg_time), perf_gflops, correct);
        return correct ? 0 : 1;
    }

    SpMV_Benchmark benchmark(filename);
    benchmark.set_kernel_name(kernelname);
    benchmark.set_report_file(reportfile);
    benchmark.print_matrix_info();
    
    std::cout << "\nRunning SpMV benchmark of "<< kernelname << " with preprocessing..." << std::endl;
    auto [preprocess_time, spmv_avg_time] = benchmark.benchmark_spmv(10);
    double perf_gflops = benchmark.calculate_performance(spmv_avg_time);
    
    std::cout << "Preprocessing time: " << preprocess_time << " microseconds" << std::endl;
    std::cout << "Average SpMV execution time: " << spmv_avg_time << " microseconds" << std::endl;
    std::cout << "Performance: " << perf_gflops << " GFLOPS" << std::endl;
    
    std::cout << "\nValidating correctness..." << std::endl;
    bool correct = benchmark.validate_correctness();
    
    // Write detailed report
    benchmark.write_report(std::make_pair(preprocess_time, spmv_avg_time), perf_gflops, correct);
    
    return correct ? 0 : 1;
}
