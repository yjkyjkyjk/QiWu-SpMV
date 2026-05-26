#include "spmv_benchmark.h"
#include <memory>

int main(int argc, char* argv[]) {
    
    std::string kernelname = "default_spmv";
    std::string reportfile = "";
    
    std::cout << "SpMV Benchmark Test" << std::endl;   
    std::cout << "===================" << std::endl;
    std::cout << "Precision: " << spmv_precision_name() << std::endl;

    std::unique_ptr<SpMV_Benchmark> benchmark_ptr;

    if (argc > 3) {
        // 从命令行参数读取.mtx文件
        std::string filename = argv[1];
        kernelname = argv[2];
        reportfile = argv[3];
        std::cout << "Loading matrix from file: " << filename << std::endl;
        benchmark_ptr = std::make_unique<SpMV_Benchmark>(filename);
    } 
    else {
        // 参数不全
        std::cout << "Missing Inputs!! Please  Ensure 2 Inputs (path/to/matrix, YourKernelName). " << std::endl; 
        exit(0);
    }

    // Create benchmark object
    SpMV_Benchmark& benchmark = *benchmark_ptr;
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
