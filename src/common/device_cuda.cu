// CUDA device memory helpers shared by SpMV and SpMM.
#include "benchmark_device_interface.h"
#if defined(CUDA_ENABLED) && CUDA_ENABLED
#include <cuda_runtime.h>
#endif
#include <cstdio>

// CUDA memory management functions
void* allocate_device_memory(size_t bytes) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    void* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA memory allocation failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return d_ptr;
#else
    return nullptr;
#endif
}

void copy_host_to_device(void* dst, const void* src, size_t bytes) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA host-to-device copy failed: %s\n", cudaGetErrorString(err));
    }
#endif
}

void copy_device_to_host(void* dst, const void* src, size_t bytes) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA device-to-host copy failed: %s\n", cudaGetErrorString(err));
    }
#endif
}

void memset_device(void* dst, int value, size_t bytes) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    cudaError_t err = cudaMemset(dst, value, bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA memset failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
    }
#endif
}

void free_device_memory(void* ptr) {
#if defined(CUDA_ENABLED) && CUDA_ENABLED
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "CUDA memory free failed: %s\n", cudaGetErrorString(err));
        }
    }
#endif
}
