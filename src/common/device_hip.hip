// HIP device memory helpers shared by SpMV and SpMM.
#include "benchmark_device_interface.h"
#if defined(HIP_ENABLED) && HIP_ENABLED
#include <hip/hip_runtime.h>
#endif
#include <cstdio>

// HIP memory management functions
void* allocate_device_memory(size_t bytes) {
#if defined(HIP_ENABLED) && HIP_ENABLED
    void* d_ptr = nullptr;
    hipError_t err = hipMalloc(&d_ptr, bytes);
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP memory allocation failed: %s\n", hipGetErrorString(err));
        return nullptr;
    }
    return d_ptr;
#else
    return nullptr;
#endif
}

void copy_host_to_device(void* dst, const void* src, size_t bytes) {
#if defined(HIP_ENABLED) && HIP_ENABLED
    hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP host-to-device copy failed: %s\n", hipGetErrorString(err));
    }
#endif
}

void copy_device_to_host(void* dst, const void* src, size_t bytes) {
#if defined(HIP_ENABLED) && HIP_ENABLED
    hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP device-to-host copy failed: %s\n", hipGetErrorString(err));
    }
#endif
}

void memset_device(void* dst, int value, size_t bytes) {
#if defined(HIP_ENABLED) && HIP_ENABLED
    hipError_t err = hipMemset(dst, value, bytes);
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP memset failed: %s\n", hipGetErrorString(err));
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP sync failed: %s\n", hipGetErrorString(err));
    }
#endif
}

void free_device_memory(void* ptr) {
#if defined(HIP_ENABLED) && HIP_ENABLED
    if (ptr) {
        hipError_t err = hipFree(ptr);
        if (err != hipSuccess) {
            std::fprintf(stderr, "HIP memory free failed: %s\n", hipGetErrorString(err));
        }
    }
#endif
}
