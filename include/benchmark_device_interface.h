// benchmark_device_interface.h
#ifndef BENCHMARK_DEVICE_INTERFACE_H
#define BENCHMARK_DEVICE_INTERFACE_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Device memory management
void* allocate_device_memory(size_t bytes);
void copy_host_to_device(void* dst, const void* src, size_t bytes);
void copy_device_to_host(void* dst, const void* src, size_t bytes);
void memset_device(void* dst, int value, size_t bytes);
void free_device_memory(void* ptr);
void set_cuda_sync_enabled(int enabled);
void synchronize_cuda();

#ifdef __cplusplus
}
#endif

#endif // BENCHMARK_DEVICE_INTERFACE_H
