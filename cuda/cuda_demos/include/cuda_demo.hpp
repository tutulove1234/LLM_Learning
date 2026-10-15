#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class CudaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline void cuda_check(cudaError_t status, const char* expression,
                       const char* file, int line) {
  if (status != cudaSuccess) {
    throw CudaError(std::string(file) + ":" + std::to_string(line) + " " +
                    expression + " failed: " + cudaGetErrorName(status) +
                    " (" + cudaGetErrorString(status) + ")");
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

struct Demo {
  const char* name;
  const char* description;
  void (*run)();
};

bool has_cuda_device();
int current_device();
int device_attribute(cudaDeviceAttr attr);
void print_device_info();
void demo_memory();
void demo_execution_and_streams();
void demo_graphs();
void demo_mempool();
void demo_multigpu();

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return ptr_; }
  std::size_t size() const { return count_; }
 private:
  T* ptr_{};
  std::size_t count_{};
};
