#include "cuda_demo.hpp"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <iomanip>

bool has_cuda_device() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
    cudaGetLastError();  // clear the per-thread runtime error slot
    return false;
  }
  CUDA_CHECK(status);
  return count > 0;
}

int current_device() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

int device_attribute(cudaDeviceAttr attr) {
  int value = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, current_device()));
  return value;
}

void print_device_info() {
  int runtime_version = 0, driver_version = 0, count = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaGetDeviceCount(&count));
  std::cout << "runtime=" << runtime_version << " driver=" << driver_version
            << " devices=" << count << '\n';

  for (int i = 0; i < count; ++i) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, i));
    char pci_bus_id[32]{};
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), i));
    int by_pci = -1;
    CUDA_CHECK(cudaDeviceGetByPCIBusId(&by_pci, pci_bus_id));
    std::cout << "GPU " << i << ": " << p.name << " UUID=";
    for (unsigned char byte : p.uuid.bytes)
      std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    std::cout << std::dec << std::setfill(' ') << " PCI=" << pci_bus_id
              << " lookup=" << by_pci << " CC=" << p.major << '.' << p.minor
              << " globalMiB=" << p.totalGlobalMem / (1024 * 1024)
              << " SMs=" << p.multiProcessorCount << '\n';
  }

  CUDA_CHECK(cudaSetDevice(0));
  std::size_t free_bytes = 0, total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::cout << "memory free/total MiB=" << free_bytes / (1024 * 1024) << '/'
            << total_bytes / (1024 * 1024) << '\n';

  cudaFuncCache old_cache{};
  CUDA_CHECK(cudaDeviceGetCacheConfig(&old_cache));
  CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferNone));
  // Kept as a compatibility/design example. CUDA 13 marks these two bank-size
  // configuration calls deprecated; new code normally leaves this unchanged.
  cudaSharedMemConfig old_bank{};
  CUDA_CHECK(cudaDeviceGetSharedMemConfig(&old_bank));
  CUDA_CHECK(cudaDeviceSetSharedMemConfig(old_bank));
  std::size_t stack = 0;
  CUDA_CHECK(cudaDeviceGetLimit(&stack, cudaLimitStackSize));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack));
  std::cout << "UVA=" << device_attribute(cudaDevAttrUnifiedAddressing)
            << " managed=" << device_attribute(cudaDevAttrManagedMemory)
            << " async-engine-count=" << device_attribute(cudaDevAttrAsyncEngineCount)
            << " stack-limit=" << stack << '\n';

  // Ask the Runtime for a versioned Driver API entry point. This avoids directly
  // linking libcuda and makes the ABI-version selection explicit.
  void* entry_point = nullptr;
  cudaDriverEntryPointQueryResult entry_status{};
  CUDA_CHECK(cudaGetDriverEntryPointByVersion(
      "cuCtxGetCurrent", &entry_point, 12000, cudaEnableDefault, &entry_status));
  CUcontext context{};
  if (entry_point) {
    auto get_current = reinterpret_cast<PFN_cuCtxGetCurrent_v4000>(entry_point);
    CUresult result = get_current(&context);
    if (result != CUDA_SUCCESS)
      throw CudaError("cuCtxGetCurrent returned driver error " +
                      std::to_string(static_cast<int>(result)));
  }
  std::cout << "driver entry status=" << int(entry_status)
            << " runtime-created context=" << context << '\n';
  unsigned int device_flags = 0;
  CUDA_CHECK(cudaGetDeviceFlags(&device_flags));
  std::cout << "device flags=" << device_flags << '\n';
}
