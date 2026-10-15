#include "cuda_demo.hpp"

void demo_mempool() {
  if (!device_attribute(cudaDevAttrMemoryPoolsSupported)) {
    std::cout << "SKIP: stream-ordered allocator is unsupported\n";
    return;
  }
  const int device = current_device();
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaMemPool_t default_pool{}, current_pool{};
  CUDA_CHECK(cudaDeviceGetDefaultMemPool(&default_pool, device));
  CUDA_CHECK(cudaDeviceGetMemPool(&current_pool, device));
  std::uint64_t threshold = 4 * 1024 * 1024;
  CUDA_CHECK(cudaMemPoolSetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  std::uint64_t read_threshold = 0;
  CUDA_CHECK(cudaMemPoolGetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold,
                                     &read_threshold));

  void* async_pointer = nullptr;
  CUDA_CHECK(cudaMallocAsync(&async_pointer, 1 << 20, stream));
  CUDA_CHECK(cudaMemsetAsync(async_pointer, 0x2a, 1 << 20, stream));
  CUDA_CHECK(cudaFreeAsync(async_pointer, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.handleTypes = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id = device;
  cudaMemPool_t explicit_pool{};
  CUDA_CHECK(cudaMemPoolCreate(&explicit_pool, &properties));
  CUDA_CHECK(cudaMemPoolSetAttribute(explicit_pool, cudaMemPoolAttrReleaseThreshold,
                                     &threshold));
  CUDA_CHECK(cudaDeviceSetMemPool(device, explicit_pool));

  int* pool_pointer = nullptr;
  CUDA_CHECK(cudaMallocFromPoolAsync(&pool_pointer, 1024 * sizeof(int), explicit_pool, stream));
  CUDA_CHECK(cudaMemsetAsync(pool_pointer, 0, 1024 * sizeof(int), stream));
  CUDA_CHECK(cudaFreeAsync(pool_pointer, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::uint64_t reserved = 0, used = 0;
  CUDA_CHECK(cudaMemPoolGetAttribute(explicit_pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
  CUDA_CHECK(cudaMemPoolGetAttribute(explicit_pool, cudaMemPoolAttrUsedMemCurrent, &used));
  CUDA_CHECK(cudaMemPoolTrimTo(explicit_pool, 0));
  CUDA_CHECK(cudaDeviceSetMemPool(device, current_pool));
  CUDA_CHECK(cudaMemPoolDestroy(explicit_pool));
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::cout << "default threshold=" << read_threshold
            << " explicit pool reserved=" << reserved << " used=" << used << '\n';
}

