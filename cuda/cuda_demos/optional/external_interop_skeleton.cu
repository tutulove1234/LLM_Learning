// NOT RUN BY DEFAULT: external handles must come from Vulkan/D3D/another producer.
// This file is a compact ownership/lifetime template for CUDA external interop.
#include "cuda_demo.hpp"

#if defined(__linux__)
void consume_external_resources(int memory_fd, std::size_t bytes,
                                int semaphore_fd, cudaStream_t stream) {
  cudaExternalMemoryHandleDesc memory_desc{};
  memory_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memory_desc.handle.fd = memory_fd;  // ownership transfers to CUDA on successful import
  memory_desc.size = bytes;
  cudaExternalMemory_t external_memory{};
  CUDA_CHECK(cudaImportExternalMemory(&external_memory, &memory_desc));

  cudaExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.size = bytes;
  void* mapped = nullptr;
  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&mapped, external_memory, &buffer_desc));

  cudaExternalSemaphoreHandleDesc semaphore_desc{};
  semaphore_desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  semaphore_desc.handle.fd = semaphore_fd;
  cudaExternalSemaphore_t semaphore{};
  CUDA_CHECK(cudaImportExternalSemaphore(&semaphore, &semaphore_desc));
  cudaExternalSemaphoreWaitParams wait_params{};
  CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&semaphore, &wait_params, 1, stream));
  CUDA_CHECK(cudaMemsetAsync(mapped, 0, bytes, stream));
  cudaExternalSemaphoreSignalParams signal_params{};
  CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&semaphore, &signal_params, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(mapped));
  CUDA_CHECK(cudaDestroyExternalSemaphore(semaphore));
  CUDA_CHECK(cudaDestroyExternalMemory(external_memory));
}
#endif

