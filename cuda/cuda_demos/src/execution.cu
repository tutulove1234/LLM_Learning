#include "cuda_demo.hpp"

#include <atomic>
#include <chrono>
#include <thread>

__global__ void saxpy_kernel(float a, const float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

__global__ void cooperative_kernel(int* counter) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(counter, 1);
}

static void CUDART_CB host_completion(void* data) {
  static_cast<std::atomic<bool>*>(data)->store(true, std::memory_order_release);
}

void demo_execution_and_streams() {
  constexpr int n = 1 << 16;
  const std::size_t bytes = n * sizeof(float);
  DeviceBuffer<float> x(n), y(n);

  int least_priority = 0, greatest_priority = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  cudaStream_t producer{}, consumer{};
  CUDA_CHECK(cudaStreamCreateWithPriority(&producer, cudaStreamNonBlocking, greatest_priority));
  CUDA_CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
  unsigned int flags = 0;
  int priority = 0;
  CUDA_CHECK(cudaStreamGetFlags(producer, &flags));
  CUDA_CHECK(cudaStreamGetPriority(producer, &priority));

  cudaEvent_t start{}, ready{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, producer));
  CUDA_CHECK(cudaMemsetAsync(x.get(), 0, bytes, producer));
  CUDA_CHECK(cudaMemsetAsync(y.get(), 0, bytes, producer));

  int min_grid = 0, block_size = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, saxpy_kernel, 0, n));
  int active_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, saxpy_kernel, block_size, 0));
  cudaFuncAttributes function_attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&function_attributes, saxpy_kernel));
  CUDA_CHECK(cudaFuncSetCacheConfig(saxpy_kernel, cudaFuncCachePreferL1));

  // Generic launch API: the argument array contains addresses of host-side argument values.
  float a = 2.0f;
  const float* x_arg = x.get();
  float* y_arg = y.get();
  int n_arg = n;
  void* arguments[] = {&a, &x_arg, &y_arg, &n_arg};
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(saxpy_kernel),
                              dim3((n + block_size - 1) / block_size), dim3(block_size),
                              arguments, 0, producer));
  CUDA_CHECK(cudaEventRecord(ready, producer));

  // Cross-stream dependency without synchronizing the host.
  CUDA_CHECK(cudaStreamWaitEvent(consumer, ready, 0));
  saxpy_kernel<<<(n + 255) / 256, 256, 0, consumer>>>(1.0f, x.get(), y.get(), n);
  CUDA_KERNEL_CHECK();
  std::atomic<bool> callback_ran{false};
  CUDA_CHECK(cudaLaunchHostFunc(consumer, host_completion, &callback_ran));
  CUDA_CHECK(cudaEventRecord(stop, consumer));

  cudaError_t query = cudaEventQuery(stop);
  if (query != cudaSuccess && query != cudaErrorNotReady) CUDA_CHECK(query);
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  if (!callback_ran.load(std::memory_order_acquire))
    throw std::runtime_error("host function did not run");

  // Cooperative launch is capability-dependent and therefore guarded.
  if (device_attribute(cudaDevAttrCooperativeLaunch)) {
    DeviceBuffer<int> counter(1);
    CUDA_CHECK(cudaMemset(counter.get(), 0, sizeof(int)));
    int* counter_arg = counter.get();
    void* cooperative_args[] = {&counter_arg};
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(cooperative_kernel), dim3(1), dim3(32),
        cooperative_args, 0, consumer));
    CUDA_CHECK(cudaStreamSynchronize(consumer));
    std::cout << "cooperative launch: supported and executed\n";
  } else {
    std::cout << "cooperative launch: SKIP (device unsupported)\n";
  }

  std::cout << "stream flags=" << flags << " priority=" << priority
            << " priority-range=[" << greatest_priority << ',' << least_priority << "]\n"
            << "occupancy suggestion block=" << block_size << " min-grid=" << min_grid
            << " active-blocks/SM=" << active_blocks
            << " registers/thread=" << function_attributes.numRegs
            << " elapsed-ms=" << milliseconds << '\n';

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(ready));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaStreamDestroy(consumer));
  CUDA_CHECK(cudaStreamDestroy(producer));
}
