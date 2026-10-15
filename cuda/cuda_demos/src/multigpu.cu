#include "cuda_demo.hpp"

#include <vector>

void demo_multigpu() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count < 2) {
    std::cout << "SKIP: peer demo needs at least two CUDA devices\n";
    return;
  }
  int can_01 = 0, can_10 = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_01, 0, 1));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_10, 1, 0));
  int rank = -1, native_atomics = 0;
  CUDA_CHECK(cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, 0, 1));
  CUDA_CHECK(cudaDeviceGetP2PAttribute(&native_atomics,
                                       cudaDevP2PAttrNativeAtomicSupported, 0, 1));
  std::cout << "0->1=" << can_01 << " 1->0=" << can_10
            << " performance-rank=" << rank << " native-atomics=" << native_atomics << '\n';
  if (!can_01 || !can_10) {
    std::cout << "SKIP: bidirectional peer access unavailable\n";
    return;
  }

  constexpr int n = 256;
  constexpr std::size_t bytes = n * sizeof(int);
  int *on_0 = nullptr, *on_1 = nullptr;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
  CUDA_CHECK(cudaMalloc(&on_0, bytes));
  std::vector<int> input(n, 42), output(n, 0);
  CUDA_CHECK(cudaMemcpy(on_0, input.data(), bytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
  CUDA_CHECK(cudaMalloc(&on_1, bytes));
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMemcpyPeerAsync(on_1, 1, on_0, 0, bytes, stream));
  CUDA_CHECK(cudaMemcpyAsync(output.data(), on_1, bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (output.front() != 42 || output.back() != 42)
    throw std::runtime_error("peer copy verification failed");
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(on_1));
  CUDA_CHECK(cudaDeviceDisablePeerAccess(0));

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(on_0));
  CUDA_CHECK(cudaDeviceDisablePeerAccess(1));
  std::cout << "peer access/copy: verified\n";
}
