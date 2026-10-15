#include "cuda_demo.hpp"

#include <algorithm>

__global__ void add_scalar_kernel(int* values, int n, int delta) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) values[i] += delta;
}

void demo_graphs() {
  constexpr int n = 256;
  const std::size_t bytes = n * sizeof(int);
  DeviceBuffer<int> device(n);
  int* host = nullptr;
  CUDA_CHECK(cudaMallocHost(&host, bytes));

  cudaGraph_t graph{};
  CUDA_CHECK(cudaGraphCreate(&graph, 0));
  cudaGraphNode_t memset_node{}, kernel_node{}, copy_node{}, empty_node{};
  cudaMemsetParams memset_params{};
  memset_params.dst = device.get();
  memset_params.value = 0;
  memset_params.pitch = 0;
  memset_params.elementSize = sizeof(int);
  memset_params.width = n;
  memset_params.height = 1;
  CUDA_CHECK(cudaGraphAddMemsetNode(&memset_node, graph, nullptr, 0, &memset_params));

  int* pointer_arg = device.get();
  int n_arg = n;
  int delta_arg = 1;
  void* args[] = {&pointer_arg, &n_arg, &delta_arg};
  cudaKernelNodeParams kernel_params{};
  kernel_params.func = reinterpret_cast<void*>(add_scalar_kernel);
  kernel_params.gridDim = dim3(1);
  kernel_params.blockDim = dim3(n);
  kernel_params.kernelParams = args;
  CUDA_CHECK(cudaGraphAddKernelNode(&kernel_node, graph, &memset_node, 1, &kernel_params));

  cudaMemcpy3DParms copy_params{};
  copy_params.srcPtr = make_cudaPitchedPtr(device.get(), bytes, n, 1);
  copy_params.dstPtr = make_cudaPitchedPtr(host, bytes, n, 1);
  copy_params.extent = make_cudaExtent(bytes, 1, 1);
  copy_params.kind = cudaMemcpyDeviceToHost;
  CUDA_CHECK(cudaGraphAddMemcpyNode(&copy_node, graph, &kernel_node, 1, &copy_params));
  CUDA_CHECK(cudaGraphAddEmptyNode(&empty_node, graph, &copy_node, 1));

  std::size_t node_count = 0, edge_count = 0, root_count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
#if CUDART_VERSION >= 13000
  CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edge_count));
#else
  CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, &edge_count));
#endif
  CUDA_CHECK(cudaGraphGetRootNodes(graph, nullptr, &root_count));
  cudaGraphNodeType node_type{};
  CUDA_CHECK(cudaGraphNodeGetType(kernel_node, &node_type));

  cudaGraphExec_t executable{};
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphUpload(executable, nullptr));
  CUDA_CHECK(cudaGraphLaunch(executable, nullptr));
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  if (!std::all_of(host, host + n, [](int x) { return x == 1; }))
    throw std::runtime_error("explicit graph result mismatch");

  // Change the instantiated executable without rebuilding it.
  delta_arg = 3;
  CUDA_CHECK(cudaGraphExecKernelNodeSetParams(executable, kernel_node, &kernel_params));
  CUDA_CHECK(cudaGraphLaunch(executable, nullptr));
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  if (!std::all_of(host, host + n, [](int x) { return x == 3; }))
    throw std::runtime_error("updated graph result mismatch");

  cudaGraph_t clone{};
  CUDA_CHECK(cudaGraphClone(&clone, graph));
  cudaGraphNode_t clone_kernel{};
  CUDA_CHECK(cudaGraphNodeFindInClone(&clone_kernel, kernel_node, clone));
  cudaGraphExecUpdateResult update_result{};
  cudaGraphNode_t error_node{};
  CUDA_CHECK(cudaGraphExecUpdate(executable, graph, &error_node, &update_result));
  std::cout << "explicit graph nodes=" << node_count << " edges=" << edge_count
            << " roots=" << root_count << " kernel-node-type=" << int(node_type)
            << " update-result=" << int(update_result) << '\n';

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(clone));
  CUDA_CHECK(cudaGraphDestroy(graph));

  // Convert normal stream work into a graph with capture.
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaGraph_t captured{};
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(cudaMemsetAsync(device.get(), 0, bytes, stream));
  add_scalar_kernel<<<1, n, 0, stream>>>(device.get(), n, 5);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaMemcpyAsync(host, device.get(), bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
  cudaStreamCaptureStatus capture_status{};
  unsigned long long capture_id = 0;
  CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &capture_status, &capture_id));
  cudaGraphExec_t captured_exec{};
  CUDA_CHECK(cudaGraphInstantiate(&captured_exec, captured, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(captured_exec, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (!std::all_of(host, host + n, [](int x) { return x == 5; }))
    throw std::runtime_error("captured graph result mismatch");
  std::cout << "stream capture: status-after-capture=" << int(capture_status)
            << " id=" << capture_id << '\n';

  CUDA_CHECK(cudaGraphExecDestroy(captured_exec));
  CUDA_CHECK(cudaGraphDestroy(captured));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFreeHost(host));
}
