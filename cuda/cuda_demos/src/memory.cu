#include "cuda_demo.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

__device__ int device_symbol[4];
__constant__ float constant_scale;

__global__ void scale_kernel(float* data, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= constant_scale;
}

__global__ void texture_surface_kernel(cudaTextureObject_t texture,
                                       cudaSurfaceObject_t surface,
                                       int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    float value = tex2D<float>(texture, x + 0.5f, y + 0.5f);
    surf2Dwrite(value + 1.0f, surface, x * int(sizeof(float)), y);
  }
}

void demo_memory() {
  constexpr int n = 1024;
  const std::size_t bytes = n * sizeof(float);
  std::vector<float> host(n);
  std::iota(host.begin(), host.end(), 0.0f);

  // Classic linear device allocation/copies/memset and symbol APIs.
  DeviceBuffer<float> device(n);
  CUDA_CHECK(cudaMemset(device.get(), 0, bytes));
  CUDA_CHECK(cudaMemcpy(device.get(), host.data(), bytes, cudaMemcpyHostToDevice));
  const float scale = 2.0f;
  CUDA_CHECK(cudaMemcpyToSymbol(constant_scale, &scale, sizeof(scale)));
  scale_kernel<<<(n + 255) / 256, 256>>>(device.get(), n);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaMemcpy(host.data(), device.get(), bytes, cudaMemcpyDeviceToHost));

  void* symbol_address = nullptr;
  std::size_t symbol_size = 0;
  CUDA_CHECK(cudaGetSymbolAddress(&symbol_address, device_symbol));
  CUDA_CHECK(cudaGetSymbolSize(&symbol_size, device_symbol));
  int symbol_values[4] = {1, 2, 3, 4};
  CUDA_CHECK(cudaMemcpyToSymbol(device_symbol, symbol_values, sizeof(symbol_values)));
  std::fill(std::begin(symbol_values), std::end(symbol_values), 0);
  CUDA_CHECK(cudaMemcpyFromSymbol(symbol_values, device_symbol, sizeof(symbol_values)));
  std::cout << "linear/symbol: last=" << host.back() << " symbol-bytes=" << symbol_size
            << " address=" << symbol_address << '\n';

  // Page-locked and mapped host memory. cudaHostGetDevicePointer illustrates UVA.
  float* pinned = nullptr;
  CUDA_CHECK(cudaHostAlloc(&pinned, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
  std::fill_n(pinned, n, 3.0f);
  float* mapped_device_pointer = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&mapped_device_pointer, pinned, 0));
  CUDA_CHECK(cudaMemcpyAsync(device.get(), pinned, bytes, cudaMemcpyDefault));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFreeHost(pinned));

  // Register ordinary host storage for DMA, then unregister it.
  void* ordinary = std::malloc(bytes);
  if (!ordinary) throw std::bad_alloc();
  CUDA_CHECK(cudaHostRegister(ordinary, bytes, cudaHostRegisterPortable));
  unsigned int host_flags = 0;
  CUDA_CHECK(cudaHostGetFlags(&host_flags, ordinary));
  CUDA_CHECK(cudaHostUnregister(ordinary));
  std::free(ordinary);
  std::cout << "pinned/mapped/registered: host flags=" << host_flags << '\n';

  // Managed allocation, policy advice, migration, pointer introspection.
  float* managed = nullptr;
  CUDA_CHECK(cudaMallocManaged(&managed, bytes, cudaMemAttachGlobal));
  std::fill_n(managed, n, 1.0f);
  const int gpu = current_device();
  const bool can_prefetch = device_attribute(cudaDevAttrConcurrentManagedAccess) != 0;
#if CUDART_VERSION >= 13000
  // CUDA 13 generalized locations from an integer device id to cudaMemLocation.
  cudaMemLocation gpu_location{};
  gpu_location.type = cudaMemLocationTypeDevice;
  gpu_location.id = gpu;
#endif
  if (can_prefetch) {
#if CUDART_VERSION >= 13000
    CUDA_CHECK(cudaMemAdvise(managed, bytes, cudaMemAdviseSetPreferredLocation,
                             gpu_location));
    CUDA_CHECK(cudaMemAdvise(managed, bytes, cudaMemAdviseSetReadMostly,
                             gpu_location));
    CUDA_CHECK(cudaMemPrefetchAsync(managed, bytes, gpu_location, 0));
#else
    CUDA_CHECK(cudaMemAdvise(managed, bytes, cudaMemAdviseSetPreferredLocation, gpu));
    CUDA_CHECK(cudaMemAdvise(managed, bytes, cudaMemAdviseSetReadMostly, gpu));
    CUDA_CHECK(cudaMemPrefetchAsync(managed, bytes, gpu));
#endif
  }
  scale_kernel<<<(n + 255) / 256, 256>>>(managed, n);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaPointerAttributes attributes{};
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, managed));
  if (can_prefetch) {
#if CUDART_VERSION >= 13000
    CUDA_CHECK(cudaMemAdvise(managed, bytes, cudaMemAdviseUnsetReadMostly,
                             gpu_location));
#else
    CUDA_CHECK(cudaMemAdvise(managed, bytes, cudaMemAdviseUnsetReadMostly, gpu));
#endif
  }
  std::cout << "managed/UVA: value=" << managed[0]
            << " pointer-type=" << int(attributes.type)
            << " prefetch=" << (can_prefetch ? "yes" : "skipped") << '\n';
  CUDA_CHECK(cudaFree(managed));

  // Pitched 2-D memory and 2-D copies.
  constexpr std::size_t width = 17, height = 11;
  float* pitched = nullptr;
  std::size_t pitch = 0;
  CUDA_CHECK(cudaMallocPitch(&pitched, &pitch, width * sizeof(float), height));
  std::vector<float> matrix(width * height, 7.0f);
  CUDA_CHECK(cudaMemcpy2D(pitched, pitch, matrix.data(), width * sizeof(float),
                          width * sizeof(float), height, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset2D(pitched, pitch, 0, width * sizeof(float), height));
  CUDA_CHECK(cudaFree(pitched));

  // 3-D pitched allocation/copy/memset.
  cudaExtent volume_extent = make_cudaExtent(8 * sizeof(float), 4, 3);
  cudaPitchedPtr volume{};
  CUDA_CHECK(cudaMalloc3D(&volume, volume_extent));
  CUDA_CHECK(cudaMemset3D(volume, 0, volume_extent));
  cudaMemcpy3DParms volume_copy{};
  std::vector<float> volume_host(8 * 4 * 3, 5.0f);
  volume_copy.srcPtr = make_cudaPitchedPtr(volume_host.data(), 8 * sizeof(float), 8, 4);
  volume_copy.dstPtr = volume;
  volume_copy.extent = volume_extent;
  volume_copy.kind = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpy3D(&volume_copy));
  CUDA_CHECK(cudaFree(volume.ptr));

  // CUDA array + resource/texture/surface objects.
  cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();
  cudaArray_t array = nullptr;
  CUDA_CHECK(cudaMallocArray(&array, &channel, width, height, cudaArraySurfaceLoadStore));
  CUDA_CHECK(cudaMemcpy2DToArray(array, 0, 0, matrix.data(), width * sizeof(float),
                                width * sizeof(float), height, cudaMemcpyHostToDevice));
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = array;
  cudaTextureDesc texture_desc{};
  texture_desc.addressMode[0] = cudaAddressModeClamp;
  texture_desc.addressMode[1] = cudaAddressModeClamp;
  texture_desc.filterMode = cudaFilterModePoint;
  texture_desc.readMode = cudaReadModeElementType;
  texture_desc.normalizedCoords = 0;
  cudaTextureObject_t texture = 0;
  cudaSurfaceObject_t surface = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texture, &resource, &texture_desc, nullptr));
  CUDA_CHECK(cudaCreateSurfaceObject(&surface, &resource));
  dim3 block(8, 8), grid((width + 7) / 8, (height + 7) / 8);
  texture_surface_kernel<<<grid, block>>>(texture, surface, width, height);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy2DFromArray(matrix.data(), width * sizeof(float), array, 0, 0,
                                  width * sizeof(float), height, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDestroySurfaceObject(surface));
  CUDA_CHECK(cudaDestroyTextureObject(texture));
  CUDA_CHECK(cudaFreeArray(array));
  const float texture_error = matrix[0] - 8.0f;
  if (texture_error > 1e-5f || texture_error < -1e-5f)
    throw std::runtime_error("texture/surface check failed");
  std::cout << "pitched/3D/array/texture/surface: value=" << matrix[0] << '\n';
}
