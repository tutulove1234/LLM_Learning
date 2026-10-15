# CUDA Runtime API Tour

这是一个面向 **API 设计学习** 的 CUDA C++ 示例集。默认程序按功能族覆盖 CUDA Runtime API
的主要抽象，并且会查询设备能力后跳过不适用的路径。它不是性能 benchmark。

> 基线：CUDA Toolkit 12.0+、CMake 3.24+、C++17。源码只使用稳定的 Runtime API；
> NVIDIA 当前文档可能包含更新的 API。默认生成 `sm_75`，请按你的 GPU 修改架构。

## 快速运行

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
./build/cuda_api_tour --help
./build/cuda_api_tour all
ctest --test-dir build --output-on-failure
```

较老的 CMake 不认识 `native` 时，查 GPU compute capability 并填数字，例如：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

各主题也能单独运行：

```bash
./build/cuda_api_tour info
./build/cuda_api_tour memory
./build/cuda_api_tour execution
./build/cuda_api_tour graphs
./build/cuda_api_tour mempool
./build/cuda_api_tour multigpu
```

没有 GPU/驱动时，`--help` 正常返回；硬件测试返回码为 77（CTest 视为 skip）。

## 阅读顺序

1. [`include/cuda_demo.hpp`](include/cuda_demo.hpp)：统一错误处理和 RAII 设备缓冲区。
2. [`src/device_info.cu`](src/device_info.cu)：Runtime 初始化、设备选择、属性、limit、
   Runtime/Driver API 互操作。
3. [`src/memory.cu`](src/memory.cu)：线性/页锁定/映射/注册/统一/二维/三维内存，
   symbol、CUDA array、texture object、surface object。
4. [`src/execution.cu`](src/execution.cu)：kernel launch、通用 launch、stream、event、
   host function、occupancy、cooperative launch。
5. [`src/graphs.cu`](src/graphs.cu)：显式图、stream capture、实例化、上传、修改、克隆。
6. [`src/mempool.cu`](src/mempool.cu)：`cudaMallocAsync`、默认/显式 pool、属性和 trim。
7. [`src/multigpu.cu`](src/multigpu.cu)：P2P 能力查询、启停 peer access、peer copy。

更完整的函数到示例映射见 [`docs/API_COVERAGE.md`](docs/API_COVERAGE.md)。

## 可选：同机跨进程 IPC

`cuda_ipc_demo` 是真实的父子进程示例，`fork()` 在任一进程初始化 CUDA **之前**发生。
父进程导出 allocation/event handle，子进程导入、等待并验证数据：

```bash
./build/cuda_ipc_demo
```

限制：Linux、同一台主机、支持 CUDA IPC 的设备/驱动环境。容器需要正确暴露 GPU，
MIG、WSL、Tegra 等环境可能有限制。IPC handle 只负责 GPU 资源；示例用 Unix pipe 传输
handle 字节。不要在初始化 CUDA 后再 `fork()`。

## 可选：外部资源互操作

[`optional/external_interop_skeleton.cu`](optional/external_interop_skeleton.cu) 展示 Linux
opaque-fd 外部 memory/semaphore 的导入、映射、wait/signal 和销毁。它默认不执行，因为有效
fd 必须由 Vulkan、OpenGL/EGL 或其他生产者创建。可只编译检查：

```bash
cmake -S . -B build -DBUILD_EXTERNAL_INTEROP=ON
cmake --build build -j
```

Windows 对应 Win32/NT handle；OpenGL、D3D9/10/11、VDPAU、EGL 的注册 API 还要求相应
图形上下文，无法做成无外部依赖且到处可运行的自动测试。

## 可选：跨机 NCCL + MPI

CUDA Runtime 本身没有“跨机器 memcpy”。[`optional/nccl_mpi_cross_host.cu`](optional/nccl_mpi_cross_host.cu)
展示常见设计：MPI 引导并广播 `ncclUniqueId`，每个进程选择本机 GPU，然后用 NCCL
执行跨 rank all-reduce。安装 MPI 和 NCCL 后：

```bash
cmake -S . -B build -DBUILD_NCCL_MPI=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j --target cuda_nccl_mpi
mpirun -np 2 ./build/cuda_nccl_mpi
```

真正跨机运行还需 launcher/SSH、网卡和防火墙配置；NCCL 可按机器条件选择
InfiniBand/RDMA 或 socket transport。

## 设计提示

- 几乎所有 Runtime 调用都通过 `CUDA_CHECK` 检查；kernel launch 后用
  `cudaGetLastError()` 检查 launch，再在明确的同步边界暴露异步执行错误。
- `cudaMemcpyAsync` 要真正与主机并行，host buffer 通常应为 page-locked memory。
- stream 内按序；stream 之间通过 event 表达依赖。销毁 stream/event 不等于任意资源都能
  提前释放，仍应遵守最后一次使用的生命周期。
- graph 的定义 (`cudaGraph_t`) 和可执行实例 (`cudaGraphExec_t`) 是两层对象；修改定义不会
  自动修改已实例化对象。
- async allocation 的生命周期也在 stream 顺序中表达；显式 pool 不能在其中仍有活动
  allocation 时销毁。
- 示例优先展示清晰的所有权和错误边界，未追求最短代码。

## 官方资料

- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

