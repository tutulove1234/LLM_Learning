# API 覆盖索引

“覆盖所有 CUDA API”并不是一个可由单个可移植程序严格实现的目标：CUDA Runtime API
包含数百个入口，其中有互斥的平台 API、已弃用 API、图形/视频栈绑定 API、只有特定 GPU
才支持的 API，以及必须由外部系统提供 handle 的 API。本项目按**概念和功能族**覆盖现代
Runtime API；同族中高度重复的同步/异步、1D/2D/3D 变体选取有代表性的调用。

| 功能族 | 代表性 API | 示例 | 默认执行 |
|---|---|---|---|
| 初始化、设备、版本 | `cudaGetDeviceCount`, `cudaSet/GetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `cudaRuntime/DriverGetVersion` | `device_info.cu` | 是 |
| PCI/UUID、资源、配置 | `cudaDeviceGetPCIBusId`, `cudaDeviceGetByPCIBusId`, `cudaMemGetInfo`, device limit/cache/shared-memory config | `device_info.cu` | 是 |
| 错误模型 | `cudaGetErrorName`, `cudaGetErrorString`, `cudaGetLastError` | `cuda_demo.hpp`, 全部 | 是 |
| Driver/Runtime 互操作 | `cudaGetDriverEntryPointByVersion` + `cuCtxGetCurrent` 函数指针 | `device_info.cu` | 是 |
| 线性内存与 copy/set | `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset[Async]` | `memory.cu`, `execution.cu` | 是 |
| Host memory | `cudaHostAlloc`, `cudaFreeHost`, `cudaHostRegister/Unregister`, `cudaHostGetDevicePointer/Flags` | `memory.cu` | 是 |
| Unified Memory/UVA | `cudaMallocManaged`, `cudaMemAdvise`, `cudaMemPrefetchAsync`, `cudaPointerGetAttributes`, `cudaMemcpyDefault` | `memory.cu` | 是 |
| Symbol/constant memory | `cudaMemcpyTo/FromSymbol`, `cudaGetSymbolAddress/Size` | `memory.cu` | 是 |
| 2D/3D 与 array | `cudaMallocPitch`, `cudaMemcpy2D`, `cudaMemset2D`, `cudaMalloc3D`, `cudaMemcpy3D`, `cudaMemset3D`, `cudaMallocArray` | `memory.cu` | 是 |
| Texture/Surface object | create/destroy texture/surface object，`tex2D`, `surf2Dwrite` | `memory.cu` | 是 |
| Kernel execution | `<<<>>>`, `cudaLaunchKernel`, dynamic shared-memory 参数 | `execution.cu` | 是 |
| Occupancy/function config | `cudaOccupancyMaxPotentialBlockSize`, active blocks/SM, `cudaFuncGetAttributes`, `cudaFuncSetCacheConfig` | `execution.cu` | 是 |
| Stream | create(flags/priority), query flags/priority, wait event, sync, destroy, `cudaLaunchHostFunc` | `execution.cu` | 是 |
| Event | create(flags), record, query, wait/sync, elapsed time, destroy | `execution.cu` | 是 |
| Cooperative launch | `cudaLaunchCooperativeKernel` | `execution.cu` | 能力检测 |
| Graph 显式构造 | memset/kernel/memcpy/empty nodes, dependencies, root/node/edge query | `graphs.cu` | 是 |
| Graph 生命周期/更新 | instantiate/upload/launch, exec-node update, exec update, clone/find-in-clone | `graphs.cu` | 是 |
| Stream capture | begin/end capture, capture info | `graphs.cu` | 是 |
| Stream-ordered allocator | `cudaMallocAsync`, `cudaFreeAsync`, default/current pool | `mempool.cu` | 能力检测 |
| 显式 memory pool | create/destroy, from-pool alloc, attribute, trim, set current | `mempool.cu` | 能力检测 |
| 多 GPU/P2P | can-access, P2P attributes, enable/disable peer, `cudaMemcpyPeerAsync` | `multigpu.cu` | 至少 2 GPU |
| 同机跨进程 | IPC memory/event get/open/close handle | `optional/ipc_demo.cu` | 独立运行 |
| 外部资源 | import/map external memory, import/wait/signal external semaphore | `optional/external_interop_skeleton.cu` | 仅可选编译 |
| 跨机 collective | MPI bootstrap + NCCL communicator/all-reduce | `optional/nccl_mpi_cross_host.cu` | 需 MPI/NCCL |

## 特意不纳入默认可执行文件

- **Deprecated API**：除 `device_info.cu` 中特意保留并注明的 shared-memory bank config
  兼容示例外，不覆盖旧 texture/surface reference、旧 thread-management、部分旧 launch API。
  学习新代码不应复制它们；需要维护遗留项目时再查对应 Toolkit 版本的 archive 文档。
- **图形/视频互操作**：OpenGL、D3D9/10/11、VDPAU、EGL 必须先创建平台图形上下文。
  外部资源 skeleton 展示现代显式 handle 模式的核心生命周期。
- **设备端高级特性**：dynamic parallelism、cluster launch、programmatic dependent launch、
  tensor core/PTX 等更偏 kernel 编程或特定架构，不是通用 Runtime API 入门路径。
- **Driver API 的 VMM、module、linker**：这是另一套更底层 API；这里只演示 Runtime 创建的
  primary context 如何被 Driver API 观察。不要把 `CUdeviceptr` 和 Runtime pointer 不加说明地混用。
- **Profiler、sanitizer、cuBLAS/cuFFT/cuRAND/NVRTC**：分别属于工具或独立库，不属于
  Runtime API。它们适合各自独立示例集。

## 同步语义速查

```text
host enqueue ──> stream A: memset -> kernel -> record(event)
                                      │
                                      └──> stream B: wait(event) -> kernel -> D2H -> host function
```

API 名称带 `Async` 只表示调用具有异步语义的可能；是否阻塞 host 还取决于 copy 方向、host
内存是否 page-locked、上下文初始化和资源压力。最终错误也可能直到后续同步 API 才报告。
