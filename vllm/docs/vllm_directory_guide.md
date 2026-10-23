# vLLM 项目目录导览 —— 聚焦调度、KV Cache 管理与存储层级

> 面向 vLLM 初学者，基于当前仓库的 V1 实现。重点讲解三个主题：
> 1. **调度**（谁在什么时候用 GPU 计算）
> 2. **KV Cache 管理**（已算过的 token 的 Key/Value 存哪、怎么复用）
> 3. **存储层级**（GPU 显存 → CPU 内存 → SSD/远端 的分层缓存）

## 0. 一句话总览

- **Scheduler** 决定"何时做"：把请求排成批次，控制每个批次算多少 token。
- **KVCacheManager / BlockPool** 决定"用哪些块"：以固定大小的 block 为单位分配 GPU 显存，并做前缀复用（prefix caching）。
- **KVConnector / Offloading 体系** 决定"数据搬到哪"：把不常用的 KV 从 GPU 卸到 CPU、文件系统、远端缓存。

---

## 1. 顶层目录

| 目录 | 作用 | 与本主题的关系 |
|---|---|---|
| `vllm/` | 核心 Python 源码（V1 引擎为主） | ★ 本文重点，见第 2 节 |
| `csrc/` | C++/CUDA 源码：attention kernel、采样、量化、通信等 | attention 后端按 block 布局读写 GPU KV 张量 |
| `tests/` | 单元/集成测试 | `tests/v1/core/`、`tests/v1/kv_offload/` 直接覆盖本主题 |
| `docs/` | 官方文档（mkdocs） | `docs/design/` 有调度与 KV cache 设计文档 |
| `examples/` | 使用示例 | 含 prefix caching、KV offload 相关示例 |
| `benchmarks/` | 性能基准 | 含 KV offload / 前缀缓存性能测试 |
| `requirements/` | 依赖清单 | — |
| `tools/` | CI 与开发辅助脚本 | — |
| `cmake/`、`CMakeLists.txt`、`setup.py` | 构建 C++ 扩展 | — |
| `kv_cache_architecture.md` | ★ 仓库内自带的 KV cache 架构中文详解（强烈推荐先读） | 正是本主题的最佳入门材料 |
| `own_ssd_offload.md` | 自研 SSD Offload 后端的设计草案（未实现） | 存储层级扩展路线 |

---

## 2. `vllm/` 核心包结构

### 2.1 引擎与调度层

| 路径 | 作用 |
|---|---|
| `vllm/engine/` | V0 引擎入口（LLM 类、AsyncLLMEngine 兼容层） |
| `vllm/v1/engine/` | **V1 引擎入口与前后端编排**（关键） |
| ↳ `core.py` | EngineCore：调度循环所在的核心进程 |
| ↳ `async_llm.py` / `llm_engine.py` | 面向用户的 `AsyncLLM` / `LLM` API |
| ↳ `output_processor.py` | 把 worker 输出转成流式 token 给用户 |
| ↳ `detokenizer.py` | 反分词，独立进程/协程 |
| ↳ `admission_control.py` | 准入控制（是否接下新请求） |
| ↳ `core_client.py` / `coordinator.py` | 客户端与多引擎（DP）协调 |
| `vllm/v1/core/sched/` | ★★ **调度器本体** |
| ↳ `scheduler.py` | V1 主调度器：从等待队列选请求、按 token 预算组批、处理抢占（preemption）、调用 KVCacheManager 分配块、给 Worker 下发调度输出 |
| ↳ `async_scheduler.py` | 异步调度变体（输出与下一轮调度重叠） |
| ↳ `diffusion_scheduler.py` | 扩散模型的专用调度 |
| ↳ `request_queue.py` | 各类等待队列/排序策略（如 priority） |
| ↳ `output.py` / `interface.py` / `utils.py` | 调度输出结构 `SchedulerOutput`、抽象接口、辅助逻辑 |
| `vllm/v1/request.py` | 请求对象 `Request`（状态机：WAITING/RUNNING/FINISHED 等） |
| `vllm/v1/executor/` | 把调度结果分发给多 GPU worker（MultiprocExecutor、RayExecutor 等） |
| `vllm/v1/worker/` | ★ Worker 侧（真正跑模型） |
| ↳ `gpu_model_runner.py` | 执行前向、维护 block table、管理 GPU KV 张量与输入批次 |
| ↳ `gpu_worker.py` | Worker 进程封装（加载权重、采样循环） |
| ↳ `kv_connector_model_runner_mixin.py` | ★ KV Connector 在执行侧的挂载点（load/store KV 的时机） |
| ↳ `block_table.py` | 逻辑块 → 物理块的映射表 |
| ↳ `cpu/`、`cpu_worker.py`、`cpu_model_runner.py` | CPU 后端 |

### 2.2 KV Cache 管理层（★核心）

| 路径 | 作用 |
|---|---|
| `vllm/v1/core/block_pool.py` | ★ **BlockPool**：空闲块池 + 引用计数 + 按前缀哈希索引的复用（prefix caching 的核心数据结构） |
| `vllm/v1/core/kv_cache_manager.py` | ★ **KVCacheManager**：请求级接口（分配/释放/命中查询），内部委派给 Coordinator + BlockPool |
| `vllm/v1/core/kv_cache_coordinator.py` | 协调多个 KV cache group（全注意力 / 滑动窗口 / Mamba 各自一种组） |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 单一类型缓存的管理实现（如滑动窗口的特殊块回收规则） |
| `vllm/v1/core/kv_cache_utils.py` | 哈希块计算、调度侧需要的 KV 元数据工具 |
| `vllm/v1/core/encoder_cache_manager.py` | 编码器（多模态）中间缓存的分配管理 |
| `vllm/v1/core/kv_cache_metrics.py` | 前缀命中/逐出等指标统计 |
| `vllm/v1/kv_cache_interface.py` | ★ KV cache 的物理布局抽象（block 数、block 大小、page/hybrid 布局描述） |
| `vllm/v1/kv_cache_layout.py` | KV 张量在显存中的布局（page 尺寸等） |
| `vllm/v1/kv_cache_spec_registry.py` | 各种 KVCacheSpec 的注册表 |
| `vllm/v1/attention/backends/` | 各模型/后端的 attention 实现（FlashAttention、FlashInfer、MLA、Mamba 等），定义各自所需的 KVCacheSpec |

**心智模型**：Scheduler 只问 KVCacheManager "这些请求能不能塞下"；KVCacheManager 通过 BlockPool 做实际块分配；而张量本体放在 `v1/worker/gpu_model_runner.py` 持有的一整块 GPU 内存里，由 attention kernel 按 block table 寻址。

### 2.3 存储层级 / KV 卸载（Offload）

| 路径 | 作用 |
|---|---|
| `vllm/v1/kv_offload/` | ★★ **GPU→外部介质的 KV 卸载框架** |
| ↳ `base.py` | 三大抽象：`OffloadingSpec`（配置入口）、`OffloadingManager`（缓存索引/淘汰）、`OffloadingWorker`（实际拷贝） |
| ↳ `config.py` / `factory.py` | 卸载配置与工厂 |
| ↳ `file_mapper.py` | 逻辑地址 → 文件/槽位映射 |
| ↳ `cpu/` | **CPU 一级卸载**：`manager.py`（CPU 缓存索引与淘汰）、`gpu_worker.py`（GPU↔CPU 拷贝执行）、`spec.py`（CPUOffloadingSpec）、`shared_offload_region.py`（共享内存区域）、`policies/`（淘汰策略） |
| ↳ `tiering/` | ★ **多级分层缓存**（GPU ↔ CPU 一级 ↔ 文件系统/对象存储/P2P 次级层）：`spec.py`（TieringOffloadingSpec）、`manager.py`（分层协调）、`fs/`（文件系统后端）、`obj/`（对象存储）、`p2p/`（实例间点对点）、`backpressure.py`（反压） |
| `vllm/v1/simple_kv_offload/` | 简化版 KV 卸载实现（`disk_backend.py` 磁盘后端、`manager.py`、`worker.py`），适合阅读入门 |
| `vllm/v1/hisparse/` | 稀疏注意力场景下的分层 KV（自有 block pool / coordinator / layout），面向超长上下文 |
| `vllm/v1/kv_hints/` | KV 位置提示（hint）机制，辅助跨层查找 |
| `vllm/v1/watermarking/` | 显存水位控制，防止分配过满导致抢占 |

**存储层级全景**：

```
GPU 显存（BlockPool 管理的 KV blocks）
   ↓ 满了/冷了 →  kv_offload/cpu/          CPU DRAM 卸载区
       ↓ 更冷 →   kv_offload/tiering/fs/   本地 SSD / 文件系统
                  kv_offload/tiering/obj/  对象存储（远端）
                  kv_offload/tiering/p2p/  其他实例的远端 KV
```

### 2.4 连接器（Connector）体系

| 路径 | 作用 |
|---|---|
| `vllm/v1/connectors/`（若存在则在此，否则分散在各后端目录） | KVConnector 接口实现，如 OffloadingConnector、NIXL、LMCache；分"调度侧"（生成 load/store 任务）与"执行侧"（真搬数据） |
| `vllm/v1/worker/kv_connector_model_runner_mixin.py` | 执行侧挂钩：在 forward 前后触发 KV 的 load/save |

### 2.5 其他重要目录（简述）

| 目录 | 作用 |
|---|---|
| `vllm/config/` | 所有配置对象（`CacheConfig`、`SchedulerConfig`、`KVTransferConfig` 等），调度与缓存行为的开关都在这 |
| `vllm/model_executor/` | 模型前向执行（layers、采样器） |
| `vllm/models/` | 各模型实现（LLaMA、QWen 等） |
| `vllm/inputs/`、`vllm/renderers/`、`vllm/tokenizers/`、`vllm/multimodal/` | 输入处理：分词、chat template、多模态预处理 |
| `vllm/entrypoints/` | OpenAI 兼容 API server、LLM 类入口 |
| `vllm/distributed/` | TP/PP/EP/DP 并行通信（KV 在 TP 下按 head 切分，与此相关） |
| `vllm/platforms/` | CUDA/ROCm/CPU/XPU 平台抽象 |
| `vllm/spec_decode/`（在 `v1/spec_decode/`） | 投机解码，调度与 KV 预算的特殊逻辑 |
| `vllm/compilation/`、`vllm/kernels/`、`vllm/triton_utils/`、`vllm/_custom_ops.py` | 编译与 kernel |
| `vllm/v1/metrics/`、`vllm/v1/sample/`、`vllm/v1/structured_output/` | 指标、采样、结构化输出（受控生成） |
| `vllm/lora/` | LoRA 适配器管理 |

---

## 3. 一次请求的完整链路（把上面串起来）

1. **进入**：`v1/engine/async_llm.py` 接收请求 → `input_processor.py` 分词 → 创建 `v1/request.py` 的 `Request`，状态为 WAITING。
2. **调度**：`v1/core/sched/scheduler.py` 每轮从 `request_queue.py` 挑请求；先向 `kv_cache_manager.py` 申请本轮 token 数对应的 block；`block_pool.py` 尝试按前缀哈希命中已有块（prefix caching 命中则零拷贝复用）。分配失败/显存不足时可能抢占低优先级请求，或等下一轮。
3. **下发**：调度器生成 `SchedulerOutput`（含 block table、采样参数、Connector 的 load/store 任务），经 `v1/executor/` 发给各 GPU worker。
4. **执行**：`v1/worker/gpu_model_runner.py` 准备输入批次 → KV Connector mixin 触发从 CPU/远端加载历史 KV（如需）→ attention kernel 按 block table 读写 GPU KV 张量 → 采样出下一个 token。
5. **回传**：token 经 `v1/engine/output_processor.py` + `detokenizer.py` 流式返回；调度器更新请求状态，重复第 2 步直到完成。
6. **释放与淘汰**：请求结束后 BlockPool 引用计数归零，块回到空闲池（内容仍在，可被前缀复用）；显存吃紧时按淘汰策略把冷块经 `kv_offload/` 体系搬到 CPU 或更低层级。

---

## 4. 初学者推荐阅读顺序

1. `kv_cache_architecture.md`（仓库根目录，中文，最贴切的入门材料）
2. `vllm/v1/core/sched/scheduler.py` —— 调度主循环
3. `vllm/v1/core/kv_cache_manager.py` → `block_pool.py` —— 块分配与前缀复用
4. `vllm/v1/kv_cache_interface.py` —— KV 布局抽象
5. `vllm/simple_kv_offload/` → `vllm/v1/kv_offload/` —— 由简到繁理解存储层级
6. `vllm/v1/worker/gpu_model_runner.py` —— 执行侧如何消费 block table
7. 配套测试：`tests/v1/core/`、`tests/v1/kv_offload/`
