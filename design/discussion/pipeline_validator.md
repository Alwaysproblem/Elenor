Pipeline Validator 架构与硬件可实现性分析

1.  总体架构

### 1.1 仓库结构

```
pipeline_validator/
├── cli.py                 # 命令行入口（258 行）
├── config.py              # 硬件配置 + 仿真配置（430 行）
├── hardware_config.yaml   # 硬件参数默认值（单一事实来源）
├── simulator.py           # 顶层仿真器（428 行）
├── tile_group.py          # Tile Group：1 Group Sequencer + 4 Compute Tile（1917 行）
├── tile_group_sequencer.py# Group Sequencer 控制器（339 行）
├── tile.py                # Compute Tile + Tile UCE + 4 引擎（1158 行）
├── engines.py             # BOA/EVU/MFE/USE 时序模型（442 行）
├── execution_ir.py        # 运行层 DTO（331 行）
├── ir_lowering.py         # IR → 运行层 lowering（692 行）
├── pmu.py                 # PMU 计数器 + stall 归因（120 行）
├── trace.py               # Perfetto/Chrome trace 生成（396 行）
├── report.py              # 报告生成（280 行）
├── stream_queue.py        # Stream Queue 模型（288 行）
├── workload_ir.py         # IR 解析 + 静态验证（677 行）
├── workload_builders.py   # 工作负载构造器
├── workloads.py           # 内置工作负载
├── dialects/
│   └── elenor.py           # xDSL 方言定义（1793 行）
├── memory/
│   ├── allocator.py        # 确定性 free-extent 分配器（552 行）
│   ├── l2_sram.py          # Group L2 SRAM 封装
│   ├── l1_slot_frame.py    # L1 Slot Frame（16-slot ABI + shadow bind FSM）
│   ├── hbm_region.py       # HBM 外部绑定注册
│   ├── transfer.py         # 传输管理器（逐腿路由 + 银行仲裁）
│   ├── noc.py              # NoC 路由器（4 虚通道 + 信用 + 仲裁）
│   └── payload.py          # Payload tracker
├── runtime/
│   ├── device_runtime.py   # Device Runtime（graph schedule lookup）
│   ├── event_table.py      # Event Table
│   ├── fault_ring.py       # Fault Ring（fault record 存储）
│   ├── firmware.py         # Firmware 层
│   ├── host_runtime.py     # Host Runtime
│   ├── kernel_driver.py    # Kernel Driver
│   ├── program_table.py    # Program Residency Manager
│   └── reset_domain.py     # Reset/Drain FSM
├── package/
│   └── package_ir.py       # 可执行包 IR
└── tests/
   ├── test_runtime.py     # 运行层测试（2489 行，85+ 测试）
   ├── test_validator.py   # 验证器测试（1711 行）
   └── test_memory_invariants.py # 内存不变量测试（911 行，52 测试）
```

总计约 18200 行 Python 代码。

### 1.2 分层模型

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ CLI (cli.py)                                                                  │
│ ├─ --ir-file                                                                  │
│ ├─ --workload                                                                 │
│ └─ --hw-override                                                              │
├───────────────────────────────────────────────────────────────────────────────┤
│ Simulator (simulator.py)                                                      │
│ ├─ run()        → standalone workload                                         │
│ └─ _run_model() → model mode (nexus.program)                                  │
├───────────────────────────────────────────────────────────────────────────────┤
│ TileGroup (tile_group.py)                                                     │
│ ├─ Group Sequencer (tile_group_sequencer.py)                                  │
│ ├─ 4 × ComputeTile (tile.py)                                                  │
│ │  ├─ Tile UCE (1 inst/cycle 控制器)                                          │
│ │  ├─ BOA Engine (dense compute)                                              │
│ │  ├─ EVU Engine (vector compute)                                             │
│ │  ├─ MFE Engine (memory flow)                                                │
│ │  ├─ USE Engine (state/control)                                              │
│ │  ├─ L1 SRAM (BankedFreeExtentAllocator)                                     │
│ │  └─ L1 Slot Frame (16-slot shadow bind)                                     │
│ ├─ Group L2 SRAM (BankedFreeExtentAllocator)                                  │
│ ├─ Transfer Manager (逐腿路由)                                                │
│ ├─ NoC Router (4-VC + credit)                                                 │
│ ├─ Stream Queues                                                              │
│ ├─ Collective Engine                                                          │
│ ├─ Fault Ring + Reset Domain                                                  │
│ ├─ Program Residency Manager                                                  │
│ └─ PMU Counter                                                                │
├───────────────────────────────────────────────────────────────────────────────┤
│ IR (dialects/elenor.py)                                                       │
│ ├─ nest.context / nest.dispatch / nest.release                                │
│ ├─ tile.program / tile.load / tile.store / tile.signal                        │
│ ├─ nexus.program / nexus.submit_context.async                                 │
│ └─ nest.alloc / nest.subview / nest.dma                                       │
├───────────────────────────────────────────────────────────────────────────────┤
│ Memory (memory/)                                                              │
│ ├─ Allocator (free-extent + owner/gen/pin)                                    │
│ ├─ HBM Region (external binding)                                              │
│ ├─ L2 SRAM (group SRAM)                                                       │
│ ├─ L1 Slot Frame (tile-local)                                                 │
│ ├─ Transfer (multi-leg routing)                                               │
│ ├─ NoC (virtual channel router)                                               │
│ └─ Payload Tracker                                                            │
├───────────────────────────────────────────────────────────────────────────────┤
│ Runtime (runtime/)                                                            │
│ ├─ Device Runtime                                                             │
│ ├─ Event Table                                                                │
│ ├─ Fault Ring                                                                 │
│ ├─ Reset Domain (drain FSM)                                                   │
│ ├─ Program Residency Manager                                                  │
│ ├─ Firmware                                                                   │
│ ├─ Host Runtime                                                               │
│ └─ Kernel Driver                                                              │
└───────────────────────────────────────────────────────────────────────────────┘
```

────────────────────────────────────────────────────────────────────────────────

2.  仿真精度层级

| 层级 | 名称          | 分配器                   | 传输路由              | NoC         | trace               |
| ---- | ------------- | ------------------------ | --------------------- | ----------- | ------------------- |
| 0    | `timing_only` | 无                       | 折叠单腿              | 无          | cycle 级时间线      |
| 1    | `runtime`     | 真实 L1/L2 + owner/gen   | 折叠单腿              | 无          | + 资源生命周期      |
| 2    | `full_memory` | 真实 L1/L2 + HBM binding | 逐腿（HBM→NoC→L2→L1） | 4-VC 路由器 | + 逐腿带宽/银行争用 |

三层共享同一 TileGroup 实例，区别仅在于 TransferManager 是否构建逐腿路由以及 NoCRouter 是否激活。

────────────────────────────────────────────────────────────────────────────────

3.  硬件配置（hardware_config.yaml）

所有硬件参数从 YAML 加载，Python HardwareConfig 是冻结 dataclass：

| 参数组              | 关键字段                | 默认值           | 冻结状态       |
| ------------------- | ----------------------- | ---------------- | -------------- |
| `system`            | `num_tiles`             | 4                | 冻结           |
| `clock`             | `core_mhz`              | 1000             | 建模基线       |
| `memory.hbm`        | `capacity_bytes`        | 16 GB            | 由后续规格冻结 |
| `memory.hbm`        | `bandwidth_gbs`         | 819.2 (8 stacks) | 由后续规格冻结 |
| `memory.group_sram` | `capacity_bytes`        | 8 MB             | 冻结           |
| `memory.group_sram` | `banks`                 | 16               | 冻结           |
| `memory.tile_l1`    | `capacity_bytes`        | 1 MB             | 冻结           |
| `memory.tile_l1`    | `banks`                 | 16               | 冻结           |
| `engines.boa`       | `num_opa × rows × cols` | 4×16×16          | 冻结           |
| `engines.evu`       | `lanes`                 | 32               | 冻结           |
| `engines.mfe`       | `bandwidth_gbs`         | 256.0            | 冻结           |
| `engines.use`       | `clock_mhz`             | 500              | 冻结           |
| `fabric.dma`        | `channels`              | 2                | 冻结           |
| `fabric.noc`        | `vc_depth`              | 8                | 由 PPA 冻结    |

--hw-override 命令行可覆盖任意 YAML 参数。

────────────────────────────────────────────────────────────────────────────────

4.  IR 方言（dialects/elenor.py）

### 4.1 三层前缀

| 前缀      | 层级           | 典型 op                                                                                                                                                                                                      |
| --------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tile._`  | Tile 级        | `tile.program`, `tile.load.async`, `tile.store.async`, `tile.pow.async`, `tile.signal`, `tile.await`, `tile.alloc`, `tile.subview`, `tile.return`                                                            |
| `nest._`  | Group 级       | `nest.context`, `nest.alloc`, `nest.subview`, `nest.dma.prefetch.async`, `nest.dma.store.async`, `nest.dispatch.tasks.async`, `nest.release`, `nest.await`, `nest.task.range`, `nest.barrier`, `nest.return` |
| `nexus.*` | Host/Device 级 | `nexus.program`, `nexus.submit_context.async`, `nexus.await`, `nexus.return`                                                                                                                                 |

### 4.2 核心类型

| 类型                               | 含义                                    |
| ---------------------------------- | --------------------------------------- |
| `!nest.event<"tag">`               | Group 级异步事件，tag 是运行时 event id |
| `!nest.l2_buffer<Dx...xDtype>`     | L2 buffer，带 slot 名                   |
| `!nest.global_memref<Dx...xDtype>` | HBM 全局输入                            |
| `!nest.global_view<Dx...xDtype>`   | 全局视图（subview 结果）                |
| `!nest.l2_view<Dx...xDtype>`       | L2 视图                                 |
| `!tile.event<"tag">`               | Tile 级异步事件                         |
| `!tile.l1_buffer<Dx...xDtype>`     | L1 buffer                               |
| `!nest.task`                       | 逻辑任务句柄                            |

### 4.3 IR 示例

```mlir
// Tile 程序：定义 tile-local kernel
tile.program @pow_kernel (%task: !nest.task, %in_buf: !nest.l2_buffer<4x128x128xbf16>) {
 %0 = tile.subview %in_buf task = %task task_dim = 0
     offsets = [0,0,0] sizes = [1,128,128] strides = [1,1,1]
     : !nest.l2_view<1x128x128xbf16>
 %1 = tile.alloc shape = [128, 128] dtype = "bf16" alignment = 256
     : !tile.l1_buffer<128x128xbf16>
 %e_load = tile.load.async %0 into %1 : !tile.event<"e_load">
 tile.await %e_load
 tile.signal input_released(%task)
 %e_pow = tile.pow.async bytes = 32768 exponent = 2 pow_ops = 65536
     : !tile.event<"e_pow">
 tile.await %e_pow
 tile.return
}

// Group 上下文：定义 Group 级控制流
nest.context @pow_task (%Y : !nest.global_memref<4x128x128xbf16>) placement = 15 {
 %l2_buf = nest.alloc slot = "l2_buf" role = "inout" shape = [4,128,128]
     dtype = "bf16" alignment = 256 : !nest.l2_buffer<4x128x128xbf16>
 %src = nest.subview %Y offsets = [0,0,0] sizes = [4,128,128]
     strides = [1,1,1] : !nest.global_view<4x128x128xbf16>
 %ev_dma = nest.dma.prefetch.async %src into %l2_buf : !nest.event<"ev_dma">
 %0 = nest.task.range from = 0 to = 4 : !nest.task_range
 %ev_grid, %ev_inrel, %ev_outready =
     nest.dispatch.tasks.async @pow_kernel tasks(%0) ins(%l2_buf) outs(%l2_buf)
     signal_policy { input_released = #nest.aggregate<all_tasks>,
                     output_ready = #nest.aggregate<all_tasks> }
     depends_on(%ev_dma)
     : (!nest.event<"ev_grid">, !nest.event<"ev_inrel">, !nest.event<"ev_outready">)
 nest.release %l2_buf depends_on(%ev_inrel)
 nest.await %ev_grid
 nest.return
}

// Model 程序：device 级调度
nexus.program @run (%Y : !nest.global_memref<4x128x128xbf16>) {
 %done = nexus.submit_context.async @pow_task(%Y) : !nexus.event<"done">
 nexus.await %done
 nexus.return
}
```

────────────────────────────────────────────────────────────────────────────────

5.  控制流模型

### 5.1 三级控制流

```
Graph Schedule (Host)
 → Group Task (Device Runtime)
   → Tile Group Sequencer (Group 级)
     → Tile UCE (Tile 级, 1 inst/cycle)
       → BOA / EVU / MFE / USE 引擎
```

Group Sequencer（tile_group_sequencer.py）：

- 逐 cycle 执行 ExecGroupAction 列表（线性 PC，无分支）
- 动作类型：INIT_STREAM, DMA_PREFETCH, DMA_STORE, DISPATCH_ROLE, WAIT_EVENT, SIGNAL_EVENT, RELEASE_L2, BARRIER_GROUP, COLLECTIVE_RUN
- 每个 action 可能等待前置 event 完成才 issue

Tile UCE（tile.py:TileUCE）：

- 1..MAX_CONTEXT_COUNT 个执行上下文
- 每周期 issue 1 条 ExecTileInst
- 指令类型：LAUNCH_BOA, LAUNCH_EVU, LAUNCH_MFE, LAUNCH_USE, WAIT, WAITALL, SIGNAL_PHASE, STREAM_POP/PUSH/ACQUIRE/RELEASE, FENCE, RET
- 上下文切换：round-robin 选择 READY 上下文

### 5.2 事件驱动

- nest.event<"tag"> / tile.event<"tag"> 的 tag 是运行时 event id
- await 阻塞直到所有 event 完成
- tile.signal <phase>(%task) 触发 group 级 phase 聚合
- Phase 聚合键：(launch_generation, grid_instance_id, signal_phase, logical_task_id)
- input_released 和 output_ready 事件在所有逻辑 task 信号后 fire 一次

────────────────────────────────────────────────────────────────────────────────

6.  内存层次模型

### 6.1 三级内存

| 层级            | 容量  | 分配器                         | 特征                                  |
| --------------- | ----- | ------------------------------ | ------------------------------------- |
| HBM             | 16 GB | `HBMRegion` (external binding) | 全局输入绑定，带宽 819.2 GB/s         |
| L2 (Group SRAM) | 8 MB  | `BankedFreeExtentAllocator`    | 16 bank，确定性 first-fit free-extent |
| L1 (Tile SRAM)  | 1 MB  | `BankedFreeExtentAllocator`    | 16 bank，per-tile，16-slot frame ABI  |

### 6.2 分配器（memory/allocator.py）

BankedFreeExtentAllocator 是核心分配器，同时用于 L2 和 L1：

- 确定性 first-fit：request 顺序、bank id 升序、extent 起始升序
- owner / generation / pin 生命周期：
  - ContextBufferOwner(name, generation, buffer_id) — L2 context buffer
  - TaskBufferOwner(name, generation, event, task, tile, context, buffer) — L1 task buffer
  - ExternalOwner(binding_name) — HBM 外部绑定
- Admission 分类（PR 3.5）：
  - INVALID_REQUEST — size/alignment 非法
  - PERMANENT_CAPACITY — 空 pool 也无法容纳
  - TEMPORARY_CAPACITY — 合法但当前 free map 不满足 → 进入 admission wait
- 不变量：失败的 plan 零副作用（不改 free map / pool version / counter / peak）

### 6.3 传输模型（memory/transfer.py）

TransferManager 构建逐腿路由，每腿是独立 TransferStage：

| 操作                    | 路由（`full_memory`）                           |
| ----------------------- | ----------------------------------------------- |
| `PREFETCH` (HBM→L2)     | HBM_READ → GLOBAL_DMA → NOC_RESPONSE → L2_WRITE |
| `GLOBAL_STORE` (L2→HBM) | L2_READ → NOC_REQUEST → GLOBAL_DMA → HBM_WRITE  |
| `TILE_LOAD` (L2→L1)     | L2_READ → LOCAL_DMA → L1_WRITE                  |
| `TILE_STORE` (L1→L2)    | L1_READ → LOCAL_DMA → L2_WRITE                  |

每腿有带宽（bytes/cycle）和固定延迟；银行仲裁（同 bank 串行，不同 bank 并行）。

### 6.4 NoC 路由器（memory/noc.py）

- 4 虚通道：VC0(command/event), VC1(DMA read rsp), VC2(DMA write), VC3(collective)
- 信用制：每个 VC 有 credit_available，发送消耗信用
- 优先级仲裁：VC0 优先，starvation 保护
- V1 简化：单路由器，无 mesh 拓扑

### 6.5 L1 Slot Frame（memory/l1_slot_frame.py）

- 16 个固定 slot（ABI 不变）
- Shadow-install 机制：prepare → bind → activate
- Frame bind FSM：8 状态（IDLE → FETCH → VALIDATE → CHECK → INSTALL → ACTIVE）
- Bank policy 强制

────────────────────────────────────────────────────────────────────────────────

7.  引擎模型

| 引擎 | 类          | 延迟模型                                  | 流水                                               |
| ---- | ----------- | ----------------------------------------- | -------------------------------------------------- |
| BOA  | `BOAEngine` | `launch_cycles + ceil(ops / peak_macs)`   | 非流水（1 job/tile）                               |
| EVU  | `EVUEngine` | `launch_cycles + ceil(ops / (lanes × 2))` | 非流水                                             |
| MFE  | `MFEEngine` | 带宽受限 + channelized                    | `mfe_load_channels + mfe_store_channels` 并行 lane |
| USE  | `USEEngine` | `launch_cycles + ceil(ops × clock_ratio)` | 非流水                                             |

MFE 特殊性：lane 分 load/store 两类，每类 first-free 分配。Lane head 提交 MemoryTransaction 到 TransferManager，engine poll 完成状态。Lane depth = mfe_pipeline_depth（描述符接收队列深度）。

────────────────────────────────────────────────────────────────────────────────

8.  L2 Admission Wait（PR 3.5）

### 8.1 状态机

```
SUBMITTED → PREPARE → ADMISSION_TRY
 ├─ ADMITTED → ACTIVATE → DONE/FAULT
 ├─ WAIT_CAPACITY → ADMISSION_WAIT (FIFO)
 │                    ↓ release_l2 final-free
 │                    ADMITTED → ACTIVATE
 └─ FAULT (INVALID/PERMANENT)
```

### 8.2 关键行为

- try_admit_l2_buffers(task, \*, context_name, launch_generation, cycle) -> L2AdmissionOutcome
- WAIT_CAPACITY sequencer 的 done=False, faulted=False；不占 UCE context、L1 frame、StreamQueue、DMA/engine work、L2 handle
- 唯一 wakeup 源：release_l2() 的 allocator final-free
- strict FIFO：队首暂时不 fit 时不检查后续 ticket
- l2_admission_wait_cycles 在 ticket admitted/cancelled 时一次性累加 terminal_cycle - enqueue_cycle

### 8.3 PMU 计数器

| 计数器                         | 含义                 |
| ------------------------------ | -------------------- |
| `l2_admission_wait`            | 进入 WAIT 的事件数   |
| `l2_admission_retry`           | FIFO retry 次数      |
| `l2_admission_wakeup`          | WAIT→ACTIVE 转换数   |
| `l2_admission_wait_cycles`     | 等待周期累计         |
| `l2_admission_queue_peak`      | 队列峰值             |
| `l2_admission_permanent_fault` | 永久 admission fault |

────────────────────────────────────────────────────────────────────────────────

9.  Fault / Reset 模型

### 9.1 Fault Ring（runtime/fault_ring.py）

- 固定大小环形缓冲区
- FaultRecord：abi_version, code, source, severity, fault_record_index, context_id, ...
- 11 种 fault code（INVALID_DESCRIPTOR 到 SLOT_PERMISSION_FAULT）

### 9.2 Reset Domain（runtime/reset_domain.py）

- 9 状态 FSM：IDLE → FAULT_DETECTED → STOP_QUEUE → FREEZE_DISPATCH → DRAIN_SAFE → MARK_EVENTS → RESET_DOMAIN → CLEAR_CREDIT_CACHE → RESUME → DONE
- 4 个 fault domain：QUEUE, TILE, GROUP, DEVICE
- Reset 期间冻结新 work，drain 进行中的 transfer/engine

────────────────────────────────────────────────────────────────────────────────

10. Program Residency

ProgramResidencyManager（runtime/program_table.py）跟踪 program 在 tile 上的驻留状态：

| 状态            | 含义                              |
| --------------- | --------------------------------- |
| `HBM_ONLY`      | program 在 HBM，未安装到任何 tile |
| `FETCHING`      | Group/Tile DMA 正在传输           |
| `TILE_RESIDENT` | 已安装到 tile-local program SRAM  |
| `EVICTED`       | slot 被回收，需要重新 fetch       |

- Cold launch = residency miss → 隐式 fetch/verify/install
- Warm launch = program resident → patch descriptor/context/shape only
- 正确性由 program_id + version + hash + epoch 保证

────────────────────────────────────────────────────────────────────────────────

11. 工作负载

内置工作负载（workloads.py + workload_builders.py）：

| 工作负载         | 描述                    | 引擎使用              |
| ---------------- | ----------------------- | --------------------- |
| `PowWorkload`    | 元素级 pow（GEMM-like） | BOA + MFE             |
| `DenseAttention` | 密集 attention          | BOA + EVU + MFE       |
| `PagedAttention` | 分页 KV cache attention | BOA + EVU + MFE + USE |
| `MoE`            | Mixture of Experts      | BOA + EVU + MFE + USE |
| `SSM/Mamba`      | 状态空间模型            | BOA + EVU + USE       |
| `MultiModel`     | 多模型并发              | 全引擎                |

────────────────────────────────────────────────────────────────────────────────

12. 硬件可实现性评估

### 12.1 可直接硬件实现的部分

| 组件                       | 可实现性 | 依据                                       |
| -------------------------- | -------- | ------------------------------------------ |
| BOA (Block Outer-product)  | 高       | 4 OPA × 16×16 外积 tile，MAC 阵列标准 RTL  |
| EVU (Enhanced Vector Unit) | 高       | 32-lane 向量 FMA，类似 GPU SIMT lane       |
| MFE (Memory Flow Engine)   | 高       | DMA channel + descriptor queue，标准 RTL   |
| L2 SRAM (Group SRAM)       | 高       | 16-bank SRAM + free-extent 管理器          |
| L1 SRAM (Tile SRAM)        | 高       | 1 MB 16-bank + 16-slot frame               |
| NoC Router                 | 高       | 4-VC + credit-based flow control，工业标准 |
| Stream Queue               | 高       | FIFO + credit + EOS，标准 RTL              |
| Tile UCE                   | 高       | 1 inst/cycle 微控制器（RISC-V 级）         |
| Group Sequencer            | 高       | 线性 PC + event-wait，简单 FSM             |
| Program Residency          | 中高     | 需要 program SRAM 管理 + DMA 加载          |
| Slot Frame                 | 中高     | 16-slot ABI + shadow-install FSM           |
| Fault Ring                 | 中       | 环形缓冲 + fault record struct             |
| Reset Domain               | 中       | 9 状态 FSM，需要 drain 逻辑                |

### 12.2 需要进一步规格化的部分

| 组件              | 当前状态                  | 阻塞项                             |
| ----------------- | ------------------------- | ---------------------------------- |
| USE Engine        | 时序模型存在，无 datapath | scan/recurrence datapath 未规格化  |
| Collective Engine | 1-cycle command 模型      | reduce/broadcast datapath 未规格化 |
| HBM Controller    | 带宽 + outstanding 模型   | channel mapping 冻结               |
| L2 Bank 仲裁      | first-fit + bank 模型     | bank policy 编码未冻结             |
| Descriptor Patch  | PATCH_DESC 占位           | patch FSM 细节未规格化             |
| Package Binary    | IR 定义存在               | 二进制 layout 未冻结               |
| Runtime ABI       | v0 draft                  | command/event ABI 未冻结           |

### 12.3 架构与硬件的映射关系

```
Validator 代码模块          →  硬件实现
─────────────────────────────────────────────
BankedFreeExtentAllocator  →  L2/L1 bank 仲裁器 + free-extent 管理
TransferManager            →  Group DMA 控制器 + tile-local DMA 控制器
NoCRouter                  →  NoC 路由器 RTL
StreamQueue                →  Stream Queue FIFO + credit 逻辑
TileUCE                    →  Tile UCE 微控制器 (RISC-V)
TileGroupSequencer         →  Group Sequencer 控制器
BOAEngine                  →  BOA MAC 阵列
EVUEngine                  →  EVU 向量单元
MFEEngine                  →  MFE DMA channel + descriptor queue
USEEngine                  →  USE 状态引擎
SlotFrame                  →  L1 Slot Frame FSM
FaultRing                  →  Fault Ring 硬件
ResetDomain                →  Reset/Drain FSM
ProgramResidencyManager    →  Program SRAM 管理 + DMA 加载
HBMRegion                  →  HBM Controller
```

### 12.4 结论

当前 validator 的架构和程序逻辑可以直接映射到硬件实现。

核心引擎（BOA、EVU、MFE）、内存层次（HBM→L2→L1）、控制流（Group Sequencer → Tile UCE）、通信（NoC + Stream Queue）都有明确的 RTL 对应物。validator 的 cycle-accurate
时序模型（引擎延迟、带宽、银行争用）直接来自硬件规格参数，不是抽象近似。

阻塞硬件实现的不是 validator 本身，而是以下未冻结的规格：

1.  USE Engine datapath — scan/recurrence 的具体运算单元未定义
2.  Collective Engine datapath — reduce/broadcast 的数据通路未定义
3.  Descriptor patch FSM — 运行时 descriptor 修补的具体状态机未规格化
4.  Package binary layout — 可执行包的二进制格式未冻结
5.  Runtime ABI — command/event 的二进制 ABI 未冻结

这些是架构文档（design/ELENOR_Architecture_Design_v1.md）中已知的 P0 阻塞项，不影响 validator 作为 cycle-accurate 性能模型的有效性。

────────────────────────────────────────────────────────────────────────────────

13. 调度模型

### 13.1 总体调度层级

```
nexus.program body (Device PC loop)
 │  submit / await / return  — 线性 PC，每个 cycle 推进 0..N 条
 │
 ├─ Device slot 分配
 │   pin != None → slot = pin if !busy else None
 │   pin == None → slot = first-free(range(count))
 │   slot None   → device_submit_wait (stall, 下个 cycle 重试)
 │
 ├─ load_context_task(slot)
 │   prepare → admit → activate (或 WAIT_CAPACITY)
 │
 └─ group.step(cycle) — 单 TileGroup lockstep 推进
     │
     ├─ 1. NoC router step + TransferManager step
     │     完成 transaction → notify sequencer event + forward to tiles
     │
     ├─ 1b. Collective jobs 到期 → notify event
     │
     ├─ 2. Stream queue tick (PMU occupancy + trace counter)
     │
     ├─ 3. Group Sequencer step (if !freeze)
     │     for seq in _active_sequencers:
     │       seq.step(cycle)  — 线性 PC，1 action/cycle
     │     _retry_pending_context_admissions(cycle)  — PR 3.5 FIFO 唤醒
     │
     ├─ 3b. activate staged admissions
     │     for ticket in _pending_activations:
     │       _activate_admitted_context(ticket, cycle)
     │     _pending_activations.clear()
     │
     ├─ 4. Tile step (4 tiles lockstep)
     │     for t in tiles:
     │       t.step(cycle, freeze_new_work)
     │         ├─ engine tick (BOA/EVU/MFE/USE)
     │         │   完成 → notify UCE event
     │         └─ UCE step
     │             ├─ context select (round-robin READY)
     │             ├─ issue 1 inst/cycle
     │             │   LAUNCH_* / WAIT / SIGNAL_PHASE / STREAM_* / ...
     │             └─ terminal drain (DONE/FAULT)
     │       drain_context_terminals()
     │         done  → release L1 + notify group sequencer
     │         fault → fault issuing sequencer + trigger_reset
     │
     ├─ 5. aggregate PMU
     │
     ├─ 5b. reset/drain FSM step (if active)
     │
     └─ 6. prune done sequencers + reclaim queue IDs
         all_done = (_active_sequencers 空 && _pending 空 && _pending_activations 空)
```

### 13.2 Device 级调度（Simulator.\_run_model）

**单 PC 线性推进**：model body 是 `submit → await → ... → return` 的线性列表，Device PC
每个 cycle 向前推进 0..N 条：

- **submit**：选 slot（pin 固定 or first-free），调用 `load_context_task`。
  slot 全忙 → `device_submit_wait` stall，PC 不推进，下个 cycle 重试。
- **await**：检查 event_tag 是否在 `done_events` 中。
  未完成 → `device_await_wait` stall，PC 不推进，下个 cycle 重试。
  完成 → PC++。
- **return**：设置 `returned = True`，PC++。

**终止条件**：`returned && !any(slot_busy)`。

**多 context 并发**：同一 cycle 内 PC 可以连续推进多条 submit（只要 slot 有空），
不等待前一个 context 完成。这是 PR 3.5 overlap 的前提——两个 submit 在同一 cycle
提交到两个 slot。

### 13.3 Group Sequencer 调度（TileGroupSequencer.step）

每个 active sequencer 每 cycle 执行 1 条 `ExecGroupAction`：

```
step(cycle):
  if done or task is None → idle
  if _pending (waiting for events):
    if all events done → resolve _pending, return (wait_resolved)
    else → stall (WAIT_EVENT), return
  if action_index >= len(actions):
    if all roles_done && all phases_done && outstanding_jobs == 0:
      done = True
    else → drain_wait
    return
  issue action_index → _issue(ins, cycle)
    action_index++ (if action committed)
```

**动作执行规则**：

| 动作             | 行为                    | 推进条件                               |
| ---------------- | ----------------------- | -------------------------------------- |
| `INIT_STREAM`    | 创建 StreamQueue        | 总是推进                               |
| `DMA_PREFETCH`   | 提交 MemoryTransaction  | 总是推进（失败 → fault）               |
| `DMA_STORE`      | 提交 MemoryTransaction  | 总是推进（失败 → fault）               |
| `DISPATCH_ROLE`  | 原子 dispatch 到 4 tile | `can_dispatch_role` 全部 tile UCE 可用 |
| `WAIT_EVENT`     | 设 \_pending            | 总是推进（进入等待态）                 |
| `SIGNAL_EVENT`   | 标记 event done         | 总是推进                               |
| `RELEASE_L2`     | 释放 L2 buffer          | dependency_events 全部完成             |
| `BARRIER_GROUP`  | 屏障                    | 总是推进                               |
| `COLLECTIVE_RUN` | 提交 collective job     | 总是推进                               |

**stall 类型**：`WAIT_EVENT`（等待前置事件）、`dispatch_wait`（UCE context 不可用）、
`drain_wait`（所有 action 已 issue，等待 role/job 完成）。

### 13.4 Tile UCE 调度（TileUCE.step）

每个 tile 有 1..MAX_CONTEXT_COUNT 个执行上下文，每 cycle issue 1 条指令：

```
step(cycle, tile, freeze_new_work):
  _drain_completion_queue()  — 处理刚完成的 engine event
  _ensure_context_traces(cycle)

  current = _select_context(cycle)
    if current.state == READY → 选 current
    else → round-robin 找下一个 READY context
    找到 → 切换 (_switch_context, PMU uce_context_switch)
    找不到 → 尝试 WAIT_STREAM context retry

  if current == None:
    idle → return

  _advance_accept(ctx, cycle)  — prepare cycles 倒数
  _advance_frame_bind(ctx, cycle)  — frame bind FSM
  if ctx.state == READY:
    _issue_context(ctx, cycle)
      issue ctx.program.insts[ctx.pc]
      pc++ (if instruction committed)
  elif ctx.state == WAIT_STREAM:
    retry stream pop/acquire

  _sample_context_counters(cycle)
``+
**上下文状态机**：

```

EMPTY → ACCEPT → FRAME_BIND → READY → WAIT_EVENT → READY → ... → DONE → EMPTY
↑ ↓
WAIT_STREAM ──────┘
``+
**上下文选择**：round-robin，从 `\_current_ctx` 开始扫描，选第一个 READY 上下文。
如果当前上下文 READY 则直接用（不切换）。WAIT_STREAM 上下文可以 retry stream pop。

**指令 issue**：

| 指令                 | 行为                                 | 推进条件                 |
| -------------------- | ------------------------------------ | ------------------------ |
| `LAUNCH_BOA/EVU/USE` | 提交 descriptor 到 engine            | engine `is_busy` → stall |
| `LAUNCH_MFE`         | 路由到 first-free load/store lane    | 全部 lane 满 → stall     |
| `WAIT`               | 设 wait_refs，ctx → WAIT_EVENT       | event done → 回到 READY  |
| `WAITALL`            | 设 wait_refs (all)，ctx → WAIT_EVENT | 全部 event done → READY  |
| `SIGNAL_PHASE`       | 调用 `_phase_signal_callback`        | 总是推进                 |
| `STREAM_POP`         | 从 StreamQueue pop                   | queue 空 → WAIT_STREAM   |
| `STREAM_PUSH`        | 向 StreamQueue push                  | 总是推进                 |
| `STREAM_ACQUIRE`     | 获取 credit                          | credit 空 → WAIT_STREAM  |
| `FENCE`              | PMU fence cycle                      | 总是推进                 |
| `RET`                | ctx → DONE                           | 总是推进                 |

### 13.5 L2 Admission 调度（PR 3.5）

Context submit 时 `load_context_task` 走 prepare → admit → activate 三阶段：

```
load_context_task(slot):
  ticket = _prepare_context_launch(...)
    deep-clone task, namespace event/stream IDs, validate pin,
    create TileGroupSequencer + _PendingContextAdmission

  outcome = _try_admit_prepared_context(ticket, cycle)

  if WAIT_CAPACITY:
    sequencer.admission_status = WAIT_CAPACITY
    _enqueue_pending_admission(ticket)  — FIFO tail
    return sequencer  (slot 保持 busy，不占任何资源)

  if FAULT:
    sequencer.faulted/done = True
    return sequencer

  _activate_admitted_context(ticket, cycle)
    admission_status = ACTIVE
    register _live_launches
    append _active_sequencers
    init task streams
    return sequencer
``+
**retry 唤醒**：在 `group.step()` 的 sequencer loop 之后、tile step 之前的 barrier 调用
`_retry_pending_context_admissions(cycle)`：

- 只在 `release_l2()` final-free 设置了 `_l2_capacity_change_cycle` 后才触发
- pool version 自上次 retry 未变 → 跳过（无 busy-poll）
- strict FIFO：只处理队首，成功后继续处理下一 ticket
- 成功的 ticket 写入 `_pending_activations`，在当前 barrier 后 activate
- activated sequencer 的第一条 group action 最早在 `cycle + 1` issue

**调度不变量**：

- WAIT_CAPACITY sequencer 不在 `_active_sequencers` 中
- WAIT_CAPACITY sequencer 不在 `_live_launches` 中
- WAIT_CAPACITY sequencer 无 L2 handle / L1 handle / UCE context / stream / DMA
- `all_done = (_active_sequencers 空 && _pending 空 && _pending_activations 空)`

### 13.6 Dispatch Role 调度（原子两阶段）

`DISPATCH_ROLE` 动作调用 `TileGroup.dispatch_role`，分两阶段原子执行：

```

Phase 1 (plan, zero side effect):
for tile in selected_tiles:
context_id = tile.available_context_id(pin)
if None → fail (rollback all prior plans)
l1_plan = tile.l1_allocator.plan_bundle(l1_requests)
if AdmissionFailure → fail (rollback)
admissions.append(admission)

Phase 2 (commit + bind):
for admission in admissions:
tile.l1_allocator.commit(l1_plan)
tile.l1_frames[ctx_id].prepare(l1_handles)
\_pin_grid_l2(grid, binding, task_identity) — pin L2 consumer
for admission in admissions:
tile.load_program(prog, context_id, memory) — bind UCE context

# publish role bookkeeping

\_role_event_tile_mask[ev] = tile_mask
\_grid_signals[grid] = signal_state
\_role_trace[ev] = role_trace
``+
任何 Phase 2 失败 → rollback：unbind context + release L1 + unpin L2 + clear grid。
Issuing sequencer faulted，不在 `\_active_sequencers` 中残留。

### 13.7 调度时序图（单 cycle 内顺序）

```
Cycle N:
  ┌─ Device PC advance (if !fault)
  │   submit → load_context_task → activate (or WAIT)
  │   await → check done_events
  │   return → set returned
  │
  ├─ group.step(N):
  │   1. NoC.step + Transfer.step
  │      → completed txn → notify_event → forward to tiles
  │   1b. Collective jobs 到期 → notify
  │   2. Stream queue tick
  │   3. for seq in _active_sequencers: seq.step(N)
  │      → issue 1 action (prefetch/dispatch/wait/release/...)
  │   3. _retry_pending_context_admissions(N)
  │      → FIFO head retry → stage activation
  │   3b. for ticket in _pending_activations: activate
  │   4. for tile in tiles: tile.step(N)
  │      → engine tick → complete → notify UCE
  │      → UCE step → select context → issue 1 inst
  │      → terminal drain → release L1 → notify sequencer
  │   5. aggregate PMU
  │   5b. reset/drain FSM (if active)
  │   6. prune done sequencers
  │      all_done check
  │
  ├─ slot completion scan
  │   for slot in range(count):
  │     if slot_busy and seq.done → done_events.add(tag), slot free
  │     if slot_busy and seq.faulted → fault_reason = seq.fault_reason
  │
  └─ termination check
      if returned && !any(slot_busy) → model complete
      if fault → wait for reset_domain.is_done → break
```

### 13.8 调度公平性

| 层级              | 公平性                 | 机制                                           |
| ----------------- | ---------------------- | ---------------------------------------------- |
| Device slot       | first-free（无优先级） | `next(i for i in range(count) if not busy[i])` |
| Group Sequencer   | 无仲裁（每个独立 PC）  | 所有 active sequencer 每 cycle 各 step 1 次    |
| L2 admission wait | strict FIFO            | 队首不 fit 时不检查后续                        |
| Tile UCE context  | round-robin            | 从 current_ctx 开始扫描第一个 READY            |
| MFE lane          | first-free in class    | load/store 分开，各自 first-free               |
| HBM channel       | 地址哈希               | `(addr // burst_bytes) % channels`             |
| L2/L1 bank        | 确定性 first-fit       | bank_id 升序、extent 起始升序                  |
| NoC VC            | 优先级仲裁             | VC0 > VC1 > VC2 > VC3, starvation 保护         |
