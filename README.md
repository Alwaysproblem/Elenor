# ELenor

> 一个面向未来 AI 推理与轻训练场景的统一 AI 加速器架构设计仓库。README 以“由 AI 辅助设计的 AI 加速器”这一定位来组织内容，核心设计文档内统一使用 **ELENOR** 命名。

![ELENOR overview](./image/Elenor_v0.png)

## 项目简介

ELENOR 是一套面向未来 5 到 10 年 AI 工作负载的加速器架构设计，目标覆盖：

- Dense Transformer
- Paged Attention
- MoE
- SSM / Mamba / RWKV
- Dynamic Shape
- 多模型并发
- 边缘到数据中心的统一复用

这套架构的核心判断是：**未来 AI workload 不再只是 dense GEMM 问题**。因此，ELENOR 不走单一路线，而是把计算、控制和数据搬运明确拆开：

> **Compute != Control != Data Movement**

硬件执行边界也非常清晰：ELENOR 硬件消费的是 **command buffer、descriptor、Region Program 和 Tile Program**，而不是直接解释高层 graph。

## 核心设计理念

ELENOR 将能力拆成四类核心引擎：

| 子系统 | 职责 | 典型工作负载 |
| --- | --- | --- |
| **BOA** | Dense Compute | GEMM、Conv、QK、AV、Expert MLP |
| **EVU** | Irregular Compute | Softmax、Norm、RoPE、Activation、Gather/Scatter、Tail 处理 |
| **MFE** | Memory Flow | Page Stream、Segment Stream、Sparse/布局变换相关数据流 |
| **USE** | State / Control | Scan、Recurrence、Dynamic Shape Assist、Token Routing、Event Assist |

这意味着 ELENOR 不是单纯的矩阵乘芯片，而是一套围绕真实 AI 系统瓶颈设计的、**面向 dense + irregular + stateful + streaming** 的复合型加速器架构。

## 顶层架构概览

```text
Host / System SoC
        |
        v
+--------------------------------------------------------------------+
| ELENOR Device                                                      |
|                                                                    |
| Host Interface  <->  Runtime Processor  <->  Global PMU            |
|        |                    |                    |                  |
|        +------ Global Scheduler / Event Fabric / Command Queue -----+
|                             |                                       |
|               +-------------+-------------+                         |
|               |                           |                         |
|         Global DMA                Memory Controller                 |
|               |                           |                         |
|               +------------- NoC / Router -+------------------------|
|                                             \                       |
|                                              +--> Tile Group x N    |
+--------------------------------------------------------------------+
```

芯片级控制面、数据面和观测面共同闭环：

- **控制面**：doorbell、command queue、event、barrier、fault、interrupt
- **数据面**：HBM/DDR/LPDDR -> L2 -> L1 -> Engine
- **观测面**：PMU、stall attribution、bandwidth、fault capture

## 目标配置

| 配置 | Tile 数量 | Tile Group 数量 | Memory | 目标场景 |
| --- | ---: | ---: | --- | --- |
| Edge | 8–16 | 1–2 | LPDDR | 小模型推理、移动端、车端、小 batch |
| Balanced | 64 | 8 | HBM 或高带宽 DDR | LLM inference、Paged Attention、MoE、多模型并发 |
| High End | 128 | 16 | HBM | 数据中心长上下文推理、多模态、轻训练 |

## 仓库结构

```text
.
├── design/             # 架构设计主文档与分模块设计文档
├── image/              # 架构图、时序图、模块示意图
├── scripts/            # 文档导出脚本
├── gen_docs_config/    # PDF 导出模板
└── LICENSE
```

## 文档导航

### 总体架构

- [`design/ELENOR_Architecture_Design_v1.md`](./design/ELENOR_Architecture_Design_v1.md) — 总体架构、边界、执行模型、阶段目标
- [`design/elenor_chip_top/ELENOR_Chip_Top_Design.md`](./design/elenor_chip_top/ELENOR_Chip_Top_Design.md) — 芯片顶层集成、命令/事件/复位/PMU
- [`design/elenor_runtime_abi/ELENOR_Runtime_Command_Event_ABI_Design.md`](./design/elenor_runtime_abi/ELENOR_Runtime_Command_Event_ABI_Design.md) — 运行时命令与事件 ABI

### 计算与执行引擎

- [`design/elenor_boa/ELENOR_BOA_Design.md`](./design/elenor_boa/ELENOR_BOA_Design.md) — Dense Compute 主引擎
- [`design/elenor_evu/ELENOR_EVU_Design.md`](./design/elenor_evu/ELENOR_EVU_Design.md) — 向量/非规则计算引擎
- [`design/elenor_mfe/ELENOR_MFE_Design.md`](./design/elenor_mfe/ELENOR_MFE_Design.md) — Memory Flow Engine
- [`design/elenor_use/ELENOR_USE_Design.md`](./design/elenor_use/ELENOR_USE_Design.md) — Unified State Engine

### 片上组织与数据流

- [`design/elenor_tile_group/ELENOR_Tile_Group_Design.md`](./design/elenor_tile_group/ELENOR_Tile_Group_Design.md)
- [`design/elenor_compute_tile/ELENOR_Compute_Tile_Design.md`](./design/elenor_compute_tile/ELENOR_Compute_Tile_Design.md)
- [`design/elenor_memory_noc/ELENOR_Memory_NoC_Design.md`](./design/elenor_memory_noc/ELENOR_Memory_NoC_Design.md)
- [`design/elenor_global_dma/ELENOR_Global_DMA_Design.md`](./design/elenor_global_dma/ELENOR_Global_DMA_Design.md)
- [`design/elenor_collective/ELENOR_Collective_Engine_Design.md`](./design/elenor_collective/ELENOR_Collective_Engine_Design.md)

### 软件栈与系统支撑

- [`design/elenor_compiler/ELENOR_Compiler_Stack_Design.md`](./design/elenor_compiler/ELENOR_Compiler_Stack_Design.md) — 编译器 lowering、descriptor template、package 生成
- [`design/elenor_driver_firmware/ELENOR_Driver_Firmware_Runtime_Design.md`](./design/elenor_driver_firmware/ELENOR_Driver_Firmware_Runtime_Design.md) — driver / firmware / runtime 分工
- [`design/elenor_workload_mapping/ELENOR_Workload_Mapping_Design.md`](./design/elenor_workload_mapping/ELENOR_Workload_Mapping_Design.md) — 工作负载映射方法
- [`design/elenor_verification_bringup/ELENOR_Verification_Bringup_Design.md`](./design/elenor_verification_bringup/ELENOR_Verification_Bringup_Design.md) — 验证与 bring-up 计划

## 建议阅读路径

如果你第一次接触 ELenor，建议按下面顺序阅读：

1. **总体架构**：先看 `ELENOR_Architecture_Design_v1.md`
2. **芯片与层级组织**：再看 Chip Top / Tile Group / Compute Tile / Memory NoC
3. **四大核心引擎**：BOA、EVU、MFE、USE
4. **软件与 ABI**：Compiler Stack、Runtime ABI、Driver/Firmware
5. **落地与验证**：Workload Mapping、Verification Bring-up、PPA/Timing/Power

## 文档导出

仓库自带 PDF 导出脚本：

```bash
bash scripts/generate_pdf.sh -f design/ELENOR_Architecture_Design_v1.md
```

如需导出其他设计文档，将 `-f` 后的路径替换为对应 `.md` 文件即可。

## 适用对象

本仓库适合以下角色：

- AI 芯片架构师
- RTL / SoC 设计工程师
- 编译器与 Runtime 工程师
- Driver / Firmware 工程师
- 验证、Bring-up、性能分析工程师
- 希望系统化理解 AI 加速器分层设计方法的研究者

## 项目状态

当前仓库以**架构设计文档**为主，覆盖芯片顶层、计算引擎、数据流、运行时、编译器、验证与 bring-up 等多个维度，适合用于：

- 架构评审
- RTL 拆解
- 编译器 / Runtime 任务拆解
- Driver / Firmware ABI 讨论
- 性能建模与验证规划

## License

See [`LICENSE`](./LICENSE).
