# 局限性

## 这套模型模拟了什么、没模拟什么

已模拟能力（V1 scope）：

- 控制流层级 Group Task->Tile role->Engine 的逐周期推进；
- 4 引擎延迟（Roofline）与 launch/wait 重叠；
- Stream Queue credit/backpressure/EOS；
- role 完成聚合 -> group task 推进；
- PMU stall 归因（WAIT_EVENT/WAIT_OPERAND/STREAM_CREDIT/WAIT_L1_BANK/
  WAIT_L2_BANK/WAIT_NOC_CREDIT/WAIT_DMA_QUEUE/WAIT_HBM_OUTSTANDING/NONE）；
- credit 不变量每周期校验；
- PR 2 物理内存模型：HBM 外部 binding、L2/L1 free-extent 分配
  （owner/generation/pin/对齐/跨 bank segment）、逐腿 transfer route
  （HBM/Global DMA/NoC/L2/L1 bank/Local DMA）、容量/owner/generation/
  use-after-release 错误一律 fault、不产生 success event。
- PR 3 task-bound phase/release protocol：`tile.signal <phase>(%task)`
  以 logical task 驱动每个 grid 的 `all_tasks` 聚合；显式 event、
  owner/generation 和 consumer pin 共同门控 role-aware L2 release。

有意简化/未模拟（需明确告知用户的边界）：

- 单 Tile Group：不模拟多 Group 间的 NoC/Collective 竞争（Collective 有
  1-cycle command/window 模型和 trace event，但 reduce datapath/bandwidth
  仍未建模）；
- Group DMA 的 HBM outstanding/NoC credit/L2 bank 竞争按
  `full_memory` 的独立 stage 建模（同 bank 串行、不同 bank 重叠）；
  `timing_only`/`runtime` 折叠为单腿延迟，不保留 per-bank 计数；
- L1 SlotFrame 固定 slot ABI（WORKSPACE/PER_TILE_PROGRAM），bank policy
  编码未冻结（V1 全部通过）；
- 引擎流水有限：BOA/EVU/USE 单 job 非流水；MFE 通道化（N 条 load lane +
  M 条 store lane，每 lane 内串行），lane 间不共享资源仲裁；
- 无真实 tensor 数值：transfer 的 bytes 是数值参数，payload tracker 只
  记录 layout/地址元数据（PR 2 起地址来自真实 transaction，不再 CRC 伪造）；
- residency/cold-warm load 由 `ProgramResidencyManager` metadata 路径
  管理（program_id/version/hash/epoch）。
- PR 3 聚合只支持 `#nest.aggregate<all_tasks>`；没有 quorum、subset 或
  其他 aggregation mode；
- logical task 到 physical tile 的 scheduler 仍固定为当前 1:1 映射；
- 未实现 PR 4 gather；
- 未实现 PR 5 memory trace lane/counter（现有 `tile_signal` instant 是
  control-flow trace，不是 memory trace lane/counter）。

## V2 fidelity 边界（PR 2）

- `timing_only`：无 handle/allocation，src/dst 视图为空，单腿折叠延迟。
- `runtime`：真实 L1/L2 分配与 HBM binding（owner/generation/容量/
  生命周期检查），但 transfer 折叠为单腿带宽+launch 开销。
- `full_memory`：在 `runtime` 之上增加逐腿 route（HBM channel/outstanding、
  NoC VC credit、Global DMA channel、L2/L1 per-bank segment、Local DMA）。
- 三种模式都必须通过全局 binding/permission/静态 view 验证。

## V1 输入参数与逻辑地址 IR 限制（PR 1 冻结）

- `nest.subview` / `tile.subview` 的 `strides` 必须全 1（语法接受非 1，
  verify 拒绝）。
- 无 inline 下标糖（不实现 `%Y[0:4, ...]`），view 必须经显式 subview op。
- `nest.context` formal 仅限 `!nest.global_memref`；submit actual 仅限
  `nexus.program` block arg（不能传 view/切片）。
- `tile.subview` 只支持单一 `task_dim`（无 `tile.task.id`）；logical task
  依赖偏移由 runtime 在 `tile.subview task_dim` 维度解析。
- 无 view 链：`nest.subview` src 必须是 context global formal，
  `tile.subview` src 必须是 tile.program L2 formal。
- dispatch actual 仅限整个 `!nest.l2_buffer`（context body 无 L2 view
  producer）。
- transfer 字节数从 view/buffer shape 推导；旧 `bytes = N` 语法已删除。
- 子视图必须连续 row-major（对任一 `sizes[i] > 1` 的维度，所有尾随维度
  必须满足 `sizes[j] == backing_dims[j]`）；非连续视图被 verifier 拒绝
  （physical transfer model 只支持单段 byte range）。
- 输入绑定按名字匹配（program block arg name_hint 为键），无按位置
  隐式绑定备选路径。
