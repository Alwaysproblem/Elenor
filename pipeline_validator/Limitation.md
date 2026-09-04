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
- PR 3.5 L2 admission wait：submit 被 device slot 接受后，若原子 L2
  bundle 暂时放不下（合法但当前 free map 不满足），context 进入
  `ADMISSION_WAIT`（占 slot，不占 UCE/L1/L2/stream/DMA，不 fault）；
  只有 `nest.release` 的 allocator final-free 才会按 strict FIFO 唤醒
  队首，且唤醒与 first action 固定为 `admit_cycle` / `admit_cycle + 1`。
  非法或空池也无法容纳的 bundle 立即 fault、永不排队。
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
- PR 3.5 admission queue 固定 strict FIFO：没有 utilization-first
  bypass、优先级或抢占；head-of-line blocking 是 V1 明确行为（PMU
  `l2_admission_queue_peak`/`l2_admission_wait_cycles` 观测，后续策略
  独立 PR）；唤醒源只有 release final-free，没有跨层
  `nexus.await_phase` 或 nest event export；
- logical task 到 physical tile 的 scheduler 仍固定为当前 1:1 映射；
- Gather 仅实现 deterministic profiled V1：不解释真实 index tensor，
  没有 tagged `AddressProvider`，不报告 address-accurate/value-accurate
  cache hit rate；没有 Scatter/ScatterReduce/FIFO delivery；
- PR 5 memory trace lane/counter 已实现：每个 transfer leg 在接受时记录、
  完成时发射 `X` slice（反映真实 accept/complete，cancel 路径不留虚假预测）；
  counter 为变化采样（非逐 cycle，change-only 去重）；flow 用 Chrome `s`/`t`/`f`
  串联同一 transaction 的多腿跨车道。保真边界：collapsed-leg 路线（`timing_only`
  /`runtime`）每条 txn 只有一条腿，flow 退化为 `s`+`f` 两个事件；Gather 仍为
  deterministic profiled（不 address-accurate/value-accurate）。车道顺序由
  `process_sort_index`/`thread_sort_index` 元数据固定（见 README §Profiling）。

## V2 fidelity 边界（PR 2）

- `timing_only`：无 handle/allocation，src/dst 视图为空，单腿折叠延迟。
- `runtime`：真实 L1/L2 分配与 HBM binding（owner/generation/容量/
  生命周期检查），但 transfer 折叠为单腿带宽+launch 开销。
- `full_memory`：在 `runtime` 之上增加逐腿 route（HBM channel/outstanding、
  NoC VC credit、Global DMA channel、L2/L1 per-bank segment、Local DMA）。
- 三种模式都必须通过全局 binding/permission/静态 view 验证。
- Gather 是三种 fidelity 的例外：lookup/MSHR/refill/ordered-write
  状态机不折叠；`timing_only` 只是不物化 src/dst 地址。`line_token` /
  `merge_group` 始终是 opaque profile identity，禁止 hash 成伪 bank、
  cache set 或物理地址。

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
