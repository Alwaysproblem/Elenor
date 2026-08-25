# 局限性

## 这套模型模拟了什么、没模拟什么

模拟了（V1 scope）：

- 控制流层级 Group Task→Tile role→Engine 的逐周期推进；
- 4 引擎延迟（Roofline）与 launch/wait 重叠；
- Stream Queue credit/backpressure/EOS；
- role 完成聚合 → group task 推进；
- PMU stall 归因（WAIT_EVENT/WAIT_OPERAND/STREAM_CREDIT/NONE）；
- credit 不变量每周期校验。

有意简化/未模拟（需明确告知用户的边界）：

- 单 Tile Group：不模拟多 Group 间的 NoC/Collective 竞争（Collective 有 1-cycle command/window 模型和 trace event，但 reduce datapath/bandwidth 仍未建模）；
- Group DMA 是纯延迟：不模拟 L2 SRAM 容量、bank 冲突、L2 占用，但 DMA trace slice 现在携带 op/desc/bytes/L2 slot；
- L1 SRAM 是带宽预算：不模拟容量、bank 冲突（tile.py:8-9 注释明说 V1 leave frozen）；
- 引擎流水有限：BOA/EVU/USE 单 job 非流水；MFE 通道化（N 条 load lane + M 条 store lane，每 lane 内串行），lane 间不共享资源仲裁；
- MFE 多 channel 共享 backend（L1 port / NoC / 聚合带宽）竞争未建模：每 channel 按 `mfe_bandwidth_gbs` 独立计带宽；
- 无真实数据：descriptor 的 bytes/ops 是数值参数，不搬真实 payload，只算延迟；
- **residency/cold-warm load**：dispatch_role 直接 load_program，不模拟 program residency miss/cold launch（设计文档 15 的 residency 契约未进模型）。

## V1 输入参数与逻辑地址 IR 限制（PR 1 冻结）

- `nest.subview` / `tile.subview` 的 `strides` 必须全 1（语法接受非 1，
  verify 拒绝）。
- 无 inline 下标糖（不实现 `%Y[0:4, ...]`），view 必须经显式 subview op。
- `nest.context` formal 仅限 `!nest.global_memref`；submit actual 仅限
  `nexus.program` block arg（不能传 view/切片）。
- `tile.subview` 只支持单一 `task_dim`（无 `tile.task.id`）；task 依赖
  偏移表达式留在 DTO，物理 tile 绑定在 PR 2。
- 无 view 链：`nest.subview` src 必须是 context global formal，
  `tile.subview` src 必须是 tile.program L2 formal。
- dispatch actual 仅限整个 `!nest.l2_buffer`（context body 无 L2 view
  producer）。
- transfer 字节数从 view/buffer shape 推导；旧 `bytes = N` 语法已删除。
- 物理地址、IOVA 容量/延迟模型、L2 allocator 消费 `role`/`alignment`
  属 PR 2；`tile.signal(%task)` 属 PR 3。
- 输入绑定按名字匹配（program block arg name_hint 为键），无按位置
  隐式绑定备选路径。
