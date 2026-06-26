# 局限性

## 这套模型模拟了什么、没模拟什么

模拟了（V1 scope）：

- 控制流层级 Region→Tile→Engine 的逐周期推进；
- 4 引擎延迟（Roofline）与 launch/wait 重叠；
- Stream Queue credit/backpressure/EOS；
- stage 完成聚合 → region 推进；
- PMU stall 归因（WAIT_EVENT/WAIT_OPERAND/STREAM_CREDIT/NONE）；
- credit 不变量每周期校验。

有意简化/未模拟（需明确告知用户的边界）：

- 单 Tile Group：不模拟多 Group 间的 NoC/Collective 竞争（Collective 有 1-cycle command/window 模型和 trace event，但 reduce datapath/bandwidth 仍未建模）；
- Group DMA 是纯延迟：不模拟 L2 SRAM 容量、bank 冲突、L2 占用，但 DMA trace slice 现在携带 op/desc/bytes/L2 slot；
- L1 SRAM 是带宽预算：不模拟容量、bank 冲突（tile.py:8-9 注释明说 V1 leave frozen）；
- 引擎非流水：一引擎一 job，无流水线深度（engines.py:14-15）；
- 无真实数据：descriptor 的 bytes/ops 是数值参数，不搬真实 payload，只算延迟；
- **residency/cold-warm load**：dispatch_stage 直接 load_program，不模拟 program residency miss/cold launch（设计文档 15 的 residency 契约未进模型）。
