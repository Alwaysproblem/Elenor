# P20 · C7-B validator RTL 联仿

状态：未开始
对应原方案：§5.3（cycle 权威表与 cycle c 顺序）、§5.4（内存与真实 UCE 接入的取样规则）、§7.2 C7-B
验证层级：R（RTL 联仿）为主，附 C 层对拍基准；不冒充板级 B。

## 1 目标与不做

目标：把 P19 的 C ABI 桥接成 validator 的可选 BOA backend（RTL 模式），使同一 artifact 在 Python 周期后端与 RTL 后端下可对拍：

1. 内存真实性：RTL发出的mem_req驱动实际bank/transfer/NoC服务，不从STEP预先构造整矩阵伪读取。读字节在对应L1实际read stage服务时取样；write service完成更新ByteStore后再ack。
2. 单一权威：RTL维护AM/结果接收；Python factory禁用周期SharedAmManager/BoaResultService驱动，仅映射请求、完成与观测。C ABI不能自动发现Python对象，互斥必须落实在实例构造和所有调用点。
3. 无预测完成：RTL 任务不构造预测 `EngineJob.finish_cycle`（`engines.py:63/308` 现为周期后端模型）；完成事件只来自 ABI 事件批。
4. 旧 generation 安全：过期 RTL 响应按身份丢弃并计数，不写入新 owner。

不做：Rocket/RoCC（P21）、固件（P22）、矩形/多阵列配置扩展（P25/P26 只复用本任务的 backend）、周期后端自身数值/拓扑改造（P04–P07/P09 已收敛）、板级。

## 2 前置与输入

- P09：软件全链路 gate 通过（新数值/终态合同生效后的基线）。
- P19：`cosim-abi` target 通过；`edge`/reset/generation/事件批契约可用。
- 输入：P09最新typed artifact与配置（不能假定P06仍从v4起升级）、P03 oracle bytes、真实workload及完整A/B/C binding。

## 3 当前证据、必读与修改边界

- 周期次序现状：`tile_group.py:1750 TileGroup.step` 先 NoC/transfer 再 `_step_tiles`；`tile.py:2028 ComputeTile.step` 为 `engines.tick → boa_results.submit/advance → uce.step`。内存服务早于 Tile tick 已成立，RTL edge 插入点就在 Tile tick 的 BOA engine 段。
- 取样现状：`tile.py:1335 _make_boa_completion` 在 completion 闭包内 `store.read_view(a_view/b_view)`，`tile.py:1349` 无 ByteStore 即报 "typed matmul requires full_memory ByteStore data"，`tile.py:1372` `write_view` 在输出阶段。这是周期后端软件模型；RTL 模式不得复用该闭包充当"实际读服务"。
- AM 权威现状：`runtime/shared_am.py`（request/commit_step/release_final，`bound_engine=slot.group` 于 `shared_am.py:427`）与 `runtime/boa_result.py::BoaResultService`（reserve/submit/advance、WAIT_COMMIT/WAIT_OUTPUT 状态机）是周期后端权威；RTL 模式下只读其计数口径、不驱动状态迁移。
- generation 现状：`TransferManager` 已有 `generation_validator`、tombstone、`reference_acquire/release` 钩子（`memory/transfer.py:500` 起），旧响应过滤复用该机制，不新造平行方案。
- 必读：§5.3 权威/禁止表、P18/P19 evidence、`pipeline_validator/config.py:70-78`（`engines.boa.*` 映射惯例）。

## 4 接口与不变量

- backend选择：CLI拟用 `--hw-override boa_backend=rtl_cosim`，默认cycle。本执行实现选择单独存run manifest，不使同一合法target artifact仅因换后端而失配；array/profile/tile_k/容量仍严格校验fingerprint。本任务接入现有config，不依赖P05预先定义未知backend。
- cycle c 顺序（RTL 模式）：内存服务（NoC→transfer，含上一 edge 到期的 BOA 读/写事务）→ Tile tick 内：交付到期响应 → 推进一个 RTL edge（P19）→ 采样请求/完成批 → 完成进 UCE-local 完成队列 → `uce.step` 发射形成下一 edge 输入。RTL 发出的新内存请求最早下一轮内存服务返回，该边界延迟计入对拍与 report。
- ByteStore唯一。数据在实际read stage取样并保存至response，后续NoC/响应延迟不重新取样；C在write service完成后ack。不能把最终transaction回调无条件当作更早的实际read时刻。
- 终态偏序：每链 `compute_done ≤ step_committed ≤ output_visible ≤ slot_free ≤ chain_done ≤ UCE event` 由 RTL 事件时间戳核验；trace 保留各时刻。
- report 新增字段（拟新增）：`boa_backend`、`rtl_event_count`、`stale_rtl_response_count`、边界延迟分布；`credit_invariant` 等检查在 RTL 模式改用 RTL 导出计数，不与 Python 后端状态混算。

## 5 实施步骤

1. 接入config、能力检查、factory、CLI及manifest；cycle路径保持P09基线，RTL路径不得创建可修改AM的周期模型，观测adapter不得调用其request/commit/release。
2. AM_REQUEST/SUBMIT经协议端口进入RTL；采样每edge真实mem_req（地址/segment、req_id、mask、owner/generation、读写data），构造对应MemoryTransaction走既有L1/bank/transfer/NoC。不能在STEP issue自行读整矩阵代替operand feeder。
3. read stage取样：若TransferManager只有最终回调，补最小实际读服务hook；响应延迟只运送已取样字节，验证等待期间覆写不会造成时间穿越。
4. C drain beat随写事务保持；完成时先验identity、更新ByteStore再ack。旧响应不写新owner，但仍结清旧transaction/input引用与credit，不是只计drop。
5. ABI事件接入既有UCE通知路径，区别非FINAL step_committed与FINAL chain_done，保持context/epoch/event/handle/seq身份；不得预测finish或调用Python结果服务发第二份完成。
6. 观测接线：PMU/trace/report 新字段；AM 申请计数（request/grant/park/wake/cancel）从 RTL 导出，检查 `park = wake + cancel`（成功闭合运行另核 `request=grant=ALLOC`，按总则 §5 分母规则）。
7. 对拍与回归：同一 artifact 双后端运行比较；Q07 类背压在 RTL 模式经 RTL 输出 credit 注入（P16 交付的注入口），不用 Python `set_output_blocked`。
8. 注册 `validator-cosim` target cases；更新 `examples/run.sh` 的 `fidelity-of` 表（typed matmul 已是 full_memory，沿用；不硬编码 fidelity override，遵守 run_all 约定）。

## 6 验收表

| Case ID                | 构造                                         | 必须观测                                                                        | 失败特征                     |
| ---------------------- | -------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------- |
| cosim-matmul-bytes     | typed matmul artifact + input-data，RTL 后端 | C bytes 与 P03 oracle 逐 bit 一致；与周期后端一致                               | oracle 缺位或 compare 容差化 |
| cosim-event-order      | 同上                                         | 事件偏序与周期后端一致；终态偏序逐链成立                                        | 只比总数不比偏序             |
| cosim-counts           | 同上                                         | request/grant/ALLOC/FINAL/slot_free 非时序计数双后端一致；cycles 允许不同且记录 | 把周期数拉平断言             |
| cosim-mem-contention   | BOA 读与 MFE 流量争同 bank                   | 边界延迟与服务争用可见（对照无争用 run），读写走真实 bank/NoC 计时              | read_view 即当完成           |
| cosim-stale-generation | 制造 reset/回收后旧响应到达                  | 丢弃并计数，新 owner 数据不受污染                                               | 旧字节写入新 lease           |
| cosim-backpressure     | RTL 输出 credit 注入背压后撤除               | lease 停留至恢复，无需 UCE 轮询即自主完成（Q07 硬件层）                         | Python 侧代答背压            |
| cosim-nosecondam       | RTL 后端 + 尝试挂 Python AM 驱动             | 创建/运行被拒，report 只有一份 AM 权威                                          | 双计数混算                   |
| cosim-fault            | 注入步骤错误                                 | ERROR 终态、无副作用、计数入独立分母                                            | 伪成功                       |

## 7 可执行命令

```bash
# 周期后端基准（现有能力）：boa-am-matmul 是 examples/run.sh:283 的 alias（含 --ir-file、
# A/B/C 三个 binding 与 A/B input-data），用它生成 artifact 与周期基准 report：
bash examples/run.sh boa-am-matmul --compiled-output /tmp/p20/boa_am_matmul.compiled.json \
  --json --report /tmp/p20/cyc.json
# 同一 artifact 的 compiled-only 重放必须带 alias 同款完整 bindings/inputs（typed matmul 需 full_memory）：
conda run -n elenor-validator python -m pipeline_validator --compiled-file /tmp/p20/boa_am_matmul.compiled.json \
  --hw-override boa_am_slot_capacity_bytes=65536 --sim-override fidelity=full_memory \
  --input-binding A=0x100000:16384:r --input-binding B=0x200000:65536:r \
  --input-binding C=0x300000:65536:w \
  --input-data A=examples/workloads/boa_am_inputs/boa_am_matmul/A.bin \
  --input-data B=examples/workloads/boa_am_inputs/boa_am_matmul/B.bin \
  --max-cycles 200000 --json --report /tmp/p20/cyc_replay.json
# RTL 后端（本任务交付 --hw-override 取值，拟新增；bindings/inputs 同上完整集合）：
conda run -n elenor-validator python -m pipeline_validator --compiled-file /tmp/p20/boa_am_matmul.compiled.json \
  --hw-override boa_am_slot_capacity_bytes=65536 --hw-override boa_backend=rtl_cosim \
  --sim-override fidelity=full_memory \
  --input-binding A=0x100000:16384:r --input-binding B=0x200000:65536:r \
  --input-binding C=0x300000:65536:w \
  --input-data A=examples/workloads/boa_am_inputs/boa_am_matmul/A.bin \
  --input-data B=examples/workloads/boa_am_inputs/boa_am_matmul/B.bin \
  --max-cycles 500000 --json --report /tmp/p20/rtl.json --trace-json /tmp/p20/rtl.trace.json
# 统一入口（本任务注册，拟新增）：
hardware/scripts/check validator-cosim --config hardware/configs/single_array.yaml \
  --out /tmp/p20 --seed 7 --case cosim-stale-generation
```

预期：两后端非时序计数与bytes一致，允许不同周期。直接CLI示例仅在加载RTL能力与配置匹配时使用；统一target必须用显式单阵列配置先编译匹配artifact再跑两后端，拒绝不兼容library，不能fallback。runner/CLI按已验证规则创建输出目录。

## 8 产物、交接与停止条件

产物：backend 选择、RTL executor、取样/回报通路、report/trace 字段、target cases、`plan/evidence/P20.md`。
交接：P21/P22 以本任务 backend 为 RTL 层载体；P25/P26 只改配置不改 backend；文档同步 §5.3/§5.4 引用的现状表述。
停止条件：双后端对拍任一必测项不可证；P18/P19 接口缺口；不得以单后端通过宣称联仿通过。
