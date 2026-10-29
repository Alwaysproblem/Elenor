# P10 · C4 / §4 ABI：协议 Schema 与接口冻结

| 项   | 值                                 |
| ---- | ---------------------------------- |
| ID   | P10（对应原方案 C4 / §4 接口协议） |
| 前置 | P02、P09                           |
| 状态 | 未开始                             |

## 1. 目标与不做

**目标**：建立权威 wire/descriptor schema `hardware/boa/protocol/boa_protocol_v0.json`（拟新增），由这一份 schema 生成 Scala / C / Python 三份 codec 与常量表，禁止三处手写编号。冻结：命令/响应字段、握手语义（一次申请、PARKED/WAKE、FULL、ERROR）、请求重放保护、有限 ID 的回绕规则、全部有限容量常量，以及「AM 授予不得依赖完成事件槽」的编码级约束。同时把 C4/C5 真硬件（P14 分配器/ChainTable、P15 事件、P16 credit/AM 服务、P17 epilogue/drain）列为该 schema 的强制集成门：下游只准消费生成 codec。

**不做**：不实现 allocator/event/credit/epilogue 硬件本体（P14–P17）；不冻结 RoCC opcode/funct 具体编码（P21 在 schema 预留的 encoding 槽内冻结）；不冻结 NaN/INT8 数值编码（P03 已冻结的按引用带入）；不建立第二套配置解析（字段落点接 P05 的 `hardware/configs/*.yaml` 格式）；不改 pipeline_validator 源语义（backend 接线属 P20）。软件 artifact 版本（P06 cutover 后的当前版本）与 wire ABI `v0` 是不同版本域，互不联动，不提供自动 fallback。

## 2. 前置与输入

- P02 通过：`hardware/scripts/env`、`hardware/scripts/check` 分派器可用（总则 §4.2），Scala/C 工具链可编译生成物。
- P09 通过：软件 gate 新基线；`SharedAmManager` 一次申请/挂起/唤醒、bundle、分阶段结果服务的合同即本协议的语义来源。
- 原方案必读：§2.2（身份分层/handle 表）、§2.3（配额与 bundle）、§3.5（O/R credit、pending_am_updates）、§4.1（AM_REQUEST 合同）、§4.2（SUBMIT_STEP 与 descriptor 字段表）、§4.3（事件/等待/释放、mem_req/mem_resp 字段）、§4.5（指令语义表）、§7.2 C4/C5、§8.1 T 表、§8.1b Q 表；总则 §3、§4.1、§4.2。
- 现有仓库证据（语义参照，非待迁移代码）：`pipeline_validator/execution_ir.py:286` `ExecBoaMatmulDesc`；`pipeline_validator/runtime/shared_am.py:206 request / 323 _grantable / 402 _allocate_lease / 460 begin_step / 493 commit_step / 532 release_final / 549 abort_owner / 613 snapshot`；`pipeline_validator/runtime/boa_result.py:72 BoaResultService`；`pipeline_validator/hardware_config.yaml:83-112`（容量默认值）。ERROR reason 枚举必须与 `shared_am.py::request` 的永久错误一致：logical_am∉{0,1}、字节数超 slot 容量、bundle_slots 越界、该 owner 已有挂起请求、同 logical_am 有未受理 FINAL 的 live 链、持有 lease 期间改 bundle、未受理 FINAL 链数达 bundle（自锁）。

## 3. 必读文件与修改 ownership

- 必读：本文件、`plan/00_执行总则与共享合同.md`、`plan/README.md`、前置 evidence `plan/evidence/P02.md`、`plan/evidence/P09.md`。
- 本任务新建（全部拟新增）：`hardware/boa/protocol/boa_protocol_v0.json`（权威 schema，含 `freeze` 段）、`hardware/boa/protocol/gen_protocol.py`（codec 生成器）、生成物 `hardware/boa/src/main/scala/boa/protocol/BoaProtocol.scala`、`hardware/boa/sim/c/boa_protocol.h`、`hardware/boa/protocol/py/boa_protocol.py`；`hardware/configs/protocol-v0.yaml`（P05 格式实例，承载容量常量供一致性检查）。
- 本任务在 `hardware/scripts/check` 分派器中注册并实现真实 target `boa-protocol`（未实现前该 target 必须非零报错）。
- 不修改：`pipeline_validator/**` 源语义、P14–P17 的模块文件（它们按生成 codec 消费）、`hardware_config.yaml`（软件模型键不动）。

## 4. 输入输出、接口与不变量

schema 必须覆盖（字段名可评审调整，语义不可缩水）：

1. `AM_REQUEST(owner, logical_am, bundle_slots, output_shape, acc_format, layout, output_destination, chain_config, request_seq) -> GRANTED(lease_handle, bound_engine, completion_token) | PARKED(request_id) | ERROR(reason)`；`WAKE(request_id, lease_handle, bound_engine, completion_token)` 恰好一次；`CANCEL(owner)`。
2. `SUBMIT_STEP(lease_handle, seq, A_descriptor, B_descriptor, input_ready_tokens, input_usage_handles, k_extent, INIT_or_ADD, FINAL) -> ACCEPTED(step_token) | FULL | ERROR(reason)`。
3. `EVT_STATUS / EVT_TAKE / REC_REAP / CTX_CANCEL`（§4.5 语义）；`mem_req/mem_resp`（req_id、owner/generation、解析地址、访问类型、byte mask、data/status）。
4. descriptor 字段组按 §4.2 五列表：版本/身份、不变链配置、输入 view、步骤、内部状态（设备维护，compiler 不猜）。

硬性不变量（编码层可校验）：

- **completion_token 不是授予资源**：`completion_token` 仅为该链完成事件的引用，指向 context admission 时按 Tile Program 静态事件数分配的槽身份 `(ctx_id, epoch, event_id)`。GRANTED 资源清单只能是 slot/quota/ChainTable 项/current+next mailbox/已准入输出绑定；schema lint 必须静态拒绝把事件槽取得或等待列入授予前提。
- **重放保护**：`(owner, request_seq)` 唯一；schema 声明重放返回同一结果、不得二次授予（行为验收在 P14）。
- **有限容量全部显式**：slots_per_group=2、slots_per_context=2、每链 mailbox=当前执行记录+1 个 next、挂起 FIFO 深度=每 Tile context 数、terminal 诊断环有界且满则覆盖最旧并计溢出、事件槽=程序静态事件数、结果 FIFO/credit 深度字段（数值由 P16 冻结，schema 预留字段与范围）。
- **回绕规则**：generation 每 slot 单调递增；epoch 仅在 context 安全退休后复用；request_seq/event seq 回绕必须定义 quiescence/背压条件，禁止以「位宽够大不会回绕」替代。
- **typed 字段一等公民**：transpose（A/B 各一，枚举四组合）、epilogue typed 参数块（bias/scale/clip/quant 字段+逐项使能位）、generation（allocation/generation、epoch、request_seq、事件 generation）必须是 schema 中序列化的 descriptor 字段，由唯一 compiler lowering 填写；任何下游（含测试 harness）不得使用 schema 之外未序列化的隐藏配置注入这些能力。发现字段缺口时，缺口清单交回 compiler 实现补齐，不建测试专用旁路。
- 位宽冻结程序：按 §6.1 容量域推导候选（如 context 1..8、logical_am 2、engine 1..4），在 `freeze` 段记录选择依据与复核日期；候选值标注「本任务评审冻结」，不得声称原方案已给编码。

## 5. 实施步骤

1. 从协议章节提取字段表，逐字段标注owner、类型、位宽候选、回绕/重放与冻结任务。P03数值规则已前置，不再“延后P03”；P21补RoCC二进制编码，P16补物理服务容量，当前冻结其字段与合法范围。
2. 编写 `boa_protocol_v0.json`：机器可校验（字段、类型、范围、枚举 reason、资源清单约束）；内置 lint 规则执行 §4 不变量（含 completion_token 约束的静态断言）。
3. 实现 `gen_protocol.py`：单一 schema → Scala object（常量+case class+二进制编解码）、C header+codec、Python codec；生成器幂等，输出 hash 写入 manifest。
4. 跨语言一致性测试：同一 golden 消息字节在三个 codec 下解码结果一致、再编码 byte-identical；负例（非法 logical_am=2、bundle_slots=3、未知 reason、截断流、未知版本）全部非零失败。
5. 统一比较具有相同语义/粒度的配置：Tile-local group/slot/quota、数值profile/shape及事件身份。软件step粒度的FIFO/credit/commit_cycles与硬件微块参数分别带单位/粒度，不要求数值相等；同一语义参数两个来源不一致才拒绝。
6. 注册boa-protocol的PC01–PC07；公共case过滤由P02实现。本任务真实编译运行三语言codec，schema源检查只能辅助，不能代替跨语言边界消息/错误行为验收。
7. 写 `plan/evidence/P10.md`：含 §4.1(4) 要求的 producer/consumer/codec/config/docs 调用方清单——P14 消费 AM*REQUEST/WAKE/CANCEL；P15 消费 EVT*\*/CTX_CANCEL/terminal 环字段；P16 消费 SUBMIT_STEP/FULL/mem_req 与容量字段；P17 消费 FINAL/drain/事件次序字段；P19 消费 C header；P21 在 encoding 槽冻结 RoCC opcode/funct。

## 6. 验收表

| case | 构造                                                                   | 必须观测                                                                                | 失败特征 / 层                      |
| ---- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------- |
| PC01 | 三 codec 对同一 golden 消息集解码再编码                                | byte-identical，计数一致                                                                | 任一 codec 差 1 byte 即失败（F）   |
| PC02 | 非法字段集：logical_am=2、bundle=3、未知 reason、截断、未知版本        | 全部非零错误并指明字段                                                                  | 静默接受或 0 退出（F）             |
| PC03 | 旧事件未reap、slot换代消息的三codec往返                                | request/wake与静态event_id独立保留，无消息字段覆盖；授予不等事件的动态证明由P14/P15负责 | 将事件槽当AM资源或身份混淆（F）    |
| PC04 | 同语义配置错配；另设step/microblock不同credit值                        | 前者拒绝，后者按各自粒度合法接受并完整编码                                              | 不分粒度强求所有常量相等（F）      |
| PC05 | 干净重复生成并编译codec，再运行同向量集                                | 生成物确定、真实编解码结果相同；作为构建smoke而非只测文件拷贝                           | 编码漂移（F）                      |
| PC06 | 有限位宽最大值/溢出/未知版本消息；回绕策略评审                         | 边界编码精确、超界拒绝；回绕行为后续P14/P15验证                                         | 隐式截断或“大位宽无需回收”（F）    |
| PC07 | 三codec round-trip四转置和全部epilogue使能/参数、allocation generation | typed字段不丢失，非法dtype/范围拒绝；使用P09真实descriptor生成消息                      | 隐藏harness参数代替可执行字段（F） |

反例边界：不得以「先冻结一份编码再让下游适配」绕过评审；不得把 P03 数值编码、P21 指令编码提前冒充已冻结。

## 7. 可执行命令（契约；target 由本任务实现）

```bash
conda run -n elenor-validator pre-commit run -a   # 仓库文档/py 门
hardware/scripts/check boa-protocol \
  --config hardware/configs/protocol-v0.yaml --out artifacts/p10 --seed 1
hardware/scripts/check boa-protocol --config hardware/configs/protocol-v0.yaml \
  --out artifacts/p10-case --seed 1 --case PC02     # 单 case 复跑
hardware/scripts/reproduce artifacts/p10/manifest.json --out artifacts/p10-repro --offline
```

预期manifest包含PC01–PC07真实编解码/错误行为结果，辅助schema审查另标。缺target/生成器或任一必要case未运行必须非零，不得echo成功。

## 8. 产物、交接与停止条件

- 产物：schema+freeze 段、三份生成 codec、生成器、`boa-protocol` target、`protocol-v0.yaml`、`plan/evidence/P10.md`。
- 交接：P14–P18、P19、P21 只消费生成 codec；schema 后续任何变更必须更新 evidence 的调用方清单并要求下游重验（总则 §4.1）。
- 停止条件：跨语言一致性或负例不通过；lint 放行事件槽进入授予路径；生成物不确定；伪造任何「已冻结」编码而不留评审记录。

## 9. 版本、cutover 与文档同步

- wire ABI `v0` 与软件 artifact 版本域（P06 cutover 后的当前版本，不硬钉 v4/v5）独立；schema 的 `min_reader_version` 变更走本任务版本评审，不随软件 artifact 联动。
- 若 P05 最终配置键拼写与本文件候选不同，以 P05 evidence 交接为准更新本 schema 一致性检查，不建第二套解析。
- 原方案 §4.5「opcode/funct 由后续规格冻结」的后续规格即本任务 schema 的 encoding 槽 + P21 冻结动作；evidence 必须写明这一分工，防止下游把槽位当成已冻结编码。
