# CN-G1：合同生成、容量实证与联合封印（C009–C020）

阅读[执行总指南](../execution_guide.md)后逐卡执行。以下源文件均为未来创建目标；`cn unit --suite contracts` 的 runner 已由 C003/C008 提供，**每卡新增自身的实际 validator 和正负例**，不能让空 suite 返回成功。合同草稿可用于构建校验工具和 bootstrap ELF；正式运行仅消费 C020 联合封印。

<a id="c009"></a>

## C009 — 边界身份、消息、预算与 CSR 空间合同

- **任务**：W1-01
- **前置**：C008
- **建议 commit**：`feat(coral-protocol): define typed boundary identities and lifetime budgets`
- **输入/先读**：[W1 计划](../implementation/W1_Boundary_Cosimulation_Plan.md) §7；总计划 §3.3；`V/execution_ir.py` 的 Grid/Task 身份与上游 `Csr.scala` 占用表。
- **修改边界**：新增 `H/protocol/{cn_protocol_v1.yaml,channel_budgets.yaml,lifetime_bounds.yaml,csr_name_map.yaml,extract_upstream_csr.py}`、contracts suite 的身份/预算测试；本卡不改 TileGroup 运行路径。
- **实现步骤**：
  1. 定义 wire ID 与恢复映射：Task、Context、指令、子操作、事务和 transport 序列分别有域；重复表达的 root/grid generation 必须相等，不能按 PC 匹配完成。
  2. 登记 CTRL/EVENT/PROGRAM/MEMORY/DEBUG 五通道及独立预算，已有 Stream 消息归 MEMORY 子队列。P1 唯一接受、P3 只提交 P1 快照、P4 下拍响应写入协议。
  3. 从固定来源提取已占 CSR，新增声明只登记逻辑名和权限；按容量/发射率/消息寿命的同单位上界推导约束，禁止把“条目数＋周期数”直接相加或写死 0x7c0 可用。
- **验证**：本卡增加 contracts case；同一个 txn 的 HOLD/FINAL 必须是不同合法阶段，transport 重放不能重复副作用。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：合法身份往返后仍能区分同 PC 双 Context、同 slot 不同代；重复 stage/stale generation/CSR 冲突各有明确错误，单 Context 不产生零宽字段。C010–C017 消费 draft schema；此时未封印、不启动 CS1。

<a id="c010"></a>

## C010 — 原生 owner、指令、配置与退休合同

- **任务**：W2-01
- **前置**：C009
- **建议 commit**：`feat(coral-contract): specify native owner tags and retirement boundaries`
- **输入/先读**：[W2 计划](../implementation/W2_Context_UCE_Plan.md) owner/接口任务；上游 `hdl/chisel/src/coralnpu/RetirementBuffer.scala`，不是错误的 `scalar/scoreboard.scala` 路径。
- **修改边界**：新增 `H/rtl/scalar/doc/{rob_tag,ctx_ctrl,retire_record,oracle_requirements,perctx_issue}.yaml`、`H/protocol/field_requests/w2.yaml`、contracts 验证。不得在本卡只改 Core IO 而留调用方不可编译。
- **实现步骤**：
  1. 定义覆盖 scalar store/fault/长延迟写回/RVV 子操作的完整 owner tag；ROB slot 必有 generation 或等价 op_seq，不用 Context＋裸 slot 防复用。
  2. 指定四 lane 同 Context 前缀接受、eligible RR、定向 fetch epoch；架构配置退休提交、操作 dispatch snapshot、flags delta 先存条目后在退休时合并。区分普通 GPR completed 写回与 CSR 提交时刻。
  3. 给 W3/W4/W6/W7 提供明确 ready/valid、cancel、park、retire-record 字段和 ownership；非译码/上下文对象不跨 C ABI。Context 复用须旧义务关闭与可靠 completion 消费。
- **验证**：使用合法/错配的结构化样本校验 consumer 解码及资源边界，不以搜索 OwnerTag 字符串作证明。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：同 Context 重用 ROB slot 的旧结果不能被解释成新 op；C1 常量退化、C2 双分区均可表达。W2-01 自身合同完成；C020 对齐全部包，C026 起才修改实际 RTL。

<a id="c011"></a>

## C011 — 精确数值合同与固定上游 KAT

- **任务**：W3-01
- **前置**：C009
- **建议 commit**：`feat(coral-numeric): specify bit semantics before implementing the oracle`
- **输入/先读**：[W3 设计 §6/§8](../components/W3_RVV_VME.md)、[W3 计划](../implementation/W3_RVV_VME_Plan.md) 数值任务、C006 原核基线。旧草稿中的示例向量必须重新按数值合同计算，不能原样抄成 golden。
- **修改边界**：新增 `H/protocol/w3_numeric_profile_v1.yaml`、`H/verification/numerics/{numeric_contract.md,known_vectors.json,upstream_probe/}`、相应 contracts/numeric-bootstrap suite；W7 的独立 oracle 不在本卡实现。
- **实现步骤**：
  1. 明确有限宽乘树、jam、TK 分组、内部 ROD、乘级 widening 溢出 Inf、累加 ROD 溢出 max-normal、signed zero/NaN/subnormal/mask。最终 BF16 RNE 是另一边界，VME OF/NV 不限制其他 RVV 指令自身的异常位。
  2. 对固定上游生成参数和小型实际 RTL 单元作定向探针，保存输入/期望/actual bits 与参数证据；它是 DUT 合同核对，不冒充独立 oracle。
  3. 用可独立推导的精确向量及边界值固定规范；masked Inf/NaN 不参加 special 分类、不因乘零生成 NV。KD-01 登记 raw Spike/MPACT 的累加溢出差异，不改已选 RTL 合同来迎合参考。
- **验证**：本卡实现 numeric-bootstrap suite（用 C004 已锁上游工具，不需要 CS1 target）。

```bash
cn unit --suite contracts --out "$EVIDENCE/contract"
cn unit --suite numeric-bootstrap --config "$P0" --out "$EVIDENCE/numeric"
```

- **通过/交接**：规范每条规则有定向输入与可复核结果；覆盖真实溢出、mask、抵消、TK 分组，不能以 grep 常量或高精度“接近”替代。产物标 experimental；C047 独立重建 oracle，尚不标数值能力 supported。

<a id="c012"></a>

## C012 — E1/E2 区域声明与 kernel 级模式表

- **任务**：W3-01
- **前置**：C010、C011
- **建议 commit**：`feat(coral-contract): separate prequeue lease requests from declaration consume`
- **输入/先读**：W3 设计 §3.2；[W3 计划 §16/§20](../implementation/W3_RVV_VME_Plan.md) 最新勘误；[W5 设计 §4.2.1](../components/W5_Kernel_Image_Runtime.md)。
- **修改边界**：新增 `H/protocol/w3_vme_decl_abi_v1.{md,json}`、`field_requests/w3.yaml`，扩展 numeric profile 的模式表 schema、contracts case；实际 CSR 数字仍归 C017。
- **实现步骤**：
  1. 定义 E1：目标指令到 Context 发射头且 valid、未进共享 RVV 队列时申请 lease；无实例由 W3 登记唯一 waiter 并 park。E2：grant 后真正被前端接受时一次消费 pending。E1 可读 pending，不可消费；不能等 accepted 才登记等待。
  2. 定义 grant 后下游背压、取消/redirect、未消费声明与已接受操作的不同清理路径；grant 已取得的资源即便目标尚未 accepted 也必须结算，不宣称“未 accepted 必然无副作用”。FINAL 不自动 drain，END 不自动释放，可靠 region_done 后才转交。
  3. 模式/转移表绑定当前封印 kernel。参考 seed/drain `(16417,18)`、BF16 `(16418/16450,265)` 仅属参考模板；块间 CLEARED→恢复合法，其他尾块有自己受检表。通用拒绝针对超出当前表/权限/精度，不做三编码全局白名单。
- **验证**：本卡增加可执行的状态/合同校验样本；同一合法尾块表必须能被 generator 与 loader schema 接受，缺恢复前导的访问拒绝。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：样本覆盖 E1 未 grant 时队列为空、grant 后 stall pending 不被吃掉、一次 E2 后重复消费拒绝；尾块表正例和 CLEARED 正例通过。C020 收录完整 W3 合同；真实进展证明在 C044/C053，不把合同校验当 RTL PASS。

<a id="c013"></a>

## C013 — 内存、页授权、资源预算和端口合同

- **任务**：W4-01
- **前置**：C010
- **建议 commit**：`feat(coral-contract): define memory ownership and bounded resource lifetimes`
- **输入/先读**：[W4 计划](../implementation/W4_Memory_Subsystem_Plan.md) 合同任务；`V/memory/{transfer,arena,page_pool,mshr}.py`、`hardware_config.yaml` 的现状。先读既有 AccessLease 事务集合，不把已存在的 acquire/release 再实现一遍。
- **修改边界**：新增 `H/protocol/{w4_memory.contract.yaml,field_requests/w4.yaml}`、`H/configs/w4_memory_constraints.yaml`、`H/scripts/derive_w4_bounds.py`、contracts tests。
- **实现步骤**：
  1. 声明 L1 bytes/cache/MSHR/Frame/hold/bound 归 RTL、L2/NoC/HBM 归 Python；返回字段包括 Context/事务/分片/视图分配代，不能以普通 binding 名替代实际 allocation provenance。
  2. 固定 scalar/RVV 本地窗口、MFE typed binding＋offset 全宽外地址、逐页原子 acquire 与终态 ACK release；取消、确认隔离、旧响应、信用和在途引用分别记账。
  3. 用静态包络计算 bank/Frame/MSHR/响应/窗口预算。输入早释放按 Task/all_tasks，剩余 pin 退休幂等；不修改 L2 物理回收粒度。Profile gate 只挡新数据 issue，不挡完成/维护/程序通路。
- **验证**：本卡新增推导器与 contracts suite，输出与独立小规模手算一致，越界/负容量/权限/单位混淆拒绝。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：完整生命周期和守恒式能覆盖 hold→bind→retire、refill waiter 存活、外写真实结算；本卡合同完成，W4-01 到 C020 联合 seal 才关闭。C023/C029/C055 消费，不能另建 Python L1 执行账本。

<a id="c014"></a>

## C014 — 映像包、显式 binding 与 helper 合同

- **任务**：W5-01
- **前置**：C012、C013
- **建议 commit**：`feat(coral-contract): specify sealed kernel bindings and resource manifests`
- **输入/先读**：[W5 计划](../implementation/W5_Kernel_Image_Runtime_Plan.md) W5-01、W5 设计 CMP-18～20；当前 schema3/v3 控制产物。
- **修改边界**：新增 `H/protocol/{w5_package.contract.yaml,field_requests/w5.yaml}`、`H/kernels/bindings/schema.json`、contracts validation；本卡不伪称 package runtime 已存在。
- **实现步骤**：
  1. 定义独立 `cn_package/v1`：真实 image、entry/link layout、完整 control artifact、kernel/source-node 映射、actual views、shape/dtype/alias/init、workspace/clobber/事件/模式表、target/profile/hash/patch_mask。
  2. 固定多预链接 slot 与 mutable 参数区隔离、按包身份驻留、zero-ref＋失效 ACK 后复用；raw 控制产物的估算 text_bytes 不得当真实 ELF 长度。
  3. 列出每个 helper 的可观察语义、编译器重排屏障、参数读写和资源约束，CSR 仅引用 C017 生成名；无 runtime 猜 operand 或补 await。为 C018 提供候选 linker/资源约束格式，不要求先有 sealed package。
- **验证**：contracts suite 对完整绑定正例接受，对缺实际 view、非法 alias、未声明资源、未知字段、错误版本域和没有受控来源的 image 拒绝。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：consumer 有足够信息生成真实调用与检验 effect，不以字段非空代行为验证。W5-01 的运行时编解码/Loader 仍在 C037，不能提前把整任务标 DONE。

<a id="c015"></a>

## C015 — 事件、终态、取消与复位合同

- **任务**：W6-01
- **前置**：C010、C012、C013、C014
- **建议 commit**：`feat(coral-contract): close event wait and safe reset ownership`
- **输入/先读**：[W6 设计 §3/§5](../components/W6_Events_Fault_Reset.md)、`V/runtime/{event_table,reset_domain}.py`、当前 fault 路径。
- **修改边界**：新增 `H/protocol/{w6_event_facility.contract.yaml,w6_terminal_reset.contract.yaml,field_requests/w6.yaml}`、contracts cases；不改运行中的复位域。
- **实现步骤**：
  1. 定义完整事件键、PENDING/四终态、frontier 两计数和 WAIT/WAITALL 完成优先原子登记；PHASE_SIGNAL 是方向通知，TASK_COMPLETION 是外部终态，不强行合为一个 local-event kind。
  2. 指定 Group 复位顺序与 watchdog 唯一权威；DRAIN 检查真实操作/传输义务，**不要求 PENDING=0**，随后 MARK_EVENTS 授权 RTL 写本地表并 ACK。RETIRE 授权前不能等父 R/pin 先归零。
  3. 按静态 live-event frontier/响应上界推导槽预算；运行期满表背压、静态超包络加载拒绝。POST_EVENT 只准 sealed manifest 授予的 Group producer，旧代拒收/完整键恢复、first-fault 明确。debug 门控、FIFO 深度推导与受限导出字段均在本卡完整登记，不留到 C061 追加旧 seal。
- **验证**：本卡实现合同验证的合法与冲突图输入；制造 DRAIN→PENDING→MARK_EVENTS 环、缺少 terminal reserve、混淆 Group/Tile writer 都须拒绝。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：无循环等待且每个 accepted 类都有成功/取消/失败结算归属；本卡合同完成，W6-01 到 C020 联合 seal 才关闭。C033/C034 仍须实际并发/背压验证，合同通过不当功能通过。

<a id="c016"></a>

## C016 — 日志封窗、T 矩阵和证据适用性合同

- **任务**：W7-01
- **前置**：C014、C015
- **建议 commit**：`feat(coral-verify): specify trace windows and complete acceptance applicability`
- **输入/先读**：[W7 计划](../implementation/W7_Verification_Trace_Plan.md) trace/checker/T 矩阵；根需求 T-01～T-68。
- **修改边界**：新增 `H/verification/contracts/{trace_window,log_record,checker_verdict,replay_bundle,perf_gates}_contract_v1.yaml`、`H/verification/matrix/t_matrix.yaml`、`H/protocol/field_requests/w7.yaml`、schema validators/tests。
- **实现步骤**：
  1. 定义整数 cycle/phase/seq、双向实际握手、payload/blob、初态、source/image/config 身份、捕获边界与封窗计数；记录 seq 与 transport message_seq 不混用。
  2. 逐项登记 68 条及其子项的实现任务、证据类型和 gate。BUILD/LOADER/Python 基线/真实 RTL 各有适用性，不能所有行机械要求 CS1_LOG；仅 CS2/CS3 与明确 N2 子项延期。PS 子行按批准范围登记 `not_run`，不是把 P0/P1/PN 等 mandatory 项改为延期。
  3. 校验合法故障的 case-PASS 与 DUT-ERROR、不可重放缺口与数值 FAIL 的区别；矩阵此时为待执行，禁止预置 PASS。场景来源可登记计划标识，最终 gate 必须有实际可取回 evidence。
- **验证**：本卡向 C007 evidence checker 加入 schema/适用性负例；前缀连续但无正确封窗、缺 blob、非授权 deferred 均应拒绝。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：T-01～T-68 唯一完整、每个 mandatory 子项有类型正确的验收要求，未运行项不可能被解析成 PASS。本卡合同完成，W7-01 到 C020 联合 seal 才关闭；C041/C054/C064 等逐步实现 C1–C7 的真实运行判据。

<a id="c017"></a>

## C017 — 协议生成器、codec 与 CSR 分配校验

- **任务**：W1-01
- **前置**：C009、C010、C012、C013、C014、C015、C016
- **建议 commit**：`feat(coral-protocol): generate checked codecs and collision-free CSR bindings`
- **输入/先读**：C009–C016 全部 field requests/合同；W1 最终 C ABI 名称。缺一个必需 fragment 就失败，不能 `w1-inferred` 代替。
- **修改边界**：`H/protocol/{gen_protocol.py,derive_widths.py,csr_allocator.py,csr_lint.py,freeze_check.py}`、`generated/{cn_messages.h,cn_offsets.svh,cn_layout.json}`、新增 `V/cosim/wire.py`；contracts suite 的 C/SV/Python consumer tests。
- **实现步骤**：
  1. 合并完整逻辑 schema，固定排序生成布局/消息长度/错误枚举与 CSR helper，未知字段/非法长度/保留位/溢出在入口拒绝。
  2. 构建真实 C 与 SV 解码消费者，用同一输入驱动有效消息、截断、错代、重放与双阶段事务；验证跨语言解析的行为，而非只比较生成文本或做 Python 自己 pack/unpack。
  3. 生成器幂等，碰撞/不支持容量失败且不污染既有生成物。实现 freeze checker，但**本卡不写正式 PROTOCOL_FREEZE**，C018/C019 还需完善资源与配置输入。
- **验证**：contracts suite 已可运行，C004 已提供其 C/SV 工具链；新增测试必须至少包含错误消息的消费端拒绝与 ledger 不变。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：C/Python/SV 一致理解同一真实字段与错误边界，CSR 与原核占用/权限不冲突。C018 消费 draft 生成器构建测量 helper；C020 负责一次 seal，不能此时声称 ABI 已最终冻结。

<a id="c018"></a>

## C018 — bootstrap helper ELF 与静态资源测量

- **任务**：W5-02
- **前置**：C017
- **建议 commit**：`build(coral-kernel): measure bounded bootstrap ELF resources before target seal`
- **输入/先读**：W5 设计 CMP-20、C014 包 schema、C017 生成的 helper 声明、C004 原生工具链。
- **修改边界**：新增 `H/kernels/bootstrap/` 的真实 C/asm helper 源、候选 linker 输入、`H/scripts/lib/measure_kernel.py`、`H/verification/kernel_bootstrap/` 测试；注册 `unit --suite kernel-bootstrap`。不实现假 package loader。
- **实现步骤**：
  1. 生成最小可审的 control sequence 与资源测量 ELF；它只用于测量，不作为可装载 CS1 包。候选地址来自待验证布局，不能把随机常量写进固定位编码。
  2. 从真实 ELF/map/调用图计算 text、静态数据、stack/spill/scratch 的可证明上界，证明至少两个不同链接区域可同时容纳；动态峰值不足以证明上界。
  3. 输出与 C013 静态资源包络相容的候选布局/参数，超限回到构建阶段重新规划，绝不在 Loader 修补。后续真实 kernel 仍须逐个复核，超出候选 target 需新 target hash，而非扩容同一身份。
- **验证**：本卡提供独立 bootstrap 构建测量 suite，不调用 C037 才实现的 `package`。

```bash
cn unit --suite kernel-bootstrap --out "$EVIDENCE/unit"
```

- **通过/交接**：实际 linked ELF 资源报告、两个不重叠地址布局和超限/重叠拒绝负例通过。C019 消费测量值；C038/C052 才完成可实际执行的 kernel 链，W5-02 尚未关闭。

<a id="c019"></a>

## C019 — C1/C2 实验配置生成与容量证明

- **任务**：W0-04
- **前置**：C008、C017、C018
- **建议 commit**：`feat(coral-config): generate sourced C1 and C2 experimental targets`
- **输入/先读**：全部包预算与 C018 ELF 资源；C1/C2、N=1、full RetirementBuffer 决策。P0 是独立 `kind: p0_upstream`。
- **修改边界**：新增/完善 `H/configs/{cs1_c1.yaml,cs1_c2.yaml,params.d/}`、`H/rtl/targets.bzl`、`H/scripts/handlers/build_config_stage.sh`、module suite registry；保持根源码不产生 Bazel 缓存。
- **实现步骤**：
  1. 汇总实际来源/量纲与每 bank/slot/credit/事件约束，生成 Scala/SV/C/包配置所消费的同一 target 指纹；缺来源、缺预算或相互冲突拒绝生成。
  2. 实现 `build --prepare-only`，只输出生成配置与合法性 verdict；不谎称完整 Tile 已能编译。module suite 构建只使用已实现的独立模块目标，后续卡原子注册其真实 tests。
  3. 将实验 target 与 numeric/kernel 支持状态分开；协议逻辑字段内容摘要可预计算，正式 `PROTOCOL_FREEZE.json` 下一卡生成。禁止目标文件引用包含自身摘要的 freeze 文件而形成 hash 环。
- **验证**：本卡实现 prepare-only 与按 target 的 module runner；可运行性通过实际编译生成配置消费者，而非仅字段相等。

```bash
cn build --prepare-only --config "$C1" --out "$EVIDENCE/c1"
cn build --prepare-only --config "$C2" --out "$EVIDENCE/c2"
cn unit --suite contracts --out "$EVIDENCE/unit"
```

- **通过/交接**：C1 不产生零宽索引，C2 资源包络可装入，两者同源生成且失配被拒。C020 冻结本版合同；完整 `build`/`cosim` 目标由 C040 提供，不能抢先用不存在的 top。

<a id="c020"></a>

## C020 — 一次性联合封印与 CN-G1 门

- **任务**：W1-01、W0-04、W7-06
- **前置**：C011、C017、C018、C019
- **建议 commit**：`feat(coral-protocol): seal complete contracts and enforce the G1 gate`
- **输入/先读**：C009–C019 的产物及其实际校验记录；执行总指南的不可变 seal 与任务关闭规则。
- **修改边界**：`H/protocol/PROTOCOL_FREEZE.json`、`freeze_check.py` 的 seal/check 入口、C007 gate evaluator 新增 `H/verification/gates/G1.yaml`、所有生成物引用；本卡是共同合同的集成 owner。
- **实现步骤**：
  1. 重验全部 fragment、numeric、helper、target、预算和生成物的内容/hash，producer/consumer 均存在且可消费；不能以“包状态 DONE”代替实际产物一致。
  2. 一次性生成 seal：内容清单摘要不包含自身字段，也不要求把未来 commit SHA 写进自身；新内容新 hash，不向已封印文件追加。新版本必须迁移全部消费者并重新跑受影响 gate。
  3. G1 evaluator 验证实际 codec/资源/CSR/生成参数证据；缺字段、旧 artifact、半封印、篡改生成物、未关闭预算各非零。冻结不改变后续数值能力的 experimental 状态。
- **验证**：seal/check 入口由本卡完成；gate 命令骨架已在 C007，G1 在本卡注册完整判据。

```bash
cn unit --suite contracts --out "$EVIDENCE/unit"
cn gate --stage G1 --out "$EVIDENCE/gate"
```

- **通过/交接**：全部必需合同和资源证据闭合；联合 seal 的故意缺项/陈旧输入负例拒绝。W1-01 可关闭；W0-04 静态目标生成完成但其生成到真实 RTL 的消费者证明仍由 C040/C065 验收。CN-G1 关闭，C021 以后实现只消费明确的 frozen 输入。
