# W2 Context 与 UCE 详细实施计划

> **逐 commit 执行入口**：[execution_guide.md](../execution_guide.md)。先用该入口选择前置已满足的卡片；本页保留工作包细节，commit 顺序、工具可用时点、文件边界和原任务关闭条件以执行 guide 为准。架构语义仍遵守已批准设计，不按旧草稿中的占位命令或文字存在性检查宣称完成。

> 状态：评审计划，非实现记录。对应设计：[W2_Context_UCE.md](../components/W2_Context_UCE.md)；总体裁决见[总体设计](../Coral_Nexus_Tile_Design.md)；包间合同见父合同 [../plan.md](../plan.md) §2/§3

## 1. 已选方案摘要（服从父合同）

本包落父合同第 5 条与设计 §3–§5 并细化为 5 个任务。基线结构判定（来自上游，详见 §3）：Coral scalar 核是**单条 4 lane in-order 前端流水**（UncachedFetch→SCore dispatch→shared units→RetirementBuffer），最小改造路径不是"每 Context 一条完整 Core"，而是保持一条 issue 流水、把**状态侧按 Context bank 化**。

已选方案（本轮按推荐默认执行，每条都有对应任务与验证）：

1. **4 lane 连续前缀 × 单 Context 单 bundle（W2-03）**：每周期调度器选 1 个 Context，dispatch bundle 的 4 lane 全来自该 Context 且是程序序连续前缀（上游"首拒后不得再接受"断言 [A02] 保留）。不收窄 lane、不做 aging、不做 quantum（未证明，不引入；SCH-03 公平性由"RR 必选 eligible 头 + 停顿归因"承担，不把吞吐保证偷换成公平性承诺）。
2. **RR 头选择**：调度器在 `contextCount` 上做 RR，从上周期选中槽位 +1 起扫描 eligible 头，**只有真正选中（本周期 issue 该 Context）才推进 RR 指针到选中槽位 +1**；未选中（头部因 operand/queue/CSR/region 不 eligible）该 Context 不推进指针（对齐 validator `_select_context` 语义 tile.py L635-653）。eligible 判据 = 头部预译码 + 分级资源 probe（本 Context scoreboard/ROB 空间/共享 queue 容量/CSR gate/region 投影位），**不是单个静态 ready 位**；W2 不完整复制 DispatchV2，但 head 预译码与各资源 probe 是 W2-03 义务，probe 接口由 W2-03 定（消费方 W3/W4/W6）。A 头等待 region 时头部停在 START（MAT-20，W2-03），B 独立补位（SCH-02）。
3. **GPR/FPR/VRF 按 Context 语义完全隔离；首版全复制存储 bank（W2-02）**：`Regfile` / `FRegfile` 按 Context 复制（候选 A），共享单元（ALU/BRU/MLU/DVU/FloatCore/LSU/RVV）在 execute 边界按 owner 取读，写回按 owner 回写。scoreboard 是每 bank 内部位图（上游 Regfile `scoreboard/flushed_scoreboard` 结构 [A10] 按 bank 复制），不共享、不跨 bank 匹配。
4. **scalar ROB 静态私有分区（W2-02）**：`RetirementBuffer` 实例按 Context 复制，每 partition 独立 `enqPtr/deqPtr/nSpace/credit`；**不引入共享环形 + 每 Context head 的方案**（该方案回收 tail 仍要跨 Context 越过他 Context 老条目，RET-03 不成立，也不做未证明的共享省面积断言）。RVV ROB 的 owner 化归 W3-02，本包只定义 owner 标记接口与 tag 互认。
5. **PC-keyed 完成/故障/寄存器号匹配切换（W2-02）**：上游 `RetirementBuffer` 完成匹配全部按新键改写——store 完成、非 RVV fault、scalar/float 长延迟写回不再按 PC/mepc/寄存器号裸匹配 [A01]；统一键为 `(ContextKey, owner_op_seq)`，向量侧沿用 `rob_tag` 并升级为 `(ContextKey, rob_tag)`（详见 §4 中的设计表）。故障归属与停发配合 W6，故障匹配键同样按 owner+op_seq 判定。
6. **配置 dispatch snapshot + ordered flags/CSR 互锁（W2-04）**：`vset/mset/frm/float` dispatch 拍取 snapshot（Coral SV 域 H14 先例扩展到 scalar CSR），flags delta 挂所属 ROB 条目、在该条目退休时按 Context 内程序序合并，CSR 读 / pending-config 做 Context 局部互锁，不做转发优化（无充分先例，不引入）。
7. **fetch epoch 定向（W2-03）**：`FetchReorderBuffer` 的 `flush` 改为携带 ContextKey+`fetch_epoch` 的定向取消（对齐设计 §3.5），替代全局 `flushTx`；其他 Context 的取指缓冲不动。
8. **等待补位/park 及安全退休（W2-04）**：等待 region/event/context 资源的 Context 头部停靠；已接受工作照常执行、写回允许到未被选中的 Context（STA-06）；`EXEC_DRAINED` 之后由 W6 推进 RETIRE_READY/RETIRE/TASK_COMPLETION；Context 槽保守等 TASK_COMPLETION 接收、旧义务闭合且身份隔离后复用。

**本阶段（CS1 scalar 阶段）非目标**：RVV/VME 数值与区域语义（W3）、RTL LSU/MFE 实存（W4 提供端口，本包不实现存储）、kernel/ELF/package（W5）、reset/watchdog/debug 导出（W6）、oracle/trace 生成器本体（W7；本包固定 oracle 依赖与 retire record schema）。也不做：aging/quantum 调度、ROB 动态共享 free-list、RF bank 化（候选 B）、CSRs forwarding 优化、FLT-02a 局部故障隔离、N≥2 VME 实例验证。

**唯一 owner 表（跨包文件所有者）**（父合同已定，本计划是它的细分）：
| 文件/状态 | 唯一编辑 owner | 其他包接口 |
| --- | --- | --- |
| `Parameters.scala`(Core/Params) | W2 | W3/W4/W6 发 `patches/reqs/` 变更请求，W2 合入后 bump 共享 header |
| `Core.scala`/`SCore.scala` | W2 | 同上 |
| `Regfile.scala`/`FRegfile.scala` | W2 | 同上（scoreboard 端口按 bank 化改动在此落地） |
| `Decode.scala`(DispatchV2) | W2 | 同上 |
| `UncachedFetch.scala` | W2 | 与 W4 端口裁决见 W2-03 |
| `RetirementBuffer.scala` | W2 | 同上 |
| `Csr.scala` | W2 | W5 配置/声明写经 `interfaces/ctx_ctrl.yaml` 消费，不直接编辑；W6 特权清单同上 |
| `FaultManager.scala`/`Bru.scala` | W2 | 同上 |
| `rvv_backend_*.sv`/`RvvCore.scala` | W3（ROB owner=W3-02） | W2 只发 `reqs` 变更请求，不直接编辑 |
| `Lsu.scala` | W4 | W2 只发 `reqs` 变更请求（owner-tag 访存端口、retire/flush 路由归 W4） |
| `CoreAxi.scala`（实体） | W2 | gate_ctl 算法/信号语义归 W6-04；W2 按 CR-W6-W2-01/02 执行实体改造 |
| oracle/schema/checker/trace | W7-02 | W2 只固定 `retire_record` 依赖（W2-01 输出） |

## 2. 前置与当前证据（区分"已核实"与"W2 自建"）

### 2.1 已核实锚点（上游真实路径，固定上游 Coral `f79522157fc0f1d7311f44e209fc36c9c8b7a039`，行号已在本轮逐条本地核对）

| 事实（已逐条核对）                                                                                                                              | 上游真实路径（锚点）                                                                                                                                                                                                                                                                                                                                       | 结论                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| scalar store 完成只按 PC 匹配（`storeComplete.bits === bufferEntry.addr`，单 `Valid(UInt(pcBits))` 端口）                                       | `hdl/chisel/src/coralnpu/RetirementBuffer.scala` L28（IO 声明）、L89（单拍寄存）、L587-589（匹配）；`bufferEntry.addr` 就是 store PC，`FaultManager` L101-142 同样把 `mepc` 当完成匹配键                                                                                                                                                                   | 同 Context 内连续同名指令在途时 PC 键已能错对（RET-01）；跨 Context 同映像同 PC 必然错乱（[INFERENCE]，基于 L587/L589 与 L494-495 语义推断） |
| 非 RVV fault 按 `mepc` 匹配、RVV fault 按 `rob_tag` 匹配                                                                                        | 同上 L493-495（`Mux(isRvvFault, rvvTagMatch, bufferEntry.addr === faultRetire.bits.mepc)`）                                                                                                                                                                                                                                                                | fault 归属同样 PC 键，须换 owner+op_seq                                                                                                      |
| scalar/float 长延迟写回按寄存器号匹配（`x.bits.addr === bufferEntry.idx`）                                                                      | 同上 L467-476（写回扫描，`idx` 是标量/浮点寄存器号，不是 ROB 序号）                                                                                                                                                                                                                                                                                        | 同 Context 内同名寄存器跨代不能假完成（`flushed_scoreboard` 处理，[A10] L110-135）                                                           |
| vector/VME 写回按 `rob_tag` 匹配                                                                                                                | 同上 L482-483；`Rob2Rt.rob_tag`（`rvv/RvvInterface.scala` L142-143）；RVV `ROB_TAG_WIDTH` 由 scalar 侧传入（`rvv_backend_define.svh` L135-138）                                                                                                                                                                                                            | 向量侧已带 tag，但 tag 不含 Context；须升 `(ContextKey, rob_tag)` 复合键（W3-02 侧细化）                                                     |
| ROB 是单 `CircularBufferMulti`（`retirementBufferSize=16`）单实例，进出 `enqPtr/deqPtr` 单尾单头                                                | 同上 L91-97、L233-240、L695-735；`Parameters.scala` L141-142（16 depth / 8 retire lanes）                                                                                                                                                                                                                                                                  | 全局共享；静态私有分区须改实例侧（W2-02）                                                                                                    |
| trap flush 是全局动作（`trapRetired → instBuffer 清空、flushed_scoreboard、RVV flush、LSU pipelineFlush、fetch.iflush`）                        | 同上 L725/L733-734；`SCore.scala` L72-80、L129-131、L259、L476-498、L540                                                                                                                                                                                                                                                                                   | 换 owner 需要把这套 flush 链全部按 Context 过滤；fault 侧停发由父级 fail-stop 兜底                                                           |
| 4 lane 前缀是 bundle 内 scan（`isJump/readsAfterWrite/branch…` 每条 lane 与前缀 AND）                                                           | `scalar/Decode.scala` L396-462（`scoreboardScan`、`readAfterWrite`/`WAW` scan）、L683-708（`canDispatch`）                                                                                                                                                                                                                                                 | 保留（W2-03 只做"选中 Context 的 bundle 送入"，不改 scan 本身）                                                                              |
| CSR dispatch 只能单条走 slot0、依赖 `retirement_buffer_empty`、`rvvIdle`/`lsuActive`/`vmeIdle` 串行化（单 Context 假设）                        | `Decode.scala` L683-708（`canDispatch`）、L707-708（CSR 等 ROB 空）、L691-697（fence）、`SCore.scala` L147（`halted`）、`Regfile.scala` L96（scoreboard 一份）                                                                                                                                                                                             | 需要 Per-Context 化 CSR 读互锁与 `coreIdle` 拆分（W2-02/W2-04）                                                                              |
| `isVxsatOrVcsr` 依赖 `rvvIdle`（跨 Context 会误伤别的 Context 的 RVV 操作）                                                                     | `Decode.scala` L976-993（vxsat 读写、`rvvIdleOrNotVxsat`）                                                                                                                                                                                                                                                                                                 | CFG-05，须 Context 化（W2-04）                                                                                                               |
| 取指总线 `IBusIO` 两相（address/data）无 txid 线；`Fetcher` 在 RTL 内部用 `IndexAllocatorShifting(2)`+`FetchReorderBuffer` 重排，靠固定拍距对齐 | `scalar/UncachedFetch.scala` L324-383（`maxConcurrentTx=2`）、L39-322（FRB 按 txid 匹配、单 flush 全清）                                                                                                                                                                                                                                                   | 定向化需加 ctx/epoch/reqid 键（W2-03）；FRB 逻辑可保留，只换键与 flush 语义                                                                  |
| 取指缓冲（`InstructionBuffer`）单 PC 单份、按 `iflush/branch/debug` 单 flush 全清                                                               | 同上 L566-583（`InstructionBuffer(p.fetchInstrSlots*2)`、`flush := iflush\|\|branch\|\|debug_pc`）                                                                                                                                                                                                                                                         | 定向 flush（W2-03）                                                                                                                          |
| `Regfile` 内含 32 深标量 scoreboard（regd/comb 双 bit）+6 写口；`linkPort` 读 x1 给 `jalr` 预测用                                               | `scalar/Regfile.scala` L16/L51-92/L110-135/L254-267；`FRegfile.scala` L35-50                                                                                                                                                                                                                                                                               | 每 Context 复制（W2-02）；`linkPort` 须随 Context 复制                                                                                       |
| `halted/wfi` 属 CSR 单份标量；`CoreAxi` 用 `core.io.wfi` 直接关钟                                                                               | `scalar/Csr.scala` L283-443、`CoreAxi.scala` L109                                                                                                                                                                                                                                                                                                          | park/quiescence 拆分（W2-04，W6 消费）                                                                                                       |
| RVV config state（`vset` 产物 `vl/vtype/vstart/vxrm/vxsat/frm/mtype`）是 CSR/`RvvCore` 单份                                                     | `scalar/Csr.scala` L22-36/L319-326/L509-557；`rvv/RvvCore.scala` L1037-1057                                                                                                                                                                                                                                                                                | W2 持有 scalar 侧 CSR 副本与 ordered delta；RvvCore configState 侧 Context 化归 W3-02                                                        |
| debug/DM 单路（`CSR req arbiter`、`dm.scalar_rs/float_rs` 读口）                                                                                | `scalar/SCore.scala` L114-116、L215-250、L292、L344-484；`CoreAxi.scala` L71-134                                                                                                                                                                                                                                                                           | 特权/debug 导出 owner W6-04；W2 保留 `dispatch.io.csr` 本征通路，`dm.*` 声明为可选（保留 CR-W6-W2-03 基本合同接口）                          |
| mini/full RetirementBuffer 判据=`enableVerification`（mini 丢 `inst`/写值）                                                                     | `RetirementBuffer.scala` L67/L72-84；`Parameters.scala` L119；`chisel/src/coralnpu/BUILD` L617-632/L821-832（`_verification_` target 已有）                                                                                                                                                                                                                | CS1 采纳 full 模式（W2-D01；c1 实测 hang 风险见 §13）                                                                                        |
| 全局共享执行单元（`SCore` ALU/Bru/MLU/DVU/FloatCore/LSU 共享+`rvvcore` 单份）                                                                   | `scalar/SCore.scala` L167-266、L270-281、L340-484（写口仲裁 L461-469）、L586-599                                                                                                                                                                                                                                                                           | 共享单元保留，issue 输入/写回输出挂 owner（W2-02/W2-03）                                                                                     |
| 单 hart 标量 ISA 测试套件（c1 单 Context 原生回归载体）                                                                                         | `tests/cocotb/coralnpu_isa/`（ecall/fence/perf*counters）、`tests/cocotb/registers.S`、`tests/cocotb/exceptions/`、`tests/cocotb/csr_test/`、`frm_test.cc`、`fcsr_frm_hazard_test.cc`、`float_csr_interlock_test.S`、`float_hazard_tests.S`、`wfi_slot*{0..3}.c`、`cf_fault_retire_test.S`、`retire_buffer_full_illegal_inst.S`、`rvv_fflags_test.{cc,py}` | W2-05 直接跑这些作 c1/退化回归（上游 fixture 归 W0/W7 统一，不在 W2-05 内维护副本）                                                          |
| verilator 顶层 C++/ELF 加载 fixture（native ISA 闭环宿主，可与 W1 C ABI 共用镜像）                                                              | `tests/verilator_sim/coralnpu/`（`core_tb.cc`、`core_mini_axi_sim.cc/tb.{h,cc}`、`elf.{h,cc}`、`coralnpu_cfg.h`、`core_if.h`、`memory_if.h`）、`sw/coralnpu_sim/spike_cosim_dpi.{h,cc}`、`spike_simulator.{h,cc}`（上游自带 Spike 独立 ISA oracle）                                                                                                        | W7-02 采 Spike 或自研；W2-05 只约定接口（`oracle_requirements.yaml`）                                                                        |
| Chisel 单元测试基建（W2 单元 runner）                                                                                                           | `hdl/chisel/src/common/{CircularBufferMulti,IndexAllocator,InstructionBuffer}{,Test}.scala`；`FetchControlSpec.scala`/`FetchReorderBufferSpec.scala`（AnyFreeSpec+ChiselSim）                                                                                                                                                                              | W2 单元测试沿用（W0-01 保证 workspace 可引用）                                                                                               |
| CSR 地址表全表（含 `VSTART 0x008/VXSAT 0x009/VXRM 0x00A/MTYPE 0xC23` 等）                                                                       | `scalar/Csr.scala` L45-101（`CsrAddress` 枚举）                                                                                                                                                                                                                                                                                                            | W2-04 的 `ctx_decl` CSR 段须从空闲地址分配（冲突清单归 W2-01/W1-01）                                                                         |
| CSR 合法性检查（decode 侧只查 `CsrAddress.safe`，写合法位段在 `Csr.scala` 内）                                                                  | `scalar/Decode.scala` L973-993（`csr_address_valid`/`csr_op_valid`）、`scalar/Csr.scala` L363-364、L594-610                                                                                                                                                                                                                                                | W2-04 把 flags 写改为 ordered delta 挂 ROB 条目；full 模式下 CSR 直写路径必须移除或替换                                                      |
| `FloatCore.fflags` 来自 fpnew `status_o`，直连 CSR（非 tag 挂 ROB）                                                                             | `float/FloatCore.scala` L310-312、L481（`io.csr.in.fflags.valid`）、L507-508                                                                                                                                                                                                                                                                               | W2-04 改为随 retire 挂 ROB entry（FloatCore 改造归 W2）                                                                                      |
| MLU/DVU 是 shared 单元、cmd 无 owner tag、单 DVU 流水                                                                                           | `scalar/Mlu.scala` L36-46、L77+；`scalar/Dvu.scala` L129（`io.req.ready := !active && ...`）、L138、L186                                                                                                                                                                                                                                                   | cmd 内需带 owner tag（W2-03），否则 dispatch 拍与 use 拍间切换 Context 会串写                                                                |
| ALU/Bru 是每 lane 单元、`bru(0).csr.csrInterlock` 路径                                                                                          | `SCore.scala` L167-236；`scalar/Bru.scala` L99-100/L103                                                                                                                                                                                                                                                                                                    | Bru/ALU dispatch 拍使用寄存器数据读、无跨拍长延迟；owner tag 在 dispatch 拍直接读 per-Context bank                                           |

### 2.2 已存在代码（可直接读/跑）

- `pipeline_validator/tile.py`：L55-64 `_UCEContextState`、L590-660 `_probe_head`/`_select_context`、L683-708（发射）、L1298-1320（`_complete_context`/`retirement_wait`）、L1016/L1046（`uce_launch_attempt/accepted` 计数）。这些是 Python 模型对发射/补位/ retire 行为的**合同观测点**，不是指令 oracle。
- `pipeline_validator/pmu.py` L15-30 `StallReason`（WAIT_EVENT/WAIT_OPERAND/STREAM_CREDIT/WAIT_L1_BANK/…/WAIT_MSHR 按序）。W2 新增 `WAIT_REGION/WAIT_FETCH/WAIT_CSR` 归因必须落进 PMU manifest（W7）。
- `pipeline_validator/config.py` L495-506（`context_count 1..8`、`device_context_count`）与 `MAX_CONTEXT_COUNT`。**cs1_c1/cs1_c2 的 context 物理值登记在 W0-04 配置生成**，本计划只写语义（1 与 2 为两份物理配置，≥3 在 CS1 不跑）。
- `tests/coralflow/`（`test_rvv_add.py` 等）是**上游固定源码里的测试**，非当前仓库本地可执行 fixture；`pipeline_validator/tests/test_*` 才是当前 repo 的 Python 基线（不改）。
- `Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md`（r5）：`CTX-01..16`、`STA-01..07`、`FET-01..07`、`CFG-01..06`、`RET-01..08`、`SCH-01..11`、`ADM-05/09/10/11`、`MAT-12/19/20/21/22`、`FLT-01/02a/04/05`、`RST-07/09/11`、`T-01..T-68`（W2 主责行：T-02/04/05/06/07/08/09/10/11/12/30/34/36/39/44/48/64 中 UCE 侧子项，见 §6 映射）。
- `design/coral_nexus_tile/components/W2_Context_UCE.md`（W2 设计稿，§3/§4/§8 已逐条对应上游真实路径）；`design/coral_nexus_tile/Coral_Nexus_Tile_Design.md`（总体裁决/§4.4/§5.x）。

### 2.3 待创建文件（W2 拥有清单；不存在，不声称已存在）

- `hardware/coral_nexus/patches/w2/` 下 W2 系列补丁（编号 = 任务号×NN，如 `0201-core-ctx-ports/0202-rob-owner-tag/0203-rf-banks/0204-csr-ctx-bank` 归 W2-02；`0302-dispatch-ctx-mux/0303-fetch-ctx-mux` 归 W2-03；`0401-..0404-` 归 W2-04；`0501-native-loop` 归 W2-05；mutant 负例放 `patches/w2/mutants/`）。编号规则见 §1，W0-01 统一 apply；**W2-01 只产 contract YAML，不产 patch**。
- `hardware/coral_nexus/rtl/scalar/`：`Context.scala`（State + slot 管理 + `ctx_generation`）、`Scheduler.scala`（RR + stall attribution + per-ctx issue/retire/credit 计数）、`UopTag.scala`（`OwnerTag`/`RobTag` 定义）、`BankedRf.scala`、`BankedFRegfile.scala`、`CtxCsr.scala`（flags delta 挂 entry、CSR 局部互锁、decl CSR、`ctx_generation`）、`RetirementPartitionBank.scala`（按 Context 分区的 RetirementBuffer 外层 + 每 Context 分区匹配合键注入，与 W2-02 patch 配合，非零改动）、`Decode.scala`（`dispatch-ctx-mux` 用 per-Context 输入的 bundle mux：`perCtxReady`/`perCtxHalt`/`perCtxRvvIdle`）、`PerCtxFetchMux.scala`。
- `hardware/coral_nexus/rtl/scalar/doc/`：`oracle_requirements.yaml`、`ctx_ctrl.yaml`、`retire_record.yaml`、`rob_tag.yaml`（W2 落，W1-01 消费其字段语义；物理位地址/编码仍归 W1-01/csr_allocator）
- `hardware/coral_nexus/protocol/field_requests/w2.yaml`（W2 交 W1-01 合并的跨界字段请求，W1 唯一 merge 点）
- `hardware/coral_nexus/verification/unit/w2/`：`W2_02_rob_owner_spec.scala`、`W2_02_rf_bank_spec.scala`、`W2_02_csr_flag_delta_spec.scala`、`W2_03_fetch_epoch_spec.scala`、`W2_03_dispatch_prefix_spec.scala`、`W2_04_sched_park_spec.scala`、`W2_04_cancel_retire_spec.scala`、`W2_05_native_loop.py`（W7 cosim runner 兼容、`tile=1`/`tile=2` 两种 config，`cosim --scenario ...` 落 `out/coral_nexus/<run_id>/`）。
- `hardware/coral_nexus/patches/reqs/w2-to-w0-*.md`、`w2-to-w3-*.md`、`w2-to-w4-*.md`、`w2-to-w6-*.md`（接口/需求变更请求）。

## 3. 任务依赖表与 freeze gate

硬依赖（父合同"任务可串行但合同级 freeze 不互等 DONE"原则，见父合同 §2.3；-01 产物用 named artifact 交叉引用，不用"前置必须 done"表述）：

| Task  | 硬前置（合同级，不依赖对方 DONE）                                                                                                                                                      | 共同 freeze 产物                                                                                                                                                                                                                                             |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| W2-01 | 无硬前置（可立即开工）；消费 `pipeline_validator/config.py`、`parameters`、设计稿；不改 Python、不执行 build                                                                           | W2-01 裁决 `rob_tag.yaml`；W1-01 message/ctx ABI、W3-01 RVV 声明、W4-04 页授权、W6-01 事件、W5-01 helper ABI（W2 只固定需求输入/owner tag 摘要）                                                                                                             |
| W2-02 | `hardware/coral_nexus/patches/reqs/w2-to-w3-rvv-rob-owner.md`（W2-01 落）；`rtl/scalar/doc/rob_tag.yaml`（W2-01 落）；W0-01 workspace、W0-02 `env build`；W2-01 五 YAML freeze hash    | W2-02 落 `0201-core-ctx-ports.patch`（Core/Parameters 集成面）+ `banked_rf/rob/CSR` 模块；接口经 `banked_rf.yaml`（W2 内部，W3-02 消费 owner-tag 输入）                                                                                                      |
| W2-03 | W2-01 `rtl/scalar/doc/{rob_tag,oracle_requirements}.yaml` hash；W2-02 `banked_rf` 模块 hash                                                                                            | W2-03 落 `per-ctx fetch/issue/scheduler` 模块与 `perctx_issue.yaml`                                                                                                                                                                                          |
| W2-04 | W2-03 模块（issue 边界、owner 路由已存在）；W2-01 `rtl/scalar/doc/{ctx_ctrl,csr_flag}.yaml`；W3-01 RVV 声明 freeze；W6-01 ctx_ctrl 事件 freeze                                         | W2-04 落 `ordered_flags/ctx park`/`ctx cancel` 模块与 `ctx_owner_cancel.yaml`                                                                                                                                                                                |
| W2-05 | W2-04 模块；W0-04 `cs1_c1/cs1_c2` target（native ISA 依赖）；W1-03 cosim 骨架（C ABI 驱动，`libcn_cosim.so` 的 `bp_*` 最终 exports）；W4-02 L1、W5-02 基础 kernel、W6-02 WAIT 逻辑实现 | `hardware/coral_nexus/patches/w2/0501-native-loop.patch`（native scalar 闭环 wiring）与 `verification/unit/w2/W2_05_native_loop.py`。验收时区分：c1 是退化功能回归，c2 才同时承载 P1（仅 1 Context 活跃）与 PN（多 Context 活跃）两类 workload（需求 §16.3） |

依赖规则（对齐父合同）：**freeze 消费级**：W2-02..04 消费 `-01` freeze hash，不依赖对方任务 DONE；若对方 `-01` 未产出 freeze hash，W2-02..04 报 `BLOCKED` 并记录缺失项名。**W2-05 的闭环必须真（不用 stub 替身）**：native scalar closed loop 必须真实跑 `W4-03/04` 的真实 LSU/L1 端口、`W5-02/04` 真实 program install 映像、`W6-02` 真实 WAIT/terminal 端点 —— 这三者的真实产物是 W2-05 的**验收**；freeze hash 只是消费顺序（若 `W4-02/03/04`、`W5-02/04`、`W6-02` 中任何一项未真实产出，W2-05 报 BLOCKED，不报 DONE，也不降级成 Python/抽象 stub 冒充闭环）。

P1/PN 口径（parent 裁定，需求 §16.3）：P1 = 改造后同一硬件仅 1 个 Context 活跃；PN = 相同物理硬件多个 Context 活跃；两者都必须跑 `cs1_c2` target、同一 RTL hash，只变 workload 的活跃 Context 数。`cs1_c1` 只是退化配置功能回归，不得当 P1 基线参与多 Context 收益宣称。原核 P0 数值等价，控制/装载路径差异单列。

## 4. 共同设计决策（先于任务落盘，W2 单方决定）

### W2-D01 RetirementBuffer 采用 full 模式、保留原类名（W2-01 定，W2-02 落地）

- [A01] 判据：full 模式才有逐条 retire 的 `data/inst`/`vecWrites`，是 native ISA oracle 对拍的**唯一**可依赖观测点；mini 模式 `inst = 0, data = 0`，[A01] L72-84/L805-851 明示。
- **采纳 `enableVerification=True`（full RetirementBuffer）**：`core_mini_verification_axi_cc_library`/`vme_core_mini_verification_axi_cc_library` 已是 upstream target，W0-03/`W0-02 env build` 直接复用；cs1_c1/cs1_c2/P0 参考配 full 同一个 RTL hash。P0 full 参考若上游无现成 cocotb 顶层，属 W0-03 添加 harness（引用上游生成 target）的义务，不声称裸 label target 已存在。
- [INFERENCE] mini/full 两种模式的 `RetirementBuffer` 分区外层结构（每 Context 一份 `RetirementBuffer` 实例）相同，只差 data 字段是否有效；W2-02 只需保证 full 模式下的行为定义，mini 模式下仍编译通过（供 `env build` 裁判）——`mini` 是性能裁剪，不是行为定义。
- 后果：CS1 所有验证、native ISA 对拍、oracle 检查都基于 full RetirementBuffer 的 retire 流；P0 原核回归/比较也走同一 full 模式 target（同一 RTL hash，不混 mini/full）；c1/c2 与 P0 reference target 均取 full 配置；若 P0 需再跑 mini 系，须在报告中以 mini 参考单列（不与 full 数据混算）。P0 full target 若上游无现成 cocotb 顶层，属 W0-03 添加 harness 引用上游生成 target 的义务；不声称裸 label target 已存在；不宣称 mini 模式的性能/面积数字。

### W2-D02 每 Context 一份 GPR/FPR/CSR/ROB/取指缓冲，共享执行单元（W2-02/03 落地）

- 结构：`Regfile`/`FRegfile`/`RetirementBuffer`/`Csr`(flags+config+trap 部分)/`UncachedFetch`(PC/FRB/IB)/`scoreboard` 按 Context 复制；ALU/BRU/MLU/DVU/FloatCore/LSU/RVV 单元共享（`SCore.scala` 原结构）。
- 选择 Context 的 mux 是新增逻辑（`Decode.scala`/`SCore.scala` 的 `dispatch.io.*` 源头），不是复制 DispatchV2；`Scoredboard/banked RF` 的读写按 owner Context 过滤（W2-02）。
- 每 Context partition depth **暂沿用上游真默认 `retirementBufferSize==16`/`retirementLanes==8`（上游 P0 已实测、可实验沿用）作为 W2-02 初始实验 baseline**，不在计划里写成未实证的新 target 硬冻结；后续若 W2-05 高水位测到需要调整，必须先落 W0-04 配置证据再改名（不得直接宣称新尺寸已冻结）。证据：W2-02 单元 spec 断言（任一时刻一个 Context 的一组 `(ctx_reg, owner_tag)` 无双活，在 per-Context partition 内）；W2-03 断言（选中 Context 的 bundle 恰好是 4 lane 前缀）。
- [待review] 每副本的 RAM/logic 实现方式（`FRegfile`/`Regfile` 保持寄存器阵列 vs 换 SRAM）不在本计划定，留给 W2-02 落地后的 PPA 裁决；本计划只保证**语义等价**（同一 owner_tag 下读写序一致）。

### W2-D03 内部完成匹配键（W2-01 定，W2-02 落地）

| 完成来源                | 上游键                                      | 新键                                                                                                                                                                                                                     | 备注                                                                                                                                        |
| ----------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| scalar store complete   | `storeComplete.bits === pc`                 | `(ContextKey, owner_op_seq)`                                                                                                                                                                                             | [A01] L28/L587-589；`LSU` 发给 W2 的事件带 `ContextKey+op_seq`                                                                              |
| 非 RVV fault            | `bufferEntry.addr === mepc`                 | `(ContextKey, owner_op_seq)`                                                                                                                                                                                             | [A01] L493-495；FaultManager 输出带 owner                                                                                                   |
| scalar/float 长延迟写回 | `x.bits.addr === bufferEntry.idx` 寄存器号  | 同上；`flushed_scoreboard` 仍负责同号寄存器跨代写回的顺序                                                                                                                                                                | [A01] L460-483；owner=该 Context bank 的 dispatch 记录                                                                                      |
| vector/VME 写回         | `rob_tag`                                   | `(ContextKey, rob_tag)`（tag 布局 `rob_tag.yaml`）                                                                                                                                                                       | [A01] L482-483；W3-02 侧 RVV ROB 按 owner 划分                                                                                              |
| RVV fault               | `rob_tag`                                   | 同上                                                                                                                                                                                                                     | 同上                                                                                                                                        |
| scalar csr 写完成       | dispatch 拍内写 CSR（`Csr.scala` L365-443） | dispatch 拍内写 CSR（**仅管理类副作用 CSR**，如 `mscratch/mstatus 部分位段`）；**架构配置 CSR（`frm`/`vtype`/`vl`/`vtype snapshot/`mtype`/`vset\*`/`mset\*`）在本 Context 退休 commit**，dispatch 拍 snapshot 与提交分开 | flags 走 ordered delta（W2-04）；`frm/vtype/vl/mtype/vset*/mset*` 的 retire-commit 在 W2-04；管理类 CSR 单独清单（W2-04 步骤 2 附属子清单） |

- `rob_tag.yaml`（W2-01 定）固定 tag 位宽与编码：`ctx_bits = max(1, ceil(log2(contextCount)))`（`contextCount==1` 走显式常量 specialization，不生成 0-width 位段）；`rob_tag` 含 slot_generation（或等价 op_seq 区分位）防同 Context 同 slot ABA [INFERENCE：`Rob2Rt.rob_tag` 位宽兼容映射，具体位宽常量在 W2-01 产物 `rob_tag.yaml` 定，不在此处硬冻结]；给冲突检查清单和负向检查。
- **不做的事**：不把 `faultRetire.bits.mepc`/`storeComplete.pc`/`bufferEntry.idx` 等旧键留在任何 `else` 分支（W2-02 步骤直接替换）；不保留任何"PC 键作为兼容路径"。旧键错乱必须由 mutation test 显式暴露而不是兼容静默。

### W2-D04 flags delta 有序合并 / CSR 读互锁（W2-04 落地）

- 每条 RVV/FloatCore flags 更新（`fflags`、`vxsat`，不扩展其他位）由 `Csr.scala`/`RvvCore`/`FloatCore` 改为**带 owner(op_seq, rob_tag) 的 delta 流**，挂到该 Context 的 ROB 条目上（`RetirementPartitionBank` 每槽扩 5+1 位 flags 槽）。
- 合并点：**仅该条目自身退休时**与架构累加器合并（Context 内程序序），禁止"先到先合并"（CFG-04；设计 §3.6 已定，本计划落实）；**不要求 `resultUpdate.valid` 与本条目 retire 同拍**——早完成条目把 delta 驻留槽内、到 retire 才合并，不丢失。
- CSR 读互锁（局部、Context 内程序序）：`csrr fflags/vxsat/vl/vtype` 及依赖 `vset/mset` 的向量 op dispatch 要求（a）同 Context 内没有在途 flags delta 尚未合并、（b）`vset/mset` 已 retire（CFG-01..03）。**不做 forwarding**；`isVxsatOrVcsr` 的跨 Context `rvvIdle` 判据删除，只看本 Context 分区（CFG-05）。scalar CSR 写（non-flags）维持 dispatch 拍原动作。
- 归属：flags delta 记录发起 Context（CFG-04），delta 的 op_seq 来自 dispatch 时 owner tag（W2-02 生成的 `owner_op_seq`）。
- 取消/清零：Context 槽复用时 flags 累加器/ROB 槽 flags 槽一并清零并计数；`ctx_generation` 防旧 delta 串代（CTX-08/09）。

### W2-D05 fetch epoch 定向（W2-03 落地）

- `UncachedFetch` 的 `flushTx`、`instructionBuffer.io.flush`、`fetch.io.iflush` 改为带 `(ContextKey, fetch_epoch)` 的定向取消；`fetch_epoch` 每 Context 单调递增、与 `ctx_generation` 分域。
- `RvviTrace.scala`/`RvvFrontEnd.sv`（RVV 指令 trace 与 `FetchInstruction`）带 per-Context 头（owner_ctx, op_seq），W7 消费。
- `coreIdle` 拆成 per-Context idle + Tile 级 quiescence 聚合（`CoreAxi` 关钟条件改造：W2 是实体编辑 owner，W6 消费 per-Context 状态驱动 `gate_ctl`），`SCore.io.halted/wfi` 相应拆分（W2-04）。

### W2-D06 Park / cancel / fail-stop 三者分离（W2-04 落地）

- `park`（WFI/region 停靠/WAIT 事件）语义只停 issue/调度，**不停已完成工作的接受与 retire**（SCH-07；`CoreAxi` 关钟不再使用）；wait 记录本体归 W6，W2 只持 WAIT 状态位。
- `cancel`（来自 W6 的 `CANCEL_TASK`）：W2 内部动作 = 停发（该 ctx 分区头部不再调度）＋"未接受 payload 丢弃/已接受在途继续完成或隔离"，**不清其他 Context 的 RF/ROB/LSU 项**；隔离/取消确认归 W6/W4。
- `fail-stop`（任一 Context FAULT → Tile 停发）：发停发给本 Tile 上下文选择器（SCH-11、FLT-01，validator `tile.py` 口径一致性见 §6 映射），fault 归属三元组 `(task, ctx, op)` 上报 W6；flush 链仍只清**本 Context** 分区状态，其他 Context 状态保留（W2-D02 副本结构天然保证）。**本阶段无 FLT-02a 局部恢复**。

## 5. W2-01 Owner/指令/配置接口

### 5.1 Scope（对齐设计 §4 + 父合同 §3）

定死三个"机器可读合同产物"和三类"内部接口 bundle"，供 W2-02/03/04 消费与 W3/W4/W6/W7 认领。物理位编码不在此定：位宽只给**派生规则式**（如 `ctx_bits = max(1, ceil(log2(contextCount)))`），常量由 W1-01 ABI 依据"最大在途数+最大响应存活期"裁决后注入 `Parameters.scala`，W2 不凭空造硅位编码 [INFERENCE：父合同 §4。本计划不给具体位数值，只给约束表达式与负向检查]。

### 5.2 步骤（机械可执行）

1. 创建 `hardware/coral_nexus/patches/reqs/w2-to-w1-message-contract.md`：列出 W2 从 W1 边界消费/输出的**全部内部消息**（`ContextKey`、`owner_op_seq`、`ACCEPT/COMMIT/RETIRE_READY/EXEC_DRAINED/TASK_COMPLETION`、`cancel/fetch_epoch`），每个消息给 (方向, 上游真实路径锚点, 字段序号, 背压策略, 谁是 owner)；同时把字段清单写成 `hardware/coral_nexus/protocol/field_requests/w2.yaml`（W1-01 单点合并、只写请求不写实现）。交 W1-01 合并进 ABI；W2 不定义位。
2. 创建 `hardware/coral_nexus/rtl/scalar/doc/rob_tag.yaml`：定义 `(ContextKey, rob_tag)` 复合 tag 的**字段集与编码约束**（如 `rob_tag_width = log2Ceil(retirementBufferSize)`、`ctx_bits`、`owner_op_seq_width` 由 W1-01 ABI 裁决；W2 补"冲突清单"：`retirementBufferIdxWidth` 上游是寄存器号宽度 [A01] `Parameters.scala:142-148`，与 `rob_tag` 宽不同，两者混用属于 INV-11 不一致）。此 YAML 是 W2-02/03/04 与 W3-02 的**唯一** tag 键来源；schema 用 versioned YAML，字段有 `required:true/false`、owner、消费方。
3. 创建 `hardware/coral_nexus/rtl/scalar/doc/ctx_ctrl.yaml`：定义 W2 从 dispatch 向 W6 发出的 **context-local 控制指令**（`WAIT/WAITALL/SIGNAL_PHASE/RET/EXEC_DRAINED/region-START/region-END/park-request`），每个字段 (op, operand kind, side-effect, 顺序语义, 归属 Context)；W6-01 消费其终态语义；`ctx_generation` 字段防旧代。
4. 创建 `hardware/coral_nexus/rtl/scalar/doc/retire_record.yaml`：定义 per-lane per-retire 的字段（`ctx_key, owner_op_seq, pc, inst, rd_wr(addr/data/mask), trap, flags_delta, rob_tag, uop_pc`），供 W7-02 native ISA oracle 与 W7-01 trace schema 消费（对齐 [A01] L805-851 与 `RvviTrace.scala` 已有字段，补 per-Context 侧字段）；W7 只读，不改字段。
5. 创建 `hardware/coral_nexus/rtl/scalar/doc/oracle_requirements.yaml`：定义 independent-ISA oracle 需要满足的最小输入契约（`ELF+entry_pc+可写段布局+per-Context 起始 CSR/RF 初值+可观察事件清单`），并给"输出契约"= per-lane `retire_record` 序列 + 架构终态（RF/CSR/内存字节）+ flags。交 W7-02 立项依据；**W2 从此不把 Python 序列对拍当指令 oracle**（总体合同 C4 数值 oracle 独立性口径，见 [../plan.md](../plan.md)）。
6. 创建 `hardware/coral_nexus/patches/reqs/w2-to-w3-rvv-rob-owner.md`、`w2-to-w4-lsu-owner-tag.md`、`w2-to-w6-ctx-quiescence.md`：明确 W3-02 侧 RVV ROB owner 划分要求、W4-03/W4-04 侧 LSU 访存带 owner+op_seq、W6-04 侧 `WFI/`park+quiescence 接管（[A01] `CoreAxi.scala` L109 行为）。

### 5.3 输出/接口（W2-01 deliverables）

- `rtl/scalar/doc/{rob_tag,ctx_ctrl,retire_record,oracle_requirements}.yaml`（版本化 schema，带 owner/consumer）；W1-01 消费 `rob_tag.yaml`/`ctx_ctrl.yaml` 字段名与语义；W7-02 消费 `retire_record.yaml`+`oracle_requirements.yaml`。
- `patches/reqs/w2-to-w{1,3,4,6}-*.md` 汇报给 parent。

### 5.4 输出格式与失败样例

每个 YAML 首行 `version: 1`；字段 `name,type,width_rule,owner,consumers[...],required`；冲突清单段（如 `retirementBufferIdxWidth` vs `rob_tag` 宽度、与已有 CsrAddress 冲突、`RVV ROB_TAG_WIDTH` 由 SV 写死 vs params 派生 [A01-`rvv_backend_define.svh` L135-138]）。**失败样例**：若 `rob_tag_width` 与 RVV `ROB_TAG_WIDTH` 宽度不一致，`rob_tag.yaml` 的 `checks` 段落列出来源文件+行号，W1-01 gate `sources verify` 必须检出；W2-01 自身也要写一个"负向检查"（故意给一个冲突值并断言 W2-02 拒绝），不依赖对方 gate 兜底。

### 5.5 优先级：P0（G1 gate 前必须可 hash）；其他并行：与 W1/W3/W4/W6 -01 同拍。

### 5.6 W2-01 的独立验证（不靠对方 gate）

- `python3 -c "import yaml,hashlib,sys; [ (lambda d: (print(f), hash(open(f,'rb').read()))) for f in glob.glob('hardware/coral_nexus/rtl/scalar/doc/*.yaml') ]"`【未来命令，W0 创建后：`bash hardware/coral_nexus/scripts/cn.sh sources verify --include rtl/scalar/doc`，检查 YAML 可解析+owner/consumer 非空+引用字段存在】
- 负向检查：手工构造两个同名字段（如 `retire_record.ctx_key` vs `ctx_ctrl.ctx_key`）不同 width_rule，断言 `checks` 检出且 `cn.sh sources verify` 非零退出。
- 退出条件：5 个 YAML+3 个 reqs 文件全部落地、版本号、owner/consumer 字段齐、负向冲突检查可复现。**不做**：不写 RTL、不跑 build（W2-02 才开始）。

## 6. W2-02 RF/CSR/scoreboard/ROB 身份

### 6.1 Scope

把 W2-D02/W2-D03/W2-D04 落地：per-Context 的 `Regfile`、`FRegfile`、`Csr`（flags/config/trap 部分）、scoreboard、`RetirementBuffer` 分区（W2-D01 full 模式），并完成 W2-D03 匹配键替换。**RVV ROB 的 owner 划分与 RVV→scalar 接口的 owner 字段消费**也在此步接入（消费 W2-01 接口冻结的 `rob_tag.yaml`、`banked_rf.yaml`）。

### 6.2 步骤

0. 落 `hardware/coral_nexus/patches/w2/0201-core-ctx-ports.patch`：改 `Core.scala`/`SCore.scala`/`Parameters.scala` 的 IO，暴露 per-Context 状态端口（`ctx_count, ctx_select, per_ctx_retire_ready` 等，字段按 `rob_tag.yaml`/`ctx_ctrl.yaml` 派生，不写常数）。此 patch 是 W2-02/03/04 消费的**集成面**，先于本步其余源码新增。
1. 落 `hardware/coral_nexus/rtl/scalar/UopTag.scala`：`OwnerTag(ctx_key, owner_op_seq)`、`RobTag(ctx_bits, rob_tag+slot_generation)`；`ctx_bits = max(1, ceil(log2(contextCount)))`（`contextCount==1` 走显式常量 specialization，不生成 0-width 位段）；`rob_tag_width = log2Ceil(retirementBufferSize)`；`rob_tag` 含 slot_generation（或等价 op_seq 区分位）防同 Context 同 slot ABA；`owner_op_seq_width`/`slot_generation` 位宽从 W1-01 ABI（W2-01 产物派生规则）注入。所有 per-Context 状态键都从这里 import。
2. 落 `hardware/coral_nexus/patches/w2/0202-rob-owner-tag.patch`：改 `RetirementBuffer.scala`
   - 每 slot 追加 `OwnerTag` 字段（L72-84 `Instruction` 扩展），
   - 替换 L28/L587-589（store 匹配）、L493-495（fault 匹配）、L467-476（scalar/float 长延迟写回匹配）、L482-483（vector/VME 写回）四处，匹配键统一 `(ctx_key, owner_op_seq)`（vector 侧 `(ctx_key, rob_tag)`），
   - 保留 L106-110 "bundle 内首拒后不得再接受"断言与 L233-240 的 `CircularBufferMulti` 结构不动，
   - `RetirementBuffer` 输入 `io.writeDataScalar/writeDataFloat/writeDataVector` 侧加 `OwnerTag`（上游 `Interfaces.scala` 对应 bundle 扩展），`io.storeComplete`/`io.fault` 侧换 `OwnerTag`（LSU/FaultManager 的输出侧改造在 W4 patch 中同步，W2 patch 只消费新字段类型），
   - flush（L725/L733-734）只清**选中 Context** 分区内的条目（per-Context `trapRetired` 派生，跨 Context 误清属 RET-06 违例）。
3. 落 `hardware/coral_nexus/rtl/scalar/BankedRf.scala`：`contextCount` 份 `Regfile`/`FRegfile`，读地址只在选中 Context bank 上有效，写回按 `OwnerTag.ctx_key` 派发；scoreboard 读写都在 bank 内部（上游结构 L96/L110-135 保留，bank 内化）；`linkPort` 输出按选中 Context 复制（上游 L254-267）。
4. 落 `hardware/coral_nexus/patches/w2/0203-rf-banks.patch`：`SCore.scala` 里 `regfile = Regfile(p)` 替换成 `BankedRf`（写口 per bank 路由，`SCore.scala` L296-323 的写口仲裁与 `assert`（L310-328）保持 per bank 不变）；`Lsu.io.rd/MLU/DVU/FloatCore` 的写回调由 W2 patch 消费 owner 字段（LSU/MFE 内部结构是 W4；本 patch 不改 `Lsu.scala`）。
5. 落 `hardware/coral_nexus/rtl/scalar/CtxCsr.scala`（+ `patches/w2/0204-csr-ctx-bank.patch`）：`Csr` 的 per-Context 副本（`mhartid` 输出 `(tile_id, ctx_slot)` 编码值，写 `mcycle/minstret` 保留原行为但计数 per Context 分区；trap CSR `mtvec/mepc/mcause/mtval` per-Context；`fflags/frm/vstart/vxrm/vxsat/vl/vtype/mtype` 保留 dispatch snapshot 机制，flags 增量在 W2-04 改 ordered delta；本步先**保留 flags 直写路径**但加 `owner_op_seq` 标记，W2-04 再移除直写），并按 `rob_tag.yaml` 补 `ctx_key`。
6. 落 `retirement_buffer` 分区外层 `hardware/coral_nexus/rtl/scalar/RetirementPartitionBank.scala`：每 Context 一个 `RetirementBuffer` 实例（W2-D01 full），`enqPtr/deqPtr/credit/nRetired` per-Context，共享 `storeComplete/writeData*/fault` 输入按 `OwnerTag` demux 到对应分区；`flush`（trapRetired）只作用于本 Context 分区（RET-06）。
7. 落 `hardware/coral_nexus/verification/unit/w2/W2_02_rob_owner_spec.scala`（ChiselSim `AnyFreeSpec`，参照上游 `FetchControlSpec.scala` 风格）：
   - 同一 image、两个 Context 同 PC store 并发、同 op_seq 键：断言（运行态行为断言，不采用源码 grep/字符串断言）：只有带相同 `OwnerTag` 的写回/完成才置位对应 slot；PC-only 匹配的旧行为已不存在，由 mutant 检验。
   - 老键负例：构造 `OwnerTag` 不同但 PC 相同，断言不被错误 retire（限定在“同映像同 PC 并发在途”前提；不 over-claim 同 Context 内任意同 PC 指令一定错对，该场景在 W2-05 单独记录）。
   - "任一时刻同 (`OwnerTag`,`owner_op_seq`) 无双活"不变量 + "分区头 A 未完成不挡 B retire"（T-09/RET-03）。
8. 落 `hardware/coral_nexus/verification/unit/w2/W2_02_rf_bank_spec.scala`：两 Context 同号寄存器、长延迟写回（MLU/DVU/FloatCore/LSU）与 scoreboard 断言（STA-02/03/06）；负例：把 owner 字段 tie 成常量，断言 spec 失败。
9. 落 `hardware/coral_nexus/verification/unit/w2/W2_02_csr_flag_delta_spec.scala`：Csr per-Context 副本，断言 `ctx_generation` 换代时旧 delta 被拒、flags 累加器被清、CSR 读互锁（W2-04 真正的 ordered 合并在 W2-04 落，本步先锁定**旧键拒绝**与**换代清零**两个不变量）。

### 6.3 输入/输出与验收

- 输入：W2-01 五个 YAML 的 hash（`PROTOCOL_FREEZE.json` 路由，W2-01 只产 contract，patch 全部在 W2-02+）；W0-01 workspace + W0-02 `env build` 环境；上游源码锚点 §2.1。
- 输出：`patches/w2/02xx-{core-ctx-ports,rob-owner-tag,rf-banks,csr-ctx-bank}.patch`（W2-02 命名，见 §1/§3 表）、`rtl/scalar/{UopTag,BankedRf,CtxCsr,RetirementPartitionBank}.scala`、三个 spec。**W2-02 完成后**这些 patch/源文件是 W2-03/04/05 的输入。
- 验收命令【未来命令，W0/W1 创建后】：
- `bash hardware/coral_nexus/scripts/cn.sh build --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/w2-02-build/`（退出 0 = 构建成功）
- `bash hardware/coral_nexus/scripts/cn.sh unit --suite w2 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/w2-02-unit/`（spec 通过 → `out/coral_nexus/w2-02-unit/result.json` per-case pass/fail）
- `bash hardware/coral_nexus/scripts/cn.sh sources verify`（patch 落地 hash 进 manifest；不使用 `--include` 后缀）
- 不使用 grep/`ls`/源码存在检查当 PASS；验收以运行态/字节/计数证据为准。
- 负例先行：spec 8/9 的"tie owner 字段成常量"用例必须**先失败后通过**（先跑负例断言失败，再跑 patch 后版本断言成功），并落 `result.json` 记录两类结果。
- 退出条件：spec 全过、patch apply 后 `grep` 源码断言 `OwnerTag` 已替换 PC/mepc/regnum 三类匹配、无新 warnings；未跑 gate G2（那是 W2-05/W7 侧）。

### 6.4 失败/成功/取消/背压/旧代行为

- 失败：spec 编译错/负例未失败/负例通过 ⇒ 任务 FAIL，不改接口，直接向 parent 报 `W2-02-blocked:<原因>`。
- 成功：上述 all-pass + patch list + spec 日志归档 `out/coral_nexus/w2-02/`。
- 取消（W6 触发）：W2-02 只做本地 `trapRetired` per-Context 清理（步骤 2 最后一句），不写 W6 记录；跨包隔离测试归 W2-04（W2-D06）。
- 背压：无（W2-02 不引入新队列）；`RetirementPartitionBank` 的 `nSpace` per-Context，不新增共享 FIFO。
- 旧代：`ctx_generation` 进 `UopTag`，旧 owner 的迟到写回/完成被拒并计数（CTX-08/09/INV-15；spec 9 覆盖）。

## 7. W2-03 Fetch/Issue/PartialAccept

### 7.1 Scope（对齐设计 §3.3、§3.5 + W2-D05）

调度器 RR 选 Context、每 bundle 4 lane 连续前缀、每 lane 的接受/拒绝/取消与 `fetch_epoch` 定向；`partial accept` 的 payload 稳定。这是 W1 分相合同（P0..P4）的 RTL 侧消费点。

### 7.2 步骤

1. 落 `hardware/coral_nexus/rtl/scalar/Scheduler.scala`：从 W6/W1 侧收 `per_ctx_ready`（READY/park 语义），输出每周期一个选中 Context + `issue_valid`；RR 指针推进规则 = **选中才 next，未选中不推进**（`_select_context` 语义 tile.py L635-653）；head 必要预译码 + 分级 resource probe 是 Scheduler 输入（W2-03 定接口，消费方 W3/W4/W6）；每 Context 维护 stall 原因计数（`WAIT_EVENT/WAIT_ENGINE_QUEUE/.../WAIT_REGION (新)/WAIT_CSR (新)/WAIT_FETCH (新)`，序进 PMU manifest，OBS-07）。park 语义（WFI/region 停靠）只挡发射，不清 RF/ROB（W2-D06）。
2. 落 `hardware/coral_nexus/patches/w2/0302-dispatch-ctx-mux.patch`：`Decode.scala`（DispatchV2）的输入 `io.inst/scoreboard/fscoreboard/halted/interlock/retirement_buffer_*/rvvIdle/lsuActive` 全部由 `Scheduler` 选中 Context 的 bank 输出提供（W2-D02 的 mux）；`io.jump/isCsr/canDispatch/logic` 不动；`isVxsatOrVcsr` 的 `rvvIdle` 换成"本 Context 分区 idle"（`Decode.scala` L976-993 的 `rvvIdleOrNotVxsat` 接入 per-Context 信号，CFG-05）。
3. 落 `hardware/coral_nexus/patches/w2/0303-fetch-ctx-mux.patch`：`UncachedFetch.scala` 的 `FetchReorderBuffer`（L39-322）与 `FetchControl`（L384-538）改造为 `(ContextKey, fetch_epoch, fetch_reqid)` 键（W2-D05）；`instructionBuffer.io.flush`（L577）换成"本 Context 定向 flush"；`SCore.scala` L128-131 的 `iflush`、L166-171 的 `branch` 按选中 Context 派发；`ibuffer`/`FRB` 按 Context bank 化（`maxConcurrentTx=2` per Context）。**`UncachedFetch.scala` 物理端口仲裁（ibus 背压）归 W4-02**：W2 在 `patches/reqs/w2-to-w4-ibus-port.md` 提"每 Context 取指请求/响应带 `OwnerTag+epoch+reqid`、每 Context 2 项在途"需求，`maxConcurrentTx` per-Context 值由 W4-02/W5-01 冻结，本步骤先把 `UncachedFetch` 的键改造落完，不动 ibus 物理。
4. 落 `hardware/coral_nexus/rtl/scalar/PerCtxFetchMux.scala`（若步骤 3 的 per-Context `UncachedFetch` 化需新的顶层拼接）：把 per-Context `FetchReorderBuffer`/`instructionBuffer` 的输入按 `Scheduler` 的选中信号 mux，写回按 `ContextKey` 派发。
5. 落 `hardware/coral_nexus/verification/unit/w2/W2_03_fetch_epoch_spec.scala`：
   - 两个 Context 独立 PC/epoch/缓冲，同拍 A 分支重定向，`fetch_epoch[ctxA]++`；构造迟到响应（旧 epoch）断言被拒 + `ctx_generation`/epoch 计数递增、信用归还（`CtxFetchCredit` 计数）、B 缓冲不变（CTX-09/FET-02/T-07 变体）。
   - 饥饿：A 长延迟 miss 连续 1000 拍，B `wait_fetch` 间隔有界（FET-03/FET-07），断言 `per_CTX fetch service gap ≤ 阈值`（阈值由 W4-02/W5-01 仲裁冻结，spec 用参数读入，不给默认值）。
   - 取消：A 分支在 fetch 在途时取消（W6 `CANCEL_TASK`），断言 A 的 in-flight fetch 被定向丢弃（`FRB.flush` 带 `OwnerTag`）+ 信用归还 + 无旧代写回（T-08/T-48 类）。
6. 落 `hardware/coral_nexus/verification/unit/w2/W2_03_dispatch_prefix_spec.scala`：
   - 每周期一个 Context 发射 bundle，lanes 全来自同 Context 且为程序序连续前缀（对应上游 L106-110 断言 + L160-171 "bundle 内 scan"），断言 `io.inst[i].fire` 若 lane i 不 fire 则 i+1..3 必不 fire（SCH-01/06/T-11）。
   - Partial accept：lane0..1 接受、lane2..3 因 unit busy 不接受，断言 (a) `owner_op_seq` 只对 lane0..1 递增、(b) PC 按 lane0..1 实际长度推进、(c) lane2..3 保持 payload 稳定直到接受或取消（SCH-05/T-11）；负例驱动 `unit_ready` 交替抖动 ≥100 拍，断言无变形（validator `T-11` 的 rtl 对应）。
   - 长度 ≠ 4：分支重定向拍 (`branch valid`) 断言 lanes 后缀全不发射（`isJump/jumped` scan 行为，上游 L407-431）。
   - RR：构造 `ctx_ready` 随机抖动 + A 头 wait_region 长期停 + B 独立就绪，断言 B 连续服务且 A 恢复后 RR 从 A 槽位 +1 继续扫描（SCH-02/T-34）。
7. 落 `hardware/coral_nexus/patches/reqs/w2-to-w1-issue-phase.md`：把 issue 的接受/完成/计数（`uce_issue/uce_launch_attempt/uce_launch_accepted/retirement_wait`）锚点映射到 Python `_select_context/_probe_head` 语义（tile.py L590-653/L1016/L1046/L1298），交 W7-01 做 trace 锚点（W2 不写 trace 生成器）。

### 7.3 输入/输出与验收

- 输入：W2-02 输出（banked RF/ROB/CSR 已在，owner tag 可生成）；W1-01 message 契约（issue/phase 逻辑字段）；W4-02/W5-01 物理端口仲裁需求（仅需求文件，非阻塞）。
- 输出：`Scheduler.scala`、`PerCtxFetchMux.scala`、patch `0302-{dispatch-ctx-mux,fetch-ctx-mux}.patch`、两个 spec、`perctx_issue.yaml`（W2-03 输出的 issue 边界表，W1-01/W7-01 消费）。
- 验收命令【未来】：`bash hardware/coral_nexus/scripts/cn.sh unit --suite w2 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/w2-03-unit/`（spec 通过 → per-case result.json）；`bash hardware/coral_nexus/scripts/cn.sh sources verify`（patch 落地 hash 进 manifest）。无 grep/源码存在当 PASS。
- 接口 freeze：`perctx_issue.yaml` 的 `issue/accept/partial_accept` 字段与 `ctx_ctrl.yaml`/`rob_tag.yaml` 无重名冲突检查（W2-01 同款 YAML 校验脚本）。
- 负例先行：spec 5/6 的负例必须先失败（旧 PC-only flush、非前缀发射）再通过。

### 7.4 失败/取消/旧代

- 失败：spec 6 的 partial-accept 负例不能稳定重现 payload 稳定或非前缀发射 ⇒ FAIL 并上报 parent；不通过改 spec 预期值来"凑通过"。
- 取消与重定向分域：W2-03 的 branch flush 只 bump 本 Context `fetch_epoch` 并丢弃本 Context 错误路径 fetch；**不做 Task 级取消/整代失效**；`CANCEL_TASK`（W6）才走整代域（`ctx_generation`/`owner_op_seq` 域），与 branch flush 在 tag/seq 域上按 `rob_tag.yaml` 与 `ctx_ctrl.yaml` 分开，不混用。
- 旧代：`fetch_epoch/ctx_generation` 拒收旧包（CTX-09，spec 5/负例覆盖），旧包只计数不写状态（INV-15）。

## 8. W2-04 Ordered flags / 局部取消 / 退休

### 8.1 Scope（对齐设计 §3.4 后半、§3.6、§3.7 + W2-D04/W2-D06）

把 W2-02 保留的 flags 直写改成 ordered delta + retire 拍合并；CSR 读/pending-config Context 局部互锁；WFI/park；局部/Tile 级 cancel；`EXEC_DRAINED`→`RETIRE_READY`（W6 主导、W2 配合）；Context 槽保守复用。

### 8.2 步骤

1. 落 `hardware/coral_nexus/patches/w2/0401-ordered-flags.patch`：`Csr.scala` flags 写路径（L365-443 的 `fflagsEn`、`fcsr` 写、`io.float.in.fflags`/`io.rvv.fflags` 累加）改为"**每 ROB 条目 5 bit fflags delta + 1 bit vxsat delta**"（W2-02 已扩 `RetirementBuffer.Instruction`），合并规则 = **该条目退休（`deqReady` 扫描中本 entry 被选中 retire）时** `fflags_arch := (fflags_arch | entry.fdelta)`，不要求 `resultUpdate.valid` 与 retire 同拍（早完成条目的 delta 驻留槽位等待，不丢）（CFG-04，禁止先到先合并）；软件 `csrw fflags/fcsr/vxsat` 的**替代动作**也在本条目自身 retire 拍合并（把写值当条目 delta，先落槽位再在 retire 拍生效），不是 dispatch 拍直接写累加器。FloatCore/RvvCore 的 `io.*.fflags.valid` 改成**带 owner 的 delta 上报**（W3-02 接同款接口，FloatCore 改造由 W2-02 拥有，本 patch 落）。
2. 落 `hardware/coral_nexus/patches/w2/0402-csr-interlock.patch`：`Decode.scala`（DispatchV2）的 CSR dispatch 拍读互锁改造：`csrr fflags/vxsat/vl/vtype/mtype`（L683-708 `canDispatch`、L976-993 vxsat 读、L502/L530/L544 `configInvalid`）增加"**同 Context 内没有在途 flags delta / `vset/mset` 未 retire**"两个 gate；**`vset*/mset*`/`frm`/`vtype`/`vl`/`mtype` 架构配置在本 Context retire-point commit（W2-04），dispatch 拍 snapshot 供 younger op；管理类副作用 CSR（如 `mscratch/mstatus` 特定字段）单独走 dispatch 拍（per-CSR 分类清单在本 patch 内落）**；替代"全局 CSR 等 ROB 空 & `rvvIdle`"（CFG-01..05；`retirement_buffer_empty`/`rvvIdle` 保留作为非 CSR 指令的既有 fence/idle 语义，不删）；`isVxsatOrVcsr` 的跨 Context `rvvIdle` 判据删（W2-03 已在 per-Context 侧生效，本步删除旧路径）。
3. 落 `hardware/coral_nexus/rtl/scalar/Context.scala` Context FSM/槽位表：`FREE→HELD→ACCEPTED/INIT→READY→WAIT_EVENT/WAIT_RESOURCE(sub: wait_engine_queue/wait_stream/wait_region/wait_fetch/wait_operand/wait_dest)→DONE/FAULT`（validator `_UCEContextState` tile.py L55-64 的超集）；`ctx_generation` 递增与旧代拒收;`park` 投影位 + W6 wait 只读摘要（wait 记录本体在 W6，W2 不落 [对齐设计 owner 表]）。
4. 落 `hardware/coral_nexus/patches/w2/0403-park-csr-quiescence.patch`：`Csr.scala` `wfi`（L437-443）与 `halted/wfi`（L243-245）语义拆为"per-Context park 输出＋Tile quiescence 聚合"两件事；`CoreAxi.scala` L109 `(!csr.io.cg && !core.io.wfi)` 钟控条件移除，**本 patch（W2 是 CoreAxi 实体唯一编辑者）**：删除直接耦合 + 暴露 tile 级 `gate_ctl` 输入接口（信号名/语义由 W6 提供、由 W6-04 消费并按 per-Context 状态驱动；W2 不写 quiescence 算法本体）。`SCore.scala` L147 `dispatch.io.halted := csr.halted||csr.wfi||dm.*` 拆 per-Context + dm 聚合。
5. 落 `hardware/coral_nexus/patches/w2/0404-cancel-iso.patch`：接收 W6 `CANCEL_TASK(generation)`，动作 = 本 Context 停发 + 已接受在途任务正常完成或被隔离、未接受 payload 丢弃（`dispatch-ctx-mux` 上的 `cancel_mask`）；RF/ROB/LSU 的清理只按 owner 进行，不误清他 Context（RET-06，spec 验证）；`CoreAxi` gate_ctl（W6-04 算法 owner、W2 实体编辑）与 W2 停发的耦合用 `quiescence` 单向信号（W1-04 消费）。fault 路径 (`FaultManager` per-Context + `flt-01` Tile 停发) 在本步统一到 fail-stop 单一路径，`FLT-02a` 明确不实现。
6. 落 `EXEC_DRAINED` 本地判定：`RetirementPartitionBank` 空 + per-Context launch 事件完成计数（validator `_complete_context` tile.py L1298-1320 口径）+ W6 wait/region 投影空闲 ⇒ 置 `EXEC_DRAINED`（W6 内部通知输入名，逻辑名，不做终态发布）；W6 推进 RETIRE_READY/`RETIRE`/`TASK_COMPLETION` 的消费由 W2-05/W6-02 真实联仿覆盖（本步只做本地判定与 W6 消息字段对齐，`ctx_ctrl.yaml` 已固定）。
7. 落 Context 槽复用 gate：旧 Task 终态 + 本地义务闭合 + `ctx_generation` 递增 + RF/CSR/ROB/flags 桶清零 + 旧 owner 拒收计数清（CTX-15 三条件；保守实现等 TASK_COMPLETION 接收，本计划采纳保守版，等 W6-02 的 TASK_COMPLETION 接收位）。
8. 落 `hardware/coral_nexus/verification/unit/w2/W2_04_sched_park_spec.scala`：
   - A 头 wait_region 停靠，B 独立运行（T-28/T-62 的 UCE 侧），断言 A 年轻指令零发射、B 正常服务、`region_done` 唤醒 A 后在原头继续（W3 授予信号只读投影，`MAT-12`）。
   - A 头 `WAIT_EVENT` park，B 补位（T-04）；同周期唤醒/发射的优先序 = 完成回写→可退休→新授予→状态推进（SCH-08，spec 用固定测试序列逐拍断言）。
   - WFI：A WFI park、B 在途、B 关钟不触发（`CoreAxi` patch 后）；全 Context park + 后端在途 ⇒ quiescence 拉低（T-27/ISA-04/CLK-01..06）。
9. 落 `hardware/coral_nexus/verification/unit/w2/W2_04_cancel_retire_spec.scala`：
   - flags 有序合并：`older FP→csrw fflags(清 0)→younger FP`、年轻者早完成（T-06/CFG-04），断言"三条全退休后架构 fflags 恰为 younger delta"，**逐条 delta 记录落 spec 日志**（本条目就是设计 §8 表中的行）。
   - CSR 互锁：本 Context 未合并 delta/未 retire `vset` ⇒ `csrr fflags` 不发射、读值在合并后 = 新值；其他 Context CSR 读不被本 gate 阻塞（CFG-05）。
   - CANCEL 局部性：cancel A，断言 B 的 scoreboard/ROB 位图逐 bit 不变（RET-06 硬不变量）。
   - 独立退休：A 队首因 store/fault 迟回被延迟期间，断言 B 的退休/完成仍推进且 retire 记录与 oracle 单跑一致（结果序与条目内容对拍，不以时间戳差为唯一判据）；配合 bounded-RR 机会计数：N 个 eligible Context 任一无限窗口内每个 eligible Context 的 issue 机会数 = `k*N + c`（c 小常数；RR 精确 N eligible 机会计数，不给泛化阈值）。负例：把 retirementPartitionBank 退化成共享 FIFO head，该负例必须被 oracle 对拍或机会计数之一检出失败。
   - 槽位复用：A 终态→`ctx_generation++`→旧 owner 注入旧 delta/旧写回/旧信号 ⇒ 全部拒收 + 计数 + 信用守恒（T-08/CTX-15/W2-D06）。
   - 饥饿：B 长期饱和（如 `div` 长链 or 长 store 链），断言 A `issue_service_interval ≤ 仲裁上限`（SCH-03），上限由 W4-02 仲裁冻结提供，未冻结前 spec 记录原始数据、不判 PASS。

### 8.3 输入/输出与验收

- 输入：W2-02/03 的 all-pass 输出；`W3-01`（RVV 声明/owner 接口）与 `W6-01`（事件/wait 记录 owner）的 contract freeze hash；`W4-03/04` 的 LSU owner-tag 变更请求接收（非阻塞，若缺失本步只记录 BLOCKED 項）。
- 输出：patch `0401-{ordered-flags,csr-interlock,park-csr-quiescence,cancel-iso}`、`Context.scala`、两个 spec、`ctx_owner_cancel.yaml`（cancel/park/quiescence 字段，W6-04/W1-04 消费）。
- 验收命令【未来】：`bash hardware/coral_nexus/scripts/cn.sh unit --suite w2 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/w2-04-unit/`（spec 通过）；`bash hardware/coral_nexus/scripts/cn.sh sources verify`。无 grep/源码存在当 PASS。
- 负例先行：spec 8/9 的"关闭互锁/关闭局部 cancel/PC-only 匹配旧键复活"三条负例，必须先失败；`ctx_owner_cancel.yaml` 的字段 freeze 有 YAML schema 校验（W2-01 同款）。

### 8.4 失败/成功/取消/旧代

- 失败：flags 有序负例未重现（older delta 抹掉）⇒ FAIL，上报 parent；`SCH-03` 饥饿上限未冻结 ⇒ 该用例标 BLOCKED（不是 PASS），等 W4-02 freeze。
- 成功：spec 8/9 全过 + patch 0401-{ordered-flags,csr-interlock,park-csr-quiescence,cancel-iso}.patch 四个全部落地 + `ctx_owner_cancel.yaml` freeze hash 进 `PROTOCOL_FREEZE.json`（jointseal）。
- 取消：本步自身不引入 cancel；W6 `CANCEL_TASK` 只作用于本 Context（W2-D06）。
- 旧代：`ctx_generation` 拒收旧 delta（spec 9 最后一例）。

## 9. W2-05 C1/C2 原生标量闭环

### 9.1 Scope

在 1 Context（cs1_c1 退化）与 2 Context（cs1_c2）配置下，把**原生标量（RV32IMF + Zicsr + Zbb + Coral 内置控制指令，不含 RVV/VME 数值）**端到端跑真：`W0-04` 的 target（c1/c2）+ `W5-02` 基础 kernel ELF + `W1-03` C ABI cosim 骨架 + `W4-02` L1 端口 + `W6-02` WAIT/terminal 逻辑 + `W7-02` independent ISA oracle。**不把 Python validator 对拍当指令 oracle**（父合同与设计 §8 的硬口径）；Python 对拍只检查 abstract contract 节点（commit/launch/phase/completion 等），与 per-Context retire 原生流分开。

### 9.2 步骤

1. 从上游 native ISA 测试套件裁 c1 子集（`tests/cocotb/coralnpu_isa/*`、`registers.S`、`csr_test/`、`exceptions/*`、`fcsr_frm_hazard_test.cc`、`float_csr_interlock_test.S`、`float_hazard_tests.S`、`frm_test.cc`、`wfi_slot_{0..3}.c`、`cf_fault_retire_test.S`、`retire_buffer_full_illegal_inst.S`、`rvv_fflags_test.{cc,py}`——W0-03/W7-01 统一 fixture 载体，W2-05 不复制副本）；对 c2 新增 `hardware/coral_nexus/verification/unit/w2/nativescalar_c2.c/.S`（两 Context 独立 ELF，共享只读 code/rodata、独立 stack/data，`T-30/T-02` 模板）。
2. 集成 `W7-02` independent ISA oracle（Spike based 或自研 binary oracle，由 W7-02 决定，不在此定）：W2-05 只负责 (a) 把 `oracle_requirements.yaml` 里的输入喂给 oracle（ELF+init state+可写段布局）、(b) 把 oracle 的 per-lane `retire_record` 与 RTL per-Context retire 流做**逐条比对**（pc, inst, rd_wr, trap, flags_delta, ctx_key）、(c) 把 oracle 架构终态（RF/CSR/内存字节）与 RTL 终态比对。比对实现脚本放 `verification/unit/w2/`，不改 oracle 本体。
3. 落 `hardware/coral_nexus/verification/unit/w2/W2_05_native_loop.py`：驱动 `cn.sh cosim --scenario nativescalar_c{1,2} --backend rtl --config cs1_c{1,2}.yaml --out out/coral_nexus/w2-05/`；比对 oracle 与 RTL retire 流 + 终态；产出 `report.json`（case → pass/fail/blocked + 阻塞项 ID）。
4. 现象级断言（验收判据，全部来自设计 §8 表和父合同，不发明新口径）：
   - **同 PC 两 store**（T-02）：两个 Context 同 image 同 PC 独立 `sw`，oracle 逐条 retire 记录按 `OwnerTag` 匹配，RTL 无错配（`storeComplete`/`fault` 键已被 W2-02 替换）；负例：`w2-02` 的 PC-only 复活变体（patch 0202 的反例 patch 放 `patches/w2/mutants/0202-pc-key-store.patch`）必须**被该 case 检出失败**。
   - **同寄存器号**（T-02/T-06）：A 写 `x5` 在途、B 读 `x5` 不被 A 阻塞（发射记录），forwarding 命中只在本 Context；负例：`rf bank` 退化（共享 scoreboard）变体被检出失败。
   - **oldfetch**（T-07 变体）：重定向后旧 epoch 迟到响应拒收、其他 Context 缓冲不变；负例：`fetch-mux` 的 epoch 键丢失变体被检出失败。
   - **partial accept**（T-11）：单元级（spec 6）+ CS1 级（native kernel 中 unit_ready 抖动注入，`W1-01 issue-phase` 消费）；无变形/无重复发射。
   - **flags-clear**（T-06/CFG-04）：`older FP→csrw fflags 0→younger FP`，逐条 delta + 终值恰为 younger delta；oracle 判定（依赖 W7-02 的 fflags delta 语义，不依赖 Python 对拍）。
   - **独立退休**（T-09/RET-03）：A 头延迟，B 照常退休（native loop 里用长延迟 load/store 构造）。
   - **饥饿**（SCH-03/T-34）：长 store 链/长 div 链饱和下另一 Context 的 `issue_service_interval` 有界（上限由 W4-02 仲裁 freeze 提供；未冻结先记录数据，不判 PASS）。
   - **park/WFI/取消**（T-27/ISA-04/T-25/T-08）：W6 接管的 case 在 W6-05/W2-04 联跑，本步只跑"正常 terminate + 独立退休"子集。
5. c1 与 c2 同跑：`cs1_c1` 是退化配置**功能回归**，`cs1_c2` 才是 P1/PN 的比较基线（父合同"性能 P1/PN 同 cs1_c2"）。两者都过且 oracle 对拍全过 ⇒ W2-05 DONE；仅 c1 过/c2 blocked ⇒ W2-05 状态 blocked，不报 DONE。

### 9.3 输入/输出与验收

- 输入：W2-04 all-pass；`W0-04` c1/c2 target freeze；`W5-02` 基础 kernel ELF；`W1-03` cosim 骨架（oneTile C ABI loop）；`W4-02` L1 端口 freeze；`W6-02` WAIT/terminal freeze；`W7-02` oracle freeze（native ISA oracle；**不以"supported profile"为前置**，experimental 状态即可对拍，supported 升级是 W7-05 的事）。
- 输出：`verification/unit/w2/{nativescalar_c2.c,.S,W2_05_native_loop.py}`、`patches/w2/mutants/` 下负例 patch（见每个 case 的负例引用）、`out/coral_nexus/w2-05/report.json`。
- 验收命令【未来，W2-05 实现侧】：
- `bash hardware/coral_nexus/scripts/cn.sh cosim --scenario hardware/coral_nexus/verification/unit/w2/scenarios/nativescalar_c1.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c1.yaml --out out/coral_nexus/w2-05-c1/`
- `bash hardware/coral_nexus/scripts/cn.sh cosim --scenario hardware/coral_nexus/verification/unit/w2/scenarios/nativescalar_c2.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/w2-05-c2/`
- 每产生 `out/coral_nexus/w2-05-c{1,2}/{trace,report}.json`（W7 消费）；**无 grep/源码存在当 PASS**。
- 成功判定：oracle 逐条对拍 0 差、终态 byte 相等、报告里 blocked 清单空（或每条都给出具体缺失 freeze 的 ID）。
- 失败判定：任一 case fail 且负例 patch 未被检出 ⇒ 该 case 报 fail，W2-05 不 DONE，上报 parent 定位接口缺陷；**不改 oracle、不改 checker、不改设计**。

### 9.4 失败/取消/旧代/背压

- 失败：oracle/ELF/端口任一 freeze 缺 ⇒ 用例 BLOCKED，不 PASS；`report.json` 列条目。
- 取消：cancel 类 case 归 W6-05 联跑，本步不重复实现。
- 旧代：`ctx_generation` 拒收在 native loop 中由 T-08 子项验证；不在本步重写。
- 背压：`W1-01 issue-phase` 的随机背压在本步消费（partial-accept case 的 CS1 级），单元级已在 W2-03。

## 10. 现象 → 现象判据 → 依赖 oracle 映射表（验收快查）

| 现象                       | 任务     | RTL 观测点（真实位置）                                                                  | oracle 依赖                                              |
| -------------------------- | -------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 同 PC 两 store 混流        | W2-02/05 | `storeComplete`/`writeData*` 带 `OwnerTag`（W2-02 patch）；native loop 逐条 retire 对拍 | 独立 ISA oracle（W7-02）；负例 = PC-only 复活 patch 检出 |
| 同号寄存器假依赖           | W2-02/05 | per-Context banked scoreboard；`issue_service_interval`                                 | oracle；负例 = 共享 scoreboard 变体                      |
| oldfetch 响应错投          | W2-03/05 | `FetchReorderBuffer` `OwnerTag+epoch+reqid` 键 + 信用归还                               | oracle；负例 = epoch 键丢失变体                          |
| partial accept 变形        | W2-03/05 | `RetirementBuffer` L106-110 前缀断言 + payload 稳定                                     | oracle（逐条 retire 序）；validator contract 节点序列    |
| flags-clear 顺序合并       | W2-04/05 | per-ROB 槽 5+1 bit delta；retire 拍合并                                                 | **必须** oracle 的 fflags delta 语义；不用 Python 对拍值 |
| CSR 读互锁穿透             | W2-04/05 | `canDispatch` CSR gate；`csrr fflags` 读值 = 合并后                                     | oracle；负例 = 去掉 gate 变体                            |
| 独立退休（A 头阻塞）       | W2-02/05 | 分区 head 与 credit 独立；retire 时间戳差                                               | oracle；负例 = 共享 FIFO head 变体                       |
| 饥饿                       | W2-03/05 | `Scheduler` stall 计数 + issue_service_interval                                         | oracle（延迟分布），上限由 W4-02 仲裁 freeze 提供        |
| 槽位复用/旧代注入          | W2-05    | `ctx_generation` 拒收 + 信用守恒（CTX-09/15）                                           | oracle；负例 = generation 位失效变体                     |
| WFI≠event wait/quiescenc e | W2-04/05 | `Context` park 投影 + `CoreAxi` wfi 移除 patch + quiescence 接口                        | oracle（WFI retire 记录）；W6-04 联跑                    |
| fail-stop                  | W2-04/05 | fault 归属三元组 + Tile 停发 + 隔离排空                                                 | oracle（fault pc/cause 逐条）；W6-05 联跑                |

## 11. 需求族映射与 T 覆盖（W2 主责/参与行）

| T    | 主责  | W2 覆盖内容                                                                                          | 其他包参与                                 |
| ---- | ----- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| T-01 | W2+W3 | 逐条原生 ISA 指令（scalar 侧 W2，RVV 数值 W3）；W2-05 c1 native loop 的 oracle 对拍                  | W3-05（RVV/VME 侧）W7-02（oracle）         |
| T-02 | W2    | 同 PC/same reg 的 OwnerTag 键替换 + 负例 patch 变体（W2-02/05）；`ctx_generation` 不变量（W2-04/05） | W7-02 oracle                               |
| T-05 | W2    | A 在途切 B：写回按旧 ContextKey 归属（W2-02）、停发无 flush/drain（W2-03 单元）                      | W4（LSU 在途）W6（切换事件）               |
| T-06 | W2+W3 | flags ordered delta、CSR 读互锁、per-Context 配置快照（W2-04）                                       | W3（配置快照与向量侧 flags）、W7-02 oracle |
| T-07 | W2    | fetch epoch 定向取消、迟到回包拒收（W2-03）                                                          | W4（LSU 回包）W7-02 oracle                 |
| T-08 | W2    | 槽位复用 gate + 旧 owner 拒收 + generation 回绕负例（W2-04/05）                                      | W7-02 oracle                               |
| T-09 | W2    | A 头阻塞不挡 B 退休（W2-02/05）                                                                      | W7-02 oracle                               |
| T-10 | W4    | 多 Context 乱序/分片访存（W4 主责）；W2 侧 `OwnerTag` 派发正确性（W2-02 单元）                       | W4-03/04 W7                                |
| T-11 | W2    | issue 接口随机背压 + bundle 部分接受（单元+CS1 级）                                                  | W7-01（issue phase 锚点）                  |
| T-12 | W2    | ROB/scoreboard/in-flight 按 Context 灌满（W2-02/03 单元 + T-12 in CS1）                              | W7-03 C3 守恒                              |
| T-30 | W5    | 同 image 双 Context（W2 侧重在 per-Context 可写区隔离、`OwnerTag` 派发）                             | W5-02 W4 W7                                |
| T-34 | W2    | 长短任务混合 + 饱和 VME 混合（W2 scheduler、共享 unit 公平性）                                       | W4-02 仲裁 freeze、W7                      |
| T-36 | W7    | per-Context PMU 计数 + W2 stall 归因（W2-03 提供原始数据）                                           | W7-01 trace、W7 PMU schema                 |
| T-39 | W1    | Python↔RTL 联仿对拍（W2 提供 per-Context retire 流与 contract 节点锚点）                             | W7-01/02                                   |
| T-44 | W1    | UCE 钉住（`WAIT_SLOT` 归因在 W1/W6）；W2 侧 per-Context ready 输入（W2-03）                          | W1-04 W6-05                                |
| T-48 | W4    | 迟到回包与代际（LSU 侧主责）；W2 侧重在 `OwnerTag`/`ctx_generation` 拒收                             | W4-04 W7-03                                |
| T-64 | W1    | CS1 消息边界（W2 侧提供 per-Context retire 流）                                                      | W1-02/03/04 W7                             |

不在 W2 直接覆盖、但 W2 提供原始数据/接口的：`STA-01..07`（W2-02/03 硬件侧实现）、`CTX-01..16`、`FET-01..07`、`CFG-01..06`、`RET-01..08`、`SCH-01..11`、`MAT-20/21/22`（W3 主责区域声明）、`OBS-07/14..18`（W7/W6）。

## 12. 低能力实现 agent 手册（按任务执行）

1. **先读**：本文件 §1/§2/§4 → `design/.../W2_Context_UCE.md` §3/§4/§8 → 本文件对应任务节 → 上游锚点（`/tmp/coral_src/RetirementBuffer.scala` 或 git submodule 里同 commit 文件）。
2. **不改**：W3/W4/W6/W5/W7 的任何文件；`Lsu.scala`/`rvv_backend_*.sv`/`RvvFrontEnd.sv`/`CoreAxi.scala` 一旦要动 ⇒ 写 `patches/reqs/w2-to-wn-*.md` 交给 owner（W2 是这些请求的发起者，不是接收者）。
3. **先写 spec 后写 RTL**：每个任务先在 `verification/unit/w2/` 落 spec 骨架（空 case），再落 patch；patch 顺序按 task 内编号（0010→…）。
4. **负例先行**：每任务至少一条"故意破坏不变量"的负例 patch（放 `patches/w2/mutants/`），先跑失败再跑正例；不允许"非空/编译过/不报错"当 PASS。
5. **接口缺失时**：给 owner 的最小证据 = YAML 字段名 + 上游路径锚点（文件:行）+ 一段失败现象（哪个断言、什么输入），不自行改合同；冲突直接报告 parent（不在计划内部裁决）。
6. **何时标 done**：该任务的"验收命令"+"负例先失败后通过"+"输出文件存在且版本化"三者齐；不跑 G2/G3 gate（那是 W7/parent 的事）。
7. **禁止**：把 validator 对拍当指令 oracle；把"非空/编译过/不报错"当 PASS；把性能数字（加速比/延迟）写进本轮交付；把 [INFERENCE] 条目当已验证事实。

## 13. 风险与"报告给 parent"清单（本轮只列出，不自行裁决）

1. **跨包文件冲突**：`Dec ode.scala`/`SCore.scala`/`Core.scala`/`Parameters.scala`/`Regfile.scala`/`FRegfile.scala`/`RetirementBuffer.scala` W2 独家可编辑（父合同已定）；若 W3/W4/W6 要求反向修改它们，必须走 `patches/reqs/`，不得直接写 patch。**冲突清单**：`RetirementBuffer.scala` 的 `OwnerTag` 扩展与 W3-02 RVV ROB 侧 tag 消费必须同 patch 序列落地；`Lsu.scala` 的 owner-tag 改造（W4-03）与 W2-02 `RetirementBuffer` 匹配键升级互相依赖，须 W0-01 统一 patch 排序。
2. **full RetirementBuffer 与 c1 的关系**：W2-D01 采纳 `enableVerification=True`，`CoreMiniVerificationAxi`（non-RVV）target 在上游历史上曾有 RVV-off+Float-on 时 full ROB 模式 hang 的注释 [A01 `Parameters.scala:80-94`；上游 `BUILD` 注释]。**本轮只把 full 模式作为行为定义**，c1 native 闭环（W2-05）依赖的是 RVV/VME-enabled 配置（`VmeCoreMini*`/`RvvCoreMini*` 系）下的 full 模式，W0-03/W2-05 必须在 c1 上实测；若实测出现 hang，报告 parent（这是上游注释遗留风险，不是新发现）。
3. **`CoreAxi.scala` 实体编辑 owner = W2（parent 定）**：W2-04 的 patch 0403 做"删除 WFI 直接关钟耦合 + 暴露 `gate_ctl` 输入接口"；quiescence/gate_ctl **算法消费方是 W6-04**（W6 提供信号语义与算法，不直接改 CoreAxi 文件）；W2-04 与 W6-04 计划若不同步，走 `patches/reqs/w2-to-w6-ctx-quiescence.md`，不得平行双份修改。
4. **`Lsu.scala` owner-tag 改造与 W2-02 的匹配键升级互赖**：W0-01 的 patch 序列必须先 apply W4-03 的 LSU owner-tag patch，再 apply W2-02 的 RetirementBuffer 匹配键 patch（或反向，由 W0-01 统一确定）；W2 计划不能单独编译 W2-02 patch，这在 G2 gate 前是正常状态。
5. **native ISA oracle 独立性**：W7-02 采用 Spike（上游 `sw/coralnpu_sim/spike_cosim_dpi.cc` 已存在）或自研 binary oracle，W2 均只依赖 `oracle_requirements.yaml` 输入接口；**不把 Python 序列对拍当指令 oracle**（parent 硬口径）。
6. **P1/PN 口径**：`cs1_c1` 是退化功能回归，`cs1_c2` 才是 P1/PN 基线（parent 补充口径）。W2-05 的 c1/c2 分开判定，不用 c1 当 PN 基线。
