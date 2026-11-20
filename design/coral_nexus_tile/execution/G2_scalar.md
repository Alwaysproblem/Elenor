# CN-G2 标量执行阶段执行指南（C021–C041）

本页把 CN-G2 的标量执行工作拆成 21 张 commit 卡。公共规则与 shell 变量见[执行总指南](../execution_guide.md)；每卡使用新的 `EVIDENCE` 目录。所有 `cn` 命令均在其 producer commit 交付后才可调用。文末记录相对包计划的执行顺序澄清，不改变已批准架构范围。

阶段总原则：

- 完整 CS1 = 4 真实 RTL Tile + Python Group/L2/HBM；CN-G2 只要求 oneTile（`--rtl-tile-mask 0x1`）标量闭环，不等全 CS1。
- 现有 Python validator 行为保持的切分（如 C022 TilePort 接缝）与真实 cutover（producer 与 consumer 同 commit 迁移）分开；任何接口切换 commit 必须迁移全部 caller，不留双轨。
- 独立 RTL module 可在模块 tb 上真实验证（unit 证据）；`cn build`/`cn cosim` 的完整真实 Tile target 自 C040 才可运行；stub（`cn_protocol_stub.sv`/`libcn_cosim_stub.so`）仅限 unit，永不作 CS1/final top 证据。
- 证据交接约定：每个 `cn` run 的 `--out` 目录产 **run 级 `manifest.json`**；多 run 汇总为 `cs1_manifest.json`（aggregate）；gate 从自身 `out/inputs.json` **显式**读前置 run 的路径+hash+检查类型——先从前置 run 的 `manifest.json` 取 receipt 填入 `inputs.json`，不接受"指向任意 manifest"式引用。

<a id="c021"></a>

## C021 — 独立 scalar ISA 参考驱动与 bit oracle 基础

- **任务**：W7-02（子步 W7-02a/02b/02c/02d/02e 中标量相关部分）
- **前置**：C020
- **建议 commit**：`feat(coral-verify): independent scalar ISA reference driver and bitwise oracle core`
- **输入/先读**：`../implementation/W7_Verification_Trace_Plan.md` §7（W7-02 全节）；C020 的 `PROTOCOL_FREEZE.json` 联合封印 hash（`cn_protocol_v1.yaml`/`csr_name_map.yaml`/W3-01 数值合同文本 hash）；W0-01 锁定的 `riscv-isa-sim@93a10ae6` tarball（sha256 `064f4c1f…`，C001 物化）。
- **修改边界**：H=`hardware/coral_nexus`。新增 `H/verification/oracle/spike_ref_driver.py`（**固定 pin Spike 为主 scalar ISA 参考**：`LoadProgram/Step/IsHalted/Read*/Write*` API，只执行上游 ISA 语义）、`bitwise_scalar.py`（**窄集补充，不重造整 RV32IMF**：仅 Spike 无法独立给出或需 delta 级信息的语义——fflags/vxsat delta、frm 舍入、CSR 读写期望值；纯整数位运算，禁 numpy 浮点、禁读 Zvt RTL）、`oracle_main.py`、`nexus_extend.py`（scalar CSR/helper 适配，不假称上游支持；向量/数值参考扩展由各消费卡补）、`oracle_capability.json`/`coverage_matrix.json` 生成、`selftest/` 固定向量、`known_oracle_deviation.md`。BF16/ROD/mt bit oracle 归 C047（G3），本卡对这些类登记 `NONE→BLOCKED(归 C047)`，CN-G2 不等待。注册 `cn unit --suite w7` 的 oracle 自测 case（runner 由 C019 提供）。
- **实现步骤**：
  1. `spike_ref_driver.py`：加载 ELF→逐条 Step→读寄存器/内存快照，产出 per-lane retire 参考；`bitwise_scalar.py` 只补 delta/CSR 语义；能力探针逐指令分类 `oracle_kind∈{ISA_REF,BIT_ORACLE,BOTH,NONE}`，Nexus 私有面直接 NONE 并走 `nexus_extend.py` 扩展（非 TODO）。
  2. 用独立可手算的 scalar ELF 验证整数/浮点边界、分支、访存与陷阱；从固定 Spike 接口取得实际参考结果，再按 C010 退休记录格式归一化。私有 helper 的完成/事件输入是显式环境输入，不从 DUT 的目的寄存器值反推参考输出。
  3. 生成门：`oracle_capability.json`/`coverage_matrix.json` 生成；**判据＝逐条实际已覆盖指令产出期望输出 bits**（寄存器/内存/flags 终态逐位比对），不以"向量数≥3"或"非空"作通过；`oracle_main.py` 暴露 `oracle.check(t_id, run_dir) -> verdict` 供 C4 checker 与 C041 消费。负例：G2 mandatory 行 NONE→`BLOCKED(oracle 证据不足)` 条目；BF16/VME 类 BLOCKED 注明 `归 C047`、不计入 G2 阻塞。
- **验证**：runner 与 suite 注册由 C019 提供，oracle case 本卡添加：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit"
  ```

  断言固定自测的寄存器、内存和 flags 位模式均与独立预期一致；翻转结果或破坏一条舍入/异常规则必须定位具体不等位，不能以向量条数或文件非空代替判据。

- **通过/交接**：`oracle_capability.json` 标量域逐条非空且每条附已覆盖指令的输出 bits 证据、BF16/VME 类逐条 `BLOCKED(归 C047)`、生成物 hash 稳定；一项负例如上。消费者：C038（scalar kernel helper 适配）、C041（C1/C2 逐条 retire 对拍）、C047（BF16/ROD/mt/VME bit oracle 扩展，G3）。W7-02 本卡只覆盖标量子集，DONE 需后续 commit 齐全。

<a id="c022"></a>

## C022 — Python TilePort 无行为改变接缝

- **任务**：W1-02（TilePort/PythonTileBackend 与 tile_group 拆分）
- **前置**：C020
- **建议 commit**：`refactor(coral-cosim): introduce TilePort seam with byte-identical Python behavior`
- **输入/先读**：`../implementation/W1_Boundary_Cosimulation_Plan.md` §8（W1-02 全节，含 §8.2 拆分表与行号锚点）；C017 生成的 `pipeline_validator/cosim/wire.py`；C020 封印 hash。
- **修改边界**：V=`pipeline_validator`（新 `cosim/tile_port.py`、`python_backend.py`——**仅完整 Python 侧 TilePort Protocol 与 PythonTileBackend**；`rtl_backend` 本卡不创建，真实 RTL adapter 归 C025（stub 驱动）与 C040（真 top），不对外提供空 API/占位通道；`memory/program/event/caps/reset_phase` 服务文件留 C023/C024）；共享接线 `simulator.py`/`tile_group.py`/`device.py`/`runtime/group_port.py`/`cli.py`（W1 唯一集成点，按 §8.2 表拆分：`__init__` 工厂按 `backend_kind`、`_step_tiles` 拆 `_step_python_tiles` 等）。H 侧无改动。
- **实现步骤**：
  1. 完整实现 `TilePort` 与 `PythonTileBackend`；`send()` 不推进模拟，`step(cycle)` 每 Tick 只推进一次。切分不能引入额外排队周期；保留既有父控制阶段与 Tile 推进的相对次序。跨界仅接受声明的值类型，不传运行对象。
  2. 按 §8.2 表切分 `tile_group.py` 各函数（`_step_task_admission` 移交协调器这一步**留给 C035**，本卡仅引入 `backend_kind` 分派与 `_step_python_tiles` 保语义等价）；逐 tick 断言通道 credit 守恒。
  3. 改造前后各跑 pytest 全集与四个既有 runner：`l2-admission-wait`、`transformer-prefill-attention`、`transformer-prefill-attention-baseline`、`profile-reconfiguration`。保留各 runner 的配置/fidelity，登记 cycles、逐 Tile PMU、credit 与 trace 偏序到 `H/protocol/baseline_python.json`；输出实物在本卡证据目录。旧 launch/Context generation 的 TASK_PLAN 必须拒绝且无副作用，不能拿指令 op_seq 代替 launch 身份。
- **验证**（当前可执行，不依赖任何新 cn 命令；参数按 `pipeline_validator/cli.py` 实际支持项）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -q
  conda run -n elenor-validator python -m pipeline_validator \
    --ir-file examples/scenarios/l2_admission_wait.mlir \
    --hw-config examples/configs/profile_l2_256k.yaml \
    --hw-override num_dma_channels=2 --hw-override hbm_fixed_latency_cycles=10 \
    --context-mode 2 --device-context-mode 2 \
    --input-binding A_IN=0x100000:131072:rw --input-binding A_OUT=0x200000:131072:rw \
    --input-binding B_IN=0x300000:131072:rw \
    --sim-override fidelity=full_memory --memory-trace \
    --compiled-output "$EVIDENCE/a.json" --trace-json "$EVIDENCE/t.json" \
    --json --report "$EVIDENCE/r.json"
  for scenario in transformer-prefill-attention transformer-prefill-attention-baseline profile-reconfiguration; do
    bash examples/run.sh "$scenario" --trace-json "$EVIDENCE/$scenario.trace.json" \
      --json --report "$EVIDENCE/$scenario.report.json"
  done
  ```

  四个场景前后输入、配置和 fidelity 必须相同；按稳定身份比较 trace 偏序，不比较含路径/时间戳的整个文件摘要。保留原样例 timing-only 与 full_memory 的证据边界，不宣称这些模型执行了数值 kernel。

- **通过/交接**：pytest 通过、四场景的定义字段逐项相等；旧 generation 请求无副作用，发送不推进且每 Tick 不双推进。消费者 C023/C024/C035/C040；W1-02 的关闭以总指南 §8 为准。
- **冲突上报**：旧 W1 计划把 `_step_task_admission` 移交也写在 W1-02；本指南按共享表移至 C035（准入协调器真实现），避免接口破坏分两 commit。

<a id="c023"></a>

## C023 — 真实 Tile 外部内存腿服务分离

- **任务**：W4-02（外段腿部分）
- **前置**：C022
- **建议 commit**：`feat(coral-mem): external memory leg service split from TransferManager`
- **输入/先读**：`../implementation/W4_Memory_Subsystem_Plan.md` §5.2（W4-02）、§4.2 表（L1 腿切除）；`../implementation/W1_Boundary_Cosimulation_Plan.md` §8.2 `memory_service.py` 行；C022 的 TilePort 端口。
- **修改边界**：V（新 `cosim/memory_service.py` 与真实 API 的字节验证 `tests/test_w1_memory_service.py`；拆分 `memory/transfer.py` 外部腿并原子迁移调用方）。W4 提供外部传输实现，W1 是本跨包 commit 的 Python 接缝集成 owner。只分离 Tile 外部腿；L1 执行仍只在对应后端，RTL bank/cache/MSHR 在 C029/C030。
- **实现步骤**：
  1. `memory_service.py`：MEM_REQ（L2 地址/范围/权限）→ 合同校验（C013 语义）→ TransferManager/ByteStore 真实读写 → MEM_RSP 带 write visibility scope → MEM_TERMINAL_ACK 归还终态引用。
  2. `transfer.py` 外部腿按 §4.2 表拆出服务边界；纯 Python 路径回归不破（C022 基线仍 equality）。
  3. 负例闭环：伪造已释放 view 的 txn_id→拒绝且账本不变；伪造 credit 归还→守恒断言失败。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -q                                  # 回归基线保持
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_w1_memory_service.py -q        # 本卡新建：真实 bytes/终态/credit 断言
  ```

  新建测试断言输入 bytes 与 oracle 相等、`unclosed_l2_objects()` 清零；负例（伪造已释放 view txn_id、伪造 credit 归还）在新建测试内注入并先 RED 后 GREEN。

- **通过/交接**：外腿事务逐条终态、终态引用账本清零、基线 equality 保持；消费者：C024（program/event 服务同层）、C031（LSU 外部腿桥）、C032（MFE 外侧路径）。

<a id="c024"></a>

## C024 — MEMORY/PROGRAM/EVENT 与 Stream 外侧服务

- **任务**：W1-02（外侧服务 completion）
- **前置**：C023
- **建议 commit**：`feat(coral-cosim): program/event/stream outer services on TilePort`
- **输入/先读**：`../implementation/W1_Boundary_Cosimulation_Plan.md` §8.2（program_service/event_service/caps/reset_phase/stream 逻辑子队列）；C017 生成 `wire.py`；C015 事件合同。
- **修改边界**：V（新 `cosim/{program_service,event_service,caps,reset_phase}.py` 与 `tests/test_w1_services.py`）。PROGRAM 服务提供真实 image 字节、消费 Tile 安装 ACK，不伪造安装完成；事件服务连接 Group 表与 PHASE_SIGNAL 聚合。权限取符合 C014 schema 的不可变已校验元数据，正式 sealed Loader 在 C037 原子接入，不先实现假 Loader。STREAM 使用 MEMORY 的独立逻辑子队列；EVENT_RETIRE 广播的 producer/callers 同卡迁移。
- **实现步骤**：
  1. 四个服务文件落地并注册到 TilePort 入站分发（`_service_rtl_inbound`）；每个服务消息逐字段走 `wire.py`，禁裸 dict。
  2. PHASE_SIGNAL 聚合/duplicate/stale 保留 `_on_phase_signal` 现语义（duplicate→协议故障计数恰 +1）；Stream 子队列与 MEMORY 主队列背压独立。
  3. 负例：未授权 POST_EVENT→拒绝+计数；重复 PHASE_SIGNAL→`tile_signal_duplicate` +1（复用现测试口径）。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -q                          # 回归基线保持
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_w1_services.py -q       # 本卡新建
  ```

  新建测试直接驱动四个服务的真实 API：PHASE_SIGNAL 聚合恰一次、重复→协议故障计数恰 +1（断言本卡服务计数器/事件表状态）、Stream 子队列背压独立、未授权 POST_EVENT 拒绝——不依赖既有测试文件名/用例名。

- **通过/交接**：服务级消息 round-trip、事件表终态、Stream 背压独立可观测；负例如上。消费者：C025（五相位驱动消费服务）、C035（admission/retirement 协调器）、C039（program_service 真实驻留接入）。W1-02 的 DONE 以总计划 master §8 更晚消费者门为准，此处不提前宣告闭合。

<a id="c025"></a>

## C025 — 八函数 C ABI 与五相位单元驱动

- **任务**：W1-03
- **前置**：C024
- **建议 commit**：`feat(coral-sim): cn_abi eight-entry C ABI with stub-backed five-phase unit driver`
- **输入/先读**：`../implementation/W1_Boundary_Cosimulation_Plan.md` §9（W1-03 全节，`cn_abi.h` 最终权威命名：bp_abi_info/bp_begin/bp_layout/bp_drive/bp_tick/bp_collect/bp_status/bp_destroy；bp_end/FREEZE_INDEX/cn_wire.h 命名作废）；C017 的 `H/protocol/generated/{cn_messages.h,cn_offsets.svh,cn_layout.json}`；C002 容器 Verilator 工具链。
- **修改边界**：H（`sim/cabi/{cn_abi.h,cn_abi.cc,BUILD.bazel}` 与 `sim/unit/cn_protocol_stub.sv`）；产物 `libcn_cosim_stub.so` 只用于单元端点，正式 `.so` 到 C040 才链接真实 Tile。V（`cosim/abi.py`、单元用 `rtl_backend.py`、`tests/test_cn_abi_phase.py`/`test_cn_abi_crash.py`）。固定容量缓冲预分配，steady-state 每 Tick 无可避免分配/整容量复制；不引入回调、CMake 或别名 ABI。
- **实现步骤**：
  1. 按 C020 合同实现八函数的参数、长度、生命周期和错误映射；`bp_drive` 独立执行 P0，`bp_tick` 只执行 P1–P3，`bp_collect` 在 P3 后/P4 前。布局只读查询不能改变功能状态；init 输入生命周期和复制范围按 ABI 固定，不能擅自增加“查询一次后禁止再查”的限制。
  2. `cn_abi.cc` 封装 stub top：P1 唯一握手判定点、每 tick 恰一上升沿、P3 只提交 P1 接受集；staged 输出仅下 tick 候选。
  3. Python 五相位驱动接 `rtl_backend`（stub 模式）：正例随机 ready 拉低（固定 seed）断言 payload 稳定/接受集恰一次/credit 守恒；负例：漏 bp_drive 调 bp_tick→`CN_ERR_STATE`、双 collect→`CN_ERR_STATE`、collect 缓冲不足→`CN_ERR_RANGE` 不截断、可捕获 fatal→`CN_ERR_FATAL_DUT`+落盘 INCOMPLETE、stub `abort()` 真 crash→外层以无 `run_end.json` 判 INCOMPLETE。
- **验证**（stub 单元；runner 由 C019 提供，本卡注册 w1 case）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w1 --config "$C2" --out "$EVIDENCE/unit"
  ```

  断言 stub ABI hash 与 `bp_abi_info()` 匹配、每 tick `posedge_count` 恰 +1；负例五条全命中。

- **通过/交接**：正负例全 PASS + crash 注入 INCOMPLETE 证据文件落盘；消费者：C035（admission 经真 ABI stub 对拍）、C039（PROGRAM 通道）、C040（真 top 替换 stub）。stub 永不进 CS1 判定（`backend_kind=stub` 标记，任何 CS1 场景引用 stub 报告即 FAIL）。

<a id="c026"></a>

## C026 — Context RF/ROB 与全路径返回身份

- **任务**：W2-02
- **前置**：C020
- **建议 commit**：`feat(coral-scalar): per-context RF/CSR/ROB with OwnerTag return identity`
- **输入/先读**：`../implementation/W2_Context_UCE_Plan.md` §6（W2-02 全节，含上游锚点 L28/L467-483/L493-495/L587-589）；C017 的 `csr_name_map.yaml`、C010 实际产物 `rob_tag.yaml`/`ctx_ctrl.yaml`/`retire_record`/`oracle_requirements`/`perctx_issue`（无 banked_rf.yaml；region 语义取 C012 `w3_vme_decl_abi_v1.json`）；SRC=out 物化上游源码（patch 最终提交，不改 submodule）。
- **修改边界**：H（**W2 为本切换集成 owner，同 commit 迁移全部实际返回 producer 与 consumer**：patch `H/patches/w2/0201-core-ctx-ports.patch`（Core/SCore/Parameters per-Context 端口，caller 全迁移、编译闭合）、`0202-rob-owner-tag.patch`、`0203-rf-banks.patch`、`0204-csr-ctx-bank.patch`，以及 W4 侧 LSU `storeComplete` **接口 patch**（仅 bundle 字段扩展+握手；Lsu 多在途内部实现归 C031）；新 `H/rtl/scalar/{UopTag,BankedRf,CtxCsr,RetirementPartitionBank}.scala`；spec ×3）。匹配键：store/fault/scalar-float 长延迟写回（MLU/DVU/FloatCore 写回调携带 owner 字段）用 `(ctx_key, owner_op_seq)`，**vector/VME 写回按 C010 合同用 `(ctx_key, rob_tag)`**；FaultManager fault 匹配同 commit 迁移。不留悬空新 port；上游未改路径行为原样保留；per-Context flags ordered 路径在 C027 完整接（本卡 CtxCsr 不引入 flags 直写临时变体）。
- **实现步骤**：
  1. patch 0201+LSU 接口 patch：暴露 per-Context 状态端口并在同一 commit 接通 SCore/LSU/FaultManager 全部 caller（无悬空 port）；`UopTag` 定义 `OwnerTag(ctx_key, owner_op_seq)`/`RobTag(ctx_bits, rob_tag+slot_generation)`（`contextCount==1` 显式 specialization 不生成 0-width）。
  2. patch 0202：替换 RetirementBuffer 四处匹配键——store/fault/scalar-float 写回 `(ctx_key, owner_op_seq)`、vector/VME 写回 `(ctx_key, rob_tag)`；flush 只清选中 Context 分区（RET-06）；保留上游断言与 `CircularBufferMulti` 结构。
  3. `BankedRf`（per-Context Regfile/FRegfile/scoreboard，写回按 ctx_key 派发）、`CtxCsr`（trap CSR per-Context，接口按 C010 `ctx_ctrl`）、`RetirementPartitionBank`（full 模式分区 demux，共享输入按 OwnerTag demux）。
  4. spec ×3（同 PC 双 Context store 并发、owner tie 常量负例、换代拒收旧 delta，先 RED 后 GREEN）+ **集成级证据**：patch 后 Core 级编译闭合/ChiselSim 冒烟证明真实 producer 携带 tag 传播，不以模块 tb 喂 tag 代替。
- **验证**（runner C019 提供，本卡注册 w2 case；模块级 ChiselSim tb 属 unit 证据）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w2 --config "$C2" --out "$EVIDENCE/unit"
  cn sources verify
  ```

- **通过/交接**：spec+集成冒烟全 PASS（含负例先失败留证）、`sources verify` patch hash 入 manifest、无悬空 port；不跑 G2 gate。消费者：C027/C028（消费 owner 键）、C031（LSU 多在途内部实现）、C040（顶层组装）。

**2026-11-13 静态验收复核（未重跑 C026）**：用户确认现有测试通过；但当前
`test_w2_c026.py` 及三个 Scala spec 尚不足以证明本卡“全路径返回身份”全部完成：
Core 冒烟仅执行 ALU/MLU/store，未执行 DVU、FloatCore、load 返回及 FaultManager
生产链；FPR 用例虽名为 stale-generation rejection，正文未注入旧代写回；
mutation 判定仅检查非零退出码，未区分行为断言失败与构建/工具失败。因此本卡的
**已有覆盖有效，但完整功能验收证据仍有缺口**，不能把 pytest 绿灯等同于整卡完成。
ROB 同 PC 双 Context store、错误 op_seq、独立退休、局部 fault flush、slot ABA，
以及 GPR/CSR 换代隔离已有行为断言；本轮继续 C027，不据此改写 C026 已通过的结果。

<a id="c027"></a>

## C027 — 架构配置提交与 ordered flags

- **任务**：W2-04
- **前置**：C026
- **建议 commit**：`feat(coral-scalar): retire-point config commit and ordered fflags delta merge`
- **输入/先读**：`../implementation/W2_Context_UCE_Plan.md` §8（W2-04：0401 ordered-flags / 0402 csr-interlock / 0403 core-clock-gating / 0404 context-lifecycle、`Context.scala` FSM）；C026 产物。
- **修改边界**：H（`patches/w2/0401..0404`、`rtl/scalar/`、六个 `W2_04_*_spec.scala`）；原生返回接线按 W3 文件所有权落 `patches/w3/0402..0404`，覆盖 RVV/VME flags、配置 snapshot 与完整 parent identity，不改变张量算术算法。私有细节落 `rtl/scalar/doc/ctx_owner_cancel_impl.yaml`；**C020 一次封印后不向 PROTOCOL_FREEZE 追加任何 hash**。本卡完成配置/flags、Context FSM/park/cancel 与 CoreAxi/CoreTlul `gate_ctl` 端口暴露；完整 oneTile 顶层组装仍按原分工归 C040，gate-control 算法归 W6/G4。
- **实现步骤**：
  1. patch 0401：每 ROB 条目 5 bit fflags delta+1 bit vxsat delta，retire 拍合并（`fflags_arch |= entry.fdelta`），早完成 delta 驻留不丢；`csrw fflags/fcsr/vxsat` 替代动作同样在自身 retire 拍生效。
  2. patch 0402：同 Context "无在途 flags delta / `vset/mset` 未 retire" gate；架构配置（vset*/mset*/frm/vtype/vl/mtype）本 Context retire-point commit，dispatch 拍 snapshot 供 younger；管理类 CSR 单独 dispatch 拍（per-CSR 分类清单入 patch）；删除跨 Context `rvvIdle` 判据。
  3. `Context.scala` FSM（`FREE→HELD→ACCEPTED/INIT→READY→WAIT_*(子状态)→DONE/FAULT`）+ `ctx_generation` 旧代拒收；patch 0403/0404 park 拆分与 cancel 隔离（cancel 只作用本 Context，B 的 scoreboard/ROB 位图逐 bit 不变）。
  4. spec：`older FP→csrw fflags 0→younger FP` 终值恰为 younger delta（逐条 delta 落日志）；槽位复用旧 owner 注入全拒收+计数。
  5. 原生配置返回携带字段 mask，零值仍是有效写入；早到 GPR 写回不能因配置未到而丢失。原生身份使用含 Context generation、op_seq、slot generation 的私有完整键，不凭共享标量 slot 推断 owner；VME 每个 PE block 独立携带异常键，不能跨指令 OR 混合。
- **验证**（独立 C027 入口不重跑 C026；`cn unit --suite w2` 是包含两卡的全量入口）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  mkdir -p "$EVIDENCE"
  conda run --no-capture-output -n elenor-validator python -m pytest \
    hardware/coral_nexus/verification/unit/w2/test_w2_c027.py \
    -n0 -v --basetemp="$EVIDENCE/pytest"
  cn sources verify
  ```

- **通过/交接**：六个 Scala spec、三个原生 SV tb 与 CoreAxi elaboration 全 PASS；四个 mutant 必须为实际行为断言失败，构建失败不算检出；cancel 下的完整退休记录与 C021 pinned Spike 独立单跑逐条对拍。私有元数据落盘且未改任何已冻结合同；SCH-03 由本卡仲裁 tb 按 C019 Context 数证明 N 个 accepted eligible 机会内服务，不等待 C030，也不把它声称为内存延迟界。消费者：C028（park/ready）、C040（顶层）、G4 C059/C060（gate_ctl 算法）。

**本轮执行结果：C027 完成。**

- 全量基线（attempt-045）：独立入口 `test_w2_c027.py` **3 passed**；六个 Scala spec 共 **38 个行为用例**、三个原生 SV tb、CoreAxi elaboration 全通过。四个 mutant 均由真实行为断言检出，不以构建失败充数。
- 真 Core 验证覆盖 FloatCore ordered flags／动态舍入、RVV 配置提交／VXSAT／浮点 flags；独立 native 控制用例覆盖 fault 与 peer CSR 同拍、旧 generation 拒收和退休拍 VSTART 清零。cancel 场景的完整退休记录与 pinned Spike 独立执行逐条一致。
- 主证据：`out/coral_nexus/C027/attempt-045/pytest/c0270/evidence/`；含逐 target 日志、JUnit、native tb 日志、`result.json` 和 `cancel-oracle/comparison.json`。
- `sources verify`：`out/coral_nexus/C027/attempt-044/sources/sources_report.json` 为 PASS；upstream 保持 pristine，patch ownership 唯一。`PROTOCOL_FREEZE.json` 前后 SHA-256 均为 `1673ce7799832fdbe8aaa9139c74302fa2e19590eff0d55e093a8e334212707c`。
- 未重跑 C026，未运行 G2 gate；C026 的静态覆盖缺口仍按上节记录。原生资源仿真保留上游 warning，不将本结果称为 lint 零警告；策略见 `hardware/coral_nexus/MAINTENANCE.md`。
- 补充修复（attempt-046）：延迟 clear 只登记赢得本 Context VSTART 仲裁的请求，避免同拍 fault／CSR 更新胜出后仍被下一拍清零。定向 native 回归 **1 passed**，同时检查 `csr.vstart` 与 `configState.bits.vstart`，覆盖两类同拍更新、正常 clear、旧代拒收与复用；证据为 `out/coral_nexus/C027/attempt-046/vstart-priority.{log,xml}`。本次补充未重跑全量 C027 或 C026。

<a id="c028"></a>

## C028 — 定向取指、eligible RR 与前缀发射

- **任务**：W2-03
- **前置**：C027
- **建议 commit**：`feat(coral-scalar): directed per-context fetch, eligible RR and prefix issue`
- **输入/先读**：`../implementation/W2_Context_UCE_Plan.md` §7（W2-03 全节）；C026/C027 产物；`H/patches/reqs/w2-to-w4-ibus-port.md` 需求（物理端口仲裁归 C030）。
- **修改边界**：H（新 `H/rtl/scalar/Scheduler.scala`、`PerCtxFetchMux.scala`；SRC patch `0302-dispatch-ctx-mux.patch`、`0303-fetch-ctx-mux.patch`、`0304-bru-csr-phases.patch`；`rtl/scalar/doc/perctx_issue_impl.yaml`（本卡私有实现元数据，字段名与 C010 已封印的 `perctx_issue.yaml` 合同一致，不改旧合同）；spec `W2_03_{fetch_epoch,dispatch_prefix,core_issue}_spec.scala`；`H/patches/reqs/w2-to-w1-issue-phase.md`）。**本卡不依赖任何 VME RTL 实现**：Scheduler 头部预译码的分级 resource probe 中 region/VME 相关项仅消费 C012 冻结的 `w3_vme_decl_abi_v1.json` 投影（合同字段），不实例化 VME；VME 实装归 G3。
- **实现步骤**：
  1. `Scheduler.scala`：RR 选 Context（选中才 next）、每 bundle 4 lane 程序序连续前缀、park 只挡发射不清 RF/ROB、stall 原因计数（含新增 WAIT_REGION/WAIT_CSR/WAIT_FETCH）进 PMU manifest。
  2. patch 0302/0303/0304：DispatchV2 输入由选中 Context bank 提供；`PerCtxFetchMux` 的 per-Context reorder ledger 用 `(ContextKey, fetch_epoch, fetch_reqid)` 键，定向 flush、指令缓冲 bank 化，替换旧单流 fetch 路径。每 Context 在途上界取重开后 C019 已生成的 `constraints.fetch_outstanding_per_context`，不硬编码 2，也不等待 C030 再冻结。BRU 的请求拍 CSR 目标读取、执行拍中断读取和 fault 更新分别保留原 owner，避免通过组合 fault owner 反馈到调度。
  3. spec：迟到旧 epoch 响应拒收+信用归还+他 Context 缓冲不变；partial accept payload 稳定 ≥100 拍抖动；分支重定向拍后缀不发射；A 头长期停靠 B 连续服务、A 恢复 RR 从 A+1 继续。负例：旧 PC-only flush 复活、非前缀发射必须先 FAIL。
- **验证**（独立 C028 入口；`cn unit --suite w2` 会同时运行 C026/C027）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  mkdir -p "$EVIDENCE"
  python3 hardware/coral_nexus/scripts/lib/gen_frontend_params.py --check
  conda run --no-capture-output -n elenor-validator python -m pytest \
    hardware/coral_nexus/verification/unit/w2/test_w2_c028.py \
    -n0 -v --basetemp="$EVIDENCE/pytest"
  cn sources verify --out "$EVIDENCE/sources"
  ```

- **通过/交接**：spec 全 PASS、`perctx_issue_impl.yaml` 私有元数据不改冻结合同；饥饿/有界性由本卡独立仲裁 tb 以 C019 容量约束证明（N eligible 机会数=k\*N+c），C030 只验证真实端口满足该界（失败→回修或重做 target，不回填旧 seal），本卡 PASS **不等待 C030**。消费者：C040（顶层取指接线）、C041（partial accept CS1 级注入）、G3 C042/C043（RVV 侧消费）。

**本轮执行结果：C028 完成。**

- 全量基线（attempt-010）：独立入口 `test_w2_c028.py` **2 passed**；取指 **15**、前缀发射 **8**、真实 Core／BRU-CSR **3** 个 Scala 行为用例全部通过，合计 **26**。两个 mutant 均为真实行为断言失败（Bazel exit 3 + failing JUnit），不以构建失败充数；CoreAxi elaboration 通过。
- 覆盖旧 epoch／generation、乱序与重复回包、信用守恒、同拍响应、1000 拍 A 延迟时 B 的服务、≥100 拍 partial-accept 抖动，以及真实 MLU／长延迟 DIV 依赖／LSU SW-LW／分支回环／Context RF 隔离。park 时未完成的取指仍阻止安全复用与关钟；未 accepted 的空槽只在自身义务已排空时计入 quiescence。
- BRU 读侧定向用例用不同 Context 的非零 MTVEC／MEPC 验证 ECALL、MRET、fault 与中断目标；请求拍目标读取与已寄存执行拍中断读取不互借，未靠延迟或屏蔽故障消除组合环。
- 主证据：`out/coral_nexus/C028/attempt-010/pytest/c0280/evidence/`，含逐 target 日志、JUnit、两个负例和 `result.json`。`sources verify` 证据：`out/coral_nexus/C028/attempt-009/sources/sources_report.json` 为 PASS，upstream pristine、patch ownership 唯一。
- 消费 C019 重开后的 `protocol_seq=3`；本卡验证前后 seal SHA-256 均为 `5cd30e2e080a6c3ea46e152e295560b100b0a13151e30c691a87f3815ef2ef46`。旧单流前端及其实现专属 specs 已移除，既有 W2 调用方已迁移到新 lifecycle／debug 角色。
- 未重跑 C026/C027，未运行 G2 gate；本结果不关闭 C030 的真实物理端口仲裁或 C040 的完整 Tile 顶层。C1 elaboration 仍有单元素 Vec 索引宽度提示，不将功能通过称为零警告 lint。
- 补充修复（attempt-011）：执行控制写 `mode/halt/fault/wfi` 按 `bruExecOwner` 路由，`mcause/mepc/mtval` 按 trap/fault payload owner 路由；两类旧代拒收分别归属，同一旧指令同时占两组时不重复计数。扩展后的 BRU 定向回归 **1 passed**：A 分别执行 WFI、MPAUSE、EBREAK、MRET，与 B 同拍 fault，检查两个 bank 的 WFI／halt／fault、MEPC／MCAUSE；MRET 还验证 A 从 Debug 返回 Machine、B 保持 Debug。证据：`out/coral_nexus/C028/attempt-011/bru-write-owners.{log,xml}`。本次补充未重跑全量 C028 或 C026/C027。
- 全量复验（attempt-012，含上述 BRU 写侧修复）：由 subagent 运行未过滤的独立入口，**2 passed in 288.85s**；取指 **15**、前缀发射 **8**、真实 Core／BRU-CSR **3**，合计 **26** 个 Scala 行为用例全部通过。两个 mutant 均以真实行为断言失败检出，CoreAxi elaboration 通过；seal SHA-256 不变。主证据：`out/coral_nexus/C028/attempt-012/pytest/c0280/evidence/`；本次 `gen_frontend_params.py --check` 与 `sources verify` 也通过，证据分别为同 attempt 的 `frontend-params.log` 与 `sources/sources_report.json`。未重跑 C026/C027 或 G2 gate。

<a id="c029"></a>

## C029 — L1 bank、Frame 与 hold/bind/retire

- **任务**：W4-02, W4-05（bank/Frame/arena/hold-retire 部分）
- **前置**：C020
- **建议 commit**：`feat(coral-mem): L1 banks, slot frames and hold-bind-retire arena ledger`
- **输入/先读**：`../implementation/W4_Memory_Subsystem_Plan.md` §5.2 步骤 1–3、§5.5 步骤 3（hold-retire）；`V/memory/l1_slot_frame.py:127-295`、`V/memory/arena.py:1424-1435` 基线；`V/memory/byte_store.py:75-84` 的 L1 allocation/generation/profile 独立有效性域；C013 合同 fragment hash。
- **修改边界**：H（新 `H/rtl/memory/{l1_bank,slot_frame,arena_ledger}.sv`（ledger 含 hold-retire 扩展）+ 各 tb；`scripts/lib/gen_memory_params.py` 将已封印 C013/C019 参数消费为私有 `rtl/memory/c029_params_pkg.sv`，不形成第二套容量权威；注册 `cn unit --suite w4` case；`V/tests/test_w4_hold_retire.py` 新增）。共享表把 W4-02 的 cache/MSHR/程序端口拆到 C030、外腿拆到 C023；本卡 = bank/Frame/ledger。
- **实现步骤**：
  1. `l1_bank.sv`：逐 bank busy/holder、`try_issue` 多 bank 同步占用/释放；SPM/cache 分区和保留容量取 C013/C019 生成参数。模型默认值是来源证据，不直接复制为新的 RTL 固定容量。
  2. `slot_frame.sv`：prepare/bind_view/bind/release/assert_slot_binding；"8 态×1 cycle"是 Python 模型估时参考，RTL 按实际检查/握手推进。Slot 容量只约束同时 live 的绑定，不约束整程序 buffer 数；PREPARE 流式验证 immutable layout，VIEW_BIND 消费对应的只读 layout lookup，允许不同 lifetime 复用 Slot/地址并改变 shape。tb 测量 latency，以实际 layout record 数与 C013/C019 bank/Slot 容量给出工作量界，输入背压单独计量。新绑定发布时范围必须无旧有效字节：reset 初态、此前 VIEW_FREE 的精确范围清理 ACK、以及 RETIRE 的整 Arena 清理 ACK 维持此不变量；不必清空 SRAM 字节，不得清掉 peer 范围的有效性。
  3. `arena_ledger.sv`：HOLD/COMMIT/COMPENSATE 迁移 + hold-retire 链（HOLD→COMMIT_FINAL→RETIRE_READY→RETIRE 授权→本地回收 view/Frame/Arena）；逐 bank `allocated+free==user_spm_per_bank`、`allocated=held+bound`。每 Context/bank 至多一个 reserve extent，空闲区由保留表的补集导出；分配 ID 由 ledger 单调产生，禁止重绑定产生 ABA。执行排空信号须匹配当前 Context generation；Frame shadow lookup 对外提供真实绑定，closing view 不再可查。Profile FSM 仍归 C058，本卡只消费选定的 sourced profile。
  4. tb：按已知 48 B indexed fixture 的地址/字节序列驱动 bank 端口，验证读回与竞争；此处只证明 bank，不声称尚未实现的 gather/scatter FSM 已执行。越界/越 reserved 下限/双 free、hold 期间重复预留均须拒绝。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  python3 hardware/coral_nexus/scripts/lib/gen_memory_params.py --check
  cn unit --suite w4 --config "$C2" --out "$EVIDENCE/unit"
  conda run -n elenor-validator python -m pytest \
    pipeline_validator/tests/test_memory_invariants.py -n0 -q \
    -k "double_release or stale_generation"
  ```

- **通过/交接**：tb 正负例 PASS（先 RED 后 GREEN）；消费者：C030（cache/MSHR 挂 bank 端口）、C035（准入 hold 消费 ledger）、C040（顶层）。

**本轮执行结果：C029 完成。**

- 实际命令入口 `cn unit --suite w4 --config cs1_c2.yaml`：**6 passed in 169.17s**，含完整 RTL 验收驱动及 5 个 Python 参考回归。`double_release or stale_generation` 定向回归另有 **2 passed，74 deselected**。
- `tb_l1_bank`、`tb_slot_frame`、`tb_arena_ledger` 在 C1/C2 下共 **6 次完整执行全 PASS**；**11 个 mutant** 均先编译成功、再由真实 RTL 行为断言检出，构建失败／超时不算检出。全套没有跳过用例，也没有运行 C026/C027/C028 或 G2 gate。
- bank 已验证全 bank 原子占用、无效请求不抢占仲裁、disjoint 进展、背压、实际 48 B 字节序列，以及同 txn ID 跨 lifetime 的旧 fin 拒收。bank 自己生成 holder generation；profile commit 与新请求／invalidation 不同拍接收；精确有效性清理逐 word 推进并等待当前 holder 排空。
- Frame 已验证整程序 buffer 数大于 live Slot 数、不同 lifetime 改变 shape 后复用 Slot、当前 live view 的 overlap 拒绝、真实 shadow lookup、Context/profile/Arena identity、空 Frame 和响应背压。准备时间按实际 record 数、bank 数与输入等待计量，不把模型 8 cycles 冒充 RTL 时序。
- ledger 驱动真实 Frame 和真实 bank：first-fit／capacity／fragmentation、HOLD 不超卖、授权后禁止补偿、引用未闭合拒绝 FREE、分配 ID 不发生 ABA、旧 generation 排空信号拒收、精确 tail 清理保留邻居有效字节、RETIRE 后回收 Arena／Frame，以及 completion 保留到精确 ACK 均通过。Python 只用既有参考对象，无第二套 L1 执行账本。
- 主证据：`out/coral_nexus/C029/attempt-006/unit/pytest/c0290/evidence/`（`result.json`、`source-sha256.json`、逐 case build/run 日志与 rc）；命令摘要为同 attempt 的 `unit/summary.json`；定向回归为 `memory-invariants/pytest.log`。`out/` 为 gitignored 运行证据，不等同归档。
- C013/C019 与 `protocol_seq=3` seal 未改；验证前后 SHA-256 均为 `5cd30e2e080a6c3ea46e152e295560b100b0a13151e30c691a87f3815ef2ef46`。C029 变更路径的 pre-commit hooks 通过（mypy 局部类型声明修正后通过）。不声明 cache/MSHR、Gather/Scatter FSM、运行期 Profile FSM 或完整 Tile top 已完成；消费者边界仍为 C030/C035/C040/C058。

<a id="c030"></a>

## C030 — L1 cache/MSHR 与程序物理端口

- **任务**：W4-02（cache/MSHR/程序端口部分）
- **前置**：C029, C028
- **建议 commit**：`feat(coral-mem): L1 cache, MSHR table and program physical port arbitration`
- **输入/先读**：W4 计划 §5.2/§4.4、`V/memory/mshr.py`/`cache.py`；C013/C019 的冻结端口/容量约束与 C028 实际取指接口。C028 已完成自身有界性验证，本卡把它接到真实物理端口后再验证，不存在回填前置 BLOCKED。
- **修改边界**：H（新 `H/rtl/memory/{mshr_table,cache,transfer_outbridge,l1_program_port}.sv`（含 `tile_program.sv` 安装/取指端口仲裁）+ tb；`V/tests/test_w4_mshr_cache.py`）。profile 12 步 FSM 本体归 G4 C058；本卡 cache 的 `reconfigure`/`begin_maintenance` 语义就位供其消费。
- **实现步骤**：
  1. `mshr_table.sv`：token 分配/合并/cancel、fill 缓冲及 validity、profile generation 分域；line 长度与表容量取 C019，不再自定常量。全部 waiter/响应/终态闭合前 token 不复用。
  2. `cache.sv`：tag+validity+data+dirty 行、refill/clean/invalidate 携 generation、`reconfigure` 仅 commit 后生效。
  3. `l1_program_port.sv`：安装/取指共享端口公平仲裁（RES-12）、如实计数；text 写入仅在无执行引用（FET-06 改写保护）；per-Context 取指在途以 **C019 容量约束与 C028 已证之界**为准，本卡只验证真实端口满足该界——不满足则回修模块或重做 target（新 hash），不回填旧 seal。
  4. tb：MSHR token 早复用/早释放注入负例；program 端口安装与取指并发公平性计数。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w4 --config "$C2" --out "$EVIDENCE/unit"
  cn sources verify
  ```

  交接：附真实安装/取指端口符合 C019 资源界与 C028 仲裁规则的证据；若失败修复本卡实现，不能原地改 seal 或追认未通过的前置。

- **通过/交接**：tb 全 PASS、token 生命周期断言常开；消费者：C031（LSU miss 走 MSHR leader）、C039（安装服务消费端口）、C058（profile FSM）。

**本轮执行结果：C030 完成（RTL 验收）。**

- 实际命令入口 `cn unit --suite w4 --config <绝对路径>/cs1_c2.yaml`：**21 passed in 870.86s**。`--config` 必须是路径，裸文件名 `cs1_c2.yaml` 会在每个 runner 的 setup 阶段 `FileNotFoundError`（耗时 3 s，未触及 Docker）。此前 `commands.yaml` 的 `w4` 只登记 C029 与 hold-retire，C030 的 4 个 runner 与 `test_w4_mshr_cache.py` 不会被该入口执行，本卡已登记。
- 四个 RTL 切片均在 CONTEXT_COUNT 1 与 2 下跑绿：MSHR（831 checks，16 entries）、程序端口（C1 339 / C2 404 checks）、集成 miss path（35932 checks）、原生 C028 取指集成（C1 196 / C2 271 checks，`fetch_outstanding=2`、`fetch_bytes=32`，对真实生成的 `PerCtxFetchMux` 在 contextCount 1/2、sealed 256 位取指宽度下驱动真实端口）。
- 行为 mutant 全部先成功编译、再由已执行的断言以退出码 1 检出（构建失败不计）：C030 四个 runner 共 47 个死于 `C030_ASSERT`（MSHR 9、程序端口 8、cache+bridge 22 [15+7]、取指集成 8）；另有 bank 的 cache 数据端口 11 个 `bank-cp-*`，由 C029 runner（`bank-*` family）执行、死于 `C029_ASSERT`，该 family 共 13 个 bank mutant 在同一次套件运行中全部检出。cache 的 15 个按重写后的 `cache.sv` 重新生成，旧 7 个对新源码全部无法 apply。
- cache 为顺序行扫描、slot 条带化到真实 `l1_bank`、容量取已封印 profile 的聚合值；miss bench 驱动真实 bank，覆盖多 lane victim 选择、cache 与 SPM 持有者在同一 bank 上的仲裁（cache 操作在持有期内等待而非静默跳过）、以及 bank profile 与 cache profile 不一致时拒绝并保持行状态。
- 已知限制：集成 miss bench 的 `CONTEXT_COUNT` 只放大 SPM 侧，两个 Context 数下 check 数相同（35932），C1 仅证明可 elaborate 与运行，不增加第二 Context 覆盖；mutant 02/08/09 同被“原行被淘汰”断言检出，未单独证明跨 lane 组折叠的失败模式。
- C013/C019 与 seal 未改；`cn sources verify` 通过（`out/coral_nexus/C030/final-008/`），作用于所属文件的 pre-commit 全部通过且未改写任何冻结产物。主证据：`out/coral_nexus/C030/final-007/pytest/`。`out/` 为 gitignored 运行证据，不等同归档。未重跑 C026/C027/C028，未运行 G2 gate；runtime Profile FSM 归 C058，完整 Tile top 归 C040。

<a id="c031"></a>

## C031 — 真实多 Context LSU 在途与完成

- **任务**：W4-03
- **前置**：C026, C030, C023
- **建议 commit**：`feat(coral-mem): real multi-context LSU with in-flight settlement and owner keys`
- **输入/先读**：`../implementation/W4_Memory_Subsystem_Plan.md` §5.3（W4-03 全节）；上游 `Lsu.scala:2029-2036`（单 SuperSlot）、`Parameters.scala:172`（txid=8）锚点；C026 OwnerTag、C029/C030 端口。
- **修改边界**：H（新 `H/rtl/memory/lsu_mslot.sv` + tb；`V/tests/test_w4_lsu.py`）。本卡交付**真实 LSU RTL**（非模型桩）：per-Context ingress FIFO、InFlightOp FSM、bank/cache/MSHR 接线、外部腿经 C023 `memory_service`。
- **实现步骤**：
  1. per-Context ingress + bank 端口 RR（先非阻塞 head 后 blocked-but-wakeable，禁全局队首串行化，MEM-02）。
  2. InFlightOp 状态机 `REGISTERED→ISSUING_L1→{COMPLETED|REFILLING→COMPLETED}→TRANSFERRED→CREDIT_SETTLED→COMPLETED`；**已 accepted 的本地 op 在本卡内闭合 settle 接口**（cancel 标记→本地三终态结算→credit 归还，unit 测试覆盖），跨模块 cancel_tracker 接线与外侧取消扩展归 C032——本卡不引用未来模块。
  3. store 完成/写回匹配键 `(ContextKey, owner_op_seq)`（与 C026 ROB 消费对拍断言逐条落地）；per-ctx credit 独立计数/终态归还，下溢/双释放断言常开。
  4. tb：双 ctx 独立 load/store 逐 bank 忙位、单 ctx 长延迟不阻塞其余；负例 credit 下溢/双释放/泄漏、`WAIT_L1_BANK`/`WAIT_MSHR` 归因不丢请求响应。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w4 --config "$C2" --out "$EVIDENCE/unit"
  python -m pytest pipeline_validator/tests/test_tile_indexed.py -q
  ```

- **通过/交接**：tb 全 PASS、storeComplete 键断言落地、无队首串行化回归；消费者：C032（MFE 共享 bank 端口）、C040（顶层 LSU 实例）、G3 C043（RVV 访存）。

**本轮执行结果：C031 完成（RTL 验收）。**

- 交付两个真实 RTL owner 与两个 slice：`rtl/memory/lsu_mslot.sv`（per-Context 16 槽记录表、RR 分配、真实 `l1_bank` 实例、fence、逐记录取消、owner/代验证、跨 stripe 分段、符号/零扩展、精确 bank 完成代捕获）与 `rtl/memory/lsu_cache_path.sv`（C030 cache/MSHR/outbridge 的真实并发读客户端，完整 96 位 provenance key，settle 按 `(slot, entry_gen)` 匹配，取消只标记本地 waiter）。`l1_miss_path` 增 `MSHR_KEY_W`、`m_rsp_data/m_rsp_mask/m_rsp_waiter`、filled/cancelled/refill-pend 掩码、settle/fault 脉冲与 MEMORY ctx/arena/scope sideband；`mshr_table` 只增只读掩码与 settle 脉冲；`transfer_outbridge` 的 MEM_REQ/TERMINAL 改为最早 txn 选择，避免背压下字段跳变。
- 每 Context 在途上限取已封印 `configs/w4_capacity_decisions.yaml` 的 `per_ctx_outstanding_credit_cap=16`（`LSU_CREDIT_CAP`），保留 waiter 预算 `LSU_MSHR_WAITERS=32`；未沿用上游单 Context 的 `lsuSlotTxidEntries=8`，也未另立容量权威。
- 单元验收：`lsu_mslot` slice 在 CONTEXT_COUNT 1/2 下 `C031_PASS checks=664/777`；`lsu-cache` 集成 slice（真实 C023 `MemoryService` FIFO 字节与终态 ACK）`C031_PASS checks=27456/44996`，外部请求 15/33 与终态 ACK 15/33 一一闭合、零残留引用。集成 bench 覆盖跨行 1/2/4 字节、暖命中零外发、旧 refill 挂起时同伴推进、MSHR 满 16 项 + 32 保留 waiter 合并、同 line 取消隔离、终态背压、注入 FAULTED、重复响应、fence 有序、binding_id/代不可别名。
- 行为 mutant 全部先成功编译、再由已执行断言检出（构建失败不计）：`lsu-core-*` 10 个（fence 阻断、fence 过宽、取消同拍气泡、credit 在 valid 而非 fire 释放、最新终态呈现、单记录 bank 引擎、符号扩展、CWAIT 记录 id、ben gating、profile 传递）与 `lsu-cache-*` 10 个（含 drain 提前、fault→ISOLATED、取消杀死共享 refill、key 截断、piece-1 丢失、leader 跳过 refill、refill 零行、waiter id 归零、同拍取消遗漏、bridge 选择回退）全部检出。`lsu-cache-03` 初版为等价变异（取消 token `{0,0}` 永不匹配 MSHR 代），已重写为真实转发 cancelled leader token 到 bridge cancel 端口后检出。
- 真实原生返回键：memory slice 运行期从 materialized Core 重新发射**完整** C026 `RetirementPartitionBank`（`--enableVerification=True`，不是 C030 的 mini ROB），同 PC/同 rd/同 seq 的双 Context store/load/FLW 独立退休；错误 op_seq 与旧 `ctx_generation` 的注入完成被真实 partition 拒收并计数。
- 该集成测试发现并修复了一个真实 RTL 缺陷：`lsu_cache_path` 两处 `else` 误绑定到内层 `if (b < r_n0)`，导致 piece-0 消费同时污染 `r_p1d`、piece-1 与命中路径不写入，跨行读返回错字节。修法为显式 `begin/end`；修前所有 mutant 都死于同一条坏 bench 断言，修后 10 个 mutant 的杀死断言才分散到各自语义。另修正 drain 不变量：OP_TERMINAL 在释放槽的同拍发 settle 脉冲，排空记录晚一拍采样，原不变量会在该拍误报；改为容忍该记录自身在途、token 匹配的 settle 脉冲（未删除断言，脉冲可达性由构造保证）。
- 参考回归：`pipeline_validator/tests/test_w4_lsu.py`（bank 全占原子性/旁路进展、终态 ACK 不可替代、旧 binding 代拒绝）与 `test_tile_indexed.py` 通过；只作独立 oracle，不执行 RTL、不建第二套 L1 执行账本。
- 全量入口：`cn unit --suite w4 --config <绝对路径>/cs1_c2.yaml` **26 passed in 1268.84s**（含 C029/C030 原有测试）；`cn sources verify` 通过。作用于所属文件的 pre-commit 通过。
- 已知限制：本卡是模块级 LSU/缓存客户端，不是 C040 完整 Tile top；runtime Profile 12 步 FSM 归 C058；外侧取消/`cancel_tracker` 与 MFE 直搬归 C032；页授权逐段 lease 归 C055–C057。`mem_rsp` 的延迟/乱序/重复由 bench 注入，`MemoryService` 只提供真实外部字节与终态引用；没有任何仿真数值 kernel。主证据：`out/coral_nexus/C031/acc-core/`、`acc-mem4/`、`full/`、`sources/`；`out/` 为 gitignored 运行证据，不等同归档。

<a id="c032"></a>

## C032 — MFE 直搬与逐事务结算

- **任务**：W4-04, W4-06
- **前置**：C024, C031
- **建议 commit**：`feat(coral-mem): MFE direct-move engine with per-txn settlement and cancel isolation`
- **输入/先读**：W4 计划 §5.4/§5.6；`V/engines.py` 的直搬语义、`V/memory/transfer.py` 的取消/隔离路径；C031 端口、C024 外侧服务。
- **修改边界**：H（`rtl/memory/mfe_top.sv` 直搬路径、`cancel_tracker.sv` 逐事务终态与 tb）；V（`memory/transfer.py` 的真实隔离触发点及其全部 caller，`tests/test_w4_cancel_isolation.py`）。segment/lease/页授权扩展归 C055–C057；直搬必须已有合法 binding 检查与真实外写结算，未实现的 indexed 不接生产、不留空模块。
- **实现步骤**：
  1. mfe_top 直搬 FSM：descriptor 解析→分段 L2↔L1 搬运→逐事务 `COMPLETED|ISOLATED|FAULTED` 终态；`CANCEL_REQUESTED` 只在真实完成或隔离确认后结算，无"超时→成功结算"捷径。
  2. `confirm_isolation` 生产触发点接入；隔离失败 poison（拒新启动至显式静止复位，poison 状态 Group 独占）；同周期 cancel/complete 恰一终态、重复终态/双释放全拒。
  3. 授权 binding 门：直搬 descriptor 逐字段按 C013 合同校验（范围/权限），未授权→fault 不静默。
  4. tb：直搬实际字节与 `V/engines.py` 参考一致；越权 descriptor、重复终态、cancel 后伪成功结算必须被检出。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w4 --config "$C2" --out "$EVIDENCE/unit"
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_w4_cancel_isolation.py -q
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_memory_invariants.py -q -k "cancel"
  ```

- **通过/交接**：直搬 byte 相等、三终态/取消结算语义、零泄漏终局断言；消费者：C034（drain 判据消费传输静止）、G4 C055–C057（indexed/segment/页授权在其上扩展）。

**本轮执行结果：C032 完成。**

- 交付 `rtl/memory/mfe_top.sv`（直搬 FSM：S_PARSE→S_AUTHORIZE→S_SEGMENT→S_MOVE→S_WAIT→S_TERM→S_DONE；分段 L2↔HBM binding、写 payload 取本事务真实读回字节）、`rtl/memory/transfer_cancel_tracker.sv`（每 txn 恰一终态、obligate/ack 计数、release 前置、sticky poison）与 `rtl/memory/c032_mfe_pkg.sv`。`tb/tb_mfe_top.sv` 通过 FIFO 接真实 C023 `MemoryService`（读写双向），断言字节相等、三终态、重复终态/未 ACK 释放被拒、poison 拒新启动直到 quiescent reset、账本闭合。
- 取消语义两域一致：已接受并完成的**外写** leg 把事务绑定到 COMPLETED（效果已发生），纯入向或未发外写才可 ISOLATED；同拍 cancel/settle 只产一个终态；reset 在非静止时被拒且不清 poison。Python 侧以 `terminal_ack` 作为边界 TERMINAL_ACK 接缝（ISOLATED→`confirm_isolation`、COMPLETED→真实完成路径、FAULTED→fault 终态且永不伪装隔离），新增 `isolation_failed`/`clear_poison`（仅静止可清）并把 `_advance_leg` 的取消结算改为"外写已完成即为 DONE"；`tile_group.release_context_memory` 遇 poison 直接抛出，不再永久返回 False。
- 真实缺陷（Python，先 RED 后 GREEN）：原 `_advance_leg` 把取消后完成的**外写**事务结算成 CANCELLED（违背 MEM-09/MEM-12）；边界宣告 COMPLETED 后仍会被 `_finish_cancelled` 覆盖。新增 `pipeline_validator/tests/test_w4_cancel_isolation.py` 6 用例钉住：写腿取消→COMPLETED 且可见提交、读腿唯一终态、COMPLETED 绑定完成路径、FAULTED≠ISOLATED、poison 到静止清除、非取消态确认被拒。
- 初次验收（`c6cb756909`）：C032 slice `C032_PASS tb_mfe_top contexts=1 checks=2224 external=18 terminals=18 absorbed=1`，18 个服务引用均由唯一 ACK 关闭。当时 8 个 `mfe-*` 都编译成功且运行退出码为 1，但 `acc-mfe2/c032mfe0/evidence/red-0.log` 中的 `mfe-01` 仅耗尽等待终态的 30000 拍上限；按「超时不计行为检出」规则，初次结果为 **7 个行为检出、1 个进展超时**。`mfe-03` 由取消后不得授予 wire txn 的不变量检出，`mfe-05` 则实际触发了带活跃记录的 reset 拒绝。
- 初次全量入口：`cn unit --suite w4 --config <绝对路径>/cs1_c2.yaml` **33 passed in 1294.26s**；`cn sources verify`、作用域 pre-commit、validator 全量均通过。这些是当时的结果，不包含后加的乱序 ACK 场景。
- 已知限制：只做直搬；indexed/gather/scatter 与页授权逐段 lease 归 C055–C057（未接生产、未留空模块）；`mem_rsp` 的延迟/背压由 bench 注入，`MemoryService` 只提供真实外部字节与终态引用；没有任何数值 kernel。主证据：`out/coral_nexus/C032/acc-mfe2/`、`full/`、`sources/`；`out/` 为 gitignored 运行证据，不等同归档。

**补验：同拍取消与乱序 wire ACK。**

- 对同一笔已接受的单段 `TILE_LOAD`，Python 分别先送 `COMPLETED` ACK、先送取消，再排空真实 L1 写腿。`test_w4_c032.py` 用 RTL 同拍场景返回的 64 B 驱动这两种顺序，对比本地终态和写入数据：均为 `DONE`（RTL `COMPLETED`），ACK 本身不提前发布目标字节。出向写腿已完成但下一腿尚未发出的取消、只读 DMA 与拒收事务的 ACK 另有 Python 回归。
- 两笔请求以 wire ID 20、21 发出，21 的响应先到。原先 20 的响应会替换正在背压的 ACK；修复后 valid 和完整 payload 保持到握手，服务按 **21、20** 各释放一次。C032 定向 slice：2422 checks，20 请求 / 20 唯一 ACK / 零残留引用。9 个 mutant 均以成功构建后的行为断言检出；重选停置 ACK 的 `mfe-09` 命中稳定性断言，重写的 `mfe-01` 命中外写后错误终态而非超时。证据：`out/coral_nexus/C032/ack-order-final-ia6q89ME/rtl/c032mfe0/evidence/`。
- T30（超时后的 cache clean）：命令受理 500 拍后如期报告故障，issue gate 仍关闭。8 条脏行中 5 条在故障前已完成出向 DMA，必须通过唯一 HBM 信用逐笔写回；其余 3 条不得写入。回归在取消前记录这些范围，逐笔检查排空进展，并在排空后、显式恢复后各核对全部 512 B。同一测试在旧提交 `c6cb756909` 首次字节不一致位于偏移 64（应为 `0x02`，实为 `0x00`）；修复后通过。红/绿对照：`out/coral_nexus/C032/ack-order-final-ia6q89ME/{t30-before-fix,t30}.log`。
- 补验后的 validator 全量：**608 passed，1 skipped**（`out/coral_nexus/C032/ack-order-final-ia6q89ME/validator-final.xml`）。本次 RTL 只重跑 C032 定向 slice，未重跑整个 W4 或 G2 gate。

<a id="c033"></a>

## C033 — 本地持久事件、WAIT 与阶段信号

- **任务**：W6-02
- **前置**：C020, C026
- **建议 commit**：`feat(coral-event): persistent local events, wait records and phase signals`
- **输入/先读**：`../implementation/W6_Events_Fault_Reset_Plan.md` §5.2（W6-02 全节，U2-a..i）；`V/tile.py:978-992,569-588,1298-1318`、`V/runtime/event_table.py` 锚点；C015 事件合同 hash。
- **修改边界**：H（新 `H/rtl/w6/event_table.sv`（**本地事件表真实实现**：持久终态存储+`consumer_refs`/`future_uses` 两计数字段——对齐 W6 计划 §4.1 与 `V/runtime/event_table.py:183-200`；**槽回收仅当 `terminal && consumer_refs==0 && future_uses==0`**）、`wait_record.sv`（读该真实表，非空表引用）、`completion_bus.sv`、`terminal_store.sv` + tb；注册 `cn unit --suite w6` case）。INPUT_RELEASED/OUTPUT_READY 是 PHASE_SIGNAL 消息语义，不登记本地事件表、不分配 local_event_id。
- **实现步骤**：
  1. wait record：完整键 CAM（ContextKey+local_event_id+event_generation+sequence+scope，不含字符串键）、原子 check-register、armed 位整记录裁决后才清；WAIT/WAITALL acquire 时对逐操作数 `consumer_refs++`，裁决/唤醒后递减；编译期已知的后续消费者登记 `future_uses`，消费者实际注册时 frontier 迁移（`future_uses--`、`consumer_refs++`）。**负例：事件已终态但未来 consumer 尚未注册（`future_uses>0`）时不得复用槽位**——提前复用即丢唤醒/串代，tb 注入必须检出。
  2. WAIT/WAITALL：先到直接命中不 park、同拍完成广播优先判决、后到完整键匹配逐操作数终态、WAITALL 任一 ERROR/RESET/TIMEOUT 走失败出口不伪装成功；TIMEOUT 命中与 ERROR 同路径且排空判据永不含 timeout。
  3. SIGNAL_PHASE：一次 accepted 指令→恰一条 PHASE_SIGNAL（逐次发射逐次发，Tile 无抑制门）；重复由 Group 侧 `tile_signal_duplicate` 判协议故障。
  4. terminal_store：TASK_COMPLETION/RETIRE_READY 可靠保存，通道满时稳定保存不丢（状态可查、无丢失计数器）；终态集合 `{DONE,ERROR,TIMEOUT,RESET}`。
  5. tb 覆盖 U2-a..i（同拍/后到/键失配/背压/RETIRE*READY 后取消介入），场景 YAML 本卡生成：`H/verification/scenarios/w6/u2{a..i}*\*.yaml` 显式 seed 数据与期望状态（armed/wake/计数/consumer_refs/future_uses 终值）。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w6 --config "$C2" --out "$EVIDENCE/unit"
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_runtime.py \
    -q -k "duplicate_signal_faults_without_advancing_phase"
  conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_transformer_multicontext.py -q
  ```

- **通过/交接**：U2-a..i 全 PASS（负例键失配四类分别计数且状态不变）；消费者：C034（first-fault/drain 消费终态）、C035（退休链消费 terminal_store）、C038（kernel WAIT helper）。

**C033 完成（本地 RTL 单元验收）。** C020 joint seal 的两份 C015 fragment 是事件和终态语义来源。C1/C2 来自封印 target；C019 没有冻结本地事件表深度或 WAITALL 上限。`FRONTIER=2`、`OPERANDS=3` 取自 C015 测试夹具，另用 `OPERANDS=2` 验证最小合法 WAITALL；4 位计数器只用于实测饱和，不代表硅参数。

- 终态存储只接收 W2/W4/Group 的排空、退休授权和回收证据，不生成这些证据。`RETIRE_READY` 不是成功；`TIMEOUT` 不是隔离证据。已接受但还未发出的 `PHASE_SIGNAL` 阻止本 Context 报告排空，不阻塞其他 Context。终态端口背压时保留被选槽号，直到握手后才允许复用身份。
- Group 侧使用现有实现判重复与聚合，不在 Tile 去重。原有重复信号回归 1 passed；Transformer 多 Context 阶段偏序回归 7 passed、1 skipped（prefill 不适用 decode 分区交接测试）；validator 全量 608 passed、1 skipped。这些是 Python 结果，不能替代 RTL 验收。对应收据在 `out/coral_nexus/C033/accept-OFmQmTbq/`。
- 实测反例：把 WAITALL 操作数计数缩回一位，等待者在第一个 DONE 即错误唤醒；相同 Verilator 探针用 `$clog2(OPERANDS+1)` 后只在第二个 DONE 释放两个引用（`width-probe/results/minwidth0/evidence/`）。修正前把非法 WAIT 操作数计入完成通知的错误类别，集成 bench 在 U2-e 第 53 拍检出（`order-before/semantic.log`）。以上路径相对于本次收据目录。
- 注册入口 `cn unit --suite w6 --config "$C2" --out "$EVIDENCE/unit"`：**1 passed in 148.77s**。常规测试（`OPERANDS=3`）：C1 **708**、C2 **837** 项检查；最小测试（`OPERANDS=2`）：C1 **708** 项检查。每次均执行全部九个 YAML 场景；C2 另验证同名事件跨 Context 不串扰、同一共享事件在一拍唤醒两个等待者并归还两个引用，以及一方阶段消息被背压时另一方仍能发送终态。三个仿真分别回放实际握手的 PHASE_SIGNAL 到真实 Group：重复的两条只触发一次协议故障；合法的八条在各自四个 Task 到齐后才完成相应阶段。此回放比较消息与状态，不比较 Python/RTL 周期。
- **9/9 行为 mutant** 均先成功编译，再由非 watchdog 的 `C033_ASSERT` 检出（运行退出码均为 1）。覆盖提前回收、漏掉同拍完成、WAITALL 误作 WAIT_ANY、错误 scope、遗漏 frontier 迁移、阶段指令与排空同拍、遗漏同拍取消、未等 W4 证据即完成，以及背压中撤回终态。证据：`unit/pytest/c033w60/evidence/` 内各 `red-*.log`、`green-*.log` 和 32 项源码哈希；`cn sources verify` **57/57**。本卡不宣称 C034 复位、C035 退休链、C040 完整 Tile top 或 G2 gate 已通过。

<a id="c034"></a>

## C034 — 可靠终态、first-fault 与基础安全排空

- **任务**：W6-03
- **前置**：C032, C033
- **建议 commit**：`feat(coral-event): first-fault latch, tile reset endpoint and safe drain`
- **输入/先读**：`../implementation/W6_Events_Fault_Reset_Plan.md` §5.3（W6-03 全节，U3-a..i）；`V/runtime/reset_domain.py:11-233`、`V/runtime/fault_ring.py:16-115`、`V/tile_group.py:2364-2375` 锚点。
- **修改边界**：H（`rtl/w6/first_fault_latch.sv`、`reset_endpoint.sv` 与 tb）；V（`runtime/reset_domain.py` 加入真实 Tile 执行/ACK 观察）。保留当前 watchdog 语义，显式配置字段由 C035 接入；生产 fault domain 仍为 GROUP，不把诊断 tile_id 当局部恢复授权。
- **实现步骤**：
  1. first-fault latch：复位未激活且未完成才触发；first-fault-wins，同拍多 fault 按 `source_id→engine_id→ctx_slot→op_seq` 字典序仲裁（不用哈希）；保留至 Host 显式 clear，复位不清。
  2. reset endpoint FSM：`FAULT/RESET_CMD→FREEZE_ISSUE→CANCEL_UNISSUED→DRAIN→MARK_EVENTS→INVALIDATE_RESIDENCY→REBUILD_CREDIT→RESUME/DONE`；Group 唯一顺序权威，Tile 被动执行并 ACK；MARK_EVENTS 由 Tile 实改本地表（PENDING→RESET 计数进 ACK），Group 只写 Group 表；DRAIN 判据只门控真实义务闭合（accepted 计数、MSHR/传输静止、无未发 PHASE_SIGNAL/POST_EVENT），**PENDING 事件不进 DRAIN 判据**（防 MARK_EVENTS 死锁）。
  3. CANCEL_TASK：live-未知 vs stale 严格区分（stale 幂等 ACK 计数；live 冲突→协议错误不任意丢）；已接受外写不可撤销，三终态结算后回收；隔离失败 poison（Group 独占 poison 状态）。
  4. tb 覆盖 U3-a..g、U3-i；U3-h（VME region 失败）在本卡以 terminal_store 注入 `kind=region_done, status=ERROR` 等价驱动（真实 VME 场景归 G3 C046），不靠 BLOCKED 声称通过；注入场景期望"安全失败"本身 PASS（fault_record_idx≥0、reset DONE、终态唯一非成功、零泄漏），缺陷 mutation 才 FAIL；场景 `H/verification/scenarios/w6/u3*.yaml` 显式注入计划与期望状态。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w6 --config "$C2" --out "$EVIDENCE/unit"
  python -m pytest pipeline_validator/tests/test_runtime.py -q -k "TestFaultReset or TestCycleCap or TestSuccessExitClosure"
  ```

- **通过/交接**：U3-a..g/i 及 U3-h 注入等价全 PASS、`is_quiescent_snapshot` 字段可被 Python 侧消费；消费者：C035（退休链/复位接线）、C040（顶层端点）、C046（真实 VME region 失败场景）、G4 C059（完整复位/poison 扩展）。

<a id="c035"></a>

## C035 — 跨边界准入、补偿与剩余资源退休

- **任务**：W1-04（admission/retirement 部分）
- **前置**：C025, C029, C034
- **建议 commit**：`feat(coral-cosim): cross-boundary admission with compensation and residual-resource retirement`
- **输入/先读**：`../implementation/W1_Boundary_Cosimulation_Plan.md` §10（W1-04 全节）；`V/tile_group.py:1229-1350`（admission 现块）、`:970-1025`（pin/unpin）、`:1523-1548`（retire_context_arena）锚点；C024 服务端点。
- **修改边界**：V（新 `cosim/admission.py`、`retirement.py`；`tile_group.py` 的 `_step_task_admission`/`_commit_route_task` 整块移交协调器——producer 与全部 caller 同 commit 迁移，`route.admissions` 对 python Tile 保留、rtl Tile 由 coordinator 票据替代；`config.py` 新增显式 `max_drain_cycles: int = 100` 且 `reset_domain.py:89` 改读显式字段去 getattr fallback；测试 `V/tests/test_w1_{admission,phase_signal}.py`）。gate 唤醒拆到 C036。
- **实现步骤**：
  1. `AdmissionTicket`：`PLAN→PARENT_PREP（父 L2 pin 预留+R 扣减立即入账）→HOLD_REQ→FINAL_COMMIT→ACCEPTED`；授权前失败→COMPENSATE（父 unpin/退 R/释放事件预留；票据保持到各补偿 ACK 收齐）；最终授权后不 ROLLBACK，取消走 CANCEL/drain 两分支皆测。
  2. HOLD 与 FINAL 同 txn_id 不同 message_type 不得误去重（FINAL 独立执行）；ticket 队列有界允许多 ticket 并发、hold 占用实时入账、FINAL 不超卖。
  3. `retirement.py`：链序 `EXEC_DRAINED→RETIRE_READY→RETIRE 授权（不等父资源结清——禁环）→本地回收→TASK_COMPLETION→清剩余父资源（_unpin_task_l2 幂等+R remove）`；root/Grid 收尾须等**全部 Route/tasks 退休、自身 L2 views 释放、outstanding jobs 清零**后才 `retire_context_arena`（不因单个 Task TASK_COMPLETION 即 retire root）；Python 不再执行 L1 retire（rtl Tile 经 TilePort 消息）。
  4. 输入早释放：`PHASE_SIGNAL(input_released)` 每 Task 每方向 ≤1（软件/Loader 静态保证）、收齐 all_tasks 后 `_unpin_grid_readers`；部分 Tile 不解除；重复 input_released 由 **Group** 判协议故障（Tile 每次逐条发、无抑制门，C033 口径）；安全退休另清剩余 pin 幂等；早释放不还 L1/Frame/UCE/R/程序引用。
  5. 负例：HOLD 失败必须完整补偿，FINAL 后失败只走真实取消/排空；空输入阶段按实际绑定核对 pin 效果。gate 变化唤醒与“恒关不轮询”的完整证明留 C036，不把未来行为写成本卡前置。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w1 --config "$C2" --out "$EVIDENCE/unit"
  python -m pytest pipeline_validator/tests/ -q
  ```

  断言补偿后 `unclosed_l2_objects()` 全零、`max_drain_cycles=3` 时 drain **超过 3 步**即强制 cancel（现源语义为超过上限触发，非恰 N 步；显式字段生效负例）。

- **通过/交接**：Python 侧与 stub 驱动版对拍（相同背压/取消注入下相同接受集、补偿终态、unclosed 清零）；消费者：C036、C039（INSTALL 经准入）、C040（oneTile 全链）。本卡不宣称 CS1 级 PASS。

<a id="c036"></a>

## C036 — gate 版本变化唤醒 Route 与 Task

- **任务**：W1-04（gate 唤醒部分）
- **前置**：C035
- **建议 commit**：`feat(coral-cosim): event-driven L1-gate wake for routes and task submission`
- **输入/先读**：`../implementation/W1_Boundary_Cosimulation_Plan.md` §10.1 项 3、§10.2 步骤 6；`V/profile_controller.py:713-715`（OPEN_ISSUE 置 gate 开）、`V/runtime/group_port.py:355-377`（唤醒令牌）、`V/tile_group.py:2070-2078`（admission_version 无 L1 gate 的实证缺口）锚点。
- **修改边界**：V（新 `cosim/gate_wake.py`；`V/group_scheduler.py` 仅限 gate 事件唤醒共享接线——受阻 BACKPRESSURE 记录打 `wait_gate_epoch` 暂离扫描队列、`gate_wake.register(level)` 推回并广播 GATE_NOTIFY，**非 gate 类重试的既有因果与顺序保持不变**；调度语义/准入行为改造归 owner 包；`runtime/group_port.py` 令牌**统一 tuple shape**（增 `gate_epoch` 维）并在同 commit 迁移全部 consumer——不保留 cosim/非 cosim 双 shape 兼容分支；测试 `V/tests/test_w1_gate_wake.py`）。
- **实现步骤**：
  1. `gate_wake.py`：`profile_controller._profile_gates` 翻转/`OPEN_ISSUE` 一步内触发 register；Route 登记（dispatch_role BACKPRESSURE 路径）与 Task 提交（hold 请求）两条等待路径重试，事件驱动非逐 tick 轮询。
  2. GATE_NOTIFY（G→T）通知 Tile 侧 hold/issue 状态机（消息字段走 C017 wire）。
  3. 正例：gate 关→Route BACKPRESSURE 登记；OPEN_ISSUE→唤醒→登记恰一次成功；负例：gate 恒关+无 OPEN_ISSUE+反复 tick→断言无重试探针（无轮询）。纯 Python 基线经 C022 等价门复验。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w1 --config "$C2" --out "$EVIDENCE/unit"
  python -m pytest pipeline_validator/tests/ -q
  ```

- **通过/交接**：唤醒恰一次、无轮询探针、group_scheduler 改动最小 diff 供 parent 复核；消费者：C040（oneTile gate 路径）、G4 C058（profile 12 步真 FSM 消费同名唤醒）。W1-04 的 DONE 以总计划 master §8 更晚消费者门为准，此处不提前宣告闭合。

<a id="c037"></a>

## C037 — 严格包编解码、封印与只读 Loader

- **任务**：W5-01（实现部分；合同已由 C014 冻结）
- **前置**：C020
- **建议 commit**：`feat(coral-image): strict package codec, seal and read-only loader`
- **输入/先读**：`../implementation/W5_Kernel_Image_Runtime_Plan.md` §4.1（W5-01）；C014 冻结的 `w5_package.contract.yaml` hash；C012 的 helper/CSR 语义引用。
- **修改边界**：V（新 `V/package/binding_ir.py`（编解码+严格校验：未知字段/重复键/非有限数拒绝）、`V/package/seal.py`（sha256 覆盖 image+binding+resource+target+patch_mask；重放加载一致；篡改合同层即使重算 hash 也语义拒绝）、只读 Loader `V/package/loader.py`——统一放 pipeline_validator 包内，不在 `H/kernels/cn_package` 建第二套包）；H（`cn.sh package` handler 由本卡实现（C003 只注册已实现命令）：**真实执行 构建→资源复核→封印 链，不只 roundtrip**；本卡生成最小自测 binding `H/kernels/bindings/selftest_roundtrip.yaml` 与其 ELF（构建产物由本卡工具链产出，可复用 C018 bootstrap ELF 则显式引用其来源与 hash）；`cn unit --suite w5` 注册 codec case）。
- **实现步骤**：
  1. `cn_package/v1` schema 编解码 round-trip 逐字段相等。
  2. 封印/重验：hash 不符、篡改 binding、target 不符→加载拒绝且错误指名字段。
  3. 负例（先失败后通过）：缺 ins/outs、alias 非法、重复键、未知字段、非有限数全部拒绝。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w5 --config "$C2" --out "$EVIDENCE/unit"
  cn package --binding "$H/kernels/bindings/selftest_roundtrip.yaml" --config "$C2" --out "$EVIDENCE/package"
  ```

- **通过/交接**：round-trip 与封印重载一致、负例全检出；消费者：C038（scalar kernel 封印）、C039（安装消费封印产物）、G3 C048–C052。`cn package` 命令自本卡起可用。

<a id="c038"></a>

## C038 — 真实 scalar 控制 kernel 与 ISA helper 适配

- **任务**：W5-02, W7-02（scalar kernel 侧）
- **前置**：C018, C021, C033, C037
- **建议 commit**：`feat(coral-kernel): real scalar control kernels with sealed slot freeze`
- **输入/先读**：`../implementation/W5_Kernel_Image_Runtime_Plan.md` §4.2（W5-02）；C018 bootstrap ELF 测量（不得装载正式 CS1）；C021 oracle（`oracle_requirements.yaml` 输入/retire_record 口径）；上游 `coralnpu_v2_binary` 链接脚本与 RV32 工具链（C001/C002）。
- **修改边界**：H（`H/kernels/library/` 首批真实 kernel——**标量+MFE helper 搬运控制集，不含任何 RVV/VME 指令；标量不得自由发起 L1↔L2 load/store，所有跨 L1/L2 搬运经 MFE descriptor helper**（W4 §4.1：普通 scalar 只能访问 Frame 本地窗口）：结构＝「MFE L2→L1 → WAIT → 标量 L1 内 load/算术/store（含 fflags 路径）→ MFE L1→L2 → WAIT → 阶段信号 → 正常退休链」；**G2 不使用矩阵/region helper（不 emit VME END 无 START）**，task 终止走退休链而非 C012 的 END_REGION helper；每 kernel binding yaml；两个不同链接地址脚本；`scripts/handlers/package.sh`；实测报告 `slot_freeze.json`；oracle 侧 helper/CSR 名→汇编封装适配入 `nexus_extend.py`）。本卡产出**首个真正 sealed kernel**（区别于 C018 bootstrap）。RVV/VME kernel 归 C048（G3）。
- **实现步骤**：
  1. 写 kernel（asm/C，RV32IMF+Zicsr+Zbb+Coral 内置控制指令：MFE descriptor 发起/WAIT/标量 L1 循环/SIGNAL_PHASE/退休链终止，helper 白名单与 C012/C015 合同一致）；**golden 用例**：输入 8×i32（L2，小端）= 1..8，MFE L2→L1 后标量 `x+7`，MFE L1→L2，期望输出 8×i32（小端）= 8..15（padding 按 binding 声明逐字节固定），golden bytes 随 package selftest 落盘并供 C040/C041 场景引用；编译→两个固定地址链接→readelf/map 提取实测 text/静态数据。
  2. 静态分析 stack/scratch/spill 上界（受控模板调用图）；逐 bank Arena/指令存储累计复核→冻结 slot 参数进 manifest；封印两个 package。
  3. oracle 适配：helper 名→CSR 逻辑名→Spike custom-instruction 扩展，逐条 retire_record（pc/inst/rd_wr/trap/flags_delta/ctx_key）可产。
  4. 负例：slot 容量越限 kernel→构建拒绝；地址重叠→链接期拒绝；同 (program_id,version) 不同 manifest_hash→登记拒绝。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn package --binding "$H/kernels/bindings/scalar_ctl_v1.yaml" --config "$C2" --out "$EVIDENCE/package"
  cn unit --suite w5 --config "$C2" --out "$EVIDENCE/unit-w5"
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit-w7"   # oracle helper 适配 case（独立目录避免覆写）
  ```

- **通过/交接**：两 package 并存不重叠、封印/重载 hash 一致、`slot_freeze.json` 落盘；消费者：C039（安装）、C040/C041（oneTile 执行与对拍）。RVV/VME 数值 kernel（matmul/gather-scout 的向量路径等）归 G3 C048–C051，其 RTL 前置 C042–C046 亦在 G3，CN-G2 闭合不依赖它们。

<a id="c039"></a>

## C039 — 真实程序安装、引用与多 slot 驻留

- **任务**：W5-04
- **前置**：C025, C030, C035, C037, C038
- **建议 commit**：`feat(coral-image): real program install, reference counting and multi-slot residency`
- **输入/先读**：`../implementation/W5_Kernel_Image_Runtime_Plan.md` §4.4（W5-04）；C030 `l1_program_port.sv`；C024 program_service；C038 sealed package；`V/runtime/program_table.py` 基线（现 analytic-only 语义）。
- **修改边界**：H（新 `H/rtl/w5/install_service.sv` + tb：四步 pull INSTALL 头→READ_REQ→READ_RESP→INSTALL_ACK（实测 hash/长度+本地句柄），有界 FIFO/分段背压，校验失败即 fault 无半驻留）；V（`program_table.py` 演进：pending install 记录、引用计数全局一份、累计容量账本、多 slot 位图、epoch；`V/tests/test_w5_residency.py`）。**范围限定**：本卡＝install module + 真实 program_service 合测（安装/引用/多 slot/失效），含真实 program fetch probe（取指端口读回 bytes 与 image 比对），不含 full core 执行；"运行 A 中安装 B"的真实并存执行归 C040/C063。**本卡先于 C041**。
- **实现步骤**：
  1. Group 驻留管理器状态机 `ABSENT→PENDING_INSTALL→RESIDENT`；invalidating 串行点后无新热命中。
  2. Tile 安装服务 RTL 消费 C030 端口；不占数据 Arena/MSHR/credit（RES-17/18）；Host 上传链：上传发生在 **cosim prepare 阶段**（场景驱动把 image bytes 写入 Python HBM source 区并登记），`cn package` 只做构建→资源复核→封印产物，不写运行内存。
  3. 冷→热→身份变更→单 Tile 失效→group 复位逐态（以 install/fetch probe 与 program_table 状态观测，非 full-core 运行）；warm＝零 Tile 外部 text 字节（边界计数器；local fetch 与 descriptor patch 合法，RES-10 口径）。
  4. 场景 `H/verification/scenarios/t58_cold_warm.yaml`：显式两个 package 引用、期望 READ_REQ 计数与 slot 位图终态。
- **验证**（本卡＝unit tb + program_service 集成 case；full-core 冷热并存执行归 C040/C063；`cn cosim` 完整真实 Tile target 由 C040 启用）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w5 --config "$C2" --out "$EVIDENCE/unit"
  python -m pytest pipeline_validator/tests/test_w5_residency.py -q
  ```

  正例断言：install A 完成后 install B（不同 slot）互不干扰、同映像提交只加引用共享完成点、热命中 READ_REQ=0、fetch probe bytes 与 image 逐字节相等；负例：warm 改写 resident text→拒、patch 越出 patch_mask→invalid descriptor fault、INVALIDATING 中提交→不命中、累计超容量→有界等待或拒绝。

- **通过/交接**：正负例全过 + `assert_l2_closed`；消费者：C040（顶层安装接线）、C041、G4 C063（多映像冷热/失效扩展）。

<a id="c040"></a>

## C040 — 完整 oneTile 标量顶层与原生返回链

- **任务**：W1-03, W2-04（顶层集成部分）
- **前置**：C028, C031, C032, C034, C036, C039
- **建议 commit**：`feat(coral-top): full oneTile scalar top with native retire chain and real cosim entry`
- **输入/先读**：`../implementation/W1_Boundary_Cosimulation_Plan.md` §9.2（真 top 封装）；`../implementation/W2_Context_UCE_Plan.md` §8 步骤 6（EXEC_DRAINED 判定）；C026–C039 全部产物 hash；W2 为顶层集成 owner、W1 提供接口。
- **修改边界**：H（`rtl/top/coral_tile_scalar_top.sv` 必须实例化 C026–C028 改造后的实际 Coral Core、ALU/浮点/乘除/取指/退休链，接 C031/C032/C034/C039 的 LSU/MFE/L1/安装/事件端点；不是仅把若干账本模块拼成空核）。`sim/cabi/BUILD.bazel` 链真 top 产出 `libcn_cosim.so`，注册完整 `build`/`cosim`。V（`cosim/run.py`、`incomplete.py` 与真实 RTL backend，按命令合同传播 PASS/FAIL/INCOMPLETE/异常）。
- **实现步骤**：
  1. 顶层组装 + `bp_begin` 读 `cn_layout.json` Tile top 端口映射；P0–P3 经真 Verilator top、P4 归 Python `simulator.py`/`tile_group.py`。
  2. 原生返回链接线：EXEC_DRAINED（RetirementPartitionBank 空+launch 完成计数+W6 投影空闲）→RETIRE_READY→RETIRE 授权→本地回收→TASK_COMPLETION→C035 协调器清剩余父资源；per-Context retire 流原生导出（供 C041 oracle 逐条比对）。
  3. verdict 三态：协议 fault 走 Group fail-stop 排空（完成且账本清零→FAULT）；真 crash/无 `run_end.json`/credit 破坏→INCOMPLETE（`incomplete.py` 落残留 pending set）。
  4. smoke 场景 `H/verification/scenarios/one_tile_scalar_smoke.yaml`：引用 C038 golden 用例——显式 binding、输入 bytes＝8×i32 1..8、**期望输出 bytes＝8×i32 8..15（逐字节判据）**、期望 retire 数与终态、`--rtl-tile-mask 0x1`；run 落 `$EVIDENCE/cosim/manifest.json`（run 级 receipt，含 RTL top hash/golden hash）。
- **验证**（`cn build`/`cn cosim` 自本卡启用）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn build --config "$C2" --out "$EVIDENCE/build"
  cn cosim --scenario "$H/verification/scenarios/one_tile_scalar_smoke.yaml" --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim"
  cn unit --suite w2 --config "$C2" --out "$EVIDENCE/unit"
  ```

  断言 `report.json`：`evidence_tier=="one_tile_rtl"`、`bp_status.fatal_code==0`、credit_ok、message_seq 完整、`unclosed==[]`、**输出 bytes 与 golden 逐字节相等（仅 completion/counter 不足为过）**；负例：断开 RETIRE 授权的 mutation 必须使 run 判 FAIL/INCOMPLETE 而非 PASS。

- **通过/交接**：真 oneTile smoke 全链 PASS（至少一条 `one_tile_rtl` 实际成功 run）；消费者：C041（C1/C2 闭环）、G3/G4 各 -05 批。stub 版仅 unit，本卡后任何 final 证据引用 stub 即 FAIL。

<a id="c041"></a>

## C041 — C1/C2 标量闭环与 CN-G2 门

- **任务**：W2-05, W7-03, W7-06（G2 子集）
- **前置**：C040, C038
- **建议 commit**：`feat(coral-verify): C1/C2 native scalar closed loop and CN-G2 gate`
- **输入/先读**：`../implementation/W2_Context_UCE_Plan.md` §9（W2-05 全节）；`../implementation/W7_Verification_Trace_Plan.md` §8（C1–C3 checker 子集）、§9（T 矩阵 G2 适用行）；C021 oracle、C038 sealed kernel、C040 oneTile top。
- **修改边界**：H（`H/verification/unit/w2/nativescalar_c{1,2}.c/.S`（c2 两 Context 独立 ELF、共享只读 code/rodata、独立 stack/data）、`W2_05_native_loop.py`（oracle 逐条 retire 对拍+终态 byte 比对）、场景 `H/verification/unit/w2/scenarios/nativescalar_c{1,2}.yaml`（显式 ELF/binding/seed/期望终态）；checker `H/verification/checkers/{c1_format,c2_partial_order,c3_conservation}.py` + `runner.py` G2 子集；`cn.sh` 扩展 `gate --stage G2`）。V 侧不动数值 oracle。
- **实现步骤**：
  1. c1/c2 分别构建与各自 target 匹配的 sealed 包，不能将 C2 manifest 直接喂给 C1。scalar 用例覆盖原核 ISA 子集与 C038 明确字节样本；c2 两 Context 使用不相交可写区，同时覆盖共享只读 code/rodata。
  2. 对拍：oracle per-lane retire_record（pc/inst/rd_wr/trap/flags_delta/ctx_key）与 RTL per-Context retire 流逐条比对 + 架构终态（RF/CSR/内存字节）比对；Python validator 只查 abstract contract 节点，不当指令 oracle。
  3. 现象级断言（设计 §8 表口径）：同 PC 两 store、同寄存器号跨 Context 不阻塞、oldfetch 拒收、partial accept 无变形、flags-clear 终值恰 younger delta、A 头延迟 B 独立退休；负例 patch（`patches/w2/mutants/0202-pc-key-store.patch` 等）必须被 case 检出 FAIL。
  4. checker G2 子集跑真实窗口（C1 格式/seq 唯一、C2 偏序含退休链/早释放链、C3 credit/Arena/终态守恒），负例注入（seq 断档/消费者早读/双重回收）先 RED 后 GREEN。
  5. `cn gate --stage G2`：先从 C040 smoke、C038 package、c1/c2 各 run 的 `manifest.json` 取 receipt，显式写入 `$EVIDENCE/gate/inputs.json`（路径+sha256+检查类型），再聚合 nativescalar_c1/c2 report + checker verdict 输出 `cs1_g2.json`；c1 与 c2 都过才 DONE（仅 c1 过=blocked）。
- **验证**：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn cosim --scenario "$H/verification/unit/w2/scenarios/nativescalar_c1.yaml" --backend rtl --rtl-tile-mask 0x1 --config "$C1" --out "$EVIDENCE/c1"
  cn cosim --scenario "$H/verification/unit/w2/scenarios/nativescalar_c2.yaml" --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/c2"
  cn gate --stage G2 --out "$EVIDENCE/gate"
  ```

- **通过/交接**：oracle 逐条对拍 0 差、终态 byte 相等、负例 mutant 全被检出、`cs1_g2.json` 全 PASS 且 **blocked 清单必须为空**（任何 G2 BLOCKED 即 gate 不通过；缺失前置按退出码 2 上报 parent 处置，不放行 DONE）；场景文件名为稳定确定名 `nativescalar_c1.yaml`/`nativescalar_c2.yaml`（已与 G3 页约定为回归引用），golden bytes 即 C038 golden 用例；CN-G2 门**不等全 CS1 完成**（4-Tile/mixed 归 G4 C065）。消费者：G3 入口 C042/C047；W2-05/W7-03(G2 子集)/W7-06(G2 段) 的原任务 DONE 以总计划 master §8 关闭表（更晚消费者门）为准，本卡完成不自动等同原任务 DONE。

## 执行顺序与旧计划口径澄清

1. W1 旧计划把 `_step_task_admission` 移交与 gate 唤醒都写在 W1-02/W1-04 单任务内；本指南按共享表拆为 C022（接缝）/C035（准入退休）/C036（gate 唤醒）三 commit，接口破坏不分两 commit。
2. W4-02 旧计划单任务含外腿+bank+cache+程序端口；拆为 C023（Python 外腿）/C029（bank/Frame/ledger）/C030（cache/MSHR/程序端口），各自完整 module+行为才完成。
3. W2-04 顶层集成与 gate_ctl 端口暴露分散在 C027（端口暴露/park/cancel）与 C040（顶层组装），因 oneTile top 前置（C028/C031/C032/C034/C036/C039）未齐前无法构建完整顶层。
4. 场景统一在 `H/verification/scenarios/` 或 `H/verification/unit/<suite>/scenarios/`；不存在第二套 `H/scenarios/` 注册入口。各场景显式声明输入与预期结果。
5. W5-02 旧计划首批 kernel 含 matmul/K 分块与 `vtzero/vtmv/vtle` seed/VME drain（依赖 G3 的 C042–C046 RTL 与 C047 oracle）；本页 C038 改为纯标量 load/算术/store/WAIT/阶段信号控制 kernel，RVV/VME kernel 移交 C048，消除 C041 对 G3 产物的隐式等待。
6. W7-02 旧计划把 BF16/ROD/mt bit oracle 写进 W7-02b；本页 C021 只做 scalar ISA/flags/CSR 参考，BF16/ROD/mt 类在 `oracle_capability.json` 登记 `BLOCKED(归 C047)`，CN-G2 不等待 G3。
7. C026 返回键切换由 W2 作集成 owner 单 commit 迁移全部实际 producer/consumer（含 W4 LSU storeComplete 接口 patch；vector 写回按 C010 `(ctx_key, rob_tag)`），旧文"本卡不动 Lsu.scala"作废。
8. 有界性/封印口径：C019 容量约束为权威界，C027/C028 仲裁 tb 自证、C030 只验证真实端口（失败回修/重做 target，不回填 seal）；实现私有细节独立元数据，不追加 PROTOCOL_FREEZE（C020 一次 seal）。
9. C031 已 accepted op 本卡 settle 闭合、外侧取消与 cancel_tracker 归 C032；C032 仅直搬+逐事务终态/取消，segment/lease/页授权归 C055–C057；C034 U3-h 以注入等价覆盖、真实 VME 场景归 C046。
10. C022 验证命令按 cli.py 实际参数（--ir-file/--context-mode/--device-context-mode/--input-binding/--sim-override 等，conda run 环境）；C033 源锚点为 `V/tile.py`、场景统一 `H/verification/scenarios`；C033 必须含 `event_table.sv` 真实实现（两计数 frontier）。
11. C035：重复 input_released 由 Group 判（Tile 无抑制）；max_drain_cycles 语义为"超过上限"强制 cancel；root 收尾须等全部 Route/tasks/自身 views/outstanding 才 retire root；C036 令牌统一 tuple shape 单 shape 全 consumer 迁移。
12. C037 codec/seal/loader 放 `V/package/`（不在 H 建第二包），`cn package` 真实构建→资源复核→封印；C038 标量跨 L1/L2 一律经 MFE helper、G2 不用矩阵/region helper；C039 上传移至 cosim prepare、full-core 并存执行归 C040/C063。
13. C038/C040/C041 共享 concrete golden 用例（8×i32 1..8 → +7 → 8..15）与稳定场景名 `nativescalar_c{1,2}.yaml`（已与 G3 页约定回归引用）；C022/C024 等的"W1-0x 闭合"表述一律以总计划 master §8 更晚消费者门为准。
14. 证据交接统一为 run 级 `manifest.json`→aggregate `cs1_manifest.json`→gate `inputs.json` 显式 receipt（路径+hash+检查类型）；各卡示例 `EVIDENCE` 由总入口按 attempt 注入（`: "${EVIDENCE:?…}"`），不覆写旧证据。
