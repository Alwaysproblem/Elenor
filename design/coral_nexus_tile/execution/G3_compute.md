# G3 计算执行（C042..C054）

> 本页复用总入口 `execution_guide.md` 定义的 `H/V/C1/C2/P0/EVIDENCE/cn()` 约定；每个 `cn` 子命令在本卡验证节标注提供者（前置供给或本卡实现）。阶段范围：RVV/VME 多 Context 执行与区域 lease（C042..C046）、独立 bit oracle（C047，可先行）、全部真实计算 kernel 族（C048..C052）、数值/区域正负闭环与 CN-G3 门（C053/C054）。
>
> 全页统一口径：数值/声明/模式判据唯一来源是 C011/C012 供给、C020 联合封印的固定合同：numeric 主文 `H/verification/numerics/numeric_contract.md`、profile `H/protocol/w3_numeric_profile_v1.yaml`、decl ABI `H/protocol/w3_vme_decl_abi_v1.{md,json}`（hash 随 C020 `PROTOCOL_FREEZE.json` 记录）。本页不新造 KAT 位值、不造 CSR 位值（逻辑名经 C017 `csr_name_map.yaml`）；模块级 tb/unit 证据只算 unit 证据，不算 CS1 证据（CS1 证据=真实 Tile target cosim）。产物约定：每个 `cn` run 的 `--out` 目录产 run 级 `manifest.json`，多 run 汇总为 `cs1_manifest.json`（aggregate）；gate 类命令从自身 `out/inputs.json` 显式读前置路径+hash+检查类型（先生成/取 receipt 设置输入，不指向任意 manifest）。每卡的 `EVIDENCE` 由总入口按新 attempt 设置、卡内不覆写，验证块统一用 `: "${EVIDENCE:?按总入口设置本卡新attempt}"` 守卫；测试输入/期望向量落 `H/verification/`，运行报告一律进 `$EVIDENCE`、不进源码树。

<a id="c042"></a>

## C042 — RVV VRF、队列与 ROB 的 Context 化

- **任务**：W3-02
- **前置**：C041
- **建议 commit**：`feat(coral-rvv): context-ize RVV VRF/shared queues/ROB with owner tags (W3-02)`
- **输入/先读**：`../implementation/W3_RVV_VME_Plan.md` §6（W3-02 步骤与退出）、§3.1（上游共享状态证据：`rvv_backend_define.svh:89/113-117/134-138`、`rvv_backend.sv:1194-1309/411/466/512/800/1059-1083`）；`../implementation/W2_Context_UCE_Plan.md` §7（W2-03 头部预译码/分级资源 probe、`w2-to-w3-rvv-rob-owner` 划分要求）；C041 的 oneTile 标量闭环场景（`verification/unit/w2/scenarios/nativescalar_c2.yaml`；c1 变体 `nativescalar_c1.yaml` 同目录、config 分别 cs1_c2/cs1_c1）与 cosim/unit 入口；C011/C012 固定合同与封印 hash（随 C020 联合封印记录）。
- **修改边界**：H=`hardware/coral_nexus`：`patches/w3/0201-rvv-owner-context.patch`＋`patches/w3/series` 新条目（逐上游文件 `file:lines`+意图随条目注释）、`rtl/vme/percontext/{tb_vme_pcctx.sv,tb_driver.py}`、`rtl/vme/unit/{tb_rvv_rob_slice.sv,tb_rvv_owner_cancel.sv}`；运行报告落 `$EVIDENCE` 不进源码树；SRC=out 物化上游 `rvv_backend.sv`/`rvv_backend_dispatch.sv`/`rvv_backend_rob.sv`/`rvv_backend_vrf.sv`/`RvvFrontEnd.sv` 的 owner 分片补丁（最终提交 patch，不改 submodule）；V：无（只读）。
- **实现步骤**：
  1. producer：枚举 `rvv_backend.sv` 全部共享可变状态（multi_fifo/RS/VRF/ROB/retire 路径），逐条登记处置（复刻/分片/共享+owner 换绑/fail-stop 保留/删除）成 `patches/w3` series 条目清单；`ZVT_RS_t`/`PU2ROB_t`/`RVFEXP_t` 末尾加 `owner_id` 字段（Context 维度宽度取 C019 cs1_c2 冻结值）。
  2. VRF 按 Context 隔离，共享队列与 ROB entry 带完整 owner。定向 flush 只能移除发起者尚未接受的操作；已接受的流水/访存义务继续完成或隔离，不能清 FIFO 就抹掉在途。fflags delta 保存在所属 ROB，按 C027 有序退休合并。
  3. consumer：`tb_vme_pcctx.sv` 在同一 Verilator 实例双 Context 轮换注入 UOP 层，5 case 逐个给期望写事件序与 fflags 位图：(a) 同 rd 双写不互写；(b) flags 不合并；(c) 一 Context ROB 分片等待不阻塞另一 Context 发射（无 HOL）；(d) 单 Context 取消不清另一 Context 队列；(e) 定向 flush 只作用发起者。
  4. 负例：剥掉某一条 fifo 的 owner 标签重跑，(a)/(d) 必须 FAIL（先失败后通过，报告成对留证）；随后重跑 C041 标量闭环场景确认 RVV 改动不破坏标量路径。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn unit --suite w3 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 percontext+unit cases
  cn cosim --scenario "$H/verification/unit/w2/scenarios/nativescalar_c2.yaml" \
       --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim-regress"   # 场景=C041 供给（c1 变体 nativescalar_c1.yaml+cs1_c1）；cosim 入口=C040
  ```
- **通过/交接**：`$EVIDENCE` 内 pcctx 报告 5 case 位图全 PASS+owner 剥离负例 FAIL 成对留证；`patches/w3/0201` 条目被 C043/C044/C046 消费；W3-02 任务级 DONE 以 master 关闭表为准（需 C042+C043 全部 commit+端到端 acceptance C053），本卡单独不是 DONE。

<a id="c043"></a>

## C043 — RVV 配置、flags、访存与 owner 取消

- **任务**：W3-02
- **前置**：C042
- **建议 commit**：`feat(coral-rvv): per-context vtype/mtype/frm settlement, fflags ledgers, LSU owner path and cancel (W3-02)`
- **输入/先读**：`../implementation/W3_RVV_VME_Plan.md` §6、§3.1（`RvvFrontEnd.sv:198-345` vset*/mset* 结算与 kmax 钳位、`:173` frm 采样、`Csr.scala:843-856` frmBypass、`Decode.scala:976-993` CSR 串行化）、§17（frm 非法集 {5,6,7}）；G2 页 C031（多 Context LSU 在途/完成）、C028（发射）；C017 `csr_name_map.yaml`（逻辑名 `cn_vdecl` 等，地址不手写）。
- **修改边界**：H：`patches/w3/0202-rvv-cfg-flags-lsu-cancel.patch`＋series 条目、`rtl/vme/unit/{tb_rvv_cfg_flags.sv,tb_rvv_lsu_owner.sv,tb_rvv_owner_cancel.sv}`；运行报告落 `$EVIDENCE`；SRC：`RvvFrontEnd.sv`（结算点 per-Context 配置态+字段界挂点）、`rvv_backend.sv`（flags 账本、LSU 端口 owner 标签）、`rvv_backend_retire.sv`（fcsr 按 Context 回写）；V：无。
- **实现步骤**：
  1. per-Context vtype/mtype/vl/frm 结算状态（结算点按 Context 寻址；frm 每指令前端装配拍采样，不随在途 CSR 写抖动）；字段上下界与 vill 规则挂点就位——"按当前驻留 kernel 封印表校验"在 C045 接上，本卡落状态容器与静态界；运行期结算点拒绝 `frm∈{5,6,7}`（FRDYN=7 的构建期/Loader 静态拒在 C037/C038 侧）。
  2. flags：RVV FP 指令实际产生的 flags 全集（NV/OF/UF/NX/DZ 等、按指令）经 C027 ordered-ROB retire 有序合并进该 Context 账本，不是前端直接 fcsr 累积；不施加 VME 的 OF/NV-only 限制（后者只限 VME 出口、在 C046）；FCSR 写按 owner 归属。
  3. 访存：RVV `vle/vse`（含掩码）与 Zvt `vtle/vtse` 发射挂 C031 LSU，携带 owner/txn 标签；完成按 owner 归属、背压不跨 Context 顶替 credit。
  4. cancel：取消只打标记——**不得同拍清已接受 in-flight/外写**：已发 LSU 事务继续真实返回或按合同隔离，结清后才释放 credit/槽位；本 Context 队列只清未接受项；其他 Context 继续执行直到真实 Group fail-stop（C034 合同），不伪造局部恢复；`ctx_generation` 同拍失效旧代记录、不跨代保活；负例=同拍清在途变体（在途丢失/credit 泄漏断言 FAIL）与去 LSU 完成 owner 标签变体（归属错误 FAIL）。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn unit --suite w3 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 cfg/flags/lsu/cancel cases
  ```
- **通过/交接**：cfg/flags/LSU/cancel 4 组 tb 全 PASS+负例对；消费者 C044（E1 gate 用 mode_snapshot）、C046（VME flags 出口）、C049（RVV kernel 执行路径）；W3-02 的 commit 在本卡收齐，任务级 DONE 以 master 关闭表为准。

<a id="c044"></a>

## C044 — E1 头部 lease 请求与 E2 一次消费

- **任务**：W3-03
- **前置**：C043
- **建议 commit**：`feat(coral-vme): region lease FSM with E1 pre-queue park and E2 accepted-consume (W3-03)`
- **输入/先读**：`../implementation/W3_RVV_VME_Plan.md` §7（W3-03）、§16（E1/E2 两事件定义与 U1..U6）、附录 B.4（A1..A7）；C012 供给、C020 封印的 `H/protocol/w3_vme_decl_abi_v1.{md,json}`（位域与 fault 清单）；G2 页 C033（region_done 事件记录）、C034（可靠终态/取消）；`../implementation/W2_Context_UCE_Plan.md` §7（W2-03 头部停靠/资源 probe、region START 停在头部语义）。
- **修改边界**：H：`rtl/vme/vme_region_lease.sv`（新）、`rtl/vme/unit/{tb_vme_region_lease.sv,tb_driver.py}`、`H/verification/unit/w3/tb_region_expectations.md`（期望位图=测试输入）、`patches/w3/0301-vme-region-lease.patch`＋series 条目（W2 dispatch gate 请求线与 RvvFrontEnd accepted 消费线——本卡是集成 owner，两侧 callers 一次迁移完）；V：无。
- **实现步骤**：
  1. waiter 记录 `(ctx_id,region_id,requested_at_tick,head_inst_seq,mode_snapshot,generation)`：每 Context 至多 1 条、不可共享、不占共享执行队列；W3 是唯一 waiter writer；无 grant 不入队。实例表 busy/owner/region/last_grant 为 N 维接口+N=1 实现（位宽按 cn_protocol ABI，不自行指定硅位宽）。
  2. E1 请求/停靠：declared-target 指令到达该 Context 发射头且 valid、**进入共享 RVV 队列之前**，dispatch gate 向 lease 发 START——有空闲实例当拍授予并放行；无空闲当拍登记 waiter 并回 parked，指令停头部不进共享队列（MAT-12/20）。**waiter 登记不以 accepted 为前提**：先入队再 park=HOL、gate 后等 accepted=livelock，两种实现都被冻结合同排除，tb 必须各有检出用例。只读 predecode 可在 E1 读 pending，但不可消费。
  3. E2 消费：授予放行后，该指令被 RvvFrontEnd accepted（`inst_ready_o&&inst_valid_i` 进 `valid_inst_q` 那拍）原子消费 pending，唯一消费点；stall 保持 ARMED、不早绑不重绑（U1..U4）。
  4. 公平与唤醒：等待者授予严格按到达序 old-FIFO（不饥饿、不重排），多实例时空闲实例选择按实例固定轮转 RR（N=1 退化），同周期先释放后授予、不重复授予；成功链在可靠 region_done 后事件驱动唤醒 W2 头部，不逐 tick 轮询（U6；W6 侧 ack=终态记录可靠入队保留，不等软件消费）；owner 只在可靠 region_done 后改绑（MAT-12；失败链释放见步骤 5）。
  5. parked 取消移除 waiter；普通 redirect 与 Task 取消/故障分开，后者不恢复执行。grant 后取消停止新操作，保留旧 owner/generation 的排空记录，真实完成或隔离后发布唯一失败终态，不发布成功 region_done；尚未 accepted 也必须释放已取得的 grant。Context 及实例资源在全部义务闭合前不可复用，不以清 generation 代替结算。
  6. 12 组注入（正常/竞争/冲突 × grant/park/wake/state_free 序列，含取消/故障组），期望 waiter 位图与终态位图（成功=region_done、失败=FAILED）逐格比对（`H/verification/unit/w3/tb_region_expectations.md`，仅 N=1 适用者显式标注）。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn unit --suite w3 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 region lease cases
  ```
- **通过/交接**：12/12 注入过+负例对（"accepted 后才登记 waiter"变体在竞争用例 FAIL；"grant 后 cancel 不经失败终态结算直接释放"变体实例泄漏断言 FAIL；"取消路径误发布成功 region_done"变体终态断言 FAIL）；A7 计数器（grant/park/wake/state_free/终态）出口供 C054 PMU；消费者 C045/C046/C053（T-28/T-62/T-68）。

<a id="c045"></a>

## C045 — seed/chain 与每 kernel 配置转移检查

- **任务**：W3-03
- **前置**：C044
- **建议 commit**：`feat(coral-vme): per-kernel sealed mode/transition policy store and chain declaration checks (W3-03)`
- **输入/先读**：`../implementation/W3_RVV_VME_Plan.md` §7 步骤 4..6、§20（模式/转移表：模板级参考实例非全局白名单；旧稿 FP32-MAC 16386 为笔误、正确 16417）、附录 B.5（F1..F8）；C039（多 slot 驻留生命周期）；C011 供给、C020 封印的 `H/protocol/w3_numeric_profile_v1.yaml`（参考模板 modes/transitions 行：FP32-MAC `(16417,18)` seed/drain、BF16-MAC `(16418,265)` TK=1 与 `(16450,265)` TK=2、INT8 `(16419,256)/(16515,256)` 登记存在禁用）。
- **修改边界**：H：`rtl/vme/vme_policy_store.sv`（每 program slot 转移表/链描述符/drain 白名单/等待事件白名单）、`rtl/vme/vme_decl_front.sv`＋`RvvFrontEnd.sv` 结算点校验、`patches/w3/0302-vme-policy-store.patch`＋series 条目、装载链接线（C039 install_service→transport→policy_store 实际装载当前驻留 kernel 的表，本卡接通 producer/consumer，不让表无供给）、`rtl/vme/unit/{tb_vme_policy.sv,tb_chain_decl.sv}`；V：`pipeline_validator/package/` Loader 侧把封印 manifest 的 modes/transitions/chain 表随安装链下发（消费 C014/C037 既有 manifest 结构，不新造 schema）；运行报告落 `$EVIDENCE`。
- **实现步骤**：
  1. policy store 经本卡接通的装载链（Loader→transport→C039 install_service→policy_store）随程序安装装载**当前驻留 kernel 封印 manifest 声明**的 modes/transitions；判定表驱动——mt 使用时配置必须落在当前封印表内。参考模板 4 行只是登记进 profile 的首个实例，不是全局 3 值白名单：后续封印模板（C048 尾块模板）合法声明的组合不是非法。
  2. 合法路径包含 `模式→CLEARED→模式`；块间清配置后下一计算块必须恢复。拒绝当前 kernel 表禁止的块内转移、未恢复就使用 mt、非法字段界和本版禁用 dtype；不要把参考模板的 `vset*` 限制误施加到所有其他受控模板。fullFP32 计算另由 C046 的指令级 gate 拒绝，FP32 seed/drain 配置仍合法；非法 frm 按冻结合同拒绝。
  3. 链声明：FIRST/CONTINUE/FINAL 提交进 chain table（chain_id↔dst mt 组）；单条 FIRST|FINAL 要求 dst mt 组内全部 tile 已 seed（vtzero/vtmv/vtle，无隐式清零）；FINAL 不释放、无 auto-drain；FINAL 后仅 drain 白名单指令（`vtmv.v.t`/`vfncvtbf16.f.f.w`+`vse16.v`）；END 在最终输出操作提交之后、本身不释放（释放走 C044 region_done）。
  4. F1..F8 按冻结合同作为**声明/序列类**协议故障执行（fail-stop；静态可识别者由 Loader/构建拒，归 C037/C038 消费本卡合同表）；算术溢出/数值事件不属协议 fault（由 C046 数值路径 flags 与 KD 体系处理），不借 F1..F8 把数字类异常当协议故障；每故障类至少一对先失败后通过用例；A7 计数接 OBS。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn unit --suite w3 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 policy/chain cases
  ```
- **通过/交接**：policy/chain tb 全过+F1..F8 每类正反例、CLEARED→模式恢复正例过、INT8 命中拒绝负例过；消费者 C046（drain 白名单/合法性 gate）、C048（尾块模板自有表封印）、C047（region_state oracle 按冻结合同独立重建同一 FSM，不读本 RTL）、C053（尾块模板正例）；W3-03 的 commit 在本卡收齐，任务级 DONE 以 master 关闭表为准。

<a id="c046"></a>

## C046 — VME 数值路径、受控 drain 与可靠完成

- **任务**：W3-04
- **前置**：C045, C032
- **建议 commit**：`feat(coral-vme): BF16 exec gate, controlled drain, reliable completion and cancel settlement (W3-04)`
- **输入/先读**：`../implementation/W3_RVV_VME_Plan.md` §8（W3-04）、§17/§18（锚点 NA-01..14、NC-01..07——**向量本体与期望位只从 C011 供给的 `H/verification/numerics/numeric_contract.md` §5/§6 按 hash 读取，本卡与指南都不新造位值；本卡判据=该固定 KAT，不依赖 C047 oracle（oracle 交叉判在 C053）**；旧稿附录 B.3 P10"masked lane 配 Inf 对侧 0×Inf→NaN/NV"表述有误导，正确口径=masked lane 全排除）；G2 页 C031/C043（本地 LSU 输出腿：vtmv/vtle/vtse 落 L1）与 C032（L1→L2 MFE 直搬逐事务结算）、C034（可靠终态）。
- **修改边界**：H：`patches/w3/0401-vme-bf16-gate-drain.patch`＋series 条目（`zvt_pe_array.sv` 前端 fsrc*fmt 合法性 gate、drain 白名单执行点、T-62 顺序信号；\*\*不改 `zvt_pe*\*\_fp_lane.sv`/`fp_rounding.sv` 等数值本体**）、`rtl/vme/unit/{tb_bf16_exec.sv,tb_bf16_narrow.sv,tb_mode_fault.sv,tb_cancel.sv}`、期望矩阵 `H/verification/numerics/w3_04_g0_matrix.md`（测试输入）；SRC：`zvt_pe_array.sv`前端层；运行报告落`$EVIDENCE`；V：无。
- **实现步骤**：
  1. vtfmm 合法性 gate：`fsrc_fmt != FP16ALT` ⇒ 拒入 PE 队列+`mode_fault`（禁止 fullFP32 计算；与 FP32-MAC seed/drain 模式正交，后者累加器恒 32 位合法）。
  2. 数值执行：按 C011 的 KAT 与收窄向量 bitwise 0 容差判定，覆盖 widening 溢出 Inf、累加溢出 max-normal、jam/TK 分组。mask-off lane 必须从数值 special/flags 分类中排除，不能仅将一个操作数置零后仍让 `0×Inf` 产生 NV。VME 出口 OF/NV 与 RVV 收窄的 IEEE flags 分开判。
  3. 受控 drain 与可靠完成：FINAL→白名单 drain（`vtmv.v.t`→`vfncvtbf16.f.f.w`→`vse16.v`；本地输出腿经 C031/C043 LSU 落 L1，L1→L2 搬运才走 C032 MFE 逐事务结算）→END 在最终输出提交后；tb 逐信号断言 `compute_done≤state_committed≤output_visible≤state_free≤region_done≤event`（T-62），`state_free` 不是转交点、只有可靠 region_done 后改绑 owner。
  4. cancel 停止该 owner 的新发射；已接受 mt/LSU 操作按合同完成或隔离，flags 不串 Context。N=1 时不能声称另一个 Context 同时占用该 VME；其他 eligible RVV/scalar 工作可继续直至 Group fail-stop。结算后按 C044 失败路径释放，绝不以成功 region_done 掩盖取消。
  5. 产出 `H/verification/numerics/w3_04_g0_matrix.md`：每注入×(bits, OF/NV, mode) 期望矩阵——期望值来源=C011 固定 KAT 向量（本卡判据）；C047 oracle 的交叉核对延后到 C053，本卡不依赖未来卡。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn unit --suite w3 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 exec/narrow/mode_fault/cancel cases
  cn cosim --scenario "$H/verification/unit/w2/scenarios/nativescalar_c2.yaml" \
       --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim-regress"   # 数值路径改动不破坏 G2 闭环
  ```
- **通过/交接**：KAT/收窄 bitwise 全过、mode fault 3 组、cancel 3 组、T-62 序列断言全过、g0 矩阵落盘（`H/verification/numerics/`）；g0 矩阵被 C053/C054 checker 与 C047 交叉核对消费；本卡 commit 完成，W3-04 任务级 DONE 以 master 关闭表为准。

<a id="c047"></a>

## C047 — 独立 bit oracle、RVV 收窄与 KD-01 适配

- **任务**：W7-02
- **前置**：C020, C021
- **建议 commit**：`feat(coral-verify): independent integer bit oracle for VME/RVV numerics with KD-01 policy hook (W7-02)`
- **输入/先读**：`../implementation/W7_Verification_Trace_Plan.md` §7（W7-02a..e 子步、§7.6 不变量）；C011/C012 固定合同（C020 封印 hash）：`H/protocol/w3_numeric_profile_v1.yaml`、`H/protocol/w3_vme_decl_abi_v1.{md,json}`、`H/verification/numerics/numeric_contract.md`；C021 `spike_ref_driver.py`（scalar 驱动复用）。**本卡只需冻结合同与 scalar 驱动，可与其他 G2/G3 RTL 卡并行先行；oracle 建模与自测不依赖任何 DUT 运行结果。**
- **修改边界**：H：`verification/oracle/{bitwise_bf16_f32.py,bitwise_vme_matmul.py,bitwise_rvv_fp32.py,bitwise_vle.py,region_state.py,nexus_extend.py,oracle_main.py,known_oracle_deviation.md,selftest/}`、生成物 `verification/oracle/{oracle_capability.json,coverage_matrix.json}`；V：`pipeline_validator/tests/test_w7_oracle_bitwise.py`（注册进 suite w7）。
- **实现步骤**：
  1. 按冻结合同 S1..S7 用 Python 整数位运算独立重建：BF16 乘级 widen ROD 溢出→Inf、累加级 ROD 溢出→max-normal、jam/有限宽乘树、OF·NV；RVV FP32 指令 RNE+IEEE flags 全集与 `vfncvtbf16.f.f.w` 收窄（NV/OF/UF/NX 自然产出，不受 VME 的 OF/NV-only 限制）；`vle/vtmm` 真值；region/链 FSM（E1/E2、F1..F8）从冻结 decl ABI 独立重建。**不 import numpy/scipy 浮点路径、不读 `zvt_pe_*.sv` RTL 源码推导数值**（发现 DUT 不一致按 C4 判 FAIL 写 `RTL_VS_PROFILE`，不倒向 DUT）。
  2. KD-01：DUT 永远按 normative max-normal 判定；Spike/MPACT 的 ±Inf 结果只进入独立参考诊断列。不能由运行 manifest 选择 known-deviation 模式来改变 DUT 合格条件；修改 profile 是新合同版本，不是容差开关。
  3. 对各参考真正支持且语义相同的子集，与固定 Spike/MPACT 逐位交叉核对；mask 等没有参考支持的类用独立精确向量与整数推导验证。其他参考差异逐类定位，不先假定除 KD-01 外两个参考必然一致，更不能据其修改 normative。
  4. 能力/覆盖生成：逐语义类 `oracle_kind∈{ISA_REF,BIT_ORACLE,BOTH,NONE}`；NONE 行 → `BLOCKED(oracle 证据不足)`，不写 TODO；每规则 ≥1 向量、mask/TK/溢出类 ≥3。
  5. 实现 `oracle_main.check(t_id, run_dir)` 与 `verification/checkers/c4_numeric.py`，同卡注册进已有 `cn verify` 执行路径。固定输入/独立已知输出与翻位负例通过真实 CLI 比较，证明不是孤立的 oracle 自测；不依赖 C041 的未来接线，C007 的入口已可扩展。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；suite w7 自 C021 起注册，本卡追加 oracle cases
  ```
- **通过/交接**：全部 mandatory 数值类都有可执行独立判据及边界自测；不存在未实现的必需 oracle。输出翻位被检出，KD-01 raw 参考差异可见且不影响 normative 裁决。消费者 C048–C054；capability/coverage 是证据索引，不能以有条目替代精确结果。

<a id="c048"></a>

## C048 — BF16 matmul/conv/QK/AV 与尾块模板

- **任务**：W5-03
- **前置**：C046, C047, C038
- **建议 commit**：`feat(coral-kernels): real VME BF16 kernels (matmul/conv/QK/AV) with sealed tail template (W5-03)`
- **输入/先读**：`../implementation/W5_Kernel_Image_Runtime_Plan.md` §4.3、§1.3；`../implementation/W3_RVV_VME_Plan.md` §20（尾块模板须独立 modes/transitions，非参考表复制）；G2 页 C038（scalar 控制 kernel 与 ISA helper 适配）；C047 oracle；`examples/workloads/transformer_*`（QK/AV 形状语料）。**依据**：conv/QK/AV 均出自已批设计语料——W5 组件设计风险表"多归约 conv 链"（与 W3 联评、以 MAT-10 表示法为准）与 W3 组件设计 §8 transformer 系 q/k/v（QK^T/AV 形状来源），本卡按共享表 C048 标题落成 VME 张量族，非新增范围。
- **修改边界**：H：`kernels/library/{matmul,conv,attention}/{*.S,*.c,binding*.yaml,链接脚本}`（attention=QK^T 与 AV 变体）、尾块模板 `kernels/library/matmul/tail_tmpl/`（自带 modes/transitions 声明）、`kernels/bindings/cs1_reference.yaml` 登记；V：`pipeline_validator/tests/test_w5_vme_kernels.py`（注册 suite w5）。
- **实现步骤**：
  1. 实现 matmul K 分块累加：START→合法 seed 模式及 seed→计算模式→FIRST/CONTINUE/FINAL→受控 drain→最终输出→END；不能先访问 mt 再恢复模式。conv 的布局/im2col、QK^T/AV 的实际张量视图、形状与方程均由 binding 显式给出，不猜 operand。
  2. 尾块模板：独立受控模板+自有 modes/transitions（不复制参考模板 3 行白名单），构建期静态检查（字段界、`mset*` 位宽一致）+封印；正例路径=Loader 接受、运行时按其自有表通过、输出 bitwise 对（见步骤 3，不全量推迟到 C053）。
  3. 数值判据本卡闭合：每族 ≥1 可独立精确计算正例（matmul/conv/QK/AV 输出由 C047 oracle 整数模型按封印指令序精确算出，expected bytes 经 `oracle_main` 生成嵌进场景）——oneTile cosim 真跑后 outputs+flags 与 oracle bitwise 比对，不以"执行到 TASK_COMPLETION"替代数值判定；每族 ≥2 形状变体（含 K 尾/mask 边界）同判；BF16 入出/FP32 内部/尾部显式 `vfncvtbf16.f.f.w`；负例=INT8 行命中运行时拒绝、输出 dtype f32 冒充 bf16 构建拒、缺 ins/outs 字段构建拒。
  4. 封印+资源实测（text/静态数据/stack/scratch/spill）进 slot 台账，供 C052 复核。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn package --binding "$H/kernels/bindings/matmul_kblock.yaml" --config "$C2" --out "$EVIDENCE/package"   # package handler=C037
  cn unit --suite w5 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 VME kernel cases
  cn cosim --scenario "$H/verification/scenarios/w5_matmul_smoke.yaml" \
       --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim"   # cosim 入口=C040；场景+schema 本卡生成（explicit package/input bindings/seed data，无 timing fallback）
  ```
- **通过/交接**：每族封印 package 可加载、binding 校验全过、每族 ≥1 精确正例 outputs+flags bitwise 过+尾块变体过；负例三拒留证；消费者 C052（资源复核/节点映射）、C053（全量交叉+区域）；W5-03 族由 C048..C051 合成，本卡后 VME 张量族完成。

<a id="c049"></a>

## C049 — RVV elementwise/reduce/RoPE 与 packing

- **任务**：W5-03
- **前置**：C043, C047, C038
- **建议 commit**：`feat(coral-kernels): RVV elementwise/reduce/RoPE/packing kernels (W5-03)`
- **输入/先读**：`../implementation/W5_Kernel_Image_Runtime_Plan.md` §4.3；`../implementation/W3_RVV_VME_Plan.md` §17（RVV FP flags 全集口径）；G2 页 C038；C043（RVV 配置/flags/访存 owner 路径）、C047（`bitwise_rvv_fp32` 指令级 oracle）。**依据**：RoPE 属 transformer 前置旋转 elementwise 族（语料 `transformer_prefill_attention_*`），按共享表 C049 标题落成，非新增范围。
- **修改边界**：H：`kernels/library/{elementwise,reduce,rope,packing}/{*.S,*.c,binding*.yaml,链接脚本}`、`kernels/bindings/cs1_reference.yaml` 登记；V：`pipeline_validator/tests/test_w5_rvv_kernels.py`。
- **实现步骤**：
  1. 实现 elementwise 的 add/relu/pow；reduce_sum 固定归约顺序；RoPE 实现声明旋转方程，cos/sin 表及来源进 binding；packing 实现真实尾块 mask/布局变换。各算子都有明确输入、独立预期与实际输出，不以某个代表算子代替整族。
  2. 全 BF16 入出/FP32 内部/RNE 收窄尾；每族 ≥2 形状变体（含尾部/掩码边界）；所用指令必须全部在 C047 `oracle_capability` 覆盖集内——`oracle_kind=NONE` 的指令不得出现在封印 kernel（构建期拒）。
  3. 数值判据本卡闭合：每族 ≥1 可独立精确计算正例（elementwise/reduce/packing 逐元素/归约精确可算；RoPE 由封印 cos/sin 表+旋转方程精确可算）——oneTile cosim 真跑后 outputs+flags 与 C047 oracle bitwise 比对，不以"执行到 TASK_COMPLETION"替代数值判定；尾块/掩码边界变体同判；封印+binding 校验（场景同 C048 形态）。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn package --binding "$H/kernels/bindings/reduce_sum.yaml" --config "$C2" --out "$EVIDENCE/package"    # package=C037
  cn unit --suite w5 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 RVV kernel cases
  cn cosim --scenario "$H/verification/scenarios/w5_rvv_smoke.yaml" \
       --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim"   # 场景本卡生成
  ```
- **通过/交接**：四族封印可加载+每族 ≥1 精确正例 bitwise 过+尾块/掩码变体过；负例=binding 缺方程/表源字段→构建拒、使用 oracle 未覆盖指令→构建期拒；消费者 C050/C051（复用原语与 packing）、C052、C053。

<a id="c050"></a>

## C050 — softmax/online softmax/norm kernel

- **任务**：W5-03
- **前置**：C049
- **建议 commit**：`feat(coral-kernels): stable softmax, online softmax and norm kernels (W5-03)`
- **输入/先读**：`../implementation/W5_Kernel_Image_Runtime_Plan.md` §4.3；`examples/workloads/transformer_*`、`reduce_sum_*`（语料与形状）；C049（reduce/elementwise/packing 原语与封印惯例）。
- **修改边界**：H：`kernels/library/softmax/`（含 online 变体）、`kernels/library/norm/{*.S,*.c,binding*.yaml,链接脚本}`、`kernels/bindings/cs1_reference.yaml` 登记；V：`pipeline_validator/tests/test_w5_softmax_norm.py`。
- **实现步骤**：
  1. 数值稳定 softmax（max-subtract+由封印 RVV 指令序列组成的确定性 exp 近似——算法/系数/近似范围随 binding 封印，并对精确 softmax 声明可验证的误差上界）；online softmax（running max/sum 重标定，界内状态窗口由 binding 声明）；norm（mean/var 归约+scale/shift）。
  2. 方程、初值、界、事件与完成声明全部进 binding manifest（缺字段=构建拒，不猜）；每族 ≥2 形状变体（含长序列/尾块）。
  3. 数值判据本卡闭合，双层：(a) ISA 层——C047 oracle 按封印指令序重放 outputs+flags bitwise；(b) 数学 sanity 层——softmax/online softmax 对高精度精确参考的误差在封印近似界内（**ISA 对拍单独不能证明实现的是 softmax**）；norm/reduce 给 ≥1 独立精确可算正例；每族 ≥2 形状变体（含长序列/尾块）；封印+oneTile 场景本卡生成。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn package --binding "$H/kernels/bindings/softmax_online.yaml" --config "$C2" --out "$EVIDENCE/package"  # package=C037
  cn unit --suite w5 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 softmax/norm cases
  cn cosim --scenario "$H/verification/scenarios/w5_softmax_smoke.yaml" \
       --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim"   # 场景本卡生成
  ```
- **通过/交接**：softmax/online/norm 三族封印可加载+双层判据（bitwise+数学 sanity）过+负例（缺方程/系数/界字段→构建拒）留证；消费者 C052（族闭合）、C053（全量交叉）。

<a id="c051"></a>

## C051 — 有界 scan/recurrence 状态 kernel

- **任务**：W5-03
- **前置**：C049
- **建议 commit**：`feat(coral-kernels): bounded scan/recurrence state kernels with declared equations (W5-03)`
- **输入/先读**：`../implementation/W5_Kernel_Image_Runtime_Plan.md` §4.3（recurrence=KV 累加/状态机类）；T-56 口径（直线受控循环合法、非法控制流静态拒）；C049 原语。
- **修改边界**：H：`kernels/library/recurrence/`（含 scan 变体：`*.S/*.c`+binding+链接脚本）、`kernels/bindings/cs1_reference.yaml` 登记；V：`pipeline_validator/tests/test_w5_recurrence.py`。
- **实现步骤**：
  1. **真实 scan/recurrence 方程源自所选 kernel 的 binding 声明**：如 `s_t = α·s_{t-1} + x_t`（α、x_t 为声明 inputs 的逐步入界乘加）与 prefix_sum（profile 显式声明：定长步、逐步入界加法、无累加重排）；KV-cache 状态累加为每步真实 FMA 更新的状态方程实例——binding 必须声明状态方程、初值、步数上界、状态窗口与期望每步 FP 操作计数，输出状态与声明 inputs 按方程一致；**不把 KV 字节搬运当计算**：纯 `vle/vse` 搬运（KV append）不满足方程/计数声明，族校验必须拒绝。
  2. 有界性：迭代数与状态窗口按 manifest 上界静态可查（超界=构建拒）；直线受控循环形态（T-56）；`ctx_generation`/代际无关的纯计算体（区域/事件语义由外层 scalar helper 包裹，复用 C038）。
  3. 数值判据本卡闭合：≥1 独立精确可算正例（scan 方程逐步精确可算）——oneTile cosim 真跑后输出状态/outputs+flags 与 C047 oracle 按封印指令序 bitwise 比对＋方程一致性（状态=方程(声明 inputs)）；≥2 变体（不同方程/界）；封印+oneTile 场景本卡生成；负例=KV 拷贝冒充 recurrence → 族校验 FAIL（方程/FP 计数不匹配）、界超限 → 构建拒。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn package --binding "$H/kernels/bindings/recurrence_kv.yaml" --config "$C2" --out "$EVIDENCE/package"  # package=C037
  cn unit --suite w5 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 recurrence cases
  cn cosim --scenario "$H/verification/scenarios/w5_recurrence_smoke.yaml" \
       --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim"   # 场景本卡生成；PMU/engine 计数断言每步 FP 操作存在
  ```
- **通过/交接**：方程声明+每步 FP 计数校验+精确正例 bitwise/方程一致过+搬运冒充负例 FAIL 留证；消费者 C052（族闭合）、C053（全量交叉）。

<a id="c052"></a>

## C052 — 计算库资源再核验与源节点到映像映射

- **任务**：W5-02, W5-03
- **前置**：C048, C050, C051
- **建议 commit**：`feat(coral-kernels): library-wide resource re-freeze and source-node to image mapping (W5-02, W5-03)`
- **输入/先读**：`../implementation/W5_Kernel_Image_Runtime_Plan.md` §4.2（slot_freeze 机制与负例）、§4.3；C038/C048..C051 已封印 package 全集；C019 容量证明与 cs1_c2 slot 参数。
- **修改边界**：H：`kernels/bindings/cs1_reference.yaml` 闭合（集合唯一入口；冻结 slot 参数与映射本体写进 manifest/`cs1_reference.yaml`，消费方读 manifest 不读报告）；实测原始报告与映射报告落 `$EVIDENCE` 不进源码树；V：`pipeline_validator/tests/test_w5_library_closure.py`。
- **实现步骤**：
  1. 全族重测 text/静态数据/stack/scratch/spill 上界（readelf/map+受控模板调用图静态分析），对 cs1_c2 slot 容量逐 bank/指令存储累计复核（越限=构建拒；阈值取 manifest 注入，不硬编码）。
  2. 源节点→映像映射：`cs1_reference.yaml` 中每个抽象计算节点→kernel id→封印映像 sha256+binding hash；映射与实际映像反汇编/map 一致（T-32 口径）；映射报告落 `$EVIDENCE`（本体在 `cs1_reference.yaml`）。
  3. 计算族闭合：matmul/conv/QK/AV、add/relu/pow/reduce/RoPE/packing、softmax/online softmax/norm、scan/recurrence，加 G2 scalar 控制基础集；正例与边界变体必须实际执行。MFE indexed/gather/scatter 在 C055/C056，不能称其已由 G2 提供，也不能用它阻塞本计算库验收。
  4. `cs1_reference.yaml` 全集 package round-trip 重放（封印/加载/重载 hash 一致）。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn package --binding "$H/kernels/bindings/cs1_reference.yaml" --config "$C2" --out "$EVIDENCE/package"  # package=C037；全集重放
  cn unit --suite w5 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 library closure cases
  ```
- **通过/交接**：容量复核/映射一致/族闭合检查全过（证据在 `$EVIDENCE`）+负例（删族 FAIL、容量越限拒、同 (program_id,version) 不同 manifest_hash 登记拒）留证；消费者 C053（场景引用唯一库入口）、C054（族完整性 gate 输入）；W5-03 全族与 W5-02 资源面在本卡收口，任务级 DONE 以 master 关闭表为准。

<a id="c053"></a>

## C053 — 数值、区域与尾块正负端到端用例

- **任务**：W3-05
- **前置**：C052
- **建议 commit**：`feat(coral-verify): G3 malignant numeric/region scenarios with bitwise and lease-closure verdicts (W3-05)`
- **输入/先读**：W3 计划 §9/§11、W7 的 G3 适用 T 行；C041 的 C1–C3 基础、C047 数值比较 API、C046 固定矩阵、C052 实际计算库。完整系统 checker 在 C064，行为重放在 C069，本卡不得依赖它们。
- **修改边界**：H（`verification/scenarios/w3_*.yaml`、`verification/matrix/w3_scenarios.yaml` 清单、`scripts/w3_aggregate_manifests.py`）。本卡补齐并注册 G3 区域/终态判据，消费 C047 的 C4 adapter；C6 此时只验证窗口完整性。汇总按显式清单收集预期 run，缺任何场景失败。V 无新增执行路径。
- **实现步骤**：
  1. 场景矩阵：逐 T 写期望终态（成功链 region_done 序、取消/故障链唯一 FAILED 终态、协议故障类型、flags delta 位图；期望 bits 由 C047 oracle 产出、不手写）；仅 N=1 适用注入显式标注；竞争类用 cs1_c2 双 Context。
  2. 每场景真实 oneTile 执行，调用已实现的 C1–C3、C047 数值比较与本卡区域/终态判据。成功链检查 T-62 闭包顺序；取消/故障检查唯一失败终态、真实排空或隔离、不误发成功 region_done。超时未闭合不能通过。预期安全故障可以 case-PASS，但 DUT outcome 仍为故障，不要求它走成功输出链。
  3. 负例：每个合法场景配"去掉声明字段/缺 mset 前导恢复"变体必须被拒（Loader 或运行时 fault 二者之一，按合同定位）；对 checker 期望做 mutation（翻期望位）必须 FAIL，证明 checker 缺陷可检出。
  4. 尾块模板正例：C048 封印的参考表之外尾块模板——Loader 接受、运行时按其自有表通过、数值 bitwise 过，证明模式表非全局白名单；KD-01 向量按 oracle normative 判，raw Spike 差异记诊断不判 FAIL；任何 deviation 逐条列 KD，不静默。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  for s in "$H"/verification/scenarios/w3_*.yaml; do
    cn cosim --scenario "$s" --backend rtl --rtl-tile-mask 0x1 --config "$C2" \
       --out "$EVIDENCE/cosim/$(basename "$s" .yaml)"            # cosim=C040；场景本卡生成；每 run 产 run 级 manifest.json
  done
  python "$H/scripts/w3_aggregate_manifests.py" --matrix "$H/verification/matrix/w3_scenarios.yaml" \
    --runs "$EVIDENCE/cosim" --out "$EVIDENCE/cosim/cs1_manifest.json"
  cn verify --manifest "$EVIDENCE/cosim/cs1_manifest.json" --out "$EVIDENCE/verify"   # C047 的数值 adapter＋本卡 G3 判据；无 C069 行为重放
  ```
- **通过/交接**：全场景 verdict 闭合（无 pending/超时）、bitwise 全对、lease 闭包全过、负例与 mutation 成对留证、`$EVIDENCE` 内 malignant 报告偏差逐条列 KD；消费者 C054；本卡 commit 完成，W3-05 任务级 DONE 以 master 关闭表为准。

<a id="c054"></a>

## C054 — C4 数值证据与 CN-G3 门

- **任务**：W7-02, W7-03, W7-06
- **前置**：C053
- **建议 commit**：`feat(coral-verify): G3 numeric evidence aggregation and CN-G3 gate (W7-02/03/06)`
- **输入/先读**：`../implementation/W7_Verification_Trace_Plan.md` §8（C4 判据与负例：bit 翻转/精度放大/FP16 替换/以 RVV 冒充 VME 均 FAIL）、§10.4（门形态）、§5（非空 PASS/证据分级/可重放）；C053 全量产物与 manifest；C052 族台账。
- **修改边界**：H（`scripts/handlers/gate_g3.sh`、`verification/gates/g3_gate.py`），聚合 C047 已注册的 C4 adapter 与 C053 判据，复用同一 oracle；验证实际精度、flags、执行引擎和期望值。先由 C053 receipts 生成本 gate 的 `inputs.json`。V 新增 `tests/test_w7_g3_gate.py`，报告只入证据目录。
- **实现步骤**：
  1. C4 数值证据聚合：G3 mandatory 行（T-28 N=1、T-29、T-32、T-62、T-68 N=1 及计算族行）逐行 verdict+evidence hash **重算重跑**（re-hash+重跑 checker，不只看文件存在）；oracle 结果与 KD 清单（KD-01 normative=max-normal、raw 参考差异已登记）进 manifest。
  2. 族完整性 gate（**不省族**）：C052 四族+scalar 基础集全部有已执行场景与非空 PASS 证据；任一族缺席 → gate FAIL 并点名该族（BLOCKED 不顶替 PASS、不静默跳过）。
  3. checker mutation 证据：在 G3 证据集上重放至少各一例 bit 翻转/期望篡改/manifest dangling 引用 → 对应 checker 必须 FAIL（检出能力证明，成对留证）。
  4. CN-G3 门：先生成/取 receipt——把 C053 的 run 级 manifest.json 与 cs1_manifest.json 的路径+hash+检查类型显式写进 `$EVIDENCE/gate/inputs.json`（不指向任意 manifest），`cn gate --stage G3` 从该 inputs.json 读入后校验非空 PASS、证据引用闭合无 dangling、lease 闭包、KD 已裁；退出码语义与总入口一致（0=成立，1=FAIL，2=BLOCKED，3=证据缺失/hash 不符）。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn gate --stage G3 --out "$EVIDENCE/gate"        # gate 框架沿 C007/C041 模式，G3 stage 由本卡实现
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit"        # runner=C019；本卡注册 g3 gate cases
  ```
- **通过/交接**：全部 G3 mandatory PASS、计算族齐全、mutation 可检出、窗口完整且可取回。缺证据/bit 翻转/缺族分别被具体拒绝；本卡不宣称已完成 C069 的行为重放。C055 仅依赖 C020/C032，可提前执行；C064 明确消费本 G3 gate，随后 G4 汇合全部结果。原 W 任务关闭以总指南为准。
