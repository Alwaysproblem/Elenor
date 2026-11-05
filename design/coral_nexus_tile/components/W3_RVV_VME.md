# W3 组件设计：RVV / VME 执行与矩阵区域（BF16 数值、区域 lease）

> 本组件为设计评审稿，跨包合同与用户裁决见[总体设计](../Coral_Nexus_Tile_Design.md)；r5 指[需求规格](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。本轮检查见[设计验证记录](../Design_Validation.md)，不代表组件已实现。

状态：设计评审稿（未获批实施计划）。对应需求：`Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md` r5 §9（VEC-01～06、MAT-01～24）、§9.3/§9.4、§16.1/16.2（PAR-06）、D-21、D-26、D-29、X-26。本文只写设计 guide，不改 RTL/Python。上游固定版本：coralnpu `f795221`、coralnpu-compiler `d715556`（需求文档 §3.1）；本文全部数值/结构证据来自该 pin 的源码阅读，**均未 RTL 实跑**。数值口径以用户确认为准：**BF16 输入/输出、FP32 内部累加；VME 内部 BF16 乘/累加保留上游 round-to-odd（ROD）及实际有限宽/归约/overflow/flags 语义，不做 RNE 算术改造**（2026-11-05 用户确认）；最终 BF16 输出收窄推荐 frm=RNE，仍待review。r5 §9.3 “FP32 对外输出”的旧口径按用户决定分离为单列语料（见 §6.4），不改现存 fixture。

## 1. 定位、目标和 First Silicon cutline

- 本组件覆盖 Tile 内 **RVV 向量路径**（VEC-01～06）与 **VME/Zvt 矩阵路径**（MAT-01～24），包括：N 维同构 VME 实例的矩阵区域 lease（申请/停靠/授予/结束/排空/释放）、每 matmul 累加链声明（FIRST/CONTINUE/FINAL）、有界配置转移检查、BF16×BF16→FP32 数值合同与独立 oracle。
- 上游已核实（固定源码，非推测）：BF16×BF16→FP32 是**真实硬件路径**且在 VME 构建目标中默认编译开启——`zvt_pe_array.sv` L279-286（`ZVTBF16FMM_ON` 下 `fsrcFmt = SEW16 ? FP16ALT : FP32`，`fdstFmt` 恒 FP32）、`zvt_pe_mulbulk_fp_lane.sv` L5-37/L35（fpnew 多格式乘树，dst 断言必须 FP32）；宏由 Chisel 生成的 `rvv_backend_config.svh` 下发：`RvvCore.scala` L656-681 `GenerateBackendConfig`（`enableVme` ⇒ `define ZVT_ON`+`define ZVTBF16FMM_ON`，`enableVectorBf16` ⇒ `define ZVFBFWMA_ON`），VME 目标 flag 链 `hdl/chisel/src/coralnpu/flags.bzl` L17-29（RVV 公共 flag 含 `--enableFloat/--enableZfbfmin/--enableVectorBf16=True`，加 `--enableVme=True`）。编译侧 [C01] matrix_codegen.md L98-104/145-152（BF16 模式 `msetmtype(16418,265)` 单 K、`msetmtype(16450,265)` TK=2，`mtwiden=2`，`vtype=265`＝SEW16+LMUL2+altfmt1，指令 `vtfmm.alt.tvv`；累加器恒 32-bit）。cvfpu 固定 pin：`rules/repos.bzl` L97-116 → openhwgroup/cvfpu `bb65bdedd07711dfd41c621382f940a5cbb93046`，其 `src/fpnew_pkg.sv` L27/L45/L55 `FP16ALT`＝16 bit/8 exp/7 man＝bfloat16、L146 `ROD=3'b101`。注：`Core.scala` L173-187 的 `+define+` 列表只含 VLEN/TB_SUPPORT/ZVE32F_ON/ZVT_ON，BF16 相关宏走生成的 config 头，不在该列表。
- 目标不变量（r5 §5/§9.4）：一个可变状态一个 owner；区域是首版不可转交单位（MAT-05）；同 Context 区域不重叠（MAT-21）；并行只来自不同 Context；`compute_done ≤ state_committed ≤ output_visible ≤ state_free ≤ region_done ≤ 本地完成事件`（MAT-15）；不引入共享 AM/阵列组/Gemmini。
- 不做：多 VME 利用率估算（D-29）；INT8（`Zvti8i32mm`，上游存在但本轮范围外，能力矩阵中显式标注不支持）；FP16（`Zvtf8f32mm/Zvtf16f32mm` 中非 BF16 的 altfmt 变体不启用）；完整 RVV 程序 dialect（W5/后续）。

## 2. 职责、非职责和 ownership

| 状态/机制                          | owner（本组件）                                           | 说明与边界                                                                                              |
| ---------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| RVV 共享命令/uop 队列、VRF 访问    | RVV 前端（上游 `RvvFrontEnd.sv`、`rvv_backend*.sv` 改造） | 按 Context 隔离配置快照与标志（MAT-11、VEC-06）；START 声明指令在入队**之前**停靠（MAT-12/20，见 §5.1） |
| VME 实例（mt+PE 阵列）×N           | 每实例私有（上游 `zvt_mt.sv` mt 私有于实例）              | N≥1 设计维度，N=1 退化基线；实例号软件不可见（MAT-19）                                                  |
| 区域 state lease 表、等待记录      | 本组件新增区域管理模块（唯一 writer）                     | 每 Context 至多 1 条等待记录、1 个在持区域；与 W6 事件等待记录分权（W6 是 event wait 唯一 writer）      |
| 累加链声明跟踪（FIRST/…/FINAL）    | 本组件                                                    | 每链恰一个 FIRST 与一个 FINAL；链表来自封印 kernel/region 描述符（MAT-24）                              |
| 配置转移检查（mset*/vset* 结算处） | 本组件（挂接上游 `RvvFrontEnd.sv` 结算点）                | 模式枚举+转移表+字段上下界（MAT-10、CMP-19）；构建期与运行时同一份表                                    |
| numeric profile 定义与执行         | 本组件（定义进 target 指纹，MAT-09/PAR-06）               | 见 §6；validator 不产数值真值（VER-16、D-16）                                                           |
| BF16 I/O 转换（load/narrow）       | RVV 指令（`vle16.v`、`vfncvtbf16.f.f.w`+`vse16.v`）       | 上游编译侧已如此降级 [C01] L102/122/125；VME 本体不做 BF16 输出转换                                     |
| 操作数/输出 L1 view                | W4（L1/Arena/Frame）                                      | 本组件只持逻辑绑定（授予时冻结输出 view/初值/shape/profile，MAT-10）                                    |
| region_done 后的完成事件/退休报告  | W6（事件表、RETIRE_READY 协调）、W2（EXEC_DRAINED 输入）  | 本组件产出可靠终态记录；不等待软件消费事件（MAT-15）；RETIRE_READY 由 W6 协调，不由 W2 直接发布         |

接口依赖（箭头表示信息提供方→消费方，不是文件级实施顺序）：W2（UCE 发射、op_seq、配置）→本组件；本组件→W2（region 排空证据，W2 汇总后向 W6 发送 EXEC_DRAINED）；W4（L1 view/访存/仲裁、输出可见性判据）→本组件；W5（封印 kernel/region 描述符、链表与转移表、binding、指纹登记）→本组件；本组件→W6（region_done/失败终态记录、RETIRE_READY 协调；W6 是事件等待唯一 writer，本组件是区域等待唯一 writer）；本组件→W7（插桩与比对点）；上游固定源（coralnpu f795221 Zvt/RvvFrontEnd/rvv_backend\*、coralnpu-compiler d715556 matrix_codegen）为改造基线而非运行时依赖；无 Gemmini/共享 AM 依赖。性能口径：本组件不单独做性能宣称（P1/PN 由 总体设计 统一），不做利用率估算（D-29）。

## 3. 微架构和状态机

### 3.1 实例与路由（MAT-18～MAT-23）

- 结构：`vme_region_lease`（模块名建议，待review）维护每实例 `{busy, owner ContextKey, region_id, generation}` 表 + 每 Context 一条等待记录；实例数据面为上游 `zvt` 的完整复制维度（`ZVT_RS` 入口、mt、PE 阵列、vme2lsu/lsu2vme、fpexp、retire 命令队列各自独立）。
- 授予：START 到达且有空闲实例 → 硬件选择（work-conserving，平局确定轮转）；无空闲 → 停靠（FIFO）。同周期竞争规则按 MAT-12：多实例同周期释放、多等待者同周期到达时，每实例至多授予一个等待者、每等待者至多得一个实例；释放-授予同周期不丢唤醒、不重复授予、不提前转交。
- 绑定：授予即建立 Context→实例绑定，矩阵 uop 按绑定分发；每实例入口队列独立背压，一实例满不阻塞其他 Context 的无关 uop（MAT-22）。绑定保持到 region_done，不迁移不拆分；跨区域不读 mt 残留（MAT-10），故无需亲和性。
- 上游现状（改造量依据，X-26）：`rvv_backend_define.svh` L113-117 `NUM_VME=1`；`rvv_backend.sv` L1056-1084 仅实例化一个 `zvt`，ZVT_RS/res_vme2rvv/vmeRt/fpexp/vme2lsu/flush 均单路；`zvt.sv` L251-274 为单条按序 retire 命令队列（深度 8）；flush 为全局单路。N>1 是真实结构改造（每实例队列、owner+实例号关联返回/退休/fpexp/访存/flush），接口按 N 维书写，实现后延且不宣称 N≥2 通过。

### 3.2 区域 lease FSM（每 Context 一份）

```
IDLE ──START 声明──┬─有空闲实例──▶ ACTIVE（持有 lease：seed/累加/受控 drain）
                   └─无空闲实例──▶ PARKED（该 Context 唯一等待记录，不占共享队列；
                                    由前一 region 的可靠 region_done 唤醒并授予 → ACTIVE）
ACTIVE ──END 声明被接受（已接受操作集合关闭）──▶ END_DRAIN（仅后端服务；
           最终输出在绑定 L1 可见 + 旧访问结束 → state_free＝实例状态可回收（清零/失效），
           **不在此点向等待者转交**；lease 名义持有至 region_done）
        ──▶ REGION_DONE（唯一成功终态；region_done 可靠保存且旧事件身份稳定之后，
             才向等待者转交/再授予（owner 改绑 gate 在此，非 state_free）并发布 W6 完成事件；
             不等软件消费事件）
任一态 ──取消/故障（MAT-16）──▶ FAILED（唯一失败终态；绝不发布成功 region_done）
```

- 关键点：END 声明只关闭已接受操作集合（MAT-14/20），不释放 lease；输出背压期间保持 lease（已准入输出的端口/credit 竞争属输出背压，MAT-15）。上一区域已 END 未 region_done 时，同 Context 下一个 START 停靠不判故障（MAT-21）。
- 终态次序落地：`compute_done`＝该区域全部链的 FINAL 累加**真实完成**（加法器写回 mt，非指令被接受）；`state_committed`＝最终 mt 内容提交；`output_visible`＝最终输出在绑定 L1 范围可见（与 W4 的可见性判据对接，MAT-06/MAT-15）；`state_free`；`region_done`；然后才是事件（MAT-15）。同周期多标志置位时按此逻辑先后编码。
- 取消/故障（MAT-16）：停止新操作、保留 owner+generation、等待已接受操作真实完成或隔离、唯一失败终态；绝不发布成功 region_done；缺 END 的 Task 退出＝协议故障；基线 fail-stop（FLT-01，升级走 W6/W2）。

### 3.3 累加链跟踪（MAT-24）

- 每区域一个有界链表（条目：{mt 目的集合, 状态 SEEDED/OPEN/FINAL_SUBMITTED/FINAL_DONE}），上界由后续规格冻结。链的 mt 目的跨多个物理 mt：一个 16×16 FP32 输出块占 **4 个连续 mt**（每 mt 256 B＝64 个 FP32 元素），`zvt_pe_block.sv` L266-267/L391 的地址拼接保留基址高位 `readMtIdx[3:2]` 并由 block/行列生成低 2 位，故基址 `mt0` 的块物理占 `mt0..mt3`；上游编译把 `mt0/mt4/mt8/mt12` 用作**四个独立 32-bit accumulator 的基址**（2×2/1×4 布局，[C01] L9-14），而非同一块的四个物理 mt。
- 检查（静态可识别者在构建/加载拒绝；运行时协议故障）：缺 FIRST 的 CONTINUE/FINAL；重复 FIRST；FINAL 之后再累加/写该链 mt；END 或 Task 退出时存在未 FINAL 收尾的链。FIRST 前该链 mt 必须已由本区域 seed（`vtzero`/`vtmv`/tile load——上游 retire 命令类 `isZero/isMv2Vme/isLoad`，`zvt_ctrl.sv` L102-134；`vtzero` 目的 tile 编码于 rd，L168-172）。
- FINAL 后仅封印 kernel/profile 声明的受控 drain/输出指令（上游即 `vtmv.v.t` 提取 32-bit 行 + 可选 `vfncvtbf16.f.f.w` 收窄，[C01] L125）。上游矩阵乘确为**累加形式 only**（PE 数据面恒为 mt 读-改-写：`zvt_pe_block.sv` L262-288 读 mt、L310/L333 operands={mt 值,乘积}、L380-402 写回；无覆盖写路径），r5 “无覆盖式 matmul”成立。

### 3.4 有界配置转移检查（MAT-10、CMP-19）

- 检查点：`vset*`/`mset*` 结算新配置处＝上游 `RvvFrontEnd.sv` 配置流水（已实现字段级检查：tk 按 SEW8/16/32 钳位 ≤4/2/1、tm 钳位到 TE、altfmt 在 SEW≥32 判 vill、SEW×TWIDEN≤ELEN、保留位检查、vset\* 清零 mtype——L205-259/L262-343/L446-495）。本组件在此追加：新配置模式（{mtwiden,sew,altfmt,tk,tm,vl 域}的合法组合枚举）必须属于封印 kernel/profile 声明的转移表；越界＝协议故障（MAT-16）。
- 表随 kernel/profile 封印、进 lease 资源包络与 target 指纹；构建期（CMP-19）与运行时同一份。受控 kernel 内合法切换的实证范式：FP32 配置 seed/drain ↔ BF16 配置运算（[C01] L102：“clear/seed/drain 恒用 FP32 配置，因累加器恒 32-bit”）。
- 每区域授予时冻结逻辑合同（输出 view/初值/shape/profile/资源包络），区域内不得换逻辑合同（MAT-10）；指令级配置仍可按序变化。

### 3.5 RVV 路径要点（VEC-01～06）

- 保留上游 scalar→RVV command/uop 展开/执行队列/VRF/写回/退休路径并 Context 化：uop 与配置快照携带 owner（上游 ZVT_RS 已携带 `vector_csr` 快照+`rob_tag`，`rvv_backend_dispatch.sv` L493-505；RVV 各 RS 同理补 owner）。B 的 vset/mset 不得改 A 已接受指令的配置（MAT-11，快照隔离已具备基础）。
- 非法编码/非法寄存器/越界/不支持扩展确定失败（VEC-05）；Context 局部取消不清空他人向量队列与有效结果（VEC-06；上游 flush 全局单路，需按 owner 化——改造点）。
- VEC-04 能力清单（目标口径，以 actual 构建规则 march＝`rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma` 为准，不引 README 字串；均需编译+执行证据）：向量整数/浮点（Zve32f+Zbb）、BF16 转换与宽化（`vfncvtbf16.f.f.w`/`vfwcvtbf16`/`vfwmaccbf16.vf`，Zvfbfmin/Zvfbfwma，[C01] L92/125，RTL 路径 `rvv_backend_falu_unit.sv` L312-318/L442-452，`ZVFBFWMA_ON` 门控、经 `--enableVectorBf16=True` 生成的 config 头下发）、归约、shuffle；旧 EVU/USE 算子（relu/softmax/norm/pow/add）映射为显式绑定 RVV kernel（W5 binding），无操作数描述符不猜 A/B/C（X-19；validator 现状 `compiler/lowering.py` L845-854 仅 m/n/k/ops/accumulate）。

## 4. 接口、descriptor、寄存器和协议

身份字段（沿用 总体设计 §3.1）：TaskKey（root_launch_generation+GridInstanceId+task_id）、ContextKey（tile_id+ctx_slot+ctx_generation）、region_id（本组件分配，Context 代内唯一）、op_seq/subop_id（W2 提供）、txn_id。逻辑通道与物理编码评审后冻结；下表为逻辑接口：

| 接口        | 对端  | 方向 | 内容（要点）                                                                                                                             | 背压/错误                                                                                                         |
| ----------- | ----- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| uop 发射    | W2    | 入   | 矩阵/向量 uop（含 owner、配置快照、START/END、FIRST/CONTINUE/FINAL 声明、rob_tag、op_seq）                                               | START 在入共享队列前由 lease 检查：授予或停靠；实例入口队列满只阻塞该 Context（`engine_queue_full` 归因，OBS-07） |
| mt 类操作   | 内部  | —    | `vtzero/vtmv/tile load/drain`（isZero/isMv2Vme/isLoad/isStore/isPe，zvt_ctrl.sv L102-134）                                               | 上游 tile 访存被 `!peBusy` 全局门控（L105-106）——多 Context 需按 owner/实例化改造（待review 项 §9）               |
| 访存        | W4    | 双向 | 操作数 tile load、drain 输出 store（经 LSU/MFE），携带 owner+region_id+实例标识                                                          | 输出背压期间持 lease；drain 有界服务（MEM-02，与 W4 仲裁合同对接）                                                |
| region 终态 | W6/W2 | 出   | region_done（成功）/失败终态记录（唯一、可靠、有界）；W3 向 W2 提供 region 排空证据；W2 汇发 EXEC_DRAINED 至 W6，由 W6 判定 RETIRE_READY | 不等待软件消费事件（MAT-15）；W3 为 VME 区域等待记录唯一 writer（W6 管事件等待）                                  |
| 插桩        | W7    | 出   | grant（含实例号+触发指令身份）/park/wake/state_free/region_done、fpexp、链跟踪（MAT-17、OBS-10）                                         | debug FIFO 满不影响释放/唤醒；正常模式不留历史 trace                                                              |
| 配置表      | W5    | 入   | 封印 kernel/region 描述符：链表、转移表、drain 指令白名单、资源包络                                                                      | 静态越界在构建/加载拒绝（CMP-19）                                                                                 |
| 指纹        | W5/W7 | 出   | N、TE、NUM_MT、subtile、管线深度、numeric profile 标识（PAR-06，见 §6.3）                                                                | 变化＝artifact 作废重编译                                                                                         |

上游固定源码（pin `f79522157fc0f1d7311f44e209fc36c9c8b7a039`，实施入口参考，最终命名待review；实现目标目录 `hardware/coral_nexus/`）：[zvt.sv][H12]、[zvt_ctrl.sv][H15]、[zvt_mt.sv][H16]、[zvt_pe_array.sv][H17]、[zvt_pe_block.sv][H18]、[zvt_pe_mulbulk_fp_lane.sv][H19]、[zvt_pe_adder_fp_lane.sv][H20]、[fp_rounding.sv][H21]、[rvv_backend.sv][H13]、[rvv_backend_dispatch.sv][H14]、[RvvFrontEnd.sv][H11]、[rvv_backend_define.svh][H09]、[inc/rvv_backend.svh][H22]、[rvv_backend_falu_unit.sv][H23]、[RvvCore.scala][H24]、[Core.scala][H01]、[Parameters.scala][H08]、[flags.bzl][H25]、[rules/repos.bzl][H26]、[cvfpu pin fpnew_pkg.sv][H27]、[matrix_codegen.md][C01]、[官方 VME 文档][S04]。

## 5. 数据流、控制流和时序路径

1. **START（MAT-20/12）**：区域内首条触碰 mt 的指令带 START；申请发生在任何 mt 副作用之前、且在该指令进入共享向量命令/uop 队列之前。有空闲实例→硬件选择并原子建立 {owner, generation, region_id, 完成目标, 逻辑合同}；否则写入该 Context 唯一等待记录，Context 前端停靠（SCH-01 保证后续指令不越过；停靠不占共享队列入口，不挡其他 Context 与 owner 的返回路径）。START 之前不触 mt 的指令（配置、操作数装载）不受影响。
2. **seed**：`vtzero`（或 `vtmv`/tile load）初始化链目的 mt；FP32 配置。
3. **累加**：`vtfmm.alt.tvv`（BF16，TK≤2 打包）逐 K 步：乘树有限宽求和（输入经 round/sticky jam，非数学精确，见 §6.2）→一次舍入→mt 读-改-写再一次舍入；每条声明链位置。A 等自己的本地 load 期间，B 可跑普通 RVV/独立区域（不同实例）。
4. **FINAL**：链尾累加真实完成（写回）后该链 compute_done；FINAL 不释放、不自动 drain。
5. **受控 drain/输出**：显式 `vtmv.v.t`→VRF→`vfncvtbf16.f.f.w`+`vse16.v` 收窄后写绑定 L1 view；FP32 raw tile store 只能作**内部 FP32 scratch**，不得绕过收窄直接满足绑定的 BF16 输出（output_visible 判据以收窄后的 BF16 写为准）。输出可见前不 state_free；输出背压时持 lease 等待（有界服务）。
6. **END**：最终输出操作提交之后声明；关闭该区域已接受操作集合；硬件跟踪全部已接受 scalar/RVV/VME/LSU 子操作、配置/flags 写回与输出完成，不依赖原 Context 再执行。
7. **释放与转交**：输出可见+旧访问结束→state_free（实例 mt 清零/失效＝可回收条件，此点不转交、不改绑 owner）；此后 region_done 可靠保存且旧事件身份稳定，才向等待者转交/再授予（公平序）并发布 W6 完成事件，不等软件 poll；W2 消费 W3 的 region 排空证据并汇发 EXEC_DRAINED 至 W6，RETIRE_READY 由 W6 判定（Group 驱动退休合同）。等待者的唤醒 gate 是可靠 region_done，不是 state_free。
8. **取消/故障**：见 §3.2；generation 不撤销已发外部写；清理限于 owner 实例。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 dtype 矩阵（目标口径；全部条目需编译+执行证据后才可标“支持”）

| 路径                    | 输入 | 乘法/累加                                                                                                                                                                             | 输出                                     | 证据状态                                                                                                                                        |
| ----------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| VME matmul（主）        | BF16 | FP32（mt 内，内部 ROD 已用户确认）                                                                                                                                                    | FP32→BF16 收窄（RVV 侧）                 | 固定 RTL 与编译器源码证据（§1）；尚未构建/执行                                                                                                  |
| VME matmul（FP32）      | —    | —                                                                                                                                                                                     | —                                        | **历史参考、当前不交付**：上游存在该路径，但本轮不列入主支持矩阵、不需为其补编译/执行证据；splitk 的 FP32 对外输出亦非 D-21 关闭前置（见 §6.4） |
| BF16 输出收窄           | FP32 | `vfncvtbf16.f.f.w`＝fpnew F2F（src2=FP32→dst=FP16ALT，`rvv_backend_falu_unit.sv` L442-447，`ZVFBFWMA_ON` 门控）；rnd=frm（L238）⇒ frm=RNE 可实现；NaN/Inf/subnormal 按 fpnew F2F 合同 | BF16                                     | 代码证据（未 RTL 实跑）；**推荐 frm=RNE，待review 未确认**；与 VME 内部 ROD 分开单列                                                            |
| vecmat BF16（宽化 FMA） | BF16 | RVV `vfwmaccbf16.vf`＝FMADD src=FP16ALT→dst=FP32（`rvv_backend_falu_unit.sv` L312-318，rnd=frm）                                                                                      | 内部 FP32；完成绑定输出前必须收窄为 BF16 | [C01] L92；代码证据，待执行证据                                                                                                                 |
| EVU 类算子              | BF16 | RVV（Zve32f+Zvfbfmin 转换，按 actual march flag 集）                                                                                                                                  | BF16                                     | 映射设计（VEC-04），W5 binding                                                                                                                  |
| INT8/FP16               | —    | —                                                                                                                                                                                     | —                                        | 本轮不支持，能力矩阵显式标注（不静默降级）                                                                                                      |

### 6.2 舍入/顺序/特殊值/mask（上游事实，进 profile；改动任一项＝改结果比特，全量重对拍）

- **BF16 FMM 固定 round-to-odd（ROD），非 RNE——用户已确认沿用（2026-11-05）**：乘级 `zvt_pe_mulbulk_fp_lane.sv` L272（src≠dst 强制 ROD）与 `zvt_pe_block.sv` L90-91（FPMUL 且 fsrcFmt≠FP32 时 tag.rndMode=ROD）双重强制；**累加级加法器使用同一 tag**（`zvt_pe_block.sv` L311/L334）→ BF16 路径乘积与每步累加均 ROD 舍入到 FP32。FP32 FMM 用 vector CSR `frm`（`rvv_backend_dispatch.sv` L505 直 cast）。ROD 完整实现与编码（101）见 `Zvt/fp_rounding.sv` L66-105。**用户决定：保留上游内部 ROD 及实际有限宽（jam）/归约顺序/overflow/flags 语义，不做 RNE 算术改造**；以上任一项后续若变＝改结果比特，按 MAT-09 全量重对拍。
- **FMA 顺序（确定性；乘树并非数学精确——更正）**：每条 vtfmm 内 ≤TK=2 个 K 元素乘积进批归一化定点树**求和后一次舍入**，但树输入在 `ALIGN_PRODUCT_TO_MSB` 下先把 round/sticky jam 进 significand LSB（`zvt_pe_mulbulk_fp_lane.sv` L415-419），即输入是截断+jam 的有限宽表示，非未截断精确乘积；跨指令 K 步按程序序对 mt 读-改-写；M 向按 `zvt_pe_array.sv` cnt 分片序（stripmine）确定。顺序唯一确定、oracle 按此有限宽合同比特建模（§8.1）；与数学 exact 参考的差异只作诊断，不作相等必过门槛。
- **溢出不对称（代码证据，未 RTL 实跑；语义按用户确认沿用）**：widen（BF16→FP32）乘级溢出→Inf（`zvt_pe_mulbulk_fp_lane.sv` L561-564 注释 "widen-mode ROD overflows to inf, not max-normal"）；累加级溢出按舍入模式（ROD/RTZ→max-normal，RNE/RMM→Inf，`zvt_pe_adder_fp_lane.sv` L149-151）。列入 oracle 必验（§8.2）；profile 精确条目的最终封印以实跑＋bitwise oracle 通过为准。
- **特殊值**：NaN→规范 qNaN `0x7FC00000`；NV＝invalid（Inf−Inf、0×Inf）；Inf 传播；subnormal 输入/输出走 subnormal-align 路径（`fp_align`/`fp_mulfront`；mulbulk 对 subnormal×subnormal 有负指数批归一化处理，L113-121）。**VME 只向 fpexp 上送 OF/NV**；NX/UF/DZ 不出 VME（`zvt_pe_block.sv` L202-210/L363-374 只 OR OF/NV）——沿用固定上游内部数值路径；能力声明仍需独立 oracle 与实跑验证（§9）。
- **mask/tail**：被 mask 的 K 元素**不参与**数值与 special 分类/fflags——上游在乘级入口先 mask 后分类（`zvt_pe_mulbulk_fp_lane.sv` L83-89 `operands_masked` 置零发生在 `fp_mulfront`/special 判定之前，masked lane 不产生 NaN/Inf/NV；`fmt_product_enable` 同时把它移出加法树），全 mask 的输出元素不加不写（mt undisturbed：`zvt_pe_block.sv` L106 maskIn、L136 乘级 in_valid、resMask 门控加法器 L315 与写回 L398）；部分字节 mask 被断言禁止（L164-169）。**不得默认“乘零无害”**：若后续实现把 mask 实现为乘 0，被 mask lane 撞上未 mask 的 Inf 即 0×Inf→NaN/NV——故合同要求“mask 先于操作/special 分类”的证据（现存 source 即如此），否则设专项 gate 验证。tm/tk 尾部由封印 kernel 的合法 K 尾处理覆盖（MAT-10）。
- **frm=DYN 禁止**：dispatch 将 frm 直 cast 为 roundmode（L505），`FRDYN(7)` 会落入未定义舍入路径；区域 kernel 合同必须钉 frm∈{RNE,RTZ,RDN,RUP,RMM}（静态拒绝 DYN）。

### 6.3 参数与指纹（PAR-06/PAR-08）

进入 target 指纹的 VME/RVV 字段（建议名，映射现行 `boa_*`，CMP-16）：`vme_instance_count`（=N）、`vme_te`（=VLEN/8，默认构建 VLEN=128→TE=16，`Parameters.scala` L100-121）、`vme_num_mt`（16）、`vme_subtile_bytes`（16）、`vme_pipe_mulbulk/adder`（上游默认 3/3，`zvt_pe_block.sv` L34-35）、`rvv_vlen`、`rvv_queue_depths`、`numeric_profile_id`。现行 validator 指纹字段 `boa_dtype_bytes=2`/`boa_acc_bytes=4`（`config.py` L252-253、`execution_verifier.py` L82-83）语义上与 BF16-in/FP32-acc 一致，须重命名映射并保持数值语义（BF16 I/O：input_bytes=2、accumulator_bytes=4、**output_bytes=2** 为新增字段，待review）。validator 现状 BOA 仅 latency 模型（`engines.py` L205 `latency = launch_overhead + ceil(ops/peak_macs)`），launch 进 `opaque_events`、FREE_L1 前须全部 await（`execution_verifier.py` L791/L827）——RTL 接入后该保守保护由真实 VME 完成点替代（对齐 W5/W7 改造，不在本组件内改 Python）。

### 6.4 旧 FP32 输出语料（单列，不修改现存 fixture）

- r5 §9.3 参考负载最低要求原文含“FP32 累加**与输出**”；用户本轮收窄为 BF16 I/O。已核实（按**对外 L2 输出 dtype** 划分，不把内部 f32 L1 buffer 计入）：`matmul_gather_add*.mlir` 与 transformer 系（如 `transformer_prefill_attention_pipeline.mlir` 的 q/k/v L2 输出均 bf16）本就兼容 BF16 I/O——transformer 的 f32 仅是内部 L1/scratch；`matmul_splitk_pipeline.mlir`（bf16 入、`%c_l2` f32 对外输出、f32 S scratch）属旧 FP32 对外输出口径。处置：BF16-I/O 主验收语料为前者＋新增 BF16 输出定向用例（§8）；splitk 类列为“FP32 输出扩展语料”记录，非 D-21 关闭前置，**不改写其 dtype 假设、不冒称已覆盖 BF16 输出**。

## 7. RTL/软件实现建议

1. 新增 `vme_region_lease`（名待review）：实例表+等待记录+链表+转移表检查+终态次序编码；上游 zvt 数据面按实例参数化复制（N=1 时退化为单实例字面行为）。retire 命令队列、fpexp、vme2lsu/lsu2vme 每实例独立并携带 owner+实例号（MAT-22）。
2. START/END/FIRST/CONTINUE/FINAL 声明编码：三种载体候选——指令内标志位、伴随 CSR 写、自定义指令——**并列待review，首版评审定一种**（Coral Zvt 现有编码是否有可用位未核对，MAT-20 原文）；无论选哪种，声明快照必须**一次性原子采集**并与所属指令/区域绑定，不被中断/异常/Context 切换污染或重复消费（本设计不预设先后实现顺序）。
3. 配置转移检查挂在 `RvvFrontEnd.sv` 结算点（§3.4），表由封印产物下发（与 W5 的 manifest 字段对齐）；静态拒绝在 Loader（CMP-19）。
4. flush 按 owner 化：上游 `flush` 全局清空（`zvt.sv` L273 `clear(flush)` 等），多 Context 下改为 per-owner 取消路径（MAT-16/22），Group fail-stop 域仍由 W6/W2 主导。
5. tile 访存 `!peBusy` 门控（`zvt_ctrl.sv` L105-106）改 per-owner/per-instance 判定，避免 A 的 drain 被 B 的 PE 占用无界阻塞（MEM-02 有界服务）。
6. 低能力实现 agent guide（不另造降阶模拟器）：先读上游 Zvt 数据面与 `RvvFrontEnd` 结算逻辑→实现不变量（§3.2/§3.3 次序、每 Context 一等待记录、链表闭合、转移表命中）→用 T-62/T-68 定向序列证明；停止条件＝不变量全成立且恶性用例清单（§8.3）全过；周期数不对拍 Python（时间权威在 Python Simulator，总体设计 §5.1 P1/P3）。

## 8. 验证、bring-up 和验收标准

### 8.1 独立数值 oracle 策略（W7 协同，本组件定义合同）

- 独立实现：**先冻结合同、再写 oracle**——以 §6.2 的有限宽/jam/ROD/溢出不对称/special/mask-undisturbed 合同为唯一规范，用 Python 整数位运算独立重建比特级 oracle（不读 RTL 代码推导，保持独立性）；与 RTL 联仿逐元素比对要求**比特一致（容差 0）为唯一必过 gate**。数学 exact 参考（mpmath/高精度）只作**诊断**（量化有限宽误差），不是相等必过条件；**发现不匹配时先查实现与规格，不得反向修改 oracle 去凑 DUT**；oracle 变更只能来自合同变更（MAT-09 全量重对拍）。numpy/torch FP32 参考仅 sanity。validator 不产数值真值（VER-16）。
- 对拍点：每 vtfmm 后 mt 子tile、drain 后 VRF/L1 输出、fpexp 标志。

### 8.2 恶性用例（数值）

ROD 边界（guard/round/sticky 全组合、恰好半 ULP 奇偶差）；Inf×0、Inf−Inf、±Inf 累加溢出（两级不对称各验）；subnormal×subnormal（负批归一化指数）；BF16 最大值累加到 FP32 溢出；K=1（TK=1 模式 16418）与 K=2（16450）混合及**TK=1/TK=2 分组差**（同一 K 序列不同分组逐比特对比）；**乘树有限宽专项**：大指数差乘积对、异号大幅抵消、round/sticky jam 边界（`zvt_pe_mulbulk_fp_lane.sv` L415-419），与数学 exact 的差异按 §8.1 作诊断并给出量化；K 尾奇数；mask 全零/半 mask（断言路径）及 **masked lane 配 Inf/NaN 对侧操作数**（验证 mask 先于 special 分类：不得出现 0×Inf→NaN/NV，无 mask-before-op 证据时为专项 gate）；**链跨物理 mt**（基址 mt0 的 16×16 块占 mt0..mt3）与**多基址布局**（mt0/4/8/12 四个独立 accumulator，[C01] L9-14）；FP32 seed（含非零 `vtmv` 初值）→BF16 运算→FP32 drain 切换；收窄 `vfncvtbf16` 的 RNE/ties 边界（若 frm=RNE 获批）；FRDYN 静态拒绝；链未 FINAL 即 END。

### 8.3 区域/多实例用例（T 映射）

- **T-62（必验，N=1 即可）**：A 跨多次 K 累加等待本地 load、B 同号寄存器 RVV 并申请区域；A 受控切换 seed/drain/运算配置；最终输出背压、取消、迟到回包；注入越界配置转移。期望：A 状态与配置保持、不污染 B；B 独立推进；申请只入队一次、安全释放后授予；输出可见+旧访问清零后 state_free 再 region_done；越界静态拒/运行时协议故障；故障不报成功、零重复释放、循环依赖程序被拒。
- **T-68（必验）**：起止与链声明的全部非法序列（缺 START、无绑定触 mt、持区域再 START、END 后触 mt、缺 END 退出；单步 FIRST|FINAL、多步链、缺 FIRST 的 CONTINUE/FINAL、重复 FIRST、缺 FINAL 即 END/退出、FINAL 后再累加/写 mt、FINAL 后受控 drain 合法）；FINAL 后输出背压持 lease；END 后 region_done 前下一个 START 停靠不判故障；compute_done 以 FINAL 真实完成为准。
- **N≥2 项（T-68 尾段）**：设计接口与等待/授予仲裁按 N 维书写；**N=2 的 RTL 验证随 N>1 实现阶段延期**（D-29），本轮不宣称通过；N=1 必须全过。两 Context 并行于不同实例、结果与 N=1 串行比特一致等条目列入延期清单并在实现阶段补验。
- 相关：T-05/T-06（在途切换、交替改配置不串扰）、T-28（矩阵/scalar/RVV 交错）、T-29（尾部/mask/特殊值/累加初始化——oracle 主战场）、T-34（饱和 VME 混合公平性）。

### 8.4 出口判据（本组件视角）

CS1 出口：T-62/T-68（N=1）全过；§8.2 数值恶性用例 bitwise oracle 全过（含 masked-Inf 专项）；numeric profile 的精确定义/ID 在构建前冻结并进入实验 package 指纹；§6.1 主支持条目必须在实跑＋bitwise oracle 通过后才标 validated/supported（不能先宣称支持，测试所需 package 也不能等待测试通过才封印）；D-21 以 BF16 主路径实跑证据关闭（FP32 输出扩展语料另列、非前置）；停顿归因映射（VME 忙按实例，OBS-07/09/10）落地。

## 9. 风险、取舍和后续细化方向

1. **cvfpu 有公开固定 pin（更正先前结论）**：`rules/repos.bzl` `cvfpu_repos()`（L97-116）`http_archive` 指向 openhwgroup/cvfpu `bb65bdedd07711dfd41c621382f940a5cbb93046`（sha256=0723e6a6…，带 0001-0010 patch 列表）；该 pin 的 `src/fpnew_pkg.sv` 可公开读取，含 `FP16ALT`={8,7}（L27/L45/L55）与 `ROD=3'b101`（L146）。本文数值证据统一引用该 pin（+patch 后的语义仍以仓内 `Zvt/fp_*.sv` 实现为准），manifest 记录 pin+patch 集（PAR-01）。
2. **Zvt v0.4 规范 PDF 不可达**：lists.riscv.org attachment 本环境无法下载；以官方 VME 页面（developers.google.com/coral/guides/hardware/vme-matrix-engine，已核对引文）+ 固定 RTL 内 “Zvt §12.1.1.x” 注释为准；如后续可获取 PDF 再补规范级引用。
3. 声明编码空间（START/END/链位置）未核对 Zvt 编码余量（MAT-20）：三种载体（指令内标志/伴随 CSR 写/自定义指令）并列待review，首版定一种并一次性原子快照（§7.2）。
4. `boa_*`→VME 指纹字段重命名与 `output_bytes=2` 新增、`frm=DYN` 静态拒绝位置（前端 vs Loader）——均待review，不视为用户已确认。
5. 上游单路结构（全局 flush、`!peBusy` 门控、单 retire 命令队列）在多 Context/多实例下的改造规模是主要进度风险；接口已按 N 维设计，但任何 N≥2 的时序/面积结论本轮不做（含利用率，D-29）。
6. BF16 输出收窄在 RVV（`vfncvtbf16.f.f.w`＝fpnew F2F，rnd=frm，代码证据未实跑）而非 VME 内：**推荐 frm=RNE，待review 未确认**；获批后与内部 ROD 一并进 profile 与 oracle。
7. **数值决策已定项（2026-11-05 用户确认）**：VME 内部 BF16 保留上游 ROD 及实际有限宽（jam）/归约顺序/overflow 不对称/仅 OF·NV flags 语义，不做 RNE 算术改造；本文 ROD 条目按 pin 源码书写并标“未 RTL 实跑”，profile 定义/ID 先冻结供构建封印，能力在实跑＋bitwise oracle 通过后才标 supported（§8.4）。
8. 构建口径：actual march＝`rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma`（W0 核对，与 README 字串不同，以 actual 构建规则为准）；`ZVTBF16FMM_ON`/`ZVFBFWMA_ON` 经 Chisel 生成 config 头启用（§1），若新 Tile 改用自建构建须显式重下发这些宏并进 manifest（PAR-01/PAR-02）。

[C01]: https://github.com/google-coral/coralnpu-compiler/blob/d715556c3853719e414182c0bf5f4a10f248f1d9/docs/design/matrix_codegen.md
[S04]: https://developers.google.com/coral/guides/hardware/vme-matrix-engine
[H01]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/Core.scala
[H08]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/Parameters.scala
[H09]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/inc/rvv_backend_define.svh
[H11]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/RvvFrontEnd.sv
[H12]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/Zvt/zvt.sv
[H13]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/rvv_backend.sv
[H14]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/rvv_backend_dispatch.sv
[H15]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/Zvt/zvt_ctrl.sv
[H16]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/Zvt/zvt_mt.sv
[H17]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/Zvt/zvt_pe_array.sv
[H18]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/Zvt/zvt_pe_block.sv
[H19]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/Zvt/zvt_pe_mulbulk_fp_lane.sv
[H20]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/Zvt/zvt_pe_adder_fp_lane.sv
[H21]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/Zvt/fp_rounding.sv
[H22]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/inc/rvv_backend.svh
[H23]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/rvv_backend_falu_unit.sv
[H24]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/rvv/RvvCore.scala
[H25]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/flags.bzl
[H26]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/rules/repos.bzl
[H27]: https://github.com/openhwgroup/cvfpu/blob/bb65bdedd07711dfd41c621382f940a5cbb93046/src/fpnew_pkg.sv
