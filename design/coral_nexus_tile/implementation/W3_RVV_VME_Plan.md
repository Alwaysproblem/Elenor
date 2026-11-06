# W3 实施计划：RVV / VME 执行与矩阵区域（对应 components/W3_RVV_VME.md）

> **逐 commit 执行入口**：[execution_guide.md](../execution_guide.md)。先用该入口选择前置已满足的卡片；本页保留工作包细节，commit 顺序、工具可用时点、文件边界和原任务关闭条件以执行 guide 为准。架构语义仍遵守已批准设计，不按旧草稿中的占位命令或文字存在性检查宣称完成。

> 状态：中文文件级实施计划（本轮只写计划，不执行构建/代码修改/测试；见[总体计划](../plan.md) §2/§3 与共享授权）。数值/结构事实均引自固定上游源码 pin（coralnpu `f79522157fc0f1d7311f44e209fc36c9c8b7a039`、coralnpu-compiler `d715556c3853719e414182c0bf5f4a10f248f1d9`、cvfpu `bb65bdedd07711dfd41c621382f940a5cbb93046`），已逐行核对并在 W3-01 产物中冻结证据 manifest；**均未 RTL 实跑**，凡"未确认"处都给出实证任务而非默认值。硬件/镜像路径全部是待创建计划目标，不代表现状。本文件不引用 `local://` 共享合同（不持久），与 parent 的口径冲突均列于 §13。

## 1. 已选方案摘要（承接总体计划/合同裁决）

- **路线**：Coral 派生完整 Tile RTL（W1 联仿边界外保留 Python），VME 承担原 BOA 矩阵职责、RVV 承担向量计算；`hardware/coral_nexus/` 为待创建 workspace。
- **数值（用户已确认，2026-11-05）**：BF16 张量 I/O、FP32 内部累加；**VME 内部保留上游 round-to-odd（ROD）与实际有限宽乘树/累加路径/溢出不对称/仅 OF·NV flags 语义，不做 RNE 算术改造**；最终 BF16 输出收窄 frm=RNE（已获准）。无 tolerance 放宽；独立整数 bit oracle 由 W7-02 按 W3-01 冻结合同实现，W3 不写 oracle 本体。乘级 widen ROD 溢出→Inf、**累加级 ROD 溢出→max-normal**（固定 RTL 合同为准，Spike/MPACT F32AddOdd→Inf 差异由 W7 登记为已知偏差 KD-01，不回写合同）。
- **声明载体（已选推荐方案）**：受控 helper（kernel 内紧邻指令对）＋每 Context 伴随 CSR 声明（`CSR_VME_DECL`，逻辑 carrier；物理地址由 W1-01 csr_name_map/csr_allocator 冲突核查后冻结，不在本计划指定数值）。START/链位置 pending metadata **原子绑定下一条声明目标指令**；**请求/停靠与消费两事件分离**：START 请求/停靠发生在目标指令到达该 Context 发射头且 valid、进入共享 RVV 队列**之前**（W2 dispatch 侧 gate，无空闲实例即登记等待记录并停在头部，不占共享队列）；pending 只在获授予、该指令真正 accepted（进入 RvvFrontEnd）时消费；stall 保持 pending，redirect/cancel 显式清空；END 是显式无目标声明（csrrwi 立即数式，上游 `Decode.scala:1194/1339` 已支持）；FINAL 不释放、无 auto-drain；END 在最终输出操作提交之后；`state_free` 不是转交点，只有可靠 region_done 后才改绑 owner（两事件精确定义见 §16）。
- **模式**：`vset*`/`mset*` 结算处追加模式枚举＋转移表＋字段上下界（MAT-10/CMP-19）；构建期与运行时同一份表；`frm=DYN(7)` 构建与 Loader 静态拒绝、运行期仍在结算点检查。
- **N 维**：所有 lease 表/仲裁/返回/插桩/指纹接口按实例维度书写；本轮只实现/验证 **N=1**（上游 `rvv_backend_define.svh:114` `NUM_VME=1`，`rvv_backend.sv:1059` 单 `zvt` 实例）；N>1 的 RTL 实现与 T-68 N≥2 子项、T-59/60 延期（D-29），不宣称通过。
- **性能口径（parent 2026-11-05 裁定）**：P1=同一硬件（cs1_c2 同 RTL hash）仅 1 Context 活跃；PN=同一硬件多 Context 活跃，两者 workload active-context 数不同、其余相同；cs1_c1 仅是退化功能回归，不参与 P1/PN 对比。W3 不出面积/Fmax/利用率数字（D-29）。

## 2. 非目标与唯一 owner

非目标（本轮 W3）：完整 RVV 程序 dialect、通用 LLVM/IREE 后端（W5）；INT8/FP16 VME 数值支持（显式不支持）；`Zvti8i32mm`/INT8；共享 AM/多 mt bank lease/阵列组；逐 tick 轮询唤醒（L1 gate 事件驱动）；N≥2 RTL 验证；利用率/PPA 承诺；把 boa_legacy timing 别名伪装成 VME 配置。

| 唯一 owner（本轮文件级）                                                                             | W3（本人）                                           | 边界                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hardware/coral_nexus/rtl/` 内 RVV/Zvt 域 RTL 与补丁序列中 RVV 前缀条目                              | ✔                                                    | `Core.scala`/`SCore.scala`/`Csr.scala`/`Decode.scala` 集成由 **W2** 执行（W3 出请求单）；`Parameters.scala`/`flags.bzl` 由 **W0**；LSU/MFE/L1 由 **W4** |
| `hardware/coral_nexus/rtl/vme/` 新增 `vme_region_lease.sv`、`vme_policy_store.sv`、decl CSR 消费前端 | ✔                                                    | decl CSR 的 CSR 端口/地址分配在 W1-01 协议域 + W2 Csr.scala 集成                                                                                        |
| `pipeline_validator/`                                                                                | ✘ 只读证据引用                                       | 行为适配（事件表/驻留/取消）归 W6/W5；共享接线归 W1                                                                                                     |
| oracle 实现与 CS1 checker                                                                            | ✘（合同在 W3，代码归 W7-02/W7-03）                   | `verification/numerics/oracles` 归 W7                                                                                                                   |
| 固定 C/asm kernel 库、封印                                                                           | ✘（合同引用 W3 文件，代码归 W5-02/03/04）            | `kernel/` 序列由 W5                                                                                                                                     |
| `cn.sh` 与 gate CLI                                                                                  | ✘                                                    | 由 parent/W0 冻结；G 名为 CN-G0..CN-G5                                                                                                                  |
| `protocol/cn_protocol_v1.yaml`、`csr_name_map.yaml`                                                  | ✘（W3 写需求碎片 `protocol/field_requests/w3.yaml`） | 由 W1-01 单点合并                                                                                                                                       |

## 3. 当前文件证据与待创建清单

### 3.1 已存在代码证据（2026-11-05 已逐行核对，源 = 固定 pin URL）

| 主题                                  | 文件:行                                                                                                                                                                                                                                                             | 内容                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| VME 单实例/维度                       | `rvv_backend_define.svh:113-117`(NUM_VME=1)、`:200-225`（ZVT_RS_DEPTH=2·ZVT_LMUL、NUM_MT=16、SUBTILE_SIZE=16、NUM_BLK=4、TE=VLEN/8、COMPRATIO=4/TE、PROCESS_DELAY=TE/4、ZVT_LMUL=32·TE/VLEN）、`:89`（NUM_VRF=32）、`:84/:134/:138`（ROB_DEPTH=16/ROB_TAG_WIDTH=4） | N=1、mt/子tile/TE 几何；W3-02 复刻与 W3-03 N 维参数化的改造量依据                |
| BF16 真实路径                         | `zvt_pe_array.sv:276-288`（OPFVV: FPMUL，`ZVTBF16FMM_ON` 下 fsrcFmt = SEW16 ? FP16ALT : FP32，fdstFmt 恒 FP32）                                                                                                                                                     | SEW16 ⇒ BF16(fsrcFmt=FP16ALT)，SEW32 ⇒ FP32（本交付排除后者，见附录 B.1 模式表） |
| 乘级强制 ROD                          | `zvt_pe_block.sv:90-91`（FPMUL 且 fsrcFmt≠FP32 ⇒ tag.rndMode=ROD）；`zvt_pe_mulbulk_fp_lane.sv:272`（src≠dst ⇒ rnd=ROD）                                                                                                                                            | BF16 乘级＋批归一化树均 ROD                                                      |
| 累加级消费同一 tag                    | `zvt_pe_block.sv:311,334`（adder `.rnd_mode(mulbulkCmd.tag.rndMode)`）                                                                                                                                                                                              | 每步累加 ROD                                                                     |
| ROD 编码/溢出 saturate                | `fp_rounding.sv:91`(ROD round_up 公式)、`:99-104`(of_saturate：RTZ/ROD→max-normal)                                                                                                                                                                                  | 累加级溢出→max-normal；非对称证据                                                |
| widen 溢出→Inf                        | `zvt_pe_mulbulk_fp_lane.sv:561-564`（src≠dst ⇒ overflow_result=Inf）                                                                                                                                                                                                | 乘级与累加级不对称                                                               |
| 特殊值/mask 先于分类                  | `fp_mulfront.sv:64-99`（classifier/Inf×0/NV/Inf）、`zvt_pe_mulbulk_fp_lane.sv:84-89`（operands_masked 先置零）、`:152-231`(special 组合)、`zvt_pe_block.sv:136,315,398,194`（in_valid/resMask/写回门控）                                                            | masked lane 不参与/不产生数值与 special                                          |
| 仅 OF/NV 出 VME                       | `zvt_pe_block.sv:202-210`、`:363-374`；`zvt.sv:23-25`（fpexp 输出）；`rvv_backend_retire.sv:222-252`（zvt fpexp 直写 fcsr）                                                                                                                                         | VME 侧无 NX/UF/DZ 输出                                                           |
| 累加读改写/无覆盖写                   | `zvt_pe_block.sv:262-288`(读 mt 拼 BLKID)、`:299-348`(adder)、`:379-402`(写回)；RAW 保护 `zvt_pe_array.sv:195-207`                                                                                                                                                  | 链跨 4 mt 的物理布局证据                                                         |
| tile seed/drain 指令与 `!peBusy` 门控 | `zvt_ctrl.sv:101-139`、`:140-199`、`:592-599`（uopRdy）、`:630-655`（`!peBusy` 全局门）                                                                                                                                                                             | 本轮多 Context 改造点                                                            |
| 配置结算/转移检查挂点                 | `RvvFrontEnd.sv:198-287`（vset*/mset* 解析）、`:288-345`、`:346-470`（msettn/msettm/msettk、kmax 钳位、vill 规则）                                                                                                                                                  | W3-01 转移表挂点                                                                 |
| frm 采样                              | `RvvFrontEnd.sv:173`(`frm = frm_i`)；`Csr.scala:843-856`(frmBypass)                                                                                                                                                                                                 | 每指令在前端装配时采 frm，不随在途 CSR 写抖动                                    |
| Zvt 指令分类器                        | `RvvDecode.scala:169-333`（isMset\*/isVmeLdSt/isVtfmm/isVtzero/isVtmv/isVtdiscard/readsTile/writesTile/accessesTile）                                                                                                                                               | 静态 Loader 与前端触mt谓词的位证据                                               |
| CSR 现状与 custom 占用                | `Csr.scala:45-100`（CsrAddress：0x001/002/003、0x008-00a、0x300-344、0x7a0-7b3、0x7c0-7c7、0x7e0/7e1、0xb00-0xb02/0xb80/0xb82、0xc20-0xc23、0xf11-f14、0xfc0-0xfd4）                                                                                                | 候选自由段与 W1-01 唯一冲突 manifest 的输入                                      |
| CSR 串行化规则                        | `Decode.scala:976-993`（CSRs 须 ROB empty、仅 lane 0、mtype 读 stall 到 RVV idle）；`:1093/:1339`（csrrwi 立即数支持）                                                                                                                                              | decl carrier 需 W2 放宽 ROB-empty 门的原因                                       |
| RVV 标量/矩阵退休与 flags             | `rvv_backend.sv:1194-1309`（rob/retire/vrf 实例与端口）                                                                                                                                                                                                             | W3-02 复用/分区依据                                                              |
| trap flush 全局清空                   | `rvv_backend.sv:411/:466/:512/:800/:1059-1083`（所有 fifo/RS/.flush=trap_flush_rvv）                                                                                                                                                                                | VEC-06 owner 化改造点                                                            |

### 3.2 已核实固定上游 URL（全部可公开读取，实施时再读）

固定 pin 文档/源码：coralnpu `f795221…` `hdl/verilog/rvv/design/Zvt/{zvt,zvt_ctrl,zvt_mt,zvt_pe_array,zvt_pe_block,zvt_pe_mulbulk,zvt_pe_mulbulk_fp_lane,zvt_pe_adder,zvt_pe_adder_fp_lane,fp_rounding,fp_align,fp_mulfront,fp_addfront,fp_absaddsub,fp_lza}.sv`、`hdl/verilog/rvv/design/{RvvFrontEnd,rvv_backend,rvv_backend_dispatch,rvv_backend_falu_unit,rvv_backend_rob,rvv_backend_retire,rvv_backend_vrf}.sv`、`hdl/verilog/rvv/inc/{rvv_backend_define,rvv_backend}.svh`、`hdl/chisel/src/coralnpu/scalar/{Csr,Decode,SCore,Bru}.scala`、`hdl/chisel/src/coralnpu/rvv/{RvvCore,RvvDecode,RvvInterface}.scala`、`rules/repos.bzl`（common_cells pin `pulp-platform/common_cells@6aeee85d0a34fedc06c14f04fd6363c9f7b4eeea`，`lzc.sv:16-22` **记录 in_i 全零 ⇒ cnt_o=WIDTH-1、empty_o=1**）；cvfpu pin `bb65bde…` 的 `src/fpnew_pkg.sv`（FP16ALT={8,7}、ROD=3'b101，`:35-64`、`:146`）**须加 coralnpu `third_party/cvfpu/0001-0010` 补丁后随仓内构建**，数值证据以补丁后构建为准（`rules/repos.bzl:97-115` 已核对补丁清单）；coralnpu-compiler `d715556…` `docs/design/matrix_codegen.md`（[C01] §1/§2.1/§2.3）；官方 VME 页 [S04]。**`Zvt v0.4` 规范 PDF 本环境不可达**（W3 计划 §8），以 RTL 注释与官方页为准；如可获取再补规范级引用。

### 3.3 待创建（本轮只列计划目标路径）

| 路径                                                                                                   | owner 任务 | 说明                                                             |
| ------------------------------------------------------------------------------------------------------ | ---------- | ---------------------------------------------------------------- |
| `hardware/coral_nexus/rtl/vme/vme_region_lease.sv`                                                     | W3-03      | 每 Context lease FSM ＋ 实例表 ＋ 等待记录（N 维接口）           |
| `hardware/coral_nexus/rtl/vme/vme_policy_store.sv`                                                     | W3-01/03   | 每 program slot 转移表/链描述符/drain 白名单/等待事件白名单      |
| `hardware/coral_nexus/rtl/vme/vme_decl_front.sv`（+ 接入 RvvFrontEnd 的 patch 条目）                   | W3-01/02   | 每条指令携带 pending 声明的接受/消费前端改造                     |
| `hardware/coral_nexus/protocol/field_requests/w3.yaml`                                                 | W3-01      | 字段请求碎片（decl CSR 名称/宽/访问级；region 消息字段）         |
| `hardware/coral_nexus/protocol/w3_numeric_profile_v1.yaml`                                             | W3-01      | 精确数值 profile：模式/舍入/溢出策略/flags/mask/tail 谓词        |
| `hardware/coral_nexus/protocol/w3_vme_decl_abi_v1.md`＋`w3_vme_decl_abi_v1.json`                       | W3-01      | 声明消费合同：原子绑定/stall/redirect/cancel 状态机与 fault 清单 |
| `hardware/coral_nexus/protocol/w3_vme_contract_freeze.json`                                            | W3-01      | 上述 3 文件 sha256 + 上游 7 文件 hash 引用                       |
| `hardware/coral_nexus/kernels/numeric_unit/`（上游单模块 harness `tb_upstream_fp.sv` + `run_unit.sh`） | W3-01      | **上游**固定 pin RTL 的合同核对 harness（独立于 DUT）            |
| `hardware/coral_nexus/rtl/vme/{percontext,unit}/` 各 `tb_*.sv`/`*.py`                                  | W3-02/03   | 单元协议 driver                                                  |
| `hardware/coral_nexus/scripts/w3_check_claims.sh`                                                      | W3-01      | 跨文档行号/谓词一致性审计脚本（只读 grep 校验）                  |

共享/他包文件（W3 计划引用但**不改**）：`pipeline_validator/{config.py,tile_group.py,devices.py}`, `runtime/program_table.py`；W8 调优文件（`_HW_DEFAULTS` 等）只读、修改走 W8。

## 4. W3 内部任务 DAG 与共同 freeze 门

```
W0-01/02/03(基线上游与容器) ─▶ W3-01(合同草稿 + 上游证据) ──JF-GATE(W3-01+W1-01+W5-01 产物落盘即可，非互等)──▶ W3-02 / W3-03 / (W4-03 的 vtle/vtse 合同片 / W5-02 的转移表内容引用)
W3-01 ─▶ W3-04 数值/收窄/模式/取消实装 ─▶ W3-05 数值+区域恶性集成 ─▶ (下阶段 CS1 集成，不在本计划内)
```

共同 freeze 门（parent 2026-11-05 口径）：**-01 产物落盘（文件+hash 可见）即可视为其他包"可引用"，不要求另一个 -01 先 DONE**。冲突核查：W3-01 与 W1-01/W5-01 互查字段碎片后由 W1-01 单点合并。W3-02/03/04/05 均须"合同已冻结"作为硬前置，引用 `w3_vme_contract_freeze.json` 的 hash；W3-05 还须 W3-02/03/04 全 DONE。

## 5. W3-01：数值合同/声明/模式合同冻结（G1，硬前置：无；建议顺序：本包第一个）

**产物（全部待创建，本任务写）**：`protocol/w3_numeric_profile_v1.yaml`、`protocol/w3_vme_decl_abi_v1.md`＋`.json`、`protocol/w3_vme_contract_freeze.json`、`protocol/field_requests/w3.yaml`、`scripts/w3_check_claims.sh`、`kernels/numeric_unit/{tb_upstream_fp.sv,run_unit.sh,vectors/*.json,tb_report.md}`、`design/coral_nexus_tile/implementation/w3_numeric_contract.md`（合同主文本：完整逐步算法；本计划附录 B 为其骨架与机检谓词索引）。

**输出/接口被谁消费**：W7-02 oracle（只读合同文件，不读 RTL）；W5-02/03/04（转移表内容、声明 helper 写法、target 字段）；W1-01（field_requests 碎片合并进 cn_protocol_v1/csr_name_map）；W3-02/03/04（实现时的谓词与 fault 清单）。

**机械步骤（引用附录 B 的参数表/谓词/状态机，正文在合同文件中）**：

1. 按附录 B.2 的 S1–S7 步骤展开为 `w3_numeric_contract.md` §3 的 11 组逐步算法：每步标注 `(源file:lines)` 与所引源文件 hash（快照值见 §14）。
2. **实证冻结任务（非待定）**：
   a. `VLEN` 编译期取值与 TE/NUM_MT/SUBTILE_SIZE/NUM_BLK/PROCESS_DELAY/COMPRATIO/ZVT_LMUL/NUM_VRF/ROB_DEPTH/RS 深度枚举：从上游 Chisel target 构建产物 `gen/verilator/generated/` 拿头文件+顶层端口宏与（可获取时）elaborate 日志 grep；把 grep 输出存 `kernels/numeric_unit/derived_params.txt` 并写进 freeze json。**若无法拿到 Chisel 生成头（沙箱无 Bazel），回退** `rvv_backend_define.svh` 行证据 + 在 harness `tb_upstream_fp.sv` 里用 `localparam` 自检打印（`$display` 每个 VEC_LEN/GUARD_BITS/OVERFLOW_BITS/PROD_SIG_BITS/TREE_OUTPUT_SIG_WIDTH），把打印文本存档为第二证据；两证据不一致时以打印为准回写本合同（MAT-09：重跑全部对拍）。
   b. `common_cells@6aeee85` `lzc.sv` 空 input 行为：clone（pinned URL）→ 模拟输入 0 → 断言 `empty_o && cnt_o == WIDTH-1`；写入 evidence 行。
3. 用 §18 的 14 个 KAT 逐个喂 `tb_upstream_fp.sv`（上游 RTL，单模块 harness）与"合同算法参考实现"（contract_ref.py，与 tb 同目录，**由 W3-01 从合同伪码转写、非 W7 oracle**），三方对比，全部 bitwise 相等才冻结（退出条件之一）。
4. `scripts/w3_check_claims.sh`：只读 grep，核对 (i) 合同文件里所有 `(file:lines)` 引用的行在固定 pin 快照内仍真实存在（grep 字面字符串）；(ii) 附录 B.3 的 P1–P10 每条谓词在 yaml 中以 `predicate` id 出现且无歧义变量；(iii) mode 表 3 行合法编码与上游 `msetmtype`/`msetmtypei` 位模式一致（grep 常量）；(iv) 附录 B.1/B.6 表格每行 3 列；失败退出 1 并打印差异行；成功存 `out/.../w3_check_claims.txt`。
5. `field_requests/w3.yaml` fragment：逐条列 `decl csr` 需要的 (逻辑名 cn_vdecl、宽 32、访问级 RW、读写属性 w-only-per-context、绑定窗口说明)；同时列出 region/chain/grant 消息字段名与位宽需求数（数值宽度不自行指定硅编码，由 W1-01 `derive_widths.py` 产机器可读 ABI）。
6. 与 W1-01/W5-01 做 CSR 地址/位段**冲突核查**：grep 双方 fragment 中同一地址名；如冲突报 parent，不在本计划定数值。
7. 冻结 manifest（`w3_vme_contract_freeze.json`）：sha256(三个合同产物) + 上游数值本体文件（附录 B.1 表列出的 Zvt/fp/fpnew 文件集）的 URL+hash，取值见 §14 表；`freeze_check.py`（W1 提供的 runner）验证非 None、字段齐全、签名一致。

**成功准则（独立可验证）**：

- ✅ 正向：w3_check_claims.sh 退出 0；`tb_upstream_fp.sv` 14 KAT bitwise 全对且 `derived_params.txt` 的 grep 输出与 附录 B.1 表逐格相等；tb_report.md 有 PASS 行。
- ❌ 失败注入（先失败后过的负例）：把 `ZVT_MAXCOMPUTING` 打开后重跑上游 harness（COMPRATIO 1/4）、把 `lzc.sv` 换成本地修改版（返回 cnt_o=0 on empty）、把合同里 `msetmtype(16418)` 改成 16419——三者都必须让 tb 或 check_claims 报 FAIL。
- 退出条件：freeze json 被下游（W3-02+ / W5 / W7-02）引用的 hash 存在且 tb KAT 全绿。**W3-01 不等 W7-02 oracle 完成**（parent 指示）。

**T 映射**：MAT-09/VEC-04/PAR-06/PAR-08、CMP-19、T-29（数值主战场）、T-39/T-40（oracle 与 ISA 模型 profile hook；W3-01 只产合同）；T-29 mask 专项用附录 B.3 谓词。

**风险与取消/旧代处理**：数值合同一旦被修改 ⇒ MAT-09 全量重对拍（oracle 重跑、W7 KD 重判）；profile 升级路径：experimental（本任务产出）→ 实 RTL + oracle bitwise 通过后 supported（W7-04 后）。

## 6. W3-02：RVV 多 Context / VRF / ROB（G1 合同 → G2/G3 段实装）

**硬前置**：JF-GATE（W3-01 freeze hash 在）；W2-01/02 的接口冻结文档已产出（不要求 DONE，只要 interface spec 可引用）；`cs1_c1.yaml`/`cs1_c2.yaml` 存在（W2 的配置文件由 W0-04 产出；若未产出，W3-02 先用上游默认 1 Context 数值验证，c2 用法推迟不阻塞 W3-02 完成）。

**产物**：`rtl/vme/percontext/{tb_vme_pcctx.sv,tb_driver.py,pcctx_report.md}`、`rtl/vme/unit/tb_*`（ROB 分片/取消/多 Context 交错若干微 tb）、`hardware/coral_nexus/rtl/patch_series/0003-rvv-owner-context.md`（补丁序列条目，含每个上游文件被改的行号区间与意图，供 W0-01 应用）。

**机械步骤**：

1. 列上游全部共享可变状态清单（用本计划 §3.1 表 + 逐个 grep 补足未列出的 FIFO：`rvv_backend.sv` 各 multi_fifo 逐个读 depth/owner 现状），给每个条目标记 5 类处置之一（复刻/分片/共享+owner 换绑/全局 fail-stop 保留/删除）。这一步只是清单，**不改任何一行代码**。
2. 按清单逐条落到 patch_series 子条目：每条给精确 `file:lines`、新类型 field 插入位置（如 `ZVT_RS_t` 末尾加 owner_id 16b）、不改数值路径的守护断言。
3. `tb_vme_pcctx.sv`：driver 在同一 Verilator 实例中双 Context 轮换 inject（W2 已有 dispatch 侧驱动者；W3 侧只注入 UOP 层），验证：a) 相同 rd 地址双写不互写（VRF 隔离）；b) flags 互不合并（fflags 各自 delta 分离）；c) 一 Context 处于 ROB 分片等待另一 Context 继续发射（非 HOL）；d) 单 Context 取消不清另一 Context 队列（VEC-06）；e) trap flush 只清发起 Context 的 FIFO 而非全核（替代上游全局 flush）。
4. 成功/失败/取消/背压/旧代覆盖：每条 case 都给注入波形字段与期望 fflags 逐 bit 数。
5. **退出**：5 case 全对 + patch_series 条目评审通过（W2 owner review 即可，不互相等）。

**验证输入/期望**：输入=固定 `ZVT_RS_t`/`PU2ROB_t`/`RVFEXP_t` 类型；期望=逐 case 写明寄存器写事件序与 fflags 位图。

**消费方/T 映射**：W2-02 消费 patch_series；T-02/T-11/T-25/T-29/T-33/T-62 部分子项；VEC-01~06。

## 7. W3-03：区域 lease / 链声明 / N 维实例路由（G1 合同 → G3/G4 实装；D-29 下 N=1 交付）

**硬前置**：JF-GATE；W3-02 的 owner id/queue 归属接口已可见；W6-01 事件合同碎片字段（region_done 记录位宽、事件名编码）已可引用。**N=1 走全 RTL；N>1 只做接口与表声明**（D-29），在计划产出文档明确标注哪些条目仅 N=1 验证通过。

**产物**：`rtl/vme/vme_region_lease.sv`、`rtl/vme/unit/tb_*`（grant/park/wake/state_free 终态序）、`patch_series/…`条目、`protocol/field_requests/w3.yaml` 增补 region 记录字段。

**机械步骤**：

1. 定义**每 Context 一条等待记录** 数据结构：waiter=(ctx_id, region_id, requested_at_tick, head_inst_seq, mode_snapshot, generation)；写死不变量：每 Context 至多 1 条、不可共享；不占共享执行队列。
2. 实例表 `busy/owner/region/last_grant` 的 N 维接口 + N=1 退化实现（N 维字段宽度按 cn_protocol ABI 提供，不自行指定硅位宽）。state_free 的公平唤醒：`select_waiter()` 用固定轮转（确定性），同周期释放+申请时必须先释放后授予（防重复授予）；多 waiters 同周期到达：按到达序 FIFO，不饥饿。
3. START 请求与停靠时序（两事件分离，MAT-12/MAT-20，详见 §16）：**E1 请求/停靠**＝声明目标指令到达该 Context 发射头且 valid、尚未进入共享 RVV 队列时，W2 dispatch 侧向 `vme_region_lease` 发 START 请求——有空闲实例当拍授予；无空闲当拍登记该 Context 唯一等待记录并回 parked，**该指令停留在 Context 头部、不进入共享 RVV 队列**（不占队首、无 HOL，MAT-12）。**E2 消费**＝获授予后 W2 放行该头部指令，其被 RvvFrontEnd accepted 的那一拍才原子消费 pending 声明（附录 B.4 A2/§16）。停靠期间 pending 保持 ARMED：不早绑到声明写、不重绑到更后指令。取消/redirect 清 pending 并撤销等待记录；已在共享队列内的其他 Context 指令不受影响。
4. 链声明消费：FINAL/FIRST 提交进 vme_policy_store 的 chain table，绑定 chain_id 与 dst mt 组；单条链 FIRST|FINAL 时 dst mt 组内所有 tile 已 seed 才算 legal。
5. region_done：与 W6-02 的可靠记录接口（W6 是唯一 writer），W3 产出 region_done 信号＋region_id＋owner+generation，W3 侧只等待 W6 ack 才释放实例（不丢唤醒）。
6. owner 变更：只有可靠 region_done 后才允许 owner 改绑（MAT-12）；同周期释放与申请并存时，释放侧先标 released，再在下一拍从 waiter 里挑（公平序、不重复授予、不丢唤醒——写一条 `wake_coalesce` 谓词到合同）。
7. 单元 tb：每步骤给 3 组注入（正常、竞争、冲突）× 每组给期望 wait 记录位图。

**验证**：输入=tb_driver.py 的 grant/park/finalize 序列；期望=最终 region_done 终态位图 + 等待记录数不增不减。**退出**：全部 12 个注入用例过（列表见 `unit/tb_region_expectations.md`，含 N=1 显式标注）。

**消费方/T 映射**：T-62、T-68（N=1 部分）、MAT-12/17/18/19/20/23/24；W5-03（链声明消费）、W6-02（region_done 事件）、W7-03（checker）。

## 8. W3-04：BF16 执行/收窄/模式/取消（G3）

**硬前置**：JF-GATE；W3-02 完成的 owner 分片；W3-03 region lease 可触发；`cs1_c1.yaml` target 可装载（W0-04）。

**产物**：`rtl/vme/unit/{tb_bf16_exec.sv,tb_bf16_narrow.sv,tb_mode_fault.sv,tb_cancel.sv}`、`patch_series/…` 条目（pe_array/block 的 owner 标签注入与 vtfmm 合法性 gate）、`reports/w3_04_g0_matrix.md`（每个注入用例的 bits/flags 期望矩阵，作为后续 W7-03 checker 的比对表）。

**机械步骤**：

1. **vtfmm 合法性 gate**（pe_array 前端层，不动数值路径）：注入 `fsrc_fmt != FP16ALT` ⇒ 拒绝该指令进入 PE 队列并产生 `mode_fault` 事件（对应附录 B.1"M-BF16 模式仅 vtfmm.alt"行）。这直接禁止 fullFP32 matmul 模式（fpnew RNE 路径被排除在交付外）。**不改 zvt_pe_mulbulk_fp_lane.sv 本体**。
2. 收窄 RNE 收敛证据：新增 `tb_bf16_narrow.sv` 复用上游 cvfpu fpnew（cvfpu pin b65bde… src/fpnew_cast_multi.sv），从 `w3_numeric_contract.md` §5 的收窄 7 向量表（NC-01..07）逐一比较（RNE，仅 OF/NV/UF/NX 语义）；bit-exact（0 容差）；结果记录 `tb_bf16_narrow_kat_report.md`；**不做** RNE→非 RNE 的变体。
3. 模式 fault：tb_mode_fault 注入 SEW32/altfmt=1/mtwiden=0 三种非法组合，期望统一 `mode_fault`（MAT-10/CMP-19 合同表为唯一判定依据）。
4. 取消：tb_cancel 验证 (a) 正在跑的 matmul 的 owner cancel ⇒ 该 owner 的 mt tile 写事件停止、其他 Context RVV 继续跑；(b) flags delta 只属于被取消 Context 不泄漏；(c) 已完成的合法指令仍可见（不因取消被清）。
5. 每步给出失败注入的期望（必须先 FAIL 后 PASS 的负例：把 owner 剥离后重跑，期望隔离被破坏）。
6. 产出 g0 矩阵：每注入×(bits, flags of/nv, mode) 的期望值，作为后续 checker 输入。

**验证**：输入=固定 seed 向量表（`w3_numeric_contract.md` §6 的 14 KAT（NA-01..14） + tb 内 20 组注入）；期望=矩阵逐格比对。

**退出**：收窄 7 KAT bitwise 全过；mode fault 3 组全过；cancel 3 组全过；g0 矩阵落盘且后续 W7-03 checker 有据可引。

**T 映射**：T-29（数值/收窄/masked-Inf）、T-62（取消+配置切换）、MAT-09/11/16/24、VEC-04/05/06。

## 9. W3-05：数值与区域恶性集成（G3 收口）

**硬前置**：W3-02/03/04 全部门通过；`hardware/coral_nexus/sim/` 集成入口可用（W1-04 的 oneTile loop 已在验证，不需要 W1-05）。

**产物**：`hardware/coral_nexus/sim/scenarios/w3_malignant.yaml`（场景清单）、`out/.../w3_malignant_report.md`、`patch_series`（如集成需要的前端接线补丁）。

**机械步骤**：

1. 场景矩阵：把 §11 的逐 T 映射表逐项转为 YAML scenario：每条标 "期望终态（region_done/协议 fault/flags delta）"，并标注仅 N=1 适用的注入。
2. 跑场景（**此条在 parent 继续推进 RTL 后才可执行**；本轮只写场景定义与执行命令合同，不执行）：cn.sh cosim … --scenario w3_malignant（命令不存在，标注"未来命令"）。
3. 每 scenario 输出 trace 与 report，对照 g0 矩阵；任何 deviation 逐条列 KD（不静默）。
4. 失败注入必须先失败后过：对每个合法场景再跑一次"去掉某个声明字段"的变体，期望被拒。

**退出**：场景清单全部有期望+全部在 CS1 入口可执行（对 W3-05 的验收是"场景可运行"，不是数值通过率——数值由 W7 oracle 判）。

**T 映射**：T-62、T-68、T-31、T-32、T-06/T-05、T-28。

## 10. 给能力有限实现 agent 的分工作业手册

先读（顺序）：① 本计划 §1/§2 的已选方案与 owner 边界；② 组件设计 `../components/W3_RVV_VME.md` §2/§3/§4/§8.1–8.3；③ 合同 `w3_numeric_profile_v1.yaml` + `w3_vme_decl_abi_v1.md`（**不要反向从 zvt_pe_mulbulk_fp_lane.sv 直接推数值语义，那一步已在合同里完成且已经 tb_upstream_fp 验证过**；RTL 改动仅限 owner/fault/mode gate 层）。可做：按自己所属任务号（W3-02/03/04/05）在对应 unit/ 目录加 tb 和 patch*series 子条目。**不能**：改 `zvt_pe*\*\_fp_lane.sv`/`fp_rounding.sv`/`fp_align.sv`/`fp_mulfront.sv`（数值本体，除非任务号=4 且变更已在合同里落）；改 python validator；宣称 N>1 通过；宣称 supported（W7-04 后才可能）；把 fullFP32 matmul 引回（即便 fpnew 路径可运行也算 out-of-scope）。**合同缺失时**：先报 parent（附 W3-01 产出物的 file hash），不要自行补语义。

## 11. 逐 T 场景/需求族映射（适用范围）

| 需求族/场景                  | 落地任务                                       | 验证方式                                     |
| ---------------------------- | ---------------------------------------------- | -------------------------------------------- |
| MAT-01~09                    | W3-01/03/04                                    | 上游 harness（数值位）+ tb（协议位）         |
| MAT-10/24/13/20~23           | W3-01 模式表+region 表, W3-02/03 触发, W5 封印 | Loader 静态拒 + runtime fault 两种负例都要有 |
| MAT-11/12/14~17              | W3-03                                          | tb_region（N=1 退化）期望位图                |
| MAT-15/16/18~23（N 维）      | W3-03 接口                                     | N>1 仅合同/接口（D-29），不宣称通过          |
| VEC-01~06                    | W3-02                                          | tb_pcctx 5 case                              |
| T-05/06/28/31/32/T-29 主战场 | W3-04/05                                       | g0 矩阵 + 恶性场景                           |
| T-62/T-68                    | W3-03/05                                       | 对应 tb 与 scenario 两条独立证据             |
| D-21                         | W3-01/04（BF16 路径交付）                      | 收窄 7 KAT                                   |

不在本计划验证范围（只引用合同/接口，不宣称通过）：T-59/T-60（CS2/CS3）、T-68 N≥2 子项、性能 P1/PN 收益主张（W7-05 归，W3 只提供 per-context 计数器/停顿归因）。

## 12. 未来命令与产物登记（不执行）

- 未来命令：`bash hardware/coral_nexus/scripts/cn.sh unit --suite w3 --config <path> --out <path>`（W3-02/03/04/05 tb 的统一入口；W0-02 尚未创建，不声称可执行）；`cosim --scenario w3_malignant …`（W3-05 用）。
- 产物：`out/coral_nexus/<run_id>/{w3_check_claims.txt,w3_tb_report.md,w3_g0_matrix.md,w3_malignant_report.md}`。
- 当前可执行（本轮已做核对）：`python pipeline_validator/*.py` 只读证据引用、`grep` 源码行号、`curl http://raw.githubusercontent.com/google-coral/...`（已在本轮核对全部路径可公开读取）。

## 13. 发现的合同冲突与 parent 需整合事项

1. **fullFP32 matmul**：固定上游存在 SEW32 FPMUL 路径（`zvt_pe_array.sv:283-285`）；parent 裁决排除 fullFP32 为交付目标，本计划在 W3-04 步骤1 加 `fsrc_fmt != FP16ALT ⇒ mode_fault` gate 与静态 Loader/编译侧双向拒。已核对该 gate 不破坏 BF16 主路径（fsrcFmt 恒 FP16ALT under SEW16+ZVTBF16FMM_ON）。
2. **Spike/MPACT 溢出差异**： KD-01（W7-02 登记）：Spike f32_add_odd 溢出→Inf 与 coralnpu RTL 累加级 ROD→max-normal（`fp_rounding.sv:99-104`）不一致；parent 裁决"以已批准 W3 RTL 合同为准"。已写进 §1 与 W3-01 合同；W7 oracle 按 max-normal 建，并在 KD-01 记录参考差异。
3. **flags-only-OF/NV**：parent 补：VME 出口只走 OF/NV（`zvt_pe_block.sv:202-210`）；RVV RNE 收窄/其他 FP 指令的 flags（NV/OF plus NX/UF per IEEE）**不施加 NV/OF-only 限制**，由 cvfpu fpnew_cast_multi 自然产出（`fpnew_cast_multi.sv:757-768`）。已在 W3-04 步骤 2 与 §3.1 分开表述。
4. **CSR 地址**：W1-01 csr_name_map/csr_allocator 是唯一权威（含 `Csr.scala:45-100` 自定义窗口、`Decode.scala:1194` csrrwi 支持）；W3 只在 decl 前端消费 pending 声明值，**不分配地址**。已写进 §5/§7。
5. **START 请求/停靠与 pending 消费的两事件分离**：已在 W3-03 步骤 3 与附录 B.4 A2/§16 落实——请求/停靠（E1）发生在 target 指令头部 valid、进入共享 RVV 队列之前（W2 gate）；pending 消费（E2）只在获授予后该指令真正 accepted 时发生。**等待记录的登记不以 accepted 为前提**（若以 accepted 为前提：指令先入队则 parked 指令占住共享队列产生 HOL，违反 MAT-12；指令被 gate 则永远无法 accepted、等待记录永远无法建立，livelock——两种实现都被本合同排除）。与 parent 2026-11-05 裁决及 W2 计划 §7（START 声明指令位于头部即 park、不进共享向量队列）一致。若 W1-01 csr 编码窗口迫使位域布局变化，仅改 decl ABI json 的位域字段，不动谓词（MAT-09 不变）。
6. **多 Context 配置文件**：`cs1_c1.yaml`/`cs1_c2.yaml` 是 W2/W0 的待创建配置，W3 计划引用它们但**不创建**；若 W0-04 未产出导致 W3-02/04 无法装载 target，按 §6 前置条款退化到"上游 1 Context 数值验证先跑"。

## 附录 A. W3-01 输出文件的确切接口契约（给 W7/W5/W1 的可消费点）

- `w3_numeric_profile_v1.yaml`：字段 `profile_id: "vme_numeric_rode_v1"`；`profile_kind: "experimental"`；`status` 枚举 experimental|supported（受 MAT-09/W7-04 后才可能 supported）；`params`：VLEN(上游 derive)、TE、NUM_MT、SUBTILE_SIZE、COMPRATIO、PROCESS_DELAY、ZVT_LMUL、NUM_VRF、fsrc/fdst、guard_bits/overflow_bits/prod_sig_bits；`semantics`：逐步算法（引用 `w3_numeric_contract.md`）+ 14 KAT 的 hash；`acc_overflow_policy: "max_normal"`（KD-01 已登记）；`flags_scope: {vme: {OF,NV}, rvv_fp: {NV,OF,UF,NX}}`；`mask_predicates`: 附录 B.3 的 P1–P10 十条（机器可读，`status: experimental`；W7-02 oracle 与 W3-02/03 RTL 实现同以该谓词为唯一规范）。
- `w3_vme_decl_abi_v1.{md,json}`：CSR 逻辑名 `cn_vdecl`（w-only per-context，32-bit 值载体）；位域字段：`region_start`(1) / `chain_first`(1) / `chain_final`(1) / `mt_granule`(1) / `reserved`(28)（**位地址由 W1-01 冻结**）；消费状态机图 + stall/redirect/cancel 期望 + fault 清单（MAT-16 的 8 组）；原子绑定谓词 `bind_at_accepted`（附录 B.4 A1–A7）。
- `w3_vme_contract_freeze.json`：`sha256(profile.yaml)+sha256(decl_abi.md)+sha256(decl_abi.json)`、上游 7 项 hash、`msetmtype 模式表 3 行` 编码与 `derived_params.txt`。

## 附录 B. 精确数值合同主表（逐步，从固定源码逐行取；正文在 `w3_numeric_contract.md`）

### B.1 数值参数推导（固定来源：`fpnew_pkg.sv`、`rvv_backend_define.svh`、`zvt_pe_block.sv`、`zvt_pe_mulbulk.sv`）

| 参数                          | 取值（推导，未确认前不得直接当硅参数写 RTL）                                                                                                                                                                                                                               | 依据            |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `VLEN`                        | 待 W0 上游构建 derive（`rvv_backend_define.svh:6-14` 分支 128/256/512/1024，默认 128；`Parameters.scala:100-121` rvvVlen 选值）。**写计划时不冻结，由 `derived_params.txt` 冻结**。                                                                                        | 计划 §3.2       |
| `TE`                          | `VLEN/8`（`rvv_backend_define.svh:208`）                                                                                                                                                                                                                                   | derive          |
| `NUM_MT/SUBTILE_SIZE/NUM_BLK` | 16/16/4（`:203-206`）                                                                                                                                                                                                                                                      | derive          |
| `COMPRATIO/PROCESS_DELAY`     | 4/TE 与 TE/4（非 ZVT_MAXCOMPUTING 分支，`:214-216`）；ZVT_MAXCOMPUTING=1/4、4（`ZVT_MAXCOMPUTING` 分支未被上游 flags.bzl 开启）                                                                                                                                            | derive          |
| `ZVT_LMUL`                    | `WORD_WIDTH*TE/VLEN` = 4（TE=16、VLEN=128）（`:219`）                                                                                                                                                                                                                      | derive          |
| `ZVT_RS_DEPTH`                | `ZVT_LMUL*2` = 8（`:201`, `:114`）                                                                                                                                                                                                                                         | derive          |
| `ROB_DEPTH/ROB_TAG_WIDTH`     | 16/4（`:84/:134-138`）                                                                                                                                                                                                                                                     | derive          |
| `NUM_VRF`                     | 32（`:89`）                                                                                                                                                                                                                                                                | derive          |
| `fsrc/fdst/FP_FMT_CONFIG`     | FPMUL: FP16ALT→FP32；FP_FMT_CONFIG=5'b10001（FP32+FP16ALT）（`pe_block:108-115`, `pe_array:276-288`）                                                                                                                                                                      | 逐行            |
| `VEC_LEN`                     | `WIDTH/宽格式宽度`，**在 NUM_FP_FORMATS 枚举上循环取最大**，含被禁用的 FP8=8bit ⇒ `VEC_LEN = 32/8 = 4`（`pe_mulbulk_fp_lane:46-56`；`fpnew_pkg.sv:36-64`：NUM_FP_FORMATS=5，FP8='d3）。**注意：这不等于"每个 word 装 4 个 BF16"，它只是驱动 GUARD/OVERFLOW_BITS 的常量**。 | 推导+待构建证实 |
| `GUARD_BITS/OVERFLOW_BITS`    | `$clog2(VEC_LEN)` = 2/2（`pe_mulbulk_fp_lane:57-58`）                                                                                                                                                                                                                      | 推导            |
| `PROD_EXP_BITS/PROD_SIG_BITS` | SUPER_EXP_BITS+2=10 / SUPER_MAN_BITS+GUARD+1=26（`:65-66`）                                                                                                                                                                                                                | 推导            |
| `TREE_OUTPUT_SIG_WIDTH`       | PROD_SIG_BITS+OVERFLOW_BITS+2 = 30（`:318`）                                                                                                                                                                                                                               | 推导            |

> ⚠️ 本表参数均为"从固定源码读取后可推导"的常量；**冻结**动作 = 把 `derived_params.txt` 的 grep 输出与本表逐格相等，作为 W3-01 出口条件之一（§5 步骤 3）。

### B.2 逐步数值算法（W3-01 以此为唯一规范源，oracle 不读 RTL 推导）

逐步语义（每步给 (源file:lines)）：S1 mask 先于分类与 special（`pe_mulbulk_fp_lane:84-89`）；S2 Inf/NaN 分类（`fp_mulfront:64-99` + `Inf−Inf` 组合 `pe_mulbulk_fp_lane:224-229`）；S3 乘级 ROD（`pe_mulbulk_fp_lane:272`、`pe_block:90-91`）；S4 乘树 2-term 批归一化（`pe_mulbulk_fp_lane:318-490`：`fp_absaddsub(sum+neg/LZA, j=0: k=0)` 输入 `{lead,man|jam}`，`fp_align` 用 `align_minimum=align_trimmed=max(prod_exponent_norm[i])`，溢出行 →`vec_out_overflow`，`overflow_result=∞`）；S5 round to dst FP32（`fp_rounding:66-94` ROD 决策 `~guard&(round|sticky)`，OF 判定与 of_saturate 由 **dst==FP32 的 `round_of_to_max_norm=0` ⇒ 累加级走 `max_normal`、乘级走 `∞`**，`pe_mulbulk_fp_lane:561-564`）；S6 加法器（`pe_adder_fp_lane:42-71,130-144,149-151`：`fp_addfront`→`fp_align`，`do_subtract=0`，`OPFVV⇒rndMode=ROD(tag)`, `INTMUL/UINTMUL` 独立 lane）；S7 溢出特殊 mux（`pe_adder_fp_lane:154-166`：NaN→qNaN 0x7FC00000 + NV；Inf→±Inf no NV；`round_status.OF→max_normal`）。**完整位级步骤**（包括 `fp_align` 的 POSTFIX 分支与 `lzc` 空输入 cnt=WIDTH-1 语义）逐行写在 `w3_numeric_contract.md` §3 的 11 组步骤里，oracle/W5 按 yaml 只引用它，不再读 SV 文件。

### B.3 mask/tail/特殊值谓词（机检，对应 MAT-09/24 与组件设计 §6.2）

P1 `mask 先于 operands`：`&operands_masked == 0`（`pe_mulbulk_fp_lane:84-89`）；P2 masked lane 产生 zero product 且不进 special 路径（`152-231`）；P3 `tailTm/tailTn` 决定 per-PE `maskIn`（`pe_array:86-92`）；P4 K 尾：tk∈{1,2}（SEW16，`pe_array:93-101`），K=odd 经 TK=1 单步（KD-01 已登记 Spike 差异）；P5 **fullFP32 matmul 声明非法**（W3-04 gate + Loader 静态拒）；P6 `Inf×0/Inf−Inf` 只产 NaN 不产 Inf（`fp_mulfront:86-99`）；P7 sNaN 输入 NV=1、qNaN 输入 NV=0（`fp_mulfront:92-98`）；P8 `qNaN 0x7FC00000` 唯一 NaN 结果（`pe_mulbulk_fp_lane:569-571`、`pe_adder_fp_lane:156-158`）；P9 subnormal×subnormal 不上送 align_overflow（`fp_mulfront:219-221` 断言）；P10 `masked lane 配 Inf/NaN 对侧`：0×Inf→NaN/NV（`zvt_ctrl.sv:105-106` `!peBusy` 全局门 + `pe_array:86-92` 尾掩码先于乘级 special 分类）。

### B.4 声明消费状态机（W3-01 原子绑定合同核心；A1–A7 谓词）

状态（每 Context 一条 pending 记录）：EMPTY→ARMED(kind, chain_id, mt_granule)→(目标指令 accept 拍)→CONSUMED。谓词：

- A1 `csrw cn_vdecl`：仅当 pending==EMPTY 才接受，否则 protocol fault（MAT-16/21 嵌套 START）。
- A2 `bind_at_accepted`（两事件分离）：**E1 请求/停靠点**＝该 Context 的 declared-target 指令到达发射头且 valid、尚未进入共享 RVV 队列时（MAT-20：申请发生在任何 mt 副作用之前、且在该指令进入共享向量执行队列之前；由 W2 dispatch 侧发起，W3 在该点返回 grant 或 park，parked 指令保持在头部不占共享队列，MAT-12）；**E2 消费点**＝获得授予、指令被放行并被 RvvFrontEnd accepted 的那一拍，pending 声明原子消费（不早消费、不重绑）。
- A3 stall：pending 保持 ARMED；目标指令头部 stall 不清 pending。
- A4 redirect/取消/Task 退出：pending 清空，且若目标指令已 accepted 但尚未提交（MAT-14），按协议故障处理（T-68）；若 target 尚未 accepted，则无副作用。
- A5 `END`：显式无绑定指令（csrrwi 立即数，`Decode.scala:1194`），在最终输出操作提交后（W4 可见性判据，MAT-06/15）才触发 state_free 候选——END 本身**不释放**，MAT-15 顺序由 vme_region_lease 的终态编码保证（W3-03）。
- A6 FINAL 不释放、不自动 drain，FINAL 后仅合同 whitelist 指令（`vtmv.v.t`/`vfncvtbf16.f.f.w`+`vse16.v`，[C01] L125）。
- A7 计数/插桩：grant/park/wake/state_free/终态计数进 OBS-07/09/10（W3 → W7/W2）。

### B.5 声明合法/非法序列表（MAT-20/24 的 8 组 fault；F1–F8）

按 `RvvDecode.scala:169-333` 的 `accessesTile()`（读/写/清零/discard）与 `isMsetAny()` 谓词判别。非法情形枚举（F1–F8；判定方式 = 附录 B.4 A1–A7 状态机 + 合同 decl ABI fault 清单）：

- F1 无绑定而触碰 mt：`accessesTile()` 为真且无 region 绑定（MAT-20 运行时协议故障）。
- F2 已持有区域再次 START：ARMED 或 ACTIVE 时第二 `region_start` 声明（MAT-21/MAT-13 嵌套）。
- F3 END 之后再触碰 mt：END 声明被接受后任何 `accessesTile()` 指令（MAT-20）。
- F4 缺 END 的 Task 退出（MAT-16/MAT-20）。
- F5 redirect/intervening-control 落在声明与 target 之间：helper 写后、target accepted 前发生被接受的 taken-branch/jump/trap-return ⇒ pending 清空＋sticky 计数；其后的 target 类指令若为 mt 触碰 ⇒ 叠加 F1（无静默 rebind）。
- F6 目标类不匹配：声明 kind 与实际 target 类不一致（START|chain 落在非 vtfmm/非 seed 的 declared-target 类上）。
- F7 缺 FIRST 的 CONTINUE/FINAL；重复 FIRST；缺 FINAL 即 END；FINAL 之后再累加/写该链 mt（MAT-24，静态可识别者 Loader 拒，运行时协议故障）。
- F8 final 溢出/欠载/收窄/FRDYN：合法模式表（附录 B.1）之外的位编码，或收窄指令在 `frm∉{0}` 下发出。

静态可识别者在 Loader/构建拒绝（CMP-19），运行时判 MAT-16 protocol fault 并按 fail-stop 处理；二者任一触发即 stop-and-signal，不静默放过；W3-02/03/04 各自 tb 针对每组至少一条先-失败-后-通过的正反例。

## 14. 已核对的固定源码 hash（W3-01 冻结 manifest 的输入；本轮 2026-11-05 下载核对）

全部取自固定 pin URL（coralnpu `f795221`、cvfpu `bb65bde`、common_cells `6aeee85`），sha256 前 16 位：`zvt.sv=80832e80c4a87a89`、`zvt_ctrl.sv=59e01eaea151c403`、`zvt_mt.sv=93ae84e558745d4f`、`zvt_pe_array.sv=1cf4b5dbb290e0f3`、`zvt_pe_block.sv=833e9f056dc59de2`、`zvt_pe_mulbulk.sv=29a1d4550af7b4c2`、`zvt_pe_mulbulk_fp_lane.sv=7c659e59b6e49c3a`、`zvt_pe_adder.sv=bd78ed10879875d2`、`zvt_pe_adder_fp_lane.sv=365d70de6e071f4f`、`fp_rounding.sv=f9b5e4db3049eac6`、`fp_align.sv=68616a5dbb36138f`、`fp_mulfront.sv=f22b8de1c461c66c`、`fp_addfront.sv=a4fb868d6521145c`、`fp_absaddsub.sv=90e2cdaed5d797a4`、`fp_lza.sv=5eb3add8b01b05b4`、`fp_classifier.sv=5f7f6e3dbfcc379e`、`RvvFrontEnd.sv=a49ee6c2bcc718ca`、`rvv_backend.sv=c92c0af5c7a24870`、`rvv_backend_dispatch.sv=a2341fc4eb303003`、`rvv_backend_falu_unit.sv=11006be82893953a`、`rvv_backend_define.svh=e439b6d969cb509f`、`rvv_backend.svh=e9e24be87964a089`、`rvv_backend_rob.sv=9641e764d61042f5`、`rvv_backend_retire.sv=f076018b7154f4bd`、`rvv_backend_vrf.sv=a408880ba38551f8`、`RvvCore.scala=8c18c364a9a77cb8`、`RvvDecode.scala=14695577058214c5`、`RvvInterface.scala=bed89b7cf6c9a3ba`、`Csr.scala=fc7717a465fdc5a0`、`Decode.scala=fda071bba9ed411e`、`SCore.scala=9d926475fe99e5d5`、`Bru.scala=fbea2a624b1d3cef`、`fpnew_pkg.sv=2abe80eb1ef173f4`（cvfpu pin）、`fpnew_cast_multi.sv=abb8e0420facb0d1`、`fpnew_rounding.sv=67d0a15d9e418159`、`common_cells/src/lzc.sv=c47046cab67cd9e9`。

> cvfpu/common_cells 是带补丁的 vendored 依赖；hash 以上表 pin 为快照锚，实施时以 W0-01 patch 后工作树重算并写进 freeze manifest。

## 15. W3-01 冻结 manifest 内容清单（附录 A 的落盘明细）

`w3_vme_contract_freeze.json` 逐条登记（缺一即 seal 失败）：① `w3_numeric_profile_v1.yaml` sha256＋`status: experimental`＋profile_id；② `w3_vme_decl_abi_v1.{md,json}` sha256×2；③ 上游证据：本表（§14）全部 36 项 hash＋`VLEN`/`VEC_LEN`/`GUARD_BITS`/`PROD_SIG_BITS` 的 `derived_params.txt` 输出；④ KD-01（累加级溢出 max-normal、Spike→Inf 差异，不回写合同）。变更任一 → 新 hash 新 seal（不可原地改），触发 MAT-09 全量重对拍。

## 16. START 请求/停靠与 pending 消费的两事件定义（附录 B.4 A2 的补充定义；机可读进 decl ABI json）

两个事件必须分离，否则二选一的坏实现都会出现：指令先入队再 park ⇒ parked 指令占住共享 RVV 队列产生 HOL（违反 MAT-12"不占共享执行队列入口"与 VEC-06）；指令被 gate 到授予后才入队、而等待记录又以 accepted 为登记条件 ⇒ 等待记录永远无法建立，授予无从发生（livelock）。本合同显式排除两种写法：

- **E1 请求/停靠（request/park）**：declared-target 指令到达该 Context 发射头且 valid、**尚未进入共享 RVV 队列**（W2 dispatch 侧 gate 点）。W2 在此点向 `vme_region_lease` 发 START 请求：有空闲实例→当拍授予并放行；无空闲→当拍登记该 Context 唯一等待记录（waiter）并回 parked，指令保持在头部，不进入共享队列（MAT-12/20）。**等待记录的存在不以指令 accepted 为前提。**
- **E2 消费（consume）**：授予后 W2 放行头部指令；该指令在 P1 pre-edge 被 RvvFrontEnd 接受（`inst_ready_o[gi] && inst_valid_i[gi]`，进入 `valid_inst_q` 那拍，`RvvFrontEnd.sv:131-135`）的**那一拍**原子消费 pending 声明。这是唯一消费点。

属性（全部机检）：
U1 **不早绑**：`csrw cn_vdecl` 本身不是 mt 副作用、不触发 E1；E1 只由 target 指令头部 valid 触发。
U2 **不早消费**：pending 在 E2 之前不被任何单元读走；被 gate/scoreboard 挡住的指令 pending 保持 ARMED（不重绑）。
U3 **不晚绑定**：一旦 ARMED，声明只能被同一 Context 程序序中第一个走到 E2 的 target 类指令消费；stall/redirect/cancel 后按附录 B.5 处理，不重新挂给更后的指令。
U4 **stall 保持**：parked（等待记录已登记、指令在头部被 gate）或 RvvFrontEnd queue 满时 pending 不清；W2 不重发（重复 csrw＝ARMED 冲突 fault，A1）。
U5 **redirect/cancel 清**：分支/跳转解析、LSU flush、W6 fault/cancel 清 pending、撤销等待记录并使头部指令重新 eligible；target 已 accepted 但未提交的按协议故障（MAT-16）。task 代际（`ctx_generation`）同拍清空，不跨代保活。
U6 **授予唤醒**：可靠 region_done 后按公平序授予等待者并事件驱动唤醒 W2 头部（不逐 tick 轮询，对齐 L1 gate 原则与 MAT-12）；唤醒后指令走正常发射并到达 E2。

## 17. 逐 T 场景/需求族映射补充：mask/尾 K/收窄/输出/flags（本节即 W3-04/W7 的验收锚点）

| 主题               | 声明值（W3 合同唯一来源；W7 oracle 按"合同=真理"建而非照抄 Spike）                                                                                                                                               | 依据                                                         |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| mask 无效 lane     | masked lane 先置零 operands 再进分类；不参与数值/特殊值/`Inf×0`/`NaN 传播`/OV/NV 累加                                                                                                                            | 附录 B.3 P1–P2（`zvt_pe_mulbulk_fp_lane.sv:84-89/:152-231`） |
| 尾 K               | K-tail 按 tk∈{1,2}（SEW16），K=奇 数的尾步经 TK=1 单步（模板固定，非运行时决定）；尾 tk=0 由 pe_array 默认 mask 处理为"全 K 保留"（MAT-24 合法 K 尾处理）                                                        | 附录 B.3 P4（`zvt_pe_array.sv:93-101`）                      |
| 尾 Tm/Tn           | 尾行/尾列不产生 valid 输入，不进 special；尾外 mt undisturbed                                                                                                                                                    | 附录 B.3 P3（`zvt_pe_array.sv:78-92`）                       |
| 收窄（输出 RNE）   | cvfpu `F2F` `FP32→FP16ALT`：RND 按 `vfncvtbf16.f.f.w` 的 `frm`；NV/OF/UF/NX 从 `cvfpu fpnew` 自然出（IEEE 语义）；`qNaN 0x7FC0` 唯一 NaN 结果；`is_boxed_i='1'`（`rvv_backend_falu_unit.sv:560-606`）            | W3-04 步骤 2 独立证据                                        |
| VME 输出 flags     | 仅 OF/NV 出 VME（`zvt_pe_block.sv:202-210/:363-374`），其他 IEEE 位不出                                                                                                                                          | 附录 A profile `flags_scope`                                 |
| RVV FP 指令 flags  | NV/OF/UF/NX per IEEE，由 fpnew 自然产出；不施加 NV/OF-only 限制                                                                                                                                                  | parent 2026-11-05 补                                         |
| 累加级溢出         | RTZ/ROD→max-normal（`fp_rounding.sv:99-104`）；乘级 widen ROD→Inf（`zvt_pe_mulbulk_fp_lane.sv:561-564`）；Spike/MPACT F32AddOdd→Inf 差异为 KD-01 已登记偏差，不回写合同                                          | §13 项 2                                                     |
| frm                | 构建期＋Loader 拒绝 `frm=7(FRDYN)`；运行期 RvvFrontEnd 结算点仍检查（MAT-10/CMP-19）。frm=5 编码=fpnew ROD：**禁用**（运行时检查拒绝；`fpnew_pkg.sv` 的 RVE/RDD 借用 ROD=101 事件由 W3-04 单测验证；不声称无害） | 附录 B.3 P10；合同文件                                       |
| 输出 R（RNE bits） | 0 容差 bitwise 全等；无 tolerance 放宽；oracle 变更只能来自合同变更（MAT-09 重跑全部对拍）                                                                                                                       | 全局                                                         |
| seed 来源          | vtzero（+0.0，body mask 内）/ vtmv.t.v（VRF→tile 行或列）/ vtle（EEW32 tile load），每链 FIRST 前由封印 kernel 显式 seed；无隐式 0 初始化                                                                        | 附录 B.1 约束 ①                                              |

> mask/tail 的**可机检谓词**（W7-02 oracle 与 W3-02/03 RTL 实现同用）：P1 mask 先于 operand 进位（`&operands_masked == 0`）、P2 masked lane 不产 special、P3 尾 Tm/Tn 之外不产 valid、P4 K 尾 tk∈{1,2} 恰好一组保留、P5 masked Inf 对侧 0×Inf→NaN/NV（**不默认乘零无害**）、P6 masked lane 的 `fmt_product_enable` 置 0（不进加法树）、P7 尾 mask 半字节禁（`zvt_pe_block.sv:164-169` 断言）、P8 frm 非法集 = {5,6,7}（5 与 6 无 fpnew 合法语义；7=DYN）、P9 qNaN 输入传播不置 NV、P10 sNaN 输入置 NV（附录 B.3 P1–P10 的正式定义以 `w3_numeric_contract.md` §4 为准，本表是其计划层索引）。

## 18. 收窄 7 向量与 14 KAT（W3-04 验收锚点；正文逐向量在 `w3_numeric_contract.md` §5/§6）

收窄（`tb_bf16_narrow.sv` 消费）：NC-01 边界 tie-to-even、NC-02 tie-odd 校验、NC-03 sNaN 输入、NC-04 超下溢、NC-05 最小 subnormal、NC-06 最大 normal→±Inf+OF、NC-07 RNE/ROD/RTZ 三模对比（只 RNE 交付，其余仅负例）。
KAT（`tb_upstream_fp.sv` + contract_ref.py 三方 bitwise）：NA-01 masked-Inf 隔离（va=a\*，Inf；vb=NaN 非 mask，逐元素无 NaN）、NA-02 尾列 masked-Inf、NA-03 乘级溢出 max×0x3F80+→+Inf、NA-04 累加级溢出 0x3F00+2^16×2^30→max-normal、NA-05 大指数差相消 jam bit（非数学 exact 诊断）、NA-06 混 subnormal/normal 批归一化、NA-07 尾 K tk=1（无 NV）、NA-08 尾 Tm/Tn 15×15（尾元素不写不产 flags）、NA-09 链跨 mt{2k}×4 连续 tile、NA-10 多基址 mt0/4/8/12 并行、NA-11 收窄向量集（NC-01..07 的 KAT 子集）、NA-12 FRDYN 静态拒、NA-13 Inf−Inf→qNaN+NV、NA-14 sNaN 传播（qNaN+NV=1）。

> 每个 KAT 的输入 bits/期望 bits/期望 flags/源码行依据以 `w3_numeric_contract.md` §5/§6 最终表为准；本计划列的是锚点与验收方式，向量本体不在这里复写，避免三处漂移。

## 19. 依赖表（parent 要求；硬前置 = 必须先于本任务完成的任务，共同 freeze 产物不要求另一 -01 先 DONE）

| 任务  | 硬前置                                                                                                                                     | 说明                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| W3-01 | 无任务级硬前置（读上游固定 pin URL 即可）；W0-01/02 的基线构建仅是步骤 2 实证冻结的输入，允许在 W3-01 自身产物落盘后补入 freeze            | ① `w3_numeric_profile_v1.yaml`（experimental，附 schema+hash）→ W7-02 oracle、W5-03/05-04 experimental 封印直接消费；② `w3_vme_decl_abi_v1.{md,json}` → W3-02/03、W5-01 helper、W7-01；③ `field_requests/w3.yaml` → W1-01 单点合并；④ `w3_vme_contract_freeze.json` → joint seal（`PROTOCOL_FREEZE.json`，不可追加，新内容新 hash）。W3-01 不等 W7-02 oracle、不等 W3-04/05 硬件跑通（W7-04 后才判 supported） |
| W3-02 | JF-GATE（W3-01 产物 hash 可引用）；W2-01/02 接口可见（不要求 DONE）；W4-03 访存端点                                                        | 与 W2 标量侧并行，顶层合并由 W2 owner 执行；RVV patch 自证（不依赖 W2-05 闭环）                                                                                                                                                                                                                                                                                                                                |
| W3-03 | JF-GATE；W3-02 完成（owner id/queue 归属）；W6-01 region_done 记录字段（**读即可**，事件名/位宽以 PlanW6 计划落盘为准，不等待 W6-01 DONE） | N=1 交付；N>1 仅接口                                                                                                                                                                                                                                                                                                                                                                                           |
| W3-04 | W3-01（experimental profile 足够；不设 W7-02 oracle 前置）＋W3-02                                                                          | W5-03 kernel 库只依赖 W3-01；真实封印在 oracle bitwise 通过后由 W7-04 评估 supported                                                                                                                                                                                                                                                                                                                           |
| W3-05 | W3-02+03+04 全 DONE；W1-04 的 oneTile loop/`sim/cabi` 端点已可用（**不依赖 W1-05**，防集成循环）                                           | 交付可运行恶性场景集+trace 对拍锚点；数值/区域 verdict 由 W7-03/04 checker 引用                                                                                                                                                                                                                                                                                                                                |

> 本表是 parent `../plan.md` §5 「W3」行的细化；W3-05 不作为 W2-05/W5-05/W6-05 的前置。

## 20. 附录 B.1′ 模式/转移表（MAT-10 硬件可检查表示；**模板级参考实例，非全局白名单**）

**权威结构**：模式/转移表是**每个受控 kernel 模板随封印 manifest 声明**的数据（W5 binding manifest），W3 运行时检查（`RvvFrontEnd` 结算点）按**当前驻留 kernel 的封印表**校验，不是本计划写死的一份全局清单。下表只是**上游参考模板（[C01] 矩阵 codegen 家族，TM=16、TN∈{8,12,16}、TK∈{1,2}）的首个已实证实例**，登记进 `w3_numeric_profile_v1.yaml` 作为该模板的 `modes:`/`transitions:`（W3-01 freeze）；其他 M/N 尾块模板（mask/packing、部分 tile、不同 vtype 组合）由 W5 配套**独立受控模板＋各自转移表**封印，W3-05 验收含至少一个此类尾块模板的**正例**（loader 接受、运行时按其表通过、数值 bitwise 对拍）。**早期草稿中 FP32-MAC 写 16386 属笔误（16384\|2＝w2/tk0 非法组合），正确编码为 16417。**

上游参考模板的模式行（4 行，前 3 行该模板启用）：

| 模式                   | msetmtype(mtype, vtype)     | mtwiden / vtype 含义      | 状态                                     |
| ---------------------- | --------------------------- | ------------------------- | ---------------------------------------- |
| FP32-MAC（seed/drain） | `(16417, 18)`               | w=1，SEW32/LMUL4          | 启用（仅 seed/drain/收窄交替）           |
| BF16-MAC TK=1          | `(16418, 265)`              | w=2，SEW16/LMUL2/altfmt=1 | 启用（张量计算主模式）                   |
| BF16-MAC TK=2          | `(16450, 265)`              | w=2，SEW16/LMUL2/altfmt=1 | 启用（K 偶数对模式）                     |
| INT8                   | `(16419,256)`/`(16515,256)` | w=3                       | 登记存在、本版张量通路禁用（命中即拒绝） |

该模板的合法转移：`∅→FP32-MAC`（seed）、`FP32-MAC→BF16-MAC`（运算装载前）、`BF16-MAC→FP32-MAC`（drain/收窄交替）、`FP32-MAC→FP32-MAC`（TN 重编程 16↔8/12）、`模式→CLEARED→模式`（**块间 `vset*` 清 mtype/altfmt、由下一 asm 块的 `mset*` 前导恢复——上游 [C01] 明示 LLVM 会在块间插 `vset*`，属已建模合法路径，不是拒绝事件**）。CLEARED 不是独立计算模式：mt 使用时配置必须处于当前封印表的声明模式内。

**非法判定按表驱动，不按本清单硬编码**：asm 块**内**出现 `vset*`（所有模板通用）；mt 使用时配置不在**当前封印 kernel 的表**内（前导恢复缺失）；命中国际禁用行（INT8）。参考模板之外、由后续封印模板合法声明的 mtype/vtype 组合**不是非法**——只要其表经过构建期静态检查（字段上下界、与 `Csr.scala`/`mset*` 位宽一致）并以实测证据封印。`fullFP32` **计算**（vtfmm fsrc_fmt=FP32）由 W3-04 步骤 1 的 fsrc_fmt gate 拒绝——该指令级排除与模式表正交，不禁止 FP32-MAC 模式用于 seed/drain（累加器恒 32 位）。

拒绝条件（与 W5 设计 §4.2.1 逐字一致）：asm 块**内**出现 `vset*`；mt 使用时配置不在声明模式内（前导恢复缺失）；命中 INT8 行。`fullFP32` **计算**（vtfmm fsrc_fmt=FP32）由 W3-04 步骤 1 的 fsrc_fmt gate 拒绝——该指令级排除不禁止 FP32-MAC 模式用于 seed/drain（累加器恒 32 位），两者勿混。

## 21. 本计划未声称的事项

- 本轮未运行任何 tb／pytest／RTL 仿真／build；所引行号与 hash 都是从固定上游 pin URL 下载后核对（§14），不是本仓库 RTL 现状。
- `ZVT_MAXCOMPUTING` 分支（COMPRATIO 1/4）从未在上游 flags.bzl 打开——本计划显式排除（W3-01 步骤 2 负例再验）。
- 硬件/镜像路径是待创建目标；计划里未来命令**当前不存在**，不声称可执行。
- oracle 由 W7-02 按 W3-01 冻结合同实现；W3 本轮只产合同+锚点，不产 oracle 代码；W7-04 后才可能 supported。
