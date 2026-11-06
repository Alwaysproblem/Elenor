# W5 内核映像与运行时（Kernel Image Runtime）实施计划（W5-01..W5-05）

> **逐 commit 执行入口**：[execution_guide.md](../execution_guide.md)。先用该入口选择前置已满足的卡片；本页保留工作包细节，commit 顺序、工具可用时点、文件边界和原任务关闭条件以执行 guide 为准。架构语义仍遵守已批准设计，不按旧草稿中的占位命令或文字存在性检查宣称完成。

> 状态：实施计划（本轮只写计划；不执行 build/test/formatter，不创建任何目标代码）。所有 `hardware/coral_nexus/**` 与 `pipeline_validator/package|runtime` 新文件均为待创建目标。父计划与裁决见 [../plan.md](../plan.md)（§2 已选方案、§3 owner 与产物命名）；组件设计见 [../components/W5_Kernel_Image_Runtime.md](../components/W5_Kernel_Image_Runtime.md)；需求见 r5 §11.6/§13/§14。上游固定 pin：coralnpu `f795221`、coralnpu-compiler `d715556`。

## 1. 已选方案摘要、非目标与 owner

1. **CoralTilePackage 独立版本域 `cn_package/v1`**：外层结构 `{package_schema:"cn_package/v1", control_plane: CompiledProgram(schema3/ABI v3, 原样包裹), image:{bytes,sha256,entry,link_layout}, binding_manifest, resource_manifest, target_fingerprint, numeric_profile_id, patch_mask}`。现 `pipeline_validator/package/package_ir.py` 的 `ElenorPackage/PackageHeader(package_abi_version=1)` 是 mutable 旧设计对象、无真实封印链（X-17）：**不继承其版本号、不改造其状态机**；新链独立新增，Python-only 旧产物保持原样。
2. **多固定预链接 slot**：每映像构建期固定装载地址写入 manifest；至少两个链接地址不重叠的真实映像并存（T-58 冷装新映像期间其他 Context 驻留程序继续运行）；Loader 不重定位、不自动淘汰；引用归零＋显式失效收齐 ACK 后才释放 slot。slot 数与每 slot 容量**由 W5-02 实测 ELF/资源上界冻结**（本计划不写占位值）。
3. **kernel 族不缩水**（每族真实 C/asm kernel＋显式 binding manifest，禁止只做 matmul smoke）：matmul（K 分块累加＋`accumulate` 语义）、gather、scatter、softmax 系（含 online softmax）、norm、reduce_sum、pow、add、recurrence（KV-cache 累加/状态机类）。P0 workload registry（`kernels/p0/` 归 W0-03）与 W5 kernel 库共享源，由 `p0_workloads.yaml` 做接口。
4. **Host 上传真实字节链**：`cn.sh package`（或 cosim 前置）把真实 image bytes 写入 Python 侧 HBM program source 区并登记（范围＋hash＋长度）；PROGRAM_INSTALL→READ_REQ→READ_RESP 走真实字节；INSTALL 校验（hash/长度）失败＝fault（VB-11）。Host 上传时间单列报告，不并入 Tile cold cycles（RES-16 口径）；禁止 testbench 直填 SRAM 绕过安装端口。
5. **helper ABI**：受控 C/asm helper＋Context-local CSR（逻辑名 `cn_vdecl` 等，与 W3-01 decl ABI 一致；物理地址由 W1-01 `csr_name_map/csr_allocator` 分配，本包只消费生成 helper，禁手写数值）。START/链位置 pending 只在目标指令 accepted 时消费；END 显式（csrrwi 立即数式）。
6. **numeric profile** 引用 W3-01 `w3_numeric_profile_v1.yaml`（`status: experimental` 即可封印首轮测试；`supported` 由 W7-04 实跑＋bitwise oracle 后判）。BF16 输入/输出（RNE 收窄）、内部 FP32 ROD 累加；`frm=DYN` 构建期拒绝。
7. **新 target 指纹字段**：`tensor_input_bytes=2 / accumulator_bytes=4 / tensor_output_bytes=2`、`rvv_vlen`、`vme_te` 等（映射由 W5-01→W0-04 落进 cs1 配置）；旧 Python `boa_*` 时序字段保留原语义、**不作 VME 别名**（validator 合同变更须升版）。

**非目标**：完整 RVV 程序 dialect、通用 LLVM/IREE 后端（D-27 后评估）、INT8/对外 FP32 模式、自动淘汰/替换策略（RES-14 只留接口）、CS2/CS3、FPGA。

### 1.1 唯一 owner 表

| 资产                                                                                            | owner                    | 边界                                         |
| ----------------------------------------------------------------------------------------------- | ------------------------ | -------------------------------------------- |
| `pipeline_validator/package/{binding_ir.py,seal.py}`（新）、`package_ir.py` 扩展点              | W5                       | 消费 CompiledProgram 编解码；不改 schema3 域 |
| `pipeline_validator/runtime/program_table.py` 演进（pending install/引用计数/累计容量/多 slot） | W5                       | 共享接线（tile_group 调用点消息化）归 W1-04  |
| `hardware/coral_nexus/kernels/`（除 `kernels/p0/` 归 W0-03 外全部）                             | W5                       | binding/build/链接脚本/资源清单              |
| Tile 侧安装服务 RTL 逻辑（安装 FSM/元数据）                                                     | W5                       | 物理阵列/端口仲裁归 W4 `TileProgramMemory`   |
| helper 逻辑名与消费语义                                                                         | W5（声明 ABI 归 W3-01）  | CSR 地址分配归 W1-01                         |
| build/package/cosim 生成链                                                                      | W5-02（`cn.sh package`） | cn.sh 框架/退出码归 W0-02                    |

## 2. 当前文件证据与待创建清单

### 2.1 已存在证据（HEAD 814bdf9）

| 事实                                                                                                                                                | 位置                                                                                                                  | 用途                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `package_ir.py`：`PackageState`/`PackageHeader(package_abi_version=1)`/`ElenorPackage` mutable 对象、无封印/加载执行链                              | `pipeline_validator/package/package_ir.py:15-139`                                                                     | 明确"不继承"依据（X-17）                    |
| 驻留为纯解析模型：`ensure_resident` 同步置 TILE_RESIDENT、无 bytes/无 pending/无引用计数；`FETCHING` 从未置位、`evict()` 无 caller、`hbm_iova` 恒 0 | `runtime/program_table.py:17-21,79-111,130`                                                                           | W5-04 演进基线                              |
| 提交期驻留判定＋冷装载计费                                                                                                                          | `tile_group.py:1284-1350`（commit→pin→program 登记→ensure_resident→load_program；`program_cold_load` PMU 直加 L1350） | W1-04 消息化切分点                          |
| 计算 descriptor 无 A/B/C view（无 operand）                                                                                                         | `compiler/lowering.py:799-855`                                                                                        | "禁止猜 operand"依据（X-19）                |
| 封印/解析 hash＋白名单机制                                                                                                                          | `compiled_program.py:1027-1100`（`artifact_digest/seal_program/parse_compiled_program`）                              | 外层 package 编解码参考实现                 |
| 上游 CSR 已占地址（MCONTEXT0-7=0x7c0-7c7、MPC/MSP=0x7e0/7e1、MTYPE=0xc23 等）                                                                       | 上游 `Csr.scala:45-100`（W1-01 提取脚本权威）                                                                         | helper CSR 地址冲突核查输入                 |
| coralnpu_v2_binary 产出真实 ELF；ITCM/DTCM 默认 8k/32k、stack 128B                                                                                  | 上游 `rules/coralnpu_v2.bzl:20-36,112,255`、`rules/linker.bzl:26-33`                                                  | 链接脚本/资源上界起点                       |
| 上游 mset 编码 `mtype = 16384\|(tk<<5)\|w`；FP32-MAC(16417,18)/BF16-MAC(16418,265 单 K、16450,265 TK=2)；[C01] L9-14/L102/L125                      | coralnpu-compiler `CoralNPUMatrixCodegen.cpp`（W3/W5 设计已逐值核对）                                                 | 转移表与 kernel 模板来源                    |
| 集成指南：S_AXI 写 ITCM→start PC→时钟门控→复位（FPGA 每次 Run 重写）                                                                                | [S08]                                                                                                                 | "不得出现在 Nexus 正常 Task 启动路径"的反证 |

### 2.2 待创建清单（owner 标注）

| 路径                                                                                                                                    | 任务     | 说明                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------------------------- |
| `hardware/coral_nexus/protocol/w5_package.contract.yaml`＋`field_requests/w5.yaml`                                                      | W5-01    | package/binding/资源 manifest 字段 fragment（逻辑名；交 W1-01 合并） |
| `pipeline_validator/package/binding_ir.py`、`seal.py`                                                                                   | W5-01    | KernelBindingManifest 编解码、CoralTilePackage 封印/重验             |
| `hardware/coral_nexus/kernels/library/{matmul,gather,scatter,softmax,norm,reduce,pow,add,recurrence}/`（`.S/.c`+binding yaml+链接脚本） | W5-02/03 | 每族真实 kernel＋显式 binding                                        |
| `hardware/coral_nexus/kernels/bindings/cs1_reference.yaml`                                                                              | W5-03    | 父计划引用的 binding 集合入口                                        |
| `hardware/coral_nexus/rtl/w5/install_service.sv`＋tb                                                                                    | W5-04    | Tile 安装服务（四步 pull；物理端口经 W4）                            |
| `pipeline_validator/runtime/program_table.py` 演进＋`pipeline_validator/tests/test_w5_*.py`                                             | W5-04    | pending/引用/累计容量/多 slot＋单元                                  |
| `scripts/handlers/package.sh`                                                                                                           | W5-02    | `cn.sh package` handler                                              |

## 3. 任务依赖表

| 任务  | 硬前置                                                                       | 并行窗口                                     |
| ----- | ---------------------------------------------------------------------------- | -------------------------------------------- |
| W5-01 | 无任务级（读 W3-01 decl ABI 草稿与 W1-01 字段表，hash 引用即可）             | 与全部 -01 并行；JF 后被 W5-02..05 消费      |
| W5-02 | W5-01＋W0-01/02（容器＋RV32 工具链）                                         | 与 W2-02/W3-02 并行                          |
| W5-03 | W5-02＋W3-01 numeric profile hash（experimental 即可）                       | 与 W3-04 并行                                |
| W5-04 | W5-01＋W4-02（TileProgramMemory 物理端口合同）＋W1-03/04（PROGRAM 通道端点） | 与 W3-03 并行；oneTile 验证不依赖 W1-05 DONE |
| W5-05 | W5-02/03/04＋W2-05/W3-05/W4-05/W6-02 端点就绪（oneTile）                     | 最终链路；4 Tile 汇合归 W1-05                |

## 4. 逐任务

### 4.1 W5-01 binding/package/helper 合同（G1）

- **输入**：§2.1 证据、W3-01 `w3_vme_decl_abi_v1.{md,json}` 草稿、W1-01 `csr_name_map` 生成物（或其字段表草稿）、r5 CMP-14/15/18/19、RES-01..19。
- **产物**：`w5_package.contract.yaml`（package 外层结构、KernelBindingManifest 全字段：kernel id/version、ins/outs/acc view、shape/dtype/布局、alias 规则、初值、scratch/stack/spill 上界、clobber、事件与完成声明、区域/链声明引用、转移表引用、patch_mask、资源 manifest、target 字段）；`binding_ir.py`（编解码＋严格校验：未知字段/重复键/非有限数拒绝）；`seal.py`（sha256 覆盖 image+binding+resource+target+patch_mask；重放加载结果一致）；`field_requests/w5.yaml`。
- **步骤**：① 定义 `cn_package/v1` schema（版本域独立，登记进 PROTOCOL_FREEZE）；② binding manifest 逐字段（每个 abstract 计算节点→actual view/shape/dtype/事件显式映射，缺＝构建拒绝，不猜 operand）；③ helper 语义引用 W3-01 decl ABI（只写消费侧：helper 名→CSR 逻辑名→汇编封装生成）；④ 封印/重验（hash 不符/篡改合同层即使重算 hash 也语义拒绝）；⑤ 冲突核查交 W1-01 csr_lint。
- **验证**：正例＝round-trip 编解码逐字段相等＋封印后重载一致；负例（先失败后通过）＝缺 `ins/outs` 字段、alias 非法、重复键、未知字段、非有限数、target 不符→全部构建/加载拒绝且错误信息指名字段。退出＝fragment 被 JF 收录＋负例全检出。
- **T**：CMP-14/18/19、T-61/T-63（构建期部分）。

### 4.2 W5-02 基础 kernel 构建链与 slot 冻结（G2）

- **输入**：W5-01＋上游 `coralnpu_v2_binary`/链接脚本（§2.1）；RV32 工具链（W0-01 锁）。
- **产物**：`kernels/library/` 首批（matmul 单步/K 分块＋accumulate；gather/scatter 索引搬运；add/pow 标量-向量控制流最小集）＋每 kernel 的 binding yaml＋两个不同链接地址的链接脚本；`scripts/handlers/package.sh`；**实测报告 `slot_freeze.json`**（每 kernel 的 text/静态数据/stack/scratch/spill 实测上界→两映像并存所需 slot 数与容量）。
- **步骤**：① 写 kernel（asm/C，`csrrwi` 式 END helper、`vtzero/vtmv/vtle` seed、`vtmv.v.t+vfncvtbf16` drain 与 W3-01 白名单一致）；② 编译→链接到两个不同固定地址→`readelf`/map 提取实测大小；③ 静态分析 stack/scratch/spill 上界（受控模板调用图；动态 high-water 只作旁证）；④ 逐 bank Arena/指令存储累计复核→冻结 slot 参数进 manifest；⑤ 封印两个 package。
- **验证**：正例＝两个 package 在同一 manifest 下并存且地址不重叠、封印/加载/重载 hash 一致；负例＝slot 容量越限（构造超限 kernel）→构建拒绝；地址重叠→链接期拒绝；同 (program_id,version) 不同 manifest_hash→登记拒绝。退出＝slot_freeze.json 落盘且被 W5-04/W0-04 消费。
- **T**：CMP-05/06/20、RES-05/14、T-30/T-58（构建期）。

### 4.3 W5-03 计算库全族与 BF16 映射（G3）

- **输入**：W5-02 链路＋W3-01 numeric profile（experimental hash）＋softmax/norm/reduce/recurrence 语料（`examples/workloads/transformer_*`、`reduce_sum_*` 等 dtype 核对）。
- **产物**：全族 kernel（§1.3 清单）＋binding＋`bindings/cs1_reference.yaml`（集合入口）；BF16 输出 dtype 的派生语料 binding（不改旧 FP32 fixture，splitk 类单列）。
- **步骤**：① 逐族按 r5 §9.3 参考负载最低要求写 kernel（softmax 的 online 变体、norm、reduce_sum、recurrence 的 KV 累加状态机）；② BF16 输入/输出 binding（输出收窄 RNE 在 kernel 尾部显式 `vfncvtbf16.f.f.w`）；③ 每族至少 2 个形状变体（含 K 尾/mask 边界）；④ 封印＋资源复核；⑤ 数值判据交 W7-02（本任务只保证"可构建、可封印、binding 完整"）。
- **验证**：正例＝每族封印 package 可加载、binding manifest 完整（W5-01 校验全过）；负例＝族缺失（人为删 softmax）→`cs1_reference.yaml` 校验 FAIL（禁止静默缩水）；输出 dtype 写 f32 冒充 bf16→构建拒绝。退出＝全族在 manifest 且两变体齐。
- **T**：CMP-19/21、T-29/T-32/T-61（kernel 侧）、T-16（gather kernel）。

### 4.4 W5-04 程序安装/驻留/多 slot 执行（G4）

- **输入**：W5-01/02＋W4-02 物理端口合同＋W1-03/04 PROGRAM 通道端点＋现 `program_table.py` 基线。
- **产物**：`rtl/w5/install_service.sv`＋tb（四步 pull：INSTALL 头→READ_REQ→READ_RESP→INSTALL_ACK(实测 hash/长度+本地句柄)，有界 FIFO/分段背压，校验失败即 fault 无半驻留）；`program_table.py` 演进（pending install 记录、引用计数全局一份、累计容量账本、多 slot 位图、epoch）；`test_w5_residency.py`。
- **步骤**：① Group 驻留管理器状态机（ABSENT→PENDING_INSTALL→RESIDENT；invalidating 串行点后无新热命中）；② Tile 安装服务 RTL（消费 W4 端口；不占数据 Arena/MSHR/credit，RES-17/18）；③ Host 上传链：package 命令把 image bytes 写入 Python HBM source 区＋登记（§1.4）；④ 冷→热→身份变更→单 Tile 失效→group 复位逐态；⑤ warm＝零 Tile 外部 text 字节（边界计数器；local fetch 与 descriptor patch 合法）。
- **验证**：正例（oneTile，未来命令 `cn.sh cosim --scenario hardware/coral_nexus/verification/scenarios/t58_cold_warm.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/w5`）＝冷装 A 运行中冷装 B（不同 slot）无全 Tile stall；安装中提交同映像→只加引用共享完成点；热命中→READ_REQ 为 0。负例＝warm 改写 resident text→拒；patch 越出 patch_mask→invalid descriptor fault；INVALIDATING 中提交→不命中；累计超容量→有界等待或拒绝（无永久等待）。退出＝全部正负例过＋`assert_l2_closed`。
- **T**：RES-01..19 主区、T-49/T-58/T-66、CMP-18、T-30。

### 4.5 W5-05 编译→封印→加载→执行全链（G4 收口）

- **输入**：W5-02/03/04 全部＋W2-05/W3-05/W4-05/W6-02 端点（oneTile；不依赖 W1-05 DONE）。
- **产物**：端到端场景（每族 kernel 各 1 条 oneTile 场景＋多程序并存场景）＋manifest/报告；`test_w5_e2e.py`。
- **步骤**：① `package`→`cosim` 串联（真实上传→安装→accepted→执行→retire→TASK_COMPLETION）；② 同映像双 Context 重入（可写区不相交、引用守恒）；③ 失效/引用交错（进入 INVALIDATING 后新 COMMIT 不获热命中）；④ 全部负例复跑（W5-01..04 的构建期拒绝在链路中复现）。
- **验证**：正例＝全族 oneTile 执行完成且资源零泄漏；负例＝删除任一 manifest 必需字段→加载前拒绝（非运行期 trap）。退出＝mandatory 子项 PASS＋`cs1_manifest` 段登记。**不以"封印成功"替代执行。**
- **T**：T-61/T-63/T-30/T-31（执行侧）、T-66、VER-15 子项。

## 5. T 映射（W5 主责）

| T                   | 任务     | 说明                                         |
| ------------------- | -------- | -------------------------------------------- |
| T-30/T-58/T-66/T-49 | W5-04    | 同映像重入/冷热/并存/装载与 gate 共存        |
| T-61/T-63/T-32      | W5-03/05 | 显式 binding/真实映像封印/报告与实际映像一致 |
| T-31/T-33/T-53      | W5-05/01 | 并发 runtime/拒绝边界/loader 级              |
| T-56                | W5-03    | 直线受控循环；非法静态拒绝                   |
| T-59/T-60           | —        | CS2/CS3（不跑，登记）                        |

## 6. 风险与上报

1. compiler 自带 Coral pin（782675ac）≠ 硬件 pin（f795221）：两 pin 保留；kernel 机器码兼容性实测前相关组合标 experimental（W0 R2，已裁决）。
2. slot 容量/stack 上界依赖实测：W5-02 冻结报告未出前，W5-04 容量断言以 manifest 值注入（不硬编码）。
3. Zvt 编码空位（START/END/链声明载体）：已选伴随 CSR（W3-01），地址冻结归 W1-01；若上游译码空间迫使字段布局变化，只改 decl ABI json 位域，不改谓词（W3-01 §13.5 同口径）。
4. `HostRuntime` 未实例化（RES-16）：本包以 `cn.sh package` 实现最小真实上传模型并单列时间，不实现完整 HostRuntime。

## 7. 命令（当前 vs 未来分开）

| 命令                                                                                                                                                                                                                                   | 状态                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_w5_*.py -q`                                                                                                                                              | 未来（本包交付后可跑；当前文件不存在不冒称） |
| `bash hardware/coral_nexus/scripts/cn.sh package --binding hardware/coral_nexus/kernels/bindings/cs1_reference.yaml --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/package`                                   | 未来（W0-02 框架＋本包 handler）             |
| `bash hardware/coral_nexus/scripts/cn.sh cosim --scenario hardware/coral_nexus/verification/scenarios/<t5_xx>.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/<run_id>` | 未来（场景路径 W7-01 schema 冻结后填）       |
| `python -m pipeline_validator --workload pow ...`（Design_Validation §2.1）                                                                                                                                                            | 当前可执行（回归基线，不改）                 |

## 8. 本轮检查记录

- 未执行任何 build/test/formatter；未创建任何目标代码/目录。
- 全部容量/地址类数值均写为实证冻结任务产物（slot_freeze.json/manifest），本计划正文零占位 SHA、零虚构位编码。
- 与 W0/W1/W3/W4/W7 的接口均以 hash/fragment 引用（不互等 DONE）；本文件无 `local://`/`agent://` 持久引用。
