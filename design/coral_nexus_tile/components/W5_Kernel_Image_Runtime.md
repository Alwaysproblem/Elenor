# W5 内核映像与运行时（Kernel Image Runtime）设计评审稿

> 本组件为设计评审稿，跨包合同与用户裁决见[总体设计](../Coral_Nexus_Tile_Design.md)；r5 指[需求规格](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。本轮检查见[设计验证记录](../Design_Validation.md)，不代表组件已实现。

状态：草案，待用户 review；本文是可执行级设计 guide，不是文件级实施计划。日期 2026-11-05。
上游固定版本：Coral RTL [S01](https://github.com/google-coral/coralnpu/commit/f79522157fc0f1d7311f44e209fc36c9c8b7a039)、Coral compiler/runtime [S02](https://github.com/google-coral/coralnpu-compiler/commit/d715556c3853719e414182c0bf5f4a10f248f1d9)；[C01] matrix codegen 设计文档已实读核对（§4.2.1 上游证据）；[S08] IP integration guide：<https://developers.google.com/coral/guides/hardware/integration>。需求基线：[r5](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)（§11.6／§14／§9.4／§13）。除标注外，仓库内证据均指当前 `pipeline_validator` 源码，未做任何构建或仿真。

---

## 1. 定位、目标和 First Silicon cutline

W5 是"计算绑定 → 真实映像 → 封印 package → 驻留／装载 → 失效"这条链的唯一设计 owner。它把 r5 的简版计算合同（CMP-18～CMP-21）与程序驻留合同（RES-01～RES-19）落成具体对象、状态机与消息合同；物理取装通路的数据面分离归 W4，消息编码与传输归 W1，取指消费归 W2/W3，故障与复位联动归 W6。

当前 validator 的四条沉默项是本组件存在的理由（r5 §23.7）：

| 编号 | 现状（源码证据）                                                                                                                                                                       | W5 要补的                                            |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| X-19 | 计算描述符只有标量字段：`TileBoaOp` 仅 `m/n/k/ops/accumulate`、`TileEvuOp/TilePowOp` 仅 `ops/exponent/bytes`，无 source/destination 操作数（`compiler/lowering.py:799-854`）           | 受控 kernel 库＋显式 binding，不猜 operand           |
| X-21 | `text_bytes = max(len(insts)*4 + len(descriptors)*16, 1024)` 是估算公式，哈希对象是抽象 `ExecTileProgram`，无机器码（`compiler/api.py:58-79`、`compiled_program.py:1027`）             | 真实映像 hash／长度／资源，复核后才 seal             |
| X-17 | 驻留表在 Task 提交时同步置 `TILE_RESIDENT`，无 `FETCHING` 中间态、无取数／校验失败路径，字节不经 HBM/NoC/L2 模型（`runtime/program_table.py:17-21,79-100`；`tile_group.py:1311-1318`） | 真实冷装载 bytes、共享安装完成点、累计容量、引用计数 |
| X-18 | 无运行时 descriptor patch 事件，热启动"patch"不可观察（`loader.py:14-25`）                                                                                                             | `patch_mask` 声明式合同，warm 不改 resident text     |

目标：

1. 小 kernel 受控库（人工编写／构建检查的固定模板），每条计算区域显式声明 ins/outs/initial/scratch/clobber/事件/numeric profile 与配置转移集合（CMP-19、MAT-10）。
2. 构建→真实编译／汇编／链接→测量实际 text/stack/scratch/spill→按最终资源复核→封印→只读加载（CMP-20），Loader 不修补、不运行时 codegen。
3. Group 侧驻留管理器＋Tile 侧安装服务的消息合同：冷装载真实字节、共享 pending install、累计容量与引用记账、warm 外部程序文本流量为 0、失效的引用条件与 ACK 顺序。
4. 普通任务启动不得执行 Coral 基线的全核复位＋重写起始 PC 序列（RT-01）。

非目标：完整 RVV 程序 dialect、Coral compiler/LLVM 接入（D-27 后评估）、运行时替换／淘汰策略实现（RES-14 只留接口）、INT8 或对外 FP32 计算模式、FPGA 路径。

## 2. 职责、非职责和 ownership

每项可变状态唯一 owner（总体设计 合同；IF-12）：

| 状态                                                                                                            | Owner                                                                                                                      | 说明                                                                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| kernel 库源、binding 表、manifest、封印产物（sealed package）                                                   | Python（Host 侧，构建期）                                                                                                  | 不可变对象；封印后只读（CMP-14）                                                                                                                                             |
| `(package program_id, version, package_hash, epoch, Tile)` 驻留表、引用计数、累计容量账本、pending install 记录 | Python：Group 驻留管理器（现 `runtime/program_table.py` 的后继）                                                           | 判冷／热、发布 `PROGRAM_READY` 的逻辑权威，镜像 Tile installed 元数据；引用计数全局唯一一份；package_hash 绑定完整合同摘要（§4.1），不继承抽象 IR ID（RES-01/02/11、CMP-18） |
| Tile 本地驻留位图、descriptor 缓存 tag、本地 slot/offset/epoch 句柄                                             | RTL：Tile 侧安装服务（消息/状态语义归 W5；安装 FSM 由 W5 负责，驱动 W4 `TileProgramMemory` 物理端口，见 §7.2 与 W4 §2/§7） | Tile installed 元数据为"本地实际驻留"权威；**不维护引用计数副本**；只接受 owner 消息驱动                                                                                     |
| program SRAM 阵列、安装写端口、取指读端口的物理仲裁                                                             | W4（RTL）                                                                                                                  | RES-17 物理分离／保留区；公平仲裁如实计数（RES-12）                                                                                                                          |
| 取装消息的通道、credit、编码                                                                                    | W1（逻辑通道 PROGRAM）                                                                                                     | 本节只定义语义字段                                                                                                                                                           |
| 消费 resident 句柄取指、ACCEPT→executable 门控                                                                  | W2                                                                                                                         | PROGRAM_READY 先于首次取指（RES-01/19）                                                                                                                                      |
| kernel 内区域协议执行（START/END、FIRST/CONTINUE/FINAL）                                                        | W3                                                                                                                         | manifest 声明，W3 执行检查                                                                                                                                                   |
| 安装失败／校验失败的故障升级、复位时失效                                                                        | W6                                                                                                                         | fail-stop，VB-11                                                                                                                                                             |

与相邻组件的输入输出：

- 来自 W0：固定 pin 的上游编译器／汇编器调用方式与 ITCM/DTCM 布局事实（[S08]）。
- 来自 W2：Task commit 时的程序身份请求（`COMMIT` 携带 kernel ID/version/entry/manifest 身份，§13.2）。
- 给 W2：`PROGRAM_READY`（program_id/version/hash/epoch/本地句柄）；热命中直接满足。
- 给 W4：安装写请求（目标 slot/offset/bytes，走保留通路，不进用户 Arena/MSHR/数据 cache）。
- 给 W7：RES-15 计数器（冷/热命中、装载字节/周期、安装开始/完成、等待原因）与边界事务日志。

## 3. 微架构和状态机

### 3.1 Package 构建与封印状态机（Python，构建期）

`AUTHORED（模板＋binding）→ COMPILED（真实编译/汇编/链接）→ MEASURED（实测 text/静态数据/stack/scratch/spill 上界）→ RECHECKED（逐 bank Arena、指令存储累计、事件预算复核）→ SEALED → LOADED（只读验证）`。

- 每步产物不可变；任一步失败回到 `AUTHORED`，禁止在 Loader/adapter 修补（CMP-20）。
- MEASURED 方法：链接器/map 文件给出 text 与静态段实际大小；stack/scratch/spill 取**有界静态分析**上界（只有固定模板＋受控 kernel 才可审，CMP-13）；单次运行动态 high-water 只是旁证、不构成 bound 证明；后端或代码变化必须重新核验（CMP-05）。
- 封印对象 = 旧 `CompiledProgram`（schema 3／ABI v3，`compiled_program.py:197-218,1089`，严格 JSON/类型/opcode 白名单与重复键拒绝已具备）**外层包裹**：`{control_plane: CompiledProgram, image: {bytes/hash/entry/layout}, binding_manifest, resource_manifest, target_fingerprint, patch_mask, package_version}`。这是 r5 明示的"简版可把旧 CompiledProgram、显式 binding、真实映像与 manifest 一起封印"（CMP-18），新版本域独立记账（CMP-15），不暗改旧对象 schema。
- 不从旧 `ExecEngineDesc` 猜 A/B/C：BOA/EVU descriptor 仍按现行为进入 control plane 的 `opaque_events` 保护（`execution_verifier.py:733,791`），真实操作数只来自 binding manifest。

### 3.2 驻留状态机（Group 侧，每 `(image_id, Tile)`）

```
ABSENT ──install req──▶ PENDING_INSTALL ──installed+verified──▶ RESIDENT
  ▲                        │   ▲ (+ref: 同映像后续 Task 只加引用,  ▲
  │                        │   │  共享同一完成点, 不重复装载 RES-11/19)
  │   RESIDENT ──PROGRAM_INVALIDATE(与 Task COMMIT 同串行点原子进入)──▶ INVALIDATING
  │   INVALIDATING: 停新引用/热命中 → 等执行引用+待执行Task+安装事务归零
  │                 → Tile 失效+收齐 ACK → 归还 slot/容量 → 允许新安装 ──▶ ABSENT
  │              取数/校验/容量故障 → fail-stop (W6), 无半驻留记录
  └── TILE_RESET / GROUP_RESET(epoch++) ──┘
```

- `PENDING_INSTALL` 是共享单点：所有等待该映像的 Task 挂在它上面，首次取指一律晚于安装完成点（RES-13）——这是对 validator 现状"提交即 `TILE_RESIDENT`"（`tile_group.py:1311-1318`）的修正，validator 把 ACCEPT 倒计时（`tile.py:279` `prepare_total=1+prepare_cycles`）当准备态，RTL 不得照此放行取指。
- 容量按**所有驻留＋安装中程序累计**计账（RES-14）：单程序超限在构建/加载拒绝；累计不足时未 accepted 的 Task 零新增资源等待——等待出口仅两种：引用归零后**显式失效**（收齐失效 ACK 才退还累计容量与 slot，§5.4）或拒绝上报；**Task 退休只减少引用，不释放驻留容量、不卸载 resident text**（保留供下一次 warm 命中，RES-10/14），不得因退休虚减账本或逐次卸载；已 accepted 的共享 pending 只加引用不加指令容量。
- 程序引用随 Group 驱动退休链递减（CTX-15/ARN 基线，总体设计 §4）：W2 Context 内 `EXEC_DRAINED` → RTL 无旧在途/VME 结束后报告 `RETIRE_READY`（非 Task success）→ Group `RETIRE` 授权关闭本地 view/Arena/Frame → RTL 完成本地回收并可靠保存 `TASK_COMPLETION` → Group 接收后清理剩余父 L2 pin、归还 R 并结算 Grid 记录。引用在 Context 旧义务闭合（`TASK_COMPLETION` 接收且记录与新代隔离；保守可等完成握手）后递减，归零才可失效；输入纯读 pin 可按总体设计 §4.4.1 先解除，但 `INPUT_RELEASED` 不递减程序驻留引用。Python 不再执行 L1 retire、不以估时或猜测提前释放。
- validator 现状差异表：`FETCHING` 枚举存在但从未置位、`evict()` 无调用者、`hbm_iova` 恒 0（`program_table.py:17-21,130`；X-17）——CS1 联仿中 Python 侧必须真的走 bytes 与安装事务，周期不对拍（X-17 裁决），顺序对拍。

### 3.3 Tile 侧安装服务（RTL 合同）

四步 pull 协议（与 W1 对齐）：Group 发 `PROGRAM_INSTALL` 头（身份/epoch/slot/bytes/HBM 段，不含字节）→ Tile 安装服务发 `PROGRAM_READ_REQ` 分段要字节 → Group 回 `PROGRAM_READ_RESP`（真实字节，保留通路，不占数据 Arena/MSHR/credit，RES-17/18）→ Tile 写 program SRAM（W4 保留端口）、身份校验（hash/长度）后回 `PROGRAM_INSTALL_ACK`（实测 hash/长度＋本地句柄）。有界 FIFO、可分段背压；校验失败即故障（VB-11），不留半驻留；descriptor 缓存 tag 带 sequence/epoch（RES-04）。权威关系：Tile installed 元数据（本地句柄/实测 hash/驻留位）是"本地实际驻留什么"的权威，Group 驻留表镜像它并作为判冷／热与 `PROGRAM_READY` 的逻辑权威；引用计数只在 Group 一份，RTL 不设副本。

## 4. 接口、descriptor、寄存器和协议

### 4.1 逻辑消息（W1 的 PROGRAM 通道承载；字段版本化、有界、可序列化）

| 消息                     | 方向                  | 载荷要点                                                                                                                                                                                                               | 错误/背压                                                                                                                                                                                                                     | Owner                 |
| ------------------------ | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `PROGRAM_REGISTER`       | Host→Group 驻留管理器 | 外层 package 身份（program_id＋version＋**package_hash**，覆盖 image_hash＋binding/resource manifest＋target_fingerprint＋patch_mask 完整摘要，不继承抽象 IR `program_digest`）＋text_bytes＋HBM 段位置＋manifest hash | 同 package_hash 重复登记→幂等复用；**image bytes 不变但 binding/resource/target/patch_mask 任一变化→新身份、旧身份失去热命中（CMP-18/RES-02）**；同 (program_id, version) 而 package_hash/manifest_hash 不同→拒绝（身份歧义） | Python                |
| `PROGRAM_INSTALL`（头）  | Group→Tile            | 身份、epoch、目标 slot/offset/bytes、HBM 只读段位置（不含字节）                                                                                                                                                        | FIFO 满背压；不得依赖用户数据释放（RES-18）                                                                                                                                                                                   | Python 发起, RTL 执行 |
| `PROGRAM_READ_REQ`       | Tile→Group            | 分段读请求（offset/长度）                                                                                                                                                                                              | 有界 credit；程序服务独立进展（RES-17/18）                                                                                                                                                                                    | RTL                   |
| `PROGRAM_READ_RESP`      | Group→Tile            | 真实程序字节段                                                                                                                                                                                                         | 走保留通路；取数错误→fault（VB-11）                                                                                                                                                                                           | Python                |
| `PROGRAM_INSTALL_ACK`    | Tile→Group            | 身份、epoch、实测 hash/长度、本地句柄                                                                                                                                                                                  | 校验失败→fault（VB-11）                                                                                                                                                                                                       | RTL                   |
| `PROGRAM_READY`          | Group→W2              | 身份＋epoch＋本地句柄                                                                                                                                                                                                  | 必须先于首次取指；热命中直发（§13.1）                                                                                                                                                                                         | Python                |
| `PROGRAM_INVALIDATE`     | Group→Tile            | program_id/epoch＋作用域（group/单 Tile）                                                                                                                                                                              | 与 Task COMMIT 同串行点原子进入 INVALIDATING（停新引用/热命中）→ 执行引用/待执行 Task/安装事务归零后下发（RES-04、§5.4）                                                                                                      | Python 特权           |
| `PROGRAM_INVALIDATE_ACK` | Tile→Group            | 身份＋失效后 epoch                                                                                                                                                                                                     | 引用归零＋descriptor 缓存已失效后才 ACK                                                                                                                                                                                       | RTL                   |

消息重放不重复执行；恰好一次提交、可靠终态、确认后信用归还（总体设计 边界合同、IF-11）。所有 `PROGRAM_*` 消息带 op_seq/txn_id，身份使用统一分域键：`program_epoch` 与 `ctx_generation`/`arena_alloc_gen`/`profile_gen`/`run_gen` 互不混用（TaskKey＝root_launch_generation＋GridInstanceId＋task_id，PhysicalTile 路由单列）。本文消息名为逻辑合同，物理 opcode 名称与位编码待 review 冻结，不自造。descriptor patch 不在装载消息上：`PROGRAM_INSTALL` 只携带身份与位置，受 `patch_mask` 约束的 descriptor 字段更新随 Task 提交路径解析（W2，RES-07 分层），warm 不改 resident text（已与 W1 §3.2/§4.2 对齐，字段级出入以本文为准，review 时统一）。

### 4.2 kernel binding manifest（封印内容，CMP-19；缺字段即构建拒绝）

| 字段组              | 内容                                                                                                                                                                                                                                                                                                                                                                                                                             | 拒绝条件（构建/加载期）                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| kernel 身份         | kernel ID＋version＋受支持清单登记                                                                                                                                                                                                                                                                                                                                                                                               | 未登记 kernel ID；版本不符                                                                                                                                                                                  |
| `ins`/`outs`        | 显式 view：handle、shape、dtype、连续布局、权限、合法 alias 规则                                                                                                                                                                                                                                                                                                                                                                 | 缺 `ins`/`outs`；与模板签名不匹配；非法 alias（如 out alias 只读 in）                                                                                                                                       |
| `initial`           | L1 累加初值/seed 方式（vtzero/vtmv/tile load，MAT-24）                                                                                                                                                                                                                                                                                                                                                                           | 无 seed 的 FIRST 链；依赖未声明 mt 残留                                                                                                                                                                     |
| `scratch`/`clobber` | 本地工作区、寄存器 clobber 集、stack/spill 上界                                                                                                                                                                                                                                                                                                                                                                                  | 未声明工作区；实测超上界（T-63）                                                                                                                                                                            |
| 事件                | 前置/完成事件名（Context 内唯一，ISA-09）、phase 契约                                                                                                                                                                                                                                                                                                                                                                            | 缺完成事件、重名                                                                                                                                                                                            |
| numeric profile     | 张量计算数据通路 BF16 I/O、FP32 内部累加；内部 FP32 seed/drain/scratch 与标量控制/地址/索引整数（i32 等）属合法配套，不算新增数值精度；舍入/特殊值/累加顺序合同引用 W3 数值规格——**不预设通用 FP32 FMA 顺序**：Coral BF16 VME 路径内部为 ROD（round-to-odd，**用户已确认保留**，默认绑定 W3 已确认的内部 ROD 语义）、BF16×BF16→FP32 为宏门控真实路径；BF16 最终输出转换的舍入模式单列，建议 RNE（待 review），以真实指令证据为准 | profile 不在 kernel 支持矩阵；张量通路越权加 INT8 或对外 FP32；BF16 MAC 需 `+zvtf16f32mm`、TK=2 需 K mod 2=0、BF16 narrowing 输出需 `+zvfbfmin` 且 M mod 4=0——不满足即构建拒绝或显式声明 RVV 降级，不得静默 |
| 配置转移表          | MAT-10 硬件可检查表示：模式枚举＋模式间转移表＋tm/tk/vl 字段上下界；首版内容按 §4.2.1 从固定 compiler 实读落地                                                                                                                                                                                                                                                                                                                   | 静态可识别越界转移；表与封印 profile/target 指纹不一致                                                                                                                                                      |
| 区域声明            | 每区域 START/END 位置与每条累加链 FIRST/CONTINUE/FINAL（W3 执行）                                                                                                                                                                                                                                                                                                                                                                | 缺 START/END、链缺 FIRST/FINAL、重复 FIRST（静态可识别部分）                                                                                                                                                |
| `patch_mask`        | 可 patch 字段白名单（RES-07 分层：Runtime→UCE→MFE）                                                                                                                                                                                                                                                                                                                                                                              | 未声明字段被 patch、越界＝invalid descriptor 故障                                                                                                                                                           |
| 命名 binding        | HBM/参数按名字绑定（RT-12）：名字、权限、最小范围、relocation                                                                                                                                                                                                                                                                                                                                                                    | 位置匹配；区间重叠；重放静态容量（`context_count` 等）不符                                                                                                                                                  |

数值合同（本轮用户确认）：**张量计算数据通路 BF16 仅 I/O、FP32 内部累加，不新增 INT8**；内部 FP32 seed/drain/scratch 与标量控制／地址／索引整数（i32 等）是合法配套，不属被禁的新精度，也不得误删。r5 §9.3 参考负载最低要求是"BF16 输入、FP32 累加与**输出**"——本设计把对外张量输出收窄为 BF16，须在组件文档记录该差异并另建 BF16 语料，不改旧 fixture、不冒称旧 FP32 输出用例已覆盖（validator 现有 `boa_dtype_bytes=2/boa_acc_bytes=4` 与 BF16-in/FP32-acc 一致，对外 dtype 是新维度）。

#### 4.2.1 首版受控 kernel 条目与配置转移表内容（固定 compiler [C01] 实读核对，非猜测）

[C01](https://github.com/google-coral/coralnpu-compiler/blob/d715556c3853719e414182c0bf5f4a10f248f1d9/docs/design/matrix_codegen.md) §1/§2.1/§2.3/§3 的直接 codegen 事实是首版模板库与转移表的来源：

| 条目                 | 上游事实（[C01]）                                                                                                                                                                                                                                                                                         | 进入 binding/manifest 的方式                                                                                                                                  |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| K-VME-BF16-ACC       | BF16×BF16→FP32 累加：`vtfmm.alt.tvv`（`+zvtf16f32mm`）；TM=TN=16，TK=2 且要求 K mod 2=0；一个 16×16 FP32 块占 4 个连续 mt；mt0/4/8/12 是四个独立块的基址（例如基址 mt0 使用 mt0..mt3）；布局 2×2(32×32)／1×4／4×1／1×2／2×1／1×1 按 M、N 整除性与 DTCM fit 静态选择，不 fit 则缩 tile，再不行回 RVV       | shape→布局映射进模板签名；K mod 2≠0 用 TK=1 单步或拒绝；布局不匹配即拒绝（T-61）                                                                              |
| seed/drain           | 清零／播种／排空块恒用 FP32 配置（累加器恒 32 位）：`vtzero`、`vle32.v`＋`vtmv.t.v` 逐行 seed、`vtmv.v.t`＋`vse32.v` drain；计算型 init（如 bias）上游留在 RVV                                                                                                                                            | `initial` 仅 vtzero／vtmv／tile load（与 MAT-24 一致）；bias 类 init 必须显式声明走 RVV helper，不得猜成 seed                                                 |
| BF16 narrowing drain | `vfncvtbf16.f.f.w`＋`vse16.v`，需 `+zvfbfmin` 且 M mod 4=0，每 4 行在 FP32↔BF16 配置间交替；无该 feature 时 truncf 落在 consumer 行循环（RVV）                                                                                                                                                            | `outs` dtype=BF16 的模板把该交替序列写入转移表；缺 feature 时构建拒绝或显式声明 RVV 收尾降级，不静默                                                          |
| 配置前导             | 每个汇编块自带前导（`li t5,<mtype>; li t6,<vtype>; msetmtype; msettn`）：固定源码 `emitTileConfigAsm` 注释明示 **LLVM 可能在块间插 `vset*`（清 mtype 与 vtype.altfmt），故逐块重发前导**；`msetmtype` 在 mtwiden≠0 时按 SEW 推导 LMUL 并清 `vl`，`msettn` 恢复 TN=16（partial-N seed/drain 用 TN∈{8,12}） | 构建期静态检查限定为：asm 块内无 `vset*`、每块以 `mset*` 前导开始（下一次 mt 使用前恢复）；块间 `vset*` 合法且被前导吸收（§4.2.1 转移表 `模式→CLEARED→模式`） |
| 部分 tile            | M<16 用 `vid.v`＋`vmsltu.vx` 掩码装载；N∈{8,12} 走 `msettn`；非整块维用 loop peeling 产生静态子 view                                                                                                                                                                                                      | binding shape 与模板支持集不符即拒绝                                                                                                                          |
| scratch/代码尺寸     | 非融合链 drain 到 workgroup_local `memref.alloc`（16×64 f32 tile 超 4 KiB 栈）；K-loop 为固定寄存器 asm 循环（t6 计数）＋限长展开，代码尺寸有界（ITCM 小）                                                                                                                                                | scratch/spill 入资源 manifest（CMP-05）；text_bytes 以实测封印（CMP-18/20）                                                                                   |
| K-RVV-EVU            | 逐元素／vecmat 走 RVV；VLEN=128 且 `+zvfbfwma` 时 vecmat K-loop 用 `vfwmaccbf16.vf`                                                                                                                                                                                                                       | EVU 类 binding 显式绑定 RVV helper kernel 与完成事件，数值合同同上                                                                                            |
| 数值模式             | 上游 `--coralnpu-fast-math` 默认开（`CoralNPUApproximateMathPass` 替换 tanh/exp/erf 近似）                                                                                                                                                                                                                | 封印 profile 必须显式声明近似选项与范围（CMP-09），禁止静默继承上游默认                                                                                       |

**转移表候选实例（非通用形状上限，待 review）**（模式值为 [C01] §2.3 上游实值，且已由固定源码 `compiler/Transforms/CoralNPUMatrixCodegen.cpp` 的 `emitTileConfigAsm`（`mtype = 16384|(tk<<5)|w`）逐值核对；编码归属上游）：合法模式枚举 `FP32-MAC`（`msetmtype(16417,18)`，mtwiden=1，vtype=18＝SEW32/LMUL4）、`BF16-MAC`（`(16418,265)` 单 K 或 `(16450,265)` TK=2，mtwiden=2，vtype=265＝SEW16/LMUL2/altfmt=1）；INT8 模式（`(16419,256)`／`(16515,256)`，mtwiden=3）登记为存在但本版张量通路禁用。合法转移：`∅→FP32-MAC`（seed）、`FP32-MAC→BF16-MAC`（运算装载前切换）、`BF16-MAC→FP32-MAC`（drain／narrowing 交替）、`FP32-MAC→FP32-MAC`（TN 重编程 16↔8/12）、`模式→CLEARED→模式`（块间 `vset*` 清 mtype/altfmt 后由下一块 `mset*` 前导恢复——LLVM 可能在 asm 块间插 `vset*`，这是逐块前导存在的原因，属已建模合法路径而非拒绝事件）。拒绝条件：asm 块内出现 `vset*`、或 mt 使用时配置不在声明模式内（前导恢复缺失）。本组上游模板的字段范围：TM=16、TN∈{8,12,16}、TK∈{1,2}；其他 M/N 尾块必须配套独立受控模板、mask/packing 及其转移表，不可将本表冒充全部需求的支持范围、`vl` 由 `msetmtype` 清零后按块内装载需求重建。该表随 kernel/profile 封印，进入 lease 包络与 target 指纹，构建期（CMP-19）与运行时（W3）同一份表；若首版模板进一步禁止区域内一切 `vset*`，属项目自加的更严限制（待 review），不是上游性质。

## 5. 数据流、控制流和时序路径

### 5.1 冷启动（T-58 主路径）

1. 准入分层（总体设计 §4）：Group 有界 ticket／无副作用 `PLAN` → 父 L2 pin/R 预留 → RTL 本地资源 hold（含程序 slot 引用／安装等待记录；不 bind 可执行 UCE、不取指）→ Group 确认父／本地全部持有后最终 `COMMIT` 授权 → RTL 原子安装 Task 绑定并 ACK accepted；授权前失败补偿释放并 ACK（票据保留到释放证据），最终授权后不 ROLLBACK、走 `CANCEL`/drain；无跨 Tile gang。accepted ≠ installed ≠ executable（RES-19）：程序安装可与准入并行，`PROGRAM_READY` 本身永不授权未 accepted Task 的执行（IF-13）。
2. Group 驻留管理器判冷：该 `(identity, Tile)` 无驻留或 epoch 不符（RES-03）→ 若无 pending 则发 `PROGRAM_INSTALL`，建立共享完成点。
3. Tile 安装服务按 §3.3 四步 pull 取真实字节（Host 已随 package 放入 HBM，RES-16；Host 上传开销另行报告）→ 写 SRAM → 校验 → `PROGRAM_INSTALL_ACK`。
4. Group 发布 `PROGRAM_READY`；W2 在 Frame/Context 就绪＋Profile gate 全满足后才取指。因果序上首次取指晚于安装完成与 `PROGRAM_READY`（含安装中又提交的同程序 Task）；同周期允许已定义的逻辑先后（总体设计 统一采样/提交相位），不要求 tick 数值严格大于。
5. 冷装载只阻塞等待该映像的 Context；其他 Context 照常发射退休，无全 Tile stall/flush/全局 start PC 改写（RES-12、RT-01）。
6. 装载失败：fault → fail-stop（W6），取消/排空后回收，无半驻留记录。

### 5.2 热启动（T-49/T-58）

命中条件＝该 Tile 驻留有效且 `program_id+version+hash+epoch` 全匹配（RES-03）→ 外部程序文本流量 0 字节、装载服务周期可为 0、不改写 resident text；但 `PROGRAM_READY`/控制路径非 0——热启动不是零启动开销：Task 提交仍付 ACCEPT 与 Frame 绑定；本地 ITCM/program SRAM/cache 命中的正常取指照常、不计入（§11.6.1 定义）。动态量只来自 Loader actual binding、提交时 Task 字段与 `patch_mask` 内 descriptor 字段（RES-10）。warm 入口 Profile 必须等于 artifact 记录，不匹配走显式静止恢复，不自动 reset/切档（RT-11）。

### 5.3 同映像重入（多 Context，CMP-12）

同映像 readonly text/常量全 Tile 共享一份驻留（引用计数＋N）；可写状态（参数区、stack/scratch/spill、L1 accumulator）每 Context 独立实例，由 binding manifest 的 per-Context 段落声明。禁止每 kernel 全局清零（RT-06）。重入不产生第二次安装、不重复占指令容量。

### 5.4 失效与复位

`PROGRAM_INVALIDATE`（特权）的关门线性化：Group 驻留管理器与 Task COMMIT 共用同一串行化点（IF-13 有界 ticket 协调器），失效请求在该点原子将条目置入 `INVALIDATING`——此后该程序不再接受新引用、不再热命中（单 Tile 失效不改 epoch，同样由该门控防止发出即将失效的旧句柄），新提交排队为未来冷装或被拒绝。随后等待执行中引用、待执行 Task、安装中事务全部归零（引用按 §3.2 退休链递减）；再向作用域内 Tile 发失效并收齐 ACK（Tile 先失效驻留与 descriptor 缓存带 sequence/tag 再 ACK）；收齐 ACK 后才归还 slot／容量并允许同 slot 新安装。group 级失效 epoch 递增并使全部驻留失效，单 Tile 失效只影响该 Tile；reset 使驻留失效（RES-04）。执行中 text 不得被覆盖，换程序走静止或 epoch 切换（RES-08/FET-06）。validator 现状只在复位状态机内失效（`runtime/reset_domain.py:121-123`），运行期失效是新增 RTL/联仿合同。

### 5.5 普通启动 vs Coral 基线

Coral 官方序列：Host/DMA 经 `S_AXI` 写 ITCM（默认 8 kB）→写起始 PC CSR→释放时钟门控→释放复位；FPGA 路径每次 `Run` 复位并重写起始 PC（[S08]；[C04](https://github.com/google-coral/coralnpu-compiler/blob/d715556c3853719e414182c0bf5f4a10f248f1d9/runtime/sim/fpga/simulator.cpp)，r5 §3.2）。这些全核动作不得出现在 Nexus 普通 Task 启动路径：启动＝`COMMIT`＋可选安装＋`PROGRAM_READY`＋per-Context 入口 PC（来自 manifest entry），Tile 级初始化与任务 admit 分离（RT-01/02）。

## 6. 配置、PPA、性能模型和 PMU

- 容量：`tile_program_sram.capacity_bytes: 65536` 是 validator 模型现值（`hardware_config.yaml:78-79`），**不是 RTL 目标**；CS1 目标容量与 slot 数由后续 SRAM profile 冻结（由后续规格冻结）。程序 footprint 分真实映像、只读内容、对齐与受保护元数据计账，不混用户 Arena（RES-05）；累计容量按 §3.2 记账。
- 版本域（strict）：package 版本、control plane 的 schema 3/ABI v3、硬件 YAML schema 2、command/event/日志 schema 各自独立记账（CMP-15）；`target_fingerprint` 必须绑定 manifest 且把 VLEN、VME 几何、Context 数、程序 SRAM 容量等新字段纳入（CMP-16；现指纹函数 `execution_verifier.py:123`）；不匹配在加载前拒绝（CMP-11）。
- 计数器（RES-15）：冷/热命中次数、冷装载字节/周期（PMU `program_cold_load`、`cold_load_cycles` 沿用）、安装开始/完成、等待原因；RTL/adapter 补真实安装事务并区分程序流量与数据流量。正常模式不留历史日志，时间线由 Host/debug 捕获（VER-14，W7）。
- 性能：validator 解析式冷代价（两段带宽取整＋1 ACK，`program_table.py:102-111`）只作参考口径，CS1 周期不对拍（X-17）；真实性能以 RTL cycle 证据为准；Host 上传开销不含在冷启动周期内，单独报告（RES-16；`host_session.py:197` 每次 host_call 固定 `host_patch_cycles`，冷热相同）。
- 装载位置策略：主建议**多固定预链接 slot 表**——每映像构建期固定装载地址写入 manifest（零重定位、校验最简），多 slot 支持多程序并存（T-58：冷装新映像期间其他 Context 的驻留程序继续运行）＋引用归零后显式失效释放 slot；**并存前提是链接地址互不重叠且累计容量足够，Loader 不做重定位、不把固定地址映像搬到其他 slot**，同 slot 换包须先走 §5.4 失效；slot 数与每 slot 容量参数待 review。单一共享基址（同刻仅 1 映像）不满足 T-58，弃用；动态参数一律走独立参数区，不重写常驻指令（§14.3.1）。

## 7. RTL/软件实现建议

1. **monorepo 位置**：kernel 库源与构建脚本统一放 `hardware/coral_nexus/kernels/`（`library/` 模板＋binding；构建产物写 `out/coral_nexus/`；不新增顶层 `kernels/`）；binding/manifest 编解码建议扩展现有 `pipeline_validator/package/`（已有 `package_ir.py`，但未提供本设计真实映像与绑定链，X-17；新 API 为 proposal）；驻留管理器演进建议仍放 `pipeline_validator/runtime/program_table.py` 同域（新增 pending/引用/累计容量）；抽象 timing 语义仅在**显式 python reference 模式**保留（timing_only 恒 0，RES-03），CS1 联仿强制新封印 package——缺 binding/manifest 即拒绝，无旧 API fallback shim；RTL 侧（Tile 安装服务等）设计产物归 `hardware/coral_nexus/`，不依赖也不改动现有 `hardware/{gemmini,boa,toolchain,scripts}` 空目录。
2. **Tile 侧安装服务归属**：本设计按"逻辑驻留/安装状态归 W5 合同、物理阵列与端口仲裁归 W4"切分；安装控制与 installed 元数据由 W5 唯一写，W4 只执行存储端口/仲裁；可物理组合模块，但不能重复维护两套安装 FSM。
3. **mutable/readonly 分区**：readonly（text、常量、封印 descriptor）映像级共享；mutable（参数、stack/scratch/spill、L1 accumulator）按 Context 实例化，全部入资源 manifest 与 Task 准入（CMP-05/06）。
4. **ISA helper 表达**：区域 START/END、链 FIRST/CONTINUE/FINAL、await/signal 等控制语义按 ISA-03 允许 custom instruction／CSR/MMIO／受控 helper 三类候选表达；本组件只在 manifest 中以逻辑名声明这些点，**不造 opcode、不冻结编码**（编码由 ABI 评审冻结；Coral Zvt 现有编码是否有空位未核对，r5 MAT-20 已记录）。
5. 封印算法与信任边界：`artifact_digest`/`seal_program`/`parse_compiled_program` 的 hash＋白名单机制（`compiled_program.py:1041-1100`）作为外层 package 编解码的参考实现；真实 `image_hash` 覆盖实际装载字节（CMP-18）。**hash 不是信任证明**：重写映像并重 seal 得到的是新 package，必须经可信构建链 whitelist（受控 kernel 库＋可信构建产物登记）验证；Loader 只重验合同层语义、依赖与 actual binding，不宣称独立证明任意机器码（CMP-14 信任边界显式记录）；篡改合同层后即使重算 hash 仍被语义校验拒绝。

## 8. 验证、bring-up 和验收标准

### 8.1 用例映射（正面）

| 用例                                                                                 | 断言要点                                                                                                                                                                                                                                                                                                                                              | 需求                    |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| 同形状不同 ins/outs/累加对象                                                         | 固定 kernel 按显式 view 执行、结果分别正确；不从 shape/ops 猜操作数                                                                                                                                                                                                                                                                                   | CMP-19、T-61            |
| 真实映像封印                                                                         | image hash/长度/entry/binding 一致；实测资源复核后才 seal                                                                                                                                                                                                                                                                                             | CMP-18/20、T-63         |
| 冷→热→身份变更→group 复位→单 Tile 失效                                               | 外部文本流量冷>0、热=0（边界计数器/事务日志证明，本地取指照常）；身份变更全部重冷；单 Tile 失效只影响该 Tile                                                                                                                                                                                                                                          | RES-03/10/04、T-58/T-49 |
| 安装中再提交同映像                                                                   | 后续 Task 只加引用、共享完成点、首次取指不早于安装完成点                                                                                                                                                                                                                                                                                              | RES-11/13/19、T-66      |
| 同映像多 Context 重入                                                                | 引用计数守恒；无重复安装；per-Context 可写区独立、互不污染                                                                                                                                                                                                                                                                                            | CMP-12、RT-06           |
| 冷装载期间其他 Context 运行                                                          | 无全 Tile stall；程序服务不占数据 Arena/MSHR、不饿死                                                                                                                                                                                                                                                                                                  | RES-12/17/18、T-66      |
| 失效引用/ACK/交错顺序                                                                | 失效与 Task COMMIT 在同一串行化点原子进入 INVALIDATING：进入点后新 COMMIT 不获热命中/新引用（含单 Tile 失效 epoch 不变），先等执行引用＋待执行 Task＋安装事务归零，收齐失效 ACK 后才归还 slot/容量并允许新安装；ACK 晚于 descriptor 缓存失效；epoch 递增正确                                                                                          | RES-04、§13.1           |
| 同机器码、改合同（image_hash 相同，binding/scratch/resource/patch_mask/target 变化） | 旧 package 身份失去热命中，按新身份走冷装；**双驻留仅当两包链接地址互不重叠且累计容量足够**——仅改 scratch/binding 的两包常链接到同一 slot，此时新包须等旧引用归零、显式失效 ACK 收齐后再冷装该 slot（§5.4）；Loader 不得把固定地址映像搬迁到任意空 slot 制造并存（无重定位，§6）；同 (program_id, version) 但 manifest_hash 不同→登记拒绝（身份歧义） | CMP-18、T-63            |

### 8.2 负面用例（拒绝/故障，均为构建/加载期拒绝或 fail-stop）

| 注入                                                                                             | 期望                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| package 篡改（改 manifest/image 字节、删合同层步骤）                                             | 加载前拒绝：hash 不符、重复键、未知字段、非有限数（CMP-14，`compiled_program.py:1081-1100` 机制）；改合同层后即使重算 hash 仍被语义校验拒绝；重写 image 并重 seal＝新 package，须过可信构建链 whitelist（§7.5），Loader 不宣称证明任意机器码         |
| binding 缺 `ins`/`outs`/`initial`、alias 非法、kernel 不匹配                                     | 构建/加载拒绝，不猜（T-61）                                                                                                                                                                                                                          |
| 实测 spill/scratch/代码大小越界；累计驻留超目标容量（模型现值 64 KiB 仅为参数示例，非 RTL 目标） | 前者构建拒绝；后者 admission 有界等待或单程序超限拒绝（RES-05/14、T-63）                                                                                                                                                                             |
| 配置转移越出封印表                                                                               | 静态可识别→构建拒绝；运行时→协议故障交 W3/W6（MAT-10、T-62 注入项）                                                                                                                                                                                  |
| 取数错误、身份校验失败                                                                           | 故障＋无半驻留记录（RES-13、VB-11）                                                                                                                                                                                                                  |
| warm 尝试改写 resident text／patch 越出 `patch_mask`                                             | 拒绝/invalid descriptor 故障（RES-08/07）                                                                                                                                                                                                            |
| Zvt **asm 块内**出现 `vset*`、块内 mt 使用前无 `mset*` 前导恢复、或转移表外模式切换              | 构建期拒绝。块**间**的 `vset*` 合法：固定源码 `emitTileConfigAsm` 逐块发前导正是因为 LLVM 可能在块间插 `vset*`（清 mtype/altfmt，由下一块前导吸收），不得据块间 `vset*` 拒绝（[C01] §2.3；`compiler/Transforms/CoralNPUMatrixCodegen.cpp` 固定 pin） |
| `target_fingerprint` 不符、旧身份命中新程序                                                      | 加载前拒绝（CMP-11/16、RES-02）                                                                                                                                                                                                                      |
| 失效进行中（INVALIDATING）提交同程序新 Task 请求热命中                                           | 不得命中：进入 INVALIDATING 线性化点之后不再发出旧句柄（尤其单 Tile 失效不改 epoch 的场景）；新 Task 排队为未来冷装或拒绝                                                                                                                            |

### 8.3 实现代理 guide（面向低能力实现代理：读→实现→自证；非 Python 降阶模型、非实施计划）

1. **读**：`runtime/program_table.py` 全文、`tile_group.py:1284-1350`（提交期驻留判定）、`compiled_program.py:1027-1100`（封印/解析）、`loader.py:14-25`、`compiler/api.py:58-90`（身份估算）、`compiler/lowering.py:799-855`（无 operand 现状）、r5 §11.6/§14 与本文 §3～§5。
2. **实现**（在真实归属位置，不建第二套降阶模型）：Group 侧 Python 驻留管理器（§3.2 状态机＋全局唯一引用计数）与封印/加载校验（§3.1/§7.5）；Tile 侧按 §3.3 四步 pull 与 W4/W1 合同接线；每条不变量落到可断言位置（Python assert/测试钩子、RTL 边界事务日志）。
3. **自证不变量**：(a) 外层 package 身份（program_id＋version＋package_hash，package_hash 覆盖 image_hash＋binding/resource manifest＋target_fingerprint＋patch_mask 的完整摘要，不继承抽象 IR `program_digest`）唯一决定驻留命中——**image bytes 不变但任一合同摘要变化＝新身份，旧热命中失效**（CMP-18）；同 (program_id, version) 而 manifest_hash 不同＝身份歧义，拒绝；(b) 因果序上首次取指晚于安装完成与 `PROGRAM_READY`（同周期按 总体设计 统一提交相位判逻辑先后，不比 tick 数值）；(c) 程序引用计数守恒且只沿 Group 驱动退休链递减（W2 `EXEC_DRAINED`→RTL `RETIRE_READY`→Group `RETIRE` 授权关闭本地 view/Arena/Frame→RTL 完成本地回收并可靠保存 `TASK_COMPLETION`→Group 接收后清理剩余父 L2 pin/R/Grid），归零才可失效；退休只减引用，不释放驻留容量、不卸载 resident text（保 warm，RES-10/14）；容量退还仅经显式失效收齐 ACK；`PROGRAM_READY` 本身不授权未 accepted 的 Task；(d) 累计驻留＋安装中字节 ≤ 目标容量恒成立；(e) warm 外部程序字节数恒 0（边界计数器）；(f) 封印产物不可变、重放加载结果一致；(g) 空间不足等待必有出口（显式失效释放容量后重试或拒绝上报），无永久等待；(h) 失效线性化：进入 `INVALIDATING` 的串行点之后无新引用/热命中（单 Tile 失效 epoch 不变时同成立），slot/容量归还与新安装允许均晚于该程序全部失效 ACK。
4. **停止条件**：上述不变量在语料上全部成立且 §8.2 负面用例行为正确即达目的；安装微架构周期数、通道物理编码、数值 oracle（W3/W7）不在本 guide 范围（X-17 周期不对拍）。

## 9. 风险、取舍和后续细化方向

| #   | 项                                                                                       | 影响                              | 处置建议                                                                                                                                                             |
| --- | ---------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Tile 安装控制与物理端口边界                                                              | 联仿接线与唯一 owner              | W5 安装 FSM/元数据驱动 W4 `TileProgramMemory`；仅物理编码与预算仍待 W1，不存在第二安装 FSM                                                                           |
| 2   | slot 数与每 slot 容量                                                                    | 多程序并存能力与碎片              | 主建议已收敛为多固定预链接 slot（§6）；单一共享基址因不满足 T-58（冷装新映像时其他 Context 须继续运行）弃用；slot 数/容量参数待 review                               |
| 3   | BF16 对外语料与 r5 §9.3"FP32 输出"差异；BF16 最终输出转换舍入模式（建议 RNE，待 review） | 验收口径与数值 profile 分歧       | 内部 ROD 已由用户确认保留（§4.2 默认绑定 W3 已确认语义）；输出转换舍入单列、RNE 为建议，待 review 并以真实指令证据入封印 profile；需 W7 增 BF16 语料并在总览决策登记 |
| 4   | 累计容量与空间不足出口                                                                   | 多程序并存上限、等待 livelock     | 多 slot＋zero-ref 显式失效释放 slot；空间等待有界、可拒绝上报（§3.2/§6）；自动淘汰/替换策略仍延期                                                                    |
| 5   | `PROGRAM_*` 消息与 W1 通道预算/credit 的具体分配                                         | 死锁/饿死分析依赖 W1              | 语义已与 W1 对齐（其 §3.2/§4.2/§5.1，字段级出入以本文为准）；仅剩物理编码与通道预算由 W1 评审                                                                        |
| 6   | 配置转移表粒度                                                                           | 过严拒合法 kernel、过松漏协议故障 | 首版模式枚举/转移/上下界已按固定 compiler 实读落地（§4.2.1）；K 尾、多归约 conv 链等粒度与 W3 联评，以 MAT-10 表示法为准                                             |
| 7   | Host 上传 HBM 程序段的建模缺口（RES-16，`HostRuntime` 未实例化）                         | 冷启动端到端口径不完整            | Python 侧补最小上传模型或明确声明不含、单独报告                                                                                                                      |
| 8   | Zvt 编码空位未核对（START/END/链声明载体）                                               | manifest 声明点可能无法落到指令位 | 上游核对后由 ISA 评审冻结；本组件不预设编码                                                                                                                          |

依赖：W0（工具链事实）→ W5 → W2（取指门控）/W3（kernel 执行）/W4（安装端口）/W7（计数与日志）；与 W1 并行定义消息语义。本文件未运行任何构建/测试/格式化（按共享合同由 总体设计 统一执行）。

[C01]: https://github.com/google-coral/coralnpu-compiler/blob/d715556c3853719e414182c0bf5f4a10f248f1d9/docs/design/matrix_codegen.md
[S08]: https://developers.google.com/coral/guides/hardware/integration
