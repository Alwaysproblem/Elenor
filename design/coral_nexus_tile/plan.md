# Coral Nexus Tile CS1 详细实施计划

> 状态：**设计已通过用户 review；按推荐方案编制的实施计划 v1**。本轮授权为列计划，不自动开始 RTL／Python 实现。
> 基线：[总体设计](Coral_Nexus_Tile_Design.md)、[W0–W7 组件设计](#work-packages)、[Tile 需求 r5＋输入早释放勘误](../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。
> 本文件保留范围、42 个原任务与设计裁决；**实际逐 commit 执行入口改为 [execution_guide.md](execution_guide.md)**（76 张 commit 卡、前置依赖、文件边界、可用验证命令与交接模板）。批准方案不等于能力已实现，参数冻结与实际验收仍须完成。

## 1. 交付范围与执行纪律

### 1.1 完成目标

最终 CS1 为 **1 个 Python Group＋4 个真实 RTL Tile**，CPU／Device、Group 调度、L2 SRAM／cache／MSHR、NoC、HBM 与外部授权服务仍为 Python。每个 RTL Tile 包含改造后的 Coral scalar UCE、隔离的 Context 状态、RVV／VME、LSU／MFE、L1 bank／cache／MSHR、Frame／Arena、程序安装、本地事件、故障／复位端点与门控。Host checker 只观察，不生成 DUT 结果或完成。

- 支持 `cs1_c1` 单 Context 退化配置与 `cs1_c2` 双 Context 配置，物理配置差异进入 manifest。**P1／PN 性能比较都用同一 `cs1_c2` RTL 与 target**：P1 仅一个 Context 活跃、PN 多个活跃；`cs1_c1` 只作退化功能回归，不用于冒充同硬件性能比较。单 RTL Tile／mixed 配置是调试阶段，不是最终替换范围。
- BF16 张量输入／输出，FP32 内部累加、seed／drain／scratch；整数控制、地址和 i32 索引保留。VME 内部保留 Coral ROD 与实际有限宽语义，最终 BF16 转换采用 RNE。
- VME N=1 实现；状态、接口、实例路由及指纹按 N≥1 设计。N>1 RTL、共享 AM／阵列组、完整 RVV dialect／通用 LLVM 后端、CS2／CS3、FPGA、流片及 PPA 不自动开工。
- 真实 kernel 覆盖声明的强制验收负载；不能把 softmax／norm／gather 等范围缩为一个 matmul smoke，也不能用“unsupported”批量排除应实现条目。
- 输入早释放保持已勘误合同：Task 级整体信号、Group `all_tasks` 聚合、纯读 pin 解除、按 buffer 的 owner release、完整 backing 的物理 final-free。**不新增同 Task 按 buffer／范围分批通知或单 backing 子范围回收。**

### 1.2 当前事实与计划目标分开

本轮目录核对：仓库当前**没有 `hardware/` 与 `.devcontainer/`**；旧设计关于 Gemmini／BOA 空目录的文字属于先前快照。下文这些目录、脚本、模块和配置都是待创建目标，不代表已实现。不得恢复旧 Gemmini／BOA 代码作为 Coral 依赖。

当前 `pipeline_validator/compiled_program.py` 与 `execution_verifier.py` 接受 schema `3`／compiler ABI `v3`；`hardware_config.yaml` 的 schema `2` 属另一版本域。新真实映像包和 RTL target 有独立版本域，不能据旧 README 或其他工作树版本改写当前事实。

所有实现任务开始前：核对工作树现状与 source lock，读取对应设计和计划；用户的新改动保留。修改 exported API 先查引用；有支持代码智能的 LSP 时使用 LSP。删除 obsolete 路径与迁移调用方是同一任务，不添加临时兼容别名或无声降阶路径。

<a id="decisions"></a>

## 2. 推荐方案落实与参数冻结

### 2.1 已选方案登记

标记“沿用推荐”指原设计已推荐；“保守落定”指原设计并列方案中按用户授权选取最小风险实现，不声称用户逐项指定了具体技术细节。

| 决策          | 本计划采用                                                                                      | 依据／取舍                                                                                                                                          | 主责任务            |
| ------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| 工程组织      | `hardware/coral_nexus/` 独立 Bazel＋Bzlmod workspace                                            | 沿用推荐；validator 不纳入 Bazel 依赖图                                                                                                             | W0-01、W0-02        |
| 上游管理      | pristine submodule＋编号补丁序列；在 `out/` 工作树应用补丁                                      | 沿用推荐 A；不用未锁分支或私有 fork 作为隐含依赖                                                                                                    | W0-01               |
| 环境          | Debian trixie digest、不可变 apt snapshot、依赖版本／SHA 锁、离线种子                           | 复刻上游环境叠加隔离的 validator Python3.11；snapshot 为保守落定，缓存仅用于离线分发，不作为第二套包源                                              | W0-02、W0-05        |
| dev container | `.devcontainer/coral_nexus/devcontainer.json` 引用同一镜像                                      | 保守落定；不维护第二套 Dockerfile                                                                                                                   | W0-02               |
| RV32 工具链   | 固定预构建 GCC 归档，Bazel 9.1.0                                                                | 沿用上游实际默认；源码重建为审计备选，不做自动 fallback，也不是本轮双工具链交付                                                                     | W0-01～03           |
| 联仿载体      | 单进程、单线程、Verilator C ABI＋ctypes                                                         | 沿用推荐；C++ 不回调 Python、不执行 P4、不自推进另一时钟                                                                                            | W1-01～03           |
| RF／ROB       | GPR／FPR／VRF Context 独立存储 bank；scalar／RVV ROB 静态私有分区                               | 沿用推荐；无共享全局回收 tail，不实现动态 free-list 借容量                                                                                          | W2-02、W3-02        |
| 发射          | 保留 Coral 4 lane，单 Context 连续前缀；eligible head RR                                        | 保守沿用上游宽度，不做未证明 aging；服务界由队列和返回预算证明                                                                                      | W2-03               |
| CSR／flags    | dispatch 配置快照；架构态按本 Context 退休提交；flags delta 随 ROB 条目有序合并                 | 沿用主方案；CSR 读／pending-config 局部互锁，无跨 Context 全核 drain，无转发优化                                                                    | W2-04               |
| 声明载体      | 受控 helper＋每 Context 伴随 CSR 声明                                                           | 保守落定；不占未经核实的 Zvt 保留位。START／链位置 pending metadata 仅在目标指令 accepted 时原子消费；END 显式提交，具体 CSR 编码由前置冻结任务产出 | W1-01、W3-01、W5-01 |
| 数值          | VME 内部有限宽／jam／ROD／溢出不对称及 OF/NV；输出 `frm=RNE`                                    | 用户选择推荐方案已批准 RNE 输出；不把内部换成通用 FP32 RNE FMA                                                                                      | W3-01、W7-02        |
| 数值拒绝      | `frm=DYN` 在构建、Loader 拒绝，运行期仍检查；逐比特＋flags                                      | 主方案；不放宽 ULP 迎合 DUT，数学高精度仅诊断                                                                                                       | W3-01、W5-01、W7-02 |
| MSHR          | token 保留至已登记 waiters 获得可靠响应／数据并结清引用                                         | 沿用保守方案；cache 写入不等于数据永久可用，不做早释放 token 优化                                                                                   | W4-02               |
| 程序驻留      | 多固定预链接 slot，至少容纳两个不重叠真实映像；zero-ref 后显式失效                              | 沿用推荐；Loader 不重定位、不自动淘汰，具体 slot 容量按真实 ELF 冻结                                                                                | W5-02、W5-04        |
| 真实映像包    | 独立 v1 `CoralTilePackage`，包裹 schema3 控制产物与真实映像／binding／资源／target              | 采用已设计的简版链；保留 Python-only 产物不等于 RTL 可以无 binding fallback                                                                         | W5-01、W5-05        |
| 事件与退休    | 每 Context frontier 事件槽；check-register 同周期完成优先；可靠 completion 后保守复用 Context   | 沿用推荐；WAIT／WAITALL，不增加 WAIT_ANY；输入 pin 早解除独立于退休                                                                                 | W6-01、W6-02        |
| 取消／复位    | owner 标记与真实排空；Group fail-stop，Group 为复位顺序／watchdog 权威                          | 不清掉无关 Context 的已接受工作，不凭 generation 或 timeout 撤销外写；隔离失败 poison                                                               | W2-04、W4-06、W6-03 |
| ABA／回绕     | 停止受影响身份复用，排空到包括 ACK／响应队列的旧义务消失后才回绕                                | 保守落定；不以“大位宽大概够用”代替证明                                                                                                              | W1-01、W6-03        |
| 日志／重放    | 小 debug FIFO、默认无 HBM 历史；Host 有界完整窗口＋封窗；单侧已知故障重放                       | 沿用推荐；SHA-256 标识 blob／合同，不以 hash 代替 bytes；缺口明确失败                                                                               | W6-04、W7-01、W7-05 |
| 性能          | P1／PN 均用同一 `cs1_c2` RTL/target、同总工作量，仅活跃 Context 数不同；P0 原核比较范围差异单列 | `cs1_c1` 仅退化功能回归；PS、N>1 利用率估算与 PPA 不在本轮，不承诺速度或 Fmax                                                                       | W7-05               |

### 2.2 仍须产生证据的冻结任务

这些不是待用户重新挑方案的事项，而是实施的前置产物。任务可在明确输入下完成；产物缺失则阻断消费者，不造默认值、不提交占位实现。

| 冻结对象                               | 输入与产生方法                                                                                  | 必须输出／检查                                                                                                                                | Owner／最晚消费者                       |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| 完整工具依赖／镜像身份                 | 固定 SHA 的 Bazel、Maven、Python、GCC／Verilator 锁递归解析；解析 OCI digest 与 snapshot 包清单 | 每项版本、URL、SHA、许可、离线种子；清缓存构建与断网执行分别验证                                                                              | W0-01／02；P0 构建前                    |
| 消息布局、长度、credit／事务／终态预算 | W1 逻辑字段与各包最大在途约束；逐通道等待图                                                     | 单一机器可读 schema、生成器、跨语言 codec、容量不等式与满队列进展证明                                                                         | W1-01；W1-03／所有端点前                |
| CSR／helper ABI                        | 固定上游 CSR／译码空间、RV32 ABI、编译器约束；START／链标记原子消费与 END 关闭规则              | 可用地址与权限表、无冲突译码测试、C／asm helpers、失配拒绝；不预填硅片编码                                                                    | W1-01＋W3-01＋W5-01；真实 kernel 构建前 |
| Context／ROB／LSU 容量及回绕           | C1／C2、四 lane、最大分片／向量展开、回复存活上界                                               | 参数约束、每类 credit 上界、生成 assert、小位宽回绕测试；无负移位／零宽信号                                                                   | W0-04＋W2-01＋W4-01；对应 RTL 前        |
| 地址／memory map                       | RV32 Context-local Frame／程序窗口；外侧 L2 view 或 HBM binding＋offset 经 MFE descriptor 求址  | 采用受检窗口＋descriptor 全宽地址，普通 kernel 不持自由全局指针；按目标容量冻结外侧宽度，检查加法溢出／跨窗口／权限／generation，禁止截断高位 | W1-01＋W4-01＋W5-01；LSU／装载前        |
| numeric profile                        | W3 固定源码归纳的独立精确规范，W7 整数位运算 oracle                                             | 有限宽、jam、归约序、特殊值、mask、flags、输出 RNE 全定义，ID／SHA 入 target；先 experimental 封印供测试，验证后才 supported                  | W3-01＋W7-02；W3-04／W5-03 前           |
| 程序 slot／stack／scratch              | 实际链接 map／ELF、受控调用图、最坏 shape 与并发包络                                            | 至少两个真实程序可共存、mutable 区不别名、累计容量与静态资源上界；无动态峰值冒充最坏界                                                        | W5-02／04＋W0-04；T-30／T-58 前         |
| Profile／cache／MSHR 参数              | 当前 `hardware_config.yaml` 的实验 Profile＋W4 真实端口计费                                     | 单一生成配置、manifest 一致、合法组合矩阵与压满进展；标明模型值非 SRAM/PPA 冻结值                                                             | W4-01＋W0-04；W4-02／05 前              |
| 故障／debug／日志预算                  | 每类已接受工作的终态义务、响应上界、Host quota                                                  | 单 watchdog、终态保留容量、debug 溢出不阻塞功能、封窗完整性                                                                                   | W6-01＋W7-01；W6-03／04 前              |
| 性能通过门                             | 同工作量 P0/P1/PN，统计起止点与合法环境                                                         | 测量前登记比较配置／工作集／回归预算；功能失败时不发布性能结论                                                                                | W7-05；性能跑分前                       |

## 3. 单一状态 owner 与共享文件

### 3.1 代码归属

| 包  | 独占主目录／改造面（待创建者已标注）                                                                                                        | 共享文件处理                                                                                                                                                                                       |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| W0  | 【新】`hardware/coral_nexus/{env,configs,scripts,upstream,patches}` 的框架与 workspace、来源锁、`cn.sh`；【新】`.devcontainer/coral_nexus/` | patch 内容由功能 owner 提供；W0 只固定应用顺序和来源                                                                                                                                               |
| W1  | 【新】`protocol/` 机器可读权威、`sim/` C ABI；【新】`pipeline_validator/cosim/` driver／adapter                                             | 唯一集成 `simulator.py`、`tile_group.py`、`group_scheduler.py` 的 gate 事件唤醒、`device.py`、`runtime/group_port.py`、`cli.py` 和共享 config；`__main__.py` 只是 shim，不另建 parser 或 scheduler |
| W2  | 【新】scalar／Context RTL 与上游 Scala 补丁                                                                                                 | `Core.scala`、`SCore.scala`、`Parameters.scala`、`CoreAxi.scala` 单一集成 owner；其他包提交明确接口改动请求                                                                                        |
| W3  | 【新】RVV／VME RTL、SV frontend/backend/ROB/VRF/Zvt 补丁                                                                                    | 不与 W2 同时改顶层；数值规范主责 W3，oracle 代码归 W7                                                                                                                                              |
| W4  | 【新】`rtl/memory/`、LSU 补丁；现 `memory/transfer.py`／Arena／Profile／page pool 外侧服务必要改造                                          | `tile_group.py` 接线请求交 W1；程序 SRAM 物理端口归 W4，安装 FSM 归 W5                                                                                                                             |
| W5  | 【新】`kernels/`、安装 RTL；现 `package/`、`runtime/program_table.py`、compiler／loader 真实绑定链                                          | helper 编码消费 W1 schema，不能另写一份；共享 config 扩展交 W1                                                                                                                                     |
| W6  | 【新】event／fault／reset／clockgate／debug RTL；现相应 runtime FSM 适配                                                                    | Group 顺序经 W1 集成；CoreAxi 门控变更交 W2，外存 debug 事务经 W4                                                                                                                                  |
| W7  | 【新】`verification/` oracle／checker／scenarios／trace／replay；对应 tests／`report.py`／展示导出                                          | W1 提供只读 RTL 资源镜像，W7 负责 report 消费；C++ 采样由 W1，硬件插桩由功能 owner；checker 不写功能状态                                                                                           |

所有包共用一份来源锁、协议 schema 与 target manifest。子任务可以并行写各自文件，不可同时改同一上游文件或 generated header。跨包接口变更提交“字段／producer／consumer／失败行为／受影响 T-ID”的差异，集成 owner 一次迁移全部调用方。

### 3.2 接口产物与消费顺序

| 合同产物                      | Producer  | 必须消费的组件                                                | 不可丢失的语义                                                                      |
| ----------------------------- | --------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| source／toolchain lock        | W0-01／02 | 所有构建、W7 manifest                                         | 固定 SHA＋补丁顺序，不依赖宿主未知环境                                              |
| boundary ABI／C API           | W1-01     | W1 driver、W2–W6 RTL、W7 capture                              | TaskKey／ContextKey／message_seq／op_seq 分离；P1 唯一接受；P3 不接收新握手         |
| Context／native-op 合同       | W2-01     | W3 uop、W4 LSU、W6 completion                                 | PC 和寄存器编号都不是跨 Context 完成键；partial accept 原子                         |
| region／chain／numeric 合同   | W3-01     | W2 frontend、W5 helpers／package、W7 oracle                   | START 前无 mt 副作用；FIRST 已 seed；FINAL 无 auto-drain；可靠 region_done 后才转交 |
| memory／lease／Profile 合同   | W4-01     | W1 bridge、W2 LSU、W3 drain、W5 install、W6 cancel            | 真实字节读写；各 leg 终态；MSHR provenance；页授权原子 acquire；hold 不超卖         |
| package／kernel binding 合同  | W5-01     | compiler／Loader、W1 launch、W2 entry、W3 config、W7 manifest | image 与完整执行合同一起封印；实际视图显式绑定，不猜 operand                        |
| event／fault／reset 合同      | W6-01     | W1 Group、W2 wait、W3 region、W4 accepted、W5 invalidate      | 事件持久；完成优先；取消非删除；失败隔离 poison                                     |
| capture／oracle／verdict 合同 | W7-01     | W0 CLI、W1 capture、各功能包                                  | 完整窗口与封窗；FAIL／BLOCKED 不冒充完成；整数 cycle/phase/seq                      |

**联合冻结不是循环依赖**：W1–W7 的 `-01` 任务读取彼此草案接口并产出自己负责的完整合同，不要求另一 `-01` 先 DONE。联合冻结 gate 检查全部产物及 producer/consumer 一致后形成同一合同 hash；依赖这些接口的实现任务只能消费冻结版本。W0-04 汇总该 hash 与有来源的参数，不允许 `-01` 再反向等待 W0-04 最终配置。

### 3.3 跨包产物命名与 C ABI 边界

以下均为待创建目标，详细布局由 W1-01 生成，其他包只能引用同一权威：

- `hardware/coral_nexus/protocol/cn_protocol_v1.yaml`：逻辑字段、消息种类、分段、身份、错误和版本；`channel_budgets.yaml`／`lifetime_bounds.yaml` 分别登记容量与寿命证明。
- `protocol/gen_protocol.py`／`derive_widths.py`／`freeze_check.py`：生成并核查 `protocol/generated/{cn_messages.h,cn_offsets.svh,cn_layout.json}` 与 `pipeline_validator/cosim/wire.py`。各包通过 `protocol/field_requests/wN.yaml` 与各自合同片段提供字段；共同冻结只在全部必需产物齐全且一致时写不可变 `PROTOCOL_FREEZE.json`。后续修改产生新 hash，不向已冻结文件追加、不以推测片段代替缺失合同。
- `pipeline_validator/cosim/tile_port.py` 定义 Python／RTL 共用的 TilePort 行为接口；RTL adapter 不把 Python L1／engine 放回执行路径。
- `hardware/coral_nexus/sim/cabi/{cn_abi.h,cn_abi.cc,BUILD.bazel}` 生成 `libcn_cosim.so`，导出 `bp_abi_info`、`bp_begin`、`bp_layout`、`bp_drive`、`bp_tick`、`bp_collect`、`bp_status`、`bp_destroy`。P0 由 `bp_drive` 驱动，`bp_tick` 只执行 P1–P3，`bp_collect` 在 P3 后、P4 前取回；初始化预分配 IO／返回队列，steady-state bridge 不分配堆对象，仅复制必要有效 beat／P1 快照，不每 tick 无条件拷贝整个容量。
- C++ 不保存 Python 指针越过调用，不回调 Python；一个实例管理本次选择的全部 RTL Tile，共用一个沿。崩溃且无有效封窗的运行是 INCOMPLETE，不能由缺失 cleanup 被当作已排空。
- `message_seq` 是每 `(tile, channel, direction)` 的传输序列，`op_seq` 是 Context 内父指令，`txn_id` 是事务，`subop_id` 是展开子操作。现 `GridInstanceId` 已含 `launch_generation`；逻辑上重复表达的 `root_launch_generation` 必须与其相等，wire 可只保存一份，不可出现两个相互独立的 launch 代际。
- `csr_name_map.yaml` 经固定上游 CSR 占用表校验；禁止默认 `0x7c0` 可用（Coral 已占 MCONTEXT0..7）。分配排序、保留范围、访问级和生成 helper 都入冻结 hash；本计划不指定新 CSR 数字。
- 若所选 source Task 使用既有 Stream Queue，`STREAM_*` 作为 MEMORY 下独立逻辑子队列／预算消息化，Group StreamQueue 仍唯一 owner；不新增第六物理通道，不把流 credit 混入数据 flit 或 input_release。未使用 Stream 的语料不据此增加强制流依赖。

## 4. 优先级、阶段与依赖图

P0 为正确性与可运行闭环，P1 为全部强制场景集成及复现，P2 只用于测量／维护交付；P1/P2 不等于可省略。此处阶段用 **CN-G0～CN-G5**，避免与需求 §20.2 的 G0～G6 混同；CLI `--stage Gk` 指本计划 CN-Gk。

```text
CN-G0  来源、环境、原核基线
  |
  v
CN-G1  联合合同冻结、目标生成
  |
  +--> 并行实现与单元验证
  |      标量/访存、RVV/VME、内存/授权/Profile
  |      事件/故障、kernel/驻留、独立 oracle/checker
  |
  +--> CN-G2  双 Context 标量闭环
           |
           v
         CN-G3  RVV/VME 数值与区域闭环
           |
           v
         CN-G4  四 RTL Tile 完整 CS1（汇合全部功能分支）
           |
           v
         CN-G5  全部验收、重放、性能与复现
```

| 阶段  | 可观察出口；不是“文件已创建”                                                                                                                                                  | 需求阶段映射                                                              |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| CN-G0 | 固定来源与容器运行原核真实 ELF；基础 scalar／load-store 的已知输出或上游独立参考相符，manifest 可定位所有输入。完整 VME 数值资格由 W7-02／05 后续闭合，不作为环境门的循环前置 | 需求 G0                                                                   |
| CN-G1 | 全部 `-01` 合同一致、跨语言 encode/decode/错误拒绝具备可运行验证、合法目标可生成；不宣称已有多 Context RTL                                                                    | 需求 G1 的合同部分                                                        |
| CN-G2 | C1/C2 真实 scalar ELF 从正式安装端口执行；同 PC store、独立退休、异步访存、WAIT／取消基本链实际通过，零资源泄漏                                                               | 需求 G1 剩余功能＋G2                                                      |
| CN-G3 | BF16 RVV／VME kernel、显式区域／链、最终输出 RNE、独立逐比特结果与 flags、T-61～63／T-68 N1 通过                                                                              | 需求 G3                                                                   |
| CN-G4 | 四个 RTL Tile，无被替换 Tile 的 Python 执行影子；完整 MFE／授权／Profile／故障／debug／冷暖／输入早释放闭环                                                                   | 需求 G4 的实现部分                                                        |
| CN-G5 | 全部 CS1 适用 mandatory 子项 PASS、已知故障完整窗口单侧重放、P0/P1/PN 证据、清缓存构建与断网复跑、维护交付齐                                                                  | 需求 G4 出口＋G5 中本轮 P1/PN 度量；不自动启动完整后端、PS、CS2 或需求 G6 |

<a id="work-packages"></a>

## 5. 子任务清单与并行窗口

详细文件中的输入、逐步改造、正负例和命令是每项验收依据。下表列全部 42 项；`冻结门` 表示 §3.2 的共同合同产物，而非新增实现占位任务。

交给实现模型时先读[按 commit 执行 guide](execution_guide.md)及其[原任务关闭映射](execution_guide.md#8-原-42-个任务到-commit-的关闭映射)。本表是工作包划分，不是直接提交顺序：一个 Wn-xx 可跨多个 commit，合同发布、模块验证与真实端到端验收分别关闭；不得以其中一张卡完成代替整任务完成。

| 包／详细计划                                                      | 任务 ID 与交付                                                                                                                                                            | 硬前置／并行窗口                                                                                                        |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [W0 环境](implementation/W0_Baseline_Environment_Plan.md)         | W0-01 来源锁与补丁工作树；W0-02 容器／离线种子／CLI；W0-03 原核 P0；W0-04 支持 target／配置生成；W0-05 双复现／许可维护                                                   | 01→02→03；04 在冻结门后；05 初期准备、最终对完整产物执行                                                                |
| [W1 边界联仿](implementation/W1_Boundary_Cosimulation_Plan.md)    | W1-01 消息／身份／预算／编码；W1-02 Python TilePort／外部服务；W1-03 C ABI／五相位；W1-04 准入／退休／早释放／gate；W1-05 四 Tile 集成                                    | 01 读固定来源；02/03 在冻结门后可并行；04 需要真实本地资源／事件端点的对应子合同；05 等完整 Tile                        |
| [W2 Context/UCE](implementation/W2_Context_UCE_Plan.md)           | W2-01 owner／指令／配置接口；W2-02 RF／CSR／scoreboard／ROB；W2-03 fetch／issue／partial accept；W2-04 flags／取消／退休；W2-05 C1/C2 闭环                                | 01 参加冻结门；02–04 按共享状态依赖实施；05 等 W4 基础访存、W5 真实映像／安装、W6 事件及 W7 标量参考                    |
| [W3 RVV/VME](implementation/W3_RVV_VME_Plan.md)                   | W3-01 数值／声明／模式合同；W3-02 RVV Context／VRF／ROB；W3-03 区域／链／实例路由；W3-04 BF16 执行／收窄／配置／取消；W3-05 数值与区域集成                                | 01 参加冻结门；02 与 W2/W4 并行，顶层按 owner 合并；03/04 接受合同驱动单元验证；05 等真实 kernel、内存端点与独立 oracle |
| [W4 内存](implementation/W4_Memory_Subsystem_Plan.md)             | W4-01 memory／Frame／Arena／地址；W4-02 L1/cache/MSHR/program 端口；W4-03 LSU 多在途；W4-04 MFE/indexed/mask/授权；W4-05 Profile／维护／hold-retire；W4-06 取消／资源压力 | 01 参加冻结门；02 供 03/04/05 使用；基础访存与完整 indexed 可分阶段验证，但 W4-04 未全部通过不得 DONE                   |
| [W5 映像与运行时](implementation/W5_Kernel_Image_Runtime_Plan.md) | W5-01 binding/package/helper ABI；W5-02 基础 C/asm kernel；W5-03 RVV/VME 计算库与 BF16 映射；W5-04 安装／驻留／多 slot；W5-05 编译到执行全链                              | 01 参加冻结门；02/04 可在环境与冻结门后并行；03 依赖数值合同；05 需要真实 RTL 与 oracle，不以封印成功代运行             |
| [W6 事件故障](implementation/W6_Events_Fault_Reset_Plan.md)       | W6-01 event／terminal／reset 接口；W6-02 WAIT/phase/可靠终态；W6-03 fault/cancel/reset/watchdog；W6-04 clockgate/debug；W6-05 竞争／故障集成                              | 01 参加冻结门；02/03 的独立端点先行，不能等系统末尾才做错误处理；05 等各包实际 accepted/drain 计数                      |
| [W7 验证追踪](implementation/W7_Verification_Trace_Plan.md)       | W7-01 日志／manifest／68 T／C1–C7；W7-02 ISA／数值 oracle；W7-03 checker 负向检出；W7-04 CS1 定向验收；W7-05 重放／性能；W7-06 最终出口                                   | 01 参加冻结门；02/03 与 RTL 并行；04 分阶段跑真实集成，最终四 RTL；05/06 等完整支持矩阵与窗口证据                       |

每个任务可用独立有界单元协议驱动器验证自身机制，但驱动器不是最终 RTL、不得在 `--backend rtl` 下返回虚假计算／完成。Task 是否完成由其全部输入输出与验收判据决定；不要把“接口已声明”标成 Wn-xx 完成。

## 6. 必须贯通的四条路径

### 6.1 原生执行到可靠退休

`真实 ELF → PROGRAM 安装 → accepted＋READY＋Frame/profile 门 → per-Context fetch/issue → 完整 owner 的结果/flags → EXEC_DRAINED → RETIRE_READY → Group RETIRE → 本地资源回收 → 可靠 TASK_COMPLETION → Group 清理剩余 pin/R/Route`。

检查：共享程序并发同 PC、不共享 mutable 数据；已接受工作返回不依赖其 Context 再被调度；Context 不能凭 RET、WFI 或程序最后 PC 直接复用；程序驻留引用减少不卸载 text。

### 6.2 输入早释放而原计算继续

`Task 全部 L2 load await → input_released → Grid all_tasks → 解除尚存纯读 pin → Group 按 buffer release → backing 全引用闭合 → L2 final-free/容量唤醒`。

检查：此链可早于 Task completion；R／L1／Frame／程序引用继续持有；未完成其他消费者时不回收；安全退休只清理尚存 pin。必须用真实输入字节证明源空间复用后 L1 staging 与计算结果仍正确，模型 trace 不能代替此 RTL 检查。

### 6.3 VME 区域与受控输出

`伴随 START 原子绑定首条 mt 指令 → grant/park → seed → 每条 matmul 链声明 → FINAL 真实完成 → 显式 drain/RNE/store → END → output_visible＋旧访问闭合 → state_free → 可靠 region_done → 转交`。

检查：等待者不占共享队首、owner 的 drain 可推进；同 Context 下一 START 等旧 region_done；FINAL 不触发隐式 drain；旧代返回不写新 owner；单步链 FIRST 与 FINAL 同时成立。

### 6.4 取消、故障与重放

`停止新 issue → 保存 first-fault 与 owner → 已接受工作真实完成或确认隔离 → 本地及外侧引用结算 → 唯一失败终态 → 按 Group 复位相位 ACK → 驻留失效/代际推进`。

检查：timeout 不视作隔离证明；不可撤销外写纳入终态；隔离失败 poison 不继续分配；DEBUG 满不堵功能完成。原始完整窗口包含初态、bytes、actual handshake 与封窗边界，能重放已知故障；截断窗口拒绝“可完整重放”。

## 7. 命令与证据接口

### 7.1 未来统一入口

以下命令**当前不存在**，由 W0-02 创建宿主 wrapper，各包实现所属子命令。所有配置和场景路径相对仓库根，输出写 `out/coral_nexus/<run_id>/`；路径不存在、hash 失配、配置不支持、协议错误须非零退出，不自动切 Python timing 模型。

```bash
bash hardware/coral_nexus/scripts/cn.sh env build
bash hardware/coral_nexus/scripts/cn.sh env verify
bash hardware/coral_nexus/scripts/cn.sh sources verify
bash hardware/coral_nexus/scripts/cn.sh baseline \
  --config hardware/coral_nexus/configs/p0_coral.yaml --out out/coral_nexus/p0
bash hardware/coral_nexus/scripts/cn.sh build \
  --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/build-c2
bash hardware/coral_nexus/scripts/cn.sh unit --suite w2 \
  --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/w2
bash hardware/coral_nexus/scripts/cn.sh package \
  --binding hardware/coral_nexus/kernels/bindings/cs1_reference.yaml \
  --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/package
bash hardware/coral_nexus/scripts/cn.sh cosim \
  --scenario hardware/coral_nexus/verification/scenarios/cs1_acceptance.yaml \
  --backend rtl --rtl-tile-mask 0xf \
  --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/cs1
bash hardware/coral_nexus/scripts/cn.sh verify --manifest out/coral_nexus/cs1/manifest.json
bash hardware/coral_nexus/scripts/cn.sh replay \
  --bundle out/coral_nexus/cs1/replay --out out/coral_nexus/replay
bash hardware/coral_nexus/scripts/cn.sh bench \
  --matrix hardware/coral_nexus/verification/scenarios/performance.yaml --out out/coral_nexus/perf
bash hardware/coral_nexus/scripts/cn.sh gate --stage G5 --out out/coral_nexus/gate-g5
```

| 命令族          | 实现责任                | 最少输出与真实出口                                                                  |
| --------------- | ----------------------- | ----------------------------------------------------------------------------------- |
| env／sources    | W0-01／02／05           | source/toolchain/image 锁与 verify verdict；离线种子可导出                          |
| baseline／build | W0-03／04、各 RTL owner | ELF／生成 RTL／实际 target 配置／共享库及构建 manifest；baseline 还须运行并核对输出 |
| unit --suite wN | 各功能 owner＋W7        | 可复现具体正负例、断言和 bytes/状态 verdict；不只返回编译成功                       |
| package         | W5-01～05               | 真实 image、sealed package、资源/配置/数值约束、明确拒绝报告                        |
| cosim           | W1-03～05＋W5/W7        | 真实握手日志、输出 bytes、终态/资源快照、manifest、退出原因；崩溃为 INCOMPLETE      |
| verify／replay  | W7-03～06               | C1–C7 报告、逐 T 子项及证据引用、窗口完整性与重放等价判据                           |
| bench           | W7-05                   | 预登记测量条件、P0/P1/PN 结果与瓶颈；数值/生命周期失败则不生成通过结论              |
| gate            | W7-06，W0 集成入口      | 对指定阶段必需证据重验 hash 与 verdict；缺文件/FAIL/BLOCKED 必须非零                |

未来子任务中更细的 suite、case、target 名须在各包实现时落到真实 parser/构建目标；不得把文档中的命令留作无实现的“说明性入口”。

每条未来命令从哪个 commit 起可用，见[执行 guide §4.2](execution_guide.md#42-工具-producer-表防止验证依赖未来工具)。执行模型不能先调用尚未交付的 `cn.sh`、完整 Tile top、package、oracle 或最终 gate，再以缺工具为由提交不可运行的占位实现。

P0 与 P1／PN 配置不可混用：W0-01／02 提供 `configs/p0_coral.yaml`（`kind: p0_upstream`），W0-03 的 `baseline` 只消费未改造的上游来源与原生 kernel，拒绝 `cs1_target` 配置。`cs1_c1.yaml`／`cs1_c2.yaml`（`kind: cs1_target`）由 W0-04 在联合冻结后生成，供 build／unit／package／cosim／bench 使用；P0 不等待最终 CS1 配置。P0 基础已知输出 smoke 可先闭合环境门；完整 VME 数值与性能对比须在 W7-02 oracle 就绪后按 W7-05 补齐，不提前声称数值验证通过。

### 7.2 当前可执行基线与一次性验证

当前可运行的 Python 生命周期/字节场景见 [设计验证记录 §2](Design_Validation.md)。实现合入时遵守仓库现有门，先 pre-commit，再 pytest；这些命令不是本轮已执行声明：

```bash
conda run -n elenor-validator pre-commit run -a
conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -v
```

每个代码变更先复现对应 consumer-visible 问题或验证新行为；功能正负例、边界、状态转换可保留为回归。单纯 wiring、源码字符串、mock 回声和“不抛异常”不作为永久测试。各包开发阶段不反复运行全库格式化；共享变更整合后由一个 owner 执行完整门，并另外运行真实 CLI／C ABI／RTL 场景观察结果。

## 8. 给实现 agent 的逐项执行手册

1. 选择一个具备前置证据的 Wn-xx，读总计划 §2–3、对应设计和分包任务。确认目标是现存文件还是待创建目标，先列出会改的 API 与消费者。
2. 读取 source lock、合同 hash、target、支持状态；没有冻结接口时完成对应 `-01` 产物，不在代码里临时发明字段／opcode／默认容量。
3. 在一个 owner 的文件范围内实现完整状态转换。normal／backpressure／cancel／fault／stale 必须同时闭合；每个 accepted 请求都能追到终态及信用/引用归还。
4. 保留 C1 单 Context 退化配置；每次新增 Context/instance 维度覆盖生产者、队列、uop、写回、flags、错误、取消、退休及统计，不只在顶层加 tag。
5. 用任务中的最小实例跑真实程序，核对实际输出 bytes 或精确状态；负例须能因目标合同被拒或报错，而不是因缺输入/解析失败碰巧失败。
6. 修改配置/协议/数值语义后更新对应版本域与所有调用方、生成文件、fixture、文档；不增加过时字段别名、不回退软件计算。Python-only 模型可以保留，但同一个 RTL Tile 只能有一套执行 owner。
7. 产出完整证据：命令、source/config/image/package hashes、seed、输入与期望/实际输出、关键 trace/断言、资源收支、退出 verdict。必要引用文件全部可取回，不能只写路径名。
8. 任务全部验收通过才 DONE；真实阻断保留可复现证据及缺失产物 owner。不得将失败标为不适用、将超时标为通过、只交构建 scaffold 或把范围内工作写成后续。

## 9. 最终验收与不适用项

- [W7 实施计划](implementation/W7_Verification_Trace_Plan.md) 的 T-01～T-68 映射必须覆盖每个适用子项；至少包含 C1/C2、四 RTL Tile、真实 BF16 输出、indexed 内存字节、早释放、冷热装载、多程序并存、取消与完整窗口已知故障重放。
- T-59／T-60 属 CS2／CS3；T-68 与 T-28 中明确需要 N≥2 的子项延期。N=1 区域/链的强制部分照常验收。任何包含 L2 的用例仍要验其 CS1 的 Python 外侧协作部分，不得整行跳过。
- 性能结果仅在功能与资源闭合通过后采集。P1／PN 必须同一 `cs1_c2` RTL hash／物理配置、同 workload／总工作量／input／numeric profile／外存模型及统计起止，仅活跃 Context 数不同；不能以 C1 与 C2 的物理差异冒充调度收益。不把 `clock_mhz` 当 Fmax，不混 debug 扰动，不给多 VME 利用率估算。
- 完成后保留源码／补丁／许可、构建与运行锁、ABI 变更规则、配置合法性、操作手册和复现证据。每项的文档更新随其实现提交完成，不以一个最终“补文档”任务替代。
- 下一步实际开工入口为 W0-01／W0-02 与各包 `-01` 合同准备。当前文件交付不自动执行这些命令，不修改宿主或创建硬件实现脚手架。

## 10. 本轮计划检查记录

本节仅记录计划编制时的实际检查，不将未来命令当作已有能力：

| 本轮执行项                                                          | 观测                                                                                                                                           | 证明边界                                                                                   |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 当前源码／布局核对                                                  | schema3／ABI v3；没有 `hardware/`／`.devcontainer/`；工具目标来自固定上游设计证据                                                              | hardware 目标均待创建，不复活旧工程                                                        |
| 固定上游 Bazel 目标核对                                             | `coralflow_test` 按后端生成 `_verilator` 后缀；P0 用 `core_mini_axi_sim_cocotb_verilator`，VME 用 `vme_test_verilator`／`VmeCoreMiniAxi` model | 已同步修正 W0 设计中的旧 README 无后缀名；源码已核对，Bazel 构建／query 仍属于 W0 实施验收 |
| `docker version`                                                    | client/server 均 `28.3.0`                                                                                                                      | 仅证明已有 daemon 可达；未 build/pull/run 硬件镜像                                         |
| `l2_admission_wait.mlir`，full_memory，双 Context                   | 完成，21686 cycles，3 项检查全过；A 输入 131072 B 在 cycle 3313 回收并被 B 复用；A 四个 Task 均在此后退休                                      | Python 生命周期与容量复用，不是数值或 RTL                                                  |
| indexed gather/scatter，ByteStore，compile→serialize→parse→load→run | 完成，1710 cycles；OUT row1=`[10,11,12,13]`、row2=`[40,41,42,43]`；row0/3 未初始化；gather/scatter 各48 B，credit 与 L2闭合检查通过            | 实际 indexed 搬运字节，不证明 RVV/VME 算术；Context 容量2不等于本场景双 Context并发        |
| 早释放 trace 的 Perfetto SQL                                        | 无非零 error/data_loss；slice370、counter值2937与 raw JSON 对账一致，release_cycle字段一致                                                     | 未审完整 flow，不宣称完整边界重放                                                          |

早释放命令与 [设计验证记录 §2.4](Design_Validation.md#24-输入早释放与容量复用-smoke勘误后)相同，本次产物前缀为 `/tmp/coral-plan-baseline`；compiled artifact hash=`a9f6d524d8cce5fb95e80795fa05ca9a8befbd3f69de7bfe6dc93855cbe51a76`。indexed 输入与脚本见该记录 §2.3，本轮再次执行并核对实际字节。

本轮没有执行 pytest／全库 pre-commit、Coral 构建、RTL 仿真或 PPA；没有安装依赖、改实现代码／fixture／系统配置。参数冻结与 RTL 能力仍是上述任务的实施验收义务。

分包计划验证（2026-11-05 收口）：8 份分包计划（W0–W7，共 42 个任务 ID）与本计划通过统一验证——Prettier 格式化且幂等（二次运行逐字节不变）；全部表格列数一致、代码围栏闭合、相对链接与锚点有效；任务 ID 引用闭合（无未定义引用）；W7 场景表 T-01～T-68 连续完整；无残留非文档化的内部引用。修复了分包草稿中的表格断列／下划线转义问题与 W3 尾部重复段落。以上为文档结构验证，不代表任何未来命令已可执行。
