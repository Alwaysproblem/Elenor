# 软件管理 AM 与 BOA Matmul IR 改造计划（硬件实现前置）

- 状态：**已按本计划完成硬件前置软件 gate；§3.4 源级分支/重试 MLIR 仍不在本次范围。实施记录见 [实施报告](BOA_Software_Managed_AM_IR_Implementation_Report.md)。**
- 范围：先完成当前 pipeline validator 的 AM/Matmul 源语义、compiler、artifact、资源/数值/周期模型和调用方迁移；具体分支与重试处理留给更低层 IR／执行协议后续考虑，不作为本次软件 gate。
- 依据：[共享 AM V1](../../nexus_shared_am_v1_requirements_and_design.md)、[硬件与系统 proposal](BOA_Gemmini_Integration_Design.md)，以及本轮软件管理 AM 的用户确认。
- 本计划只展开软件/IR 前置任务；不是整个硬件任务的实施授权。下文新语法、类型、类名、测试和生成器均为**拟实现接口**，不是当前仓库已支持的能力。
- §3.4“受限控制流，不增加通用 DAG 执行器”按用户要求保留，**整节标记为本次不实现**。本次继续使用单 block 源 IR 和现有 `tile.await`，不新增源级 TRY 状态、retry token 或控制流操作。

## 1. 已确认需求与软件/硬件边界

| 议题           | 本轮确认的合同                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------------------ |
| AM allocation  | 扩展 `tile.alloc`，显式 `memory_space="am"`，返回 `!tile.am_buffer<...>`；不是按变量名或 f32 类型推断              |
| L1 allocation  | 原有 L1 语义保持；`memory_space="l1"` 可显式书写，省略时仍为 L1                                                    |
| 软件管理       | 软件决定申请位置、逻辑 AM 位置 0/1、每步 Matmul 和最终输出点；物理 group/slot 和 engine 仍由设备选择               |
| 资源不足       | BUSY、重试快照和恢复是下层资源协议；本次源 MLIR 不返回或分支处理这些状态。失败不占物理槽、不重复提交等安全合同保留 |
| Matmul operand | A、B 输入，L1 output，AM acc 都是显式 SSA operand                                                                  |
| 模式           | 保留 accumulate，再增加 output；`accumulate output` 表示 ADD + FINAL，提交最终结果到 L1                            |
| 释放           | 最终输出可见且旧访问结束后设备自动归还 AM；不依赖软件 free/poll/reap                                               |
| 输出 dtype     | 保留 f32 原精度输出；默认 BF16，可选 INT8 转换；不增加 FP32/INT8 输入 Matmul                                       |
| 时序           | 本次范围内的源 IR/模拟器 gate 仍先于硬件实现；显式分支、重试 codegen 和低层动态恢复策略后续单列，不混入此次 gate   |

继续遵守共享 AM V1：4-engine 多阵列基线、4 AM groups×2 slots=8 动态共享槽；每 tile context 跨所有 engines 最多两个 live leases；链级 affinity 在成功申请后直到 slot_free 不变；同 context 两条独立链可以交错或并行；one-shot 也计入配额。

不重新引入专用 BOA RF 或向 UCE 报内部 load 完成的协议。普通输入仍按实际最后读取与全部读者结束释放。软件管理 AM 不表示软件选择物理 bank、直接访问 AM 地址或发任意 atomic add。

## 2. 当前仓库事实与改造入口

| 层         | 当前入口与行为                                                                                                                                                    | 需要改变                                                                                     |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 源类型     | [elenor.py](../../pipeline_validator/dialects/elenor.py) `TileL1Buffer`，222–242 行                                                                               | 增加独立 AM 类型，不复用 L1 冒充 AM；本次不增加 TRY 状态/重试 token 类型                     |
| allocation | 同文件 `TileAllocOp`，1970–2012 行，只有 shape/dtype/alignment，结果固定 TileL1Buffer                                                                             | 按显式 memory_space 分派，AM 走动态租约，L1 原路径保留                                       |
| Matmul     | 同文件 `TileBoaOp`，2410–2460 行，只有 m/n/k/ops/accumulate，无 operand                                                                                           | 增加 A/B/out/acc、seq、output 和明确内存 effect；源级仍返回一个完成事件                      |
| Tile 控制  | `TileProgramDefOp.body=region_def("single_block")`；[workload_ir.py](../../pipeline_validator/workload_ir.py) `_body_ops` 同样要求单 block                        | 本次保持该约定，不新增 branch、label、循环或源级重试控制                                     |
| lowering   | [lowering.py](../../pipeline_validator/compiler/lowering.py) 845–854、892–927 行                                                                                  | AM 不再进入 ExecL1Buffer、L1 Arena 或 ALLOC_L1；Matmul 生成 typed descriptor                 |
| layout     | [resources.py](../../pipeline_validator/compiler/resources.py) `layout_buffers`、`conservative_arena_bytes`                                                       | 分别核算 L1 buffer 与 AM 动态 slot 需求，不给所有 context 永久预留两槽                       |
| 执行/codec | [execution_ir.py](../../pipeline_validator/execution_ir.py) `ExecTileProgram/ExecEngineDesc`；[compiled_program.py](../../pipeline_validator/compiled_program.py) | 增加 typed AM/Matmul 数据和闭合 codec；不为本次新增控制流 opcode；当前严格 v4/v4（本次实现） |
| 独立验证   | [execution_verifier.py](../../pipeline_validator/execution_verifier.py) 738–846 行                                                                                | 在现有线性检查上补 AM/operand/模式/生命周期验证；CFG 与路径敏感分析不在此次范围              |
| runtime    | [tile.py](../../pipeline_validator/tile.py) ALLOC_L1、launch queue、local event、RET；[engines.py](../../pipeline_validator/engines.py) `BOAEngine`               | 新增真实 AM 所有权、非阻塞申请/提交、多完成边界、自动 drain；BOA 当前仍是延迟模型            |
| 事件       | [event_table.py](../../pipeline_validator/runtime/event_table.py) owner/generation/consumer_refs；Tile-local id 为 `ctx{ctx_id}:{local_name}`                     | 复用事件管理，但新增动态尝试/链/步骤实例身份，防止重试、slot 和 context 重用误投             |

当前正式源 IR 是单 block，独立 verifier 拒绝 label，WAIT arity=1、WAITALL 为 all；**本次保持这些控制流边界**。这不是 AM 类型和 Matmul operand 改造必须先补齐的 CFG 缺口；相关低层分支与重试设计保留在 §3.4，后续再考虑。

### 2.1 用户示例在今天确实是 L1

用户给出的 allocation 位于 [matmul_splitk_pipeline.mlir](../../examples/workloads/matmul_splitk_pipeline.mlir) 的 BOA leaf；同一文件的 EVU combine 也有 f32 L1 scratch/accumulator，不能一起改 AM。

此前编写初版计划时，当前 parser 的 `load → print → reparse → print` 检查结果一致；该文件有 8 条 TileBoaOp，operand 数全部为 0，解析出的 6 个 f32 allocations 均为 `!tile.l1_buffer<64x256xf32>`。这是历史现状检查，**不是新 AM 功能验证，本次范围修订未重跑该场景**。

LSP 当前仅有 Ruff，未提供 references 能力；实现时若有支持 references 的 server，修改导出构造 API 前先查引用，否则使用精确符号搜索列出全部调用方，不能只改一个 example。

## 3. MLIR dialect 的简明合同

### 3.1 类型与分配：AM 不是 L1

新类型建议沿用现有 shape/dtype 打印风格：

| 类型                       | 含义                                                                            |
| -------------------------- | ------------------------------------------------------------------------------- |
| `!tile.l1_buffer<MxNxf32>` | 普通 L1 数据，仍可作为 MFE/EVU 等合法 operand                                   |
| `!tile.am_buffer<MxNxf32>` | 一个输出链的 AM capability；不是裸地址，不进入 L1 SlotFrame                     |
| `!tile.event<"tag">`       | 成功提交后的持久完成/失败事件，运行时附带 owner、generation、seq 和动态实例身份 |

源级类型表达数据/资源和完成依赖，不表达一次硬件尝试的返回码。L1 与 AM allocation 均返回一个 buffer 结果；BUSY/SUCCESS/ERROR 与重试信息留在下层协议，不增加源级返回值：

```mlir
%out = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "f32"
    alignment = 256 : !tile.l1_buffer<64x256xf32>

%acc = tile.alloc memory_space = "am" logical_slot = 0 output(%out)
    shape = [64, 256] dtype = "f32" alignment = 256
    : !tile.am_buffer<64x256xf32>
```

这里的 `tile.alloc memory_space="am"` **就是申请一个 AM slot 的源级语义**，不是先创建 L1 buffer 再隐式映射。`%acc` 标识该逻辑申请的结果；下层必须在真正获得 lease 后才允许依赖它的硬件访问，不能把入队或 BUSY 当作已经分配成功。具体等待、分支和重试如何展开，本次不规定，也不默认让 UCE 停在申请处阻塞已有链。

AM allocation 增加 `output(%out)` 的原因：V1 要求 OPEN 成功时已知并保护最终 L1 目的地。`%out` 必须在申请前定义、绑定、通过 memory admission；不等最后一步才另外争抢 L1 输出空间。所有 Matmul 的 `outs` 必须与该绑定相同。

- `logical_slot` 只允许 0/1，软件不指定 physical_slot、group 或 engine。
- shape/dtype/AM result type 必须一致；本轮 AM dtype 只允许 f32。
- 申请成功才建立 lease、扣 quota、固定 affinity、保留 mailbox/terminal record；失败无持久分配副作用。
- OPEN 不要求 SRAM 清零；第一次合法 INIT 初始化所有有效输出位置。
- `tile.free %acc` 对 AM 类型非法。最终 output 消费这条链并自动释放；事件 reap 是记录退休，不是 AM free。
- `tile.load/store/gather/scatter` 和 EVU 不能直接接收 AM 类型。现有合法的 L1/MFE 行为不改变。

### 3.2 Matmul：四个显式 operand，两个正交标志

沿用现有 `tile.boa.async "matmul"` 名称，但**删除无 operand 的旧签名**。拟议规范顺序：

```text
%done = tile.boa.async "matmul"
    ins(%a, %b) outs(%out) acc(%acc)
    m = M n = N k = K seq = S
    [accumulate] [output]
    : !tile.event<"tag">
```

`ins` 是两个 BF16 L1 输入，`outs` 是 L1 目的地，`acc` 是 AM capability。转置/有效 mask/后处理作为明确属性或 frozen descriptor 配置继续支持，不能因引入 AM 被删除。

m/n/k 与有效输入、输出和 AM shape 一致；不得越出声明 view。先支持二维逻辑 Matmul，物理微块/尾块由已有 tiling/adapter 合同描述。`ops` 不再由数值 Matmul 的作者手填，编译器从有效 m/n/k 推导 `2*m*n*k`；保留 trace 的 ops 统计，但不让它与真实 operand 矛盾。

| 标志                | AM 动作                   | 是否写 L1 output         | 成功完成事件   | 之后能否继续使用该 acc   |
| ------------------- | ------------------------- | ------------------------ | -------------- | ------------------------ |
| 无标志              | INIT：`acc := A×B`        | 否                       | step_committed | 可以提交后续 ADD         |
| `accumulate`        | ADD：`acc += A×B`         | 否                       | step_committed | 可以提交后续 ADD         |
| `output`            | INIT + FINAL，单步 GEMM   | 是，最终提交后自动 drain | chain_done     | 不可再提交，需重新 alloc |
| `accumulate output` | ADD + FINAL，最后一次累加 | 是，最终提交后自动 drain | chain_done     | 不可再提交，需重新 alloc |

`accumulate output` **包含本次 A×B**，不是仅搬运旧 accumulator 的 store。若需要纯 AM dump/快照，应另行批准，不把它暗中塞进此模式。

INIT 仅允许 seq=0；后续必须 accumulate，seq 连续；一条成功链恰好一个 output。output 被 ACCEPTED 后立即关闭新 STEP 入口，但 AM lease 继续占 quota，直到输出可见与旧访问 quiescent 后 slot_free。

这里的 ACCEPTED 是下层实际受理状态，不是新增加的源级 SSA 结果；源程序不需要写判断或 retry 分支。

所有 Matmul 都显式携带同一个 `%out`，但没有 output 标志时**不读取或改写其数据**；它只标识已绑定的最终目的地。该目的地在链存活期内保持合法且独占写入授权，不能被 free/rebind 或被其他链无依赖地重叠写。

在 AM 里保留 FP32，输出可为 f32 原精度直出、BF16 或 INT8；转换只在最终 drain。f32 输出能力不等于支持 f32 输入 Matmul，也不让 EVU 直接访问 AM。

### 3.3 源级结果与下层资源协议

| 层级               | 结果/状态                              | 本次边界                                                                        |
| ------------------ | -------------------------------------- | ------------------------------------------------------------------------------- |
| 源 MLIR allocation | 一个 `!tile.am_buffer`                 | 声明逻辑 AM 申请、shape 和输出绑定，不暴露物理尝试状态                          |
| 源 MLIR Matmul     | 一个 `!tile.event<"tag">`              | 表达该操作的完成依赖；非 output 对应 step_committed，output 成功对应 chain_done |
| 下层申请/提交接口  | GRANTED/ACCEPTED、BUSY、ERROR          | 保留资源安全语义；具体状态编码、分支、重试恢复策略后续考虑                      |
| 功能/周期模型      | 真实 lease、quota、依赖、完成/错误状态 | 不假设资源永远可用；未成功不得占槽、更新 AM 或虚构成功完成                      |

本次不新增 `!tile.try_status` / `!tile.retry_token` 源类型，不要求作者处理一个无效 `%acc` 或 `%done` 的失败分支。下层必须保证正确的受理与依赖边界；不能通过省略源级分支，把 BUSY 伪装成成功。

既有完成/错误机制仍需安全收尾。FINAL 的失败路径回收资源后保留失败终态，不产生成功输出；这不要求本次新增 `tile.fail` 操作。资源等待可以在模型内部表达，但不因此冻结一套新的低层 retry CFG 或重试指令。

### 3.4 受限控制流，不增加通用 DAG 执行器

> **本次不实现、不纳入本次软件 gate。** 本节按用户要求保留，作为后续更低层 IR／执行协议讨论材料；本次源 MLIR 不增加分支、循环、TRY 状态处理或重试逻辑。

以下是上一版提出的候选形式，**仅留档，不是本次 API，也不预先决定未来必须采用这些名字或放在源 dialect**：

| 留档候选（本次不实施）                                        | 原方案意图                          |
| ------------------------------------------------------------- | ----------------------------------- |
| `tile.try_branch %status success ^ok busy ^retry error ^fail` | 根据一次申请/提交状态选择后续路径   |
| `tile.await_any %event_or_retry, ...`                         | 等待未解决的完成/资源条件后重查状态 |
| `tile.fail %status`                                           | 停止正常提交并进入受控错误收尾      |
| `cf.br ^label`、`!tile.try_status`、`!tile.retry_token`       | 显式 CFG、尝试状态与重试快照表示    |

具体分支、重试、check-and-register、回边进展和恢复编码应在更低层 IR／执行协议设计时再决定；本次不改 `TileProgramDefOp` 的 single-block 约束，不放宽 label/branch 白名单，不引入 CFG/path-sensitive verifier。

现有 `tile.await` 的 all 语义与 `tile.return` 的安全退休合同继续保留。以后如果低层需要显式 CFG，仍需证明不能因新 OPEN 阻断已有链的 FINAL，也不能把整个 Tile Program 改成通用 DAG 执行器；这些是后续设计约束，不是此次新增控制流任务。

### 3.5 简单示例：两步链，末步 accumulate output

下面是**本次目标源语法的单 block body 片段**，不是今天 parser 可接受的完整 workload。进入片段前，`%a0/%b0/%a1/%b1` 已在 L1、producer 已完成；`%c_l2` 与 `%task` 已绑定，外围程序按原约定处理 input_released。片段只表达申请、计算和完成依赖，不展开下层资源不足时的分支/重试。

```mlir
%out = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "f32"
    alignment = 256 : !tile.l1_buffer<64x256xf32>
%acc = tile.alloc memory_space = "am" logical_slot = 0 output(%out)
    shape = [64, 256] dtype = "f32" alignment = 256
    : !tile.am_buffer<64x256xf32>

%k0 = tile.boa.async "matmul"
    ins(%a0, %b0) outs(%out) acc(%acc)
    m = 64 n = 256 k = 64 seq = 0
    : !tile.event<"k0">
tile.await %k0

%done = tile.boa.async "matmul"
    ins(%a1, %b1) outs(%out) acc(%acc)
    m = 64 n = 256 k = 64 seq = 1 accumulate output
    : !tile.event<"done">
tile.await %done

%stored = tile.store.async %out into %c_l2 : !tile.event<"stored">
tile.await %stored
tile.signal output_ready(%task)
tile.return
```

`%done` 成功意味着 L1 输出可见、AM 安全释放和 quota 归还；`%stored` 是后续 L1→L2 store，不能合并。旧例子的 store `%acc` 必须改为 store `%out`；没有 `tile.free %acc`。

更多 K-step 在首步与末步之间增加 `accumulate`、不带 output 的操作及必要依赖。两条独立链用各自 AM operand，不添加无数据依赖的 context-wide 顺序；本次示例不承担动态资源竞争下完整低层重试方案的证明。

## 4. 两槽推进、生命周期与资源核算

### 4.1 软件两个位置与物理八槽

compiler 管理 `logical_slot=0/1` 和每链的 seq；runtime 在成功 allocation 时动态分配 slot、决定 bound_engine 并保留终态记录。编译期可声明 `tile_am_slots_per_context` 的需求上界 0..2，但它**不是**在 context admission 时申请两个物理槽。

本次源程序用两个逻辑位置、显式 operand 和已有完成依赖表达两条链；不增加观察 BUSY、分支重试或 WAIT_ANY 的源程序模板。新 OPEN 不得阻断旧链供数、STEP、FINAL 和 drain 的进展合同仍然有效；具体如何在更低层 IR／执行端实现，后续单独设计。

本次可使用已 await 的输入和已有合法 producer 依赖完成数据验证；设备只在输入实际 ready 后发射，不提前占 engine/group 服务资格。不要为此新增源级 retry 或分支操作。

同 handle 的下一步 issue 等前步 step_committed；一个 current 加一个 next-step mailbox。不同 handle 无额外 context-wide 顺序。配额不足、未受理无副作用等可由资源模型独立验证，但不能据此声称“完整低层重试与动态进展已实现”。

### 4.2 生命周期状态与 effect

源 verifier 与独立 executable verifier 检查线性程序中的 AM 定义、INIT/ADD/output、依赖和最后使用；runtime 模型检查实际 lease 与完成边界：

```text
源级逻辑生命周期：
  AM allocation定义
  → INIT
  → accumulate（可重复）
  → output / accumulate output（消费该链）
  → 仅L1 output继续存活

下层物理生命周期：
  真正GRANTED时 quota +1
  RESERVED/WAIT_INPUT/RUNNING/COMMITTING/FINAL_READY/DRAINING/FAULT_QUIESCING均计入
  仅slot_free时 quota -1；不等软件await/reap
```

- 普通 mode 的 done = step_committed；最终 mode 的 done = chain terminal，成功才满足 chain_done。
- 无 output 的操作只写 AM；带 output 才写 L1 `%out`。source/executable verifier 必须能据此检查 store/read 的依赖。
- 输入通过普通 input_usage/ref tracking 保护；可保守到已证明不再读的 step 完成边界，不引入内部 load 通知。
- 成功提交时快照所有 view/config/owner；只复制 descriptor 元数据，不在接受时提前复制 A/B 字节。
- FINAL 接受后输出缓冲仍不可 free/rebind；AM 释放后 L1 `%out` 继续供 MFE/EVU 使用。
- 同 ctx 重绑和 slot 重用不能复用裸事件名；完成事件包含 context epoch、链实例、handle generation、seq、kind。具体 retry 实例的编码与恢复留待低层设计，不增加源级结果。
- FINAL terminal record 可比 slot 活得久；在合法 await/reap 或外部引用退休后释放记录。FINISHING 不依赖已经返回的 context 重新运行。

### 4.3 64×256 示例对容量意味着什么

`64×256×4 = 65536 bytes = 64 KiB`，仍是**一个**逻辑 AM slot 的 shape 需求，不是把它拆成不计配额的临时槽。

| 验证配置中的逻辑输出块          | 单槽有效数据下界 | 八槽有效数据下界 |
| ------------------------------- | ---------------: | ---------------: |
| 16×16 f32                       |            1 KiB |            8 KiB |
| 64×256 f32（本次示例）          |           64 KiB |          512 KiB |
| 256×256 f32（现有大 Matmul 族） |          256 KiB |            2 MiB |

这些是数值容量下界，不是产品宏容量。padding/alignment/stride/ECC/实际粒度另计。对各 fixture 使用显式 simulator AM 配置；容量不够必须拒绝或重新合法 tiling，不自动增槽、不跨邻槽越界、不靠增大 context 配额解决。

AM 独立于现有 L1/L2 profile/Arena/Frame。建议增加专用 `SharedAmConfig/SharedAmManager` 语义，而不是把 `MemoryProfile.level` 全仓扩成 am 以模拟普通 SRAM heap；保留一个驻留 context 一个 memory profile。AM 配置及数值规则进入 target fingerprint。

**不能承诺 L1 自动减少 64 KiB。** 原 `%acc` 移出 L1，但新 `%out` 仍需 L1：f32 输出常与旧 acc 一样大；BF16 输出大小不同。L1 envelope 应从新 buffer 集合、lifetimes 和条纹 padding 重算，而不是手工减一个常数。

## 5. 编译产物与清理迁移策略

### 5.1 Typed execution contract

拟议的最小结构（名称待实现时按现有命名风格落地，字段语义不得遗漏）：

| 对象                | 必要字段/职责                                                                                                                 |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `ExecAmBuffer`      | buffer_id、dims、dtype=f32、alignment、logical_slot、绑定 output view、数值/epilogue配置                                      |
| `ExecBoaMatmulDesc` | A/B 的 ExecMemoryView、L1 output view、AM buffer_id、m/n/k、seq、init_or_add、final、transpose/mask/依赖                      |
| `ExecTileProgram`   | 在 l1_buffers 之外单列 AM 声明，保持现有线性指令组织；AM 不进入 ArenaLayout                                                   |
| `ExecTileOp`        | 扩展 AM allocation 与 typed Matmul 执行语义，沿用现有等待/错误出口；具体低层 TRY/branch/retry opcode 不在本次冻结             |
| `AmLease`           | owner ctx/epoch、entry/generation、physical group/region、bound_engine、seq、mailbox、state、pending updates、terminal record |

保持已有 typed gather/scatter 的思路：在 `ExecEngineDesc.params` 中只存一个已注册的 typed matmul payload，不增加任意可绕过校验的 free-form 字段。AM capability 不伪装为 `ExecMemoryView(space="l1")`；A/B/out 仍复用真实 L1 view。

独立 verifier 从 executable 的 buffer/type/config、线性指令依赖和 effect 重建不变量，不信任 source verifier，也不从 source_ir 恢复缺失语义；本次不增加 CFG 分析。`pending_am_updates`、O/R credits、bank/adder 吞吐和 output write acknowledgements 仍是模型状态，不能从一个旧 finish_cycle 推断全部完成。

### 5.2 ABI 与旧 BOA 语法清理

该改造是 breaking change；按当前基线统一升级 **schema_version=4、compiler_abi="v4"**。同步 compiler 产生值、codec 双入口检查、registered types/enums、independent verifier、hash/target identity、loader 错误说明和全部受影响 artifact。旧 v3 明确拒绝并提示重编译，不提供自动旧格式加载 shim。

`tile.boa.async` 只保留新 operand-based 数值/资源合同；旧无 operand 签名和旧构造参数必须移除，不能通过默认空 operand 兼容。

确实只模拟 FLOPs/控制门、不具备数学 operand 的场景，迁移为**明确的 `tile.boa.estimate.async`**：独立操作/descriptor、标记 timing-estimate，不声明 AM 计算、不产生数值 C、不参与 typed AM 链，不在 numerical/RTL backend 中静默执行。它不是旧名字的别名；旧 grammar 拒绝，调用方必须显式分类迁移。

保留估算能力是为了不把用户未要求的 EVU/USE/整个模型数值化强塞进本任务。若一个场景需要证明真实 AM 行为，必须迁到 typed Matmul，不能用 estimate 避开 quota 或数据验证。

### 5.3 全部调用方的迁移分类

| 调用族                                                                                | 迁移要求                                                                                  | 必须避免                                             |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `matmul_splitk_pipeline.mlir`、`matmul_splitk_multicontext_pipeline.mlir` 的 BOA leaf | acc→AM；新增独立 L1 out；末步 accumulate output；保留 f32 partial 输出；重算资源          | 把 combine 中 EVU 的 f32 scratch/acc 改 AM           |
| `matmul_2048x512x64_boa256x256x32.mlir`、matmul_pow/tail 家族                         | 明确 A/B/out/AM；把旧 BF16 output 与新的 FP32 AM 分开；补 AM 容量配置                     | 把输出 buffer 和 accumulator 继续混为同一 L1 对象    |
| `gather_matmul*` / `matmul_gather_add*`                                               | 逐项厘清 A/B 和现有 acc_init 的数学含义；需要真实初值则显式一次性 bias/seed合同，不能忽略 | 只因已有 ByteStore 输入就宣称当前 BOA 已数值正确     |
| Transformer / PagedAttention generators                                               | 从生成器的真实 operand 数据流迁移可确定的 BOA；其余明确 estimate，不假造 buffer           | 直接手改生成 MLIR、把 EVU/Pow 输出当作已经有真实数值 |
| `examples/scenarios/nest_subgraphs/` 控制/时序门                                      | 没有矩阵输入的 BOA 改显式 estimate；相关调度验证继续成立                                  | 给控制门填零矩阵当作数值迁移                         |
| tests 内 TileBoaOp 构造                                                               | 真实内存/AM 行为用 typed Matmul；纯时序 gate 用 estimate                                  | 给新参数随意填假的 operand 只为测试继续绿            |
| 文档、CLI、缓存 artifact                                                              | 更新 IR_SPEC/README/Limitation、所有 wrapper 参数和输出格式说明；重编译旧 artifact        | 只更改 MLIR 文本、不更新 codec 或 source_hash        |

生成器权威包括 `transformer_common.py`、transformer prefill/decode 的单/多 context/dispatchparallel 变体及 `generate_paged_attention_decode.py`。当前部分 matmul fixture 无 generator，应新增本次 AM 数值场景的 generator，并保留既有手写 fixture 的合法维护方式；不要把“所有示例都是生成的”当作事实。

现有按源码正则解析 `tile.boa.async` 文本的测试不应重新固定新字符串；删除 incidental wording/source-text 断言，改为 parser/semantic effect、真实输出、事件偏序或 FLOPs 行为检查。

## 6. 优先级、依赖与文件级任务 guide

以下任务只覆盖本次明确的源语义与模型范围，完成后进入软件 gate；§3.4 留档的控制流、具体低层分支/重试 codegen 不属于这些任务，也不是它们的前置依赖。共享符号/文件仍须有一个集成 owner，不以空模块或虚假资源成功作为完成。

```text
P0-01 类型/操作/源语法 ------> P0-02 线性IR语义验证
P0-01 ----------------------> P0-03 executable/codec/lowering
P0-01 ----------------------> P0-04 L1与AM资源分离
P0-02 + P0-03 + P0-04 -------> P0-05 独立executable验证
P0-03 + P0-04 ---------------> P0-06 AM manager/事件
P0-01 + P0-05 + P0-06 -------> P0-07 TileUCE接线与snapshot
P0-06 + P0-07 ---------------> P0-08 BOA功能/周期/自动输出
P0-02 + P0-03 + P0-08 -------> P1-09 全调用方与fixture迁移
P0-05 + P0-08 + P1-09 -------> P0-10 端到端软件gate
P0-10通过 ------------------> 后续硬件实施的必要前提
```

编号是稳定交接单元，图中的依赖不是允许跳过其他任务；优先级 P1 的调用方迁移同样必须在完整 gate 前完成。

### P0-01 · 源类型、操作与单 block 语法

- **文件**：`pipeline_validator/dialects/elenor.py`；受影响的构造 helper。
- **输入**：§3.1–§3.3、§3.5 的本次语法；§3.4 留档不作为实现输入。
- **实现 guide**：新增 AM 类型，按 memory_space 分派 allocation；L1/AM allocation 都只有一个 buffer 结果，Matmul 固定 A/B/out/acc 并只返回一个完成事件，output 与 accumulate 正交。保持现有单 block 与等待语法，不注册新控制流 dialect。
- **IRDL 接口约束**：allocation 的 `result` 是 L1/AM 的受检 union；Matmul 的 `result` 仍为 TileEvent。本次不增加 status/retry 结果，不能用 AnyAttr 绕过类型/arity 校验。
- **输出**：可 parse/print 的新 dialect 与明确的非法类型/标志诊断；旧 operandless Matmul 被拒绝，estimate 有独立名字。
- **验证**：两种 allocation、四种 Matmul mode、f32/BF16/INT8 output 和错误类型的 roundtrip 语义一致；省略 memory_space 的 f32 allocation 仍是 L1；不测“打印字符串包含某词”代替语义。
- **禁止**：只让 parser 接受新关键词，lowering/runtime 仍走旧 BOA；用可选空 operand 保留旧签名。

### P0-02 · 源 IR 线性生命周期与依赖验证

- **文件**：`pipeline_validator/workload_ir.py` 的既有验证入口与 effect helper。
- **输入**：P0-01 的单 block 程序、显式 SSA operand、已有完成依赖。
- **实现 guide**：在线性指令顺序和事件依赖中检查 AM 定义、INIT/ADD/output、logical_slot 上界、seq、唯一 FINAL、output 绑定和最后使用。保护 L1 输入/output 和 Group phase；不建立 CFG 或路径敏感分析。
- **输出**：针对未定义/已消费 AM、非法模式/seq、静态不合法逻辑槽生命周期、unsafe free/store 的诊断。
- **验证**：两条独立链的合法交错通过；读未输出 C、AM 当 L1、output 后继续使用、缺 FINAL 等被拒。第三次底层资源尝试的 BUSY/no-side-effect 由 manager 测试，不要求在源程序写 TRY 分支。
- **禁止**：为了此任务放宽 single_block/label 约束；用 context-wide 顺序串行化独立链，或声称已证明所有动态重试场景的进展。

### P0-03 · Executable DTO、codec、lowering 与 ABI

- **文件**：`execution_ir.py`、`compiler/lowering.py`、`compiled_program.py`、`compiler/api.py`、`loader.py`。
- **输入**：新源 IR、typed view、§5 execution contract。
- **实现 guide**：单列 AM 声明和 typed Matmul descriptor，不让 AM 进入普通 L1 地址路径；沿用线性执行对象和已有等待结构，生成稳定 source_ref/instruction_id，ops 从 shape 推导。同步注册、codec 和 v4/v4 检查；本次不增加 TRY_BRANCH/WAIT_ANY 或重试结果 slot。
- **代码容量**：AM/typed descriptor 纳入 program identity 和代码/元数据预算，保留 program SRAM 容量检查；K-step 和输出链可静态展开，不生成新的 branch/label/retry CFG。
- **输出**：自包含线性编译产物，无需读取 source_ir 即可执行/独立验证；新的资源配置进入 target/program hash。更低层的具体分支与重试形式后续另定。
- **验证**：serialize→parse→load 保留语义；篡改 operand/shape/AM binding/seq/mode/版本后重新 seal 仍被独立 verifier 拒绝。
- **禁止**：只改一个版本常量、宽松接受未知字段、在 loader 里从源文本补回缺失状态。

### P0-04 · L1 layout、AM 配置与资源契约分离

- **文件**：`compiler/resources.py`、`profiles.py::TileResources/ResourceBudget`、`dialects/elenor.py::TileResourcesAttr`、`config.py`、`hardware_config.yaml`、`execution_verifier.py` 的 target字段清单。
- **输入**：L1/AM 两类声明、8槽/每context2配额、explicit slot_capacity配置。
- **实现 guide**：L1 allocator/Frame只处理L1；AM资源需求单列，shape/stride/alignment必须能放入一个slot。新增配置字段严格解析并纳入fingerprint；不新增可变AM memory profile或按context静态占槽。L1 envelope从新out/inputs和lifetimes重算。
- **输出**：明确分离的L1 arena预算、AM单槽容量要求和最大live声明。
- **验证**：64×256 f32申请要求64 KiB；配置不足被拒；普通f32 L1仍占L1预算；f32 out替换旧L1 acc不产生虚假64 KiB节省；8contexts各一槽可表达。
- **禁止**：直接把AM加入现有ArenaPool的普通heap、用R×2永久预留所有AM、动态扩槽掩盖容量不足。

### P0-05 · 独立 executable verifier

- **文件**：`execution_verifier.py`，必要的无 xDSL DTO helper。
- **输入**：只有v4 executable、硬件/资源配置，不信任source验证结果。
- **实现 guide**：扩展现有线性验证，独立检查 AM 声明、typed operand、依赖、handle 逻辑生命周期、effect、quota 需求和正常结束条件；不新增 successor/dominance/CFG 分析。
- **输出**：复核物理配置、L1 layout 和 AM 语义的加载门。
- **验证**：对 P0-02 的消费者可见错误做 resealed tamper；换 C、漏 FINAL、非法 ADD/shape/operand 均拒绝，既有不支持的 branch/label 继续拒绝。
- **禁止**：为本次放宽 BR/label 白名单，或只比较 source hash。

### P0-06 · 动态 AM manager 与持久事件

- **文件**：拟新增 `pipeline_validator/runtime/shared_am.py`（唯一共享AM所有权模块），复用 `runtime/event_table.py`；`tile.py` 的context身份、`tile_group.py`的安全退休连接。
- **输入**：已admit L1输出、ctx_key、logical_slot、shape/config、全局slot表。
- **实现 guide**：实现动态 lease、两槽 quota、固定 affinity、current+next mailbox、seq/代际和安全释放；资源申请全成或全败，模型可以观察受理/容量不足/错误，不能伪造成功。具体失败后的分支、重试策略与低层 token 编码后续考虑。
- **输出**：不依赖 Gemmini RTL 的资源/事件模型，以及内部受理边界；不是一套新源级 TRY API。
- **验证**：T01–T08、T14/T17/T19/T21 的资源/功能适用部分；失败不占槽、不改 seq，旧 generation 不误投、one-shot 无旁路。T16 中具体等待登记/重试竞态实现随低层设计后续验收。
- **禁止**：把记录未reap算作AM仍占用、事件id只用ctx号/源tag、失败后留下半slot。

### P0-07 · Tile UCE/模型接线与 snapshot（不新增控制流）

- **文件**：`tile.py`、既有事件/队列适配、`tile_group.py` 的 FINISHING 接口。
- **输入**：P0-05 验证后的线性程序、P0-06 资源接口与已解析 views。
- **实现 guide**：连接 AM allocation、typed Matmul 与已有等待/错误机制；只有实际受理后才生效，接受时冻结 descriptor/config/引用，不提前读字节。保证最终写回、释放和完成边界能在模型中观察，不引入源级分支或显式重试指令。
- **输出**：当前单 block 程序的 AM/BOA 模型接线、完成依赖和 snapshot。具体低层 retry CFG、重试调度策略不属于本任务。
- **验证**：线性单链及两条独立链、不同 K 长度、普通事件早到、context 返回后的自主 drain、代际隔离；不将这些测试冒称完整低层重试方案已验证。
- **禁止**：资源不足直接当成功；默认让阻塞 allocation 截住旧链却宣称无死锁；以通用设备 DAG 或新控制流 MLIR 替代本次明确范围。

### P0-08 · BOA 数值、有限周期资源和自动输出

- **文件**：`engines.py`、`runtime/shared_am.py`、`memory/byte_store.py`的既有接口适配、现有transfer/bank仲裁连接、`report.py`与trace连接；若抽取BOA模块只保留一个状态owner。
- **输入**：typed A/B/out/acc、有效shape、INIT/ADD/FINAL、当前选择的backend。
- **实现 guide**：模拟器真实读取BF16、生成FP32 partial、更新AM；无output不改C，FINAL更新完成后转换/直出并自动写L1，写可见与旧访问结束后释放。建模有限result FIFO、O+R信用、group bank/adder/route/RMW/drain延迟，不以一个旧finish_cycle代表全部事件。
- **输出**：本次源语义对应的软件数值/有限资源周期路径；具体低层分支与重试展开不在本任务，也不得通过该延期隐藏 AM 背压或提前完成。
- **验证**：中间C sentinel保持不变、AM partial正确，最终f32/BF16/INT8字节对独立oracle；T09–T12/T20/T22周期版本，L1背压恢复后无UCE polling也释放。
- **禁止**：DUT和oracle调用同一个计算函数自证、零延迟AM、无限result容量、只有event数增加却不写C。
- **数值前置**：实现前通过review冻结主proposal§3.2尚未批准的特殊值/物理分组规则；不能擅自选择FTZ、NaN或容差。本任务的接口/资源工作不依赖先实现某种WS/OS RTL。

### P1-09 · 调用方、生成器与文档清洁切换

- **文件**：§5.3所有调用族；`examples/generators/transformer_common.py`及各generator；拟新增 `generate_boa_am_matmul.py` 与其fixture/scenario；tests构造；`IR_SPEC.md`、pipeline_validator/examples README、Limitation。
- **输入**：新typed/estimate API，所有现有TileBoaOp和源码Matmul位置清单。
- **实现 guide**：分类迁移而非按变量名机械替换；BOA leaf用独立AM和L1 out，EVU f32保留L1；旧无operand操作显式迁estimate。修改generator后确定性再生fixture、更新source_hash和wrapper；所有受影响artifacts重编译。
- **输出**：仓库中不再有旧无operand的tile.boa.async或旧TileBoaOp构造签名；合法估算能力仍可使用新显式名字。
- **验证**：parse/compile全部受影响fixtures，运行代表性Matmul/split-K/gather/transformer/nest控制场景，核对新增流量与AM统计；不声称EVU/USE整图数值正确。
- **禁止**：给缺operand案例制造假数据、把f32 partial降成BF16来绕过迁移、保留隐式fallback或旧别名。

### P0-10 · 硬件前的软件集成 gate

- **文件**：已有行为测试及拟新增 `pipeline_validator/tests/test_boa_am.py`、编译/CLI回归；对应example generator/scenario。
- **输入**：本次范围内 P0-01 至 P1-09 的集成、现有环境和固定 seed 输入；不要求 §3.4 的延期项。
- **实现 guide**：按 §7 标明本次/后续边界，执行局部及统一门禁、source 与 compiled-only CLI；观察 C 字节、事件偏序、quota/credit 和资源归零。报告未覆盖的低层重试场景，不打印假的通过状态。
- **输出**：源级 AM/Matmul 与模型的可复现软件报告；把后续低层控制流和 RTL 验证明确分开。
- **验证**：本次 exit criteria 满足且受影响消费者通过；源级 CFG、TRY 状态分支和低层 retry codegen 不作为本次完成条件。
- **禁止**：用现有33,129-cycle时序smoke代替AM数值验收，或在没有真实运行时写“通过”。

## 7. 硬件之前的验收矩阵与命令

| ID   | 必须实际观察的行为                                                                                 | 关联层                      |
| ---- | -------------------------------------------------------------------------------------------------- | --------------------------- |
| IR01 | 同形状f32普通allocation仍是L1；显式AM allocation不进入L1Arena/Frame                                | dialect/layout/runtime      |
| IR02 | 64×256 AM只申请一个64 KiB需求的slot；不足拒绝、不能借邻槽                                          | config/verifier/manager     |
| IR03 | 无标志INIT初始化全部有效AM位置，旧垃圾不影响结果                                                   | 功能/数值                   |
| IR04 | accumulate更新AM但L1 C sentinel不变；不是读改写L1                                                  | 数值/内存effect             |
| IR05 | accumulate output包含最后一次乘法，最终L1字节正确并自动归还slot                                    | 数值/周期                   |
| IR06 | output 单步 INIT + FINAL 使用一个 slot，遵守相同 quota 和终态合同                                  | 功能/周期                   |
| IR07 | f32直出保留partial精度；BF16/INT8只在最终转换                                                      | 数值                        |
| IR08 | output accepted后再次提交、重复/跳seq、错误owner、AM传给MFE/EVU均拒绝                              | 双verifier/runtime          |
| IR09 | 单 block 两条独立链可表达；manager 的第三槽申请失败无副作用，不把 BUSY 当成功；具体重试推进留后续  | 本次语义/配额；重试展开延期 |
| IR10 | bound engine忙时不迁移；一个engine可以服务超过两个全局handles                                      | 功能/周期                   |
| IR11 | FIFO延迟使compute_done先到但commit/output/free不提前；O+R≤Q                                        | 周期                        |
| IR12 | descriptor接受后改写不改变执行；输入只有最后读者结束后才可覆写                                     | snapshot/内存               |
| IR13 | 既有完成事件早到、普通 await 和旧 generation 隔离正确；BUSY 快照等待/重试竞态在低层阶段验收        | 本次事件；低层等待协议延期  |
| IR14 | FINAL输出反压且UCE不poll时仍能在下游恢复后释放；终态记录仍可读取                                   | 周期/生命周期               |
| IR15 | 无FINAL返回报错；FINAL已受理的正常返回FINISHING；异步故障ERROR/CANCELLED安全收尾                   | context                     |
| IR16 | v4 artifact 无需 source 重放；resealed 的非法 operand/AM 绑定/seq/mode 仍被拒，不新增正向 CFG 用例 | codec/线性独立 verifier     |
| IR17 | 全部旧调用方清洁迁移；estimate明确标注且数值/RTL路径拒绝它                                         | 迁移/CLI                    |

IR06 的 `INIT|FINAL` 是逻辑标志组合，不是二进制编码。主 proposal 的 T01–T22 保留为完整系统目标；本次只验收源级语义、资源模型和数值/周期适用部分。T01/T13/T14/T16 等涉及具体分支、重试恢复或等待登记的端到端部分，随更低层 IR／执行协议后续设计验收；不能将其列为本次 gate，也不能因暂缓而宣称已通过。T10/T11/T12/T22 的周期结论仍需真实周期模型。

实施后使用现有入口，不新增只打印下一条命令的空脚本。以下新增fixture/scenario/test路径是本计划要求交付的文件，**本轮尚不存在**：

```bash
# 新生成器必须按仓库写fixture约定实现；此处是实施后的验收命令。
conda run -n elenor-validator python examples/generators/generate_boa_am_matmul.py
conda run -n elenor-validator python -m pytest \
  pipeline_validator/tests/test_boa_am.py \
  pipeline_validator/tests/test_compiler.py \
  pipeline_validator/tests/test_validator.py -n0

# 生成器输出固定输入数据；CLI绑定范围与fixture声明保持一致。
conda run -n elenor-validator python -m pipeline_validator \
  --ir-file examples/workloads/boa_am_matmul.mlir \
  --sim-override fidelity=full_memory --context-mode 4 \
  --hw-override boa_am_slot_capacity_bytes=65536 \
  --input-binding A=0x100000:16384:r \
  --input-binding B=0x200000:65536:r \
  --input-binding C=0x300000:65536:w \
  --input-data A=examples/artifacts/boa_am_inputs/A.bin \
  --input-data B=examples/artifacts/boa_am_inputs/B.bin \
  --compiled-output /tmp/boa-am-ir/program.json \
  --memory-trace --trace-json /tmp/boa-am-ir/trace.json \
  --json --report /tmp/boa-am-ir/report.json
```

该主 smoke 的数学形状固定为 `A[64,128]bf16 × B[128,256]bf16 → C[64,256]f32`，K 按两步 64 分块。源 binding 使用 block-packed `A[2,64,64]`、`B[2,64,256]`，避免在 row-major A 中切不连续的窄 K view；分别需要 16384、65536、65536 bytes。基础 smoke 使用 placement=1、一个逻辑 Task，硬件 context capacity=4；双链/多 context 使用独立变体，不能从这一条运行推断四 context 并发。

新配置名 `boa_am_slot_capacity_bytes` 在 P0-04 实现；其余 AM 字段使用显式定义的 4-engine/4-group/8-slot 软件 profile，不从现有 OPA 峰值参数猜实例数。64×256 是逻辑 output/AM 区域；默认 16×16×8 物理微块仍需拆分，微块归约顺序进入 numeric profile 和 oracle，不能把逻辑 k=64 当作单周期或单个物理块。

CLI之后必须读取C实际字节并对独立oracle，以及报告中的新AM检查；若当前CLI不导出C，P0-10必须提供现有ByteStore驱动的数值验证入口或明确输出参数，不能只看report completed。再用 `--compiled-file /tmp/boa-am-ir/program.json`、相同bindings/input-data/硬件配置重放，确认不读源IR。两槽不同K长、8contexts和失败注入由同一生成器的明确变体覆盖。

最终统一门禁沿用仓库 `conda run -n elenor-validator pre-commit run -a` 与 `python -m pytest pipeline_validator/tests/`；所有数值/周期smoke报告额外审阅C字节、各阶段事件和资源归零，不把pytest绿色当作唯一证明。

本次硬件前的软件 gate 包括：单 block 新语义与 estimate 分离、AM/L1 分离、四 operand/四模式数学与最终输出、合法两槽资源约束、固定 affinity、safe release/持久完成、v4 独立重放和受影响调用方/文档迁移。**不包括 §3.4 的源级 CFG、TRY 状态分支或具体低层重试展开。** 后续系统集成仍须验证动态资源争用下的完整进展；不能把本次 gate 的通过等同于该部分也已完成。

## 8. 本轮交付与验证边界

本轮交付本计划并同步系统proposal；不修改用户的共享AM需求文件，不实现上述类型/op/runtime，不生成假的通过日志。

历史 parser/roundtrip 证据见 §2.1；本次只修订文档范围与示例，不宣称新语法已通过 parser。`WS/OS`、特殊值和物理参数仍按主 proposal 的 review 门处理。AM allocation、自动释放和 f32 输出保持；分支与重试归属更低层 IR／执行协议，本次不做。

实施报告继续列出任务、文件、IR01–IR17 和 T01–T22 的实际运行层级；本次覆盖、明确延期、未运行 RTL 必须分开标注。保留 §3.4 不表示其内容已经成为当前实现或 gate。
