# NEST Context 调度测试子图规范

> 用途：指导其他 Agent 实现 NEST/Nexus 调度器、Simulator、IR Generator 与回归测试。
> 目标：覆盖主要 DAG 依赖、动态执行、分块流水，以及 NEST 的 Admission/Event/Buffer/Multi-context 边界。

## 1. 统一测试契约

每个 case 分别判定：

- **Correctness**：依赖、数据可见性、Event、Buffer 生命周期、最终结果正确。
- **Liveness**：合法且资源最终可满足的任务最终完成；永久不可满足请求明确 reject/error。
- **Scheduling Quality**：不制造伪依赖，不把局部等待扩大为无必要全局 Barrier；真实资源冲突与调度器人为串行化分开判定。

`A -> B` 表示 B 对 A 存在真实数据/状态依赖，**不等价于必须 await A.context_done 后才能 submit B**。实现中建议区分 `submit / prefetch_done / input_released / output_ready / store_done / context_done`。

---

# 2. 静态 DAG 组织

## S01 Pure Serial Chain — 纯串行链

```text
A -> B -> C -> D
```

代表：`Matmul -> BiasAdd -> Activation -> Matmul`。

**目的**：基础依赖/Event/完成语义。

**检查**：消费者只能在所需数据可见后执行；异步 Store 未完成不能错误宣布完整任务完成；Event 不得提前、遗漏、串号。

**变体**：等耗时；A/B/Store 分别长尾；全部同 Context；每节点独立 Context；`{A,B}/{C,D}` 两 Context。

## S02 Independent Chains — 独立链

```text
A0 -> A1 -> A2
B0 -> B1
C0 -> C1 -> C2
```

**目的**：局部等待隔离和 Multi-context 前进性。

**检查**：A 等待时已 Active 的 B/C 不应被全局冻结；不得生成跨链伪依赖。真实 Matrix/DMA/Bank 冲突导致的串行不算依赖错误。

**变体**：同构链；Matrix/DMA/Vector 异构链；任一链 10x/100x 长尾。

## S03 Fan-out — 单生产者多消费者

```text
      +-> B
A ----+-> C
      +-> D
```

**检查**：A 的完成可唤醒全部消费者；Event 不能被第一个 waiter 消费掉；共享 Buffer 活到最后一个真实读取者完成读取。

关键反例：B/C 已读完而 D 尚未读完时不得覆盖 A。若 D 已完整搬到私有 L1，则 L2 Buffer 可按 L2 最后读取释放，无需等 D 整体 compute 完成。

## S04 Fan-in — 多生产者单消费者

```text
A --+
B --+--> D
C --+
```

**检查**：D 等全部必要输入；单个前驱完成不能误触发；重复 Event 不能冒充另一个前驱。

到达顺序：A-B-C、C-A-B、B-C-A、同周期完成。

## S05 Diamond / Residual Fork-Join

```text
      +-> B --+
A ----+       +-> D
      +-> C --+
```

Residual：

```text
X -> F0 -> F1 --+
+---------------+-> Add
```

**检查**：B/C 独立推进；D 等 B/C；Residual 输入保持到 Add 最后读取；Identity skip 与真实 Copy skip 分开测试。

## S06 Unbalanced Diamond — 严重不平衡菱形

```text
      +-> B : 100 cycles --------+
A ----+                          +-> J
      +-> C : 1 -> D : 1 -------+
```

**目的**：发现长尾导致的无谓 Barrier/资源滞留。

**检查**：C/D 不应因 B 慢而无法执行自身可执行部分；J 仍等 B；短分支私有资源及时释放。

分别让长尾发生在 `load / compute / store`，尤其检查慢 Store 下是否错误使用 `compute_done == task_done`。

## S07 Nested Fork/Join — 嵌套分叉汇合

```text
A -> {B, C}
B -> {D, E}
{D, E} -> F
{F, C} -> G
```

**核心契约**：F 只等 D/E，不等 C；G 才等 F/C。令 C=100 cycles、D/E=5 cycles，检测局部 Barrier 是否被扩大为全层 Barrier。

## S08 N-shaped Cross Dependency — N 型交叉依赖

```text
A -----> C
         ^
B -------+
+-------> D
```

依赖：`C <- {A,B}`，`D <- {B}`。

参数：A=100、B=5、C=5、D=90 cycles。理想情况下 D 在 B 完成后即可运行，makespan 约 105。

反例：

```text
submit A
submit B
await A
submit C
submit D
```

此时 D 被伪串行化，makespan 可接近 190。结果可能仍正确，因此应允许 `Correctness=PASS, SchedulingQuality=FAIL`。

扩展：`{A,B}->J0`、`{B,C}->J1`、`{J0,J1}->O`。

## S09 Dense Skip Connections — 稠密跳连

```text
A -> B -> C -> D
C reads {A,B}
D reads {A,B,C}
```

**目的**：长 live range；区分 scheduling dependency 与 Tensor use-def。

即使 A->D 在完成顺序上可被 transitive reduction，D 仍可能直接读取 A；不能因此删除 Buffer 使用关系。

变体：A/B 分别为最大输出；L2 exact-fit；少一个 allocation quantum。

## S10 U-shaped Long Skip — U 型长跳连

```text
D2 depends on E2
D1 depends on {D2,E1}
D0 depends on {D1,E0}
```

**目的**：跨多阶段 Buffer 存活与 L2 峰值。

测试 E0 大/E2 小、E0 小/E2 大、三者均大。Admission 不能只按当前节点估算而忽略 live range。

## S11 Reduction Tree + Broadcast — 规约树后广播

```text
P0 --+
     +-> R0 --+
P1 --+        |
              +-> R -> {N0,N1,N2,N3}
P2 --+        |
     +-> R1 --+
P3 --+
```

**检查**：局部/全局规约等待集合正确；广播全部消费者；不存在的尾块 participant 不得进入等待计数。

变体：participant count=0/1/T-1/T/T+1；一个 partial 极慢；广播消费者快慢不均。浮点结果固定规约顺序或使用容差。

## S12 Attention-like Offset Join — Attention 式错位汇合

```text
X -> Q --+
         +-> QK^T -> Softmax --+
X -> K --+                     +-> PV -> Output
X -> V ------------------------+
```

**契约**：QK^T 只依赖 Q/K，不依赖 V；PV 依赖 Softmax/V。分别令 V/Q/K/Softmax 很慢。

若真实 Lowering 将 QKV 融合成不可拆 Kernel，测试必须服从真实可观察 Event，不能虚构 Q/K 提前 ready。

---

# 3. 动态执行组织 （暂时不考虑）

## D01 Dynamic Address, Static Topology — 动态地址 （当前已经有了，可以暂时不考虑）

```text
Matmul -> IndexGen --+
                     +-> Gather -> Matmul
Table ----------------+
```

**检查**：Index 未 ready 不得 Gather；Gather memory stall 不冻结独立 Context；重复读取不要误判为 Scatter 写冲突。

Index pattern：sequential、random、duplicate-heavy、hotspot、bank-dispersed、cache-friendly、cache-hostile。

## D02 Dynamic Cardinality — 动态任务数量

```text
Mask -> NonZero/Compact -> {indices,count} -> Gather -> Compute
```

**检查**：后续 task count 来自正确运行时 count；count ready 与完整 indices ready 可分阶段；`count=0` 必须产生合法完成路径，不能等待不存在的 Tile task。

边界：0、1、tile_size-1、tile_size、tile_size+1、max_supported。

## D03 Conditional Branch — 条件分支 (当前模拟器还没有添加对分枝的支持，这个例子可以忽略)

```text
                +-> true  -> A -> B --+
Condition ------+                    +-> SelectedOutput
                +-> false -> C -------+
```

**区别于普通 Diamond**：只执行选中路径。

**检查**：Join 只等待选中分支；未激活分支不得进入完成计数/资源预留。测试 always-true、always-false、请求间交替选择。

若 Lowering 为“两边都算+Select”，按真实语义测试，不能套用 selective-execution 的资源预期。

## D04 Dynamic Fan-out/Fan-in — MoE 式路由 (当前模拟器还没有添加对分枝的支持，这个例子可以忽略)

```text
Router -> Dispatch
          +-> Expert0: n0 --+
          +-> Expert1: n1 --+-> Combine
          +-> Expert2: n2 --+
```

**检查**：活跃 Expert 集合、动态任务数量、严重负载不均衡、空 Expert、Combine 完成条件。

变体：均匀；全部到一个 Expert；部分 Expert=0；一个 Expert 极慢；多个结果合并同一输出。若涉及写冲突，只使用芯片明确支持的独占分区/局部规约/顺序合并，不默认存在全局 Atomic。

---

# 4. 时间与分块组织

## T01 Chunked Load-Compute-Store Pipeline

```text
Load_A[0] --+
Load_B[0] --+-> Compute[0] -> Store[0]
Load_A[1] --+
Load_B[1] --+-> Compute[1] -> Store[1]
...
```

**目的**：验证两级 Multi-context 是否形成流水。A/B load 必须是独立 Event。

无跨 Chunk 数据依赖时不得添加整批 Barrier；但 Buffer slot 复用必须受生命周期保护：

```text
input_slot[i%2]：上次读取完成后才能覆盖
output_slot[i%2]：上次 Store 对该 slot 的读取完成后才能覆盖
```

检查 Load/Compute/Store overlap、容量反压、覆盖、丢 Chunk、重复执行。

## T02 Wavefront — 波前

```text
V00 -> V01 -> V02
 |      |      |
 v      v      v
V10 -> V11 -> V12
 |      |      |
 v      v      v
V20 -> V21 -> V22
```

内部节点依赖上方和左侧。

**检查**：局部依赖满足即可推进，不退化为整行/整列 Barrier；逻辑 task 坐标不能错误绑定固定物理 Tile/Context ID。

## T03 Loop-carried Dependency — 循环携带状态

```text
State0 -> Step0 -> State1 -> Step1 -> State2 -> Step2
           ^                 ^                 ^
         Input0            Input1            Input2
```

**检查**：后一轮读取正确状态版本；迭代 Event 带 epoch/generation，不能误用上一轮完成状态；测试 0/1/N 次与提前退出。

不要把同一 Event 实例直接连成环；有限展开后必须形成合法实例依赖。

## T04 Multi-request Shared Program — 多请求交错

```text
SharedWeights -> Request0: A0 -> B0 -> C0
              -> Request1: A1 -> B1 -> C1
              -> Request2: A2 -> B2 -> C2
```

**检查**：同程序不同实例 Event 隔离；Context slot 快速复用后旧 Event 不得完成新请求 waiter。

关键序列：Request0 很慢；Request1 很快并释放 slot；Request3 立即复用；随后旧请求/旧事务通知到达。Event key 应包含 generation/epoch 语义。

---

# 5. Context 切分测试

每个适用 DAG 至少生成三种合法映射：

```text
A. 多节点同 Context
B. 每节点独立 Context
C. 每个 branch/stage 一个 Context
```

比较 Context 内线性 PC await、跨 Context Event、Nexus submit、Admission 与 Buffer 生命周期。

## 非法切分反例

原图：

```text
A -> B -> C
```

切分：

```text
ctx0 = {A,C}
ctx1 = {B}
```

若跨 Context 只有 context_done，则 B 等 ctx0，而 C 又等 ctx1，形成：

```text
ctx0 -> ctx1 -> ctx0
```

**原始算子 DAG 无环，不代表 Context dependency graph 无环。**

期望：Compiler/Scheduler 拒绝该切分，或重切为 `{A}/{B}/{C}`。若使用更细粒度阶段 Event，必须同时证明事件可见性与资源前进性。

---

# 6. NEST 专项压力测试矩阵

| ID  | 场景                       | 构造                             | 期望                                            |
| --- | -------------------------- | -------------------------------- | ----------------------------------------------- |
| N01 | 完成阶段分离               | compute 快、Store 极慢           | 不得提前 context_done                           |
| N02 | Fan-out Buffer 生命周期    | 最后消费者延迟读取               | 不得提前覆盖共享 Buffer                         |
| N03 | WAIT_CAPACITY              | Device slot 已占、L2 不足        | 不占 L2/L1/UCE/DMA 执行资源；容量释放后重试     |
| N04 | Strict FIFO HOL            | 队头需4、后项需2、当前容量2      | 若契约 strict FIFO，后项不得绕过；记录性能代价  |
| N05 | Multi-Tile partial failure | 最后一个目标 Tile admission 失败 | 两阶段提交/全部回滚，不留部分绑定               |
| N06 | Event generation reuse     | slot 快速复用                    | 旧 Event 不得唤醒新实例                         |
| N07 | Local wait isolation       | ctx0 wait、ctx1 已 Active        | ctx1 继续推进；ctx0 同 PC 后续受线性 await 约束 |
| N08 | Empty/tail task            | 0/1/T-1/T/T+1                    | 只等待实际 participant                          |
| N09 | Impossible capacity        | 单 Context 需求永久 > HW 上限    | 明确 reject/error，不无限 WAIT                  |
| N10 | HBM visibility             | output_ready 但尚未 store_done   | 若 consumer 从 HBM 读，必须等待正确写回可见性   |

---

# 7. 参数化维度

每个核心拓扑不能只跑一组固定数字。

## 时间参数

```text
uniform
single predecessor 10x slow
single predecessor 100x slow
load-bound
compute-bound
store-bound
branch latency reversed
```

## Context 并发

```text
L2 contexts x Tile contexts
1x1
Nx1
1xM
NxM
```

比较时保持 SRAM、DMA、queue 等其他资源不变，避免把资源增加误认为 Multi-context 收益。

## 内存容量

```text
ample
exact-fit
one allocation quantum short
impossible single-request
```

## Placement / Bank

```text
same bank
separate banks
same tile-context pin
different pin
unconstrained placement
```

## Event 顺序

对所有 Fan-in/Join 类 case，随机打乱独立前驱完成顺序，并覆盖同周期完成。

---

# 8. 自动 DAG 生成与 Fuzzing

固定拓扑编号 `N0...N(n-1)`，只允许 `Ni -> Nj, i < j`，即可天然生成无环图。

候选边：

```text
Emax = n(n-1)/2
```

组合数：

```text
2^Emax
```

因此：

```text
n=5: 1,024
n=6: 32,768
```

参考生成器：

```python
from itertools import combinations


def forward_dags(n: int):
    if not 0 <= n <= 6:
        raise ValueError("n must be in [0, 6]")

    nodes = tuple(f"N{i}" for i in range(n))
    possible_edges = tuple(combinations(nodes, 2))

    for mask in range(1 << len(possible_edges)):
        edges = tuple(
            edge
            for bit, edge in enumerate(possible_edges)
            if mask & (1 << bit)
        )
        yield nodes, edges
```

**注意**：DAG 拓扑枚举只覆盖图结构，不自动覆盖：

```text
Event 阶段
Tensor use-def
Buffer size/live range
Context partition
placement
resource conflict
dynamic control flow
```

因此每个生成 DAG 还需要继续参数化。

推荐流水：

```text
DAG template / generated DAG
    -> annotate tensor edges
    -> annotate event semantics
    -> generate legal context partitions
    -> assign latency/resource/buffer/placement
    -> lower to NEST/Nexus IR
    -> run simulator
    -> check correctness
    -> check liveness
    -> compare schedule quality
    -> minimize failing graph
    -> save as regression case
```

失败后建议做 delta-debugging：删除节点、删除边、缩短 Buffer/资源参数，直到得到最小反例。

---

# 9. Agent 实现要求

生成测试代码的 Agent 应遵守以下规则。

## 9.1 Case 数据结构

每个 case 至少描述：

```yaml
id: S08
name: n_shaped_cross_dependency
nodes: []
edges: []
context_partition: []
resource_config: {}
expected:
  correctness: pass
  liveness: pass
  max_makespan_cycles: 110
  forbidden_dependencies:
    - [A, D]
```

## 9.2 不要只断言最终输出

至少采集：

```text
context_submit_time
context_active_time
prefetch_begin/end
dispatch_begin/end
input_released
output_ready
store_begin/end
context_done
buffer_allocate/release
admission_block/retry
physical_context_bind/unbind
```

## 9.3 必须建立 invariants

示例：

```python
assert consumer.start >= producer.required_ready_time
assert buffer.release >= buffer.last_read_end
assert no_event_from_old_generation_wakes_new_generation
assert context_done >= required_store_done
assert all_admitted_resources_are_released
```

性能类 invariant：

```python
assert not has_false_dependency("A", "D")
assert makespan <= reference_upper_bound
```

对于 Strict FIFO 等架构规定行为，不应写成“必须达到理论最优 makespan”。

## 9.4 Reference Scheduler

建议测试框架实现一个不考虑硬件细节的 DAG reference scheduler：

```text
node ready = all true data/control predecessors satisfied
```

然后再实现 resource-aware reference：

```text
node runnable = dependency_ready && resource_available
```

NEST Simulator 的 trace 与 reference 对比，可以区分：

```text
依赖导致等待
资源导致等待
Admission 导致等待
线性 PC / await 导致额外等待
```

---

# 10. 推荐首批回归集

若第一阶段不想一次实现全部 case，优先级建议：

## P0 — 必须先有

```text
S01 Serial Chain
S03 Fan-out
S04 Fan-in
S05 Diamond
S08 N-shaped Cross Dependency
T01 Chunked Pipeline
N01 Completion Stage Separation
N03 WAIT_CAPACITY
N06 Event Generation Reuse
```

## P1 — 很容易暴露架构问题

```text
S06 Unbalanced Diamond
S07 Nested Fork/Join
S09 Dense Skip
S12 Attention Offset Join
D01 Dynamic Gather
T04 Multi-request Interleaving
Context cyclic-partition negative test
N05 Multi-Tile rollback
N10 HBM visibility
```

## P2 — 扩展模型覆盖

```text
S10 U-shaped Skip
S11 Reduction/Broadcast
D02 Dynamic Cardinality
D03 Conditional Branch
D04 Dynamic Routing/MoE
T02 Wavefront
T03 Loop-carried State
5-node DAG enumeration
6+ node randomized fuzzing
```

---

# 11. 模型结构到测试模板的映射

这张表用于证明测试集覆盖的是“结构模式”，而不是只覆盖几个模型名字。

| 模型/工作负载结构                        | 主要模板      |
| ---------------------------------------- | ------------- |
| MLP / Sequential CNN                     | S01           |
| 多 Head / 多输出网络                     | S03           |
| Concat / Add / Fusion                    | S04           |
| ResNet / residual block                  | S05/S06       |
| Dense skip network                       | S09           |
| U-Net / encoder-decoder skip             | S10           |
| LayerNorm / normalization reductions     | S11           |
| Transformer Attention                    | S12 + S03/S04 |
| Transformer FFN                          | S01/S05       |
| KV/cache-related multi-request inference | T04 + S03     |
| Gather / embedding / dynamic indexing    | D01           |
| NonZero / compact / variable token count | D02           |
| Conditional execution                    | D03           |
| MoE / sparse routing                     | D04           |
| Tiled GEMM / convolution pipeline        | T01           |
| Wavefront tiled algorithms               | T02           |
| Recurrent/stateful execution             | T03           |
| Multi-request serving                    | T04           |

**覆盖目标不是声称有限模板可以数学意义上覆盖“所有模型”，而是覆盖模型图中反复出现的调度原语；再通过 DAG enumeration/random fuzzing 覆盖模板之外的组合。**

---

# 12. 最终验收标准

一个 NEST Context Scheduler 的测试框架至少应能回答：

1. **依赖是否正确？** 是否存在漏依赖、伪依赖、错误 Event 粒度？
2. **数据是否安全？** Buffer 是否过早释放/覆盖？不同存储层级的可见性是否正确？
3. **系统是否前进？** Admission、Context partition、Event generation 是否可能造成死锁或永久等待？
4. **并发是否合理？** 独立工作是否因线性 await 或全局 Barrier 被无意义阻塞？
5. **资源行为是否符合设计？** WAIT_CAPACITY、strict FIFO、两阶段提交、rollback 是否遵守架构契约？
6. **实例是否隔离？** Context slot/Event ID 复用是否有 generation/epoch 防护？
7. **动态工作是否安全？** 动态地址、动态数量、条件分支、动态路由是否有明确完成语义？
8. **是否可回归？** 每个 fuzz failure 能否最小化并固化为 deterministic regression case？

建议最终形成：

```text
hand-written topology tests
+ parameterized topology tests
+ negative legality tests
+ resource/admission stress tests
+ exhaustive small-DAG tests
+ randomized large-DAG fuzzing
```

这比仅用 ResNet、Transformer、U-Net 等完整模型验证调度器更容易定位 NEST Context 层面的错误，也更适合长期做架构演进和回归。
