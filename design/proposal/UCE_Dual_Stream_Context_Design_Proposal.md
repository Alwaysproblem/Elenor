# Tile UCE Dual-Context Execution Mode Proposal

版本：v0.2
定位：**在不新增 group-level fetchable program object 的前提下**，为 resident Tile Programs 提供一个更可落地的 dual-context coroutine-style execution mode。
状态：**architecture exploration / contract-convergence proposal**；First Silicon V1 仍以 `single-context + single-issue + in-order` Tile UCE 为 canonical cutline。
核心机制：**2 个 execution context + explicit wait(event/stream) suspension + pre-declared Slot Frame partition + conservative fault-drain-reset policy**。

> 术语收敛：旧稿中的 `stream context` 容易与 `Stream Queue` 混淆。本文统一改称 **execution context**；若需要兼容旧讨论，可视为同义替换。

**TODO：在后续版本中需考虑 memory order ，因为调度的时候有可能 store 还没有完成，load就已经开始了。在simulator/validator中暂时不需要验证这一点。**

---

## 1. 目标、适用边界和非目标

### 1.1 这份 proposal 解决什么问题

当前 ELENOR 已经有明确的控制层级：

```text
Runtime / Device Runtime
  -> TileGroupTask
    -> Tile Group Sequencer action list
      -> prepared tile task
        -> resident Tile Program handle
          -> Tile UCE
            -> BOA / EVU / MFE / USE / Tile DMA / Stream ops
```

这里真正缺的不是一个新的 group-level program object，而是一个**比 full OoO 小很多、但能在 tile-local 范围内产生有限 overlap** 的执行模式：

1. 每个 Tile Program 内部仍然保持简单顺序结构。
2. 只在显式 `wait` / `stream` 阻塞点切换上下文。
3. 不做任意 lookahead，不做动态猜测，不做 op-level dataflow scheduler。
4. 不改坏现有 `TileGroupTask -> Tile Program -> Tile UCE` 对象边界。

### 1.2 本文的目标

本文建议把原方案收敛为：

> **Tile UCE Dual-Context Execution Mode for Resident Tile Programs**

目标是：

1. 允许 **最多两个 resident Tile Program handle** 在同一个 Tile UCE 内同时 active。
2. 每个 execution context 内部仍保持 **in-order issue**。
3. UCE 仍保持 **单发射 shared issue pipeline**；不会在一个 cycle 内并发取指或并发 launch 两条指令。
4. architected baseline 下，context switch 只发生在：
   - 显式 `wait.event` 未满足；
   - `stream.pop` / `stream.acquire` 等现有 stream 阻塞点未满足；
   - 当前 context 已完成或进入终止路径，而另一个 context ready。
5. `engine credit / local resource` 导致的瞬时 backpressure 在 baseline 中先视为**当前 context stall**，而不是新的架构级切换语义；若后续要把“credit-stall 时切到另一个 ready context”作为优化加入，应单独标成实现可选项。
6. dual-context 准入建立在**上游 dependency/admission 决策 + UCE 本地 bind 校验**上，而不是 UCE 自己做 group scheduling。
7. L1 准入建立在 **Slot Frame partition compatibility** 上，而不是新增一个通用 `l1 allocator`。
8. event / stream 唤醒语义必须对齐现有 `event_id + expected_sequence` 和 `queue_id + producer_id + sequence_id` contract，避免 stale completion。

### 1.3 适用边界

这份 proposal 的定位是：

```text
V1.x execution-mode exploration
+ contract closure input
+ simulator / validator prototype target
```

不是：

```text
First Silicon V1 normative object-model replacement
```

### 1.4 非目标

本文**不**做以下事情：

1. 不引入 `Tile Group Program IR`。
2. 不在 UCE 内新增 `Central Program FIFO`。
3. 不定义一个脱离现有 ABI 的通用 `Token` 对象。
4. 不把 `STORE` 叙述成独立第五类 engine。
5. 不做 op-level out-of-order / sliding-window scheduler。
6. 不做硬件自动 memory alias prove。
7. 不做通用 L1 bump allocator / compacting allocator。
8. 不把 dual-context 直接写成 First Silicon 必须实现项。

---

## 2. 与现有对象模型的对齐

### 2.1 canonical hierarchy 不变

这份 proposal **不新增对象层**。执行对象仍然是：

```text
command buffer / descriptor / TileGroupTask / Tile Program
```

落点应收敛为：

```text
Runtime / Device Runtime
  -> TileGroupTask
    -> Tile Group Sequencer dispatch policy
      -> prepared tile task A / B
        -> resident Tile Program handle A / B
          -> Tile UCE dual-context execution mode
            -> engine launch / wait / stream / patch
```

### 2.2 ownership 不改写

| 决策 / 对象                                                       | owner                                                      | dual-context mode 只做什么                              |
| ----------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| inter-program dependency、role 顺序、group-level completion order | Compiler / Runtime / Tile Group Sequencer                  | **不接管**；UCE 不做 group scheduling                   |
| prepared tile task dispatch                                       | Tile Group Sequencer                                       | 向空闲 context 提交已经准备好的 tile task               |
| Tile Program residency / fetch / install                          | Program Residency Manager / Runtime / Tile Group Sequencer | UCE 只消费已经 valid 的 resident local handle           |
| Slot Frame bind contract                                          | Runtime / firmware bind，UCE enforce                       | 校验 frame generation / owner / partition compatibility |
| Stream Queue init / reset / drain                                 | Tile Group Sequencer                                       | UCE 只做 token protocol access                          |
| event table status / sequence                                     | Local Event Unit / Runtime event model                     | UCE 按 `event_id + expected_sequence` wait              |
| engine launch / wait / context switch                             | Tile UCE                                                   | 本 proposal 的唯一新增核心                              |

### 2.3 `await` 在本文里的含义

本文仍然会使用 `await` 这个词，但它只是**语义缩写**：

- `await local event` = 现有 `wait.event` / `waitall` 路径；
- `await stream token` = 现有 `stream.pop` / `stream.acquire` 的阻塞路径；
- **不是**新的 group-level ISA；
- **不是**新的通用 token object。

---

## 3. 核心执行模型

### 3.1 双 context、单发射

UCE 内部维护两个 execution context：

```text
CTX0
CTX1
```

但保留一个 shared fetch/decode/issue pipeline：

```text
single fetch
-> single decode
-> single issue
-> shared outstanding scoreboard / stream wait state
```

因此该模式的本质不是 superscalar，也不是 OoO，而是：

```text
single-core coroutine scheduler
with two resident execution contexts
```

### 3.2 每个 context 至少需要持有的语义状态

为避免把位宽和编码提前冻结，这里只定义**语义字段**，不定义最终 binary layout：

| 字段                                           | 含义                                                                             |
| ---------------------------------------------- | -------------------------------------------------------------------------------- |
| `valid`                                        | context 是否已绑定 active task                                                   |
| `ctx_id`                                       | 本地 context 编号，取值集合由后续规格冻结                                        |
| `program_local_slot`                           | resident program text 所在 local handle / slot                                   |
| `program_id / program_version / program_epoch` | resident validity 校验键                                                         |
| `pc`                                           | 当前 Tile Program PC                                                             |
| `frame_id / frame_generation`                  | 当前绑定 frame 身份，防 stale frame                                              |
| `frame_partition_id`                           | 当前使用的 Slot Frame partition                                                  |
| `desc_window_ref`                              | 当前可 patch / 可见的 descriptor window                                          |
| `wait_kind`                                    | `none / local_event / stream_token`                                              |
| `wait_ref`                                     | 当前等待对象的引用                                                               |
| `outstanding_mask`                             | 该 context 已发起、尚未 retire 的 local event / stream op 集合                   |
| `fault_state`                                  | `none / bind_fault / runtime_fault / timeout / reset` 等状态，编码由后续规格冻结 |
| `completion_event_ref`                         | 该 context 对应的 done/error 可见对象                                            |

### 3.3 context 状态机

建议的最小状态机：

```text
CTX_EMPTY
  -> CTX_BIND_PENDING
  -> CTX_READY
  -> CTX_WAIT_EVENT or CTX_WAIT_STREAM
  -> CTX_READY
  -> CTX_DONE
  -> CTX_EMPTY

error path:
CTX_BIND_PENDING / CTX_READY / CTX_WAIT_*
  -> CTX_FAULT
  -> CTX_DRAINING
  -> CTX_EMPTY or tile RESET
```

语义重点：

1. `CTX_BIND_PENDING` 期间完成 handle / frame / desc-window / partition 校验。
2. `CTX_WAIT_EVENT` 和 `CTX_WAIT_STREAM` 明确区分，避免把两类 wakeup 混成一个抽象 token。
3. `CTX_DRAINING` 是必要状态；一旦已有 engine launch 被接受，fault 路径就不能只清软件状态。

### 3.4 最小调度原则

architected baseline 下，UCE 每个周期只尝试在两个 **已经 active 且 ready** 的 context 之间做最小选择：

```text
runnable(ctx) :=
    ctx.valid
    && ctx.state == CTX_READY
```

建议的最小策略仍是 round-robin / alternating pick：

1. 当前 context 继续运行，直到它遇到**显式 wait 阻塞点**。
2. 当前 context 执行 `wait.event` 且条件未满足时，切到另一个 ready context。
3. 当前 context 在 `stream.pop` / `stream.acquire` 上阻塞时，切到另一个 ready context。
4. 当前 context 完成或进入终止路径时，若另一个 context ready，则切换过去。
5. 如果另一个 context 也不 ready，UCE idle，并等待 event/stream completion。

因此 baseline 的切换语义是：

```text
explicit wait / stream stall / task terminal transition
```

而不是：

```text
any transient local backpressure implies architected context switch
```

---

## 4. Wait object 与依赖语义

### 4.1 只保留两类等待对象

本文只保留两类等待对象：

1. **local event ref**
2. **stream token ref**

不再保留通用 `Token` 抽象。

### 4.2 local event ref 语义

`local event ref` 的语义字段至少应包括：

| 字段                | 说明                                                                                                                             |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `event_id`          | event table entry 标识                                                                                                           |
| `expected_sequence` | 防 event reuse / stale wakeup                                                                                                    |
| `producer_class`    | 调试 / PMU / fault triage 所需的 producer 类型；是否进入 compare key 由后续规格冻结                                              |
| `visibility_class`  | 若需要区分“engine done / buffer reusable / stronger visible”语义，可通过 descriptor / event class 关联；字段和编码由后续规格冻结 |

等待规则：

```text
wait.event(event_id, expected_sequence)
```

只有在：

```text
event_table[event_id].sequence == expected_sequence
&& status == DONE
```

时才允许 wakeup。若 status 为 `ERROR / TIMEOUT / RESET`，则进入 fault / trap 路径，而不是把它当成普通完成。

### 4.3 stream token ref 语义

`stream token ref` 的语义字段至少应包括：

| 字段                      | 说明                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `queue_id`                | Stream Queue 标识                                                                                            |
| `producer_id`             | token producer 身份                                                                                          |
| `sequence_id`             | 同 producer 单调递增序号                                                                                     |
| `reset_or_generation_tag` | 若 queue reset / drain 后 token 身份需要防复用，应携带 generation-like discriminator；具体编码由后续规格冻结 |

等待规则不是“读一个抽象 done bit”，而是：

```text
stream.pop / stream.acquire
only resumes when a matching token is visible
under existing queue ordering / EOS / error rules
```

也就是说：

1. wakeup 受 queue `queue_id + producer_id + sequence_id` 约束；
2. EOS / error token 仍按现有 Stream Queue contract 传播；
3. dual-context mode 不重新定义 stream ordering。

### 4.4 `await` 与 store completion 的收敛

旧稿里最容易漂移的是：

```text
await store_token
```

这在当前项目里不应保持为抽象写法。更可落地的约束是：

1. **不定义**一个通用 `store_token`。
2. 如果 lowering 需要“output slot 可复用”这个时点，必须引用一个**明确的 local event ref**。
3. 该 event ref 表示的是哪一层完成语义——例如：
   - `store accepted into ingress`；
   - `store data visible enough to release output slot`；
   - `stronger runtime-visible completion`；
     由对应 store path 的 descriptor / event class 冻结。
4. host-visible / runtime-visible completion 仍然归 Runtime Event Model，不由 dual-context proposal 发明新对象。

因此本文对 store 的态度是：

```text
reuse existing event model
instead of inventing a generic store token
```

---

## 5. Program handle、frame generation 与 descriptor window contract

### 5.1 dual-context bind 的核心不是“再塞一个 program”，而是“把第二个 bind contract 补实”

现有 UCE 单 active handle 语义已经要求：

- 校验 `program_local_slot / program_id / program_version / program_epoch`；
- 校验 `frame_generation`；
- 校验 `descriptor window`；
- 校验 `slot frame owner / permission / alignment / bank hint`。

dual-context 不是绕过这些检查，而是让 **两个 independent bind contract 并存**。

### 5.2 每个 context 的 bind 规则

| 对象                          | dual-context 下的最小规则                                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| resident program handle       | 每个 context 独立持有自己的 `program_local_slot + program_id/version/epoch`，bind 时必须逐个校验；resume 后若发现 epoch 不匹配，必须拒绝继续执行           |
| frame generation              | 每个 context 独立持有 `frame_id + frame_generation + owner + partition_id`；bind 时必须校验 current generation，reset / drain 后旧 generation 不得继续使用 |
| descriptor window             | 每个 context 必须有自己可验证的 descriptor window；V1.x 建议禁止两个 active context patch-write 同一窗口范围                                               |
| outstanding event namespace   | event table 可以共享，但 scoreboard entry 必须带 `owner_ctx` 或等价归属信息，避免 completion 唤醒到错误 context                                            |
| completion / error visibility | 每个 context 必须有自己可观察的 done/error 完成对象；但 post-issue fault 的恢复策略可保守升级为 tile-local drain                                           |

### 5.3 conservative rule：不在 dual-context 首版做“共享 patch-write window”

为降低落地难度，建议 V1.x 直接采用：

```text
active CTX0 and CTX1
must not patch-write overlapping descriptor ranges
```

允许共享的仅限：

1. immutable resident program text；
2. immutable const / metadata；
3. 只读 descriptor region；
4. event/status region 中按 event_id 隔离的 entry。

这样可以避免：

- 两个 context 竞争同一 patch shadow；
- partial descriptor visibility；
- patch ownership 漂移到 UCE 内部临时规则。

---

## 6. L1 准入收敛为 Slot Frame partition compatibility

### 6.1 不再以 `l1_footprint allocator` 作为 binding contract

旧稿里最不落地的一点，是把 dual dispatch 建立在：

```text
l1_footprint(P0) + l1_footprint(P1) <= available_l1
```

并进一步引出一个“两格 allocator”。

这会绕开现有 Slot Frame contract。更可落地的做法是：

> **dual-context 只在两个 Tile Program 都能绑定到预声明的 Slot Frame partition 时成立。**

### 6.2 最小 admission 输入

dual-context admission 至少需要以下输入，但这些输入的 carrier 可以继续沿用现有 prepared task / descriptor / frame 体系，不必发明新 IR：

| 输入                               | owner                                     | 用途                                                |
| ---------------------------------- | ----------------------------------------- | --------------------------------------------------- |
| `frame_partition_id`               | Compiler / Runtime / Tile Group Sequencer | 指出该 program 准备绑定到哪个预声明 partition       |
| `slot_usage_mask` 或等价 metadata  | Compiler                                  | 声明该 Tile Program 读/写哪些 slot role / slot 集合 |
| `frame_id + generation + owner`    | Runtime / firmware bind                   | 防 stale frame / owner mismatch                     |
| `desc_window_ref`                  | Runtime / Tile Group Sequencer            | 限定 patch / fetch 可见范围                         |
| `engine exclusivity class`         | Compiler / Runtime                        | 标记是否会占满某类本地独占资源                      |
| `dependency-free admission result` | Compiler / Runtime / Tile Group Sequencer | 明确这两个 program 允许共驻留                       |

### 6.3 建议的保守规则

V1.x 建议直接采用下面这套保守规则：

1. 每个 context 只能绑定到**一个预声明 partition**。
2. 两个 active context 的**可写 slot 集合必须互斥**。
3. 共享 resident/system slot 只允许：
   - program text；
   - descriptor / const；
   - event/status；
   - 明确声明为 resident/readonly 的 metadata。
     具体 slot 编号由后续规格冻结，不在本文拍死。
4. `BANK_PINNED`、alignment、owner、permission 仍由现有 Slot Frame checker 校验。
5. 如果两个 context 需要竞争同一可写 workspace / accumulator / output slot，则 admission 失败，不做动态搬移。

### 6.4 `l1_footprint` 还能不能保留

可以保留，但只能降级为：

```text
compiler/runtime scheduling hint
```

而不是：

```text
hardware binding contract
```

也就是说：

- `l1_footprint` 可用于上游 coarse filter；
- 真正决定 bind legality 的，仍然是 `Slot Frame partition + slot usage + owner/permission/generation`。

### 6.5 推荐的最小 admission 判定

```text
dual_context_admit(P0, P1) :=
    upstream_dependency_free(P0, P1)
    && program_handle_valid(P0)
    && program_handle_valid(P1)
    && frame_generation_current(P0)
    && frame_generation_current(P1)
    && desc_window_non_overlapping(P0, P1)
    && slot_partition_compatible(P0, P1)
    && engine_exclusive_class_compatible(P0, P1)
```

这里故意**不**包含 raw byte allocator。

---

## 7. UCE 本地调度边界：只做 runnable pick，不做 next-program selection

### 7.1 UCE 不拥有 `Central Program FIFO`

Dual-context mode 进入当前架构后，应该把问题切成两层：

#### 层 A：谁决定“下一个 program 能不能进来”

这是：

```text
Tile Group Sequencer dispatch policy / Runtime admission policy
```

的问题。

#### 层 B：两个已经 active 的 context，下一拍谁发射

这是：

```text
Tile UCE dual-context runnable pick
```

的问题。

本文只解决层 B，不把层 A 重新塞回 UCE。

### 7.2 推荐的边界切法

| 决策                                      | owner                                       | dual-context proposal 的约束                                     |
| ----------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| 哪两个 prepared tile task 允许共驻留      | Compiler / Runtime / Tile Group Sequencer   | 必须在进入 UCE 前决定                                            |
| 空闲 context 何时接收新 task              | Tile Group Sequencer                        | UCE 只暴露 `CTX_EMPTY / bind_reject / done / fault` 等可观察状态 |
| 两个 active context 谁先发射              | Tile UCE                                    | round-robin / simple fairness 即可                               |
| group-level lookahead / skip blocked head | Tile Group Sequencer future dispatch policy | 若以后要加，不属于 UCE first cut                                 |

### 7.3 为什么这是更可落地的切法

因为一旦 UCE 自己维护 `Central Program FIFO`，它就必须额外拥有：

- group-level eligibility check；
- group-level dependency visibility；
- completion order policy；
- FIFO HOL blocking 处理；
- group-level PMU attribution。

这些都已经越过了 Tile UCE 的 ownership 边界。

---

## 8. dual-context 的 fault / drain / reset 策略表

这部分是旧稿最缺的落地 contract。建议直接采用**保守而确定**、且明确区分 `未 accept` 与 `已 accept` 的策略。

### 8.1 总原则

1. **必须显式区分 `not accepted yet` 与 `accepted task`。**
2. **未 accept**：只保留**瞬时 admission / readiness 冲突**。例如：当前没有空 context、目标 partition 正被 active sibling 占用、某类 exclusivity-only local resource 当前不允许第二个 context 共驻留。此时 candidate task 还不是 active tile task，**没有 tile completion、没有 task fault terminal event**；Tile Group Sequencer 只会看到“本次未接收”，然后按自身 dispatch policy 继续保留、重试或顺序化。
3. `stale program handle / bad frame generation / invalid descriptor window` **不属于未 accept 桶**；一旦 task 已经被 UCE accept，它们都必须走 validation fault 的终止路径，而不是被降格成 backpressure。
4. **已 accept**：一旦 UCE 已经 latch 了该 context 的 task metadata，并让其成为 active task，那么后续即使 fault 发生在 `CHECK_PROGRAM_HANDLE / CHECK_FRAME_GENERATION / CHECK_DESC_WINDOW`，也必须走**正常 tile terminal path**：error/reset-like completion + fault record，而不是悄悄退回 scheduler。
5. **post-launch runtime fault** 直接升级为 tile-local drain domain。
6. tile reset / stream drain / frame generation invalidation 必须让两个 active context 都进入确定状态。
7. per-context selective kill 可以留到后续版本，不进入首版 cutline。

### 8.2 策略表

| 场景                                                               | faulting / candidate context 行为                                    | sibling context 行为                                                                                                         | 可观察结果                                                                                                                                                       | 说明                                                                                                                                                                                 |
| ------------------------------------------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| local admission gate 未通过，且 **task 尚未被 UCE accept**         | 不创建 active context；保持 `CTX_EMPTY`                              | 已 active sibling 不受影响                                                                                                   | **无 tile completion、无 task fault terminal event**；仅有本地 bind-reject / admission counter，Tile Group Sequencer 依据 dispatch policy 选择保留、重试或顺序化 | 这里只包含**瞬时 admission/readiness 冲突**，例如：无空 context、目标 partition 当前被占、exclusive local resource 当前忙；**不包含** stale handle / bad frame / invalid desc window |
| task **已 accept**，但在 `CHECK_PROGRAM_HANDLE` 失败               | 进入 `CTX_FAULT`，走该 task 的 error terminal path                   | 若 sibling 已 active，可继续；若实现更保守，也可暂时 freeze 到 fault record / completion 写回完成，但不要求 tile-local drain | 该 task 必须产出 error completion + fault record                                                                                                                 | 这里虽然还没 launch engine，但 task 已经存在 completion 语义，不能静默退回 scheduler                                                                                                 |
| task **已 accept**，但在 `CHECK_FRAME_GENERATION` / owner 校验失败 | 同上                                                                 | 同上                                                                                                                         | 该 task 必须产出 stale-frame / owner fault record + error completion                                                                                             | 防 reset/drain 后继续用旧 frame                                                                                                                                                      |
| task **已 accept**，但在 `CHECK_DESC_WINDOW` / patch legality 失败 | 同上                                                                 | 同上                                                                                                                         | 该 task 必须产出 invalid descriptor / patch fault + error completion                                                                                             | 避免进入 partial patch 状态                                                                                                                                                          |
| 任一 context 在已有 launch accepted 后发生 engine fault / timeout  | 进入 `CTX_FAULT -> CTX_DRAINING`                                     | **立即停止新 issue**，进入 `CTX_DRAINING`                                                                                    | UCE 进入 tile-local `drain_outstanding`；root-cause context 报 fault，sibling context 报 aborted/reset-like completion，具体编码由后续规格冻结                   | 这是最保守也最容易验证的共享 fault domain                                                                                                                                            |
| Stream Queue drain / reset 命中 active context                     | 等待 stream 的 context 转入 reset/error terminal path                | 另一个 active context 同步停止 issue 或按 drain policy 收敛                                                                  | 两个 active context 上持有的旧 stream wait ref 失效；active task 必须以 reset/error-like terminal path 可见                                                      | queue init/reset/drain ownership 仍在 Tile Group Sequencer                                                                                                                           |
| Tile reset                                                         | 两个 active context 全部 invalid                                     | 两个 active context 全部 invalid                                                                                             | outstanding event 写 `RESET / ERROR`；active frame invalid 或 generation 递增；旧 handle 失效；active task 以 reset/error-like terminal path 终止                | 与现有 Runtime / Slot Frame reset 语义一致                                                                                                                                           |
| 单个 context 正常完成                                              | 释放该 context 的 partition / desc window / scoreboard owner entries | 另一个 context 可继续运行                                                                                                    | 单 context done event 可单独可见                                                                                                                                 | 正常完成不需要拖累 sibling                                                                                                                                                           |

### 8.3 为什么首版只在 post-launch fault 上做 tile-local drain

因为当前架构里共享的东西很多：

- shared issue pipeline；
- shared scoreboard / event namespace；
- shared tile-local PMU / fault record path；
- 可能共享 descriptor / event/status resident region。

但也要区分 fault 发生的阶段：

1. **未 accept 前的 admission miss**：不是 task fault，只是本次不接收。
2. **已 accept、但尚未 launch engine 的 validation fault**：必须走该 task 的 error terminal path，但不一定需要把 sibling 一起拖入 drain。
3. **已 launch 后的 runtime fault**：直接视作 tile-local shared fault domain，进入 `drain_outstanding` 最稳妥。

因此首版建议的边界是：

```text
not accepted yet          => no completion semantics
accepted but pre-launch   => task-local error terminal path
post-launch runtime fault => tile-local drain domain
```

---

## 9. PMU / Debug / 可观测性

### 9.1 per-context counters

建议每个 context 至少维护：

```text
issued_mfe_load
issued_mfe_store
issued_boa
issued_evu
issued_use
wait_event_cycles
wait_stream_cycles
wait_engine_credit_cycles
bind_reject_program_handle
bind_reject_frame_generation
bind_reject_desc_window
bind_reject_partition_conflict
context_switch_out_count
task_completed_count
task_faulted_count
```

### 9.2 UCE global counters

建议 UCE 全局维护：

```text
uce_dual_ctx_active_cycles
uce_single_ctx_active_cycles
uce_both_ctx_blocked_cycles
uce_context_switch_count
uce_drain_outstanding_cycles
uce_fault_escalation_count
uce_bind_reject_count
```

### 9.3 engine taxonomy 保持现状

性能计数必须继续归属现有 engine taxonomy：

- `BOA`
- `EVU`
- `MFE`
- `USE`
- `Tile DMA`（若当前实现单独记）

因此旧稿里的：

```text
STORE active cycles
```

应改写为：

```text
mfe_store_active_cycles
or
dma_store_active_cycles
```

而不是暗示一个新的 `STORE` engine。

### 9.4 group-level dispatch 归因不属于 UCE

如果以后需要：

- central queue occupancy；
- head-of-line blocking；
- eligibility miss；
- dependency gate block；

这些计数应归：

```text
Tile Group Sequencer / runtime dispatch policy PMU
```

不是 dual-context UCE PMU。

---

## 10. 典型执行示例

### 10.1 合法示例：两个 independent resident Tile Programs 共驻留

假设上游已经决定 `Program A` 与 `Program B` 允许 dual-context 共驻留：

- `A` 绑定到 `partition P0`
- `B` 绑定到 `partition P1`
- 两者 descriptor patch window 不重叠
- 两者没有 tile-local data dependency

示意执行：

```text
CTX0 <- Program A
CTX1 <- Program B
```

`Program A` 的关键路径：

```text
launch.mfe.load   -> event_ref A_ld_done
wait.event        A_ld_done
launch.boa        -> event_ref A_cp_done
wait.event        A_cp_done
launch.mfe.store  -> event_ref A_store_reusable
wait.event        A_store_reusable
ret
```

`Program B` 的关键路径：

```text
launch.dma.load   -> event_ref B_ld_done
wait.event        B_ld_done
launch.evu        -> event_ref B_cp_done
wait.event        B_cp_done
ret
```

UCE 行为：

1. CTX0 发射 `launch.mfe.load`。
2. CTX0 进入 `wait.event(A_ld_done)`，若 event 未完成，则切到 CTX1。
3. CTX1 发射自己的 load / compute。
4. 两个 context 都只在显式 wait 点切换；没有 lookahead，没有重排。
5. `A_store_reusable` 的含义是“与 slot reuse 相关的 local event”，**不是** host-visible completion。

### 10.2 非法示例：存在依赖的两个 program 被尝试共驻留

若：

```text
Program B 读取 Program A 的 output slot
or
Program B 需要等待 Program A 的 local completion 才合法启动
```

则 V1.x 建议直接：

```text
不允许 dual-context 共驻留
```

处理方式只能是：

1. 同一个 context 顺序执行；或者
2. 等 `Program A` 完整完成后，由 Tile Group Sequencer 再 dispatch `Program B`。

换句话说，首版 cutline 不建议把“cross-context 显式 dependency”也一起做掉；那会把 admission、completion ordering 和 recovery 复杂度同时拉高。

### 10.3 非法示例：partition / descriptor window 冲突

即便两个 program 在数据流上 independent，只要满足任一条件，也必须拒绝 dual-context bind：

1. 两者 patch-write 同一个 descriptor shadow range；
2. 两者需要写同一个 output / workspace / accumulator slot；
3. 两者需要同一个 exclusivity-only local engine class；
4. 其中一个 program handle / frame generation 已 stale。

---

## 11. 落地路线

### 11.1 第一优先级：把 contract 补实，而不是先扩对象模型

最先该做的不是写新 IR，而是补齐以下 contract：

1. `await` 统一收敛为 `wait.event` / `stream wait` 两类语义。
2. dual-context bind 需要哪些最小字段：
   - program handle identity；
   - frame generation / owner / partition；
   - descriptor window；
   - completion/error visibility object。
3. post-issue fault 的 tile-local drain 策略。
4. PMU / debug / fault record 里如何区分 root-cause context 和 sibling aborted context。

### 11.2 第二优先级：做 V1.x prototype，而不是反向定义架构对象

可以用 `pipeline_validator` 或其他架构仿真环境做 prototype，但它只是：

```text
收益验证 / 复杂度验证 / PMU 归因验证
```

不是：

```text
对象模型的最终权威来源
```

prototype 路线建议：

1. 继续消费现有 `TileProgram` / `TileGroupTask` 层级；
2. 不引入 `tg.program`；
3. 用两个静态 Slot Frame partition 做 admission；
4. 只支持 `wait.event` 与现有 `stream` 阻塞点；
5. 先验证 overlap 收益是否值得进一步冻结硬件 contract。

### 11.3 第三优先级：若收益成立，再考虑扩展项

只有在上述 contract 已稳定、prototype 收益明确后，才建议继续讨论：

1. Tile Group Sequencer 侧更复杂的 dispatch lookahead；
2. 更细粒度的 per-context fault isolation；
3. store accepted / reusable / stronger visible 的多层 event class；
4. dual-context 之外的小 pending-set / ready-set。

---

## 12. 接受标准与最终建议

### 12.1 这份 proposal 若要算“更偏落地”，至少要满足

1. **不新增 group-level fetchable program object。**
2. **不新增 UCE 内 central FIFO。**
3. **所有 wait 语义都落回现有 event / stream contract。**
4. **L1 准入收敛为 Slot Frame partition compatibility。**
5. **program handle / frame generation / descriptor window / reset-drain-fault policy 明确。**
6. **PMU 仍沿用现有 engine ownership，不新增 `STORE` engine。**
7. **First Silicon V1 与 V1.x exploration 的边界明确。**

### 12.2 最终建议

当前阶段最合理的收敛结论是：

```text
Tile UCE Dual-Context Execution Mode
for Resident Tile Programs
```

它的本质是：

```text
在现有 TileGroupTask -> Tile Program -> Tile UCE 层级不变的前提下，
给 Tile UCE 增加一个双 context 的 coroutine-style 执行模式；
context switch 只发生在现有 wait/event/stream 阻塞点；
L1 准入由 Slot Frame partition 和 bind 校验保证；
post-issue fault 采用保守的 tile-local drain/reset 策略。
```

一句话定义：

```text
这不是新的 group IR，也不是小型 OoO scheduler；
它是一个建立在现有 resident handle、event identity、Slot Frame 和 Stream Queue contract 之上的 UCE 执行模式扩展。
```
