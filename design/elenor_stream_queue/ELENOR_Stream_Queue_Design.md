# ELENOR Stream Queue 设计文档

## 1. 定位、目标和 First Silicon cutline

Stream Queue 是可选的、有界 producer-consumer token/credit overlay，可为某些 Grid/Task 数据流提供背压、EOS 与 fault 状态；它不是每个 role、Task 或 transfer 的必经数据通路。ELENOR 当前主执行链是 Graph → root Context → Grid → finite Task → Tile Program → Engine，依赖与完成由显式事件/await/barrier、transfer 与 owner lifecycle 表达。

Stream Queue 不计算 payload、不拥有 L1/L2 storage，也不替代 view、event、pin、claim 或 Arena retirement。物理 FIFO、CDC、NoC packet 与 Tile Program 指令为未冻结硬件 v0 草案。当前软件 StreamQueue 只建模有限队列状态、token 元数据、credit、producer sequence/EOS、fault latch 与 reset generation。

| 能力            | 当前软件 overlay                                                                 | 尚未建模的硬件草案                          |
| --------------- | -------------------------------------------------------------------------------- | ------------------------------------------- |
| 队列作用        | 可选 per-queue producer/consumer ID 集与有限 depth                               | 全部 role 唯一数据通路、自动 fanout         |
| token / payload | token 保存 payload address/bytes 引用；payload 仍由 L1/L2 owner 管理             | inline payload、实体 FIFO、物理 payload DMA |
| credit          | 有界 valid-token credit；EOS/error 不占 payload credit；leased credit 计入不变量 | QoS refill、跨域 credit/CDC                 |
| EOS/error       | per-producer EOS bitmap；error latches first fault index                         | error packet 优先级、下游 fabric 转发       |
| reset           | queue 清空并递增 generation；由外层 reset 在隔离之后调用                         | tile/group/device 电路与 preemption/resume  |

## 2. 职责、非职责和 ownership

### 2.1 ownership matrix

| 对象                                                   | owner                                                | 当前合同                                                           |
| ------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------ |
| queue descriptor 中的 producer/consumer/depth/EOS 设置 | 编译/调用点配置；hardware v0 layout 未冻结           | Queue 模型只接收已验证构造参数，不负责 Graph/Task scheduling       |
| token、sequence、credit、EOS 与 fault latch            | optional `StreamQueue` 实例                          | bounded metadata/credit 状态；不包含完整 payload bytes             |
| payload backing 与 view                                | root Context L2 Arena / Task L1 Arena                | token 地址是引用；Queue 不分配、pin、publish、释放或持久化 payload |
| payload visibility                                     | issuing producer/consumer 的 transfer/event contract | Queue token 不代替 event/fence、`input_released` 或 `output_ready` |
| queue reset/generation                                 | Group reset lifecycle                                | 只在外层已 drain / cancel-confirm 并确认 owner isolation 后 reset  |
| FIFO/CDC/NoC/PMU hardware encoding                     | future hardware spec                                 | 本文 v0 descriptor/指令/寄存器不是 software artifact 或已实现 RTL  |

### 2.2 非职责

- 不复制 payload、不延长 Arena/view/backing/pin 生命周期；使用者必须显式满足 producer/consumer event 与 L1/L2 owner contract。
- 不提供 cache coherency；payload 可见性必须由适用的 transfer completion 与 event/fence 建立，token 仅传递引用/顺序。
- 不做 Graph、Context、Grid 或 Task scheduling，也不把所有 role 串成 Stream Queue pipeline。
- 不隐式恢复或吞掉错误；软件模型 latch first fault index，最终 Task/root fault 与隔离仍由外层 Runtime/Group 生命周期处理。

## 3. 微架构和状态机

### 3.1 当前 queue 状态与物理 FIFO 边界

当前 software `StreamQueue` 仅记录：

```text
queue_id / depth / producer IDs / consumer IDs
FIFO token metadata / credit_available / credit_leased
per-producer sequence + EOS bitmap
popped-but-unreleased valid-token count
first fault index / faulted / reset generation / PMU
```

它不拥有 token payload，也没有 per-consumer refcount、实际 NoC payload 路由或物理 CDC FIFO。Descriptor shadow、SRAM macro、payload fence scoreboard 和 ECC/parity 是未来硬件候选结构；Queue 的有限深度来自配置，产品 queue 数/深度仍需 SRAM/PPA 冻结。

### 3.2 producer token 生命周期状态机

下列 producer/consumer arrows 是可选协议示意，不是 Python `StreamQueue` 内部的逐周期状态机；该对象只实现 token/credit metadata 操作。

```text
P_IDLE
  -> P_ACQUIRE_CREDIT
  -> P_FILL_PAYLOAD
  -> P_PAYLOAD_FENCE
  -> P_PUSH_TOKEN
  -> P_WAIT_ACCEPT
  -> P_IDLE

EOS path:
P_IDLE -> P_PUSH_EOS -> P_EOS_SENT

ERROR path:
任意状态 -> P_PUSH_ERROR -> P_FAULTED
```

语义：

- `ACQUIRE_CREDIT` 成功时 `credit_available` 减一且 `credit_leased` 加一；valid token push 将 leased slot 计入 FIFO occupancy。
- 当前模型中 EOS 与 ERROR token 不消耗 payload credit；valid-token credit invariant 为 `credit_available + credit_leased + valid_tokens_in_fifo + popped_valid_not_released == depth`。
- EOS 更新相应 producer bitmap；all-producers EOS 依据配置的 producer 集判断。ERROR 在软件模型 latch first fault index 并标记 queue faulted，不等于物理 error packet 已送达 downstream。
- `sequence_id` 是同 producer 的单调序号；consumer 的 pop 返回 token metadata，不替 payload 做 load/fence。

### 3.3 consumer 状态机

下列 consumer arrows 是协议示意；软件 `pop` 返回 metadata，不执行 payload load、fence 或 downstream propagation。

```text
C_IDLE
  -> C_WAIT_TOKEN
  -> C_POP_TOKEN
  -> C_CHECK_FLAGS
  -> C_CONSUME_PAYLOAD
  -> C_RELEASE_TOKEN
  -> C_IDLE

EOS path:
C_CHECK_FLAGS(EOS) -> C_MARK_EOS -> C_IDLE 或 C_DONE

ERROR path:
C_CHECK_FLAGS(ERROR) -> C_RECORD_FAULT -> C_PROPAGATE_ERROR -> C_FAULTED
```

- `pop` 只从 FIFO 取 token 元数据；它不读取 payload，也不证明 payload bytes 已对 consumer 可见。
- 有效 token 进入 popped-but-unreleased 计数；`release` 恰好一次归还 credit。EOS/ERROR 没有 payload credit 可归还。
- Queue 只报告 token 与 credit 状态；payload 的 L1/L2 view、producer completion 和 consumer event 仍按外部 owner contract 校验。
- `BROADCAST` enum/硬件 descriptor 字段不等同于已实现多消费者 refcount 或 backing claim。

### 3.4 queue drain / reset generation contract

```text
正常运行：ACQUIRE -> PUSH(valid) -> POP -> RELEASE
EOS：     producer EOS bitmap 更新；不取得/归还 payload credit
ERROR：   latch first fault index + faulted
外层 reset：STOP 新 work -> drain / cancel-confirm accepted owners
          -> 确认 transfer、pin、view、Task/R lease 与 L2 claim 安全
          -> queue reset：empty + credit=depth + generation++
```

当前 `StreamQueue.reset()` 清 FIFO、归还 queue 内 bookkeeping credit、清 EOS/fault 状态并递增 token generation；它本身不取消或隔离 DMA、engine、Arena view、pin/backing 或 Task。调用者必须在其前完成 Group/Tile reset domain 的 owner drain/cancel-confirm。超时本身不构成隔离确认。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Stream descriptor/token binary ABI v0 (hardware draft)

```c
typedef enum {
    ELENOR_STREAM_TOKEN_VALID = 1u << 0,
    ELENOR_STREAM_TOKEN_EOS   = 1u << 1,
    ELENOR_STREAM_TOKEN_ERROR = 1u << 2,
    ELENOR_STREAM_TOKEN_FENCE = 1u << 3,
} elenor_stream_token_flags_t;

typedef enum {
    ELENOR_STREAM_Q_SPSC      = 0,
    ELENOR_STREAM_Q_MPSC      = 1,
    ELENOR_STREAM_Q_BROADCAST = 2,
} elenor_stream_queue_kind_t;

typedef enum {
    ELENOR_STREAM_EOS_SINGLE_PRODUCER = 0,
    ELENOR_STREAM_EOS_ALL_PRODUCERS   = 1,
    ELENOR_STREAM_EOS_PER_PRODUCER    = 2,
} elenor_stream_eos_policy_t;

typedef struct {
    uint16_t abi_version;
    uint16_t queue_kind;
    uint16_t eos_policy;
    uint16_t token_stride;
    uint32_t queue_id;
    uint32_t depth;
    uint32_t producer_mask;
    uint32_t consumer_mask;
    uint32_t payload_slot_id;
    uint32_t token_region_base;
    uint32_t token_region_bytes;
    uint32_t flags;
    uint32_t pmu_stream_id;
} elenor_stream_queue_desc_v0_t;

typedef struct {
    uint32_t token_id;
    uint32_t payload_addr;
    uint32_t payload_bytes;
    uint32_t flags;
    uint32_t producer_id;
    uint32_t sequence_id;
    uint32_t fault_record_index;
    uint32_t user_metadata;
} elenor_stream_token_v0_t;
```

software `StreamQueue` 使用相似的 in-memory token metadata，但其 Python object 不是上述 binary descriptor，也不序列化成硬件 ABI。当前 model 允许 configured producer/consumer ID sets；该类型字段不证明 broadcast/refcount payload sharing 已实现。

约束：硬件 descriptor 中 depth、stride、mask、payload slot 范围、错误字段和 ABI 宽度均为 v0 proposal；生产/消费元数据不授予 payload backing 所有权。

### 4.2 Tile Program stream instruction proposal

`STREAM_INIT/ACQUIRE/PUSH/POP/RELEASE/EOS/ERR/DRAIN/RESET` 的汇编示例是未来 ISA/硬件接口草案，不构成当前 source xDSL 或 compiled-program opcode list。software `StreamQueue` 的受限运行时对象只实现 token/credit/EOS/error/reset metadata 语义；真实 producer/consumer 仍需显式完成 payload transfer 与 event/fence。

### 4.3 寄存器和可观测状态

| 寄存器                      | 说明                                              |
| --------------------------- | ------------------------------------------------- |
| `SQ_CTRL[q]`                | enable、drain、reset、fault_latch_clear           |
| `SQ_STATUS[q]`              | running、empty、full、draining、faulted、eos_seen |
| `SQ_HEAD_TAIL[q]`           | head、tail snapshot                               |
| `SQ_CREDIT[q]`              | available、leased、inflight                       |
| `SQ_EOS_BITMAP[q]`          | producer EOS bitmap                               |
| `SQ_FIRST_FAULT[q]`         | first fault record index                          |
| `SQ_OCC_CYCLES[q]`          | occupancy weighted cycles                         |
| `SQ_CREDIT_EMPTY_CYCLES[q]` | producer acquire stall                            |
| `SQ_QUEUE_EMPTY_CYCLES[q]`  | consumer pop stall                                |
| `SQ_RESET_SEQ[q]`           | reset/drain generation counter                    |

上述寄存器名和字段是硬件 v0 草案，不对应当前 StreamQueue PMU/snapshot 的稳定 CSR ABI。

## 5. 数据流、控制流和时序路径

### 5.1 Optional token overlay on the current Task/transfer path

```text
root Context registers a bounded action / Grid Route
  -> finite Tasks independently acquire their Task L1 Arenas and execute
  -> explicit events, input_released/output_ready and transfer completion govern data visibility

optional producer Task:
  acquire queue credit -> prepare payload under its L1/L2 owner
  -> complete the required transfer/event/fence -> push token reference

optional consumer Task:
  pop token -> validate/use its payload reference under explicit owner/event rules
  -> release token credit
```

The queue is an overlay on a dataflow already governed by Context actions, Grid/Task admission, Arena owners, transfer completion and events. It does not launch every role, replace `nest.await` / `nest.barrier`, copy payload, pin a shared backing, or retire the producer's Arena when the token is released.

### 5.2 Ordering and dependency separation

- A producer's valid-token sequence is monotonic; tokens from different producers may interleave and imply no global order.
- `ALL_PRODUCERS` EOS is satisfied only when the configured producer bitmap is complete; EOS has no payload credit.
- The software ERROR push latches its first fault index and marks the queue faulted; downstream propagation is the outer Group/Context fault contract, not a promised StreamQueue packet hop.
- A token address/size is a reference only. Payload readability still requires its actual L1/L2 view owner, generation, transfer and event/fence to be valid.
- Reset increments queue generation and clears the FIFO; old `StreamToken` objects retain their old generation and callers must validate that tag before any later use/release. This queue bookkeeping does not isolate or retire a payload owner. `FENCE` metadata alone is not a memory visibility fence.

### 5.3 credit 和 backpressure

Credit invariant:

```text
credit_available + credit_leased
  + valid_tokens_in_fifo + popped_valid_not_released == depth
```

- Producer `acquire` stalls when no payload credit is available; producer `push` is not a second dynamic capacity allocator.
- EOS and ERROR do not consume/release payload credit; only popped VALID token `release` returns one credit.
- Consumer `pop` on empty reports queue-empty/stream-credit waiting. Queue wait is distinct from Arena `WAIT_CAPACITY`, Task `WAIT_SLOT`, engine queue credit or DMA/NoC wait.
- A double release/credit invariant error is observable in the queue PMU/state; this queue does not prove higher-level dependency graphs are deadlock-free.

### 5.4 EOS/error/reset behavior

| Token / reset | Current software overlay                                                 | Higher-level effect                                                        |
| ------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| VALID         | FIFO metadata pop/release; one payload credit returned on release        | Does not itself complete Task/Grid or free payload                         |
| EOS           | Per-producer bitmap and EOS token; no payload credit                     | Configured all-EOS condition is visible to caller                          |
| ERROR         | First fault index latched; queue becomes faulted                         | Outer Group/Context failure and cancel/drain owns propagation              |
| Queue reset   | Clear bookkeeping, restore credit, clear EOS/fault, increment generation | Must follow external accepted-work isolation; does not reset Arenas itself |

No specific hardware tile/group/device reset domain is claimed by this Python object. Group reset is responsible for stopping new work, draining accepted engines/transfers, confirming cancellation isolation, closing pins/views/claims/Task leases and only then resetting queues. A physical queue SRAM clear, CDC reset sequence and downstream error packet remain unmodeled.

## 6. 配置、PPA、性能模型和 PMU

### 6.1 Capacity boundary

The software model receives a finite queue `depth`, configured producer/consumer ID sets and EOS policy. The physical number of queues per Group, token SRAM bytes, queue port count and product depth are not frozen by the current 1 Group × 4 Tile model.

| Property            | Current model                                              | Hardware status                        |
| ------------------- | ---------------------------------------------------------- | -------------------------------------- |
| `depth`             | positive configured credit bound                           | per-product depth and token SRAM TBD   |
| producers/consumers | configured ID sets; sequence and EOS bitmap                | physical ports, broadcast/refcount TBD |
| payload storage     | external L1/L2 owner/view; queue stores reference metadata | payload DMA / slot pin integration TBD |
| clock/reset         | software cycle steps and reset generation                  | FIFO/CDC/RDC and reset waveforms TBD   |

### 6.2 Performance boundary

Queue occupancy/credit wait can bound an optional token pipeline only when producer/consumer rates and round-trip credit-release latency are separately known. It does not guarantee overlap, remove explicit dependencies, predict global schedule order or add payload bandwidth. The queue model provides cycle-accounted credit/empty/occupancy state; it is not a physical FIFO timing model.

### 6.3 PMU and attribution

Current software StreamQueue counters include `queue_full`, `queue_empty`, `occupancy`, `credit_full`, `credit_empty`, and `credit_fault`, plus the component stream-credit stall class. These are per queue/runtime component observations.

The `sq_*` register/counter names in this document remain candidate hardware PMU fields, not guaranteed Python report keys or hardware counter IDs. Attribute queue full/empty to its producer/consumer queue wait; do not merge them with NoC, Arena admission, DMA, event or engine stalls into a unique global cycle sum. Snapshots, not trace labels alone, establish queue state.

## 7. RTL/软件实现建议

Current executable semantics are the optional StreamQueue token/credit/EOS/error/reset model plus explicit L1/L2 view, transfer and event ownership outside the queue. Physical ready/valid FIFO, CDC, descriptor shadow, NoC packets, ECC, driver ABI and ISA instructions below remain proposals for later RTL/spec work.

- The physical queue descriptor may be loaded at init, but the software queue object does not implement a hardware descriptor shadow.
- Token addresses remain non-owning references; producer/consumer must perform explicit memory transfer, owner checks and event/fence ordering.
- A Group reset must stop new work and reach drain/cancel-confirm for accepted owners before queue reset; an unconfirmed timeout cannot retire a view/backing or R lease.
- Product queue ports, producer/consumer hardware IDs, async FIFO, error packet routing and ABI versioning remain to be specified.

## 8. 验证、bring-up 和验收标准

### 8.1 当前 software overlay contract

- Credit conservation includes acquired-but-not-pushed `credit_leased`; EOS/error metadata consumes no payload credit, and a released VALID token returns exactly one credit.
- Per-producer sequence/EOS state and first fault latch are observable; a queue error does not itself imply a physical downstream token was delivered.
- Queue reset restores empty/depth state and increments generation only after the Group reset path has safely isolated accepted work and owner references.
- Tokens never grant payload ownership; explicit backing/view, transfer completion, event/fence, pin and release contracts remain authoritative.
- A queue can be tested independently for acquire/push/pop/release/EOS/error/reset invariants; this does not prove all higher-level dependency graphs are deadlock-free.

### 8.2 Physical Queue/CDC bring-up (not yet performed)

Ready/valid stability, SRAM macro/ECC, CDC/RDC, NoC congestion, broadcast/refcount, physical reset and stream instruction RTL signoff remain future hardware work; no such proof is implied by the Python cycle model.

### 8.3 Cross-module contract checklist

- Scope: optional bounded token overlay, not the only path between roles or all Task outputs.
- Ownership: payload backing/view remains with L1 Task Arena or root L2 Arena; Queue does not manage claims, pinning, Store visibility or reclamation.
- Ordering: token order is producer-local; event/fence/transfer establish payload visibility; `input_released`, `output_ready`, Grid done and owner retirement remain distinct.
- Credit: invariant includes leased VALID credits; EOS/error do not consume payload credit; reset generation invalidates stale handles after isolation.
- Error/reset: queue first-fault state is distinct from Group/Context fault propagation and accepted-transfer cancel-confirm.
- PMU: model snapshot keys and hardware `sq_*` registers are separate namespaces.
- Hardware: descriptor/ISA/register encoding, FIFO/CDC/NoC/ECC/PPA and silicon behavior remain v0 proposals.

## 9. 风险、取舍和后续细化方向

| 风险                          | 影响                           | 缓解                                                                  |
| ----------------------------- | ------------------------------ | --------------------------------------------------------------------- |
| credit leak                   | queue 永久 full 或 credit 虚增 | invariant counter、reset reconcile、formal proof                      |
| EOS 语义含混                  | role 过早结束或无法结束        | descriptor 中显式 EOS policy 和 producer mask                         |
| multi-consumer 过早实现       | refcount/reset 验证爆炸        | broadcast/refcount 是未来硬件扩展；当前 queue 不拥有多 reader backing |
| error token 被普通 token 淹没 | fault 延迟不可控               | error latch + VC0 fault fabric + group task policy stop queue         |
| payload/token coherency 混淆  | consumer 读旧数据              | payload fence、event ordering、SVA 可见性检查                         |
| PMU 归因重复                  | 性能调优误判                   | primary stall owner + secondary debug tag                             |

当前软件 overlay 的 credit/EOS/error/reset 状态以本节明确的实现合同为准：valid token 占用 credit，EOS/error 不占 payload credit；物理 FIFO/CDC、broadcast/refcount、NoC packet、watchdog 与产品深度仍由后续硬件规格冻结。
