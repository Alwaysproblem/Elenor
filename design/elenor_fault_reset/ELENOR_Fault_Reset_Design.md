# ELENOR Fault / Reset 设计文档

## 1. 定位、目标和 First Silicon cutline

Fault / Reset 子系统定义 ELENOR 在 invalid descriptor、address fault、DMA timeout、event deadlock timeout、SRAM ECC、NoC poison packet、engine internal fault、stream credit leak 等异常下的记录、传播、drain、reset 和重新进入规则。它是 command -> DMA -> engine -> event -> PMU 端到端可靠性的基础。

设计目标：

- 错误必须可定位：fault record 绑定 command id、context id、program id、tile/group、descriptor、event、queue、slot。
- 错误必须可传播：stream error token、event fault bit、NoC poison/fault packet、interrupt 都有确定语义。
- reset 必须可收敛：tile reset、group reset、device reset 下 token、credit、event、DMA outstanding、NoC packet、PMU snapshot 有确定状态。
- fault path 不依赖 bulk data drain；VC0 command/event/fault 必须在 VC2 数据拥塞时仍可前进。

First Silicon cutline：

| 能力              | First Silicon V1                                                       | V1.x / V2 保留                            |
| ----------------- | ---------------------------------------------------------------------- | ----------------------------------------- |
| fault record      | basic fault record ring、first fault latch、event fault bit            | hierarchical trace packet、sampling trace |
| reset domain      | tile、group、device 三档                                               | context-only preemption、resume           |
| drain             | queue stop、DMA cancel/drain、credit reconcile、event completion/fault | retryable partial replay                  |
| error propagation | Stream Queue error token、NoC poison/fault packet、driver interrupt    | QoS-aware fault containment               |
| PMU               | fault counter、reset cycles、drain timeout、stall attribution          | PMU feedback scheduler                    |

## 2. 职责、非职责和 ownership

### 2.1 ownership matrix

| 组件                            | 责任                                                                                                    | 不负责                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- | --------------------- |
| Kernel driver                   | interrupt handling、fault record readout、context teardown、device reset request                        | Tile 内 drain 细节    |
| Firmware / Runtime processor    | command queue consume、descriptor validation、fault aggregation、reset/drain policy、profiling snapshot | engine datapath 修复  |
| Global Scheduler / Event Fabric | event fault bit、dependency unblock/poison、timeout monitor                                             | 数据 payload 修正     |
| Tile Group Sequencer            | group 内 role stop、stream queue drain、group reset sequencing                                          | 高层 graph 重新调度   |
| Tile UCE                        | Tile Program stop、engine cancel、descriptor patch fault、tile reset sequencing                         | HBM memory policy     |
| Stream Queue Engine             | error token、credit reconcile、EOS/error/reset 状态                                                     | payload 数据修复      |
| DMA / MFE                       | address fault、timeout、poison completion、outstanding cancel                                           | command policy        |
| NoC router                      | poison packet forwarding、VC credit recovery、per-VC fault PMU                                          | descriptor validation |
| PMU                             | fault/reset counter、snapshot、stall owner                                                              | 决策是否 reset        |

### 2.2 非职责

Fault / Reset 不提供透明 retry 保证，不隐式恢复 user context，不隐藏硬件 fault，不把所有错误升级为 device reset。Runtime 选择 reset tile、reset group 或 reset device；硬件必须给出足够状态让该选择可验证。

## 3. 微架构和状态机

### 3.1 fault handling pipeline

```text
FAULT_DETECT
  -> CAPTURE_LOCAL_CONTEXT
  -> LATCH_FIRST_FAULT
  -> STOP_AFFECTED_ISSUE
  -> PROPAGATE_ERROR
  -> DRAIN_OR_CANCEL
  -> WRITE_FAULT_RECORD
  -> SIGNAL_EVENT_OR_INTERRUPT
  -> WAIT_RESET_POLICY
```

检测点：descriptor validator、slot frame checker、DMA address generator、memory controller ECC/poison、NoC router poison/credit checker、Stream Queue invariant checker、event timeout monitor、engine wrapper assertion。

### 3.2 reset domain 状态机

```text
RUNNING
  -> RESET_REQUESTED
  -> QUIESCE_ISSUE
  -> SNAPSHOT_PMU_AND_FAULT
  -> DRAIN_OUTSTANDING
  -> CANCEL_UNDRAINABLE
  -> RECONCILE_CREDITS_EVENTS
  -> CLEAR_LOCAL_STATE
  -> RELOAD_OR_REBIND
  -> RESET_DONE
  -> RUNNING

任何状态 -> RESET_FAILED 当 watchdog 超时
```

reset domain：

- Tile reset：影响单 tile 的 UCE、USE state shadow、engine wrappers、Tile DMA、L1 frame shadow、tile-local stream handles。
- Group reset：影响 Tile Group Sequencer、Stream Queue Engine、Barrier/Event Engine、Group DMA、Group SRAM volatile region、group collective。
- Device reset：影响全局 scheduler、command/event fabric、NoC global state、memory controller device-side queues、全部 groups。

### 3.3 drain 和 cancel 状态机

```text
DRAIN_IDLE
  -> STOP_NEW_WORK
  -> WAIT_ENGINE_IDLE
  -> WAIT_DMA_COMPLETION
  -> WAIT_STREAM_RELEASE
  -> WAIT_NOC_CREDIT_RETURN
  -> COMPLETE_DRAIN

超时边：任意 WAIT -> CANCEL_OUTSTANDING -> POISON_COMPLETIONS -> COMPLETE_DRAIN 或 RESET_FAILED
```

原则：

- 可安全完成的 command 可以 drain；不可证明安全的 outstanding 必须 cancel 并 poison completion。
- drain 不得无限等待；watchdog 值由后续规格冻结。
- drain_done 前必须冻结 PMU snapshot，否则 fault 前后的 stall 归因不可解释。

## 4. 接口、descriptor、寄存器和协议

### 4.1 fault code 和 fault record v0

```c
typedef enum {
    ELENOR_FAULT_NONE                = 0,
    ELENOR_FAULT_INVALID_DESCRIPTOR  = 1,
    ELENOR_FAULT_ADDRESS             = 2,
    ELENOR_FAULT_DMA_TIMEOUT         = 3,
    ELENOR_FAULT_EVENT_TIMEOUT       = 4,
    ELENOR_FAULT_STREAM_CREDIT       = 5,
    ELENOR_FAULT_STREAM_ERROR_TOKEN  = 6,
    ELENOR_FAULT_SLOT_PERMISSION     = 7,
    ELENOR_FAULT_SLOT_BOUNDS         = 8,
    ELENOR_FAULT_SRAM_ECC            = 9,
    ELENOR_FAULT_NOC_POISON          = 10,
    ELENOR_FAULT_NOC_CREDIT          = 11,
    ELENOR_FAULT_ENGINE_INTERNAL     = 12,
    ELENOR_FAULT_RESET_TIMEOUT       = 13,
} elenor_fault_code_t;

typedef enum {
    ELENOR_FAULT_SRC_RUNTIME = 0,
    ELENOR_FAULT_SRC_GROUP_TASK  = 1,
    ELENOR_FAULT_SRC_TILE_UCE = 2,
    ELENOR_FAULT_SRC_STREAM  = 3,
    ELENOR_FAULT_SRC_DMA     = 4,
    ELENOR_FAULT_SRC_MFE     = 5,
    ELENOR_FAULT_SRC_NOC     = 6,
    ELENOR_FAULT_SRC_SRAM    = 7,
    ELENOR_FAULT_SRC_ENGINE  = 8,
} elenor_fault_source_t;

typedef struct {
    uint16_t abi_version;
    uint16_t code;
    uint16_t source;
    uint16_t severity;
    uint32_t fault_record_index;
    uint32_t context_id;
    uint32_t command_id;
    uint32_t event_id;
    uint32_t program_id;
    uint32_t desc_id;
    uint16_t group_id;
    uint16_t tile_id;
    uint16_t queue_id;
    uint16_t slot_id;
    uint32_t patch_id;
    uint32_t engine_id;
    uint64_t offending_addr;
    uint64_t aux0;
    uint64_t aux1;
    uint64_t pmu_snapshot_ptr;
} elenor_fault_record_v0_t;
```

`severity` 建议：recoverable tile、recoverable group、fatal device、software bug。编码由后续规格冻结。

### 4.2 reset command ABI v0

```c
typedef enum {
    ELENOR_RESET_TILE   = 0,
    ELENOR_RESET_GROUP  = 1,
    ELENOR_RESET_DEVICE = 2,
} elenor_reset_domain_t;

typedef enum {
    ELENOR_RESET_F_DRAIN_FIRST      = 1u << 0,
    ELENOR_RESET_F_CANCEL_INFLIGHT  = 1u << 1,
    ELENOR_RESET_F_CLEAR_PMU        = 1u << 2,
    ELENOR_RESET_F_RELOAD_PROGRAM   = 1u << 3,
    ELENOR_RESET_F_REBIND_FRAME     = 1u << 4,
} elenor_reset_flags_t;

typedef struct {
    uint16_t abi_version;
    uint16_t domain;
    uint32_t flags;
    uint32_t context_id;
    uint16_t group_id;
    uint16_t tile_id;
    uint32_t reason_fault_record_index;
    uint32_t timeout_cycles;
} elenor_reset_cmd_v0_t;
```

### 4.3 event、stream、NoC 协议

Event fault bit：

```c
typedef enum {
    ELENOR_EVENT_COMPLETE = 1u << 0,
    ELENOR_EVENT_FAULT    = 1u << 1,
    ELENOR_EVENT_TIMEOUT  = 1u << 2,
    ELENOR_EVENT_CANCELED = 1u << 3,
} elenor_event_flags_t;
```

Stream error token：必须设置 `ELENOR_STREAM_TOKEN_ERROR` 并携带 `fault_record_index`。如果上游 fault 已停止 queue，downstream stage 必须看到 error token 或 event fault，不允许永久等待 EOS。

NoC fault/poison：

- VC0 承载 command、event、fault packet，最高优先级。
- VC1 承载 DMA read response；poison read response 必须返回 descriptor 对应 source。
- VC2 承载 DMA write / stream；bulk data poison 不得阻塞 VC0 fault packet。
- VC3 承载 collective；collective epoch 中出现 fault 必须 poison epoch completion。

### 4.4 控制寄存器

| 寄存器                                                       | 说明                                                     |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| `FAULT_RING_BASE` / `FAULT_RING_SIZE`                        | fault record ring 地址和容量                             |
| `FAULT_RING_HEAD_TAIL`                                       | firmware 写 head，driver 读 tail 或相反，方向由 ABI 冻结 |
| `FIRST_FAULT_LATCH`                                          | 当前 domain first fault index                            |
| `RESET_CTRL`                                                 | request、domain、drain/cancel flags                      |
| `RESET_STATUS`                                               | idle、quiesce、drain、cancel、clear、done、failed        |
| `RESET_WATCHDOG`                                             | reset/drain timeout，值由后续规格冻结                    |
| `DOMAIN_QUIESCE_MASK`                                        | 被停止的新 work issue 源                                 |
| `OUTSTANDING_DMA` / `OUTSTANDING_NOC` / `OUTSTANDING_STREAM` | drain 观测计数                                           |
| `ERROR_INT_STATUS`                                           | driver interrupt source                                  |

## 5. 数据流、控制流和时序路径

### 5.1 invalid descriptor path

1. Descriptor validator 检测 ABI mismatch、字段越界、slot permission、patch owner 不匹配。
2. Tile UCE 或 firmware latch first fault。
3. 阻止 engine launch；不发出部分 DMA/NoC transaction。
4. 写 fault record，event 设置 FAULT。
5. Tile Group Sequencer 停止受影响 queue，按 policy drain。
6. Driver interrupt 或 runtime polling 观察 fault。

### 5.2 DMA timeout / address fault path

1. DMA issue 前做地址范围和 slot/IOVA 检查。
2. issue 后 memory controller 或 NoC 返回 timeout/poison。
3. DMA 停止该 descriptor 新 beat；已完成 beat 保持已完成状态，未完成 beat cancel。
4. completion event 不作为 success 置位；event 设置 FAULT 或 CANCELED。
5. 若 payload 已进入 Stream Queue，producer 必须推 error token 或 queue 被 group fault 关闭。

### 5.3 stream fault path

1. Stream invariant checker 检测 credit leak、double release、sequence gap、reset generation mismatch。
2. Queue 进入 FAULTED，停止新普通 token。
3. 已有 consumer 若等待 token，收到 error token 或 event fault。
4. credit reconcile，记录 canceled token 数。
5. group 或 tile reset 根据 fault severity 执行。

### 5.4 ordering / coherency rules

- Fault record 写入必须先于 event FAULT 可见，driver 读 event 后必须能读到对应 record。
- VC0 fault packet 不要求等待 VC2 bulk data drain；但同一 source 的 fault record index 必须全局唯一。
- Reset generation 递增后，旧 descriptor、slot frame、stream token handle 均失效。
- SRAM ECC uncorrectable 对应 cache line / SRAM row 必须 poison；后续 engine read 返回 fault，不得静默返回数据。
- descriptor cache invalidate 是 warm patch 和 reset reload 的必需步骤。
- PMU snapshot 必须在 clear local state 前完成。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 容量和面积假设

| 结构                       | First Silicon 建议             | 面积/容量说明                          |
| -------------------------- | ------------------------------ | -------------------------------------- |
| fault record ring          | 64 到 256 entries / device     | 由后续规格冻结                         |
| per-tile first fault latch | 1 entry / tile                 | 小寄存器，必须 always-on 或 reset-safe |
| per-group reset controller | 1 / group                      | 包含 drain counters 和 watchdog        |
| outstanding counters       | DMA、NoC、Stream、Event 各一组 | 宽度由最大 outstanding 冻结            |
| PMU snapshot buffer        | per domain compact snapshot    | 由 SRAM profile 冻结                   |

### 6.2 latency 和 timeout 模型

```text
T_fault_visible = T_detect + T_record_write + T_event_or_interrupt
T_reset = T_quiesce + T_pmu_snapshot + T_drain_or_cancel + T_clear + T_rebind
```

要求：

- `T_event_or_interrupt` 不应被 VC2 bulk data 无界放大。
- watchdog 必须大于合法最大 outstanding latency，小于 software timeout budget；具体值由后续规格冻结。
- reset clear 不应清掉 fault record ring，除非 driver 明确 ack。

### 6.3 PMU / error hooks

必需 counter：

- `fault_count_by_code[code]`
- `fault_count_by_source[source]`
- `reset_count_by_domain[domain]`
- `reset_cycles_by_domain[domain]`
- `drain_timeout_count`
- `canceled_dma_desc_count`
- `canceled_stream_token_count`
- `noc_poison_packet_count_by_vc[4]`
- `event_timeout_count`
- `sram_ecc_correctable_count`
- `sram_ecc_uncorrectable_count`
- `descriptor_invalid_count`
- `slot_fault_count`

PMU primary stall owner 仍遵循全局规则。进入 fault drain 后，后续周期归为 reset/drain，不再归给原 engine active/stall，避免双重计数。

## 7. RTL/软件实现建议

- Fault path 使用 sideband valid/ready，并保证能够进入 VC0 或本地 fault ring；不要通过 VC2 bulk data path 上报唯一错误。
- First fault latch 保留最早错误；后续错误计数和 secondary record 可选，但不能覆盖 first fault。
- Reset controller 分 domain 实现，tile reset 不应清掉其他 tile 的 stream queue，除非 queue 是共享 group queue 且被标记 affected。
- Descriptor validator、slot checker、stream checker、DMA checker 都输出统一 fault bus。
- Firmware reset sequence 使用表驱动，按 domain quiesce mask 停止新 work，再 drain/cancel。
- Driver 在 device reset 前读取 fault ring 和 PMU snapshot；device reset 后重建 command queue、event table、stream descriptors 和 frame binding。
- RTL 中所有 outstanding counter 都有 saturating error 检测；counter underflow/overflow 直接 fault。
- NoC credit state 在 reset 后必须由本地已知初值重建，不依赖 remote 旧状态。

## 8. 验证、bring-up 和验收标准

### 8.1 SVA / formal checks

- Fault record before event：event FAULT 置位前，对应 fault record valid 必须为真。
- First fault stable：first fault latch 在 driver/firmware ack 前不得被后续 fault 覆盖。
- No success after fault：同一 command 的 success completion 与 fatal fault 不得同时上报。
- Reset progress：若 watchdog 未超时且 downstream 响应公平，reset eventually done。
- Credit reconcile：reset done 后 Stream Queue credit invariant 恢复。
- NoC VC0 progress：VC2 full 时 VC0 fault/event packet 仍有 bounded progress。
- Outstanding non-negative：DMA/NoC/Stream outstanding counter 不得 underflow。
- Poison propagation：poison packet 必须导致 poison response、error token 或 fault record。
- Generation invalidation：reset generation 增加后旧 token/descriptor/frame handle 不得成功提交。

### 8.2 fault injection matrix

| 注入点                      | 预期行为                        | 验收                                         |
| --------------------------- | ------------------------------- | -------------------------------------------- |
| invalid descriptor ABI      | command fault，不 launch engine | fault record 字段完整                        |
| slot bounds violation       | Tile UCE patch fault            | event FAULT，slot id 正确                    |
| DMA timeout                 | DMA cancel/drain                | no success completion，PMU timeout 增加      |
| event deadlock timeout      | affected queue stop             | runtime 可 reset group                       |
| stream credit leak          | queue FAULTED                   | reset 后 credit=depth                        |
| NoC poison VC2              | fault packet VC0 上报           | VC0 不被 VC2 阻塞                            |
| SRAM ECC uncorrectable      | poison row / fault              | engine 不消费静默坏数据                      |
| engine internal fault       | engine stop + tile reset        | other tile 不受污染，除非共享 queue affected |
| reset while DMA outstanding | cancel 或 drain                 | outstanding=0 后 reset done                  |

Bring-up 顺序：Phase 1 即打通 command queue + event + barrier + DMA + basic PMU + fault record；Phase 3 增加 Stream Queue EOS/error token；Phase 6 增加 multi-context fault isolation。

### 8.3 跨模块 contract checklist

- Binary struct / protocol：fault record、reset command、event flags、stream error token 和 NoC poison/fault packet 行为均有 v0 草案或明确字段。
- State machine：fault pipeline、reset domain、drain/cancel 必须有 RTL 状态枚举、watchdog 和 coverage。
- Capacity / bandwidth / area：fault ring、first fault latch、outstanding counter、PMU snapshot buffer 与 reset controller 面积分开估算；未冻结深度由后续规格冻结。
- NoC VC behavior：fault/event/reset control 必须走 VC0；VC2 bulk stream 或 DMA write 拥塞不得阻塞 fault record 可见性；VC1/VC3 poison 必须映射到 fault record。
- Credit / EOS / error / reset：reset/drain 必须回收 stream credit；EOS 不得被 reset 静默伪造；error token 必须携带 fault record index；reset generation 使旧 token/frame/descriptor 失效。
- Patch ownership：descriptor patch fault 记录 owner；Runtime/firmware 发起 context patch，Tile UCE 执行 descriptor patch，MFE 负责 page/segment 动态地址，Fault / Reset 只记录和隔离错误。
- Ordering / coherency：fault record valid 先于 event FAULT；PMU snapshot 先于 clear；descriptor cache invalidate 先于 reset reload/warm relaunch。
- SVA / formal：first fault stable、no success after fault、VC0 progress、credit reconcile、poison propagation、generation invalidation 必须覆盖。
- PMU / error hooks：fault/reset 期间 stall 归入 reset/drain owner，避免与原 engine 计数重复。

## 9. 风险、取舍和后续细化方向

| 风险                             | 影响                        | 缓解                                             |
| -------------------------------- | --------------------------- | ------------------------------------------------ |
| fault 上报依赖拥塞数据 VC        | driver 看不到错误，系统挂死 | VC0 独立、fault sideband、formal progress        |
| reset drain 无限等待             | recovery 不收敛             | watchdog、cancel path、outstanding snapshot      |
| first fault 被覆盖               | root cause 丢失             | first latch stable，secondary counter 分离       |
| tile reset 污染 group 共享队列   | 其他 context 被误伤         | affected mask、generation、queue owner 检查      |
| PMU 在 reset 中清零过早          | 性能/故障不可解释           | snapshot before clear，driver ack 后清理         |
| retry 语义过早引入               | 验证复杂度失控              | First Silicon 使用 fail-stop + drain/cancel      |
| software/hardware ownership 不清 | reset domain 选择不一致     | fault severity、domain policy table、ABI version |

后续需要冻结：fault severity 编码、fault ring 方向和 wrap 规则、watchdog 值、reset domain affected mask 编码、NoC poison packet 精确格式、SRAM ECC policy、retry 是否进入 V1.x、context isolation 与 multi-model priority 的交互。
