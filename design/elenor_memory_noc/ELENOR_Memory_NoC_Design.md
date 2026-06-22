# ELENOR Memory / NoC 设计文档

## 1. 定位、目标和 First Silicon cutline

Memory / NoC 子系统是 ELENOR 从 host memory、HBM/DDR/LPDDR 到 Group L2、Tile L1、engine operand buffer 的确定性数据通路。它不解释高层 graph，也不直接决定 workload 调度；硬件只执行 command、descriptor、Region Program 和 Tile Program 产生的 DMA、stream、event 和 collective 请求。

设计目标：

- 为 BOA、EVU、MFE、USE、UCE、DMA 和 collective 提供可归因、可限流、可恢复的片上互联。
- 通过分层 SRAM、bank-aware layout 和 NoC virtual channel，避免大数据流阻塞 command、event、barrier 和 fault 上报。
- 给 compiler/runtime 提供可建模的容量、带宽、延迟、排序和 coherency contract。
- 给 RTL 和验证提供二进制 descriptor、状态机、PMU counter、SVA/formal 入口。

Architecture V1 覆盖 Edge、Balanced、High End 三档配置；First Silicon V1 只冻结能闭合面积、时序、功耗和验证的 profile。未冻结字段写为 `由后续规格冻结` 或 `由 SRAM profile 冻结`。

First Silicon cutline：

| 项目     | First Silicon V1                                                               | V1.x / V2 保留                                           |
| -------- | ------------------------------------------------------------------------------ | -------------------------------------------------------- |
| 外部内存 | 单 HBM stack 或高带宽 DDR/LPDDR profile，由封装选择冻结                        | 多 HBM stack、CXL.mem、chiplet memory fabric             |
| NoC 拓扑 | Edge crossbar / small mesh；Balanced 2D mesh 或 hierarchical mesh 的最小子集   | adaptive routing、QoS scheduler、multi-die global fabric |
| VC       | VC0 command/event、VC1 DMA read response、VC2 DMA write/stream、VC3 collective | 更多 QoS / debug / isolation VC                          |
| DMA      | 1D、2D、strided copy、async completion event                                   | multicast、gather list、coherent host page migration     |
| SRAM     | L1 slot frame、Group L2 stream/prefetch buffer、bank conflict PMU              | 动态 SRAM repartition、SRAM compression                  |
| 错误恢复 | poison/fault record、tile/group/device reset domain、drain                     | preemption、retry-once memory replay                     |

## 2. 职责、非职责和 ownership

### 2.1 ownership matrix

| 组件                         | 拥有内容                                                                      | 不拥有内容                               |
| ---------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------- |
| Host driver                  | IOVA 分配、pinning、IOMMU 映射、doorbell、interrupt、PMU readout              | Tile 内 buffer 编排                      |
| Firmware / Runtime processor | command consume、descriptor validation、fault aggregation、reset/drain policy | BOA/EVU/MFE datapath 内部调度            |
| Global DMA                   | HBM/DDR/LPDDR 与 Group L2 / device memory 之间的大粒度 copy                   | L2 到 L1 的 tile-local copy              |
| Memory controller            | 外部内存事务调度、ECC、read/write completion、poison 标记                     | command/event 语义                       |
| NoC router                   | VC 仲裁、credit、route、deadlock avoidance、per-VC PMU                        | tensor layout、page table walk           |
| Group DMA                    | HBM 到 L2 预取、weight/activation pipeline prefetch                           | Tile Program PC                          |
| Tile DMA                     | L2 到 L1 copy、storeback、slot frame address check                            | HBM 全局 memory policy                   |
| MFE                          | 数据相关动态内存访问、page/segment stream、reorder、stream fill               | 任意图遍历、Tile Program 主控制          |
| Tile UCE                     | launch/wait/fence、descriptor patch、slot binding、DMA/engine 编排            | page table walk 和大多数动态数据地址生成 |
| PMU                          | bandwidth、stall、congestion、bank conflict 和 fault counter 聚合             | 调度决策本身                             |

### 2.2 非职责

Memory / NoC 不提供 CPU cache coherency，不实现高层 graph memory planner，不隐式修复 descriptor 地址错误，不在数据通路里解析 tensor algebra。compiler/runtime 必须通过 descriptor、slot frame、stream queue 和 event 明确表达生命周期、权限、依赖和同步。

## 3. 微架构和状态机

### 3.1 分层数据路径

```text
Host Memory / HBM / DDR / LPDDR
        |
        v
Global Memory Controller
        |
        v
Global DMA / NoC ingress
        |
        v
Group Shared SRAM / L2
        |
        v
Tile DMA / MFE tile port
        |
        v
Tile Local SRAM / L1 slot frame
        |
        +--> Program / Descriptor / Event region
        +--> BOA operand / accumulator buffer
        +--> EVU vector buffer
        +--> MFE stream buffer
        +--> USE state cache
        +--> DMA staging buffer
```

### 3.2 NoC router pipeline

每个 router 建议拆成以下 pipeline stage：

1. ingress flit capture：按 input port 和 VC 接收 flit，更新 input credit。
2. header decode：解析 destination、VC、packet type、context id、poison、sequence。
3. route compute：维持 deterministic route；First Silicon 使用固定 XY 或 hierarchical deterministic route。
4. VC allocation：为 packet 分配 output VC；同 VC packet 保持 order。
5. switch allocation：按 age、VC priority 和 starvation counter 仲裁。
6. switch traversal：flit 穿越 crossbar。
7. egress credit update：只有 downstream credit 可用时发送。
8. PMU/error tap：记录 per-VC occupancy、stall、poison、drop-prohibited violation。

### 3.3 DMA request 状态机

```text
IDLE
  -> DESC_FETCH
  -> DESC_VALIDATE
  -> ADDR_TRANSLATE
  -> ISSUE_READ_OR_WRITE
  -> WAIT_COMPLETION
  -> WRITEBACK_STATUS
  -> SIGNAL_EVENT
  -> IDLE

错误边：
DESC_VALIDATE / ADDR_TRANSLATE / WAIT_COMPLETION
  -> FAULT_RECORD
  -> POISON_OR_CANCEL_OUTSTANDING
  -> DRAIN_ACK
  -> RESET_WAIT 或 IDLE
```

规则：

- descriptor validate 必须先于 NoC issue。
- address fault 不允许发出部分事务；若 fault 在 burst 中段由 memory controller 返回，必须 poison 后续 completion 并产生 fault record。
- completion event 只在所有 beats commit 后置位。
- async DMA 可被 tile/group reset cancel；cancel 必须进入 drain path，回收 NoC credit。

### 3.4 SRAM bank arbiter 状态机

```text
BANK_IDLE -> GRANT -> ACCESS -> RESPOND -> BANK_IDLE
                 \-> REPLAY_QUEUE_FULL -> BACKPRESSURE
ECC_ERROR -> POISON_RESPONSE -> FAULT_RECORD
```

仲裁优先级建议：event/status 小访问 > UCE descriptor fetch > BOA accumulator limited path > BOA operand burst > Tile DMA/MFE burst > EVU replay > USE state。实际权重由 `由 SRAM profile 冻结`。

## 4. 接口、descriptor、寄存器和协议

### 4.1 DMA descriptor v0

```c
typedef enum {
    ELENOR_DMA_1D        = 0,
    ELENOR_DMA_2D        = 1,
    ELENOR_DMA_STRIDED   = 2,
} elenor_dma_kind_t;

typedef enum {
    ELENOR_DMA_F_EVENT       = 1u << 0,
    ELENOR_DMA_F_POISON_ON_ERR = 1u << 1,
    ELENOR_DMA_F_L1_DST      = 1u << 2,
    ELENOR_DMA_F_L2_DST      = 1u << 3,
    ELENOR_DMA_F_WRITEBACK   = 1u << 4,
} elenor_dma_flags_t;

typedef struct {
    uint16_t abi_version;
    uint16_t kind;
    uint16_t context_id;
    uint16_t queue_id;
    uint32_t desc_id;
    uint64_t src_iova_or_l2;
    uint64_t dst_iova_or_l2;
    uint32_t bytes_per_row;
    uint32_t rows;
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t event_id;
    uint32_t flags;
    uint32_t timeout_cycles;
    uint32_t rsvd0;
} elenor_dma_desc_v0_t;
```

约束：

- `abi_version` 不匹配必须产生 invalid descriptor fault。
- `bytes_per_row * rows` 溢出、未对齐、越过 slot/L2 quota 时必须 fault。
- L1 目标地址必须通过 slot frame base/size/permission 检查。
- IOVA path 必须由 driver/firmware 保证 pinning 和 IOMMU 映射有效；硬件只消费翻译后或可翻译地址。

### 4.2 NoC packet header v0

```c
typedef enum {
    ELENOR_NOC_PKT_CMD      = 0,
    ELENOR_NOC_PKT_EVENT    = 1,
    ELENOR_NOC_PKT_DMA_RD   = 2,
    ELENOR_NOC_PKT_DMA_WR   = 3,
    ELENOR_NOC_PKT_STREAM   = 4,
    ELENOR_NOC_PKT_COLLECT  = 5,
    ELENOR_NOC_PKT_FAULT    = 6,
} elenor_noc_pkt_type_t;

typedef struct {
    uint8_t  vc;
    uint8_t  pkt_type;
    uint8_t  src_port;
    uint8_t  dst_port;
    uint16_t context_id;
    uint16_t route_id;
    uint32_t packet_id;
    uint32_t sequence_id;
    uint16_t byte_count;
    uint8_t  poison;
    uint8_t  last;
} elenor_noc_hdr_v0_t;
```

VC 映射：

| VC  | 流量                                  | 优先级 | 排序规则                             | starvation 规则                 |
| --- | ------------------------------------- | ------ | ------------------------------------ | ------------------------------- |
| VC0 | command、event、fault、barrier        | 最高   | per source-destination 保序          | 不得被 VC1/VC2/VC3 阻塞         |
| VC1 | DMA read response、memory completion  | 高     | 同一 DMA descriptor 内 sequence 保序 | bounded latency，由后续规格冻结 |
| VC2 | DMA write、MFE stream fill、bulk data | 中     | 同一 stream/descriptor 内保序        | 可被 throttle                   |
| VC3 | collective reduce/broadcast           | 中高   | collective epoch 内保序              | 不得反压 VC0                    |

### 4.3 寄存器草案

| 寄存器                  | 位宽 | 说明                                            |
| ----------------------- | ---: | ----------------------------------------------- |
| `NOC_CFG`               |   32 | topology id、VC enable、route mode              |
| `NOC_VC_CREDIT_INIT[4]` |   32 | 每个 VC 初始 credit，值由后续规格冻结           |
| `NOC_VC_OCC[4]`         |   32 | per-VC occupancy snapshot                       |
| `NOC_VC_STALL[4]`       |   64 | downstream credit 空导致的 stall cycles         |
| `NOC_POISON_CNT`        |   64 | poison packet 计数                              |
| `DMA_CTRL`              |   32 | enable、soft_reset、drain_request               |
| `DMA_STATUS`            |   32 | idle、busy、draining、faulted                   |
| `DMA_FAULT_PTR`         |   32 | fault record index                              |
| `SRAM_BANK_CONFLICT[n]` |   64 | 每 bank conflict cycles，n 由 SRAM profile 冻结 |

## 5. 数据流、控制流和时序路径

### 5.1 Cold launch 数据流

1. Host 上传 package、program、descriptor、weights 到 HBM。
2. Runtime ring doorbell；Device Runtime 校验 command / descriptor version。
3. Region Sequencer 通过 Global DMA 将 hot Region Program 加载到 Group SRAM。
4. Region 初始化 stream queue，发起 HBM 到 L2 预取。
5. Tile UCE 加载 Tile Program，绑定 slot frame。
6. Tile DMA / MFE 将 L2 数据写入 L1 slot。
7. BOA/EVU/USE 消费 L1 slot；结果经 Tile DMA、Group DMA 或 Stream Queue 推进。
8. event/fault 通过 VC0 返回，不得等待 bulk data drain 才可见。

### 5.2 ordering 和 coherency

- NoC 不提供全局 cache coherency；显式 event/fence 是唯一跨 engine 可见性边界。
- 同一 DMA descriptor 内，read response sequence 必须保序提交；不同 descriptor 可乱序完成，但 event id 必须区分。
- 同一 stream queue 的 token `sequence_id` 必须按协议递增；MFE reorder buffer 负责把乱序 memory response 恢复成 logical order。
- L1 descriptor cache 与 runtime warm patch 之间必须执行 invalidate/flush；running program text 不允许 patch。
- Tile UCE 在 launch engine 前必须确保 descriptor patch commit；patch 后 descriptor read 必须看到新值。
- SRAM write -> engine read 需要 local fence 或 event；engine write -> DMA storeback 需要 engine completion event。
- poison packet 保持原 packet order，不得绕过同一 flow 的早期 packet 造成状态不可解释。

### 5.3 关键时序路径

| 路径                                                 | 风险                    | 缓解                                             |
| ---------------------------------------------------- | ----------------------- | ------------------------------------------------ |
| NoC router switch allocation -> crossbar grant       | 高 fan-in 仲裁          | 分层仲裁、VC-local queue、one-hot grant register |
| SRAM bank conflict detect -> replay select           | EVU/MFE random access   | bank-local replay FIFO、下周期响应               |
| DMA completion -> event fabric -> VC0                | command/event latency   | VC0 高优先级、event small packet bypass data VC  |
| MFE stream fill -> L1 bank write -> BOA operand read | paged attention overlap | ping-pong stream buffer、bank pinning、PMU 归因  |
| reset/drain -> outstanding credit return             | 死锁风险                | drain watchdog、credit reconciliation counter    |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 容量和面积假设

| 配置           |        Tile L1 |         Group SRAM | 总片上 SRAM 粗算 | NoC 建议                                | 面积/封装假设                                    |
| -------------- | -------------: | -----------------: | ---------------: | --------------------------------------- | ------------------------------------------------ |
| Edge           | 1 MB x 8 到 16 | 2 到 8 MB x 1 到 2 |      12 到 32 MB | crossbar + small mesh                   | 常规 6T SRAM                                     |
| Balanced-small |      1 MB x 64 |           8 MB x 8 |           128 MB | 2D mesh                                 | 常规 SRAM + 严格 floorplan                       |
| Balanced-large |      2 MB x 64 |          16 MB x 8 |           256 MB | hierarchical mesh                       | SRAM/eDRAM/先进工艺，面积由 PPA exploration 冻结 |
| High End       |     4 MB x 128 |  64 到 128 MB x 16 |        1.5 GB 级 | hierarchical mesh + group/global router | chiplet / 3D SRAM / eDRAM / 近存缓存             |

### 6.2 带宽模型

```text
BW_sram_required =
  BW_boa_A_read
+ BW_boa_B_read
+ BW_boa_acc_read_write
+ BW_evu_lsu
+ BW_mfe_stream_write
+ BW_dma_load_store
+ BW_use_state
+ BW_program_desc_event

BW_sram_required <= BW_sram_peak * efficiency
BW_eff = BW_peak * (1 - conflict_rate)
```

NoC budget：

```text
BW_noc_required = BW_cmd_event_vc0
                + BW_dma_read_resp_vc1
                + BW_dma_write_stream_vc2
                + BW_collective_vc3
BW_noc_required <= BW_noc_peak * router_efficiency
```

所有峰值、频率、flit width、router radix、buffer depth 由后续规格冻结；文档层面冻结公式、counter 和验收方法。

### 6.3 PMU counter

必需 counter：

- `noc_vc_occupancy_cycles[4]`
- `noc_vc_credit_empty_cycles[4]`
- `noc_vc_credit_full_cycles[4]`
- `noc_route_block_cycles[4]`
- `noc_poison_packet_count`
- `dma_bytes_read` / `dma_bytes_written`
- `dma_desc_active_cycles` / `dma_wait_memory_cycles`
- `sram_bank_conflict_cycles[bank]`
- `sram_ecc_correctable` / `sram_ecc_uncorrectable`
- `l1_slot_violation_count`
- `event_latency_cycles_vc0`

PMU 唯一归因遵循：engine active > wait event > wait operand > stream credit > SRAM bank conflict > NoC backpressure > DMA wait memory > UCE program/descriptor stall > unknown。

## 7. RTL/软件实现建议

- Router 使用 per-input per-VC FIFO，禁止不同 VC 共享一个会造成 head-of-line blocking 的单 FIFO。
- VC0 command/event/fault 走独立 credit pool；VC0 不依赖 VC2 drain 才能前进。
- Bulk DMA packet 使用 burst，但 event/fault packet 使用 single-flit 或短 packet。
- SRAM bank arbiter 保留 consumer tag，用于 PMU attribution 和 SVA 检查。
- Tile DMA 的 L1 地址先转换为 slot-relative，再做 base/size/permission 检查。
- Firmware 在 reset 前读取 outstanding descriptor、packet、credit 和 event snapshot，写入 fault record。
- Driver 暴露 memory profile、NoC profile 和 SRAM profile 给 compiler/runtime；compiler 不应假设固定绝对地址。
- Warm launch 只 patch descriptor 和 context metadata；program text residency 由 Program Table 管理。

## 8. 验证、bring-up 和验收标准

### 8.1 SVA / formal checks

- VC0 progress：若 downstream credit 周期性返回，VC0 packet 必须 eventually leave ingress。
- Credit conservation：`credits_available + flits_in_flight + fifo_occupancy == credit_capacity`。
- No drop：非 reset/drain 状态下，valid packet 不得被静默丢弃。
- DMA event：descriptor 所有 beats commit 后且无 fault，completion event 必须置位一次。
- DMA fault exclusivity：同一 descriptor completion event 与 fault event 不得同时作为 success 上报。
- SRAM permission：write 到 read-only slot、DMA 覆盖 accumulator without permission 必须 fault。
- Poison propagation：poison input packet 必须产生 poison response 或 fault record。
- Reset drain：drain_done 之前 outstanding counter 必须归零或被标记为 canceled。

### 8.2 bring-up 顺序

1. 单 router loopback：VC credit、packet order、poison packet。
2. DMA 1D copy：HBM/L2/L1 三种 endpoint 的 completion event。
3. SRAM bank conflict random：PMU conflict 与 golden conflict model 对齐。
4. VC 干扰测试：VC2 bulk stream 满载时 VC0 event latency bounded。
5. Tile Slot Frame 绑定：非法权限、越界、未对齐 fault。
6. Paged attention trace：MFE stream fill 与 BOA QK overlap，PMU 可观察 BOA operand stall 降低。
7. reset/drain fault injection：DMA timeout、NoC poison、SRAM ECC、event deadlock。

验收标准：所有直接影响 command -> DMA -> engine -> event -> PMU 的路径必须有 RTL sim、SVA 或 formal 证据；性能声称必须能由 PMU counter 和 roofline model 解释。

### 8.3 跨模块 contract checklist

- Binary struct / protocol：DMA descriptor、NoC header、寄存器和 fault/poison packet 均有 v0 草案；所有未冻结位宽和容量由后续规格冻结。
- State machine：DMA、NoC router、SRAM bank arbiter、drain/reset 均必须有 RTL 状态枚举和 transition coverage。
- Capacity / bandwidth / area：L1、L2、NoC topology、VC buffer、SRAM macro 与 area profile 分离；性能公式固定，数值由 SRAM profile 冻结或由 PPA exploration 冻结。
- NoC VC behavior：VC0 command/event/fault 不被 VC1/VC2/VC3 阻塞；VC1 read response 保序；VC2 bulk stream 可 throttle；VC3 collective epoch 内保序。
- Credit / EOS / error / reset：Memory / NoC 不生成 EOS，但必须承载 Stream Queue EOS/error packet，reset/drain 必须回收 VC credit 和 DMA outstanding。
- Patch ownership：descriptor warm patch 由 Runtime/firmware 发起，Tile UCE 执行 slot-relative patch，MFE 拥有 page/segment 动态地址，Memory / NoC 只检查权限、地址和 coherency fence。
- Ordering / coherency：无隐式 cache coherency；descriptor invalidate、event/fence 和 poison ordering 是唯一跨模块可见性边界。
- SVA / formal：credit conservation、VC0 progress、DMA event exclusivity、poison propagation、reset outstanding 清零必须进入回归。
- PMU / error hooks：所有 NoC congestion、SRAM conflict、DMA timeout 和 poison 都必须有 primary owner 与 fault record。

## 9. 风险、取舍和后续细化方向

| 风险                          | 影响                          | 缓解                                                                 |
| ----------------------------- | ----------------------------- | -------------------------------------------------------------------- |
| NoC / SRAM contention         | BOA、EVU、MFE、USE 互相阻塞   | VC 分离、bank-aware layout、compiler memory planner、PMU attribution |
| VC deadlock                   | command/event/fault 不可见    | VC0 独立 credit、deterministic route、formal progress                |
| High End SRAM 面积不现实      | 设计无法落地                  | First Silicon 选择 Balanced-small 或明确 3D/chiplet memory 假设      |
| descriptor/coherency 边界模糊 | warm launch 使用旧 descriptor | descriptor cache invalidate、patch ownership、SVA 检查               |
| reset credit leak             | 下次 launch 假性满队列或死锁  | credit reconciliation、drain watchdog、fault snapshot                |
| PMU 双重计数                  | 性能归因错误                  | primary stall owner 规则、secondary tag 只用于 debug                 |

后续必须冻结：flit width、router radix、buffer depth、frequency、SRAM macro、ECC policy、NoC latency budget、VC starvation bound、per-profile bandwidth table，均由后续规格冻结。