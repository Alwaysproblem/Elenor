# ELENOR 硬件重设计考量清单

> **定位**：由 `design/` 全部模块规格、`design/proposal/`、`review/`、`pipeline_validator/`、`hardware/boa/` 归纳的**派生清单**，不是规格；与规格或当前代码冲突时以后者为准。前提：软件架构与调度不变，硬件微架构可重选。
>
> **阅读路径**：快速浏览 §0 → §5 → §8；查合同看 §1；改某模块看 §3 对应小节。
>
> **标签**：〔模型〕仅 validator 建模值；〔未冻结〕规格明确待冻结，不得补数字；〔冲突〕文档互相矛盾；〔待核〕与现状可能不符、未复核；〔推断〕文档未写、工程判断。§1 整节即“合同”，不逐条打标。出处缩写见附录。

```text
软件/调度（不变）  Graph → Context → Grid → Task → Tile Program → Engine；封印 artifact；只读 Loader
        ↓ 合同面（§1）
硬件微架构（可变）  SRAM/bank · NoC/HBM · UCE · 阵列/AM · EVU · MFE · USE · 物理实现（§3）
```

## 0. 先看这十条

1. **接口面不变**：硬件只执行 command / descriptor / TileGroupTask / Tile Program；准入原子性、退休顺序、事件 `id + sequence`、Arena 所有权、12 步 Profile 事务是软件假设（§1）。
2. **指纹字段即重编译**：拓扑、容量、bank、队列深度、context 数等进 target 指纹，旧 artifact 作废；纯时序旋钮只改周期基线（§2）。
3. **SRAM 几何是编译合同**：bank / stripe / reserved / Profile 表决定逐 bank 容量证明与 L2 no-rebind 高水位（§3.1）。
4. **“完成”六个边界不可合并**：`compute_done ≤ step_committed ≤ output_visible ≤ slot_free ≤ chain_done ≤ UCE event`（§3.5）。
5. **不可暂停的输出须先预留**：阵列输出 valid 无 ready，发射前按 O+R ≤ Q 预留；mesh 最后输出 ≠ AM commit（§3.5）。
6. **每个 ID/counter 都要有回绕规则**：靠 quiescence/背压，不靠位宽（§1.3、§4）。
7. **fault/reset 独立于数据通路**：VC0/sideband、first-fault latch、fault record 先于 event、PMU snapshot 先于 clear（§3.11）。
8. **未冻结项多、文档间有冲突**（§6）。
9. **模型周期 ≠ 器件性能**：1 Group × 4 Tile、单 router、HBM 固定延迟；不外推（§6.4）。
10. **容量先过物理现实**：High End 1.5 GB 级不能当普通片上 SRAM；VCU118 上 4 Tile × 4 阵列的 DSP 需求（推导，未综合）超器件上限（§3.1、§3.5）。

## 1. 合同：新硬件必须仍提供的能力

### 1.1 控制与调度

- **调度链**：CPU `DevicePort`（有限 outstanding）→ `GroupPort`（有界 pending 元数据，不占 L2/slot）→ 每 root `TileGroupSequencer`（registration cursor + fence）→ `GroupScheduler`（每周期 ≤1 ISSUE + ≤1 REGISTER，先处理 completion）→ 有界 Grid Route → 每 Tile 独立 Task commit → Tile UCE（eligible-head RR，单指令/周期）→ 有界 engine 队列。〔主 §7、§16；TG §3.2〕
- **三本容量账不可合并**：CPU outstanding、pending 元数据、Group execution slot + L2 Arena；message acceptance ≠ root 准入。〔GS §4.3〕
- **completion 不被表满阻塞**：credit/事件回收独立于 action 表。
- **准入**：root 与 Task 均 SAME 优先、COMPATIBLE 补位、同类不越队首。Task commit 原子取得 L1 Arena + Frame + UCE pin + 父 L2 pin + R lease；准备失败可回滚，已接受工作只能 drain/cancel-confirm；某 Tile 阻塞不回滚其他 Tile。
- **退休顺序**：在途 transfer/view 关闭 → L1 Arena 退休 → L2 pin 释放 → Frame 释放 → R lease 归还 → Grid done。`tile.free`、`input_released`、`output_ready` 都不归还资源。R lease 键 `(parent binding, launch generation, tile)`，跨该父的所有 Grid，pending 不占。
- **等待语义**：`nest.await` → WAIT_EVENT（只 fence 本 root 后续注册）；`nest.barrier` → BARRIER_GROUP（只等同 Context 更早 action）；硬件不得升级成全局 barrier。
- **等待可归因**：WAIT_CAPACITY / FRAGMENTATION / SLOT / CONTROL_RESOURCE / CONTEXT_LIMIT 各自独立；空池也放不下 = PERMANENT，直接 fault；失败的 plan 零副作用。〔04 §3；LIM〕

### 1.2 内存、Arena 与 Profile

- **所有权**：root 拥有 L2 Arena，Task 拥有 L1 Arena；`nest.release` / `tile.free` 只使 view 失效。〔主 §12.1；IR_SPEC §3.8〕
- **L1** 仅在编译器证明 lifetime 不重叠时复用 slot/offset，运行期只按既定布局重新绑定。**L2 永久 no-rebind**：每个 buffer 独占整 stripe-round padded span；`nest.release` 是永久 forfeiture；物理 final-free 才归还整 padded span，并只在此刻推进 `pool_version`。
- **容量证明逐 bank、含碎片**；总空闲字节不能掩盖 stripe 碎片。
- **L2 交接三种**：`private`（默认）；`context-local`（无锁无原子，不跨 context）；`readonly`（`nest.publish` → claim `DECLARED → BOUND → RELEASED`，限同 Group/L2 Profile epoch；producer 退休后仍保留被 claim/pin/inflight 的 backing，容量按 backing 去重）。〔L2P〕
- **Profile**：同层全员同一 mode，不支持 per-bank/per-Context 私有 Cache 分区；Cache `target_bytes` 只是 hint，不是配额。
- **Profile 切换** = 编译器生成的 await + `ProfileReconfigDesc`：12 步（ACQUIRE → CHECK_FRONTIER → CLOSE_ISSUE → DRAIN_REFERENCES → CLEAN_INVALIDATE → DRAIN_DOWNSTREAM → PREPARE → WAIT_READY_ACK → COMMIT → WAIT_COMMIT_ACK → OPEN_ISSUE → RELEASE）、唯一 writer、全成员双 ACK、generation 单调；缺任一步 Loader 拒绝。L1-only 切换保留父 L2 Arena；L2 切换先等 root 完成 frontier。〔主 §12.1；04 §4〕
- **搬运**只支持连续 row-major、源目的等字节；2D/strided/multicast 是未建模的硬件扩展。

### 1.3 事件、ID、完成与故障

- **事件匹配**一律 `event_id + expected_sequence`；裸 id/status 在 reset/reuse/drain 后会误判 stale。identity 带 owner/generation，旧 generation 的完成不得唤醒新 owner。〔ABI §3.2；GS §3.3〕
- **Event Table** 按编译器证明的 live frontier 计账（非全历史），slot 复用靠 generation。有限 ID 回绕靠 quiescence/背压，不靠“位宽够大”（protocol `wraparound`）。
- **终态**：PENDING / DONE / ERROR / TIMEOUT / RESET，terminal 不回 PENDING；fault 链不得产生成功 completion；first-fault-wins；event 的 status 与 sequence 原子写入，防 host 读到混合状态。
- **三条 host 可见序**：descriptor 先于 doorbell；event 先于 interrupt；fault record 先于 ERROR event。〔HI §4.5；FR §8〕
- **取消/故障**：已接受计算只靠 drain/cancel-confirm 收敛；超时本身不是隔离确认；旧响应不得污染新 generation。

### 1.4 Program、descriptor 与地址

- **residency**：编译器不发 `program.load/prefetch/install`；runtime 只登记 metadata；Group 保证 dispatch 前 ready；Tile UCE 只消费 resident handle（`program_local_slot/offset/bytes/program_epoch`）；身份 `program_id + version + hash + epoch`，reset 使其失效。〔主 §15；PKG §3〕
- **warm launch** 只 patch descriptor/context/shape；运行中不得 patch program text；descriptor cache 须 invalidate 或带 sequence tag。
- **patch_mask** 声明可 patch 字段，分层 owner：Runtime（context/IOVA）→ UCE（tile/group/slot）→ MFE（page/segment）→ USE（state）；未声明字段不得改；越界 = invalid descriptor fault。〔CMP §4.6；SF §3.3〕
- 硬件只见 `base/extent/stride/slot/event/stream token/descriptor`；descriptor 引用 slot 而非裸物理地址；IOVA 由 driver pin + IOMMU 提供。
- Package 版本四分（package / command / descriptor / slot_frame，另有 stream_queue_abi）：minor 只追加，major 不兼容即拒绝并写 fault record。〔PKG §4.2〕

### 1.5 数值（BOA，`bf16_fp32_mac_rne_v1`）

- BF16 输入；每个 `tile_k`（默认 8）块从 +0 起按 k 递增逐 MAC `fma_f32_rne`（单次舍入）；INIT 首个非空块覆盖 AM，其后每块 `add_f32_rne`；不得整 step 汇总再加一次。
- 保留 subnormal，禁 FTZ/DAZ；NaN 输出规范化为 `0x7fc00000` / `0x7fc0`；tininess-after（用无界 exponent 的目标精度判 tiny，不看最终编码）。
- mask 无效 lane **旁路**不做 MAC（`fma(0,0,p)` 会改 signed zero，`0×Inf` 得 NaN）；整块无效不做 AM `+0`；K=0：INIT 写 +0，ADD 逐 bit 保留旧值。
- epilogue 只在 FINAL 做一次：bias(add) → scale(mul) → ReLU → clip → 一次转换。INT8：`mul(y, inv_scale)` → RNE 取整 → 加 zero_point → 饱和 `[-128,127]`；NaN → zero_point，+Inf → 127，−Inf → −128。
- Gemmini 路线：Matmul 输入只有 BF16，INT8 仅作输出类型；Conv 恒为软件 im2col + Matmul，不做硬件 LoopConv（与主规格冲突，见 §6.2）。
- 改 `tile_k`、融合方式或特殊值策略 = 改 bits：进 target 指纹，重跑全部对拍。流水化 MAC 会改 skew/排空时序，须重跑数值与联仿。〔NUM；Gem §3.2、§7.4.7〕

### 1.6 观测

- 每个 stall cycle 只有一个 primary owner，9 级：engine_active > wait_event > wait_operand > stream_credit > sram_bank_conflict > noc_backpressure > dma_wait_memory > uce_program_or_descriptor > unknown；secondary tag 只进 debug；分域 stall 不可相加成全局周期。〔PMU §3.3；REV §4.6〕
- engine 局部细计数可保留，但须同时输出映射到这 9 级的 primary；fault/reset 期间周期归 reset/drain owner；`unknown` 非零 = 验证覆盖缺口。
- PMU 来自真实组件快照，不从 trace 事件名反推；counter 饱和加法 + overflow sticky；snapshot 用 freeze-then-read 或 double-bank。

## 2. 哪些可以改，改了付什么代价

- **可自由重选**（软件合同不规定）：SRAM macro/bank 数/端口/位宽/ECC、cache 组织（set/way/tag/MSHR）、NoC 拓扑/flit/VC 深度、HBM 控制器与 channel 映射、阵列实现与 dataflow、UCE 实现形态（RISC-V / sequencer / ROM）、时钟域划分、引擎内部流水与队列深度、容量档位。
- **进 target 指纹（`target_fingerprint`）→ 旧 artifact 作废，须重编译并升版**：`num_tiles`；Group/L1 容量与 bank；BOA 阵列几何、AM 槽数/容量、数值 profile 与 `tile_k`；`evu_lanes`；`use_state_cache_bytes`；`uce_dispatch_per_cycle`；MFE channel/队列深度/pipeline/stream buffer；HBM 容量/outstanding/channel/burst；cache line；L1/L2 MSHR；program SRAM；`noc_vc_depth`；DMA 通道数；`frame_slot_capacity`；`context_count`；Device 与 Group 调度容量；Profile Registry。
- **不进指纹 → 只改周期基线**：各类 latency、带宽、launch cycles、时钟、trace、fidelity、seed、调度策略（S0/S1）。

## 3. 模块级清单

只列模块特有的陷阱、未冻结项与〔推断〕；通用合同见 §1。

### 3.1 SRAM、bank 与 Profile

- **几何是编译合同**（§1.2）：stripe-round（模型默认 16 bank × 256 B = 4 KiB）来自 Registry；改 bank 数/宽度 → padding → L2 span → no-rebind 高水位 → `prepare_resources` 重校验。
- `system_reserved_spm` 来自不可变目标描述，缺失不得默认 0，进 Registry hash；SPM + Cache 总量守恒、分区粒度 64 B，不守恒/不取整即拒。
- **面积账**：Cache 容量与 L1/L2 数据容量同属一个总量，不可相加；Tag/MSHR/ECC/有效位/控制 RAM 另计。Cache 现为全局 LRU（非 set/way），命中率不是物理预测；写策略 read_only / write_back；维护能力 `invalidate_range` / `clean_invalidate_all` / `bypass`，无 range 能力只能全域 + 静止证明。
- **bank 分区**：Program/Descriptor/Event 区不与 BOA operand 热路径同 bank 组；USE state cache 设 protected region；MFE stream 用 ping-pong；accumulator slot 单独标记防 DMA/EVU 覆写。
- **仲裁优先级建议**：event/status > UCE desc fetch > BOA acc 限流路径 > BOA operand burst > Tile DMA/MFE burst > EVU replay > USE state，权重〔未冻结〕；arbiter 须保留 consumer tag 供 PMU/SVA。**带宽账**：7 类 consumer（BOA operand/acc、EVU、MFE stream、USE、DMA、UCE program/event）ΣBW ≤ peak × (1 − conflict_rate)，各项〔未冻结〕。
- **容量档位**：Balanced-small（1 MB × 64 + 8 MB × 8 = 128 MB）是 first-silicon 现实档；256 MB 档需 eDRAM/先进工艺论证；High End 1.5 GB 级须 chiplet/3D SRAM/eDRAM，不能默认普通 6T SRAM。〔主 §12.3；PTP〕
- **ECC/parity**：规格仅 optional（uncorrectable → poison → fault），scope 与策略〔未冻结〕；多 MB SRAM 与长驻 KV 必须定。
- **Program SRAM** 占用 = `4 B/指令 + 16 B/descriptor + 16 B/AM buffer + 4 B × 内联常量字`（bias N 字、scale 1、clip 2、inv_scale + zero_point 2），下限 1 KiB；这是软件模型 footprint，非硬件编码。改指令/descriptor 编码须同步 `required_text_bytes`，否则 Loader 拒绝。
- 主 §12.2 的 L1 分区示例（2 MB，含独立 accumulator 384K）与现行“AM 独立于 L1 Arena/Frame”的设计可能不一致〔待核〕。

### 3.2 Group 层（L2、Sequencer、Scheduler、Group DMA）

- `GroupScheduler` 的单 ISSUE + 单 REGISTER 是共享瓶颈：改宽度会改变所有 root 的 DISPATCH_ROLE/DMA/publish 时序与 R lease 归还时刻；S2 当前等同 S1。〔SEQ §6.2〕
- Sequencer 不得泛化成小 CPU（无 Group 级 PC，action 表不自修改）。扫描 probe 须无副作用，硬件需旁路探测端口，宽 scan 需分 bank 候选表〔推断〕。
- L2 准入 strict FIFO，唯一唤醒源是 final-free（无忙轮询）；旧评审（proposal/02 §9.3）的 n04-fifo-hol 场景观测到 FIFO 队头阻塞的显著代价（旧基线，不引用数字）。WAIT_CAPACITY 等待者零持有 UCE context/L1 frame/StreamQueue/DMA/L2 handle。
- 软件期望“B 准入周期 == release 周期”，硬件须让 free-map 写入与准入检查同拍可见（前递）〔推断〕。readonly 的 backing/claim 表深度 ≈ contexts × slots，`claim_id`/generation/`pool_version` 位宽须冻结并防回绕〔推断〕。
- Group DMA 只做 descriptor 已定址的 1D/2D/strided copy；cutline `copy > async event > multicast > gather list`；page/segment walk 属 MFE，L2 → L1 属 Tile DMA/UCE。

### 3.3 NoC、DMA 与 HBM

- **VC 映射**：VC0 command/event/fault/barrier（最高优先、独立 credit 池、单 flit 短包、不得被 VC1–3 阻塞）、VC1 DMA read response、VC2 write/stream、VC3 collective。这是可靠性要求而非性能选项；合并 VC 须重做死锁分析。〔NOC §4.2；FR §4.3〕
- 禁止跨 VC 共享 per-input 单 FIFO（HOL）。credit 守恒 `credits_available + flits_in_flight + fifo_occupancy == capacity`；reset 后 credit 泄漏 → 假性满队列 → 死锁，需 reconcile counter（运行期也常设 watchdog〔推断〕）。router 8 级流水须 credit 遍历寄存化、crossbar grant 一级化，switch alloc 用 age + VC 优先级 + starvation 计数。
- 拓扑：Edge crossbar/small mesh；Balanced 2D 或 hierarchical mesh。模型只有单 router、无 mesh，跨 router 延迟/吞吐无模型支撑。
- **无全局 cache coherence**：SRAM write → engine read 需 local fence；engine write → storeback 需 completion event；引入缓存须重建这些 fence 假设。
- **DMA 序**：completion event 在最后一个 write response 之后；同 descriptor 内 read response 按序提交；zero-length、4 KiB/IOMMU 页切分、max_burst〔未冻结〕；32 位 bytes/rows/stride 须防溢出，大 copy 由 runtime 拆分（规则〔未冻结〕）；outstanding tag 用 CAM 成时序路径，可限窗口或用 age-based circular tag FIFO。
- **HBM channel 映射**：模型以整腿起始地址 `(addr // burst) % channels` 选单 channel，对齐的 2 的幂块步距会全落 ch0；控制器冻结时用 xor/hash 或 strand 交织并复测，burst splitter/页边界与 channel 选择用同一对齐基数〔推断〕。HBM 模型为固定延迟 + 带宽，不含行/刷新/turnaround。

### 3.4 Compute Tile、Tile UCE 与 Slot Frame

- RR 只是 issue 仲裁，正确性不依赖它（§1.1、§1.3）；全阻塞按 WAIT_EVENT > WAIT_ENGINE_QUEUE > WAIT_STREAM 归因。UCE 接受 ≠ engine 接受 ≠ 完成，三者分开统计。
- **context**：模型 1–8（`SimConfig` 默认 1，AM 验收默认 4）。一个 hart ≠ 一个 context；context 数 ≠ CPU outstanding ≠ Group slot ≠ R lease ≠ `context_id` 隔离域。改 context 数牵动 R lease、L1 容量、event/scoreboard 预算、frame 生成；硅片数〔未冻结〕。
- **engine ingress 队列**：模型 BOA/EVU/USE 深 1，MFE load/store 深 4；深度决定 blocked-head 跳过与 `engine_queue_full` 归因，硬件深度〔未冻结〕。每周期扫全部 head 在 context 多时是时序路径，应做并行 eligible 向量〔推断〕。
- **Slot Frame**（16 slot，v0 草案；bind/patch FSM 见 SF §3.2/§3.3）：shadow 双 buffer（active/next），bind commit 原子切 generation；校验为组合逻辑 + 拍寄存，可多周期 bind。patch 单元只允许小 ALU（add/shift/stride 乘），复杂公式由编译器展开；engine wrapper 只收 slot-resolved 地址；patch 失败禁 launch。
- **故障边界**：admission 后的 validation fault 走 trap → fault record → Task terminal = ERROR；launch accepted 后的 fault/timeout 升级为 tile-local drain，止于本 Tile。
- UCE/USE 可共核，但 ownership/CSR/PMU/fault owner 须分离（`uce_*` / `use_*`）；per-context timeout 应收敛到该 Task terminal，与 Group 侧 watchdog 二选一，避免双 watchdog 竞态〔推断〕。指令编码〔未冻结〕；RISC-V / sequencer / ROM 三种实现对外语义（launch 序列、event 序列、fault 行为）须一致。

### 3.5 BOA 与 AM（Gemmini 路线）

> 现行路线是 Gemmini 脉动阵列组；旧 OPA/outer-product、reduce tree、dtype 描述不覆盖该路线（见 §6.2）。〔Gem §1〕

**必须保持**

- **AM 拓扑**：每 Compute Tile 独立，4 group × 2 slot = 8 槽，跨 Tile 不共享；group 只是物理 slot 寻址，≠ engine ≠ L2 Tile Group；每 context ≤2 个 live lease（跨本 Tile 所有 engine，V1 禁调大）。〔protocol `capacities`〕
- **授予**：`AM_REQ` 一次发出（GRANTED / PARKED / ERROR，无 BUSY/重试）；PARKED 不占 slot/quota/generation；`slot_free` 恰好唤醒一次并交付已授予 lease；每 owner ≤1 挂起。fresh 须整 bundle 放入未预留空槽、FIFO 不插队；continuation 只用自身预留。授予原子取得 slot/quota/ChainTable/mailbox/L1 目的地；slot 取首个空闲 `(group, index)`，engine 取 live-handle 最少、平局最小 id，之后 affinity 固定到 `slot_free`。quota = 2 本身不证无死锁；同 `logical_am` 有未受理 FINAL 时的第三条链是合同错误。allocator 是 Tile 内唯一串行化点，授予判断/入队/`slot_free` 重评估须同拍序，否则丢唤醒。
- **完成序**：`compute_done ≤ step_committed ≤ output_visible ≤ slot_free ≤ chain_done ≤ UCE event`；FINAL 自动 drain 到已准入 L1，不等 UCE free/poll/ACK。Q09：`FINAL(slot0) → ALLOC(slot0) → await(旧事件)` 不得挂死。
- **输入生命周期**：descriptor accepted ≠ A/B 已读；最后实际读取且全部读者结束才可复用；不设软件可见 RF/load 事件（已取消，勿恢复）。
- **不可暂停输出**：阵列输出 valid 无 ready，发射前须为最小不可暂停微块的全部输出预留接收保障；每个真实 FIFO 独立 O+R ≤ Q，arrival 只是 R→O，下游有保障后出队才释放该层容量；不同层级容量不可相加。
- **RMW**：同一物理行从读旧值到写回须地址互锁或 forwarding（含 URAM 读延迟与输出寄存级）；INIT 跳过旧值读；加 bank ≠ 加 adder 吞吐；drain 不得被持续 ADD 饿死（按年龄/轮转仲裁读端口）。
- **每链** current + next 两个 mailbox（深 1），同 handle 下一步在前步 `step_committed` 后发射；mailbox 满 = FULL 无副作用 → WAIT_ENGINE_QUEUE；每 engine 同时一个 executing step。“禁 double issue”指不依赖阵列内部双发射，保留 PE `c1/c2/propagate/tag/last/drain`。
- **故障/复位**：先 quiesce 在途 AM 访问 → free → 发布失败终态；reset 中止 lease、清挂起、per-slot generation + 1，不要求 URAM 物理清零；terminal 环（256 项）仅诊断，可覆盖并计溢出。
- **身份**：`(ctx_id, ctx_epoch)`；lease = `(handle, slot_generation)`；event token = `(ctx_id, ctx_epoch, event_id)`；`(owner, request_seq)` 防互连重放重复授予。

**可改 / 待冻结**

- 阵列几何默认 16×16、逻辑块 16×16×8；矩形（8×16 / 16×8）与四种 transpose 必达，须独立推导 skew/transpose/last/tag/drain（方形 assert 不能只删）。dataflow WS/OS〔未冻结〕：`tile_k` = 8 时利用率上限约 50%（推导，未综合），`tile_k` = 16 近 100% 但改数值 profile。
- `arrays_per_tile` V1 目标 4（软件上限 8，非硬件上限）；`slot_capacity` 候选 256 KiB（物理由 SRAM profile 冻结）；result FIFO / AM credit / L1 output credit 的硬件微块深度〔未冻结〕（软件 step 粒度 4/8/1 不得强行等同）；RoCC opcode/funct/位宽、PE 流水、Vivado 版本〔未冻结〕。
- operand buffer 宜带 generation 标记，防 prefetch 翻转与 fault drain 交错时误消费 stale buffer〔推断〕。

**FPGA 原型约束**（VCU118 / XCVU9P，首版 100 MHz 单时钟）

- AM slot = 8 块 URAM（4096 行 × 16 FP32 lane）；group = 2 slot 级联 8192 行 + 16-lane FP32 RMW；每 Tile 64 块 URAM（6.7%）；一个 Tile 的阵列 + AmRouter + AM group 不跨 SLR。
- DSP〔推导，待综合〕：HardFloat fused 约 2 DSP/MAC；1 Tile × 4 阵列 = 2,048（29.9%），4 Tile × 4 阵列 = 8,192，超 6,840。PE 浮点 MAC 是组合路径，限制 Fmax；只接受 WNS ≥ 0。早期做一次 WNS 探测可提前暴露频率天花板；AmRouter E×4 交叉的冲突窗口宜用冲突轨迹仿真定仲裁粒度〔推断〕。
- Rocket 不自动满足“微型 UCE”面积目标，需综合证据；descriptor/记录放受保护 mailbox/TCM（release/acquire），不假定 Rocket D-cache 与 L1 SPM 自动一致；RoCC resp 须有限周期返回。默认 HardFloat；为 PPA 换 MAC 实现须重做 P03/P11/P12/P20 对拍。

**牵动**：改槽数/容量/选择规则/`tile_k`/numeric profile/affinity → 进指纹，重算全部 typed fixture 并升 artifact 版本；改 AM 状态机 → 事件合同与 Tile 退休（FINISHING）；改 MAC 流水 → skew/排空、数值向量、联仿。

### 3.6 EVU（EVU-MT）

- 只访问 local slot memory（`slot_id + byte_offset`）：无 HBM/虚存/page walk/全局指针；禁止滑向 GPU（无 warp scheduler、per-thread PC、reconvergence stack、多 context）。
- `effective_active = lane_valid & active_mask & predicate`；inactive lane 无任何副作用（不写 RF、不发访存、不参与 reduce、不触发地址 fault）；tail 由硬件生成 `lane_valid`；active lane 越界必须 fault。shared-PC、单 active kernel；分支只允许 uniform branch 与结构化 `if_mask/else_mask/endif`（mask stack 深 4，溢出/下溢 fault）。
- launch/commit：`launch_ready` 仅在 IDLE 且无 pending commit、无 fault drain；整份 descriptor latch；单次 terminal commit，identity = `event_id + event_sequence`；first-fault-wins，fault record 先于 fault event；slot binding table 运行中只读。
- 基线（建议，非冻结，EVU §14.1）：32 lane、mask stack 4、64 位指令、32 位数据通路、8 级流水。时序点：addr gen 与 bank check 分拍；predicate → byte-enable 扇出大，mask 分段寄存；gather replay 只重放冲突 lane、不改逻辑 lane 序，响应可乱序但写回恢复 lane mapping，replay queue 满背压、超时 fault。
- scatter 三选一放行（编译器证明无冲突 / ordered / 重复即 fault），V1 默认关，禁 silent last-writer-wins，atomic 不进 V1；BF16/FP16 reduce 用 FP32 累加；all-inactive reduction 返回 identity 还是 fault〔未冻结〕。
- 〔未冻结〕opcode、dtype/rounding、SFU 近似包络（决定 golden 能否对齐）、slot layout/bank hint、descriptor 二进制、fault_policy、scatter duplicate policy。
- 〔冲突〕主 §9.5 是 RVV 风格 ISA + 绝对地址 `elenor_vector_kernel_desc_t`，EVU 规格是 64 位自定义 ISA + 纯 slot 描述符且禁无序 scatter；cutline 把 gather/SFU 标 P1、结构化 mask 标 P2，却被 LSU 表与验收要求（REV P0-7）。须二选一。
- 牵动：L1 bank 重排 → replay 率；bank 端口数须 ≥ 访问 lane 数的配比，文档未联动分析〔推断〕。模型仅 Roofline，不执行张量算术。

### 3.7 MFE 与 Stream Queue

**MFE**

- V1 只做 Page + Segment；Sparse/Persistent 仅预留 flags，未实现 mode 的 descriptor 必须 fault，不得 silent no-op；descriptor validator 在任何 stream write 前完成。
- `launch.mfe` ISA 不带 channel 字段：前端按 descriptor 方向分类（store → store channel，其余 → load channel）+ 同类 first-free；跨 channel 无隐式完成/可见序，同步只靠 event/stream token/barrier；hazard table 仍是唯一准入。
- 控制面（fetch/validate、fault 聚合、event/commit、PMU）每 MFE 一份，数据面 channel 复制；禁止 per-feature 命令队列与 full-tile 写数据 FIFO。乱序页返回由 reorder buffer 恢复 logical order（V1 只做相邻 burst 合并）；stream buffer 满 = fault，不覆盖未消费数据；EOS/error 各一次且在最后 data token 之后；fault/cancel 用 epoch + drain，旧 response 不写新 command。
- **store 可见性三级**：accepted → L2-visible → global-visible，单调；store completion event = 可见性。matmul → gather 的同步点 = L2 visible + 显式 L2 barrier complete，L2 侧须保留 arrival 观测点。
- credit/backpressure 必须寄存，可加小 skid；ROB tag 比较随 outstanding 增大需分 bank。〔模型〕channel 1/1、队列 4/4；Balanced+ 建议 2 load（K/V 双流）；数量由 SRAM/NoC profile + PPA 冻结，规格明确禁止伪造队列深度。page walk 缺 TLB/页表项 cache 讨论〔推断〕。性能指纹 `T_prefetch ≤ T_qk`：不满足时 BOA operand_stall 与 MFE prefetch miss 应同时上升。

**Stream Queue**

- 可选 overlay，非必经通路；token 是非拥有引用（addr/bytes/sequence），payload 由 L2/L1 Arena 管，queue 不分配/pin/释放 payload。
- credit 不变量 `available + leased + valid_in_fifo + popped_not_released == depth`；EOS/ERROR 不占 credit（〔冲突〕REV §4.10 曾建议 EOS 消耗 credit，现行 SQ 规格相反，以 SQ 为准）。
- error：latch first fault index 并标记 queue faulted，≠ 下游包已送达；error 走 VC0/fault fabric；reset 只在外层 drain/cancel-confirm + owner isolation 之后，generation + 1 使旧 token 失效。MFE buffer 满 = fault 而 SQ 满 = producer stall，归因须区分；`BROADCAST` enum ≠ 已实现多消费者。
- 〔未冻结〕queue 数/深度/token SRAM/端口、物理 FIFO、CDC、descriptor 二进制；token 物理存放位置应与 MFE stream fill 分 bank〔推断〕。

### 3.8 USE

- 功能件：State RF、State Cache（大态如 SSM hidden）、Scan/Recurrence、Event Assist（仅 local 快路径，不取代 UCE 主控制流）、Token Router Assist、Checkpoint Buffer。
- 边界：USE 不推进 UCE PC，只经 event/condition/fault 影响分支；page walk/段地址/prefetch/reorder 归 MFE；stream credit 归 UCE/SQ；state slot 标 `STATE` role + 写权限 + owner 匹配；DMA 仅经 checkpoint/restore descriptor 访问 state backing；checkpoint 必须由 descriptor/显式命令发起。
- **reset 四档**：hard（清 valid/dirty）、tile soft（当前 task rollback 或标 invalid + reset event）、state preserve（保留选定 line + 版本，之后须 restore/validate）、USE local（不动 UCE PC，返回确定性 error/reset event）；fault/reset 后 dirty state 必为 committed / rolled back / marked invalid 三者之一。
- 实现路线：仿真 v0 = ROM bytecode 解释器；RTL v1 = 小 RISC-V 或极简 sequencer + USE 状态功能单元；v2 = RISC-V + custom 指令。与 UCE 共核时 exception cause 须区分 UCE control fault 与 USE state fault，debug halt 停在 task 边界且不破坏 dirty/valid。时序条件 `T_use_state_update ≤ T_neighbor_compute_or_stream_overlap`，用 `use_active/state_miss/state_stall` 证明，不误归 UCE wait。
- 〔未冻结〕checkpoint policy 与 rollback 语义、event assist 语义（REV P0-8：V1 依赖却被推迟）、state RF/cache 容量/关联度/line、scan lane 宽、recurrence 并行度、dtype。state cache 的 checkpoint 带宽与 DMA storeback 争 L2 带宽，无预算模型〔推断〕。

### 3.9 Collective

- V1：group 内 reduce add/max/min、broadcast/multicast、可选固定 mask all-reduce；无 cross-group、无 hierarchical reduce-scatter、无 engine-to-engine bypass。`input/output_l2_base` 是 L2 offset 非 HBM 地址，storeback 归 Group DMA。
- 每个 input partial slot 的 producer 唯一（BOA/EVU/Tile DMA/MFE 四选一）；output writer 是 Collective；readiness 每 descriptor 只选一种（wait_event / stream token / participant scoreboard）。同 `collective_id + block_id + sequence_id` 的全部 required input 到齐才 reduce；缺员须显式 mask + neutral element；重复 participant 必须 fault；任一 required participant ERROR → 输出 ERROR 透传；每个 accepted command 恰好一次 DONE/ERROR/TIMEOUT/RESET，reset kill 清 scoreboard/FIFO 不留 pending done。
- **数值确定性** = 固定 op/dtype + 固定 input 顺序，换 topology/lane 数须保持（如按 tile_id 左折叠）〔推断〕；numeric policy（dtype/rounding/saturation/NaN/Inf）〔未冻结〕，软件 golden 与 RTL 可能不一致。
- 数据通路与 event/control 分 VC；L2 bank 规则：input partial 按 tile_id 分散，output 避开 Group DMA storeback，event/status 区不与高带宽 payload 共 bank，scratch 不与 input/output alias。read FIFO / compute / write FIFO 分离，scoreboard 与 L2 addr gen 寄存，fan-in mask 归一与算术树不同拍。
- 〔未冻结〕topology、lane 数、每拍元素、流水级、SRAM 端口/ECC、all-reduce/reduce-scatter 是否入 V1、大块广播最终路径。模型只有 1-cycle command/window，**无任何吞吐结论**。

### 3.10 系统控制面（Chip Top、Host Interface、Global Scheduler）

- **生命周期**：POR → RESET_ASSERTED → RESET_RELEASE_SYNC → CSR_ENUM → FW_BOOT → IDLE ↔ RUNNING/DRAINING → FAULTED（→ RESET_ASSERTED 或 DRAINING）。reset assert 可异步，deassert 须域内同步；release 顺序 protocol frontend → CSR aperture → queue manager → interrupt controller。
- **校验分工**：Runtime Processor 先校验 ABI/`cmd_size`/context/domain/descriptor 边界再交 Scheduler；Chip Top 透传 command v0 全字段，`desc_iova` 属 context/IOMMU domain，wait/signal 落 context event namespace；Host IF 不重复解析 command body。
- **Scheduler**：只消费已准备好的 command，不 lower graph、不替代 root 准入；retire 必写 `signal_event + sequence` 或 fault event，queue head 不得越过未完成 command；event table 只由 Event Fabric 集中写；timeout wheel entry 带 generation/tag，防 event_id 复用后旧 timeout 命中新 command；timeout 与 completion 同周期的优先级〔未冻结〕且须 SVA。
- **Host**：doorbell FIFO 条目含 queue_id/context_id/tail_snapshot/sequence/timestamp/flags，`doorbell_seq` + sticky overflow 检测重复/丢失，满时内部计 `doorbell_fifo_full`；posted write 顺序不足由 driver read flush 或 `ordering_adapter`（选择〔未冻结〕）。fault 中断优先于 completion，coalescing 不得掩盖 fault 且须支持 polling-only；W1C 的 mask 只屏蔽 assert、不屏蔽状态更新，同周期新 fault 不被清除（优先级〔未冻结〕）；PMU 64 位读用 snapshot 防撕裂。active queue 禁止改 base/size/domain；sticky fault 不被普通 reset 清除，除非 driver ack。
- 时序热点：doorbell 高扇出 wakeup → per-queue pending bit 分层；`wait_ref_count` 扫描设上限并分拍；waiter bitmap 分块；resource map CAM → RAM + mask compare；completion router 按 source 仲裁。
- 〔未冻结〕queue 数/entry stride/doorbell 格式/interrupt vector、event_id namespace（global vs per-context）、duplicate signal policy（模型：计数并忽略；目标倾向 fault）、`wait_ref_count` 上限、IOMMU/domain 字段、QUIESCE 时 doorbell 拒绝还是延迟。多 Group 的 completion 聚合、first-error 策略、跨 Group 公平性/QoS 完全未建模，是最大不确定区。

### 3.11 Fault 与 Reset

- fault 路径不依赖 bulk data drain：走 VC0/sideband，绝不经 VC2 上报唯一错误。统一 fault bus（descriptor validator / slot checker / stream checker / DMA checker）；first-fault latch（每 Tile 1 项，always-on 或 reset-safe），后续错误只计数不覆盖；outstanding counter 饱和，underflow/overflow 直接 fault。
- reset 次序：先停 queue arbitration、取消未 issue 的 pending，已 issue 的等 completion/timeout/下游 reset ack；可证明安全才 drain，否则 cancel + poison completion；DMA 部分 beat 已完成保持、未完成 cancel；drain 不得无限等，watchdog > 合法最大 outstanding latency 且 < 软件 timeout 预算（值〔未冻结〕）。
- reset 使 generation 递增，旧 descriptor/slot frame/stream token/frame handle 全部失效；NoC credit reset 后由本地已知初值重建，不依赖 remote 旧状态；EOS 不得被 reset 静默伪造；reset clear 不清 fault ring，除非 driver ack；tile reset 不得污染其他 Tile 的共享 group queue（affected mask + owner 检查）。
- **域**：设计文档为 tile / group / device 三档（First Silicon 至少 device + group/tile soft reset）；模型另有 QUEUE 域，FSM 为 IDLE → FAULT_DETECTED → STOP_QUEUE → FREEZE_DISPATCH → DRAIN_SAFE → MARK_EVENTS → RESET_DOMAIN → CLEAR_CREDIT_CACHE → RESUME → DONE。`CLEAR_CREDIT_CACHE` 假设 credit 状态集中；若分散在 initiator/target 两端需双向重同步〔推断〕。
- fault ring 64–256 项/device〔未冻结〕；severity 编码、ring 方向/wrap/溢出策略、watchdog 值、affected mask、NoC poison 包格式、ECC 策略、retry 是否进 V1.x 均〔未冻结〕；首版取 fail-stop + drain/cancel。
- 〔冲突〕两个同名 `elenor_fault_record_v0_t`（ABI §4.4 与 FR §4.1）字段完全不同；建议以 FR 为唯一权威。

### 3.12 PMU

- 分层聚合：engine 只上报统一 stall reason 编码（active/stall_valid/stall_reason/bytes/ops/event_id），上级不重新解释 engine 内部状态；counter ID 建议 `[31:28]` scope、`[27:24]` block、`[23:16]` instance、`[15:0]` counter，宽度〔未冻结〕。tap 只采已寄存的仲裁状态，跨层聚合走 registered bus/FIFO，不跨 router 做组合归约，跨 group 聚合用低速 snapshot 网络。snapshot 绑定 context/queue/command/event/timestamp，timestamp 与 cycle counter 同源或换算系数明确（跨域对齐〔未冻结〕）。
- 〔冲突〕各 block 使用 block-local 命名（`hi_` / `sched_` / `rs_` / `sq_` / `uce_` / `slot_` / `boa_` / `EVU_MT_PMU_` / `MFE_PMU_` / `dma_`），需统一 registry；各 engine 的 stall 粒度未统一映射到 9 级。

### 3.13 物理实现与功耗

- First Silicon 只选一个可闭合的 SRAM/NoC profile（Balanced-small 方向）；DVFS/thermal 反馈不在 V1；功耗手段 = clock gating、operand gating、SRAM sleep/retention、PMU disabled toggle。
- **高扇出控制**（reset/enable/freeze/clear/context mask/event broadcast）必须本地复制寄存器或树形分发，禁止从 Global PMU 或单一全局寄存器直扇出到所有 tile；三份规格重复强调，是首要时序风险。其他热点：SRAM macro 边界、NoC crossbar grant、descriptor patch、PMU freeze 扇出、Stream credit 组合环（需 formal）。
- **CDC/RDC**：跨域只用 async FIFO / toggle sync / req-ack bridge / gray counter；event table、fault record、PMU snapshot 跨域须 shadow/latch，禁止 host 直接采样未同步多 bit 状态。DMA 可与 MC/NoC/Scheduler 不同域，launch / read rsp / write rsp / event / PMU sample 都是显式 CDC 边界；BOA FPGA 原型为单 100 MHz 时钟，不承诺多时钟域。
- **power gating 前提**：无 inflight command、pending event、DMA/NoC packet、stream token，且无 posted CSR、doorbell FIFO、pending interrupt、PMU snapshot 事务；clock gating 条件须排除同周期新 doorbell。WAKE_RESTORE 后 descriptor cache/program SRAM/state cache 的有效性〔未冻结〕，需明确是否进 retention 列表〔推断〕。必做检查：lint、CDC、RDC、SVA、formal、STA、power。

## 4. 参数台账

状态：**冻结** = 评审已冻结；**候选**；**模型** = validator 建模值；**未冻结**。末列：哪些字段进 target 指纹（取值来自 `hardware_config.yaml` 与 `config.py`；latency/launch 单位为 cycle）。

| 项              | 取值                                                                                                             | 状态                                | 进指纹 / 不进                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| 拓扑            | 1 Group × 4 Tile                                                                                                 | 模型                                | `num_tiles` 进                                               |
| 时钟            | core 1000 MHz；USE 500 MHz；FPGA 首版 100 MHz                                                                    | 模型 / 目标                         | 不进                                                         |
| L1 / Tile       | 1 MiB，16 bank，延迟 1 cycle，峰值 512 GB/s                                                                      | 模型                                | 容量、bank 进；带宽、延迟不进                                |
| L2 / Group      | 8 MiB，16 bank，64 GB/s/bank，延迟 4 cycle                                                                       | 模型                                | 同上                                                         |
| 系统预留 / bank | L1 2048 B，L2 4096 B                                                                                             | 模型                                | 随 Registry                                                  |
| Profile mode    | L1 SPM 64/56/48/32 KiB/bank；L2 SPM 512/448/384/256 KiB/bank                                                     | 模型                                | 随 Registry                                                  |
| Cache           | line 64 B，分区粒度 64 B，MSHR L2 64 / L1 16 项                                                                  | 模型                                | line、MSHR 进                                                |
| HBM             | 16 GiB，8 channel，819.2 GB/s，burst 64 B，outstanding 32，延迟 200 cycle                                        | 未冻结                              | 容量、channel、burst、outstanding 进；带宽、延迟不进         |
| Program SRAM    | 64 KiB / Tile                                                                                                    | 未冻结                              | 进                                                           |
| Slot Frame      | 16 slot                                                                                                          | v0 草案                             | 进（`frame_slot_capacity`）                                  |
| UCE             | 1 指令/周期；context 1–8（模型默认 1，AM 验收默认 4）                                                            | 模型                                | `uce_dispatch_per_cycle`、`context_count` 进                 |
| Group 调度      | active 8、pending 32、action 16、quota 8、scan 4、event 4096、inflight 32、prefetch/store 各 8、dispatch 16      | 模型                                | 全部进                                                       |
| CPU Device      | pending 32、completion 4096、issue width 1                                                                       | 模型                                | pending、completion 进；issue width 不进                     |
| BOA             | 阵列/Tile 1（目标 4，探索上限 8）；16×16；tile_m/n 16；`tile_k` 8                                                | 模型 / 候选                         | 全部进                                                       |
| AM              | 4 group × 2 slot = 8；每 context 2；容量 256 KiB（合法 1 KiB–256 KiB）                                           | 槽拓扑冻结；容量候选                | 进                                                           |
| BOA 结果服务    | result FIFO 深 4、commit 2 cycle、output drain 1 cycle、AM credit 8、L1 output credit 1                          | 软件 step 粒度；硬件由 P16/PPA 冻结 | 不进                                                         |
| EVU             | 32 lane，BF16，launch 3 cycle                                                                                    | 建议                                | lane、dtype 进；launch 不进                                  |
| MFE             | 256 GB/s；load/store channel 1/1；ingress 队列深 4/4；pipeline 深 4；stream buffer 0 B（不强制）；launch 3 cycle | 未冻结                              | channel、队列、pipeline、stream buffer 进；带宽、launch 不进 |
| USE             | state cache 128 KiB，500 MHz，launch 2 cycle                                                                     | 建议                                | 容量进；时钟、launch 不进                                    |
| DMA             | 4 channel × 256 GB/s                                                                                             | 〔冲突〕discussion 记 2             | channel 数进；带宽不进                                       |
| NoC             | vc_depth 8，router 延迟 4 cycle                                                                                  | 未冻结                              | vc_depth 进；延迟不进                                        |
| Stream Queue    | 默认深度 3                                                                                                       | 模型                                | 进                                                           |

**BOA wire 协议 v0 位宽**（P10 评审冻结 2026-10-31；原设计未给）：`tile_id` / `ctx_id` / `ctx_epoch` / `bound_engine` / `group` / `slot_index` 各 8，`logical_am` 1，`request_seq` / `request_id` / `lease_handle` / `slot_generation` / `chain_id` / `event_id` / `seq` 各 16，`step_token` / `mem_req_id` / `byte_count` 各 32，`address` 64。`seq` 禁回绕（链长 > 65534 步 = 合同错误）；其余回绕须 quiescence：`slot_generation` 需整个 Tile AM 池无 live lease、无挂起请求、无旧代在途访问；`ctx_epoch` 另需 context-id quiescence；`mem_req_id` 需 memory service quiescence。

## 5. 耦合矩阵：改左边，复验右边

| 改动                               | 必须复验                                                                                                                                                               |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bank 数 / 宽度 / 分区              | stripe padding → L2 span / no-rebind 高水位 → `prepare_resources`；BOA layout、EVU replay、MFE ping-pong、USE protected region、DMA staging、PMU bank 冲突归属；带宽账 |
| UCE context 数 / R                 | Task 准入、Frame 生成、dispatch pin、R lease 计数与归还点、event/scoreboard 预算、L1 容量                                                                              |
| engine ingress 深度                | UCE blocked-head 跳过、`engine_queue_full` 归因、MFE credit 反压链                                                                                                     |
| Group action 窗口 / scan           | S0/S1 对比基线、PMU 指标；同时扩 event/inflight 容量是另一种资源配置，不得称“只扩可见性”                                                                               |
| Event Table 组织                   | 编译器 event frontier 配额、verifier、`event_capacity`/quota、reset 编码                                                                                               |
| Profile 成员集 / ProfileController | ACK 拓扑与成员数、控制存储（独立于用户 SPM）、generation 比较                                                                                                          |
| NoC VC 数 / 编号                   | DMA 映射、Collective 通道、event sideband、PMU by-VC 计数、credit 池、fault 优先级                                                                                     |
| DMA outstanding / buffer           | NoC credit 消费、completion tracker、timeout 控制器、Event Fabric、buffer 深度                                                                                         |
| descriptor 重编码 / patch 单元     | Loader 校验、UCE patch 责任、descriptor cache 失效、ABI 版本、package CRC/relocation                                                                                   |
| 数值（`tile_k`、MAC、NaN 策略）    | numeric target、全部 typed fixture、target 指纹、联仿、epilogue 向量                                                                                                   |
| AM 槽容量 / 组数 / engine 数       | URAM 块数与 SLR 分区、AmRouter 冲突窗口、编译器资源核算与 verifier、fixture                                                                                            |
| reset domain 粒度                  | fault record domain、firmware drain FSM、event RESET、stream credit、residency metadata、PMU snapshot                                                                  |
| PMU counter ID / stall 编码        | 编译器 PMU manifest、runtime `read_counter`、phase gate、primary-owner SVA                                                                                             |

## 6. 已知缺口、冲突与不能外推的边界

REV 的 P0 来自 2026-06-23 评审，此后代码与规格已有大量变化；除注明外**未逐项复核现状**，使用前先对照最新规格。

### 6.1 ABI 与编码

- 所有 binary ABI 仍是 v0 样例，无机器可读冻结定义（REV P0-1）。**三个版本域不可混用**：软件 artifact（`schema 6 / ABI v6`）、BOA wire（`boa_protocol_v0`）、芯片 command/event ABI v0。
- 〔冲突〕command enum：主 §18 起于 `BOA_GEMM`（隐式 0），ABI 起于 `LAUNCH_GROUP_TASK = 1`；`VECTOR_KERNEL` vs `EVU_KERNEL`。event status：enum（ABI）vs bitmask（FR）。fault record 两份定义（§3.11）。`context_id` 位宽（NOC 16 位，其余 32）；`event_id` 位宽（EVU 16 位，其余 32）。
- 〔冲突〕DMA descriptor v0：DMA §4.1 与 NOC §4.1 字段集不同；地址域靠 flag 重载（`src_iova_or_l2`），建议改 `src_domain` / `dst_domain` enum。
- BOA wire 的 Tile 本地 `event_id`（16 位，`(ctx_id, ctx_epoch, event_id)`）与命令级 `event_id + sequence`（32 位）是不同命名空间，需明确〔推断〕。
- 已部分解决：ABI 文档的 wait 已改为 `event_id + expected_sequence`（对应 REV P0-4）；SQ 已定 EOS/ERROR 不占 credit。其余冲突仍在。
- wire v0 的 `SUBMIT_STEP` 含 `input_ready_tokens` / `input_usage_handles`，各消息带 `ctx_epoch`；在 `execution_ir.py`、`compiled_program.py`、`compiler/lowering.py`、`tile.py` 中均无同名字段（grep 无命中）。编译产物 v6 是否携带等价信息，联仿前须确认。

### 6.2 版本与口径漂移

- 〔冲突〕artifact 版本：代码为 `SCHEMA_VERSION = 6` / `COMPILER_ABI = "v6"`；主 §15/§19 写 v4，Chip Top 与 GS 规格写 v2，Gem 与 README 写 v5。以代码为准。
- 〔冲突〕DMA 通道数：yaml 为 4，discussion 记 2 且标“冻结”。以 yaml 为准。
- 〔冲突〕容量档位：主 §6.1 写 2 MB/Tile；主 §12.3 first-silicon 取 Balanced-small（1 MB × 64 + 8 MB × 8）；yaml 为 1 MiB/8 MiB。
- 〔冲突〕BOA 路线：主 §6.1/§8 与 BOA/OPA 规格是 4 × OPA outer-product；现行实施是 Gemmini 阵列。
- 〔冲突〕BOA 能力面：主 §3.2/§3.3 的 V1 cutline 含 INT8/BF16 GEMM，§3.4 的 Conv = MFE WindowGen/im2col + BOA；Gem §1.2 明确不做 INT8 输入 Matmul 与硬件 Conv/im2col（INT8 仅作输出类型，Conv 为软件 im2col + Matmul）。须先裁决 BOA 输入 dtype 与 Conv 路径，再定 MFE WindowGen 是否保留。
- 〔冲突〕reset 域：设计文档三档，模型多一个 QUEUE 域。EVU 见 §3.6。

### 6.3 未决的 P0（来自 REV）

- **P0-2 program 驻留路径**：评审建议冻结 V1 路径 Global/Group DMA 预取到 group L2 → Tile DMA 装入 program SRAM/I-cache → 回 `{program_local_slot, program_epoch}` ready ack → 才允许派发；Residency Manager 拥有 epoch 并在 reset 时 invalidate。真实安装路径仍未建模。
- **P0-3 Paged Attention 跨引擎 ABI**：K/V 元素序、per-head packing、stream token 单元与 credit、EOS/error 原子性、MFE → BOA 的 producer/consumer credit；缺一组冻结的 must-work 参数向量。
- **P0-6 split-K**：partial group identity、fan-in 数、reduction order、accumulator epoch、“全部 partial 已消费”的 event 语义均未定义；duplicate append 或提前写回会静默损坏输出。
- **P0-7 EVU cutline** 与验收矛盾（§3.6）；**P0-8 USE checkpoint/event-assist** 语义被推迟但 V1 依赖（§3.8）。

### 6.4 模型边界（不能外推）

- validator 是 1 Group × 4 Tile、单 router、HBM 固定延迟 + 带宽、Collective 仅 1-cycle 窗口、EVU/USE/estimate 为 Roofline；`clock_mhz` 是模拟参数，不是 Fmax；三档 fidelity 都不是 RTL cycle 等价，只有 `full_memory` 有逐腿时序。
- P09 软件回归基线周期（默认配置：每 Tile 1 个 BOA 阵列、1 Group × 4 Tile；boa-am-matmul 11332、matmul-splitk-pipeline 53675、gather-matmul-4tiles-2contexts 14961、matmul-2048x512-boa256 79196）只用于软件回归，不是器件性能；旧 P01 周期（4164 / 30135 / 7793）属旧合同，不可与之比较。
- 故意保留的限制：单 Group、固定 4-Tile 映射、无 task stealing、FIFO 队首约束、静态 Arena、L2 永久 no-rebind。放宽任一项都是新合同，不是实现细节。
- Gather 命中序是源码 authored profile，cache 容量不预测命中率；`ByteStore`（含逐字节有效位）是验证 oracle，不是产品功能。sNaN/非规范 NaN payload 的 epilogue 向量无法经生产 BF16 MAC 链到达，只能函数级对拍（P08 recheck 记 17 条 blocked）。
- `Simulator.run` 把初始化 Profile 事件从 trace 中剔除，不能据 trace 检查初始 Profile 耗时。

## 7. 跨模块、文档未覆盖的项（均为〔推断〕；模块内的见 §3 各节）

1. **统一 generation 计数域**：allocator/event/launch/residency/profile/queue/`slot_generation` 的位宽、溢出、回绕目前只有 protocol v0 对 AM/ctx 给了规则，无全局方案。
2. **时钟/电源域划分**：tile-local 控制核、阵列、NoC、HBM PHY 多半多频；先定 CDC 边界处 event/credit 计数器语义，done/fault/timeout 的 race 才能验证；每个 clock-gating 条件的跨域输入单独同步，不合并后再同步。
3. **超时整合**：per-context timeout、Group watchdog、Profile 命令超时（模型 2,000,000 cycle）的关系；建议 per-stage 超时而非整事务单计数。
4. **ring 溢出与 slot 分配**：fault ring/trace ring 溢出策略（停写/覆盖/降采样）；fault record slot 由 host 提交还是硬件分配的并发定义。
5. **Program Residency Manager**：替换策略、I-cache/D-cache 一致性、program slot 大小；program hash 校验的信任根、安全启动、DFT/扫描链均未提。
6. **跨模块 stall owner 优先序**：同一 cycle 多模块同时 stall 时，需要全局 stall-namespace 注册与仲裁表。
7. **ECC/parity 分层 scope**：program text（必须保护，patch 无法修复）、descriptor cache/shadow table（可重建）、operand/accumulator（双缓冲可掩盖；AM 长时间 RMW 累积需 sticky/fault/重算策略）、USE state cache（最高优先级）、event/barrier/fault 表、queue/token bookkeeping SRAM；并定义对应 fault code。

## 8. 重设计自查表

**先于 RTL 冻结（Phase 0 交付物）**：ownership matrix；binary ABI v0；三条 canonical trace（dense / paged / MoE）；first-silicon SRAM profile + NoC VC 假设；Slot Frame ABI、Stream Queue Contract、auto-patch 语义；SRAM/NoC 容量-带宽-面积预算模板。

- [ ] §1 的合同逐条仍成立？放宽任何一条先改软件，再改硬件。
- [ ] 动了指纹字段？是 → 重编译全部 fixture/artifact，升版并重算基线。
- [ ] SRAM：重跑逐 bank 容量证明、L2 no-rebind 高水位、碎片分类；SPM + Cache 守恒；Tag/MSHR/ECC 另计面积；7 类 consumer 带宽账（4 engine + UCE 并发，主 §28 Q11）。
- [ ] 每个有限 ID/counter：宽度、回绕、quiescence；旧响应不得污染新 generation。
- [ ] 每个队列/FIFO：O+R ≤ Q 或 credit 不变量；满时无副作用；有死锁/HOL 分析。
- [ ] 每个 terminal 事件：恰好一次；fault 与 success 互斥；fault record 先于 event；同周期 completion/timeout 优先级已定。
- [ ] reset/drain：停新 → 排安全完成 → cancel + poison → credit reconcile → generation + 1；PMU snapshot 先于 clear；各域 CDC/RDC。
- [ ] 数值：`tile_k`/融合/NaN/INT8 策略未变；变了 = 改 bits，重跑 numeric target 与全链对拍。
- [ ] PMU：每个新 stall 映射到 9 级 primary；`unknown` 为 0。
- [ ] 验收：命令经 command queue 触发（testbench 直拉 datapath 不算证据）；功能/周期/RTL/板测分层不互相冒充；模型周期不当器件性能。
- [ ] 高扇出控制本地复制/树形分发；SRAM macro 边界与 router 关键路径已寄存。
- [ ] bring-up 顺序可独立验证：command queue/event/barrier → DMA + completion event → BOA GEMM（经 queue）→ PMU → EVU elementwise → softmax/norm → MFE Page Stream → Paged Attention e2e → EVU gather/scatter → MFE Segment → USE → MoE → 多模型。
- [ ] 改了 ownership：同步更新对应模块 §2 ownership 表、主规格与本清单。

## 附录：出处缩写

| 缩写                  | 文件                                                                                                                                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 主                    | [ELENOR_Architecture_Design_v1.md](./ELENOR_Architecture_Design_v1.md)                                                                                                                                                                                                    |
| EVU / USE / MFE       | [EVU](./elenor_evu/ELENOR_EVU_Design.md) / [USE](./elenor_use/ELENOR_USE_Design.md) / [MFE](./elenor_mfe/ELENOR_MFE_Design.md)                                                                                                                                            |
| SQ                    | [Stream Queue](./elenor_stream_queue/ELENOR_Stream_Queue_Design.md)                                                                                                                                                                                                       |
| TG / SEQ / SF         | [Tile Group](./elenor_tile_group/ELENOR_Tile_Group_Design.md) / [Sequencer](./elenor_tile_group_sequencer/ELENOR_Tile_Group_Sequencer_Design.md) / [Slot Frame](./elenor_slot_frame/ELENOR_Tile_Slot_Frame_Design.md)                                                     |
| GS / HI / FR          | [Global Scheduler](./elenor_global_scheduler/ELENOR_Global_Scheduler_Design.md) / [Host Interface](./elenor_host_interface/ELENOR_Host_Interface_Design.md) / [Fault Reset](./elenor_fault_reset/ELENOR_Fault_Reset_Design.md)                                            |
| DMA / NOC / PMU / PTP | [Global DMA](./elenor_global_dma/ELENOR_Global_DMA_Design.md) / [Memory NoC](./elenor_memory_noc/ELENOR_Memory_NoC_Design.md) / [PMU](./elenor_pmu/ELENOR_PMU_Design.md) / [Physical Timing Power](./elenor_physical_timing_power/ELENOR_Physical_Timing_Power_Design.md) |
| ABI / CMP / PKG       | [Runtime ABI](./elenor_runtime_abi/ELENOR_Runtime_Command_Event_ABI_Design.md) / [Compiler](./elenor_compiler/ELENOR_Compiler_Stack_Design.md) / [Package](./elenor_executable_package/ELENOR_Executable_Package_Design.md)                                               |
| Gem                   | [BOA Gemmini Integration](./proposal/BOA_Gemmini_Integration_Design.md)                                                                                                                                                                                                   |
| L2P / 04              | [L2 显式共享](./proposal/L2_显式共享_PLAN.md) / [Context Resource Bank](./proposal/04_NEST_Context_Resource_Bank_Contract_Design.md)                                                                                                                                      |
| discussion            | [pipeline_validator.md](./discussion/pipeline_validator.md)                                                                                                                                                                                                               |
| yaml / IR_SPEC / LIM  | [hardware_config.yaml](../pipeline_validator/hardware_config.yaml) / [IR_SPEC.md](../pipeline_validator/IR_SPEC.md) / [Limitation.md](../pipeline_validator/Limitation.md)                                                                                                |
| protocol / NUM        | [boa_protocol_v0.json](../hardware/boa/protocol/boa_protocol_v0.json) / [numeric_freeze.md](../hardware/boa/reference/numeric_freeze.md)                                                                                                                                  |
| REV                   | [ELENOR_Document_Review.md](../review/ELENOR_Document_Review.md)                                                                                                                                                                                                          |
