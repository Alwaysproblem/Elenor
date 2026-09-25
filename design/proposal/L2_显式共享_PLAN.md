# L2 显式共享与 buffer 粒度提前释放实现计划

## 背景与交付目标

实现 TODO 第 5、6 项：默认私有 L2 分配；显式跨 context 只读共享 weight；允许 single-writer 发布中间结果后供多个 context 只读消费；最后使用者结束且物理访问排空后，归还 buffer 的 striped extent。恢复同 profile 下 release 驱动的 FIFO 提前准入，并提供不同 memory profile 必须等旧 context 的 HBM store 完成、profile 切换结束后才启动后续 load 的 MLIR 对照案例。

本计划只覆盖 Tile Group 内 L2 共享及其生命周期，不顺带修改 HBM 通道交织、L1 释放语义或 device slot 调度策略。

## 已核实的实现依据

- `pipeline_validator/execution_ir.py` 的 `ExecL2Buffer` 只有私有 buffer 的 slot/shape/role/alignment/bytes；`ExecGroupActionOp` 已有 `RELEASE_L2`、`BIND_L2_VIEW`，没有共享发布/引用动作。
- `pipeline_validator/tile_group.py::release_l2` 已具备先完整 preflight 再 mutation 的结构：检查 launch generation、role、dependency events、reader `input_released`、writer `output_ready`、grid pins 与 `TransferManager.has_inflight_access`，最后调用 `ArenaPool.invalidate_view`。保留这一结构，不能先扣 refcount 或释放一部分 writer pins 后才发现错误。
- 同函数把物理 handle 的 owner 写死为 `ContextBufferOwner(context_name, generation, slot)`，并要求所有同 handle 的 grid pins 属于本 launch；共享必须分离 borrower 的逻辑引用身份与 backing 的物理身份，不能只把相同 handle 填入另一 context。
- `_activate_admitted_context` 明确新 sequencer 在 post-sequencer barrier 激活，第一条 group action 最早下一拍发射。因此“release 当拍”验收针对容量归还和 admission；实际 DMA load 仍遵守 action issue、依赖和传输延迟，不能承诺零延迟 load。
- `pipeline_validator/compiled_program.py` 使用严格、冻结的 executable DTO 与 codec；新增共享元数据必须进入 compile/load verification、hash、relocation/codec，而不能仅在 runtime 挂未验证字典。
- `compiler/resources.py::_l2_lifetimes/_l2_reuse_pairs/_apply_l2_reuse` 当前确实允许 release 后经 barrier 复用同一 offset；`layout_buffers` 以 whole-stripe-round 计算 padding，但 `ArenaLayout.reserved_bytes` 还允许大于实际 high-water 的合同余量。因此不能直接把 view 的 valid-byte segments 当作完整可归还 extent，也不能假定所有历史 L2 layout 无重叠。
- `workload_ir.py::_verify_release_graph` 当前要求 out/inout 至少一次 HBM store；共享中间结果的新 publish 路径必须显式替代这一“输出一定回写 HBM”的限制，但普通私有 out/inout 保持现合同。
- `examples/configs/profile_l2_256k.yaml` 只有 L2 mode 0，user SPM 为 262144 B；不能仅把示例 `l2_mode` 改成 1，必须新增真实双档配置。
- 现有 `.vscode/run_all_workload.sh` 只运行列出的 workloads/protocol，输出实际在 `.vscode/{full-memory,runtime}-traces`，并不覆盖 NEST subgraphs，也不生成注释所称的 report/html。最终验收必须从当前入口枚举 corpus，不沿用旧记忆中的脚本位置、数量或产物承诺。
- `memory/allocator.py::release_exact` 只接受 `_exact_live` 中曾原样 commit 的 BankSegment，不支持把 whole-arena segment 的子段直接交回来。L2 commit 必须先按 backing + slack 切成 exact units 后一次性提交，才可复用现 API；无需新增第二种 partial-free allocator。
- `memory/byte_store.py::_domain_for_view` 目前把 `allocation_id` 纳入 SRAM 字节域。共享 borrower 若只换 owner/ID 而不改 L2 byte domain，会读到另一个空域；L2 必须按物理 backing identity 归一。

## 联合合同与实施顺序

先实现 R3-6 私有 buffer 提前归还，再在同一 backing 生命周期上加入 R3-5 只读引用；两者不各自实现 free-map。最后用 profile 对照和共享案例闭合跨层验收。

实际落地按以下三个可验证批次执行；后面的“步骤”按行为展开，不代表允许提交不安全的中间状态：

1. **私有释放批次：**第一、二步 + 第五步的 transaction hooks/private cancel-reset + 第八步的物理占用/extent trace；一起完成后运行 private admission、allocator、memory invariants 回归。
2. **共享批次：**第三、四步 + 第五步的 shared claims/永久容量分类 + 第六步第1–3条的 epoch/quiescence + 第八步共享计数；一起完成后运行共享、compiler/loader、profile/late-write 回归。
3. **场景验收批次：**第六步第4–7条 + 第七步示例；运行全部 CLI 双档场景、Perfetto 审计与全量回归。真实 workload 容量等待基线在第一个批次改代码前采集。

### 固定的安全边界

- L2 backing 表示唯一物理占用，context view 表示一次逻辑借用；共享 view 有各自 owner/generation，禁止伪造 producer owner 来绕过现有检查。
- 私有 backing 只有一个 owner reference；共享 backing 同时跟踪 producer reference、尚未准入的消费者 claim 和已准入的 borrower reference。消费者未 submit/尚在 WAIT_CAPACITY 也不能丢 claim。
- 不保存可独立漂移的整数 refcount。物理 backing 只记录 `producer_live: bool` 与 `claims: dict[claim_id, _L2ClaimRecord]`；每个 claim 的唯一状态为 `DECLARED`、`BOUND`（record 附 borrower AllocationHandle）、`RELEASED` 或 `CANCELLED`。剩余逻辑引用数从 `int(producer_live) + count(state in {DECLARED, BOUND})` 推导；grid pins/accepted transaction IDs 是另两道独立排空门，不重复计消费者。`DECLARED→BOUND→RELEASED` 或 `DECLARED/BOUND→CANCELLED` 单向，run generation + callsite + local slot 使 ID 唯一。公众 `nest.release` 第二次调用必须报 invariant fault；**只有 reset/cancel 内部清理**允许已 `RELEASED/CANCELLED` 的 claim 原状态返回，防重复扣账或重复归还。
- 每次 release 只撤销当前 context 的引用；最后一个引用/claim 撤销且 pins、真实 transfer、物理写入均排空时才归还完整 stripe-rounded extent。root context 退休不等待别的 context 的共享读者，backing 可独立留存。
- 采用永久 forfeiture：L2 本地 allocation 的 padded span 全部不相交；移除 barrier 后复用旧 L2 offset 的编译路径。L1 仍按原 Task Arena / tile.free 合同，不改。
- 首版共享限定同一 model invocation、同一 Tile Group、同一 L2 profile generation。只改 L1 profile 不损坏 L2 sharing；L2 切走再切回同 mode 也算不同 generation，不能延续旧共享对象。
- 跨 context 消费者以 producer context 成功完成为 readiness 边界，复用已有 device dependency 机制；publish 负责数据封存和免除强制 HBM 中转，不另造跨层 phase-event ABI。

### 源 IR 合同

`nest.alloc` 增加 `sharing = "readonly"`（省略时为 `"private"`），仍先由生产者独占写；`%ready = nest.publish %buf depends_on(...) : !nest.event<"ready">` 封存为只读；`nexus.shared.ref %producer_done slot = "X" : !nest.l2_buffer<...>` 引用该次 submit 的已发布对象。消费者 context 接受只读 L2 formal，继续通过现有 dispatch `bindings/ins` 与 `tile.subview` 读取，并使用 `nest.release` 归还自己的引用。

model 中 shared.ref 只生成可验证的静态引用关系，不成为阻塞 CPU PC 的操作。compiler 在消费者 submit 中合并 producer completion dependency，并在整个 run 开始前登记全部消费者 claim。唯一标识采用 run generation + producer binding_id + producer slot，不能仅凭 context 名、HBM 地址或 shape 猜测共享。

### 第一步：固定 L2 forfeiture 布局，保留 L1 的局部复用

1. 在 `compiler/resources.py::prepare_resources` 的 L2 路径不再传入可缩短的 allocation lifetimes，使用现有 `layout_buffers(..., lifetimes=None, slot_capacity=max(1, len(local_buffers)))`；所有本地 L2 buffer 的 stripe-rounded span 不相交。删除仅服务 L2 回绑的 `_l2_lifetimes/_l2_reuse_pairs/_apply_l2_reuse` 路径及其 verifier 镜像证明，保留 `_tile_lifetimes` 和 L1 reuse。
2. `round_bytes = layout.stripe_bytes * banks`；每个 buffer 的物理归还范围是从 `arena_offset / banks` 起、每 bank 长度为 `ceil(logical_bytes / round_bytes) * stripe_bytes` 的完整 span，不是 `AllocationHandle.bank_segments` 的 valid bytes。尾部 padding 随该 buffer 一起归还。
3. `reserved_bytes` 大于这些 spans 总和时，余量仍由 root arena 持有，直到 root retirement 归还；不把它伪装成 buffer 释放收益。空 layout、有保留余量的 context 继续合法。
4. 编译与独立 executable verifier 都检查 L2 padded span 无交叠、边界合法、每个本地 buffer 只 bind 一次且只有一次 release。`bind_view` 运行时也拒绝已 forfeited 区域重绑。旧合同因复用而预留过小时，修订对应 source resource bytes 为现有 `conservative_arena_bytes` 的结果；若超硬件 profile 则明确编译失败，不暗改硬件容量、不恢复旧别名。
5. `IR_SPEC.md` §3.8、§8 对 L2 明确“release 永久撤销本 invocation 对该 backing 的访问权；最后引用安全撤销时归还 extent，barrier 不恢复所有权”，替换“release 不改 free-map、不唤醒 root”的旧条款；L1 的原条款保持。这里是行为合同变更，不只是修注释。

### 第二步：以 backing 为唯一物理记账单位，完成私有 buffer 提前归还

1. 扩展 `memory/arena.py` 而不新增第二个 allocator：增加 L2-only `_L2BackingRecord`，保存 `backing_id`、完整 padded `reserve`、valid segments、origin arena、allocation/profile generation、逻辑引用/claim、pins/inflight 关联与 published/released 状态。现有 `_extents.commit_exact/release_exact` 仍是唯一 free-map mutation；L1 不走新 backing 释放路径。
2. L2 `commit_arena` 仍原子预留整个 local bundle，把 reserve 划为 backing spans 与 arena slack。`ArenaHandle.reserve/reserved_bytes` 保留初始布局含义，当前占用只取 live backing + live slack，不能再按初始 reservation 重复扣减。
   **exact-unit 切分：**L2 `commit_arena` 验证原 ArenaPlan 后，从 whole-bank reserve 派生各 buffer 的 padded segments 与尾部 slack segments，按 bank/address 排序，一次调用 `commit_exact(version, all_units, cycle)`。这样 `_exact_live` 保存的就是日后释放的相同 units；不能先 commit whole arena，再把任意 subsegment 传入现 `release_exact`。L1 继续 whole-reserve commit。单元切分前后的区间并集和总 bytes 必须完全相等。
3. `AllocationHandle` 增加 `backing_id: str = ""`：L1/HBM 为空，L2 为物理对象 ID。每次逻辑借用有独立 `allocation_id`、`ContextBufferOwner`；物理地址、backing ID 以及物理 `generation/profile_generation` 相同，borrower launch generation 放 owner 中。私有阶段一个 backing 一个 owner reference，尚未执行 bind 的 owner claim 同样持有容量。
4. `invalidate_view` 对 L2 完成逻辑 release 后，只在 backing 没有任何 owner/borrower/pending claim，且所有 aliases 的 pins/inflight 已空时，通过唯一的 `_try_release_l2_backing(backing_id, cycle) -> bool` 归还完整 span、推进 `pool_version`、更新计数与 trace。`unpin/end_inflight` 的 pending-finalize 也走此路径。普通显式 `release_l2` 仍要求 preflight 全通过，不把非法 release 偷换成自动等待。
   `ByteStore._domain_for_view` 的 L2 分支改为 `l2:{backing_id}:{generation}:{profile_generation}`，其它空间不改；从 producer 写入到各 borrower 读出同一字节域，不复制数据。`PayloadTracker` 只是既有地址级 layout 元数据，本次不扩其 API，也不拿它的 base-address 命中或 region 数代替真实字节、共享容量证据。
5. L2 `retire_arena` 只归还尚持有的 slack/未释放私有残余，并退休 root 元数据；早已归还的 backing 不重复 free。下一阶段加入的已发布共享 backing 可以独立于 origin arena 生存。reset/cancel 必须复用同一 final-free 路径。
6. `tile_group.py::release_l2` 保留原子 preflight。所有真正 final-free 路径（显式 release、pending finalize、cancel/reset）同步 post-mutation 的 pool snapshot 与 `_record_l2_occupancy`，并把 `_l2_capacity_change_cycle` 记为实际归还拍。`_l2_reserved_bytes`/`_l2_live_bytes` 按 backing 维护，borrow/invalidate 逻辑 alias 不各加减一份；`retire_context_arena` 不再无条件扣初始 reservation。admission 只依赖 pool_version，不新增重复的 extent_pool_version 重试门。
7. 保留 `GroupPortAdapter._admit_pending` 的 admission-version 门控与 SAME/COMPATIBLE 分类 FIFO。Group tick 末的 completion harvest 已会调用 port 重试；确保这一处能看到当拍改变后的 pool version，不另加逐周期 busy retry，不让后来的请求越过同类队头。

### 第三步：显式只读共享的语法、编译结果和静态证明

语法固定如下（新增 API，不是现有可运行示例）：

```mlir
%w = nest.alloc slot = "W" role = "in" sharing = "readonly"
  shape = [64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<64x64xbf16>
%published = nest.publish %w depends_on(%prefetched) : !nest.event<"published">
nest.release %w depends_on(%prefetched, %published)

// nexus.program 内：引用这一次 submit，不引用 context 模板名。
%p = nexus.submit_context.async @load_weight(%W) : !nexus.event<"p">
%shared = nexus.shared.ref %p slot = "W" : !nest.l2_buffer<64x64xbf16>
%b = nexus.submit_context.async @reader(%B_OUT, %shared) : !nexus.event<"b">
%c = nexus.submit_context.async @reader(%C_OUT, %shared) : !nexus.event<"c">
```

1. `dialects/elenor.py` 增加 `NestPublishOp`（`nest.publish`）、`NexusSharedRefOp`（`nexus.shared.ref`），并给 `NestAllocOp` 增加默认 `"private"` 的 sharing 属性，只接受 `"private" | "readonly"`。复用已有 shape、字符串属性、operand group、depends_on、event parse/print helpers 与 dialect registration，不另造共享 tensor type。
2. Context formals 遵循现 Tile Program 顺序：零或多个 `NestGlobalMemref`，随后零或多个 `NestBuffer`；后者只代表只读共享输入。`NexusSubmitContextOp.actuals` 接受这两类，逐一匹配类型类别、完整 shape/dtype，不能仅比 byte count。program 的外部输入仍只允许 HBM global memref。L2 formal 名需非空、唯一，且不与本地 alloc slot 冲突。
3. 新增冻结 DTO `ExecSharedInput(slot: str, dims: tuple[int, ...], dtype: str, element_bytes: int, bytes: int, producer_binding_id: str, producer_slot: str)`；`ExecTileGroupTask.shared_inputs: tuple[ExecSharedInput, ...] = ()`；`ExecL2Buffer.sharing: str = "private"`。task 的 `l2_buffers/layout/resource_contract.l2_spm_bytes` 仍仅描述本 context 新建的分配，shared inputs 不重复计容量。
4. 新增 `ExecGroupActionOp.PUBLISH_L2 = "publish.l2"` 与 `ExecPublishRequest(buffer_slot: str, reader_dispatch_ordinals: tuple[int, ...], writer_dispatch_ordinals: tuple[int, ...], dependency_events: tuple[str, ...])`。发布是一个 synchronous Group action，`dst` 为发布完成事件；复用 event table 和 ready-action 调度，不新增 device event 类型。
5. `lower_model_ir/_lower_context` 依 submit callsite specialization 将 shared.ref 解为 producer binding ID + slot，写入消费者 task.shared_inputs；HBM actual_inputs/global_inputs 过滤后仍为原来的 dense global 索引。消费者 submit dependencies 合并对应 producer completion events，稳定去重；shared.ref 本身不生成执行指令。沿用现有 `DeviceLaunchRequest`，不把同一共享映射再复制到 device op、request 和 task 三处。
   source 的 `_verify_device_memory_dependencies` 必须按同一 explicit ∪ shared-producer dependency 集合计算 ancestors，再校验 HBM hazards；不能一边 lowering 自动补依赖、一边 source verifier 在补依赖前错误拒绝合法共享调用。独立 executable verifier 从 task.shared_inputs 重算这些必需边，而不信任已存 dependencies。
   imported formals 在 lowering objects 表中映射为 role=`"in"` 的 L2 buffer 视图描述，但不加入 task.l2_buffers。增加 `ExecGroupActionOp.BIND_L2_IMPORT = "bind.l2.import"`，每个 formal 在 task actions 开头发一个 `args=(slot, shared_input_index)`、带内部 dst bind event 的同步动作；successful admission 已创建 borrower handle，该动作只校验现有 handle、不再 acquire/refcount。dispatch/release 沿用 bind_events 依赖。更新 `TileGroupSequencer.issue_registered`（包括 L2 issue gate 列表）和独立 verifier 的 bound/released 集合为 local allocs ∪ shared_inputs；layout 仍只含 local allocs，不把 import index 错当 arena layout index。PUBLISH_L2 也在 issue_registered 分支执行并通知 dst，保持 release/terminal 可排空。
6. Source verifier 和 `execution_verifier.py` 都从实际 descriptors 计算读写集：共享 formal 可进入 `bindings/ins`、L2→L1 load、L2→HBM store 的源，不能是 prefetch/tile.store 的目的，也不能进 dispatch `outs`、再次 publish 或转借为新的 export。同一 backing 在一次 submit 中禁止重复作为两个 formals 传入；多个不同 submit 的读者合法。
7. producer 是唯一可写 context，允许其多个合法、有序的 writer dispatch 在 publish 前构建结果。每个 readonly alloc 必须恰好一次 publish；publish 的直接 dependencies 必须恰好覆盖本 buffer 全部 prefetch/store 完成、reader input_released 与 writer output_ready（稳定去重），lowering 再补内部 bind/hazard 依赖。publish 是 producer 最后一次数据访问边界，之后只允许 release，不允许其继续读/写/store；release 仍依赖全部原访问完成事件和 publish。private buffer 不允许 publish。不能发布从未初始化的 buffer：至少有完整 prefetch 或实际 writer；执行范围/phase 证明复用现有 verifier。
8. `_verify_release_graph` 接受 shared formals 的一次 release；只对已 publish 的本地 readonly out/inout 解除“必须有 HBM store”的旧要求。private out/inout 不变。这样 A 的 X 可以直接 L2 发布给 B/C，而非先写 HBM 再各自 prefetch。
   源级 release 的精确集合保持可判定：private=`R_input_released ∪ P_prefetch ∪ S_store`；exported readonly=上述集合再加 publish event；imported readonly=`R_input_released ∪ S_store`。集合内不允许重复，writers 的 output_ready 通过 publish/既有 normalization 闭合，不要求作者重复列出一套新的隐式事件。
9. `compiler/api.py::_relocations`、`compiled_program.py::Relocation` 和 `runtime/relocation.py::relocate_task` 增加 `"publish_events"`，重定位 request.dependency_events；`dst`、action dependencies 继续用通用重定位。事件预算在 publish 插入后走 `finalize_event_resources`，终止闭包包含 publish/release。
   `normalize_action_dependencies` 将 PUBLISH_L2 与 RELEASE_L2 一样加入真实 writer/read completion 依赖并回填 request.dependency_events；publish completion 作为其后 release 的硬依赖。保留原有最大事件 frontier/事件复用机制，不把 producer 的 NestEvent 搬进 consumer 的 event namespace。
10. 严格 codec 注册上述 DTO、字段与枚举；compiled schema 升为 `2`、compiler ABI 升为 `"v1"`，旧 artifact 明确拒绝并要求从 source 重编译，不做运行期兼容 shim。compile 和 load 都独立验证 producer 存在、export 已发布、consumer 关系、shape、权限、依赖与 profile epoch，修改 artifact 元数据不能绕过验证。

### 第四步：把共享 claim 加入准入、发布和最后使用者释放

1. `TileGroup.begin_launch` 从已验证的 specialized tasks.shared_inputs 一次性建立封闭的 consumer manifest；key 为 `(run_generation, producer_binding_id, producer_slot)`，claim key 为 `(consumer_binding_id, local_slot)`。此时仅登记 dormant metadata；producer backing commit 时才物化其 pending claims。dormant manifest 不占容量、不阻塞 entry prefix/profile 初始化。没有动态名称查找或按 HBM 地址自动去重。zero-consumer export 合法，producer 自己 release 即可最终归还；新的 run 开始前必须确认旧 physical backings/claims 已全部清零。
   producer backing commit 时按该 manifest 为每个未来消费者建立一条 `DECLARED` claim；未用在任何 submit actuals 的 `nexus.shared.ref` 不创建 claim。`BOUND` 只在 consumer 原子准入时写入，物理 backing 只在 `producer_live=False` 且无 `DECLARED/BOUND` claims、pins、inflight 后归还一次；归还后写终态并从 live-backing map 删除，旧 ID 不得复活。
   让 run-scoped manifest 持有每个 `_L2ClaimRecord` 直到 run 终止；活 backing 的 `claims` 只引用这些同一对象，不复制状态。backing final-free 从物理 live map 移除，但 manifest 保留 `RELEASED/CANCELLED` 终态供重复内部 cancel 查验；只有完成零泄漏断言或安全故障 drain 后才清空 manifest。这样重复 cleanup 不依赖已释放 backing 的查找，也不会保留跨 run 的历史元数据。
   静态 manifest 只按**实际消费者 submit callsite**建立 claim，不按未使用的 `nexus.shared.ref` SSA 数量或 tile 数计数。现 model IR 的程序体必须以 `nexus.return` 结束（`workload_ir.py::_verify_nexus_program`），device PC 逐个处理 submit；正常完成要求 PC 已 return 且无 pending/active launch（`device.py::CpuDeviceController.done`）。所以正常路径的未提交 claim 属 invariant fault；fault/cancel 走安全 drain，不能把缺失 reader 当已释放。
2. producer admission 只分配一次 backing；发布前 producer 独占可写，publish 后 backing immutable。producer release 撤销自己的逻辑 view，但尚未提交、WAIT_DEPS、WAIT_CAPACITY 的消费者 claims 都保留。B 释放不能让尚未准入的 C 丢数据。
3. consumer admission 的纯 plan 同时检查 private arena、所有 shared backing 的 published/readiness、run/profile generation、shape、合法 claim、事件/slot 预算；任何失败都不消耗 claim、不生成 view、不扣容量。全部成功后原子提交 private arena 与 borrower views，pending claim 转为 bound reference，不新增一份 refcount。
   missing/unpublished backing、错误 producer/run/profile identity、已消费 claim 属合同错误，按现 fault 路径拒绝；它们不是 WAIT_CAPACITY，不能无限等待。正常消费者的 producer readiness 已由 device dependency 保证，不另设共享就绪 wait queue。
4. 增加 `ArenaPool.borrow_l2_view(backing_id: str, owner: ContextBufferOwner, claim_id: tuple[str, str], cycle: int) -> AllocationHandle`；此方法仅在 successful admission commit 调用，不修改 free-map/pool_version。alias overlap 只允许同 backing 的 published readonly views，不放宽跨 backing 重叠检查。消费者 `(generation, local_slot)` 仍接入现有 `_l2_handles`、role binding、tile dispatch 路径。
   imported handle.arena_id 属于 consumer 的逻辑 arena，并登记到该 arena 的 view 列表；backing record 另存 origin_arena_id。空 private layout 的 consumer 也提交一个零容量逻辑 arena。由此 root 退休检查自己所有 views，物理 accounting 仍只遍历 backing，不按 arena.views 累加重复容量。
5. `publish_l2(request: ExecPublishRequest, sequencer: TileGroupSequencer, cycle: int) -> bool` 与 release 复用共同的 read-only access preflight；先验证全部状态再解除该 producer 已完成的 writer pins、封存 backing、发完成事件。release 的 pin检查只检查当前逻辑 view，其他 borrower 的读 pins 合法存在；final-free 再检查整个 backing 的 aliases。
6. 增加 `ArenaPool.assert_access(handle: AllocationHandle, permission: str) -> None` 与 `permissions(handle: AllocationHandle) -> str`，permission 仅 `"r"`/`"w"`；private/producer 尚未 publish 为 `"rw"`，published/borrower 为 `"r"`，invalidated/stale view 一律拒绝。物理访问检查放在 `validate_transaction_generation` 的 source/destination 分支；`_resolve_view`/tile formal materialization 填入 pool 给出的 permissions。不能只信 descriptor 的字符串，destination commit 前必须再次查 authoritative pool 状态。
   编译、publish preflight、borrow preflight、transfer acquire 和 destination commit 都检查权限；publish 后 producer 的逻辑 view 保留到 release，但不再允许 source IR 发新数据访问。INVALIDATE_PENDING 只禁止新 pin/transaction，既有已登记 transaction 仍可排空，不把它提前当成 INVALIDATED。
7. `retire_context_arena/context_cleanup_ready` 只要求本 context 的 views、routes、leases、jobs 结束，不要求 exported backing 已物理 free；否则会形成“producer 等 reader release、reader 等 producer completion”的死锁。model 正常完成时仍要求所有 claims、backings、views、transfers 清零。
8. 新增 `TileGroup.assert_l2_closed() -> None` 检查 live backings/`DECLARED/BOUND` claims/borrowed views/pins/inflight、Arena slack 和 free-map 守恒均为零。`Simulator._run_model` 现有 `if controller.succeeded` 在 `simulator.py:250–253` 会**直接 break**：成功分支先调用此断言；若失败，调用既有 `CpuDeviceController._enter_fault(reason, cycle)` 使故障对 controller 可见，再 `controller.note_fault_drain_started`、`_ensure_fault_drain`，并 `continue` 而不是 break；下一轮沿现 `if controller.faulted` 分支持续 Group step/harvest，只有 `reset_domain.is_done` 后才返回 `completed=False`。standalone 完成出口同样检查并进入真实 drain；`begin_launch` 拒绝旧 run 的 backing/claims。此守卫是泄漏探测，不能替代合法 release。

### 第五步：闭合滞后访问、异常终止和容量分类

1. 当前全库搜索 `begin_inflight/end_inflight` 没有实际调用方，不能把 `_ViewRecord.inflight == 0` 当作已成立的硬件安全证据。对所有 accepted L2 src/dst transaction 接通引用登记：`TransferManager` 增加可选 `reference_acquire: Callable[[MemoryTransaction], None]`、`reference_release: Callable[[MemoryTransaction, int], None]` hooks，由 TileGroup 提供；没有 SRAM 的独立 manager 测试可不配置，有 L2 endpoint 的实际 group 必须配置。
2. submit 完整验证后、任何 transport issue 之前，对 transaction 中去重后的所有 L2 views 一次性 preflight 并登记 `begin_inflight`，不能依据 issuer 推断只保护一个 context。提交失败零登记；登记中失败必须完整 rollback。`acknowledge(transaction_id: str, cycle: int)` 在 terminal 且实际字节/owner 后处理完成后调用 release hook，再删除 transaction；`acknowledge_all_terminals(cycle: int) -> int` 同步改签名。逐个迁移 `engines.py`、`tile_group.py`、`memory/profile_controller.py`、transfer 内部和测试调用；最终以全仓 `.acknowledge(` / `acknowledge_all_terminals(` 搜索确认无遗漏。
3. 不在请求 cancel、看到引擎 idle 或仅 `input_released` 时提前撤销 transfer ref。DONE/CANCELLED/FAULTED 都要等确认安全的 terminal acknowledgement；cancel-requested 的 accepted leg 继续 drain。保持现 `has_inflight_access` 作为 release preflight 的独立交叉检查，不用它替代 physical reference ledger。
4. 审计并覆盖：Global DMA prefetch/store；MFE load/store；Gather 的 L2/L1 refill 与最终 destination write；cache dirty clean/writeback；MSHR waiter；maintenance transfer；NoC cancel drain。缓存以 HBM provenance 为身份，SPM 和 cache 属不同 bank intervals；不能无依据在每次 SPM release 发全 cache flush。凡路径真实持有 L2 AllocationHandle 必须进入上面的 ledger；纯 cache/HBM 路径继续受 profile drain 约束。
   L2 ledger 同时在 view 上记录 transaction ID、在 backing 上记录 `(allocation_id, transaction_id)`，使 view release 只等自己的访问而 final-free 等全部 aliases；不把正常的另一个 borrower 在飞读误判为当前 borrower 非法 release。acquire 在 `_transactions` 安装 accepted transaction 时原子执行并记录成功标志：未成功 acquire 的拒绝/故障 transaction 不得在 ack 时对不存在的 ref 扣减；已 acquire 的 fault transaction 一直持有到 ack。
5. 成功的公共 `nest.release` 先做完整只读 preflight，才执行一次 `BOUND→RELEASED` 或 producer_live=True→False；对同一 owner 再调用必报 `"double release"` invariant fault，失败保持 roles/handles/claims/pins/capacity/pool_version/trace 不变。内部 fault/reset 的 cancel 路径对已 `CANCELLED/RELEASED` 的 claim 可重复调用但不修改 backing/free-map。wrong owner、stale generation、写只读 backing 仍为硬错误。
   `Simulator._run_model` 目前触及 `max_cycles` 就退出 `while` 并直接返回 incomplete（`simulator.py:255–256`）；仅调用 `_ensure_fault_drain()` **不会让 ResetDomain 前进**。改为 cap 后以 `controller.fault_reason or f"cycle cap {sim.max_cycles} reached"` 调用现 `controller._enter_fault(reason, cycle)`（保留已发生的终态泄漏原因）、`_ensure_fault_drain`，再独立执行上限为 `hw.memory_target.profile_command_timeout_cycles` 的**额外 drain 循环**：每拍 `controller.step`（故障态不发新 submit）→`group.step`→`controller.harvest_completions`，待 `reset_domain.is_done` 后报告 `completed=False`。现 `ResetDomain.max_drain_cycles` 仅升级 cancel 请求，不算隔离成功；若额外窗口耗尽或 step 报安全故障，设置 `TileGroup.poisoned_reason: str | None`，`begin_launch/try_admit_context_task` 拒绝复用，返回失败并列出未清 backing/claim/txn IDs，**保留物理占用**。显式恢复只可调用现 `TileGroup.reset()`：先确认 reset drain 已 DONE、`assert_l2_closed()` 通过、无活 route/transfer，再运行原 quiescent reset 流程并在成功末尾清 poison；不能隔离时 reset 仍拒绝，绝不强行 free。
6. fault/reset 先停止新 borrow/publish，再取消尚未准入/尚未 submit 的 manifest claims；active borrower 按原 drain 顺序关闭 routes、jobs、pins、transactions，然后释放 view/ref。未发布 producer 出错不得唤醒消费者为 SUCCESS；使用现 device ERROR 传播。不得把 reset 实现成清空 registry/free-map，否则会隐去 late write 和 double-free。
   `release_context_memory` 的顺序必须调整：现有代码在所有 invalidate/retire 之后才 `acknowledge_all_terminals`，接通 inflight 后会互相等待。改为确认 cancel/isolation 且无可继续写入的 transactions 后，先 `acknowledge_all_terminals(cycle)` 归还 transfer refs，再 invalidate views/退休 arenas；正常 engine/group/profile 终态也用相同 ack API。取消未 bind 的 owner claim、未 submit 的 consumer claim 同样需走 registry，不只遍历 `_l2_handles`。
   bulk ack 前先沿现 `MFEEngine.retire_isolated`/profile cancellation 路径退休 terminal owner jobs，确保稍后的 tile.reset 不再持有待 ack transaction；bulk ack 只收尾剩余 terminal records。不能先删 manager 记录，再让仍存活的 engine job 二次 acknowledge。

7. 维持现 empty-pool per-bank/alignment permanent-capacity 检查，以及 free bytes 不足/碎片的不同 wait reason。共享模型另外在 compile/load 验证每个消费者“去重后的必要 shared padded spans + 自有 arena”在每个允许 profile 的每 bank 容量内；不能以 consumer private bytes=0 推断模型总需求为0。
8. 增加纯查询 `ArenaPool.can_fit_with_retained(layout: ArenaLayout, backing_ids: tuple[str, ...]) -> bool`：从该 profile 的 pristine per-bank free intervals 扣除这些 live backing 的 padded segments，再复用 `_first_fit` 检查整个待准入 arena，不改变状态。port 队头失败时，构造其**必定不能先释放**的 backing 集合：某个未释放 claim 属于此队头、同类别已排在其后的 request，或其 submit completion-dependency 后继。依赖后继关系从已编译 device schedule 在 begin_launch 一次性建立，不分析 source IR；同类别 FIFO 阻挡取当前队列。只要一个 claim 满足条件，该 backing 就必须保留。
9. 若在假定其它所有 reservation 都已释放的上述最乐观 free map 上仍放不下队头，则以现有 PERMANENT_CAPACITY 机制报 `L2 capacity fault: retained shared extents prevent FIFO-head admission`；这证明等待不能解决，不靠 timeout。否则继续原 WAIT_CAPACITY/WAIT_FRAGMENTATION 和 version-gated FIFO，不因“暂时没有 active context”就贸然判永久错误。只在 pending/admission token 改变时重算；不跳过队头、不隐式 spill。测试同时覆盖 own-shared 碎片和队尾 reader 持有 backing 的 FIFO 互锁。

### 第六步：保持 profile quiescence，增加真实 load/store 对照例

1. 复用 `compiler/profile_pass.py::bind_profiles` 的 `reconfigure/wait_events`：不同 L2 mode 仍等待完整 root completion history，生成普通 device await，再发 ProfileReconfigDesc。不将 `input_released`、extent free 或 buffer publish 当作可切档 frontier。
2. 新共享静态合同还要检查 generation interval：任何 L2 改档前，该 epoch 的所有 producer 和已声明 reader 都必须在改档前已 submit，并在 frontier 内退休；若 reader 出现在改档后，编译/load 直接拒绝跨 epoch reference，不能在 controller drain 中等一个根本未提交的未来 reader。仅 L1 改档不要求清除 L2 sharing。
3. `ArenaPool.reconfigure` 及 `ProfileController._closure_ledger/_level_quiescent/_verify_member_ready` 增加 live backing、alias pins/inflight、已物化 pending/active claims 检查；仅 `live_arenas == 0` 或 origin context_done 不足以切档。dormant manifest 不在 quiescence 集合内，L1-only closure 不数 L2 backing。现有 CLOSE_ISSUE→drain→Prepare/Commit ACK→OPEN_ISSUE 顺序不变，不以“新 profile 区间容得下旧地址”为由迁移 live backing。
   还需把 `begin_launch/reset` 的“previous launch retired”检查扩到独立 backing/claim ledger；origin arena 全退休但有 reader claim 时不能启动新 run。profile 检查只看当前 epoch 已物化对象，不能把未来 epoch 的 dormant manifest 当作存活占用。
4. 新建 `examples/configs/profile_l2_256k_switch.yaml`，以现 `profile_l2_256k.yaml` 为基础：16 banks、physical 327680 B、system reserved 4096 B/bank 不变；mode 0 为 SPM 20480/cache 0 B/bank（user 256 KiB），mode 1 为 SPM 12288/cache 8192 B/bank（user 128 KiB）。两个 mode 的物理总量相同，保留 read_only cache 与原 maintenance caps，L1 继承默认。
5. 新建 `examples/scenarios/l2_admission_profile_switch.mlir`，复用 `l2_admission_wait.mlir` 的 tensor shapes、A/B programs、HBM bindings、pow_ops 和 A 的真实 HBM store。A `l2_mode=0, allowed_profiles=[0]`，B `l2_mode=1, allowed_profiles=[1]`；保持 L1 mode 0。model 连续 submit A/B，无人为插入 await；让 compiler 产生必需的完整 A frontier。
6. `examples/run.sh` 增加入口 `l2-admission-profile-switch`，复用原 `l2-admission-wait` 参数，只改 IR/config 路径；保留 `--context-mode 2 --device-context-mode 2 --hw-override num_dma_channels=2 --hw-override hbm_fixed_latency_cycles=10 --max-cycles 500000` 和三份 131072 B bindings。不写死 fidelity，允许两个档位通过额外参数选择。
7. 对照关系固定：同 profile 原场景 `A.input extent free == B.active_cycle < A.pow end`，B HBM→L2 prefetch 与 A pow 有真实时间交集；不同 profile 新场景 `A.HBM store done <= A.context done <= L2 profile command start < profile command done <= B.admit < B.prefetch start`，且 B 不在 A pow 窗口 load。真正放行边界取 command 完成/OPEN_ISSUE，而不是只看到 COMMIT stage。

### 第七步：共享 weight 与分支中间结果的可运行例

1. 新建 `examples/scenarios/l2_shared_weight.mlir`（入口 `l2-shared-weight`）：loader context 将 `[64,64] bf16` 的 W（8192 B）prefetch 一次、publish、release、return；两个独立 consumer context 各 dispatch 四个 tiles，W 的 `tile.subview` 不带 task 维，每 tile 从同一 backing 读 W。每个 consumer 只有自己的 `[4,64,64] bf16` 输出（32768 B），最终写入 B_OUT/C_OUT。
2. 新建 `examples/scenarios/l2_shared_fanout.mlir`（入口 `l2-shared-fanout`）：A prefetch A_IN（8192 B），通过真实 tile.load→tile.store 生成 L2 X（8192 B），publish X 后不对 X 发 HBM store；B/C 如上直接借用 X，分别产生 B_OUT/C_OUT。整个 module 不提供 X 的 HBM binding，防止实现无意退回 HBM 中转。
3. 两例同用现 `profile_l2_256k.yaml`、固定 L2 mode0/L1 mode0、context/device capacity 2、DMA channels 2、HBM latency 10、max_cycles 500000。入口绑定 W/A_IN=`0x100000:8192:r`、B_OUT=`0x200000:32768:w`、C_OUT=`0x300000:32768:w`；分别以实际参数名传入。loader local resource=8192，fanout A local resource=16384，consumer local resource=32768；consumer shared W/X 不计入其 local reserve。
4. 用相同 shapes/tile programs 的 private-weight 对照 fixture（不是新增另一套运行时模式）测量：两个私有 reader 各 prefetch W，shared 只一次；只对 W 流量断言 16384→8192 B，不把每 tile 的 L2→L1 load 当成被消除。consumer 仍各读一次 L2。
5. 延迟 C 的首次 read、或让 C 在 B 释放后才准入，证明 producer 退休、B release 均不归还 X/W；C 最后 release 才出现唯一 physical free。两个场景都需支持普通 run.sh 运行，而不是只存在单元测试里。

### 第八步：让记账、trace 与新生命周期一致

1. `ArenaPool.snapshot` 的 per-bank reserved/allocated/free/padding 使用 live physical backings + arena slack；共享 aliases 不重复计物理容量。每 bank 始终满足 `allocated_bytes + free_bytes == user_spm_per_bank`，物理 cache/system-reserved 另列。保留逻辑 live_views/pin_count，新增 `live_backings`、`pending_shared_claims`、`active_shared_references`；pending 只数已有 backing 的未准入 claims，不数 dormant manifest。report 的 zero-leak 检查同时检查这些字段，不再只看 live_arenas/live_allocations。
2. arena rows 的 `reserved_bytes` 改为其仍实际持有的物理容量，并加 `initial_reserved_bytes` 记录原 admission 合同；origin 已退休但存活的 shared backing 单列 backing rows，不伪装成活 root。`live_view_bytes`/protocol live bytes 按 backing 去重；pending reader 仍持有的 published data 不因 producer view invalidated 被扣成0。
3. 每次真正 final-free 发 `l2_extent_release` instant，含 `backing_id`、origin buffer/arena、run/profile generation、released bytes、各 bank released segments、release cycle、post-mutation pool_version。逻辑 `buffer_view_invalidate` 保留并带 backing ID/剩余 claim 数；非最后 borrower release 不得发 physical-free 事件。
4. `arena_lifetime` 保留但只表示 root/task 元数据生命周期，不再表示恒定容量占用。新增 `l2_backing_lifetime` complete slice 表示 physical commit→final-free；实际占用积分用 reserved bytes counter。接入现 `MemoryTrace._pool_snapshot`，report 从真实 snapshot 汇总，不能反推 trace 来凑计数。
5. 把行为定义同步到 `IR_SPEC.md` §3.8、§7–§9：local reserve vs shared references、release forfeiture、publish 替代中间结果 HBM store、profile epoch 禁止跨越。原 `l2_admission_wait.mlir` 中“runtime 不会 wait”的陈旧注释删除；runtime 与 full_memory 使用相同容量合同。

### 四类 MLIR 示例草图

以下仅展示构造差异；完整、可解析的文件按第六、七步实现。`sharing`、`nest.publish`、`nexus.shared.ref` 是本计划将新增的语法，当前 parser 不接受。dispatch 的完整 `signal_policy`、视图和输出清理不能在正式示例中省略。

**1. 同 profile 提前准入：沿用 `l2_admission_wait.mlir`。** `ctx_a` 的 2×128 KiB 私有 L2 占满 256 KiB；`ctx_b` 要申请另 128 KiB，先 WAIT_CAPACITY；A tile 发 `input_released` 后，Group 的 `nest.release` 才回收输入 extent。

```mlir
// ctx_a：完整 prefetch/dispatch/store 仍沿用现有文件
nest.release %a_input depends_on(%ev_inrel_a, %ev_pref_in)
%stored = nest.dma.store.async %a_output into %a_out_hbm
    depends_on(%ev_outready_a) : !nest.event<"stored">
nest.release %a_output depends_on(%ev_pref_out, %stored)

// nexus.program：不插入 A→B 的完成依赖
%done_a = nexus.submit_context.async @ctx_a(%A_IN, %A_OUT) : !nexus.event<"done_a">
%done_b = nexus.submit_context.async @ctx_b(%B_IN) : !nexus.event<"done_b">
```

目标是 a_input 实际 final-free 与 B root admit 同拍；B 首条 prefetch 最早下一拍 issue，HBM 数据到达更晚；A 的 pow 仍在进行。

**2. 切 L2 profile 后才能 load：新 `l2_admission_profile_switch.mlir`。** 保留同一 A（mode0，完整 HBM store），仅将 B 设为 mode1、`allowed_profiles=[1]`，小池 YAML 要同时定义 mode0/1：

```mlir
// ctx_b 的资源头
resource_contract = #nest.context_resources<
  l2_mode = 1, allowed_profiles = [1],
  logical_tasks = 4, l2_spm_bytes = 131072, requested_contexts_per_tile = 1>
// ctx_b 体内保留真实 nest.dma.prefetch.async %B_IN into %b_input
// nexus.program 仍连续提交：
%done_a = nexus.submit_context.async @ctx_a(%A_IN, %A_OUT) : !nexus.event<"done_a">
%done_b = nexus.submit_context.async @ctx_b(%B_IN) : !nexus.event<"done_b">
```

编译器已有 profile pass 应在 B 的 submit **之前**自动生成对 A 完成的 device await 与 L2 reconfig；不人为把 `nexus.await %done_a` 写进源 IR。必须观测 A HBM store 完成→A root 完成→切档/OPEN_ISSUE 完成→B admit→B 首次 prefetch。

**3. 跨 context 共享 weight：完整独立模块草案见 [`l2_shared_weight.mlir`](local://l2_shared_weight.mlir)。** 当前 parser 不支持其新语法；实现后正式源文件落在 `examples/scenarios/l2_shared_weight.mlir`，不能把 session-local 草案误认为工作树文件。loader 一次 prefetch W（8192 B）、publish 并撤销自身 view；在 `nexus.program` 创建一次 `nexus.shared.ref`，传入两次 `@reader` 调用。每个 `@reader` 通过自己的 `!nest.l2_buffer<64x64xbf16>` context formal 接收引用，将该 formal 放入 dispatch `bindings/ins`；`tile.program @copy_weight` 使用独立的 L2 formal `%w` 生成 `tile.subview %w`，绝不跨越词法作用域引用 `nexus.program` 的 `%shared_w`。两次 L2→L1 读取依然真实发生，省去的是第二次 HBM→L2 prefetch。

```mlir
// nexus.program 内：只产生并传递 capability。
%loaded = nexus.submit_context.async @load_weight(%W_IN) : !nexus.event<"loaded">
%shared_w = nexus.shared.ref %loaded slot = "W" : !nest.l2_buffer<64x64xbf16>
%b_done = nexus.submit_context.async @reader(%B_OUT, %shared_w)
  depends_on(%loaded) : !nexus.event<"b_done">

// @reader(%OUT: !nest.global_memref<4x64x64xbf16>,
//         %weight: !nest.l2_buffer<64x64xbf16>) 内：
// dispatch bindings(%weight, %output) ins(%weight) outs(%output)
// 在单独的 tile.program @copy_weight(%task: !nest.task,
//                                 %w: !nest.l2_buffer<64x64xbf16>, ...) 内：
%v = tile.subview %w offsets = [0, 0] sizes = [64, 64]
  strides = [1, 1] : !nest.l2_view<64x64xbf16>
```

**4. A 的 L2 中间结果扇出给 B/C：完整独立模块草案见 [`l2_shared_fanout.mlir`](local://l2_shared_fanout.mlir)。** 实现后正式源文件落在 `examples/scenarios/l2_shared_fanout.mlir`。A 的 tile.load→tile.store 在 L2 建立 X，`output_ready` 后 `nest.publish %x`，不为 X 定义 HBM formal/binding 或 store。B/C 分别以同一 `nexus.shared.ref` 调用 `@reader`；各 consumer 的 context L2 formal → dispatch `bindings/ins` → tile program L2 formal 链与例 3 相同，不将 nexus SSA 直接送进 tile program。

```mlir
// @producer 内：X 是唯一 writer 写入、发布后只读的 L2 allocation。
%x = nest.alloc slot = "X" role = "out" sharing = "readonly"
  shape = [64, 64] dtype = "bf16" alignment = 256
  : !nest.l2_buffer<64x64xbf16>
%published = nest.publish %x depends_on(%x_ready)
  : !nest.event<"published">
nest.release %x depends_on(%published)

// nexus.program 内：不传 X 的 HBM 指针。
%a_done = nexus.submit_context.async @producer(%A_IN) : !nexus.event<"a_done">
%shared_x = nexus.shared.ref %a_done slot = "X" : !nest.l2_buffer<64x64xbf16>
%b_done = nexus.submit_context.async @reader(%B_OUT, %shared_x)
  depends_on(%a_done) : !nexus.event<"b_done">
%c_done = nexus.submit_context.async @reader(%C_OUT, %shared_x)
  depends_on(%a_done) : !nexus.event<"c_done">
```

两个独立草案都含完整 `builtin.module`、tile programs、contexts、model 和全部输出 release/HBM store；只有拟议 ops/formal 合同尚不能由当前 parser 编译。`@make_x` 用单 tile（`placement=1`），B/C 各用四 tile。首版 B/C 等 A 完整成功退出，而非只等 publish；最后一个安全 reader release 才归还 X/W。

## 验证方案

以下命令在仓库根目录执行，使用 `elenor-validator` conda 环境。计划阶段不执行产物生成；实施阶段使用新目录 `examples/artifacts/l2-sharing-release/<run-id>/`，保存 source/config SHA-256、命令、compiled hash、report、trace 和分析结果，不能复用旧 trace 作为新实现证据。

### 先量化真实 workload 的容量等待

改代码前，对 `bash examples/run.sh list` 的 Runnable workloads 全集运行 runtime/full_memory 两档，采集 memory trace。按 request 的 `context_admission_wait`、wait_reason 变化和 context_admitted 划出 WAIT_CAPACITY/WAIT_FRAGMENTATION 区间，报告等待请求数、总 root-wait cycles、等待区间并集占 makespan 比例，另列 WAIT_SLOT/WAIT_CONTROL_RESOURCE。若容量等待为0，记录该 workload 预期没有 R3-6 直接收益；仍完成已指定的实现，不把这一步变成再次请求优先级决定。用户报告的旧 l2-admission-wait 失败直接作为已知基线，不为确认而重跑。

### 回归测试与边界

- **私有提前归还：**改写 `tests/test_runtime.py::TestRootArenaAdmission.test_t09_t10_t14_pending_root_owns_no_slot_or_arena_until_prior_root_retires`，新名 `test_pending_root_admits_on_input_extent_release`，参数化 runtime/full_memory。断言 B 的 port `active_cycle == a_input 的 l2_extent_release cycle`，小于 A context completion，`active_peak == 2`；B 等待期间不持 slot/arena；B 的真实 prefetch 与 A EVU:pow 区间交集大于0。device `admission_cycle` 不代替 port `active_cycle`。
- **striping/forfeiture：**在 `test_profiles.py`/`test_memory_invariants.py` 保留 L1 invalidate 不改 free-map 测试，新增 L2 非整 stripe 尾部 padding 一起归还、slack 留到 root retire、A 释放后 B 写新数据不被 A 回绑、退休/reset 不 double-free。使用 free-map 和读回字节验证，不只检查私有字段。
- **pending-reader 保活：**新增 `tests/test_l2_sharing.py`，独立组织跨 context 新合同（现 shared_A fixture 只覆盖单 context，不能替代）。seed W=`bytes(range(256))*32`；B/C 各4 tiles 输出应等于四份 W。延迟 C 或约束 slot 让其晚准入；B release 与 producer retirement 后 C 仍读到原字节，只有 C release 才归还 backing。断言 shared prefetch W=8192 B、private 对照=16384 B，shared physical capacity 不按 borrower 数增加。
- **fanout：**A 的真实 tile.store 完成后 publish X；B/C 输出逐字节正确；没有 X 的 GLOBAL_STORE/PREFETCH transaction。只观察模拟器支持的 copy bytes，不把 timing-only 的 pow/EVU 当作数值计算 oracle。
- **只读与静态图：**拒绝 unpublished/private export、未知 producer slot、错 shape/dtype、shared formal 作为写目的、release 后使用、重复 publish/release、重复同次借用、跨 L2 epoch。有效的无消费者 publish/release、同一模板多次 submit 不串 generation。通过构造/篡改已编译 DTO 再 load 证明不能绕过 source verifier，不只测 parser。
- **atomicity/late access：**复用 `test_profile_runtime.py` 的 ByteStore+真实 transaction 模式，覆盖 accepted-but-not-issued prefetch、cancel-requested leg、FAULTED 未 ack、最后一次 destination write、不同 issuer 的同 backing 访问；preflight 失败不归还容量、不修改 pins/refs，旧 generation late completion 不覆盖新 owner 字节。
- **profile 对照：**在 `test_profile_runtime.py` 运行新 profile-switch IR 两档；断言 compiler 自动生成 A completion await，真实 HBM store 完成后才允许 profile command，命令完成前 B 无 prefetch，最终 mode1、无旧 epoch views。另测 L1-only switch 不清理共享 L2，以及未来 reader 跨 L2 switch 在 compile/load 拒绝而非跑到 timeout。
- **容量分类/FIFO：**空池根本不够→PERMANENT_CAPACITY；有释放可恢复但当前不足→WAIT_CAPACITY；空闲总量足够但连续 aligned span 不足→WAIT_FRAGMENTATION；同类队头不被越过，shared acquire 或非最后 release 不触发无意义容量 wakeup。所有成功/取消案例结束后 refs、pins、inflight、backings、reserved bytes、routes/leases 清零。
- **泄漏封闭：**verifier 拒绝缺失/重复的 import `nest.release`；公共第二次 release 是 fault，但两次内部 cancel 清理不重复释放。成功的 B/C 与 zero-consumer export 验 `assert_l2_closed`。注入未提交 C 的 fault/cancel 或 C 仍 WAIT_CAPACITY 时的 reset，确认 `DECLARED` claim 经隔离后转 `CANCELLED`；注入终态漏 claim，验证 `controller._enter_fault` 后 Group 真的推进到 `ResetDomain.DONE`，才返回 `completed=False` 和具体 ID。用低 `max_cycles` 触发独立 post-cap drain loop；隔离无法完成时检查 `poisoned_reason`、旧 backing 未被 free、新 run/reset 被拒绝，证明不靠单纯零计数伪造安全。两次成功 run 不得串用上一 run 的 W/X。

分阶段执行与最后完整回归：

```bash
conda run -n elenor-validator python -m pytest \
  pipeline_validator/tests/test_profiles.py \
  pipeline_validator/tests/test_memory_invariants.py \
  pipeline_validator/tests/test_runtime.py -q

conda run -n elenor-validator python -m pytest \
  pipeline_validator/tests/test_l2_sharing.py \
  pipeline_validator/tests/test_compiler.py \
  pipeline_validator/tests/test_profile_runtime.py \
  pipeline_validator/tests/test_trace.py -q

conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -v
```

### CLI 端到端执行与 Perfetto 证明

实施者创建一个 throwaway Python runner（放本次 artifacts 目录，不新增维护负担），用 subprocess 逐项执行下列矩阵，先建输出目录；每条保存退出码/stdout/stderr：

```text
name = l2-admission-wait
       l2-admission-profile-switch
       l2-shared-weight
       l2-shared-fanout
fidelity = runtime, full_memory

bash examples/run.sh <name> --sim-override fidelity=<fidelity> \
  --memory-trace --trace-json <out>/<name>.json \
  --report <out>/<name>.report.json --json
```

`<out>` 固定为本次 run-id 下的 fidelity 子目录。随后枚举当前 `bash examples/run.sh list` 的所有 workload、protocol 和 NEST 名称，执行相同双档矩阵并刷新本次全部 trace；不要硬编码“136 场景”，不要把 `.vscode/run_all_workload.sh` 误当完整 corpus。已知期望失败的输入仅按其现有测试合同归类，不能为通过率改成成功；其它任何失败都逐项处理。

使用已存在的 `examples/artifacts/tools/trace_processor_shell`，通过 Python `subprocess.run([tool, trace_path, "-q", "/dev/stdin"], input=sql, text=True, ...)` 执行 SQL。若该工具缺失，只在项目 artifacts/tools 下恢复；不在项目外安装。报告至少包含：

- a_input `l2_extent_release`、B `context_admitted`、B `dma.prefetch` 的开始/完成、A `EVU:pow` 区间与交集；profile 对照中的 A HBM store、A context completion、L2 profile command 完成、B first load。
- `backing_id` 级分配/发布/逻辑 release/final-free 时间、W/X prefetch 次数与字节、共享容量峰值、每 bank 的容量守恒；物理生命周期可以跨 origin arena retirement。
- 每份 trace 的 Perfetto `stats` 中 `severity in ('error','data_loss')` 非零行，raw JSON 与 SQL 的 X/instant/counter/flow 完整性比对；独立 JSON 提取器和 SQL 的上述关键指标一致。不能静默忽略 parser_failure。

可复用的 SQL 形态：

```sql
SELECT name, ts, dur,
       extract_arg(arg_set_id, 'args.backing_id') AS backing_id,
       extract_arg(arg_set_id, 'args.context') AS context_name,
       extract_arg(arg_set_id, 'args.buffer_id') AS buffer_id
FROM slice
WHERE name IN ('l2_extent_release', 'context_admitted', 'profile_command')
   OR extract_arg(arg_set_id, 'args.summary_kind') = 'group_transfer'
   OR name GLOB 'dma.prefetch:*'
   OR name GLOB 'dma.store:*'
   OR name = 'EVU:pow'
ORDER BY ts;

SELECT name, value FROM stats
WHERE value != 0 AND severity IN ('error', 'data_loss');
```

本计划的配置保持1 GHz：Perfetto 的 ts/dur 数值可直接按 cycles 解读；raw JSON 的 ts/dur 则按 `cycle_ns()` 还原。若执行时修改时钟，统一用 `cycles = Perfetto_ns / cycle_ns()`，不套用固定1000倍。引擎程序用 args 区分 A/B，完整时间重叠取区间交集，不只比较最早/最晚一条；使用已有 transfer summary 的 transaction_id/op/source/destination，避免把多条 transfer legs 加总成多次 prefetch。

## 关键锚点

以下位置是容易漏改的非局部合同；行号仅供定位，实施前重读：

- `pipeline_validator/compiler/resources.py:308–391`：旧 L2 barrier 后 offset reuse；提前归还时必须移除，不能仅改 allocator。
- `pipeline_validator/workload_ir.py:619–716`：当前 out/inout 强制 HBM store，以及 release 仅接收 nest.alloc；共享 fanout 和 imported release 的关键阻点。
- `pipeline_validator/tile_group.py:1299–1319`：context_done 前的 root retirement；不得等待其它 context 的共享借用结束。
- `pipeline_validator/runtime/group_port.py:333–355,398–409`：version-gated 分类 FIFO 与 post-step poll，保证 release 当拍 admission 的既有路径。
- `pipeline_validator/compiler/profile_pass.py:503–538,606–617`：普通 await proof + 全 root history frontier，不能降级为 buffer release。

## 实施依赖与验收边界

- 编译布局禁止回绑可先独立落地；私有物理归还必须和 transfer reference hooks、记账、private cancel/reset 同一变更闭合后才能启用。不得先交付“能 overlap 但仍可能 late write”的中间状态。
- 共享源 IR/DTO/codec/verifier 的实现可与 backing 生命周期实现并行；共同字段和签名按本计划固定。接入 `tile_group.py`、编译器整条 pass 与 profile frontier 由同一集成人串行完成，不能让两人同时修改这些共享边界。
- profile 对照 fixture 和新共享 fixture 在语法合同固定后可独立编写；实际双档验证等集成完成后统一运行。
- 第5项完成条件是跨 context 单份物理 backing、单次 weight prefetch、fanout 无中间 HBM 往返、最后 reader 安全归还；仅保留现“同 context 同 handle”不算完成。
- 第6项完成条件是同 profile 当拍提前准入和真实 load/pow overlap，加上不同 profile 完整 store→retire→switch→load 的反向约束；只改计数器或只缩短 context 生命周期不算完成。

## 假设与预定处理

- 首版受控跨 context 传递采用 producer context completion 作为消费起点，而非其内部 publish phase 提前唤醒另一个 context；这仍消除 HBM 中转并支持 B/C 分支。若需求将来要求 producer 尚在其它计算时就启动读者，需要另行扩跨层 phase-event 合同，不在本次悄悄加入。
- 不保持旧 compiled artifact 的二进制兼容；source 默认 private 语法保持，旧 artifacts 重编译。遇到依赖旧 L2 offset reuse 的 source，按明确的 no-reuse resource summary 迁移，不用兼容开关。
- 不跨 model run/Tile Group/L2 profile generation 共享，不提供隐式 spill、自动重加载或任意 shared read-write。跨边界输入明确拒绝；L1-only mode change 沿用现有独立 domain。
- 若真实 workload 的 capacity wait 很低，最终报告只声称容量阻塞案例与重复 weight IO 的实测收益；不承诺 matmul-pow-free-slot 或 matmul17-pow-tail-overlap 恢复引擎 overlap，它们的既有 device/data/HBM 瓶颈不在本改动里。
