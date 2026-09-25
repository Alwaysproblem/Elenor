# 批次 II：显式只读 L2 共享与生命周期

> 本文是执行计划，不是已实现功能说明。唯一的行为合同来源是[总计划](../L2_显式共享_PLAN.md)；本批次只拆解其中第三、四步、第五步与共享有关的第四至第九项、第六步第一至第三项、第八步第一至第二及第五项，以及验证方案第 252–258 行。实现不得越过这些边界来缩小合同。
>
> - 前置批次：[批次 I：私有 L2 提前释放与物理 extent 生命周期](01_private_l2_release.md)
> - 下游场景：[批次 III：场景与端到端验收](03_scenarios_and_acceptance.md)
> - `plan/README.md` 由主任务维护；本文不改写该索引。

## 1. 范围、前置条件与批次门

### 1.1 前置条件

开始共享实现前，批次 I 必须已整体通过：现有单一 `ArenaPool`/free-map 以物理 backing 为唯一容量单位；padding、非重叠 extent、无 rebind 和私有提前释放合同成立；所有已接受 L2 transfer 的引用在安全终态确认后才归还；private cancel/reset 能安全 drain，物理占用/extent trace 可审计。共享只复用这些 backing、释放与 transfer 生命周期接口，不得另建 allocator、free-map 或独立可变 refcount。批次 I 有缺口时停止批次 II，不用共享代码绕开缺口。

首次代码改动前的 workload 容量等待基线属于联合计划的前置门：须已按总计划第 240–246 行采集 runtime/full_memory 两档真实 workload 结果及输入哈希。不得在本文填入或推测未采集的数值；基线缺失时先完成该前置项。

### 1.2 共同冻结合同

- `L2` 的默认语义保持 `private`；唯一新增语义是显式 `readonly` 导出/导入。共享是同一 physical backing 上的不同逻辑 owner/view，不是多个物理 allocation，也不是裸地址或全局按地址去重。
- 每个 L2 buffer 继续使用完整 padded、互不重叠、不可 rebind 的 backing。单一 free-map 仍是唯一物理占用真相；容量按 backing 去重。
- claim 集合是有限、按 run 封闭的 manifest。引用数由 producer 所有权和 claim 状态推导，不能再维护一个可独立增减的整数 refcount。
- reader 只能在 producer context completion 依赖已经满足、publish 已完成且 backing 仍属于同一 run/group/L2 profile epoch 后读取。submit lowering 自动并入 producer-completion dependency；不得只依赖作者手写 `depends_on` 或在运行时增加无界 readiness 等待队列。
- producer 在 publish 前是唯一 writer；publish 成功即不可变。producer 的 `nest.release` 只撤销 producer view，不取消未提交、`WAIT_DEPS` 或 `WAIT_CAPACITY` reader 的 claim。
- transfer 引用覆盖所有真实 L2 source/destination endpoint，直到实际 terminal acknowledgement；idle、cancel 请求、`input_released`、context completion 均不等同于安全 ack。
- L2 profile epoch 必须闭合：同一 backing 的 producer 与所有声明 reader 在该 L2 epoch 内；不同 L2 profile 的转换等待完整 root completion frontier。只切 L1 不要求清空 L2 backing。

### 1.3 本批次不接管的实现

批次 II 不重新实现批次 I 的私有 extent 归还、L1 生命周期或单一 allocator；不编写批次 III 的 `examples/scenarios/`、配置文件、`examples/run.sh` 入口及全量 CLI/Perfetto 场景矩阵。批次 II 必须提供可供批次 III 直接使用的 parser/compiler/loader/runtime 合同和测试；不得因场景文件尚未合并而把此处的单元/运行时边界测试省略。跨批次共用接口只在本文指定一份实现归属，不由下游重复实现。

## 2. 拟议 IR 与静态合同

以下是新增 API 的示意语法，**当前仓库不可解析、不可运行**：截至本计划编写时，`pipeline_validator/dialects/elenor.py` 尚无 `NestPublishOp`/`NexusSharedRefOp`，`NestAllocOp` 也无 `sharing` 属性；model/context actuals 与 formals 仍以 global memref 为准。所有后续步骤均为提案，不能将下列片段描述为可执行 fixture。

```mlir
// producer context：默认仍为 private；此 allocation 明确导出只读 backing。
%w = nest.alloc slot = "W" role = "in" sharing = "readonly"
  shape = [64, 64] dtype = "bf16" alignment = 256 : !nest.l2_buffer<64x64xbf16>
%published = nest.publish %w depends_on(%prefetched) : !nest.event<"published">
nest.release %w depends_on(%prefetched, %published)

// model：shared.ref 锚定本次 producer submit，而不是 context 模板名。
%producer_done = nexus.submit_context.async @loader(%W) : !nexus.event<"producer_done">
%shared_w = nexus.shared.ref %producer_done slot = "W" : !nest.l2_buffer<64x64xbf16>
%reader_done = nexus.submit_context.async @reader(%OUT, %shared_w)
  depends_on(%producer_done) : !nexus.event<"reader_done">
```

按以下静态规则冻结并在 source verifier 与独立 executable verifier 中各自证明：

1. `sharing` 只接受 `private` 或 `readonly`，省略等价于 `private`。context formals 按现有 Tile Program 的约定排列：零个或多个 `NestGlobalMemref`，然后零个或多个 `NestBuffer`；后者只代表只读导入。`NexusSubmitContextOp.actuals` 的类别、shape 和 dtype 必须逐项严格匹配，不可仅以 byte count 相等放行。共享 formal 名必须非空、唯一，且不冲突于本地 allocation slot；`nexus.program` 的外部输入仍只允许 HBM global memref，tile program 可以通过自身 L2 formal 读取导入视图。
2. `nexus.shared.ref` 必须引用可识别的 producer submit 实例与该实例的 readonly export slot。共享引用不产生独立执行指令；消费者 submit 的 dependency 集是显式依赖与 producer-completion dependency 的稳定去重并集。source 级 HBM hazard verifier 和 lowering 使用同一 dependency 集，不得发生“lowering 会补边、source verifier 却先拒绝合法 IR”的分歧。
3. 按实际 `bindings/ins/outs`、transfer view、dispatch effect 和 tile program 描述推导读写闭包：共享 formal 可作只读 `ins`、L2→L1 load 的源及 L2→HBM store 的源；不得是 prefetch/tile.store 目的、dispatch `outs`、新 publish 的目标、可写别名或二次 export。一次 submit 不得把同一 backing 映射为两个 formals；不同 submit 的多个 readers 合法。
4. readonly local allocation 必须恰好 publish 一次；private allocation 不可 publish。publish 直接 dependency 必须恰好闭合该 buffer 所有 prefetch/store completion、reader `input_released` 与 writer `output_ready` 事件，并稳定去重。验证 buffer 至少经过完整 prefetch 或实际 writer 完整初始化；复用现有 execution-range/phase/full-view 覆盖证明，拒绝仅部分初始化、缺 writer completion 或依赖遗漏的发布。
5. publish 是 producer 最后一次数据访问的边界：发布后 producer 不得继续读、写或 store，只可 release。writer access 只能在 publish 前完成。release 事件合同不引入隐式作者负担：private release 集为 `R_input_released ∪ P_prefetch ∪ S_store`；导出 readonly release 集再含 publish event；导入 readonly release 集为 `R_input_released ∪ S_store`。集合内禁止重复事件。已 publish 的 readonly `out/inout` 可直接交给 reader，不强制为共享中间结果先做 HBM store；private `out/inout` 的旧 HBM-store 合同不变。
6. 静态 closure 同时覆盖 import 的一次 release、读写权限、producer/consumer identity、profile epoch、事件 ancestry 和 complete initialization；release 后的 use、duplicate publish/release、未 publish export、unknown slot/producer、shape/dtype mismatch、跨 L2 epoch reference 均为 compile/load 硬拒绝，而不是容量等待或运行超时。

## 3. 有序实现步骤与文件/API 归属

### 步骤 A：扩展 source IR，并固定编译期证明

**文件：**`pipeline_validator/dialects/elenor.py`、`pipeline_validator/workload_ir.py`。

1. 在 `NestAllocOp` 增加默认 `private` 的 sharing 属性；新增 `NestPublishOp` 与 `NexusSharedRefOp` 并注册到 dialect。复用已有 shape/string/event/operand-group parse-print helpers，不新增另一种 tensor type。
2. 扩展 context block-arg、submit actual 验证：仅允许“globals 在前、readonly L2 imports 在后”；按完整维度和 dtype 精确核对；输入 formal/slot 唯一。`nexus.program` 的 HBM 外部输入不得直接替换成 L2 formal；导入只出现在 context actual/formal 与它向 tile program 传递的链上。
3. 扩展 `_verify_release_graph`、`_verify_device_memory_dependencies` 及相邻 source verifiers：从实际 IR descriptors 重算 read/write/event closures、producer completion ancestry、publish 依赖与完整初始化覆盖；按上一节的精确 release event 集检查。所有失败应在 source parse/verify 阶段给出具体合同错误。
4. 同一 model context 模板的多次 submit 必须 specialize 为不同 producer/consumer binding identity；静态引用解析不许退化成按符号名或 HBM 地址的动态查找。

**门：**parser/print 往返与 source verifier 正反例覆盖上述每条规则；source verifier 测试能区分合法只读 load/store source 与非法 destination/二次 export；不触及 runtime 才失败。

### 步骤 B：加入冻结 DTO、lowering actions 与 independent verifier

**文件：**`pipeline_validator/execution_ir.py`、`pipeline_validator/compiler/lowering.py`、`pipeline_validator/execution_verifier.py`、`pipeline_validator/tile_group_sequencer.py`。

1. 增加冻结 `ExecSharedInput(slot, dims, dtype, element_bytes, bytes, producer_binding_id, producer_slot)`；在 `ExecTileGroupTask` 加 `shared_inputs: tuple[...] = ()`，在 `ExecL2Buffer` 加默认 `sharing="private"`。`task.l2_buffers`、`layout`、`resource_contract.l2_spm_bytes` 只包含本 context 新分配，不把 imports 当作 local layout index 或再次收费。
2. 增加 `ExecGroupActionOp.PUBLISH_L2 = "publish.l2"`、`BIND_L2_IMPORT = "bind.l2.import"` 和冻结 `ExecPublishRequest(buffer_slot, reader_dispatch_ordinals, writer_dispatch_ordinals, dependency_events)`。`lower_model_ir/_lower_context` 依实际 submit callsite specialization 将 shared ref 解析为 producer binding ID + slot，写入 consumer `task.shared_inputs`；不把同一映射重复存到 device op、request 和 task。HBM actual/global input index 过滤 imports 后仍保持原有 dense global 索引。
3. lowering objects table 将每个 imported formal 映射为 `role="in"` 的 L2 buffer view descriptor，但不加入 `task.l2_buffers`。每个导入 formal 在 task actions 开头形成一个同步 `BIND_L2_IMPORT`（`args=(slot, shared_input_index)`，内部 dst bind event）。成功 admission 已创建 borrower handle；action 只验证绑定，不再次 acquire/增加 claim。dispatch/release 仍依赖绑定完成。更新 `TileGroupSequencer.issue_registered`、L2 issue gate、action completion 与 independent verifier 的 bound/released 集合为 local allocations ∪ imports；layout 始终只解释 local allocations。
4. 将 `PUBLISH_L2` 接进同一 ready-action/event-table 路径并发出完成事件。`normalize_action_dependencies` 与 event finalization 必须使 publish 等待完整 writer/read completion，publish completion 又是 release 的硬依赖；沿用事件 frontier/reuse 机制，不跨 producer/consumer 事件 namespace 传 `NestEvent`。
5. independent `execution_verifier.py` 从 serialized descriptors 重算 claims 所需的 dependency、只读闭包、producer/export/consumer 关系、full initialization、发布先后与 epoch；不可信任 DTO 预存 dependencies、manifest、权限字符串或已验证的 source summary。

**门：**执行 DTO 的边界非法值被拒；合法 producer/两 reader task 的 imports 不污染本地 layout/预算；篡改执行 DTO 的读写/依赖/shape 不会由 compiler 生成成功。

### 步骤 C：闭合 codec、ABI、relocation 与 load-time 独立验证

**文件：**`pipeline_validator/compiler/api.py::_relocations`、`pipeline_validator/compiled_program.py::Relocation`/strict codec、`pipeline_validator/runtime/relocation.py::relocate_task`、`pipeline_validator/loader.py`、`pipeline_validator/execution_verifier.py`。

1. 注册新增 DTO、动作 enum、sharing enum、shared-input/publish fields 的严格 codec；增加 `publish_events` relocation，并能重定位 `ExecPublishRequest.dependency_events`。`dst`、action dependencies 继续使用通用 event relocation。更新 `_expected_relocations` 与 relocation verifier，保证 relocation 集严格相等且无重复/遗漏。
2. 编译产物 `schema_version` 从当前 1 升为 2、`compiler_abi` 从当前 `v0` 升为 `v1`。旧 schema/ABI 产物明确拒绝并要求 source 重编译；不加运行期兼容 shim，也不把硬件 YAML 的 schema 2 与 compiled artifact version 混为一谈。
3. compile 与 load 都独立验证 producer 确实存在且被 publish、actual consumer callsites 完整、shared shape/dtype/权限、dependency closure、事件 relocation 和 L2 epoch；loader 不信任源文本声明或 artifact 中声称的 producer/claim map。DTO tampering 测试需在重建/重封合法 artifact hash 后仍因语义验证失败，不能只证明 hash mismatch 会失败。
4. 因 ABI 升级，除 `IR_SPEC.md` 外，还要同步当前明确记载 `schema_version=1`/`compiler_abi=v0` 的 `pipeline_validator/README.md` 与 `pipeline_validator/Limitation.md`，仅改受影响的 compiled-artifact 版本说明；硬件 YAML schema 保持独立。

**门：**新 artifact compile→dump→parse→load 成功；旧 schema/ABI 被拒；独立 load verifier 对 producer slot、shape、权限、dependency、epoch 等语义篡改逐一拒绝；所有 relocations 在重复 launch specialization 中绑定到正确 producer 实例，不串事件。

### 步骤 D：建立封闭 manifest、claim ledger 与原子 admission

**文件：**`pipeline_validator/tile_group.py::TileGroup.begin_launch/try_admit_context_task/context_cleanup_ready/retire_context_arena`、`pipeline_validator/memory/arena.py::ArenaPool`、`pipeline_validator/runtime/group_port.py::GroupPortAdapter._admit_pending`。

1. `begin_launch` 从已验证、已 specialized 的所有实际 consumer submit callsites 建立 run-scoped manifest。Backing key 为 `(run_generation, producer_binding_id, producer_slot)`；claim key 为 `(consumer_binding_id, local_slot)`。仅构造 dormant metadata，不预留 bytes、不阻塞 entry prefix/profile 初始化。未被任何实际 submit actual 使用的 `nexus.shared.ref` 不建立 claim；zero-consumer export 合法。开始新 run 前检查旧 backing/claims 已完全终结。
2. producer backing commit 时才按 manifest 建立唯一 `_L2ClaimRecord` 对象：reader 初态 `DECLARED`。consumer successful admission 原子地执行 `DECLARED → BOUND`；成功 release 后变 `RELEASED`；仅 fault/reset 的安全 cleanup 可将未绑定或已隔离的活 claim 置为 `CANCELLED`。ledger 保存同一 record 至 run 终止；backing 只引用这些 record，不复制状态。终态保留供重复内部 cleanup 检查，成功闭合或安全 drain 后才清空 manifest。
3. 所有 consumer submit callsite 的 claims 从 launch 起就存在，因此未提交、`WAIT_DEPS`、`WAIT_CAPACITY` 均不能被当成已释放。正常 model return 时尚未提交的 declared claim 是 invariant fault；fault/cancel 才能在隔离和 drain 后安全取消。拒绝 duplicate consumption、跨 generation/run 的旧 ID 复用、未 publish backing 和 wrong owner。
4. producer 一次 admission 只创建一个 backing；publish 前只有 producer 可写，publish 后不可变。producer release 撤销自己的逻辑 view但不丢弃任何 reader claim。B release、producer retirement、C 尚未提交/尚在等待容量都不能让 C 的 backing 被 free。
5. admission 分两段：先做完全只读的纯 plan，同时检查 private arena、所有 imports 的 published/readiness、run/group/L2 epoch、完整 shape、claim、event/slot budgets 和 per-bank 容量；任一失败不扣容量、不改 claim、不产生 view。全部检查成功后再原子提交 private arena 与 borrower views、claim 状态和 handle。缺失/unpublished/错误 identity/已消费 claim 是合同 fault，不是 `WAIT_CAPACITY`；producer readiness 靠 completion dependency，不另开共享等待队列。
6. 提供 `ArenaPool.borrow_l2_view(backing_id, owner, claim_id, cycle) -> AllocationHandle`，只在 admission commit 调用；它不改 free-map 或 `pool_version`。producer handle 与每个 borrower handle 有不同 `allocation_id` 和 `ContextBufferOwner`，但 `backing_id`、物理 `generation/profile_generation` 与地址必须相同；borrower 的 launch generation 留在其 owner 上。borrower handle 归 consumer 逻辑 arena 与 local slot，仍接入现有 `_l2_handles`、role binding、tile dispatch；backing 单独保存 origin_arena_id。import view 可登记在 consumer 的零容量逻辑 arena 中，root retire 检查本 context 所有 views，但物理容量只按 backing 遍历；alias overlap 仅容许同一个 published readonly backing，不放宽跨 backing 的 overlap 检查。
7. `context_cleanup_ready`/`retire_context_arena` 只等待本 context 的 views、routes、leases、jobs、pins/transactions；不能要求 exported backing 已物理释放，否则 producer 和 reader 会互等。只有整个 model terminal closure 要求所有 backing、claim、view、transfer 清零。

**门：**W 的延迟 C 测试证明 producer retirement 与 B release 后 C 仍能读原字节，且只有 C release 后一条物理 free；private arena 与多个 borrower alias 的容量统计无重复；admission 任一失败的 pool、claims、handles 和 version 不变。

### 步骤 E：复用批次 I 的 transaction hooks，补齐共享权限与 alias 检查

**文件：**`pipeline_validator/memory/transfer.py::TransferManager`、`pipeline_validator/tile_group.py::validate_transaction_generation/release_context_memory`、`pipeline_validator/memory/arena.py::ArenaPool`、`pipeline_validator/engines.py`、`pipeline_validator/memory/profile_controller.py`、相关 transfer/engine tests。

1. 沿用批次 I 已接通的 `reference_acquire/reference_release` 与 `acknowledge(transaction_id, cycle)`、`acknowledge_all_terminals(cycle)` API；不再实现第二套登记/释放机制。为 producer/borrower 的所有去重 L2 src/dst views 增加同一 backing 上的 alias preflight 与物理引用登记：view 记录 transaction ID，backing 记录 `(allocation_id, transaction_id)`，使 view release 只检查自己而 final-free 检查全部别名。所有 accepted L2 endpoints 都在传输 issue 前登记；无 L2 端点的 cache/HBM 路径仍归 profile drain，不因 release flush 全 cache。
2. 逐一审计并补充 Global DMA prefetch/store、MFE load/store、Gather L2/L1 refill 与最终 destination write、cache dirty clean/writeback、MSHR waiter、maintenance transfer、NoC cancel drain；只有真实持有 L2 `AllocationHandle` 的路径进入 L2 ledger，跨 issuer 的同 backing 访问也必须受保护。拒绝/登记失败保持批次 I 的完整 rollback 和 acquire 成功标志，不能在 ack 时减不存在的引用；DONE/CANCELLED/FAULTED 已接受事务均持有到字节及 owner 后处理完成的 terminal acknowledgement。核对 `engines.py`、`tile_group.py`、`memory/profile_controller.py`、transfer 内部与测试无漏改调用点。
3. 在 `ArenaPool` 提供 `assert_access(handle, permission)`、`permissions(handle)`，permission 仅 `r/w`；private/未 publish producer 是 `rw`，published producer 与 borrower 是 `r`，stale/invalidated handle 始终拒绝。`_resolve_view`/tile formal materialization 的 descriptor permissions 从 pool 派生，不能作为权威。
4. 在 transfer generation validation 的 source 和 destination 分支检查当前 pool permission；对 destination bytes 真正 commit 时再查 authoritative pool state，防止 descriptor 缓存值绕过 publish/release。`INVALIDATE_PENDING` 只禁止新 pin/transaction，已 accepted transaction 仍可安全 drain。publish preflight 必须等 writer pins/access 完成后才封存 backing 并发完成事件；此后 source IR 不可开始新的 producer access。producer release 只检查自身 logical view 的 pins；其它 borrower 的合法读 pin不应阻止 producer release，但 final-free 必须检查所有 aliases。
5. 沿用批次 I 的异常清理顺序：先禁止新 publish/borrow，按现有 cancellation/isolation 路径退休 terminal owner jobs（含 `MFEEngine.retire_isolated`），确认不能再写后 `acknowledge_all_terminals(cycle)`，再 invalidate views、撤销 claim/ref、retire arenas。不能先删除 transfer record 再让存活 engine job 二次 ack，也不能先 invalidate 再等待 ack 造成互锁。未发布 producer fault 不得令 reader 成功；仍走 device ERROR 传播。

**门：**ByteStore 与真实 transaction 测试覆盖 accepted-but-not-issued prefetch、cancel-requested leg、FAULTED 未 ack、最终 destination write、跨 issuer 同 backing access；失败 preflight 不改 refs/pins/capacity；旧 generation late completion 不能覆写新 owner bytes；publish 后写被 source/runtime 双重拒绝。

### 步骤 F：公共 release、取消/reset 与 controller-visible leak drain

**文件：**`pipeline_validator/tile_group.py`、`pipeline_validator/simulator.py::Simulator._run_model`、`pipeline_validator/device.py::CpuDeviceController`、`pipeline_validator/runtime/reset_domain.py`。

1. 公共 `nest.release` 先做完整只读 preflight，再恰好执行一次 `BOUND→RELEASED` 或 `producer_live=True→False`。同一 owner 第二次 public release 必须报 `double release` invariant fault；wrong owner、stale generation、只读写入也是硬错误。失败 preflight 不改变 roles、handles、claims、pins/inflight、capacity、`pool_version` 或 trace。仅内部 fault/reset cleanup 可重复处理已 `CANCELLED/RELEASED` claim 且不改物理状态。
2. backing 仅在 producer 已 release、无 `DECLARED/BOUND` claim、所有 alias views 已释放且 pins/accepted transaction refs 均为零后 final-free 一次。条件由 owner/claim/view 状态推导，不增设自由浮动 refcount。终态记录保留到 run closure/drain 完成；final-free 后旧 backing ID 不可复活。
3. 新增 `TileGroup.assert_l2_closed()` 检查 live backings、`DECLARED/BOUND` claims、borrowed views、pins/inflight、arena slack 与 free-map 守恒。`Simulator._run_model` 当前 `controller.succeeded` 会直接 break，故成功出口先执行 closure assertion。断言失败时调用 `CpuDeviceController._enter_fault(reason, cycle)` 令 fault 对 controller 可见，记录 drain start、调用 `_ensure_fault_drain` 并 continue；后续沿 fault 分支持续 Group step 与 completion harvest，只有 `ResetDomain.DONE` 才返回 `completed=False`。standalone 完成出口同样遵循该合同。leak guard 仅探测泄漏，不替代合法 release。
4. `max_cycles` 到达时不能直接返回 incomplete，也不能仅调用 `_ensure_fault_drain`。保留已有 terminal leak reason；否则以 cycle-cap reason 进入 controller fault，再执行独立有界 post-cap drain（上限 `hw.memory_target.profile_command_timeout_cycles`），逐拍按 controller step → Group step → harvest 推进，故障态不接收新 submit；只有 `ResetDomain.DONE` 才报告 drained failure。超时/隔离失败设置 `TileGroup.poisoned_reason`，列出残留 backing/claim/transaction IDs，保留物理占用并拒绝 `begin_launch`、`try_admit_context_task` 和不安全 reset。不得清空 registry/free-map 伪造成功；显式 reset 仅在确认安全 drain 后恢复。
5. fault/reset 先禁止新 publish/borrow，再取消未准入、未 submit claims；active borrower 按既有顺序关闭 routes/jobs、pins/transactions 并释放 view/ref。未发布 producer 出错不得令 reader 成功，仍走 device ERROR 传播。新的 run generation 不得复用旧 W/X。

**门：**覆盖缺失/重复 import `nest.release` 的 verifier 拒绝、公共 double release fault 与内部重复 cancel、成功 B/C 与 zero-consumer export closure。注入未提交 C 的 fault/cancel、C `WAIT_CAPACITY` 时 reset、终态漏 claim：验证 claim 仅在 drain 后 `CANCELLED`；terminal leak 进入 controller-visible fault 并真实推进到 `ResetDomain.DONE`，再返回 `completed=False` 与具体 ID。低 `max_cycles` 覆盖独立 post-cap drain；隔离失败验证 poison 保留 backing 并禁止复用。

### 步骤 G：保留 backing 下的逐 bank 容量证明与 FIFO 分类

**文件：**`pipeline_validator/compiler/resources.py`、`pipeline_validator/execution_verifier.py`、`pipeline_validator/memory/arena.py`、`pipeline_validator/tile_group.py`、`pipeline_validator/runtime/group_port.py`。

1. compile/load 对每个允许的 L2 profile 验证 consumer 去重后的 shared padded spans + 自有 arena 可按每 bank 容量合同容纳；不能由 consumer private bytes 为零推断总需求为零。多个 reader alias 同一物理 span 只计一次，padding/stripe 沿用真实 layout 规则。
2. 增加纯查询 `ArenaPool.can_fit_with_retained(layout: ArenaLayout, backing_ids: tuple[str, ...]) -> bool`：从 pristine per-bank free intervals 扣除指定 live backing 的 padded segments，再复用 `_first_fit` 判断完整 arena；不得改 free-map、`pool_version`、trace 或 reservation。
3. 队头失败时只保留“该队头肯定不能先释放”的 backing：未释放 claim 属于队头、同类 FIFO 中排在它之后的 request，或其 submit completion-dependency 后继。依赖后继在 `begin_launch` 从已编译 device schedule 一次建立；同类阻挡顺序取当前 FIFO，不重新扫描 source IR。关联 claim 存在即保留该 backing。
4. 若扣除必留 backing、假定其它 reservation 全部释放的最乐观 free map 仍不能容纳队头，则使用 `PERMANENT_CAPACITY`，报告 `L2 capacity fault: retained shared extents prevent FIFO-head admission`。否则使用原 `WAIT_CAPACITY`/`WAIT_FRAGMENTATION` 与 version-gated FIFO retry；不因暂时没有 active context 判永久、不越过队头、不 spill。只在 pending/admission token 变化时重算；shared acquire 与非最后 borrower release 不触发无意义容量 wakeup，真实 final-free 沿用批次 I 通知。

**门：**空池 per-bank 根本装不下→`PERMANENT_CAPACITY`；释放可恢复→`WAIT_CAPACITY`；总空闲够但 aligned contiguous span 不够→`WAIT_FRAGMENTATION`。覆盖 own-shared 碎片、队尾 reader 持有 backing 的 FIFO 互锁和同类队列不越过。正常成功/取消核对 claims、pins、inflight、backings、reserved bytes、routes/leases；poison 路径保留物理占用及残留 IDs，不强行清零。

### 步骤 H：将共享约束并入 L2 profile quiescence

**文件：**`pipeline_validator/compiler/profile_pass.py::bind_profiles`、`pipeline_validator/memory/arena.py::ArenaPool.reconfigure`、`pipeline_validator/memory/profile_controller.py::_closure_ledger/_level_quiescent/_verify_member_ready`、`pipeline_validator/tile_group.py::begin_launch/reset`、source/execution verifiers。

1. 异 L2 mode 切换沿用 `bind_profiles` 的完整 root completion history，生成普通 device await 与 `ProfileReconfigDesc`；publish、`input_released`、view invalidate、extent free 都不是切档 frontier。producer 和所有 declared readers 必须已在同一 epoch submit 并在完整 frontier 内退休。未来 epoch reader 在 source compile 与 compiled load 阶段拒绝，不能靠 runtime drain 等尚未 submit 的 reader。
2. `ArenaPool.reconfigure` 与 profile closure/quiescence/member-ready 检查 live backing、alias pins/inflight、已物化 pending/active claims；`live_arenas==0` 或 origin `context_done` 不足以准许 L2 reconfigure。live backing 不因新 profile 可容纳而迁移。现有 CLOSE_ISSUE→drain→Prepare/Commit ACK→OPEN_ISSUE 次序不变。
3. dormant manifest 不计 live/quiescence 资源，避免未来 epoch metadata 阻塞 entry prefix；L1-only closure 不清理、不等待 shared L2 backing。`begin_launch/reset` 的 previous-launch-retired 检查扩到独立 backing/claim ledger，origin arena retired 但仍有 reader 时不得开新 run。
4. 本批次实现 profile closure 与 compile/load epoch rules；profile-switch YAML/场景、CLI 和双档 Perfetto 对照属于[批次 III](03_scenarios_and_acceptance.md)，不重复创建 fixture。批次 II 仍用 runtime/profile tests 直接验证 live L2 backing 阻止 L2 重配、L1-only switch 不清理 backing、跨 L2 epoch consumer 被拒。

**门：**profile-controller/runtime 具备上述直接边界证据；批次 III 的完整 profile-switch IR/trace 尚未执行前，不声称端到端 profile 场景已通过。

### 步骤 I：共享记账、资源报告与规格文档

**文件：**`pipeline_validator/memory/arena.py::ArenaPool.snapshot`、`pipeline_validator/trace.py::MemoryTrace._pool_snapshot`、`pipeline_validator/report.py`、`pipeline_validator/IR_SPEC.md`，以及 ABI 说明涉及的 `pipeline_validator/README.md`、`pipeline_validator/Limitation.md`。

1. `ArenaPool.snapshot` 的 per-bank reserved/allocated/free/padding 按 live physical backings + 实际 arena slack 记账；aliases 不重复物理容量。每 bank 恒满足 `allocated_bytes + free_bytes == user_spm_per_bank`，system-reserved 与 cache 单列。保留逻辑 `live_views`/`pin_count`，新增 `live_backings`、`pending_shared_claims`、`active_shared_references`；pending 只计已有 backing 的未准入 `DECLARED` claims，不计 dormant manifest。report zero-leak 同时检查这些量，而不只检查 `live_arenas/live_allocations`。
2. arena rows 的 `reserved_bytes` 表示当前持有的物理容量，并新增 `initial_reserved_bytes` 记录 admission 合同。origin retire 后仍存活的 backing 独立报告，不伪装成活 root；`live_view_bytes`/protocol live bytes 按 backing 去重；producer view invalidation 后 pending reader 持有的 published data 不归零。trace counter 来自真实 post-mutation pool snapshot。
3. backing final-free 和 extent event 实现/trace 归批次 I；本批次只复用其物理终点，将共享 views/claims/counts 加入 snapshot/report，不重建 allocator 或重复定义私有 extent 事件。
4. 更新 `pipeline_validator/IR_SPEC.md` §3.8、§7–§9：local reserve 与 readonly shared reference、publish/immutability、producer release forfeiture、publish 替代中间结果 HBM store、同 run/group/L2 epoch 和禁止跨 L2 profile。`l2_admission_wait.mlir` 中“runtime 不会 wait”的陈旧注释由批次 III 在场景验收前清理；本批不抢改批次 III 拥有的 fixture。同步 README/Limitation 中受 schema/ABI 升级影响的 compiled-artifact 版本说明，硬件 YAML schema 独立不变。

**门：**每 bank 容量守恒；多 alias 只计一份物理 reserved bytes；origin arena retired 而 reader claim 尚在时 report 仍显示 backing/pending；最终 report 与 terminal closure 一致。规格文档不得把提案语法描述成当前 parser 已支持。

## 4. 必须新增或扩充的可观察测试

新增 `pipeline_validator/tests/test_l2_sharing.py`，独立覆盖跨 context 合同；现有 shared_A fixture 只覆盖单 context，不能替代。测试真实行为而非 private-field wiring、mock echo、bare not-throw 或 trace 行数变化。

| 合同                                 | 测试位置/输入                                                                             | 必须观察到的结果                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------ | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pending-reader 保活、bytes 与容量    | `tests/test_l2_sharing.py`；W seed `bytes(range(256))*32`，B/C 各 4 tiles                 | B/C 输出逐字节等于四份 W。延迟 C 首次 read/准入；producer retirement 与 B release 后 C 仍读原 W，只有 C release 后出现唯一 physical free。共享 W prefetch 为 8192 B，private 双 reader 对照为 16384 B；physical capacity 不随 borrower 数增长。                                                                                                                       |
| fanout 正确性、无 HBM 中转           | `tests/test_l2_sharing.py`；A 真实 tile.load→tile.store 产生 X                            | B/C 输出逐字节正确；X 没有 `GLOBAL_STORE/PREFETCH` transaction。只以模拟器实际支持的 copy bytes 作数值 oracle，不用 timing-only pow/EVU。                                                                                                                                                                                                                             |
| source 静态只读与 closure            | `tests/test_l2_sharing.py`、`tests/test_compiler.py`                                      | 拒绝 private/unpublished export、unknown producer/slot、错 shape/dtype、shared formal 作写目的/`outs`/二次 export、release 后 use、duplicate publish/release、同次 submit 重复导入、缺/重复 import release、跨 L2 epoch；接受合法 zero-consumer publish/release；同模板重复 submit 不串 generation。                                                                  |
| codec 与独立 load verifier           | `tests/test_compiler.py`                                                                  | 旧 schema 1/ABI v0 明确拒绝，新 schema 2/ABI v1 compile/dump/parse/load 成功。分别篡改 producer binding/slot、shape、permissions、dependencies、epoch、claim/relocation；重封 artifact hash 后仍由 load semantic verifier 拒绝，不能只触发 hash mismatch。                                                                                                            |
| accepted transfer、取消与 late write | `tests/test_profile_runtime.py`、`tests/test_l2_sharing.py`，ByteStore + 真实 transaction | 覆盖 accepted-but-not-issued prefetch、cancel-requested leg、FAULTED 未 ack、最后 destination write、不同 issuer 同 backing access；acquire 与 destination commit 各检查权限；失败 preflight 不改容量/ref/pin；accepted leg 持有到安全 terminal ack；旧 generation late completion 不能覆写新 owner bytes。                                                           |
| public release、内部 cancel          | `tests/test_l2_sharing.py`、`tests/test_memory_invariants.py`                             | 公共第二次 release 是可观察 fault，且失败前后状态不变；内部重复 cancel 幂等；producer release/非最后 B release 不物理 free；错 owner/stale/write-after-publish 拒绝；最后合法 borrower release 唯一 final-free。                                                                                                                                                      |
| profile epoch/quiescence             | `tests/test_profile_runtime.py`、`tests/test_profiles.py`                                 | live backing/alias pin/inflight/materialized claim 阻止异 L2 reconfigure；完整 root frontier 前不切档；未来 epoch reader compile/load 直接拒绝而非 timeout；L1-only switch 不清理 shared L2。完整 profile-switch fixture/CLI/trace 由批次 III 负责。                                                                                                                  |
| capacity 分类/FIFO                   | `tests/test_l2_sharing.py`、`tests/test_memory_invariants.py`、`tests/test_profiles.py`   | 不可能→`PERMANENT_CAPACITY`；释放可恢复→`WAIT_CAPACITY`；总空闲够但 aligned span 不足→`WAIT_FRAGMENTATION`；retained-set 与 FIFO/dependency successor 相符，不越过同类队头；acquire/非最后 release 不做无意义 wakeup。                                                                                                                                                |
| terminal leak、fault/reset、run 隔离 | `tests/test_l2_sharing.py`、`tests/test_runtime.py`                                       | `assert_l2_closed` 通过成功 B/C 和 zero-consumer export；正常成功无 DECLARED/BOUND claim。未提交 C fault/cancel 或 WAIT_CAPACITY reset 只在 drain 后 CANCELLED；terminal leak 先 fault、真实 drain 到 `ResetDomain.DONE`，再返回 `completed=False` 和具体 ID。cycle cap 用独立 post-cap drain；隔离失败 poison、保留 backing 并拒绝复用；两次成功 run 的 W/X 不串代。 |

正常成功与取消核对 claims、pins、inflight、backings、reserved bytes、routes/leases terminal 状态；预期 poison 的失败核对物理占用和残留 ID，不为 zero-leak 清空状态。

## 5. 执行顺序、产物与向批次 III 移交

按 A→B→C→D→E→F→G→H→I 执行：source verifier/DTO/load 合同稳定后才启用 runtime admission；transfer ref 与 fault drain 未闭合前禁止发布可运行共享 fixture。每个步骤的门全部通过后才整体交付批次 II，不交付“可 parse 但未验证”“能共享但未 drain”或需下一批次补安全性的半成品。

实施证据存入新目录 `examples/artifacts/l2-sharing-release/<run-id>/`：source/config SHA-256、命令与 exit/stdout/stderr、compiled hash、report、memory trace 与分析结果。不得复用旧 trace，不得预填未实测周期、字节流量或性能结论。

批次 II 定向回归：

```bash
conda run -n elenor-validator python -m pytest \
  pipeline_validator/tests/test_l2_sharing.py \
  pipeline_validator/tests/test_compiler.py \
  pipeline_validator/tests/test_profile_runtime.py \
  pipeline_validator/tests/test_profiles.py \
  pipeline_validator/tests/test_memory_invariants.py \
  pipeline_validator/tests/test_runtime.py \
  pipeline_validator/tests/test_trace.py -q
```

完整闭合证据包括 compile/source verification、dump/parse/load semantic tamper 拒绝、ByteStore/真实 transaction 读写、delayed-C 物理生命周期、profile/capacity 分类及成功/故障/reset closure。批次 III 只消费这些接口，增加可运行 shared weight/fanout、profile switch、双档 CLI/Perfetto 和 corpus 回归，不再实现另一套 claim ledger、permission hooks、profile gate 或 allocator。移交时明确：本文拟议 IR 仍需由批次 III 的完整场景文件验证可运行；没有端到端 trace 前只能报告批次 II 测试证据，不能声称场景验收完成。

完整项目回归 `conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -v` 在批次 III 整合后统一执行；开发中途的定向回归不能替代最终全量验收。
