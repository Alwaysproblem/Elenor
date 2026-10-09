# PagedAttention：Tile 级真实地址 Gather/Scatter 与 Host 页池执行计划

## Context

在 pipeline_validator 中扩展**现有 Tile 级 Gather**，并新增 **Tile 级 Scatter**，使 PagedAttention 的动态页池、多请求、多步 decode 能以真实地址、真实字节和真实 Cache 行为进行调度模拟。nest/group 层不新增 Gather/Scatter；group 仍只做连续 DMA、dispatch、release 等编排。允许改变旧 Gather（删除人工指定 hit/miss 的 profiled 语义），不增加 `paged_kv` 或 Attention 专用 IR；BOA/EVU 仍只计时序，不做 Attention 数值计算。

## 已核实的现状（只读调研）

- 旧 `tile.gather.global.async`（`dialects/elenor.py::TileGatherOp`）：source 是 Tile Program 的 global formal（经 `nest.dispatch.tasks.async ... globals(...)` 传入），indices/destination 是 L1 alloc；但 `engines.py::launch_gather/_resolve_byte_sources` **从不读取 indices 字节**，每个请求的 hit/miss 由 `tile.profiled.access` 的 outcome 决定。
- MFE 已有完整的 L1/L2 Cache + MSHR 机制可复用：`DeterministicLRUCache.contains/record_hit/record_miss/refill/read_line`、`MshrTable.allocate(merge_group)/wait/complete/cancel`、`_try_l1_mshr/_try_l2_mshr`、`_tick_gather_materialization`（按 ordinal 顺序写 L1 目的地）、`retire_isolated`、TransferOp `GATHER_HBM_REFILL/GATHER_L2_REFILL/GATHER_DIRECT_L1_REFILL/GATHER_DIRECT_RESPONSE/GATHER_DEST_WRITE`。Cache 身份 `CacheLineIdentity(allocation_id, allocation_generation, line_offset, ...)` 已是 allocation 限定。
- Cache 合同检查 `compiler/resources.py::_cache_path_requirements/check_program_capabilities` 与 `execution_verifier.py::_cache_path_requirements/_verify_cache_contract/_verify_parent_l2_capability` 目前按 authored outcome 推导；`compiler/profile_pass.py::insert_task_maintenance/_device_maintenance` 已能在 HBM 写与带 Cache 读之间插入 clean/invalidate。
- 当前唯一 dirty line 生产者是测试用 `ByteStore.seed_cache_line(..., dirty=True)`；运行路径不会产生 dirty line。
- `CpuDeviceController._advance_program` 只支持 submit/await/profile_reconfig/memory_maintenance/return，且 device 级 memory_maintenance 是阻塞 CPU PC 的 fence；context 内 `MEMORY_MAINTENANCE` 只阻塞该 context 后续 action 注册（`TileGroupSequencer.note_registered`）。
- `ProfileController._clean_transaction` 是 src=None、dst=HBM、inline captured_data 的计费写事务先例；`ByteStore` 读取未初始化字节会报错，不零填。
- 只读 smoke：Pow 经 compile→serialize/parse→load→run 完成，9090 cycles；内存中普通 DMA copy 以 ByteStore 搬运 `<3i>(3,0,3)` 正确，68 cycles。均为旧路径事实，不是新功能验证。

## 已冻结的设计决策

1. Gather/Scatter 只存在于 Tile Program 内，remote 端必须是 Tile 的 global formal，local 端是 L1 alloc；indices 是 L1 中的真实 `i32` 字节。group 层不新增任何 Gather/Scatter action。
2. **Gather 由真实地址查 Cache**：按 cache line 查 L1→L2，miss 走 MSHR 合并与 HBM 回填；对应 level 的 Cache 关闭（profile cache_bytes=0）时走 SPM 直通。hit/miss/merge 全由运行时地址与 Cache 状态决定，`tile.profiled.access` 删除。
3. **Scatter 绕过 Cache**：L1→HBM 写，不分配 Cache line、不产生 dirty line。后续带 Cache 读的可见性由编译期插入的 maintenance 保证，maintenance 的范围在运行时精确解析为 Scatter 实际写过的字节。
4. 动态页池由 Host/runtime 管理：新增通用 `nexus.host.call.async` 软件节点、HostRuntimeSession 与 PagePoolRegistry；页表是普通内存数据。PagedAttention 的 K/V/长度语义只存在于 examples。
5. artifact 干净切换到 `schema_version=3`/`compiler_abi="v3"`；旧 artifact 要求重编译，不写升级 shim。

## Approach

按 1→6 顺序实施。步骤 1–2 完成后，迁移后的旧 Gather 示例即可在 v3 下运行；步骤 3–4 加入 Host 页池；步骤 5–6 构造场景与对照。每步完成后已有无关测试应保持通过（Gather 相关的旧断言按步骤 2 迁移/删除）。

### 1. Tile Gather/Scatter 的 IR、DTO 与地址映射

**通用 indexed map**（`dialects/elenor.py` 新增 `TileIndexedMapAttr`，打印为 `#tile.indexed_map<index_scale=S, offset=O, task_stride=P, repeat=R, stride=T, segment=L>`）。全部单位为 remote dtype 的**元素**；S/R/L>0，O/P/T≥0。对第 i 个 index（L1 中 little-endian signed i32）、第 j 个重复段（0≤j<R）、logical task id t：

```text
remote_element = index[i] * S + O + t * P + j * T
local_element  = (i * R + j) * L
copy_elements  = L
```

local 为该 task 自己的 L1 alloc，元素总数必须恰为 `I*R*L`（I 为 indices alloc 元素数，I>0）。负 index、uint64 溢出、任一段越出 remote view 分别报 `indexed_address_out_of_bounds`/`indexed_address_overflow`；禁止 Python 负索引回绕。

**语法**（替换旧 Gather，新增 Scatter；`_TileAsyncOp` 子类，复用现有 event/operand-group 解析）：

```mlir
%g = tile.gather.global.async %pool indices(%idx_l1) into %k_l1
    map = #tile.indexed_map<index_scale = 8224, offset = 0, task_stride = 1024,
                            repeat = 1, stride = 0, segment = 1024>
    window_entries = 4 scope = "owner_0" : !tile.event<"k_gathered">
%s = tile.scatter.global.async %k_new_l1 indices(%append_idx_l1) into %pool
    map = #tile.indexed_map<index_scale = 8224, offset = 960, task_stride = 1024,
                            repeat = 1, stride = 0, segment = 64>
    window_entries = 4 scope = "owner_0" : !tile.event<"k_scattered">
```

- `TileGatherOp(source, indices, destination, tag, *, address_map, window_entries=4, scope=None)`；新增 `TileScatterOp(source, indices, destination, tag, *, address_map, window_entries=4, scope=None)`（source 为 L1、destination 为 global formal）。删除 `result_bytes/cache_target_bytes/l1_mshr_hint` 与 profile region；`TileProfiledAccessOp` 从 dialect、`__all__`、`pipeline_validator/__init__.py` 删除。`scope` 为可选非空字符串（通用托管内存 owner，见步骤 3）。
- verifier（`workload_ir.py`）：remote 必须是当前 tile.program 的 global formal；indices 为 live `[I]xi32` L1 alloc；local 为 live L1 alloc、dtype 与 remote 相同；indices 与 local 不同 alloc；window_entries>0。Scatter 的 remote formal 必须在 dispatch 的 global 写权限集合中。L1 侧顺序：indices 与 Scatter source 的每个先前异步写（load destination、Gather destination）必须已 `tile.await`，Gather destination 的每个先前异步访问也必须已 await。现有 verifier 的 await 检查只出现在 `tile.free`/`tile.signal`（`workload_ir.py` 的 `l1_access_events`），lowering 也不插隐式 wait；缺这条时 INDEX_READ/L1_READ 可能读到未写入或旧的字节。IR_SPEC §4.2.1 的 free 前 await 集合同步加入 Scatter 的 indices/source。
- **task 间不冲突**：一个 dispatch 有多于一个 task 时，Scatter 必须满足 record-domain 条件：所有 `(t, j)` 段 `[O+t*P+j*T, +L)` 两两不重叠且全部位于 `[0, S)`；否则报 `tile_scatter_task_overlap`。
- **同一 Tile Program 内**：两条 Scatter、或 Scatter 与 Gather 作用于同一 global formal，若两者之间没有 `tile.await` 前者事件，则必须满足相同 S、各自段集合都在 `[0,S)` 内且区间不相交（record-domain 证明），否则报 `tile_indexed_hazard`。一个 dispatch 的程序若既 Scatter 又 Gather 同一 formal（任何 task 组合），报 `tile_scatter_gather_same_dispatch`。
- frozen DTO（`execution_ir.py`）：`ExecIndexedMap(index_scale:int, offset:int, task_stride:int, repeat:int, stride:int, segment:int)`；`ExecTileGatherDesc(source:ExecMemoryView, indices:ExecMemoryView, destination:ExecMemoryView, address_map:ExecIndexedMap, window_entries:int, scope:str|None)`；`ExecTileScatterDesc` 同字段（source 为 L1 view，destination 为 global view）。删除 `ExecGatherDesc/ExecProfiledAccess/ExecGatherOutcome`。`ExecTileOp` 保留 `LAUNCH_GATHER` 并新增 `LAUNCH_SCATTER="launch.scatter"`；`ExecEngineDesc(kind="MFE", op="gather"|"scatter", params={"gather"|"scatter": desc})`。
- lowering（`compiler/lowering.py::_lower_engine_descriptor/_lower_program`）、codec（`compiled_program.py` `_CLASSES/_ENUMS`、`_validate_engine`）、load 验证（`execution_verifier.py::_verify_gather` 改为验证新 desc，新增 `_verify_scatter`，并重算上述 task/程序内冲突规则）同步。所有版本守卫改为 schema 3/ABI v3（`compiler/api.py::compile_program` 构造、`compiled_program.py` parse 前置检查与 `_validate_compiled_program_value`、`execution_verifier.verify_compiled_program`）。
- Tile global formal 与 `globals(...)` 语法、`ExecTileRoleBinding.global_actuals`、`tile.py::_resolve_tile_view` 的 global 解析全部保留，这是 Tile 访问 HBM 的唯一通道。
- **可见点锚定 grid 事件**：Scatter 的写 effect 与现有 Gather 的读 effect 一样锚定该 dispatch 的 grid 事件（compiler 端 `lowering.py::normalize_action_dependencies` 对 global 访问记录 `original.dst`，load 端 DISPATCH_ROLE 分支用 `grid_done`），不用 `output_ready`——它只对应真实 L2 写方向（IR_SPEC §5.4）。纯 Scatter 程序没有 L2 写：`outs()` 为空，`output_ready` 省略（空 tag）。`tile.return` 会等待全部 LOCAL 事件（`tile.py::_complete_context`），再加上 §2 对 Scatter 事件的定义，grid_done 就意味着该 dispatch 的全部 Scatter 段已提交到 HBM。
- **global effect 枚举点，每处都必须加 Scatter 写分支**：下列位置目前只认 Gather（写死 `params.get("gather")` 或 `TileGatherOp`），且一律按读处理。Scatter 的 global 端是 `destination`（`source` 是 L1），照抄 Gather 分支取 `.source` 会拿错 view。每处都带上 scope，按 §2“scope 不相交证明”的同一规则：
  1. `workload_ir.py::_context_global_accesses`（source 跨 root 检查，:1346-1362）：加 `TileScatterOp.destination` 对应 actual 的写。
  2. `compiler/lowering.py::normalize_action_dependencies` 的 DISPATCH_ROLE 分支（同 context 自动加 RAW/WAR/WAW 边，:636-639）：加 `params["scatter"].destination` 对应 actual 的写。
  3. `compiler/profile_pass.py::_action_accesses`（maintenance，:125-128）：见 §2。
  4. `execution_verifier.py`（load 端重验）：构建 `_ProgramEffects` 处（:497-544）现在对任何 tile descriptor 写 global 直接 `_fail("tile descriptor writes a global formal")`（:542-543），且只有 `global_reads`。新增 `global_writes`，只放行 Scatter 的 destination，其余 descriptor 写 global 仍然 fail；`_binding_effects`（:856-860）返回读、写两组；DISPATCH_ROLE 分支（:1448-1454）为写追加 `writing=True`、锚 `grid_done`、带 scope 与 precise 标记的 `_Access`，由 `_verify_global_hazards`（:1756）重验边和 maintenance 覆盖。
  5. 运行时 `tile.py`：`_LAUNCH_OPS` 加 `LAUNCH_SCATTER`，`_descriptor_l1_bases`（:844-864）加 Scatter 的 source/indices。否则 `_assert_free_dependencies_complete` 会跳过 Scatter launch，`_assert_launch_l1_live` 也会把它当 opaque descriptor，不做 use-after-free 检查。

  这些遗漏都是静默的：缺 1，跨 root 的 Scatter→Gather 不写依赖也能编译通过；缺 2，同 context 内 Scatter→Gather 不会自动加边，后面的 Gather 可能在 Scatter 提交前回填旧 HBM 字节；4 如果只删掉 :542-543 的 fail 而不填 `global_writes`，load 端也查不出缺边。IR_SPEC §5.4 的 signal 规则只管 L2，source 的 hazard 规则只管跨 root，没有其他检查兜底。缓存需求三处（`compiler/resources.py:226`、`execution_verifier.py:656/675`）保持只认 Gather，Scatter 不进入。

**冲突域总览**（对照 `scatter.md` §2.1：单 context 内的 `unique` 挡不住跨 context 的非原子覆盖，所以每一层都要有可验证的机制）：

| 范围                                    | 机制                                                                                                                             | 违反时                                      |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 同一条 Scatter 内重复/重叠段            | 运行时按 `(i,j)` 顺序后者覆盖（§2）                                                                                              | —                                           |
| 同一 dispatch 的不同 task               | 静态 record-domain 证明                                                                                                          | `tile_scatter_task_overlap`                 |
| 同一 Tile Program 内无 await 的两条访问 | 静态 record-domain 证明                                                                                                          | `tile_indexed_hazard`                       |
| 同一 context 的不同 dispatch            | `normalize_action_dependencies` 按整 view 推导依赖并自动加边                                                                     | 串行执行                                    |
| 不同 root                               | 现有 source 规则要求显式依赖（`test_ready_action.py::test_cross_context_hbm_hazard_requires_an_explicit_completion_dependency`） | 编译拒绝                                    |
| 不同 scope（上两行的唯一放宽）          | 静态视为不相交；运行时逐段校验落在该 scope 当前页内，root pin 期间不得改页                                                       | `memory_scope_violation` / `host_pool_busy` |

没有哪一层依赖未经验证的不相交声明，因此不引入运行时 destination claim/lease（`scatter.md` §4.1、§6）。scope 不相交由页所有权逐段校验，正好是其 §6.4 方案 A 所要求的“可验证 disjointness certificate”。

### 2. MFE 地址驱动执行：Gather 走 Cache，Scatter 绕过 Cache

在 `memory/address_provider.py` 新增纯函数层 `resolve_indexed_segments(map: ExecIndexedMap, index: int, ordinal: int, task_id: int, remote: ResolvedMemoryView, local: ResolvedMemoryView, element_bytes: int) -> tuple[tuple[ResolvedMemoryView, ResolvedMemoryView], ...]`，返回 (remote_slice, local_slice) 对，内部用 `slice_resolved_view` 并做边界/溢出检查；Gather/Scatter 共用，无其他求址实现。

**索引读取**：新增 `TransferOp.INDEX_READ="index_read"`，route 仅 `L1_READ`（src=indices L1 4 B 切片，dst=None），在 `TransferManager.submit` 单端点白名单中与 cache-clean 并列；L1_READ 完成时由 ByteStore capture，MFE 用 `captured_data()` 解码后 ack。每个 index 一次 INDEX_READ，计入 L1 bank 竞争。没有 ByteStore 时无法得到 index 值：含新 Gather/Scatter 的程序在 compile/load 拒绝非 full_memory（`indexed memory requires full_memory fidelity`），run 时无 ByteStore 在 issue 前失败（`indexed memory requires ByteStore input data`）。

**Gather FSM**（重写 `engines.py` 的 `_MFEGatherRequest/_MFEGatherJob/launch_gather/_tick_gather_request`，保留 materialization/retire 结构）：

- 每个 job 最多 `window_entries` 个 index slot 在飞；slot 从 INDEX_READ 发出到该 index 全部请求写入 destination 为止。index 解码后由 `resolve_indexed_segments` 得段；当 L1 或 L2 任一 Cache 启用时，段按 cache line（`cfg.cache_line_bytes`）切成请求（首尾可为不满行的行内部分）；两级都关闭时每段一个直通请求。请求 ordinal 全局递增，destination 写仍按 ordinal 顺序（沿用 `_tick_gather_materialization` 与 `gather_reorder_wait_cycles`）。
- 行身份：`CacheLineIdentity(remote.handle.allocation_id, remote.handle.generation, line_offset)`，provenance 为精确一行（`precise=True`）；MSHR merge_group 字符串为 `f"{allocation_id}:{generation}:{line_offset}"`。不再使用 opaque token。
- 查找时序：L1 启用时提交新 `TransferOp.GATHER_L1_LOOKUP`（route `L1_CACHE_LOOKUP`）；完成时 `l1_cache.contains(identity)` → 命中：`record_hit` 读行数据、标 response ready；未命中：`record_miss`，L2 启用则提交 `GATHER_L2_LOOKUP`（route `L2_CACHE_LOOKUP`），否则进入 L1 MSHR + `GATHER_DIRECT_L1_REFILL`。L2 查找完成时命中 → L1 启用走已有 `GATHER_L2_REFILL`（L1 MSHR 合并），L1 关闭走新 `GATHER_L2_RESPONSE`（route `NOC_RESPONSE→LOCAL_DMA`）；L2 未命中 → 已有 `_try_l1_mshr/_try_l2_mshr` + `GATHER_HBM_REFILL` 路径。L1 未启用时从 L2 查找开始。判定在查找腿完成时刻做；删除 `GATHER_L1_HIT/GATHER_L2_HIT/GATHER_MISS_LOOKUP`。
- 回填整行：行必须完全落在 remote 的 resolved view 内，否则该请求走 `GATHER_DIRECT_RESPONSE` 并计 `gather_cache_bypass_requests`；若对应 Tile/Context cache requirement 为 `bypass="forbidden"`，运行时 fault `gather_cache_bypass_forbidden`。ByteStore 模式下整行在 HBM_READ 完成时 capture **数据与逐字节 validity**（只有整行回填腿用新增的非抛错读取，其余腿仍走严格 `read_view`），validity 随行进入 L2/L1 cache；未初始化字节只在进入 `GATHER_DEST_WRITE` 的段时按现有 uninitialised 规则报错。原因：`transfer.py::_capture_source` 现用严格 `read_view`，整行回填会因同一行内与请求无关的未初始化字节（页尾 token、padding）fault，与 §6“尾部不 seed”及验证 7 微例（D=2 时一个 K 行仅 4 B，一条 64 B 行横跨多个 token 与 padding）矛盾。这对应 `scatter.md` §3.3 的 `partial_line` 契约：部分有效的行是常态，不是错误。
- 删除 `byte_store.bind_profiled_source/profiled_source_offset/resolve_profiled_source` 及 `_resolve_byte_sources` 的 profiled 分支。PMU 保留并改为真实统计：`gather_requests/gather_l1_hits/gather_l2_hits/gather_hbm_misses/gather_mshr_merges/gather_mshr_stalls/gather_cache_bypass_requests/gather_index_reads/gather_bytes`。

**Scatter FSM**（MFE 新增 `_MFEScatterJob`，复用 MFE_GATHER 同一 job 容量 `mfe_load_channels*mfe_pipeline_depth`，tile 队列新增 `"MFE_SCATTER"`、深度 `mfe_store_queue_depth`，`tile.py::_launch_queue_key/_drain_engine_queues/_build_engine_launch` 增加分支）：

- 同样按 window 读 index、求段；每段一个 `TransferOp.SCATTER_WRITE="scatter_write"`，route `L1_READ→LOCAL_DMA(store)→NOC_REQUEST→GLOBAL_DMA→HBM_WRITE`（L1_READ capture、HBM_WRITE commit）。不查 Cache、不 refill。
- 重复/重叠目的按 `(i,j)` 字典序**后者覆盖**，不是 reduce：后继段与任何更早未完成写区间相交时等前驱 DONE+ack；不相交段并行。Scatter 的 remote 写范围与本指令 indices 所在 L1 无关；与同 Tile 其他在飞 Gather 的冲突已由步骤 1 静态规则排除。
- 每个 DONE 的 Scatter 子事务把 `(allocation_id, generation, offset, bytes)` 记入 `TileGroup.precise_write_ledger`，键为 `(role_event_id)`（dispatch 的 grid 事件 runtime id，取自 `_UCEContext.role_event_id`）与 root request id（新增 `TileGroupSequencer.root_request_id`，由 `GroupPortAdapter` 在接受请求时设置；standalone 为 None）。ledger 为 run 级，`begin_launch` 清空。
- 首个 index 非法时不发任何 payload；晚项失败不回滚已提交字节，但停止新 issue、隔离在飞子事务、不发 success。子事务 DONE/FAULTED/CANCELLED 均及时 ack；`retire_isolated/begin_reset_drain/cancel_unissued` 覆盖 Scatter job 与 index slot。
- **可见点**：Scatter job 的 tile 事件只在全部段 DONE（HBM_WRITE 已 commit）且 ack 后触发；不存在“已发出/已接受”即完成的中间态（`scatter.md` §4.5 不变量 8；`design/elenor_mfe/ELENOR_MFE_Design.md` §9 风险“async store visibility 过早 DONE”）。PMU 新增 `scatter_overlap_wait_cycles`（有序覆盖造成的串行等待）与 `scatter_commit_latency_cycles`（段 issue→HBM commit，即可见点代价），让保护机制的开销可测（`scatter.md` §8.3：协议代价没建模就无法评估）。
- **partial burst**：段不足 `hbm_burst_bytes` 时，计时按现有 burst 向上取整（`transfer.py` 的 `burst_rounded`），字节按掩码精确提交（masked write，对应 `scatter.md` 的 `partial_line = masked_write`），不建模 RMW，所以 §1 的不相交证明按字节粒度成立。若 HBM 规格冻结为无字节掩码（ECC RMW），证明粒度须改为 burst，由后续规格冻结。
- **fault 记录**：first-fault-wins（与 `design/elenor_mfe` §4.3 的 page fault 口径一致）；trace 列出已提交段的 ordinal。窗口内靠后的 index 可能先提交，所以已提交集合不一定是前缀。

**Cache 合同与可见性**：

- `_cache_path_requirements`（compiler 与 verifier 两份）改为：程序含 Gather 时，某 level profile `cache_bytes>0` → `used=True, required_needed=False`；`cache_bytes==0` → `bypass_needed=True`。Scatter 不使用 Cache、也不触发 bypass 需求。`l1_mshr_hint` 容量检查删除（MSHR 满时运行时等待，已有 `WAIT_L*_MSHR`）。
- `profile_pass._action_accesses` 的 DISPATCH_ROLE 分支：Gather → 对 actual view 的读，levels 为该 Tile/Context allowed profiles 中任一 cache_bytes>0 的 level；Scatter → 对 actual view 的写，并带 `precise_writes=True`。`_GlobalAccess` 增加 `scope: str|None` 与 `precise_writes: bool`；`MaintenanceRange` 增加 `precise_writes: bool = False`（codec/验证同步）。
- 生成 postwrite invalidate 时，若与读重叠的所有先前写都来自 Tile Scatter，则 range 标 `precise_writes=True`。`ProfileController._resolve_ranges` 对这种 range 用命令 dependencies 查 ledger：group 级命令的 dependencies 是 dispatch grid 事件 runtime id；device 级命令通过修改后的 `DevicePort.note_dependencies(events, request_ids, cycle)`（CpuDeviceController 从 `_event_handles` 传 request id，GroupPortAdapter 转给 ProfileController 存入 receipt）查 root request id。结果与 range 的 `[offset, offset+bytes)` 求交；为空即无需失效。非 Scatter 写（DMA store、HostWrite）仍用整段保守范围。
- `insert_task_maintenance` 给被同 task 内后续 postwrite invalidate 覆盖的写标注 `post_invalidated_levels`；`_device_maintenance` 跳过 `post_invalidated_levels ⊇ 读 levels` 的先前写，避免在 context 内已失效后再插阻塞 CPU 的 device fence。`execution_verifier._maintenance_covers/_verify_global_hazards` 用同一规则重验。
- scope 不相交证明：两访问映射到同一 actual binding、都带 scope 且 scope 不同 → 视为不重叠（hazard 与 maintenance 都跳过）；同 scope 或任一无 scope → 整 view 保守。source verifier、`normalize_action_dependencies`、`profile_pass`、load verifier 四处同一规则。v3 `static_effects["accesses"]` 条目新增 `scope`、`precise_writes`，load 端重算比对。

### 3. 旧 Gather 示例与测试迁移

- 检索迁移点（ruff LSP 不提供 references，用 grep）：`TileProfiledAccessOp|tile\.profiled\.access|ExecGatherDesc|ExecProfiledAccess|ExecGatherOutcome|bind_profiled_source|resolve_profiled_source|profiled_source_offset|GATHER_L1_HIT|GATHER_L2_HIT|GATHER_MISS_LOOKUP|l1_mshr_hint|cache_target_bytes|deterministic_profiled_not_address_or_value_accurate`，范围 `pipeline_validator/` 与 `examples/`（排除 `examples/artifacts/`）。全部改写或删除，不留兼容分支。
- 5 个 fixture 保持路径与 Tile 级结构：`gather_profiled.mlir` 更名 `gather_indexed.mlir`（run.sh `gather` case 改指向它），`gather_matmul.mlir`、`gather_matmul_4tiles_2contexts.mlir`、`matmul_gather_add.mlir`、`matmul_gather_add_4tiles_2contexts.mlir` 原地改写：profile region 换为 `map = #tile.indexed_map<index_scale = 64, offset = 0, task_stride = 0, repeat = 1, stride = 0, segment = 16>`，16 个 index 对 i8 table 各取 16 B 写入 `[256]xi8` 目的地。
- 含 matmul 的 4 个 fixture 新增 global `acc_init`（与 output 同 shape），context 内连续 prefetch 到新 L2 buffer，Tile 在 store 前 `tile.load` 到 `matmul_dst`，消除 ByteStore 下对未初始化 L1 的 store；L2/L1 resource contract 按新增 buffer 重算，删除仅为 profiled outcome 存在的 `required = true` cache 声明（保留 `access = "read"` 与 target_bytes）。
- 新增 `examples/generators/generate_gather_inputs.py`（`--output-dir` 默认 `examples/workloads/gather_inputs`）：用 `load_workload_ir` 读取每个 fixture 的 entry 参数类型，为每个 fixture 写 `gather_inputs/<fixture_stem>/<binding>.bin`。indices 整个 binding 填 `[3,0,2,1]` 周期序列（prefetch 读整段，必须全部初始化）；table 只写前 4096 B（`offset % 251`）；lhs/rhs 用 `offset % 251`；acc_init 全零；output 不生成。run.sh 的 5 个 case 增加对应 `--input-data`。
- 测试：删除只断言 authored outcome/merge 语法或旧周期的用例；`test_validator.py` GATHER_IR、`test_runtime.py::make_gather_module`、`test_profile_runtime.py` 的 gather 片段改为新语法；cache seed/maintenance/profile 恢复类测试保留并改用真实地址 Gather。

### 4. Host Runtime、通用页池与 scope

不新增 PagedKV 类型/manager。新增 `runtime/host_session.py`、`memory/page_pool.py`；现有 `runtime/host_runtime.py`（package 固定开销模型）不改，仅复用 `hw.host_patch_cycles` 参数。

**nexus 软件节点**：

```mlir
%prepared = nexus.host.call.async "prepare_r0_s0"
    bindings(%APPEND_IDS) accesses = [{offset = 0, bytes = 4, mode = "write"}]
    scopes = ["owner_0"] depends_on(%prev) : !nexus.event<"prepared_r0_s0">
```

- `NexusHostCallOp(name, bindings, accesses, scopes, tag, depends_on=())`：bindings 与 accesses **按位置一一配对**，同一 SSA 可重复出现但其范围两两不重叠；bytes>0、范围在 formal 内、mode∈`read|write|readwrite`；0 个 binding 时 accesses 为空；scopes 去重非空。
- frozen `ExecHostAccess(input_index:int, offset:int, bytes:int, mode:str)`，input_index 始终是 nexus.program entry 的 global formal 序号；`ExecHostCall(name:str, accesses:tuple[ExecHostAccess,...], scopes:tuple[str,...])` 放入 `ExecDeviceOp.command`，op=`"host_call"`，沿用 event_tag/dependencies。v3 `static_effects` 新增 `host_accesses` 与 `scope_effects`（submit 对其 scopes 记 `use`，host_call 记 `mutate`；同 scope 的 mutate/use、mutate/mutate 必须有依赖），load 端重算。
- `Simulator.run(program, *, host: HostEnvironment | None = None)`；`HostEnvironment(handlers: Mapping[str, HostRoutineFactory], pools: tuple[HostPagePoolSpec, ...])`。`HostRoutineFactory = Callable[[DeviceHostRequest], Generator[HostCommand, HostResult, None]]`；命令 `HostRead(binding, offset, bytes, scope=None)`、`HostWrite(binding, offset, data, scope=None)`、`HostAllocPages(pool, scope, count)`、`HostFreePages(pool, scope)`、`HostDelay(cycles)`。读写必须被本调用声明范围与 mode 覆盖；托管 pool binding 的读写必须带本调用声明的 scope。缺 handler、未知 scope/binding 在 issue 前失败；含 host_call 而 host=None 报缺失。
- 验证 HostEnvironment 时，`static_effects` 中每个带 scope 的访问，其 actual binding 必须就是该 scope 所属 pool 的 binding，否则在 issue 前失败（`scope_binding_mismatch`）。这对应 `scatter.md` §4.1 对 context/dispatch/claim 三处 ID 一致的要求，避免错配拖到第一个段越界时才暴露。
- 每个 host_call 先计 `hw.host_patch_cycles`，之后每 cycle 轮转最多执行 `sim.device.issue_width` 条命令；一条 routine 同时只有一条未完成命令，结果在 harvest 后送回，下一条最早下 cycle。active routines 上限 `sim.device.pending_capacity`。
- `CpuDeviceController` 新增 keyword-only `host_port: HostPort | None = None`，host 请求与 submit 共用 request_id/event/future-ref/completion credit 命名空间，`_active_host` 独立、不占 `device_context_count`；记录加 `kind="group"|"host"`；成功条件含 `_active_host` 与未收割 host completion 为空。计数 `host_submitted/host_completed/host_failed` 单列，`report._request_timing_summary` 只统计 group。
- 时钟顺序固定：`controller.step → host.issue → group.step → host.harvest → controller.harvest_completions`。
- 运行中写入必须计费：`TransferOp.HOST_READ`（route `HBM_READ→GLOBAL_DMA→NOC_RESPONSE`，dst=None）与 `HOST_WRITE`（route `NOC_REQUEST→GLOBAL_DMA→HBM_WRITE`，src=None、inline captured_data），使用同一 TransferManager/ByteStore；运行中禁止在 callback 中直接 seed/write ByteStore。`ByteStore.seed_hbm` 只用于 cycle 0 之前。

**页池**：

- `HostPagePoolSpec(name, binding, page_bytes, page_count, initial_owners: Mapping[str, tuple[int, ...]])`：page_bytes 为正且 64 B 对齐、`page_count*page_bytes ≤ binding.size`、初始页号有效不重叠；scope 预先声明、一个 scope 只属一个 pool、pool 间 backing 不重叠；page id 可编码为 i32。
- `PagePoolRegistry`：`initialize(specs, bindings, hbm, byte_store, run_generation)`、`allocate(pool, scope, count, cycle) -> PageAllocation(scope, pages, epochs)`（最低空闲页号、原子、OOM 报 `host_pool_out_of_pages`）、`free(pool, scope, cycle)`、`pin_root(request_id, scope_bindings, cycle)`（原子，GroupPort 拒绝/背压时回滚）、`unpin_root(request_id, cycle)`、`lease(scope, binding, offset, bytes) -> AccessLease`、`abort_after_drain(cycle)`、`assert_closed()`、`snapshot()`。alloc/grow 与 free 都要求该 scope 无 pins/refs，否则立即 `host_pool_busy`；重复 free/释放后使用报 `host_scope_inactive`；新分配页调用新增 `ByteStore.invalidate_view(view)` 清有效位（不清数据、不计流量）。
- `invalidate_view` 同时清除各级 cache 中该范围副本的 validity（oracle-only：不改 tag/LRU，不计流量）。否则前 owner 的 Gather 留下的陈旧行仍带“有效”字节，新 owner 一旦越界读或漏了失效，就会静默拿到前 owner 的数据，§2 的 validity 检查对此无效（对应 `scatter.md` §5 资源表的“复用条件”列）。
- 带 scope 的 Tile Gather/Scatter：每个解析段必须完全落在该 scope 当前持有页内，否则 `memory_scope_violation`；每个子事务真正 submit 前 `lease.acquire(txn_id)`，terminal ack 后 release；`TileGroup.validate_transaction_generation` 在原检查前加 `lease.is_valid()`。带 scope 的指令只允许出现在 nexus.program 的 context 中；托管 pool binding 只能经带 scope 的 Tile Gather/Scatter 或带 scope 的 HostRead/Write 访问（普通 DMA 访问报错）。
- 初始化顺序：Profile 初始化 → `group.begin_launch`（HBM 注册、ByteStore seed 覆盖检查）→ 验证 HostEnvironment → 注册 pool（若任一 L1/L2 cache 行的 `binding_name` 属于该 pool，报 `managed_pool_has_cached_state`）→ 创建 session → CPU 开始 issue。

### 5. 调度集成、故障收敛与观测

- Tile Gather/Scatter 子事务均由所属 MFE job 消费与 ack；HostSession 在 group.step 之后只消费自己的事务 id；每个子事务 terminal 后及时 ack 一次。
- profile/range gate：MFE job 与 HostSession 在飞事务计入 `_closure_ledger/_level_quiescent` 与 `range_inflight`；index 已读完但 payload 未提交的间隙仍非 quiescent。gate 关闭只阻止新 launch，已接受工作继续 drain。
- fault/reset：冻结 CPU 与 host 新工作，取消未发 index/payload 与未执行 host 命令；已接受事务完成或隔离后 ack，再释放 lease、root pin，最后 `abort_after_drain`。仍走同一 ResetDomain，DONE 后返回失败结果。成功出口额外检查 HostSession、PagePool（无 pins/refs/存活 scope 页）闭合，漏 `HostFreePages` 走 controller 可见 fault drain。
- report：`gather_fidelity` 改为 `"address_resolved_cache"`（仅在有 Gather 时输出）；`gather_request_conservation` 改为 `requests == l1_hits + l2_hits + hbm_misses + cache_bypass_requests`；新增可选节 `scatter`（jobs、segments、bytes、index_reads）、`host_runtime`（calls、commands、software_cycles、host_read/write_bytes）、`memory_pools`（每 pool 的 initial/allocated/freed/live/peak 页、pins、refs）。新 check：`pool_page_conservation`（initial+allocated−freed=live）、`host_zero_leak`、`pool_zero_leak`。无新功能时不输出空节。
- trace 事件：`gather_index_read/gather_lookup/gather_refill/gather_response`（含 line identity、hit level、merge leader/waiter、真实地址）、`scatter_index_read/scatter_write`、`host_call/host_command/host_read/host_write`、`pool_allocate/pool_free/scope_pin/scope_unpin`、`maintenance_precise_ranges`（解析后的字节区间）。
- 观测补充：`scatter` 节加 `overlap_wait_cycles`、`commit_latency_cycles`（sum/max）；`memory_pools` 加每个 scope 的 `pin_hold_cycles`；`maintenance_precise_ranges` 同时给出静态 range 字节数和解析后字节数，用来量化 precise maintenance 的收益。
- CLI 新增可重复 `--input-data NAME=PATH`：文件非空、≤ binding size、NAME 唯一且存在，按 binding base `seed_hbm` 文件前缀；仅允许 full_memory；与 `--all/--compile-only/--print-ir` 互斥；错误 exit 2。

### 6. 场景与对照

**独立索引场景**：`examples/generators/generate_indexed_memory.py` 生成 `examples/workloads/indexed_gather_scatter.mlir` 与 `examples/workloads/indexed_memory_data/{data,gather_indices,scatter_indices}.bin`。DATA 为 `[4,4]i32`，行值 `[10..13]/[20..23]/[30..33]/[40..43]`；GATHER_IDX=[3,0,3]、SCATTER_IDX=[1,1,2]、OUT=`[4,4]i32`；map `S=4,O=0,P=0,R=1,T=0,L=4`。单 context、1 task 的 Tile Program：load 两组 indices 到 L1 → Gather DATA→L1 `[3,4]` → await → Scatter L1→OUT。OUT 未写行不读取；run 后由测试用 `ByteStore.read_hbm` 检查第 1 行=`[10..13]`（ordinal 1 覆盖 ordinal 0）、第 2 行=`[40..43]`。run.sh 新增 `indexed-memory` case，bindings DATA=0x100000:64:r、GATHER_IDX=0x200000:12:r、SCATTER_IDX=0x300000:12:r、OUT=0x400000:64:w，三份 input-data，profile 默认使 L1/L2 cache 启用（allowed_profiles=[1,2]）。

**PagedAttention（仅 examples）**：

- 新增 `examples/generators/generate_paged_attention_decode.py`（默认 `--output-dir examples/workloads`）生成 `paged_attention_decode_pipeline.mlir`、`paged_attention_decode_baseline.mlir`、`paged_attention_decode_scenario.json`（schema_version=1，字段 `initial_lengths/steps/page_tokens/physical_pages/page_padding_bytes/contexts_per_tile/l1_mode/l2_mode/seed/initial_mapping/source_hashes`）；新增 `examples/generators/run_paged_attention.py` 构建 HostEnvironment/ByteStore，compile→load→`Simulator.run(host=...)`，复用 report/trace 序列化。
- 默认参数：R=3，initial_lengths=[255,511,767]，steps=4，B=16，physical_pages=128，padding=64 B，GQA 16:4（KVH=4、HPK=4），D=64，bf16，placement=15，contexts_per_tile=4，l1_mode=1，l2_mode=1，group-policy=s1，num_dma_channels=2，hbm_fixed_latency_cycles=10，max_cycles=2_000_000。参数校验：长度非负、其余为正、初始总页数≤physical_pages。
- 普通 globals：`POOL [physical_pages, page_stride_elements]bf16`，每页按 `[2(K/V),4,B,64]` 解释，page stride=`2*4*B*64*2+padding` 字节（默认 16448，64 B 对齐）；`BLOCK_TABLE [R*max_pages]xi32`（max_pages=ceil((max(initial)+steps)/B)）、`APPEND_IDS [R*steps]xi32`、`LENGTHS [R]xi32`；`Q_IN [R,S,4,4,64]bf16`、`K_NEW/V_NEW [R,S,4,1,64]bf16`、`S_INIT [R,S,4,4,4,2]f32`、`O_INIT [R,S,4,4,4,64]f32`（第 3 维为 4 个 partition）、`OUT [R,S,4,4,64]f32`。scopes `owner_0..owner_2`，一个 HostPagePoolSpec 绑定 POOL；初始映射 `random.Random(seed=0)` shuffle；有效 KV 字节 `(17*r+31*token+7*head+byte+kind_bias)%251`（K bias 0、V bias 113）；主表未用项 -1、LENGTHS=initial_lengths；APPEND_IDS、空闲页、padding、尾部不 seed。
- 每 (r,s) 的 nexus DAG：`prepare_r{r}_s{s}`（host）→ context `step_r{r}_s{s}` → `commit_r{r}_s{s}`（host）；下一步 prepare 依赖本步 commit；最后 `release_r{r}` 依赖最后一次 commit。跨 request 不加 barrier。
  - prepare：必要时 `HostAllocPages(1)`，`HostWrite` 当前 token 所在物理页号到 `APPEND_IDS[r*steps+s]`；声明该 4 B write 与 scopes=[owner_r]。
  - step context：连续 prefetch K_NEW/V_NEW、APPEND_IDS 单项、每块的 BLOCK_TABLE 单项（最后一块改用 APPEND_IDS 单项，因为当前 token 所在页即最后一页）、Q、partition state 到 L2。append dispatch（4 task，task=KV head）：load K/V 新行与 append index 到 L1，两条 Tile Scatter（K：`S=page_stride_elements, O=token_in_page*64, P=B*64, R=1, T=0, L=64`；V 的 O 加 `4*B*64`），scope=owner_r。attention 每块一个 dispatch（4 task）：load 块 index 到 L1，K/V 两条 Tile Gather（`O=0` 与 `4*B*64`，`P=B*64, R=1, L=valid_tokens*64`，目的 L1 `[valid,64]`），scope=owner_r，随后按现有 decode 计时：QK/PV 各 `2*4*T*64` ops、score EVU `4*(T+2)`、accumulate EVU `2*4*64`。append 与 attention 之间由编译器插入 context 内 precise postwrite invalidate。
  - commit：若本步新开页，`HostWrite` 主表对应项；再 `HostWrite` LENGTHS 新长度；scopes=[]。
  - release：主表已用项写 -1、LENGTHS 写 -1，最后 `HostFreePages(owner_r)`。
- pipeline 与 baseline 的唯一差别是块级调度：pipeline 把块 b 分给 partition `b % 4`，dispatch b 只依赖 dispatch b−4 的 output_ready，最后一个 merge dispatch（每 task EVU `4*(18+8*64)` ops，含最终 normalize）合并 4 个 partition 后 store；baseline 单 partition，dispatch b 依赖 b−1 的 grid_done，最后一块追加 normalize EVU `4*64`。超过 16 个 grid 时按现有 completion-only retirement window=8。Tile Program 按 (valid_tokens, 是否末块) 复用。
- run.sh 新增 `paged-attention-decode`、`paged-attention-decode-baseline`，调用 runner 的 `--variant pipeline|baseline`；runner 支持 `--scenario`、互斥的 `--compiled-file/--compiled-output`、`--memory-trace/--trace-json/--json/--report`，并复用 CLI 的 `--hw-override/--sim-override/--group-policy/--context-mode/--device-context-mode/--max-cycles`。bindings 按 POOL、BLOCK_TABLE、LENGTHS、APPEND_IDS、Q_IN、K_NEW、V_NEW、S_INIT、O_INIT、OUT 顺序，首地址 0x1000000，其后 `align_up(prev_end, 2 MiB)`；前 4 项 rw，Q_IN..O_INIT r，OUT w。replay 先核对 scenario source_hash。
- `examples/generators/analyze_paged_attention.py` 读两份 JSON report（单元素数组）与完整 trace，输出 makespan/speedup、每 request-step 延迟（<20 样本不报 P95）、payload/index/host 字节、L1/L2 hit/miss/merge 与按 level 的命中率、页峰值/零泄漏、HBM 读与 BOA 区间重叠率（区间并集，不超 100%）。缺字段明确失败，无提速如实输出。
- analyzer 还要输出每个 request-step 的“append 完成 → 首个 attention dispatch 开始”间隔：同 scope 的读写按整 view 串行（§1 冲突域），这是不引入运行时 claim 的代价，要实测，不能假设。

## 不变量与资源生命周期

借鉴 `scatter.md` §5/§6.1：新增资源逐项写明获取、释放、故障回收与复用条件；每条不变量对应验证项。

| 资源                         | owner              | 获取                                                 | 正常释放                                                        | 故障回收                             | 复用条件                                          |
| ---------------------------- | ------------------ | ---------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------- |
| index slot                   | Gather/Scatter job | INDEX_READ 发出                                      | 该 index 全部段写入 destination（Gather）或 DONE+ack（Scatter） | 取消未发，在飞隔离后 ack             | 释放后                                            |
| Scatter 子事务与 AccessLease | Scatter job        | `lease.acquire` 后 submit                            | terminal ack 时 release                                         | 同左（含 CANCELLED/FAULTED）         | lease epoch 仍有效                                |
| root scope pin               | root request       | GroupPort 接受时 `pin_root`（原子，拒绝/背压时回滚） | root 退役时 `unpin_root`                                        | drain 完成后、`abort_after_drain` 前 | —                                                 |
| scope 页                     | scope              | `HostAllocPages`（无 pins/refs）                     | `HostFreePages`（无 pins/refs）                                 | `abort_after_drain`                  | free 后；epoch+1，HBM 与 cache 副本 validity 清除 |
| host routine                 | host_call request  | issue（占 pending 容量）                             | generator 结束且其事务全部 ack                                  | 取消未执行命令，关闭 generator       | —                                                 |
| `precise_write_ledger` 条目  | TileGroup run      | Scatter 段 DONE                                      | `begin_launch` 清空                                             | 同左                                 | —                                                 |

不变量（括号内为验证项）：

1. Scatter 事件 ⇒ 全部段已提交 HBM 且 ack；grid_done ⇒ 该 dispatch 的全部 Scatter 已可见（3）。
2. Scatter 不分配、不弄脏任何 cache 行（3）。
3. 同一 Scatter 内重叠段按 `(i,j)` 顺序提交，不重叠段并行（1、3）。
4. 任意两个可能并发的写或读写，都有 §1 冲突域表中某一层的证明或依赖，否则编译/加载拒绝（3、4）。
5. 带 scope 的段在 submit 时落在该 scope 当前页内，lease 持有到 terminal ack；root pin 期间该 scope 不可 alloc/free（5）。
6. 带 Cache 的读不会返回 Scatter 之前或前 owner 的字节（3、5、11）。
7. fault 后不再 issue；已接受事务 terminal+ack 先于其 lease/pin 释放；ResetDomain DONE 时事务/MSHR/lease/pin/页全部归零（6）。
8. 成功出口没有存活的 host routine、pin、ref 或 scope 页（5、8）。
9. 活性：所有容量取 1 时场景仍能完成（10）。

## 考虑过但不采用（对照 `scatter.md`）

| 方案                                                                      | 处置   | 理由                                                                                                                                 |
| ------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| 运行时 destination claim/lease（§4.1、§6）                                | 不采用 | §1 冲突域每层已有静态证明、显式依赖或运行时页所有权校验；代价是同 scope 读写按整 view 串行，由 analyzer 的 append→attention 间隔度量 |
| L1/L2 write-combine cache（§3.3、§3.5）                                   | 不采用 | 已冻结决策 3：Scatter 绕过 Cache；因此没有 dirty 行，也不需要 clean-cancel 协议；代价是每段一次按 burst 取整的 HBM 写                |
| ScatterReduce / atomic add（§2.2、§4.3）                                  | 非目标 | PagedAttention 不需要；将来只能做成独立 op（对应 `MFE_SEG_SCATTER_ATOMIC_ADD`），不得作为 `tile.scatter` 的模式                      |
| `duplicate_policy = reject` + dispatch 前全量校验 indices（§4.1 第 6 步） | 不采用 | 保留 `(i,j)` 有序覆盖（对应 `MFE_SEG_SCATTER_ORDERED`）与窗口流水；任何 fault 都让整个 run 失败，成功路径观察不到部分写              |
| 缩小规模的穷举形式化模型（§7.1）                                          | 不采用 | 本计划没有需要穷举的 claim 协议；活性由验证 10 的极小容量运行覆盖                                                                    |

回退路径：precise maintenance 若被证伪，`profile_pass` 不再标 `precise_writes`，退回整段 invalidate；scope 不相交若被证伪，删除该放宽，跨 root 写退回显式依赖。两者都只撤销放宽，其余协议不变。

## Critical files & anchors

| 文件                                              | 锚点                                                                                                                    | 原因                                                                                                        |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `pipeline_validator/engines.py`                   | `launch_gather`、`_tick_gather_request`、`_try_l1_mshr/_try_l2_mshr`、`_tick_gather_materialization`、`retire_isolated` | 改为地址驱动查 Cache 的核心 FSM，保持 MSHR/顺序写/隔离结构                                                  |
| `pipeline_validator/memory/transfer.py`           | `_build_route`、`submit` 单端点白名单、`_capture_source/_commit_destination`                                            | 新增 INDEX*READ/GATHER_L\*\_LOOKUP/GATHER_L2_RESPONSE/SCATTER_WRITE/HOST*\* route，字节只在真实腿完成时可见 |
| `pipeline_validator/compiler/profile_pass.py`     | `_action_accesses`、`insert_task_maintenance`、`_device_maintenance`                                                    | Scatter 写/Gather 读 effect、precise_writes、post_invalidated 跳过 device fence                             |
| `pipeline_validator/memory/profile_controller.py` | `_resolve_ranges`、`note_dependencies`、`_closure_ledger`                                                               | 运行时把 precise maintenance 解析为 Scatter 实写区间；新 job 参与 quiescence                                |
| `pipeline_validator/device.py`                    | `_advance_program`、`_validate_and_count_references`、`_admit_ready_launches`                                           | host_call 与 group submit 共享事件生命周期                                                                  |

## Verification

工作目录为仓库根，环境 `elenor-validator`。新增测试文件 `pipeline_validator/tests/test_tile_indexed.py`、`test_host_runtime.py`、`test_page_pool.py`、`test_paged_attention.py`，全部用 tmp_path，断言真实字节、事件、资源与 trace 字段。

1. **独立索引**：indexed-memory 的 OUT 两行精确匹配；只替换 GATHER_IDX 文件，同一 artifact 结果与 trace 地址随之变化；index 读 = (3+3)×4 = 24 B（L1_READ），Gather payload 48 B、Scatter payload 48 B。
2. **真实 Cache**：迁移后的 `gather_indexed.mlir`（16 个 index 周期 [3,0,2,1]、L1/L2 cache 启用）满足 `l1_hits+l2_hits+hbm_misses+bypass=16`、HBM 回填恰 4 行（256 B）、其余命中或 MSHR 合并；同一程序在 L1/L2 cache_bytes=0 的 profile 下 hits=0、全部直通；连续两次 Gather 同一行，第二次 L1 命中。cache 行跨 view 末尾时走 bypass，requirement forbidden 时 fault。
3. **Scatter 与可见性**：Scatter 不改变任何 Cache 的 resident/dirty 统计；先 Gather 缓存某行、再 Scatter 改写该行、再 Gather，后者读到新字节且 trace 显示 precise 失效只覆盖 Scatter 实写区间。同 context 内 Scatter dispatch 后接同 binding 的 Gather dispatch（同 scope 或无 scope）：产物里 Gather 的 dependencies 自动含 Scatter 的 grid 事件，并插入 precise invalidate；WAR（Gather→Scatter）与 WAW（Scatter→Scatter）同样自动加边，不同 scope 不加边。篡改 artifact 删掉该边或该 maintenance，`load_program` 分别报 `lacks a required global RAW/WAR/WAW edge` 与 `lacks required cache maintenance after HBM write`。重复目的按 ordinal 后者覆盖；不相交段并行（trace 中首个完成前均已 issue）。Scatter tile 事件的 cycle 不早于其最后一个 SCATTER_WRITE 的 HBM commit；纯 Scatter dispatch 的 grid_done 晚于全部段 commit。
4. **静态规则**：多 task Scatter 段重叠报 `tile_scatter_task_overlap`；同程序无 await 的重叠 Scatter/Gather 报 `tile_indexed_hazard`；同 dispatch 既 Scatter 又 Gather 同 formal 报错；负 index/越界/溢出/未初始化 index 在相应层失败，首项非法无 payload。indices/Scatter source 的 load 或 Gather 写未 await 就被使用时 verifier 拒绝。跨 root：两个 request 用不同 scope 且无依赖时，trace 中二者的 Scatter/Gather 时间重叠；同 scope 或无 scope 时去掉依赖，source verifier 报 `overlapping global accesses`（一侧 Scatter 写、一侧 Gather 读即可触发）。
5. **Host 与页池**：host 事件与 submit 互相依赖、cycle N 完成 N+1 唤醒；host 满不占 Group outstanding。两 owner 非 Attention 记录池：alloc 最低页、free 后复用 epoch 增长、旧事务无法写新 owner；busy alloc/free、double free、OOM 原子失败、越 owner 地址、pool 有残留 cache 行注册均拒绝。
6. **故障收敛**：在 index 读完/payload 未发、HostWrite 在飞、scope pin 期间触发 profile gate 与 fault/reset，最终 ResetDomain DONE，事务/NoC/MSHR/lease/pin/页数全部归零。
7. **PagedAttention 微例**（B=4,H=1,D=2,bf16，初始 3 token 在页 2）两步 append 跨页，新页由 allocator 分配为 0；Gather 回读字节正确，未初始化尾部从不进入 destination。在 l1_mode=l2_mode=1 下跑：K/V 行远小于 64 B 行，整行回填必然带上未初始化的尾部 token 或 padding，不得 fault；再构造一条越过有效 token 的读，fault 必须发生在 destination 写。
8. **默认场景对账**（pipeline 与 baseline 相同项必须相等）：最终长度 [259,515,771]，12 次 append，初始 96 页、新分配 3 页、释放 99 页；393 个 attention 块、有效 token 6162；Gather payload 6,309,888 B、Scatter payload 12,288 B、L1 index 读 12,960 B（393×32+12×32，每 dispatch 4 task×K/V 2 条×4 B）、index DMA 预取 1,572 B（每块 1 个 4 B 项，末块复用 APPEND_IDS 项，即 393×4）、Host 写 516 B；BOA ops 25,239,552。EVU ops：baseline 928,320，pipeline 1,017,792（多出 4 partition merge、少 normalize），从 trace `args.ops` 汇总。Cache hit 数不固定，只校验守恒与 l1_mode/l2_mode=0 时 hits=0。
9. **兼容**：v2 artifact 被拒并提示重编译；新 v3 round-trip 结果一致；无关 transformer/多 context/L2 共享/Profile/维护场景测试通过。
10. **最小容量活性**：IR 中 `window_entries=1`，并设 `l1_mshr_entries=l2_mshr_entries=1`、`mfe_load_channels=mfe_pipeline_depth=1`、`hbm_outstanding_limit=1`、`sim.device.pending_capacity=1`、`sim.device.issue_width=1`；indexed-memory 与 PagedAttention 微例都要跑完，OUT 与页内字节和默认容量下一致（`scatter.md` §7.2：存在一条通往终态的路径不等于活性，极小容量用来暴露循环等待）。
11. **页复用与陈旧行**：r0 的 Gather 让页 p 的行驻留 L1/L2 → `HostFreePages` → r1 分配到 p。r1 先 Scatter 再 Gather，读到新字节；构造一条越过有效 token 的读命中陈旧行，必须在 destination 写报未初始化，而不是静默返回 r0 的字节。

```bash
conda run -n elenor-validator python -m pytest -n0 \
  pipeline_validator/tests/test_tile_indexed.py pipeline_validator/tests/test_host_runtime.py \
  pipeline_validator/tests/test_page_pool.py pipeline_validator/tests/test_paged_attention.py
PYTHONPATH=. conda run -n elenor-validator python examples/generators/generate_gather_inputs.py
PYTHONPATH=. conda run -n elenor-validator python examples/generators/generate_indexed_memory.py
PYTHONPATH=. conda run -n elenor-validator python examples/generators/generate_paged_attention_decode.py
bash examples/run.sh indexed-memory --memory-trace --trace-json /tmp/indexed.trace.json \
  --json --report /tmp/indexed.report.json
bash examples/run.sh gather --memory-trace --trace-json /tmp/gather.trace.json
bash examples/run.sh matmul-gather-add --memory-trace --trace-json /tmp/mga.trace.json
bash examples/run.sh paged-attention-decode --memory-trace \
  --trace-json /tmp/pa-pipeline.trace.json --json --report /tmp/pa-pipeline.report.json
bash examples/run.sh paged-attention-decode-baseline --memory-trace \
  --trace-json /tmp/pa-baseline.trace.json --json --report /tmp/pa-baseline.report.json
conda run -n elenor-validator python examples/generators/analyze_paged_attention.py \
  --pipeline-report /tmp/pa-pipeline.report.json --baseline-report /tmp/pa-baseline.report.json \
  --pipeline-trace /tmp/pa-pipeline.trace.json --baseline-trace /tmp/pa-baseline.trace.json
bash examples/run.sh transformer-decode-kv --report /tmp/legacy-decode.report.txt
conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -v
```

交付时报告两种变体的周期、各类字节、L1/L2 命中/miss/merge、页峰值与重叠率；若 pipeline 不快于 baseline，保留真实结果并指出等待链，不调参数掩盖。

## Assumptions & contingencies

- 用户决定：Gather/Scatter 只在 Tile 级；Gather hit/miss 由真实地址查 Cache；Scatter 绕过 Cache、依赖编译期 maintenance；动态页池由 Host/runtime 管理；允许删除 profiled Gather 与旧 artifact 兼容。
- 新地址驱动功能要求 full_memory + ByteStore；仅以 runtime/timing_only 运行旧无关 workload 不受影响。
- 若默认场景资源合同超出 L1/L2 profile，按 `transformer_common.contract_bytes` 修正新生成器的合同，不调大硬件默认容量。
- 若 precise maintenance 的 device 级路径在实现中发现 `note_dependencies` 签名变更影响其他调用方，按 grep `note_dependencies\(` 全部同步更新，不保留旧签名。
- 与 ELENOR 设计文档对齐（`scatter.md` §8.1 的教训：跨文档不一致要显式记录）：Tile Gather/Scatter 对应 `design/elenor_mfe/ELENOR_MFE_Design.md` §4.2 的 `MFE_SEG_GATHER_ONLY`/`MFE_SEG_SCATTER_ORDERED`；跨 tile 只允许已证明不相交的写，符合 §4.3“Cross-tile update 不在 MFE V1 内解决”。偏离：不建模 MFE Page Stream 的硬件 page walk。PagedAttention 结果度量的是 BLOCK_TABLE→L2→L1→INDEX_READ 的软件 page walk 路径，不能当作 Page Stream 的性能。
