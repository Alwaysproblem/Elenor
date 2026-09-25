# 批次 I：私有 L2 提前释放与物理 extent 生命周期

本文件是原计划的**批次 I 执行清单**：实现私有 L2 的非重叠布局、buffer 粒度物理释放，以及 transaction、异常清理、占用记账和 trace 闭环。它是计划，不表示下面的新 API 或行为已经存在。

- [原始联合计划](../L2_显式共享_PLAN.md)：本批次覆盖第一、二步、第五步中 transaction hooks 与 private cancel/reset、以及第八步的物理占用和 extent trace；基线与验收以其“验证方案”为准。
- [批次 II：显式只读 L2 共享与生命周期](02_explicit_l2_sharing.md)：在本批次的 backing/final-free 生命周期上增加跨 context claims；本批次不得提前实现或假定 shared claim 已闭合。
- [批次 III：场景与端到端验收](03_scenarios_and_acceptance.md)：执行全 CLI 双档场景、端到端 trace 审计和全量回归。
- `plan/README.md` 由父任务维护；本文件不改写该入口。

## 范围和冻结合同

- L2 默认仍是 private；本批次不加入 `sharing`、publish/import、consumer claim、cross-context ready 等语法或行为。same-profile admission 只由 buffer final-free 的实际 pool-version 变化唤醒；不同 L2 profile 仍等完整 root completion history / full frontier，不能把 `input_released`、view invalidation 或 extent-free 当成 profile-switch frontier。
- 每个私有 L2 buffer 有一个独立且不重叠的**padded backing span**。`nest.release` 永久放弃该私有 view 的访问权；所有实际访问和 transfer reference 都安全排空后，才可把该 buffer 的完整 stripe-rounded span 归还给同一个 L2 free map。`nest.barrier` 不恢复已释放 owner 的权限，也不允许 L2 地址回绑。
- Arena 初始化时仍原子预留整个本地资源合同；各 buffer span 之间未分配的部分是 arena slack，由 root 持有到 root retirement。释放一个 buffer 绝不把其余 buffer 或 slack 一并归还，也不以逻辑 `AllocationHandle.bank_segments` 的 valid bytes 代替物理 padded span。
- 同 L2 profile 下，A 的 input backing 在自己的读访问和 prefetch transaction 完全安全后物理释放即可唤醒同类 FIFO 队头 B；不能等 A 的 output、pow、store 或 context completion。不同 L2 profile 仍以完整 root completion/frontier 为 quiescence 边界，不把 buffer free 当成切档 frontier。L1 的 Task Arena、`tile.free`、L1 invalidate/free-map 和 L1 profile 规则完全不变。
- 不保存可独立漂移的 integer refcount，不新增第二个 allocator，不加逐周期 admission busy-retry，不越过原有 SAME/COMPATIBLE FIFO 队头。
- 本批次为后续扩展保留一个 backing-centric finalization seam（唯一 `_try_release_l2_backing(backing_id, cycle)` 与统一 backing record）；不得加空壳/无语义 shared 字段。producer/claim 状态机、pending-reader 保活、共享永久容量分类、跨 epoch / shared quiescence、shared 计数与 shared read/write/publish 检查全部明确留给批次 II。

## 0. 改代码前：冻结 workload 容量等待基线

这一步必须先于**任何批次 I 源码修改**；保存值和原始产物，不填估算值，不以一次失败重跑代替已有证据。此前已报告的旧 `l2-admission-wait` 失败是已知基线，**不要为了确认它而在修改前重跑**；它属于 `Protocol scenarios`，不是下列 Runnable workloads 基线集合。

1. 从仓库根目录创建唯一 run-id 目录并冻结实际列表：

   ```bash
   RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
   OUT="examples/artifacts/l2-sharing-release/${RUN_ID}/baseline"
   mkdir -p "$OUT"
   bash examples/run.sh list | tee "$OUT/run-list.txt"
   ```

   只解析 `Runnable workloads:` 到 `Protocol scenarios:` 之前的名称；不能把 protocol scenarios 或 NEST 子图混进 workload 分母，也不能手抄/固定今天的列表。对列表中的**每一个**名称分别执行 `runtime`、`full_memory` 两档。使用 throwaway runner 写在本次 `OUT`（不增加仓库维护脚本）：逐项调用以下形式，保存完整 argv、退出码、stdout、stderr、trace、report；即使一项失败也完成矩阵的其余项，然后把失败标为基线缺项，不伪装成功。

   ```bash
   bash examples/run.sh <runnable-name> \
     --sim-override fidelity=<runtime|full_memory> \
     --memory-trace \
     --trace-json "$OUT/<name>.<fidelity>.trace.json" \
     --report "$OUT/<name>.<fidelity>.report.json" --json
   ```

   以下 runner 可原样从仓库根目录运行；它只读 `run-list.txt` 的 Runnable 区间，逐个执行全矩阵并把每条命令和 stdout/stderr 留在 `OUT`。单项失败不会阻止其余 baseline 运行，最终以非零状态提示缺项：

   ```bash
   OUT="$OUT" python3 - <<'PY'
   import json
   import os
   import re
   import subprocess
   from pathlib import Path

   out = Path(os.environ["OUT"])
   text = (out / "run-list.txt").read_text()
   names = []
   lines = text.splitlines()
   try:
       start = lines.index("Runnable workloads:") + 1
       end = lines.index("Protocol scenarios:", start)
   except ValueError as exc:
       raise SystemExit("missing workload/protocol list boundary") from exc
   for index in range(start, end):
       match = re.search(r"workloads/[^\s]+\.mlir\b", lines[index])
       if match is None:
           continue
       label = lines[index][:match.start()].strip()
       if not label:
           label = lines[index - 1].strip()
       if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", label):
           raise SystemExit(f"unrecognized workload entry at line {index + 1}")
       names.append(label)
   if not names or len(names) != len(set(names)):
       raise SystemExit("Runnable workload list is empty or duplicated")
   failures = []
   for name in names:
       for fidelity in ("runtime", "full_memory"):
           stem = f"{name}.{fidelity}"
           argv = [
               "bash", "examples/run.sh", name,
               "--sim-override", f"fidelity={fidelity}", "--memory-trace",
               "--trace-json", str(out / f"{stem}.trace.json"),
               "--report", str(out / f"{stem}.report.json"), "--json",
           ]
           (out / f"{stem}.command.json").write_text(json.dumps(argv, indent=2) + "\n")
           proc = subprocess.run(argv, capture_output=True, text=True, check=False)
           (out / f"{stem}.stdout").write_text(proc.stdout)
           (out / f"{stem}.stderr").write_text(proc.stderr)
           (out / f"{stem}.exit-code").write_text(f"{proc.returncode}\n")
           if proc.returncode:
               failures.append(f"{stem}: exit {proc.returncode}")
   if failures:
       raise SystemExit("\n".join(failures))
   print(f"completed {len(names)} workloads x 2 fidelities")
   PY
   ```

   Runner 必须原样传入现有 `examples/run.sh` 为该名称配置的 IR、硬件配置、绑定、硬件 override 和 cycle cap；不要换成只覆盖目标容量等待的手工 IR。对应地，对每次实际源 IR、`examples/run.sh`、所有 `examples/configs/*.yaml`、参与编译/验证/allocator/runtime/trace 的当前源文件保存 SHA-256 清单；运行命令和清单写入相同 run-id 目录。至少覆盖 `pipeline_validator/compiler/resources.py`、`execution_verifier.py`、`memory/arena.py`、`memory/allocator.py`、`memory/byte_store.py`、`tile_group.py`、`runtime/group_port.py`、`memory/transfer.py`、`engines.py`、`memory/profile_controller.py`、`simulator.py`、`trace.py`。保留每份 report 自带的 `compiled_artifact_hash`、`registry_hash`、cycles/completed 和有效配置；配置 YAML 没有被某个 workload 使用时仍保留其文件 hash，但不要声称它被加载。

   ```bash
   sha256sum examples/run.sh examples/workloads/*.mlir \
     examples/configs/*.yaml \
     pipeline_validator/compiler/resources.py \
     pipeline_validator/execution_verifier.py \
     pipeline_validator/memory/arena.py \
     pipeline_validator/memory/allocator.py \
     pipeline_validator/memory/byte_store.py \
     pipeline_validator/tile_group.py \
     pipeline_validator/runtime/group_port.py \
     pipeline_validator/memory/transfer.py pipeline_validator/engines.py \
     pipeline_validator/simulator.py pipeline_validator/trace.py \
     > "$OUT/source-config.sha256"
   ```

2. 从每档原始 trace / 对应 report 提取 **root request** admission 等待，而不是 Tile task wait。只采 `Scheduler:L2` 上 `context_admission_wait`、其 `wait_reason` 变化和对应 `context_admitted`；以 `request_id` 配对并以实际 admission 结束该请求最后一个 wait interval。将 `WAIT_CAPACITY`、`WAIT_FRAGMENTATION` 分开，报告：进入过该 reason 的 distinct request 数、每 reason 与合计 root-wait cycles（并发请求的和）、把等待区间合并后的 union cycles / report makespan 占比。另列 `WAIT_SLOT`、`WAIT_CONTROL_RESOURCE`（并保留若出现的其他 wait reason）的请求数、cycles；不得把 slot/control wait 混计作容量收益。事件没有 `args.cycle` 时按该 run 的有效硬件 `cycle_ns` 用 `round(ts * 1000 / cycle_ns)` 转回 cycles；不把 JSON 的 `ts` 当原始 cycle。用 trace event 证明等待区间，用 report 的 `cycles` / `makespan.span_cycles` 作分母，并将抽取结果、提取器和 source/config hash 同档存档。

3. 基线通过条件：列表中的所有 runnable-name × 两档都有可审计退出状态；每个完成运行的 request wait 状态变化闭合到 `context_admitted`，等待总和和 union 可复算。若容量 wait 为 0，原样记录“该 workload 预期没有 R3-6 直接收益”，但不据此取消已指定实现；若基线或 hash 不完整，停在本阶段，不开始改代码。

## 1. 固定编译布局与独立 executable verifier

**改动面：**`pipeline_validator/compiler/resources.py`、`pipeline_validator/execution_verifier.py`；资源合同文字落在 `pipeline_validator/IR_SPEC.md` §3.8、§8。

- 在 `prepare_resources` 的 L2 分配路径不再传入可缩短 buffer lifetime 的值；用 `layout_buffers(..., lifetimes=None, slot_capacity=max(1, len(local_buffers)))` 生成各本地 L2 buffer 的独立、不相交 stripe-rounded span。删除只为 L2 地址回绑服务的 `_l2_lifetimes`、`_l2_reuse_pairs`、`_apply_l2_reuse` 和对应 verifier 镜像的“释放后 barrier 才可回绑”证明。**保留** `_tile_lifetimes`、L1 layout/reuse/FREE 验证，不能改变 L1 lifetime 或 Task Arena 行为。
- 每个 buffer 的 padding 采用完整 stripe round：`round_bytes = layout.stripe_bytes * banks`；总 padded bytes 为 `ceil(logical_bytes / round_bytes) * round_bytes`，每 bank span 为该值除以 bank 数。padding 属于 buffer span 并随它一同 free；不能把 `AllocationHandle.bank_segments` 的 valid bytes 当 padded span。
- 当无回绑 layout 高水位超过原 source resource declaration 时，将对应 source resource bytes 修正为现有 `conservative_arena_bytes` 对所有本地 buffer 的 no-reuse 结果；随后按允许 profile 正常校验。超硬件 profile、bank 或 slot 上限就明确 compile/load 失败；不暗改硬件 capacity、不恢复别名，也不在 verifier 放宽合同。
- compiler 和独立 executable verifier 分别验证：每个 executable local L2 buffer 恰有一个 layout、一个 bind、一个 release；padded per-bank span 在 Arena reserve/profile 用户区间内、正确对齐且彼此不相交；总 span + slack 与每 bank reservation 精确守恒。verifier 从冻结 DTO/actions 自行重算，不信任 source verifier 已经接受、hash 自洽或 compiler 生成的布局；旧的重叠/reuse artifact 必须被独立拒绝并要求 source 重新编译，不能以调高 resource bytes 掩盖旧别名。
- `IR_SPEC.md` §3.8、§8 将 L2 明确为永久 forfeiture：view release 撤销当前 invocation 的该 view；最后一个 private reference 与 pin/transfer 安全排空时归还完整 padded backing；`nest.barrier` 不能重获 owner 或恢复 L2 回绑。将 §3.8 现有“successful release 不改 free map / 只在 Arena retirement 归还容量”和 §8 现有“`invalidate_view` 永不改 free map / 只有 `retire_arena` 归还 extents”的 L2 条款替换为新合同；写清楚 L1 仍保持原合同。

**阻断门：**任一现有 workload 仅因移除 L2 alias 而越过 profile/静态合同、编译器与独立 verifier 对布局或 action 次数不一致、或 L1 layout hash/lifetime/测试发生额外变化时，停工；修正源 resource declaration 或实现问题并重新从这一整项证明，不准把重叠当“临时可用”。

## 2. 单一 L2 free map 上的 backing、exact-unit 切分与私有早期 free

**改动面：**`pipeline_validator/memory/arena.py`、其唯一 exact allocator `pipeline_validator/memory/allocator.py`、`pipeline_validator/tile_group.py`、`pipeline_validator/memory/byte_store.py`、相关 memory snapshots/tests。不要添加第二个 allocator，也不要借 L1 `BankedFreeExtentAllocator`/Task Arena 改 L2 行为。

1. 新增 L2 专属 `_L2BackingRecord`，以唯一 `backing_id` 表示一个物理对象，保存 exact padded reserve、逻辑有效 segments、origin arena、allocation/profile generation、当前 private owner-reference 状态、生命周期状态及该 backing 的 pins/accepted-transaction 关联。private 阶段一个 backing 只有其唯一 owner reference。`AllocationHandle` 增加物理 `backing_id`（L1/HBM 为空）；逻辑借用仍有各自 `allocation_id` 和 `ContextBufferOwner`。物理 backing generation/profile_generation 跟物理对象一致；launch generation 留在 owner，不允许借改 owner/ID 隔离真实 backing。物理 identity 不能退化为 address 或未经验证的 shape。
2. `commit_arena` 先完整重验原 immutable ArenaPlan、version、profile、owner 和 deterministic first-fit，再将 whole-bank reserve 切成之后确实会原样 free 的 exact units：按各 buffer `arena_offset / banks`、每 bank padded stripe 长度生成 backing units；把余下不属于任何 padded buffer 的区间显式切为 owner-arena slack units。检查 unit 在每个 bank 对齐、范围内、互不重叠，且 buffer units ∪ slack units 与原 whole-bank reserve 的区间并集和总 bytes 完全相同；再通过 `_extents.commit_exact(expected_version, all_units, cycle)` **一次**原子 commit。不能先让 `_exact_live` 记录 whole arena 再把不同 subsegment 交给 `release_exact`，因为后者要求返回完全相同的 exact units。L1 继续原有 whole-reserve commit。
3. 物理 backing 之外的 arena slack 仍占 root 容量直到成功 `retire_arena`；`ArenaHandle.reserve/reserved_bytes` 保留 admission 时初始合同语义。`retire_arena` 只 release 当前仍由 root 持有的 slack/未释放 private 残余并退休 root metadata；已释放 backing 不得二次 free。`bind_view` 只能绑定自己的新 backing；运行时发现同一 forfeited 区域被重绑、重复 bind、错 owner、stale generation/profile 一律 invariant fault。
4. `invalidate_view` / `TileGroup.release_l2` 仍先作完整 read-only preflight：release deps 已完成、声明 reader 的 `input_released`、writer 的 `output_ready` 已满足、owner/role/generation 正确、该 private view 无 pin 与 in-flight use。preflight 失败时 roles/handles/pins/free map/pool version/occupancy/trace 全部不变；公共重复 `nest.release` 必须是 `double release` invariant fault。只有 reset/cancel 内部清理可对已取消/已释放对象幂等，不能重放公共释放。
5. 成功逻辑释放只撤销当前 owner/view。所有私有引用撤销且 backing 的 pins 与实际 accepted transactions 都为零后，唯一 `_try_release_l2_backing(backing_id, cycle)` 才调用 `_extents.release_exact` 归还完整 padded units、推进 pool version 和 occupancy；`unpin`、`end_inflight` 的最后完成也走同一 finalizer。若物理访问尚未安全，保留 backing/free-map，不能把非法 public release 偷换成自动等待，也不能提前 wake FIFO。`ByteStore._domain_for_view` 的 L2 byte domain 按 `backing_id + physical generation + profile_generation` 稳定，producer 写入的真实字节不因 view owner / allocation ID 改变而变空域；其它 memory space 不变。`PayloadTracker` 继续只作地址级 layout 元数据，不扩 API、不以其命中代替字节或容量证据。
6. `TileGroup` 的 `_l2_reserved_bytes`、`_l2_live_bytes`、peak 与 `_record_l2_occupancy` 按物理 backing/slack 更新，不按逻辑 alias 重复加减。buffer invalidate 可改变逻辑 live-view bytes，但只有 exact backing final-free 改变物理 reserved/free bytes、pool version 和 `_l2_capacity_change_cycle`；root retirement 只减实际退回的 slack/残余，不得无条件扣 `ArenaHandle.reserved_bytes` 初始 reservation。
7. 重用原 `ArenaPool` / `BankedFreeExtentAllocator` 这一个 L2 free map 和现 admission-version 门：`GroupPortAdapter._admit_pending` 仅在现 admission token（包括 L2 `pool_version`）变化时重试；确保同拍 exact final-free 的新版本可被 Group tick 后的 FIFO retry 看见。保留 SAME/COMPATIBLE 分类和各自队列队头，不越队、不逐拍 busy retry，也不再加重复 `extent_pool_version` retry gate。队头若空池永远不可能容纳，返回 `PERMANENT_CAPACITY`；可由已占区间安全释放后容纳但当前 free bytes 不够为 `WAIT_CAPACITY`；总 free bytes 足够但 aligned contiguous bank span 不足为 `WAIT_FRAGMENTATION`。root 正在等待容量期间不得预占 device slot、event 或 arena。
8. L2 backing record 以物理身份和单一 finalizer 为扩展点；本批次只验证 private owner。不要在本批次加 shared borrow、published 权限、claims 状态、共享 reader accounting 或它们的“零计数”断言；这些由批次 II 将 claims/pins/transactions 接入同一 backing finalizer，不另造 allocator / free-map。

**阻断门：**exact units 有一字节区间缺口/交叠、tail padding 被漏还、slack 提前释放、release 导致 pool version 不变或重复 free、容量 wait 不是由真实 final-free 唤醒、B 越过同类队头、byte readback 不等于写入字节，任一出现都不得进入 transfer hook 阶段或提交。

## 3. Accepted transfer 引用：submit 原子登记，ack 后释放

**改动面：**`pipeline_validator/memory/transfer.py`、`tile_group.py`、`engines.py`、`memory/profile_controller.py` 及测试；逐个审计所有 L2 源/目的端，而不从 transaction issuer 猜被保护的 owner。

1. `TransferManager` 增加可选 `reference_acquire: Callable[[MemoryTransaction], None]` 和 `reference_release: Callable[[MemoryTransaction, int], None]` hooks。无 SRAM endpoint 的 manager 单元测试可不设；使用物理 L2 endpoint 的实际 `TileGroup` 必须配置并 fail closed。hook 在物理层对去重后的所有 L2 src/dst views 一次预检与登记；private 阶段按其 backing/view 关联一个 accepted transaction。为后续 aliases 保持 backing identity 是唯一物理键，但 shared alias 映射与授权由批次 II 实现。
2. 完成 transaction 全部验证、安装至 `_transactions` 时，且在任何 transport issue 之前一次性登记全部相关 L2 references。无论 Global DMA prefetch/store、MFE load/store、Gather L2/L1 refill 与最终 destination write、cache dirty clean/writeback、MSHR waiter、maintenance transfer 还是 NoC leg/cancel drain，只要实际持有 `AllocationHandle` 就不得漏过。纯 HBM/cache-only 路径不虚构 L2 reference，继续受原 profile drain 约束；不得因每次 SPM release 而无证据 flush 整个 cache。
3. 登记须原子：transaction 预检/accept 拒绝时零登记；登记中任何失败时对已登记 view/backing 完整 rollback；事务记录持有明确的 acquisition-success 状态。拒绝或未成功 acquire 的 transaction 后续不能在 ack 时扣一个不存在的 reference。对 view 和 backing 都记录 transaction ID 关联，使 private view release 和物理 backing final-free 以真实的引用集合判断。
4. 只在 `acknowledge(transaction_id, cycle)` 收到已确认的 terminal transaction，在它真实字节访问及 source/destination owner 后处理完成后调用 release hook，再移除 manager record；同步改成 `acknowledge_all_terminals(cycle) -> int`。DONE / 已隔离后安全 CANCELLED / FAULTED 均由**terminal acknowledgement**归还；`CANCEL_REQUESTED`、引擎看似 idle、只看到 `input_released` 都不是终态，不能撤 reference。accepted-but-not-issued 的 transaction 和 cancel-requested leg 一样继续保活，直到确认终态并 acknowledge。保留 `has_inflight_access` 为 release preflight 的独立交叉检查，不能以它替代物理 reference ledger。
5. 迁移并核对所有真实 ack 路径：`engines.py` 中各 engine/MFE owner retirement；`tile_group.py` 的 global transfer/completion/reset；`memory/profile_controller.py` 的 cache maintenance/clean；`TransferManager` 内部 bulk ack 和所有对应单元测试。最终全仓确认每个 `.acknowledge(` / `acknowledge_all_terminals(` 调用都有真实 cycle，且 cleanup 不会先删 manager record 再让仍存活 engine job 二次 ack。terminal byte-store/source capture/destination commit 顺序不变。

**阻断门：**任何物理 L2 accepted transaction 缺 acquire/release、ack 早于字节/owner 后处理、拒绝事务造成 underflow、accepted fault 或 cancel-requested leg 在 terminal ack 前释放 extent、旧 generation late completion 覆盖新 owner 的 bytes，都属于安全失败，不得提交。

## 4. Private fault/cancel/reset 与安全恢复闭环

**改动面：**`pipeline_validator/tile_group.py`、`simulator.py`、`runtime/group_port.py`、transfer/engine cleanup；复用现有 reset/fault machinery，不以清空数据结构模拟隔离。

1. fault/reset 一开始停止新的 release/bind/access，并取消尚未准入、尚未 submit 的 private owner 请求。已经准入的 sequencer 按现序关闭 routes、jobs、pins、transfer references；accepted transfer 请求 cancel 后仍走 transport drain / 确认 isolation / terminal acknowledge。只有安全 ack 归还 inflight refs 后，才按“view invalidate → root slack/residual retirement”收尾；调整 `release_context_memory` 次序，避免当前先 invalidate/retire、随后又等待 inflight ack 的环。bulk ack 前先通过 `MFEEngine.retire_isolated` 等现 owner 路径退休 terminal jobs，避免 tile reset 后仍有 engine job 对相同事务二次 ack。
2. 对未 bind / 未 submit 的 private owner 与 admission ticket 作一次安全取消，不留容量、slot、event 或 owner 悬挂。不得把 reset 写成清空 `_l2_handles`、backing registry、allocator exact-live/free map、transactions 或 trace；不得先 free、后检查 late write。wrong owner、stale allocation/profile generation、write-after-release 仍按硬 invariant fault。
3. 在 `Simulator._run_model` 的 success 出口先运行 private 阶段的 L2 闭合检查：无 live private backing、arena slack/reservation、view、pin、accepted transaction，free map 与 per-bank capacity 守恒。现 controller success 分支会直接 break；若闭合检查失败，必须先让 fault 对 CPU device controller 可见（`controller._enter_fault(reason, cycle)`），调用 `note_fault_drain_started` / `_ensure_fault_drain` 并继续 controller + Group + completion harvest，只有 `ResetDomain.DONE` 后才返回 `completed=False` 及具体 backing/claim/transaction ID。只调用 `group.trigger_fault()` 或在 success 已 break 后 raise 都不算 drain。独立/standalone 成功出口保留同一 private 闭合证明，但本批次只核查可达的 private 对象；shared-specific assertions 留给 II。
4. 修正 `_run_model` 的 cycle cap：当前 `while self.cycle < max_cycles` 直接以 incomplete 返回，不能把“达到 `ResetDomain.max_drain_cycles`（它只升级 cancel 请求）”当成 isolation proof。达到 cap 后先调用 `controller._enter_fault`，reason 保留已有终态 fault（否则用 `cycle cap ... reached`），通知 drain 并 `_ensure_fault_drain`；再启动独立、最多 `hw.memory_target.profile_command_timeout_cycles` 拍的 post-cap drain loop。每拍次序保持 `controller.step(cycle)`（faulted 时禁止新 submit）→`group.step(cycle)`→`controller.harvest_completions(cycle)`，直到 ResetDomain DONE 后返回 `completed=False`。额外窗口耗尽、step 遇到安全故障或隔离无法证明时，设置 `TileGroup.poisoned_reason`，保留旧物理 backing/free-map reservation，返回具体未清对象 ID；`begin_launch` / `try_admit_context_task` 拒绝复用。仅显式 `TileGroup.reset()` 在确认 reset drain DONE 且物理状态安全后恢复，不允许成功 run 或重复 reset 擦除 poison/旧 backing。
5. fault/reset 前后的同 generation 生命周期必须有可见证明：无消费者/未 submit 的 private owner 可按内部 cancel 单向终结；重复内部清理不再改容量；第二次成功 run 不得继承上一 run 的 backing、transaction、ByteStore 逻辑域或 trace state。shared manifest/claim 状态、pending consumer 的 DECLARED→CANCELLED、shared terminal leak 演练由批次 II 接入同一 drain，而非在本批次造一个独立通道。

**阻断门：**ResetDomain 未到 DONE 就返回“已清理”、低 `max_cycles` 后只返回 cycle-cap 错误而未跑额外 drain、poison 场景把未隔离 extent 归还或允许 Group 新 run、成功出口 leak 只由日志提示、accepted late write 可碰到新 owner，全部必须视为失败；冻结组和证据后回滚整个不安全批次。

## 5. Physical snapshot、extent trace 和账务守恒

**改动面：**`ArenaPool.snapshot` / `MemoryTrace` / `TileGroup._record_l2_occupancy` 与报告快照。只接受 allocator mutation 后的真实 snapshot，不从 trace 反推计数、不从逻辑 live bytes 伪造物理使用量。

- per-bank `reserved/allocated/free/padding` 统计 unique live padded backings + root-held arena slack；`live_view_bytes` 单独表示有效逻辑 view bytes。各 bank 始终满足 `allocated_bytes + free_bytes == user_spm_per_bank`；物理 cache/system-reserved 独立列出，不混入 user SPM free map。arena rows 中 `reserved_bytes` 是当下真正由其持有的容量，同时保留 admission 时 `initial_reserved_bytes`；root retirement 与 buffer final-free 分开。
- snapshot 新增本批次确实可计算的 `live_backings`，并让完成/zero-leak 检查同时核对它、现有 views/pins/inflight、slack/reserved bytes 和 exact free-map 守恒。`pending_shared_claims`、`active_shared_references` 不是本批次的零值“占位字段”；批次 II 才加入其真实统计与 dormant/active 定义。
- 每次真实 final-free 发 `l2_extent_release` instant：带 `backing_id`、origin buffer/arena、run/allocation/profile generation、released padded bytes、每 bank released segments、release cycle、**post-mutation** `pool_version`。只有真实 final-free 才发；非最后 view invalidate、owner retirement 但尚有 backing、失败 preflight 都不得伪报 extent release。`buffer_view_invalidate` 保留为逻辑事件，携 backing ID 和当前物理状态，不能暗示 free map 变化。
- `arena_lifetime` 仍只代表 root/task 元数据生命周期，不再用它代表恒定容量占用。新增 `l2_backing_lifetime` complete slice，起于 backing 的物理 commit，止于唯一 final-free；trace counter/snapshot 在每个真实 commit/final-free/retirement 后更新。容量积分应从 `reserved bytes` counter 读取，而不是根据 slices 或 event 名估算。

**阻断门：**任何时刻 bank 守恒不成立、snapshot 与 free map/exact units 不一致、buffer 释放后 arena slack 变零、普通 view invalidate 发出 physical-free、per-buffer padding 未计或重复计、trace release cycle/version 早于实际 allocator mutation，都不得通过验证。

## 6. 有顺序的实现/回归门与必需证据

只在上一门通过后进入下一门；每次都在同一工作副本检查当前实现，测试建议保持 deterministic、断言消费者可见字节/容量/时序，不锁内部字段的无意义复制。

1. **Layout/verifier：**`pipeline_validator/tests/test_compiler.py` 与 `pipeline_validator/tests/test_validator.py` 覆盖 no-rebind padded spans、合同上限及编译产物；新增/更新 compiled DTO 篡改测试，证明 independent `execution_verifier` 对 overlap、错边界、重复/缺失 bind/release 拒绝，不能只测 parser/source compiler。回归中保留 L1 lifetime / `tile.free` 复用合同。
2. **Allocator/free map：**`pipeline_validator/tests/test_profiles.py` 新增（或扩展现有 ArenaPool 局部视图用例）非整 stripe 尾部 padding 随 buffer 一起归还、exact-unit split + release 后可按原始容量重新分配、slack 保留到 root retirement。每项同时读 before/after snapshot 与 per-bank free intervals；验证无重叠和 total free/allocated 守恒，不能只 assert 私有字段/版本号。现有 `test_local_view_invalidation_never_changes_global_free_map` 是 **L1 不变**边界，必须保留通过。
3. **Bytes/owner/transaction：**`pipeline_validator/tests/test_memory_invariants.py` 用 exact allocator 与真实 transaction 覆盖完整 padded segment 原子 commit/free、owner/stale/double-release 拒绝与无副作用；`pipeline_validator/tests/test_profile_runtime.py` 的 ByteStore + real transaction 测 source/destination 实字节、accepted-but-not-issued prefetch、cancel-requested leg、FAULTED but unacknowledged、最终 destination write 尚未 commit，以及 ACK 后才能重用 backing。覆盖跨 issuer 且同一 private handle 的 source/destination reference；preflight 失败后旧字节、pins/inflight、free map 均不变，late old-generation completion 不能覆盖新 owner。不要以“`ack` 被调用一次”作替代证明。
4. **Root admission：**将 `pipeline_validator/tests/test_runtime.py::TestRootArenaAdmission.test_t09_t10_t14_pending_root_owns_no_slot_or_arena_until_prior_root_retires` 改名为 `test_pending_root_admits_on_input_extent_release`，`runtime` / `full_memory` 参数化。等待期间 B 只有 pending request，不持 active slot、L2 arena/backing 或 partial event reserve；释放后同拍准入，`active_peak == 2`。严格断言 B **port** `active_cycle == a_input 的 l2_extent_release cycle < A context completion`；device launch `admission_cycle` 不可代替 port `active_cycle`。用真实 transaction trace 确认 B `op=prefetch` 的 HBM→L2 transaction 首 leg 至末 leg时间窗与 A `EVU:pow` span 交集 `> 0`；不能用计划事件、仅 `uce_issue`、cycle 数推算或 `context_admitted` 替代传输证据。

   当前 `examples/scenarios/l2_admission_wait.mlir` 的注释声称 runtime-only 会让 A/B 同时 admission，而原始计划要求同一容量合同和此 release-driven proof 在 `runtime` / `full_memory` 两档都成立。实现时先用测试暴露并解决此合同冲突（让真实 profile/fixture 在两档都覆盖可观测容量等待，随后修正文档中的陈旧注释）；不得因 runtime 难测就把 equality / overlap 改成只测 `admission_cycle`、conditional skip 或空断言。若无法证明，批次阻断，不以 plausible trace 交付。

5. **分类/FIFO/L1 regression：**在 `test_profiles.py`、`test_memory_invariants.py` 与 `test_runtime.py` 覆盖 PERMANENT_CAPACITY、WAIT_CAPACITY、WAIT_FRAGMENTATION 的实际边界及无副作用；同类 queue 队头不能被越过，release 当拍的 capacity/version 变化唤醒队头而非周期 retry。可回收容量不足与总空闲字节足够但连续 aligned span 不够必须分开。`test_local_view_invalidation_never_changes_global_free_map` 和现有 L1 profile / allocator tests 继续通过，证明 L2 改动未使 L1 release/free map 改义。
6. **fault/reset/cap/trace：**`pipeline_validator/tests/test_runtime.py`（现有 `TestFaultReset`、`TestL2AccessRelease`、`TestRootArenaAdmission`）、`test_memory_invariants.py`、`test_profile_runtime.py` 与 `pipeline_validator/tests/test_trace.py` 覆盖一次公共 release、重复内部 cancel、safe reset、低 cycle cap 额外 drain 到 DONE、无法隔离时 poison + 保留 physical units + 拒绝新 run；校验 exact `l2_extent_release` args 与物理 counters 在 post-mutation 时间点一致，成功输出 zero private live backing/slack/pins/inflight/transaction。验证旧 L2 full-Arena 与 L1 trace 未被冒充为 buffer free。

固定私有回归命令：

```bash
conda run -n elenor-validator python -m pytest \
  pipeline_validator/tests/test_profiles.py \
  pipeline_validator/tests/test_memory_invariants.py \
  pipeline_validator/tests/test_runtime.py -q

conda run -n elenor-validator python -m pytest \
  pipeline_validator/tests/test_profile_runtime.py \
  pipeline_validator/tests/test_trace.py \
  pipeline_validator/tests/test_compiler.py \
  pipeline_validator/tests/test_validator.py -q
```

若有任意私有 regression fail，先冻结失败 trace/report/source hash，不能把失败 test 注释掉或只跑新窄测试；修到根因后重跑上列两组。Shared claim tests、profile-switch/shared epoch、shared counts 与跨 context late reader 由批次 II / III 做，不在本批次添加假数据或 zero-claim echo。

## 7. CLI 真正运行与端到端 acceptance

在批次 1–6 全部通过且变更已作为安全整体落地后，才用当前 fixture 做新实现的 smoke；这不是修改前重跑旧 failure。命令从仓库根目录执行，`run.sh` 已为该 scenario 配好 exact profile、A/B bindings、context/device capacity、DMA channels、HBM latency、cycle cap：

Smoke 前保存本次 fixture、config、runner 与实现源文件的 hash，并连同 reports 中的 `compiled_artifact_hash` / `registry_hash` 一起保留：

```bash
set -euo pipefail
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="examples/artifacts/l2-sharing-release/$RUN_ID/private-smoke"
mkdir -p "$OUT"
sha256sum examples/run.sh examples/scenarios/l2_admission_wait.mlir \
  examples/configs/profile_l2_256k.yaml \
  pipeline_validator/compiler/resources.py \
  pipeline_validator/execution_verifier.py \
  pipeline_validator/memory/arena.py \
  pipeline_validator/memory/allocator.py \
  pipeline_validator/memory/byte_store.py \
  pipeline_validator/tile_group.py \
  pipeline_validator/runtime/group_port.py \
  pipeline_validator/memory/transfer.py pipeline_validator/engines.py \
  pipeline_validator/memory/profile_controller.py \
  pipeline_validator/simulator.py pipeline_validator/trace.py \
  > "$OUT/source-config.sha256"
```

```bash
set -euo pipefail
OUT="examples/artifacts/l2-sharing-release/${RUN_ID:?set-run-id-before-smoke}/private-smoke"
mkdir -p "$OUT"
for fidelity in runtime full_memory; do
  bash examples/run.sh l2-admission-wait \
    --sim-override fidelity="$fidelity" --memory-trace \
    --trace-json "$OUT/l2-admission-wait.$fidelity.trace.json" \
    --report "$OUT/l2-admission-wait.$fidelity.report.json" --json \
    > "$OUT/l2-admission-wait.$fidelity.stdout" \
    2> "$OUT/l2-admission-wait.$fidelity.stderr"
done
```

Smoke 证据必须在 raw report/trace 中闭合：result completed；`device.port.request_records` 的 ctx_b `active_cycle` 等于同 run/同 backing 的 `l2_extent_release.release_cycle`，严格早于 ctx_a `completion_cycle`；ctx_b 在此前只有 pending，不占 slot/arena；`active_peak == 2`。使用 trace 中真实 B HBM→L2 `prefetch` transaction 的跨腿 span 与 `TileN` lane 上 `EVU:pow` 的相交 cycles，严格大于 0；同时确认 A 的 output store/context 尚未完成。不能用 device `admission_cycle` 代替 port active cycle，也不能只引用某条旧 trace、SAMPLE interval 或报告里的预期检查文字。`l2_extent_release` 的 cycle、`pool_version`、per-bank segments 必须与 post-mutation snapshot/free map 对齐。

当前 source fixture comments 中的 runtime 行为若与上条冻结 acceptance 冲突，必须先按第 6 项解决、更新过时注释并证明两档合同一致；smoke fail 不可解释成“场景注释本来就允许”。修改前已报告的 failure 不重放；以上仅在完整实现后的新版本运行，并保留新 hash/退出码/完整 stdout+stderr 和 trace/report。

## 8. 原子提交与不可越过的批次边界

本批次只有在下列证据都齐全后才允许**一次性 commit/deliver**：

1. 修改前全 runnable-workload 双 fidelity baseline、源/config hash 和 root wait metrics 完整；
2. 静态无重叠编译布局、独立 DTO verifier 与 L1 不变证明通过；
3. exact padded backing/slack 单位守恒，private owner/ByteStore/readback 与 release-driven FIFO admission 正确；
4. 每条持有 L2 的 accepted transfer 在真实 terminal acknowledgement 前保活；取消、fault、reset、cap drain、poison 与安全 recovery 均通过；
5. snapshot 与 trace 反映真实物理 occupancy；所有指定 targeted tests 和两档实际 CLI smoke 的 active-cycle / input extent release equality / real prefetch-vs-pow overlap 通过。

任一项未闭合：冻结可复现 evidence，**不得提交/交付任何剥离 root capacity、提前返回物理 extent、漏 transfer hook、没有 cancel/reset drain 或靠 trace/计数假安全的中间版本**；整批修复或完整回滚。不得用“剩余功能下个批次再接”把本批次留成不安全状态。批次 II 只在此 batch I atomic boundary 后，把 claims 加到同一 backing finalizer；批次 III 之后再跑全部 CLI 双档场景与 Perfetto/全量回归。
