# 批次 III：场景与端到端验收

> 本文是 `L2_显式共享_PLAN.md` 第六步第 4–7 条、第七步和第八步第 3–5 条的可执行计划；仅规定实施顺序、可运行场景和验收证据，不代表拟议语法已经实现。共享语法与 runtime 合同实现前，文中的 `sharing`、`nest.publish`、`nexus.shared.ref` 示例均不可解析、不可运行。
>
> 相关计划：批次 I [私有 L2 提前释放](./01_private_l2_release.md)、批次 II [显式只读 L2 共享与生命周期](./02_explicit_l2_sharing.md)、[原始联合计划](../design/proposal/L2_显式共享_PLAN.md)。计划目录总览由 `plan/README.md` 维护。

## 0. 开始条件与不可变边界

本批次是场景验收批次，必须在批次 I、II 的实现和各自安全门全部通过之后开始运行端到端场景。场景文件可按冻结的 IR 合同准备，但在前两批未完成时不得把它们当作可执行验收，也不得以跳过等待、关闭 late-write 检查、只改 trace 或只缩短 root 生命周期的方式得到通过结果。跨层安全状态不完整时，不发布“场景通过”的部分交付。

开始前逐项确认：

- [ ] 批次 I 已证明 L2 私有 buffer 的 padded、stripe-rounded span 永久 forfeiture；同一 backing 的物理记账和 final-free 唯一；accepted transfer 引用直到真实 terminal acknowledgement 才排空；fault/reset、cancel、late access 不会提前归还或 double-free；真实同 profile 容量释放可唤醒 FIFO 队头。
- [ ] 批次 II 已证明封闭的 shared claim manifest、producer completion readiness、只读 view、最后安全 borrower final-free、producer root 可先退休而 backing 可存活、跨 L2 profile epoch 被拒绝；fault/reset 安全 drain 和 run 间闭合也已通过。
- [ ] 两批的 `IR_SPEC.md` 行为合同与本批 profile-switch/shared fixtures 使用的语法一致；实现未改变 HBM striping、L1 Task Arena/tile.free 合同、device-slot/context 配置语义。
- [ ] 已按批次 I 的先决要求取得真实 workload 容量等待 baseline；不在本批伪造或补写测量值，也不把缺少直接收益的 workload 改成优先级决策。此前已报告的旧 `l2-admission-wait` 失败作为已知基线，不为确认而重跑；本批只运行前两批安全门完成后的 fresh acceptance。

本批保持以下边界：L2 默认 private；共享只读、限同一 model invocation / Tile Group / L2 profile generation；消费者以 producer context 成功完成作为 readiness；L1-only profile change 不销毁 L2 sharing；L2 切走再切回相同 mode 仍是新 generation；不改 HBM striping、L1 复用和设备槽位；不增加隐式 spill/reload 或任意 shared read-write。即使 trace 结果良好，也不得声称本次恢复了 `matmul-pow-free-slot` 或 `matmul17-pow-tail-overlap` 的引擎重叠。

## 1. 场景创建：先冻结配置与 IR

场景文件、`run.sh` 注册和回归测试是不同交付面。每个场景必须有完整 `builtin.module`、可解析 IR、resource contracts、task ranges、dispatch 的完整 `signal_policy`、正确 global/L2 views 与输出清理；草图或 session-local 示例不是可接受替代。优先复用 `examples/scenarios/l2_admission_wait.mlir` 的 programs、tensor shapes、bindings、pow 参数和事件结构，避免建立第二种惯例。

### 1.1 Profile-switch 的两档 YAML

新建 `examples/configs/profile_l2_256k_switch.yaml`，以当前 `examples/configs/profile_l2_256k.yaml` 为基础；以下字段和数值必须保持精确：

```yaml
schema_version: 2
memory:
  target:
    profile_command_timeout_cycles: 2000000
    l2:
      system_reserved_spm_per_bank: 4096
      reset_mode: 0
      spm_mapping_id: striped_arena_v0
      cache_org_id: profiled_lru_v0
      cache_write_policy: read_only
      maintenance_caps: [invalidate_range, clean_invalidate_all, bypass]
  profile_source:
    l2:
      modes:
        0: { spm_bytes_per_bank: 20480, cache_bytes_per_bank: 0 }
        1: { spm_bytes_per_bank: 12288, cache_bytes_per_bank: 8192 }
  group_sram:
    capacity_bytes: 327680
```

约束与核对：

- 保留 16 banks、每 bank 4096 B system reserve、327680 B physical group SRAM、只读 cache、原 maintenance caps、默认继承的 L1 配置；mode 0/1 的每 bank `SPM + cache` 都是 20480 B，不改变硬件总量。
- mode 0 user L2 = `(20480 - 4096) × 16 = 262144 B`；mode 1 user L2 = `(12288 - 4096) × 16 = 131072 B`。逐 bank 核对容量不变量，cache 与 system-reserved 不可误算作 user SPM。
- 此 YAML 是新 profile-switch fixture 专用；原 `profile_l2_256k.yaml` 继续是单 mode0 的 admission-wait / shared fixtures 配置。不得为了让切档“成功”搬迁 live backing 或在 mode1 沿用 mode0 地址。

### 1.2 不同 L2 profile 的完整 quiescence 例

新建 `examples/scenarios/l2_admission_profile_switch.mlir`，沿用 `l2_admission_wait.mlir` 的 A/B tensor shapes、programs、HBM views/bindings、`pow_ops` 和 A 的真实 HBM store。A 使用 `l2_mode=0, allowed_profiles=[0]`，B 使用 `l2_mode=1, allowed_profiles=[1]`；两者 L1 均为 mode 0。保留各自完整 `logical_tasks`、`l2_spm_bytes`、`requested_contexts_per_tile` resource contract，不把 profile 属性藏在注释或仅依赖 runner 默认值。

`nexus.program` 连续提交 A、B，不手写 `%done_a` 的 `nexus.await`，也不在 source IR 人工插入 profile 命令。compiler 的 `bind_profiles` 必须基于完整 A root completion history，自动生成 device await 和 profile reconfiguration；buffer input release、extent final-free、publish 或 output-ready 均不得替代该 quiescence frontier。运行 trace 要能看到完整因果链：A HBM store terminal → A context/root completion → profile command/OPEN_ISSUE 完成 → B admit → B 首次 prefetch。

完整模块中的 A/B contracts 与 model 提交顺序如下（仅为 contract/fence 位置摘录，最终 fixture 必须包含完整 context/program/body，不可直接把省略体当作可运行文件）：

```mlir
// ctx_a resource_contract
resource_contract = #nest.context_resources<
  l2_mode = 0, allowed_profiles = [0], logical_tasks = 4,
  l2_spm_bytes = 262144, requested_contexts_per_tile = 1>

// ctx_b resource_contract
resource_contract = #nest.context_resources<
  l2_mode = 1, allowed_profiles = [1], logical_tasks = 4,
  l2_spm_bytes = 131072, requested_contexts_per_tile = 1>

// nexus.program: 连续 submit；不得在两行间插入 nexus.await
%done_a = nexus.submit_context.async @ctx_a(%A_IN, %A_OUT)
  : !nexus.event<"done_a">
%done_b = nexus.submit_context.async @ctx_b(%B_IN)
  : !nexus.event<"done_b">
```

在 `examples/run.sh` 增加 `l2-admission-profile-switch`，保留已有 `l2-admission-wait` 的硬件覆盖和三份 bindings，只换 IR/YAML 路径；fidelity 由额外 CLI 参数选择，不在入口写死：

```bash
bash examples/run.sh l2-admission-profile-switch \
  --sim-override fidelity=full_memory \
  --memory-trace --trace-json OUT/profile-switch.trace.json \
  --report OUT/profile-switch.report.json --json
```

入口参数必须包含 `--context-mode 2 --device-context-mode 2`、`--hw-config examples/configs/profile_l2_256k_switch.yaml`、`--hw-override num_dma_channels=2`、`--hw-override hbm_fixed_latency_cycles=10`、`--max-cycles 500000` 及精确三份 binding：`A_IN=0x100000:131072:rw`、`A_OUT=0x200000:131072:rw`、`B_IN=0x300000:131072:rw`。runner 不可因 extra args 顺序或 wrapper 丢弃这些选项。

### 1.3 跨 context 共享 weight：完整词法传递链

新建 `examples/scenarios/l2_shared_weight.mlir` 并注册 `l2-shared-weight`。loader 对 `[64,64] bf16` 的 W 做唯一一次 HBM→L2 prefetch（8192 B），完成 publish 后撤销自己的逻辑 view 并返回；两个独立 reader context 各 dispatch 四个 tiles，从同一物理 backing 读取 W。每个 reader 仅有自己的 `[4,64,64] bf16` 输出（32768 B），最终分别写入 B_OUT、C_OUT。

正式模块必须让只读 capability 沿现有词法/绑定边界逐层传递，不能越过 region 直接引用外层 SSA：

1. `nexus.program` 中 `%loaded = nexus.submit_context.async @load_weight(%W_IN)`，随后 `%shared_w = nexus.shared.ref %loaded slot = "W" : !nest.l2_buffer<64x64xbf16>`。
2. `nexus.program` 将 `%shared_w` 作为 actual 传给 `@reader` 的每次 submit；submit 保留 `%loaded` producer completion dependency。
3. `@reader` context 声明独立 formal，例如 `%weight: !nest.l2_buffer<64x64xbf16>`；context body 只使用 `%weight`，将它放入 dispatch `bindings(...) ins(%weight)`。
4. 独立的 `tile.program @copy_weight` 再声明自己的 L2 formal `%w: !nest.l2_buffer<64x64xbf16>`；从 `%w` 建 `tile.subview` 并执行真实 L2→L1 load。tile program 内严禁引用 `nexus.program` 的 `%shared_w`，也不能把共享 actual 当作 tile operand 越过 context/dispatch formals。

共享输入不计入 reader 自己的 `l2_spm_bytes`；loader local resource=8192 B，each reader local output reserve=32768 B。weight view 是 input-only：不可用于 prefetch/store destination、dispatch `outs`、二次 publish 或转借 export。两 reader 的 L2→L1 reads 仍都要发生；省掉的只有第二份 HBM→L2 W prefetch。

### 1.4 L2 中间结果 fanout：不得制造 X 的 HBM round trip

新建 `examples/scenarios/l2_shared_fanout.mlir` 并注册 `l2-shared-fanout`。producer A 只接收 HBM A_IN（8192 B），prefetch 后通过真实 tile.load→tile.store 在 L2 创建 X（`shape = [64,64]`、`dtype = "bf16"`、`alignment = 256`，共 8192 B），待所有 writer completion/output_ready 后 publish X；A 不对 X 发 HBM store。B/C 以相同的只读 context-formal → dispatch `bindings/ins` → tile-program L2 formal 链读取 X，分别写自己的 32768 B B_OUT/C_OUT。

文件和 runner 必须都没有 X 的 HBM global formal、binding 或 address；trace/report 不得出现以 X 为来源/目的的 GLOBAL_STORE、HBM PREFETCH 或其他中转 transaction。producer 本地资源 reserve=16384 B（A_IN + X），B/C 各自本地输出 reserve=32768 B，共享 X 不重复算进消费者 local reserve。fanout 源 IR 在 `nexus.program` 以 producer completion 建共享引用并把它传给两次 reader submit；不以 publish event 偷换成功完成依赖。

### 1.5 两类共享例共同 runner 合同与 private-weight 对照

两例都使用现有单 mode `examples/configs/profile_l2_256k.yaml`，固定 L2/L1 mode0；入口固定 `--hw-config examples/configs/profile_l2_256k.yaml --context-mode 2 --device-context-mode 2 --hw-override num_dma_channels=2 --hw-override hbm_fixed_latency_cycles=10 --max-cycles 500000`。入口参数名按各自 module 传递三份 8192/32768 B bindings：

- `l2-shared-weight`：`W=0x100000:8192:r`、`B_OUT=0x200000:32768:w`、`C_OUT=0x300000:32768:w`。
- `l2-shared-fanout`：`A_IN=0x100000:8192:r`、`B_OUT=0x200000:32768:w`、`C_OUT=0x300000:32768:w`。

入口均接受 `--sim-override fidelity=runtime|full_memory`，并接受 `--memory-trace --trace-json ... --report ... --json`。共享场景必须经 `bash examples/run.sh <name>` 正常执行，不能只存在 parser fixture 或单元测试中。

另准备同一 tensor shapes 和 reader tile programs 的 private-weight comparison fixture：唯一变量是 W 由两个 consumer 各自独立 prefetch/持有；它是测量对照，不新增运行时模式，不替代上述四个指定的 focused scenario。比较只对 W 的 HBM→L2 输入流量断言：private = 2 × 8192 = 16384 B，shared = 8192 B；不把 B/C output write、A_IN 读取或每个 tile 的 L2→L1 load 算成节省的 W HBM 流量。记录 fixture/执行命令/输入 hash，使对照可以复现；若作为 run.sh 可列场景注册，则另计入动态 corpus，不得改变四个 focused scenario 的定义。

### 1.6 late-C 持有 backing 的确定性示例

共享重量与 fanout 例必须至少有一个可重复的 reader 延迟路径：延迟 C 的首次 read，或限制 slot 令 C 在 B release 后才准入。用固定 W 字节样本 `bytes(range(256)) * 32`（8192 B）与真实 ByteStore/transaction 读回作为数值 oracle；B/C 的各四 tile 输出必须逐字节等于预期四份 W。fanout 对 X 也用真实 tile.load/store 和读回数据证明内容，不以 timing-only pow/EVU 作为数值正确性证据。

验收顺序是 producer 完成/退休、B 释放时 C 仍持有其 `DECLARED`/`BOUND` claim，故 backing 与容量继续存在且 C 能读取原内容；只有 C 最后安全 release、pins 和 accepted transfers 均排空后，才对该 backing 发出唯一一次 physical `l2_extent_release`。producer root retirement 和 B 的逻辑 release 都不能伪造 final-free；共享 aliases 不得重复计物理容量。只观察内部计数器不够，须同时检查字节读回、backing_id 生命周期与 per-bank free-map。

## 2. 端到端运行：focused scenarios 与动态 corpus 分开留证

### 2.1 每次 run 的不可变 artifact

本次产物统一放在 `examples/artifacts/l2-sharing-release/<run-id>/`，按 focused/corpus、scenario 名称、fidelity 分目录；每条命令有单独输出，避免覆盖。

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="examples/artifacts/l2-sharing-release/$RUN_ID"
mkdir -p "$OUT"
```

每次 run 保存：源 IR 与 YAML SHA-256、受影响 compiler/runtime/test/doc 源文件和 `examples/run.sh` 的 SHA-256、完整 argv/命令文本、工作目录与 fidelity、stdout/stderr、exit status、JSON report、原始 trace JSON、memory report、Perfetto SQL 与结果、`stats`、raw JSON/Perfetto 完整性 audit、独立 extractor 输出和最终 acceptance summary。通过实际编译/序列化路径记录与本次 run 对应的 compiled/executable package hash；若 CLI 不直接输出 compiled bytes，使用编译 API 提取并核对相同输入与选项，不能省略或推测 hash。保存当前 `bash examples/run.sh list` 原文及解析得到的 scenario 清单；不得只保存场景总数。Hash 覆盖输入文件内容而不是文件名，源变化后必须新开 run-id；旧 trace/report 仅作历史，不得作为本次证据或拼接新旧数据。

实施者可用一个 throwaway Python runner 依序调用 `bash examples/run.sh`，放在本次 artifacts 目录而非产品源码；逐次 `subprocess.run` 保存返回码/stdout/stderr 和精确 argv，不丢弃额外 flags。运行完保留可审计的命令清单和全部证据，清除 runner 的临时文件/缓存，不把临时执行框架变成新的维护 API。仅在仓库内恢复缺失的 `examples/artifacts/tools/trace_processor_shell`；禁止向项目外安装工具。

### 2.2 四个 focused 场景 × 两种 fidelity

必须显式执行以下四个名字各 runtime、full_memory 一次，共 8 次 focused runs；不得用 full corpus 的同名记录取代 focused 记录：

```text
l2-admission-wait
l2-admission-profile-switch
l2-shared-weight
l2-shared-fanout
```

命令形态（每个 name/fidelity 使用独立目录及 trace/report 文件名）：

```bash
bash examples/run.sh <name> \
  --sim-override fidelity=<runtime|full_memory> \
  --memory-trace \
  --trace-json <out>/<name>.trace.json \
  --report <out>/<name>.report.json \
  --json
```

`--memory-trace` 必须存在，才能审计 L2 backing/extent/memory lanes；`--json` 必须显式出现。保存 stdout/stderr 和退出码；不要以 shell wrapper 丢弃 `"$@"`。两个 fidelity 都使用相同容量/正确性合同，不能保留或复制 `l2_admission_wait.mlir` 中“runtime 档不会 wait”之类陈旧注释/假设。因 fidelity 延迟模型不同，不对两档周期数设相等要求；按同一档内的因果顺序及数据正确性验收。

### 2.3 由 `run.sh list` 动态枚举所有 runnable corpus

在 focused runs 后重新执行并归档：

```bash
bash examples/run.sh list
```

throwaway runner 从这次真实输出动态取得 Runnable workloads、Protocol scenarios、NEST subgraphs 的全部合法入口名；按打印出的名字逐项执行 runtime 与 full_memory 双档，并使用与 focused runs 相同的 `--memory-trace --trace-json --report --json` 证据参数。NEST 名必须采用 list 显示的 `nest-<stem 下划线转连字符>` 入口名；不能拿 `.vscode/run_all_workload.sh`、目录 glob 的局部样本或既往清单替代。不得写死“136 个”或其他 corpus count；核对的是 list 本身的完整消费、每个名字两档都有结果。Corpus run 单独分目录，不覆盖 focused artifacts。

若 `list` 中存在已有测试合同规定的预期失败输入，只按该合同标记预期结果并保留完整退出证据；不得为提高通过率把它改成成功。除此以外的任意失败都必须定位修复后从新 run-id 重跑，不得静默跳过、降级为 expected failure 或引用旧 trace。

## 3. Perfetto、因果关系与数据完整性验收

### 3.1 提取与单位

用项目内 `examples/artifacts/tools/trace_processor_shell`，通过 subprocess 将 SQL 经 stdin 送入，保存 SQL、stdout、stderr、返回码和 trace hash；典型调用：

```python
subprocess.run(
    [trace_processor_shell, trace_path, "-q", "/dev/stdin"],
    input=sql, text=True, capture_output=True, check=False,
)
```

在默认 1 GHz 配置下，Perfetto 的 `ts`/`dur` 数值可直接按 cycles 解读；raw JSON 的 `ts`/`dur` 必须按该 run 的 `cycle_ns()` 还原。若时钟配置变更，统一用 `cycles = Perfetto_ns / cycle_ns()`，不得固定套用 1000 倍或混用 JSON/SQL 单位。每份报告记录 `cycle_ns()`、clock 配置和换算路径。

SQL 至少提取物理与逻辑生命周期、admission、profile、真实传输及 EVU 计算窗：

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

按已有 transfer summary 的 `transaction_id/op/source/destination` 去重还原传输；不要把一个 transaction 的多个 leg 合计成多次 prefetch。用事件 args 区分 A/B、producer/reader 和 buffer；计算 overlap 要对完整区间集合求交集，不能只比较最早/最晚事件。

### 3.2 每条 trace 的 parser 完整性和独立交叉审计

对每份 raw trace JSON 解析原始 event counts：X complete slices、instant（`ph=i`）、B/E 区间（同时检查配对和未配对端点）、counter、flow；与 Perfetto `slice`、`counter`、`flow` 对应表的数量及可比较字段逐类对账，并用独立 Python JSON extractor 与 Perfetto 的关键事件/参数/周期结果交叉比较。将原始 JSON counts、SQL counts、差异及 parser 状态全部写入 artifacts；关键时序/共享流量/生命周期指标必须零差异。

必须读取并报告 `stats` 中 `severity IN ('error','data_loss')` 且 `value != 0` 的每一项，特别记录 `json_parser_failure`。出现非零 parser failure 时不能静默称 trace “完整”或“干净”：定位损失事件类别，核实是否影响本验收所需的 X/instant/B-E/counter/flow 数据；存在任何必需事件缺失即该 trace 验收失败、修复生成/解析路径并重新运行。即使已知旧工具可能只损 legacy flow，本次也必须对本次 trace 重做 JSON-vs-SQL counts 和独立 extractor 比较，不从历史审计继承结论。

每 bank 容量对账使用真实 post-mutation pool snapshot：`allocated_bytes + free_bytes == user_spm_per_bank`，system-reserved/cache 单列；共享 aliases 不重复计物理 `reserved/allocated`。同时核对 `live_backings`、pending claims、active references、pins/inflight、arena slack、generation、per-backing commit→final-free 区间；`arena_lifetime` 仅是 root/task metadata 生命周期，不能代替 `l2_backing_lifetime` 或从 view 数量反推物理容量。
每次真正 final-free 都须有唯一 `l2_extent_release` instant，并核对 `backing_id`、origin buffer/arena、run/profile generation、`released_bytes`、每 bank released segments、release cycle、post-mutation `pool_version`；对应的逻辑 `buffer_view_invalidate` 带 backing ID 与剩余 claim 数，非最后 borrower release 不产生 physical-free 事件。`l2_backing_lifetime` complete slice 覆盖 physical commit→final-free；`arena_lifetime` 只覆盖 root/task 元数据。occupancy 积分取 `reserved_bytes` counter，MemoryTrace/report 来自真实 `_pool_snapshot`，不能由 trace 反推计数。

### 3.3 必须成立的可观察时序与流量断言

以下每项均逐档从新生成 trace/report 和相关 byte-level 回归证据验证；用具体事件 cycle/assertion 结果写入 summary，不在计划阶段填入未经测量的值。

**同 profile 提前准入（`l2-admission-wait`，runtime 与 full_memory 各自成立）：**

- A 的 `a_input` 只有在 prefetch 与全部 tile `input_released` 后合法 release；其唯一 physical `l2_extent_release` 对应真实 padded extent 释放。
- B 的 GroupPort `active_cycle` 与 A `a_input` extent final-free cycle 相同；该时刻早于 A context completion 和 `EVU:pow` 完成。不要用 device `admission_cycle` 冒充 port `active_cycle`。
- B 等待期间未持有 slot、arena 或部分 L2 allocations；释放同拍后 FIFO 队头 admission 成功，B 首个 group action 最早在下一 cycle 发起（可能受依赖/传输延迟继续推迟）。报告 `active_peak` 并依源计划核对值为 2。
- B 的真实 HBM→L2 prefetch 与 A `EVU:pow` 的完整执行 interval 有非零交集；不能仅以“B admit 时 A pow 尚未结束”代替真实数据搬运/计算 overlap。

**不同 L2 profile 完整切档（`l2-admission-profile-switch`，两档各自成立）：**

`A HBM store terminal <= A root/context completion <= profile command start < profile command completion/OPEN_ISSUE <= B admission < B first prefetch start`。确认 compiler 生成的是 A 完整 root history await；B 在 profile command 完成前无 prefetch/load；B 不在 A `EVU:pow` 执行窗内 load；最终 profile 为 mode1 且旧 L2 epoch 无 live views/backings/claims。`COMMIT` stage 本身不是开放 B 的证据，必须使用 command 完成/`OPEN_ISSUE` 与实际 B action。

**共享 weight / fanout：**

- W 场景每次成功 invocation 对 W 只有一次真实 HBM prefetch、8192 B；private comparison 是 16384 B。B/C 各自仍执行四 tile 的 L2→L1 W loads，不能把它们报告成已消除的 HBM transaction。
- A→B/C fanout 中 X 只有一次 L2 producer 数据写；没有 X 的 HBM binding、HBM PREFETCH 或 GLOBAL_STORE。实际 tile store/read 的结果逐字节正确。
- B release 和 producer context/root retirement 后，C 尚未完成时 W/X 对应同一个 backing_id 仍 live、物理容量仍被占用，C 的 late read 数据仍正确；只有 C 最后一次合法 release 且所有 pins/accepted transfer refs 排空，才出现该 backing 唯一一次 physical `l2_extent_release`。非最后 reader 的逻辑 release 只记 `buffer_view_invalidate`，不得发 physical free。
- 共享容量峰值按唯一 backing 计，W/X 不因 borrower 个数重复扣容量；最终成功 run 中 live backing、claims、views、pins、inflight、arena slack 与 routes/leases 均清零，且 report/trace 汇总与 snapshot 一致。

## 4. 回归、规范同步与清理

### 4.1 按原计划跑完全部 targeted 与 full regression

以下命令在仓库根目录、`elenor-validator` 环境执行；批次 I/II 实现已落地后按顺序完整执行，不因本批主要是例子而省略 allocator、compiler/loader、profile、trace 和 memory invariants：

```bash
conda run -n elenor-validator pre-commit run -a

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

除 pytest/pre-commit 外，还要执行上述四个 focused × 双 fidelity CLI runs、动态完整 workload/protocol/NEST corpus 双档 runs、Perfetto stats/JSON/SQL/独立 extractor cross-audit 和实际 output byte comparison；测试通过不能替代真实场景与 trace 证明。记录真实命令和返回结果，不填写预期或推测性通过值。

### 4.2 `IR_SPEC.md` 与跨批次文档合同

`IR_SPEC.md` 的更新随实现分批进行：批次 I 修改 §3.8 与 §8 中 L2 release/free-map/forfeiture 合同，同时保留 L1 原合同；批次 II 扩充 §7–§9 的 local reservation 与只读 shared reference、publish 后不可变并替代中间结果 HBM store、producer completion readiness、L2 profile generation 边界及 L1-only switch、accepted transfer/fault/reset/late completion。批次 III 的集成者逐条核对并补齐遗漏，不能等到本批才修正先前批次已经改变的行为规范；不能把 L2 更新误写为 device-wide/general allocator 行为。

同步核对批次 I、II 文档与 IR_SPEC 同一字段/API/事件/失败合同，解决矛盾而非另造术语或兼容路径。按父任务要求在 `plan/README.md` 登记本页与两个兄弟计划及依赖关系（由 README 所有者修改）。清理 `examples/scenarios/l2_admission_wait.mlir` 中声称 runtime 不会 wait、需要 full_memory 才触发容量等待的陈旧注释，并搜索和修正由该旧前提产生的 stale comments；不得保留与 runtime/full_memory 同容量合同相冲突的说明。

### 4.3 收尾

- [ ] 删除 throwaway runner 临时文件、临时 SQL 输出、解析缓存；保留可复核的 run-id 输入 hashes、命令清单、reports、traces、stats、SQL/JSON cross-audit 和 summary。
- [ ] 不删除已有 artifacts、用户数据或 sibling 文件；本批只应修改/新增本计划所列场景/config/runner入口、规范/注释及所需验收代码，范围外改动先停止并协调。
- [ ] 从 fresh run 确认 focused 场景、dynamic corpus、per-bank capacity、parser完整性、output bytes、last-reader final-free、profile fence、所有测试与 pre-commit 均符合合同；任何重跑使用新的 run-id 和输入 hash。
- [ ] 更新 `plan/README.md` 的依赖/链接；该文件由父任务所有者维护，不在本批子计划中抢改。

## 5. 批次 III 完成标准

只有全部成立才可标记完成：

1. profile-switch YAML 与 IR 确实表示两档 16-bank user L2 256 KiB/128 KiB，compile/load 自动实施完整 A root quiescence，B 不能越过 profile command completion。
2. 同 profile、切 profile、shared-weight、shared-fanout 四个 CLI 场景均可由 `run.sh` 执行，两种 fidelity 及真实输出/因果 trace 均通过；共享 examples 的 formal 链完整、没有跨词法作用域 SSA 或 phantom HBM X。
3. W shared/private HBM 读取分别为 8192/16384 B，且只对 W 输入流量作此比较；每 tile L2→L1 reads 仍存在。Late-C 真实数据可读直到其安全 release，之后才唯一 final-free。
4. 四个 focused scenarios × 两 fidelity 全部单独留证；`run.sh list` 当次列出的每个 workload/protocol/NEST 入口均动态执行 runtime/full_memory，无硬编码 corpus 数量、无旧 trace 复用、无未分类失败。
5. 每份 trace 的 Perfetto stats、raw JSON X/instant/B-E/counter/flow vs SQL counts、独立 JSON extractor 均有结果；parser_failure 不被隐去，必需数据完整且关键指标交叉一致；物理 capacity 与 pool snapshot/per-backing lifetime 守恒。
6. 完成 targeted pytest、全量 pytest、pre-commit；`IR_SPEC.md`、场景陈旧注释、计划交叉链接均更新一致，临时脚手架清除、artifact evidence 可审计。
7. 私有释放与共享 backing 各自满足批次 I/II 安全门；若任一门未过，本批不得以缩小场景、降级异常、关闭检查或部分交付宣称完成。
