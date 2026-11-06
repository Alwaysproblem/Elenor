# 设计评审稿验证记录

> 对象：[Coral Nexus Tile 总体设计](Coral_Nexus_Tile_Design.md)及 W0–W7 组件设计；输入释放勘误另覆盖[总架构](../ELENOR_Architecture_Design_v1.md)与[Tile 需求规格](../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。此记录区分**当前模型的实跑证据**与**未来 RTL 的验收要求**；不声明任何新增硬件已实现。检查日期：2026-11-05。

## 1. 本轮变更与未执行项

初次交付位于 `design/coral_nexus_tile/`：总体设计、8 份组件设计、本文及 3 张架构图的 SVG 源／PNG 展示版。随后按用户 review 同步输入释放勘误：总架构 §16.3.1、需求 ARN-05／09 与 §12、Coral 总体 §4.4 及 W1／W2／W4／W5／W6／W7，明确阶段通知、纯读 pin 解除、按 buffer release、物理 final-free 与剩余资源退休相互独立；不新增按范围释放能力。未修改 `prompt.md`、`prompt_temp.md`、现有 Python／RTL／测试／fixture、图示或系统配置。

用户确认的设计方向：Coral 派生完整 Tile RTL 与 pipeline validator CS1；BF16 张量输入／输出、FP32 内部累加；沿用 Coral BF16 VME 内部 ROD。最终 BF16 转换已随用户通过设计 review、采用推荐方案而确定为 RNE；此批准不是 RTL 实跑证据。

本轮未运行：pytest／pre-commit 全库门、Coral Bazel 构建、RTL elaboration／simulation、容器镜像构建、综合、FPGA 或 PPA。没有依赖安装。所列未来命令只属于设计接口，不代表已有脚本。

## 2. 当前 Python 基线 smoke

### 2.1 执行环境

- 已有解释器：`/home/yongxiy/miniconda3/envs/elenor-validator/bin/python`，`--version` 实测 `Python 3.11.15`。
- `docker version --format '{{.Client.Version}} {{.Server.Version}}'` 实测 `28.3.0 28.3.0`；未 pull／build／run 硬件镜像。
- 当前编译产物由源码与实跑核对为 schema `3`／ABI `v3`，不是旧 README 的 `2/v2` 或其他 Nexus 工作树的版本。

### 2.2 原生 CLI 调度与资源基线

已执行（从仓库根目录，使用上述已有解释器）：

```bash
python -m pipeline_validator --workload pow --context-mode 2 \
  --input-binding Y=0x100000:524288:rw \
  --sim-override fidelity=full_memory --memory-trace \
  --trace-json /tmp/coral-cs1-design-20261105/pow-bound.trace.json \
  --compiled-output /tmp/coral-cs1-design-20261105/pow-bound.compiled.json \
  --json --report /tmp/coral-cs1-design-20261105/pow-bound.report.json
```

结果：退出码 `0`，`completed=true`，`cycles=9853`。报告 7 项检查全部通过：task completed、credit invariant、MFE active ratio、EVU active ratio、stream stall ratio、multi-stage Group I/O、Arena zero leak。L2 与全部 L1 的 Arena／view／pin／in-flight／reserved bytes 归零。

此场景未注入 ByteStore；EVU pow 不执行平方算术。它是当前 Python 调度／资源闭包 smoke，不是数值或 RTL 证据。首次未提供 `Y` binding 的尝试被 Loader 拒绝（退出码 `2`，`actual bindings do not exactly match executable global inputs`），补全输入后运行成功；未绕过校验。

### 2.3 实际 indexed gather／scatter 字节检查

从现有 [indexed_gather_scatter.mlir](../../examples/workloads/indexed_gather_scatter.mlir) 编译，serialize→parse→load 后运行，配置 `HardwareConfig().with_overrides(num_dma_channels=2)`、`SimConfig(fidelity="full_memory", context_count=2, max_cycles=200000, memory_trace=True)`。该 fixture 只映射 Tile 0 的一个逻辑 Task；Context 容量 2 不等于本场景已证明两个 Context 并发。

| 输入／绑定                   | 内容                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------- |
| DATA，`0x100000:64:r`        | 4×4 little-endian i32，四行为 `[10..13]`、`[20..23]`、`[30..33]`、`[40..43]` |
| GATHER_IDX，`0x200000:12:r`  | `[3,0,3]`                                                                    |
| SCATTER_IDX，`0x300000:12:r` | `[1,1,2]`                                                                    |
| OUT，`0x400000:64:w`         | 未 seed，仅实际写过的字节应初始化                                            |

观测：退出成功、`cycles=1710`；OUT row 1 为 `[10,11,12,13]`（后 ordinal 覆盖前写），row 2 为 `[40,41,42,43]`；读取 row 0／3 各触发未初始化字节错误。gather 与 scatter 都是 `48 bytes`。`credit_invariant_ok=true`，调用 `sim.group.assert_l2_closed()` 成功，快照中全部 L1／L2 的 Arena／view／backing／pin／in-flight／reserved bytes 为零。

以下是本轮 smoke 的可复现核心；使用已有环境运行，不需要新增永久测试或安装依赖：

```python
from struct import pack, unpack

from pipeline_validator.compiler import compile_program
from pipeline_validator.compiled_program import parse_compiled_program, serialize_compiled_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.allocator import MemoryInvariantError
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import load_workload_ir

bindings = {
  "DATA": GlobalBinding("DATA", 0x100000, 64, "r"),
  "OUT": GlobalBinding("OUT", 0x400000, 64, "w"),
  "GATHER_IDX": GlobalBinding("GATHER_IDX", 0x200000, 12, "r"),
  "SCATTER_IDX": GlobalBinding("SCATTER_IDX", 0x300000, 12, "r"),
}
store = ByteStore()
rows = [tuple(10 * (r + 1) + i for i in range(4)) for r in range(4)]
store.seed_hbm(0x100000, b"".join(pack("<4i", *row) for row in rows))
store.seed_hbm(0x200000, pack("<3i", 3, 0, 3))
store.seed_hbm(0x300000, pack("<3i", 1, 1, 2))
hw = HardwareConfig().with_overrides(num_dma_channels=2)
config = SimConfig(fidelity="full_memory", context_count=2, max_cycles=200000, memory_trace=True)
module = load_workload_ir("examples/workloads/indexed_gather_scatter.mlir")
artifact = compile_program(
  module, hw, config, binding_assumptions=bindings,
  source_name="examples/workloads/indexed_gather_scatter.mlir",
)
artifact = parse_compiled_program(serialize_compiled_program(artifact))
sim = Simulator(hw, config, byte_store=store, enable_tracer=True)
result = sim.run(load_program(artifact, hw, config, actual_bindings=bindings))
assert result.completed and result.credit_invariant_ok
assert unpack("<4i", store.read_hbm(0x400010, 16)) == rows[0]
assert unpack("<4i", store.read_hbm(0x400020, 16)) == rows[3]
for offset in (0, 48):
  try:
    store.read_hbm(0x400000 + offset, 16)
  except MemoryInvariantError:
    pass
  else:
    raise AssertionError("unwritten output row became initialized")
assert result.pmu.events["gather_bytes"] == result.pmu.events["scatter_bytes"] == 48
sim.group.assert_l2_closed()
print(result.cycles, "indexed bytes and lifecycle verified")
```

### 2.4 输入早释放与容量复用 smoke（勘误后）

从仓库根目录使用 §2.1 的已有解释器，实跑未修改的 [l2_admission_wait.mlir](../../examples/scenarios/l2_admission_wait.mlir)。使用已有小容量 Profile，使 A 的输入与输出合计占满用户 L2，B 因 `WAIT_CAPACITY` 等待；不是通过扩大容量绕过等待。

```bash
python -m pipeline_validator \
  --ir-file examples/scenarios/l2_admission_wait.mlir \
  --hw-config examples/configs/profile_l2_256k.yaml \
  --hw-override num_dma_channels=2 --hw-override hbm_fixed_latency_cycles=10 \
  --context-mode 2 --device-context-mode 2 \
  --input-binding A_IN=0x100000:131072:rw \
  --input-binding A_OUT=0x200000:131072:rw \
  --input-binding B_IN=0x300000:131072:rw \
  --sim-override fidelity=full_memory --memory-trace \
  --compiled-output /tmp/coral-input-release-correction.compiled.json \
  --trace-json /tmp/coral-input-release-correction.trace.json \
  --json --report /tmp/coral-input-release-correction.report.json \
  --max-cycles 500000
```

退出码 `0`，`completed=true`，总周期 `21686`；task completed、credit invariant、Arena zero leak 三项检查全部通过。编译产物 hash 为 `a9f6d524d8cce5fb95e80795fa05ca9a8befbd3f69de7bfe6dc93855cbe51a76`。

| 观察点                                          |        cycle | 证据                                                                                                  |
| ----------------------------------------------- | -----------: | ----------------------------------------------------------------------------------------------------- |
| A 的 input_released 聚合与输入 backing 物理回收 |         3313 | `seen=expected=4`；`a_input` 归还 131072 B，pin/in-flight 为 0，`origin_retired=false`                |
| B 的 root 准入激活并取得该容量                  |         3313 | report 的 `active_cycle` 与 `b_input.commit_cycle` 相同；B 与已回收 A 输入的逐 bank segments 完全一致 |
| A 的四个 Task 归还 R lease                      | 19833～19857 | 四个 `task_lease_release` 均晚于输入物理回收，未因输入通知提前释放 R                                  |
| A 的 root 完成                                  |        21683 | report 的 `ctx_a.completion_cycle`                                                                    |

Perfetto SQL 提取的 `EVU:pow` 窗口与输入回收时刻重叠，证明这是“输入 backing 已复用、原计算仍进行”的模型路径。SQL 的 `stats` 无非零 error/data_loss；slice `370`、counter 值 `2937` 与 raw JSON 对账一致，释放字段与 report 的整数周期交叉一致。未核对完整 flow 链，不声明可完整重放。

该 smoke 未 seed ByteStore，EVU pow 不执行真实平方算术；证据仅覆盖既有 Python 调度、输入聚合、引用关闭与容量复用，不证明数值、细粒度分批释放或 RTL 行为。所有新运行产物置于 `/tmp`，未新增永久测试或脚手架。

## 3. Perfetto 导入检查

用仓库已有 `examples/artifacts/tools/trace_processor_shell` 执行 SQL；不是自定义 HTML viewer。两份 trace 都检查 `stats` 的 error／data_loss，并把 raw JSON 的 `X+i+B` 与 `slice` 数、counter payload 数与 `counter` 数比较。

| 场景    | raw 预期 slice | SQL slice | raw 预期 counter | SQL counter | overlap-spill |
| ------- | -------------: | --------: | ---------------: | ----------: | ------------: |
| pow     |            775 |       775 |             4770 |        4770 |             2 |
| indexed |            513 |       513 |             1470 |        1470 |             3 |

唯一非零 error 为 `slice_spill_overlapping_complete_event`：Perfetto 说明重叠 complete slice 被移动到 overflow track，未丢弃；slice／counter 数对账一致。本轮**未证明 legacy flow 链完整**（SQL flow 46／20，不能与原始 s/t/f 事件数量直接相等比较），不把此 trace 当完整边界重放输入。

SQL 最大 `ts+dur` 为 `9853 ns`／`1710 ns`，本次模型 `clock_mhz=1000`，故数值恰好等于 cycles。一般换算是 `cycles = ts_ns * clock_mhz / 1000`；不能跨配置沿用数值相等。源码 `cycle_to_us` 会量化到 1 ns，新联仿原始日志须直接保留整数 cycle／phase／seq。

```sql
SELECT name, value, severity FROM stats
WHERE value != 0 AND severity IN ('error', 'data_loss') ORDER BY name;
SELECT 'slice' AS kind, COUNT(*) AS count FROM slice
UNION ALL SELECT 'counter', COUNT(*) FROM counter
UNION ALL SELECT 'flow', COUNT(*) FROM flow;
SELECT MIN(ts) AS first_ns, MAX(ts + dur) AS end_ns
FROM slice WHERE dur >= 0;
```

## 4. 图示检查

三幅 SVG 均为静态、自包含、无脚本／外部引用，使用主题颜色变量／currentColor、不画背景；PNG 展示版由 Chromium 在明确的浅色页面中渲染，嵌入文档。SVG 源用于编辑／内联主题显示，不作为跨主题外部图片的唯一展示。

已执行 XML 解析、正交连接端点与节点边界、边线不穿框／不交叉检查；浏览器对全部 `<text>` 执行 `getBBox()`，文本在画布内且与所属框保持边距。另在暗色页面加载 3 张 PNG，全部正常解码；像素检查四角为白色、非空笔画存在，防止透明 SVG 外部渲染成全黑。

| 图        | 尺寸    | 节点 | 有向边 | 文本边界／图片加载 |
| --------- | ------- | ---: | -----: | ------------------ |
| CS1 边界  | 960×745 |   12 |     14 | 通过               |
| VME 区域  | 940×900 |   14 |      7 | 通过               |
| Tick 相位 | 960×370 |    5 |      5 | 通过               |

## 5. 文档检查方法

仅格式化本次涉及的 Markdown，不运行会重写无关文件的全库 pre-commit。使用现有 pre-commit 缓存 Prettier `3.8.4` 与仓库 `.prettierrc`；格式后检查幂等和内容等价（忽略空白／表格分隔 padding），并检查九节模板、表格列数、代码围栏、相对链接／锚点、需求 ID／来源标签及 W7 的 T-01～T-68 完整性。

初次评审的实测结果（10 份 Markdown：总体设计、本文、W0–W7）：

| 检查                                            | 结果                                                                        |
| ----------------------------------------------- | --------------------------------------------------------------------------- |
| `prettier --check` 与二次 `--write` 幂等        | 通过；文件逐字节不变                                                        |
| 格式化前后内容等价                              | 仅表格 padding、空行与散文中字面 `*`／`_` 的转义（如 `vset\*`）；无文字增删 |
| 九节标题（总体设计与 W0–W7）                    | 9 个标题与仓库模板逐字一致                                                  |
| 表格列数／代码围栏闭合／相对链接                | 无不一致、无未闭合、无缺失目标                                              |
| 需求 ID、来源标签                               | 无未定义引用                                                                |
| W7 场景表                                       | 连续 T-01～T-68，共 68 行                                                   |
| 残留旧措辞（brief／parent／已撤销的待裁决措辞） | 无                                                                          |

检查中发现并已修复一处排版缺陷：W1 准入流程原写成嵌套反引号的行内代码，渲染会断裂，已改为 `text` 围栏块（语义不变，仅拆行）。同一扫描确认其余文档没有奇数反引号行。本节仅证明文档格式和内部引用一致，不证明任何 RTL 行为。

## 6. 可复核来源与产物指纹

以下 SHA-256 锁定**初次评审**读取与运行的基线，不是新的 ABI 版本；其中根需求规格的 hash 是本次输入释放勘误之前的版本，不代表勘误后的当前文件。本次勘误 smoke 的命令与产物身份见 §2.4。临时日志路径可清理；未来复现应重新运行并按 manifest 保存，不依赖 `/tmp` 目录常驻。

| 文件                                             | SHA-256                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| `Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md`   | `ac58196579ba0f1d9b78573b12f88f390c0be0a70953e1ef6747e5098d7198d8` |
| `prompt.md`                                      | `dd99576424c0c25a88ab2d12f68c70a15f045ef376d267b84ea08900eb2f61a8` |
| `prompt_temp.md`                                 | `be810883578377869bfa1e5991fb0ef473288804a113819c1832a6c76957bed3` |
| `pipeline_validator/simulator.py`                | `b9cf3c088356348c0d4d073e9907c4a2592aff878e41a4c840dd636e3aa773a0` |
| `pipeline_validator/tile_group.py`               | `29d318c55137560770e3596c67bf555d8bb1e34c2e854fae60f505d9cc4f1a77` |
| `pipeline_validator/tile.py`                     | `b24abef106eb3f17d743414cc706b72fb0e0a8784ea9391437f18779991c9d25` |
| `pipeline_validator/engines.py`                  | `f816ec500e3d6cb54ab71ce56601159ce5ad0b1bd05e5fc24a68e7b382d75ae6` |
| `pipeline_validator/compiler/lowering.py`        | `8b3cbd70f06c6a21c86d071914736a68a7ba22cd2190827b1f1af8960367da9b` |
| `pipeline_validator/compiled_program.py`         | `0875f77ba0f509ffd226f25ed081d98efea34ecd72bfc7b8484a75b3798a5662` |
| `pipeline_validator/runtime/program_table.py`    | `cfb46396eb1c7380ecba52b485e2b58069ad2e85ab5a118fd5b742275656b6b4` |
| `pipeline_validator/trace.py`                    | `088e0640035bba40b956e189cd456a8fea4b38017df045ad7f886cdcc76b65fd` |
| `examples/workloads/indexed_gather_scatter.mlir` | `dd21a3e06bea0ae484011018a2d449189bb434f51317ff48c0baf28faa9d49d9` |

| 实跑产物                       | SHA／身份                                                          |
| ------------------------------ | ------------------------------------------------------------------ |
| pow compiled artifact hash     | `d5555fcd45fab4db1f08468ab8b53aa91232079c29264b6bb0a0d142c325d05b` |
| indexed compiled artifact hash | `2b635556d370b0608e255dbfda0961b9630a8c64dd97435f0acf596293c5c817` |
| pow-bound trace SHA-256        | `f745a91be02790654acb7377d368571626b4ac309b156ce0df48228791a65e5c` |
| indexed trace SHA-256          | `bdb97f0859c05a27721224b9e9152fe60f600cd52386a4f06e5fc7443917f30d` |
