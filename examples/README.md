# Pipeline Validator Examples

这个目录只保留四类内容：

```text
examples/
├── workloads/   # 可直接运行、适合复制修改的完整模型
├── scenarios/   # 验证 runtime / memory 协议边界的场景
├── fixtures/    # 测试输入；不作为用户示例入口
├── artifacts/   # 已生成的 trace/report JSON
└── run.sh       # 单示例运行入口
```

## 快速运行

```bash
# 查看所有可运行示例
bash examples/run.sh list

# 运行单个示例
bash examples/run.sh gather
bash examples/run.sh gather-matmul
bash examples/run.sh matmul-gather-add
bash examples/run.sh gather-matmul-4tiles-2contexts
bash examples/run.sh matmul-gather-add-4tiles-2contexts

# 追加任意 pipeline_validator 参数
bash examples/run.sh gather-matmul \
  --trace-json /tmp/gather-matmul.json \
  --json
```

`run.sh` 固定使用 conda 环境 `elenor-validator`，并为目录内置示例提供正确的
input bindings、context 数量、memory fidelity 和必要的硬件 override。未知示例名会明确失败，
不会选择默认模型。

## 可运行 workload

| 名称                                 | 编辑文件                                            | 主要路径                                                                      |
| ------------------------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------- |
| `gather`                             | `workloads/gather_profiled.mlir`                    | deterministic profiled Gather                                                 |
| `gather-matmul`                      | `workloads/gather_matmul.mlir`                      | Gather → BOA Matmul                                                           |
| `matmul-gather-add`                  | `workloads/matmul_gather_add.mlir`                  | BOA Matmul → Gather → EVU Add                                                 |
| `gather-matmul-4tiles-2contexts`     | `workloads/gather_matmul_4tiles_2contexts.mlir`     | `placement=15`，4 tiles × 2 contexts，Gather → Matmul                         |
| `matmul-gather-add-4tiles-2contexts` | `workloads/matmul_gather_add_4tiles_2contexts.mlir` | `placement=15`，4 tiles × 2 contexts，Matmul → Gather → Add                   |
| `pow-dual-context`                   | `workloads/pow_dual_context.mlir`                   | 两个同 shape context 并发                                                     |
| `pow-dual-context-mixed-shapes`      | `workloads/pow_dual_context_mixed_shapes.mlir`      | 两个不同 shape context 并发                                                   |
| `matmul-pow-parallel`                | `workloads/matmul_pow_parallel.mlir`                | 2 matmul + 2 pow context 全并发，BOA/EVU 并行                                 |
| `matmul-pow-free-slot`               | `workloads/matmul_pow_free_slot.mlir`               | matmul 先占满 slot，pow 等首个空槽提前调度                                    |
| `matmul-pow-data-dep`                | `workloads/matmul_pow_data_dep.mlir`                | pow 消费 matmul 输出 C，只等生产者、不过早也不过度串行                        |
| `matmul17-pow-tail-overlap`          | `workloads/matmul17_pow_tail_overlap.mlir`          | 17 个 tile context（4 x placement15 + 1 x placement1），pow 与尾 context 重叠 |
| `pow-sequential-contexts`            | `workloads/pow_sequential_contexts.mlir`            | 两个 context 串行提交                                                         |
| `matmul-2048x512-boa256`             | `workloads/matmul_2048x512x64_boa256x256x32.mlir`   | 2048x512x64 matmul，BOA 256x256x32，K tile 内展开，4 context                  |

多 context trace：

```bash
bash examples/run.sh gather-matmul-4tiles-2contexts \
  --trace-json /tmp/gather-matmul-4t2c.json \
  --json

bash examples/run.sh matmul-gather-add-4tiles-2contexts \
  --trace-json /tmp/matmul-gather-add-4t2c.json \
  --json
```

两份模型都在 `nexus.program` 中连续 submit 两个 context，没有中间 await；每个 context
使用 `placement = 15` 和 `task.range 0..4`，分别固定到 UCE context 0/1。

`matmul-2048x512-boa256` 展示 2048x512x64 bf16 matmul 按 BOA shape 256x256x32 的切分方式
（placement 全 15 ⇒ 每 dispatch 4 task ⇒ 4 context x 4 task = 16 个输出块）：

- **K tiling 在一个 tile 内完成**：K=64 展开为 2 个静态 `tile.boa.async`（k=32），
  第二个带 `accumulate`，无任何循环；两个 K-step 的输入 load 一次性发射，
  MFE 与 BOA 流水重叠。BOA 描述符严格保持 256x256x32。
- **M/N tiling 在不同 context 下完成**：context 网格 2x2（`@mm_m0n0`..`@mm_m1n1`，
  m_sup/n_sup 超块），每个 context 的 `task.range 0..4` 沿 M 把超块切成 4x256 行。
- **全部 `placement = 15`**：每个 context 的 4-task grid 铺满全组 4 tiles；
  device slot pin（`nest.context context = 0..3`）与 UCE context pin
  （dispatch `context = 0..3`）一一对应，每 tile 用 4 个 UCE context 分别承载
  4 个 slot 的 task（`--context-mode 4`，超出 V1.x 每 tile 2 context 上限，
  作为 what-if 探索）。
- **block-packed 全局布局**：`A[2,4,2,256,32]`/`B[2,2,32,256]`/`C[2,2,4,256,256]`
  把 tiling 维全放前导维，所有 subview/DMA 都是连续 row-major 区间。

```bash
bash examples/run.sh matmul-2048x512-boa256 --trace-json /tmp/matmul-boa256.json --json
```

## matmul + pow 调度验证组

四个示例基于 `matmul_2048x512x64_boa256x256x32.mlir` 改造，验证 BOA matmul 与
EVU pow 在不同依赖/资源约束下的调度行为。每个示例都建议带 `--trace-json`
运行，在 Perfetto 里看 `Device / Slot:N`、`BOA`、`EVU`、`MFE_LD0/ST0` 轨道：

```bash
bash examples/run.sh matmul-pow-parallel       --trace-json /tmp/ex1.json
bash examples/run.sh matmul-pow-free-slot      --trace-json /tmp/ex2.json
bash examples/run.sh matmul-pow-data-dep       --trace-json /tmp/ex3.json
bash examples/run.sh matmul17-pow-tail-overlap --trace-json /tmp/ex4.json
```

| 示例                        | 结构                                                                                                                                                                                                                                | 验证点（实测）                                                                                                                                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `matmul-pow-parallel`       | M=1024 半边 matmul（2 context）+ 独立 Y 上的 pow（2 context），全部 placement=15、4 slot 从 t=0 并发 submit                                                                                                                         | EVU:pow 与 BOA:matmul 时间窗重叠（实测 overlap ≈ 3.7 µs，16 BOA + 16 EVU 事件）                                                                                                                        |
| `matmul-pow-free-slot`      | 完整 4-context matmul 先占满 4 个 device slot，2 个 pow context（不 pin slot）紧随 submit；pow 输入是独立 Y，无 data 依赖                                                                                                           | pow 的 submit 阻塞（`device_submit_wait`），在**第一个** matmul context 完成释放 slot 时立即被接管（实测 106 µs，早于最后一个 matmul 的 163 µs），并与仍在运行的 matmul context 并发                   |
| `matmul-pow-data-dep`       | 同 4-context matmul，但 pow 的输入是 matmul 写回 HBM 的 C（`:rw` binding）；`@pow_np_c0` 只 await 生产 C[m0] 半边的两个 context，`@pow_np_c1` 只等 m1 行                                                                            | pow 绝不早于生产者启动（实测 c0 在 117 µs = m0 行完成时刻），但不等无关工作（c0 与仍在跑的 m1 行 matmul 重叠）——依赖感知、不过度串行                                                                   |
| `matmul17-pow-tail-overlap` | M=5120 → 20 块：4 x placement=15 context（16 个 tile context，块 0..15）+ 1 x placement=1 尾 context（第 17 个 tile context，单 task 串行算剩余 4 块）；2 个 pow context 消费前 16 块（`--context-mode 5 --device-context-mode 5`） | 5 个 matmul context 从 t=0 分开并发调度；尾 context 的 BOA 只落 Tile0；pow 不等尾 context，在尾 context 仍在跑（0..221 µs）时用其余空闲 tile context 提前运行（pow EVU 188..271 µs，实测重叠 18.4 µs） |

设计说明：

- ex2/ex3 是同一拓扑的对照组：pow 输入从独立 Y 换成 matmul 输出 C 后，启动
  约束从"slot 空闲"变成"生产者完成"。
- ex4 的 17 个 tile context 不是 16 的倍数：4 个满 placement context（4 task 各）
  - 1 个 placement=1 context（1 task，task 内串行展开剩余块）。尾 context 只占
    tile 0 的 1 个 UCE context，其余 15 个 tile context 空闲时被 placement=15 的
    pow 立即复用。
- pow 的 tile program 每 task 处理 2 个 chunk（load → pow → store × 2），
  L1 buffer 顺序复用；`input_released` 在最后一次输入 load 之后发出。

所有 Gather runnable example 都包含完整输出路径：

```text
final L1 buffer
  → tile.store.async
  → L2 role=\"out\"
  → nest.dma.store.async
  → writable HBM output binding
```

## 协议场景

| 名称                                | 编辑文件                                           | 主要路径                                         |
| ----------------------------------- | -------------------------------------------------- | ------------------------------------------------ |
| `l2-admission-wait`                 | `scenarios/l2_admission_wait.mlir`                 | 精确 L2 容量下的 admission wait / release wakeup |
| `sequential-release-counterexample` | `scenarios/sequential_release_counterexample.mlir` | 中间 await 导致严格串行的反例                    |

运行：

```bash
bash examples/run.sh l2-admission-wait --json
bash examples/run.sh sequential-release-counterexample
```

## 复制后自行修改

最简单的工作流：

```bash
cp examples/workloads/gather_profiled.mlir /tmp/my_gather.mlir

bash examples/run.sh file /tmp/my_gather.mlir \
  --input-binding table=0x200000:8388608:r \
  --input-binding indices=0xA00000:4096:r \
  --input-binding output=0xB00000:256:w \
  --sim-override fidelity=full_memory \
  --max-cycles 200000 \
  --json
```

修改时优先关注：

1. `nexus.program` 与 `nest.context` global formal 的 shape/dtype 必须一致。
2. `nest.dispatch.tasks.async` 的 `globals`、`ins`、`outs` 必须与 Tile Program formal 对齐。
3. 每个 async event tag 在所属 body 内必须唯一。
4. Gather profile 的 request bytes 总和必须等于 `result_bytes`。
5. `tile.await` 决定 engine 的执行顺序。
6. 输出路径需要同时声明 `tile.store.async`、`output_ready`、L2 `role="out"` 和
   `nest.dma.store.async`。

当前 `tile.boa.async` 和 `tile.evu.async` 是 timing descriptor，没有显式 L1
operand/result。组合示例沿用 Pow 的隐式 L1 原地约定：`%matmul_dst` 作为最终工作
buffer，并在最后一个 engine event 完成后显式 Store。该路径是 output lifecycle /
timing accurate，不是 tensor value accurate。

## Fixtures 与 artifacts

- `fixtures/pow_single_context.mlir`：测试使用的单 context IR，不出现在 `run.sh list`。
- `artifacts/`：历史生成的 trace/report。运行新实验时建议输出到 `/tmp`，避免把大 JSON
  与可编辑 MLIR 混在一起。
