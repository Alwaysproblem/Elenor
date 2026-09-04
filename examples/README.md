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

| 名称                                 | 编辑文件                                            | 主要路径                                                    |
| ------------------------------------ | --------------------------------------------------- | ----------------------------------------------------------- |
| `gather`                             | `workloads/gather_profiled.mlir`                    | deterministic profiled Gather                               |
| `gather-matmul`                      | `workloads/gather_matmul.mlir`                      | Gather → BOA Matmul                                         |
| `matmul-gather-add`                  | `workloads/matmul_gather_add.mlir`                  | BOA Matmul → Gather → EVU Add                               |
| `gather-matmul-4tiles-2contexts`     | `workloads/gather_matmul_4tiles_2contexts.mlir`     | `placement=15`，4 tiles × 2 contexts，Gather → Matmul       |
| `matmul-gather-add-4tiles-2contexts` | `workloads/matmul_gather_add_4tiles_2contexts.mlir` | `placement=15`，4 tiles × 2 contexts，Matmul → Gather → Add |
| `pow-dual-context`                   | `workloads/pow_dual_context.mlir`                   | 两个同 shape context 并发                                   |
| `pow-dual-context-mixed-shapes`      | `workloads/pow_dual_context_mixed_shapes.mlir`      | 两个不同 shape context 并发                                 |
| `pow-sequential-contexts`            | `workloads/pow_sequential_contexts.mlir`            | 两个 context 串行提交                                       |

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
