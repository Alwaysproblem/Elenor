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

## 调度方法与代表实例

这些收益是满足依赖、队列 credit、带宽与资源准入时的调度机会，不是固定加速比。
运行绑定与配置以 [`run.sh`](run.sh) 为准；`nexus`（CPU）→ `nest`（Context/L2）
→ `tile`（Task/L1/引擎）不是 GPU thread/warp 模型。

| 方法                        | 代表输入                                                                                                                                                                                                  | 条件与优势                                                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 分层 Tile-SPMD、静态 tiling | [`matmul-2048x512-boa256`](workloads/matmul_2048x512x64_boa256x256x32.mlir)、[`reduce-sum-multiuce`](workloads/reduce_sum_multiuce.mlir)                                                                  | Context 分超块，Grid Task 按 placement 映射 Tile，Tile Program 内展开 K chunk；复用模板，减少逐 Tile 控制描述。                      |
| 显式异步流水                | [`reduce-sum-single-context`](workloads/reduce_sum_ktiled_single_context.mlir)、[`matmul-splitk-pipeline`](workloads/matmul_splitk_pipeline.mlir)                                                         | HBM→L2 prefetch、L2→L1 双缓冲和常驻累加器分别有依赖；独立 buffer、credit、带宽允许时才可重叠。                                       |
| 异构引擎互补                | [`matmul-pow-parallel`](workloads/matmul_pow_parallel.mlir)、[`ready-action-branch`](scenarios/ready_action_branch.mlir)                                                                                  | 独立 Context 或分支可以用 BOA/EVU；Context 生命周期重叠与引擎区间重叠是不同证据。                                                    |
| 有限资源补位                | [`matmul-pow-free-slot`](workloads/matmul_pow_free_slot.mlir)                                                                                                                                             | 无数据依赖的 pow 不必等全部 matmul；本例先受 CPU outstanding 限额阻塞，之后仍需 Group/Tile 准入。                                    |
| 只等待真实生产者            | [`matmul-pow-data-dep`](workloads/matmul_pow_data_dep.mlir)、[`device-dependency-submit`](scenarios/device_dependency_submit.mlir)                                                                        | 消费者等生产者完成，不等无关 root；源码显式 await 与 submit `depends_on` 分开。                                                      |
| 尾部利用                    | [`matmul17-pow-tail-overlap`](workloads/matmul17_pow_tail_overlap.mlir)                                                                                                                                   | `placement=1` 尾 Task 只占 Tile0 相应 UCE context；其他 Task 可在满足 pin/资源条件时继续，无迁移、抢占或 task stealing。             |
| 有界 ready-action           | [`ready-action-branch`](scenarios/ready_action_branch.mlir)                                                                                                                                               | S0 只考虑各 Context 队首；S1 可在有限扫描窗口跳过 PENDING 依赖，均受 await/barrier、配额和 credit 限制。                             |
| Context 内 L2 交接          | [`matmul-splitk-multicontext-pipeline`](workloads/matmul_splitk_multicontext_pipeline.mlir)、[`reduce-sum-splitk-multiuce`](workloads/reduce_sum_splitk_multiuce.mlir)                                    | leaf→partial→combine 用 `sharing="context-local"`，省去 partial HBM 中转；占用 L2 Arena，不跨 Context 导出。                         |
| 跨 Context 只读共享         | [`l2-shared-weight`](scenarios/l2_shared_weight.mlir)、[`l2-private-weight`](scenarios/l2_private_weight.mlir)、[`l2-shared-fanout`](scenarios/l2_shared_fanout.mlir)                                     | publish/shared.ref/claim 复用 backing、减少重复 HBM→L2 prefetch；每 Tile 的 L2→L1 load 仍存在，producer 退休不等于最后 reader 释放。 |
| 资源合同与 Profile          | [`l2-admission-wait`](scenarios/l2_admission_wait.mlir)、[`profile-reconfiguration`](scenarios/profile_reconfiguration.mlir)、[`l2-admission-profile-switch`](scenarios/l2_admission_profile_switch.mlir) | 静态证明与实际准入区分容量、碎片、槽位、控制资源、R lease；兼容补位不切档，真实切档有等待、维护、ACK。                               |

[`gather-matmul`](workloads/gather_matmul.mlir) 的 Gather→BOA 命中结果由源码
`tile.profiled.access` 指定，不是根据 Cache 容量计算出的硬件命中率。

## 可运行 workload

| 名称                                         | 编辑文件                                                    | 主要路径                                                                                                                                                                                            |
| -------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gather`                                     | `workloads/gather_profiled.mlir`                            | deterministic profiled Gather                                                                                                                                                                       |
| `gather-matmul`                              | `workloads/gather_matmul.mlir`                              | Gather → BOA Matmul                                                                                                                                                                                 |
| `matmul-gather-add`                          | `workloads/matmul_gather_add.mlir`                          | BOA Matmul → Gather → EVU Add                                                                                                                                                                       |
| `gather-matmul-4tiles-2contexts`             | `workloads/gather_matmul_4tiles_2contexts.mlir`             | `placement=15`，4 tiles × 2 contexts，Gather → Matmul                                                                                                                                               |
| `matmul-gather-add-4tiles-2contexts`         | `workloads/matmul_gather_add_4tiles_2contexts.mlir`         | `placement=15`，4 tiles × 2 contexts，Matmul → Gather → Add                                                                                                                                         |
| `pow-dual-context`                           | `workloads/pow_dual_context.mlir`                           | 两个同 shape context 并发                                                                                                                                                                           |
| `pow-dual-context-mixed-shapes`              | `workloads/pow_dual_context_mixed_shapes.mlir`              | 两个不同 shape context 并发                                                                                                                                                                         |
| `matmul-pow-parallel`                        | `workloads/matmul_pow_parallel.mlir`                        | 2 matmul + 2 pow context 全并发，BOA/EVU 并行                                                                                                                                                       |
| `matmul-pow-free-slot`                       | `workloads/matmul_pow_free_slot.mlir`                       | matmul 先占满 slot，pow 等首个空槽提前调度                                                                                                                                                          |
| `matmul-pow-data-dep`                        | `workloads/matmul_pow_data_dep.mlir`                        | pow 消费 matmul 输出 C，只等生产者、不过早也不过度串行                                                                                                                                              |
| `matmul17-pow-tail-overlap`                  | `workloads/matmul17_pow_tail_overlap.mlir`                  | 17 个 tile context（4 x placement15 + 1 x placement1），pow 与尾 context 重叠                                                                                                                       |
| `pow-sequential-contexts`                    | `workloads/pow_sequential_contexts.mlir`                    | 两个 context 串行提交                                                                                                                                                                               |
| `matmul-2048x512-boa256`                     | `workloads/matmul_2048x512x64_boa256x256x32.mlir`           | 2048x512x64 matmul，BOA 256x256x32，K tile 内展开，4 context                                                                                                                                        |
| `reduce-sum-single-context`                  | `workloads/reduce_sum_ktiled_single_context.mlir`           | 256x4096 Reduce-Sum，单 context：1 dispatch 4 task，K chunk 双缓冲 + f32 acc                                                                                                                        |
| `reduce-sum-splitk-multicontext`             | `workloads/reduce_sum_splitk_multicontext.mlir`             | 256x16384 Reduce-Sum，split-K 跨 8 context 分区 + HBM 部分和 combine                                                                                                                                |
| `reduce-sum-multiuce`                        | `workloads/reduce_sum_multiuce.mlir`                        | 512x4096 Reduce-Sum，UCE supertile × task 行块两级 M 分工，全 tile 4 context（--context-mode 4）                                                                                                    |
| `reduce-sum-gpu-tree`                        | `workloads/reduce_sum_gpu_tree.mlir`                        | 256x4096 Reduce-Sum，GPU reduce-tree：4 leaf 按 K 配对 + 两级合并树                                                                                                                                 |
| `reduce-sum-splitk-multiuce`                 | `workloads/reduce_sum_splitk_multiuce.mlir`                 | 256x4096 Reduce-Sum，全 tile 4 UCE context 沿 reduce axis 切分 + context-local L2 扁平合并                                                                                                          |
| `matmul-splitk-pipeline`                     | `workloads/matmul_splitk_pipeline.mlir`                     | 256x256x512 matmul，reduce-K tiling 三级流水（split-K leaf + scratch 合并）                                                                                                                         |
| `matmul-splitk-multicontext-pipeline`        | `workloads/matmul_splitk_multicontext_pipeline.mlir`        | 512x512x512 matmul，M/N 2×2 四 context；各 context 以 context-local L2 合并 split-K partial                                                                                                         |
| `transformer-prefill-attention`              | `workloads/transformer_prefill_attention_pipeline.mlir`     | Transformer Prefill attention block（GQA 16:4，seq=512）：单 root 内 QKV K-chunk 流水 + 4×4 blocked attention（L1 KV ping/pong）+ Wo prefetch overlap + N-split 输出投影；baseline 对照 `_baseline` |
| `transformer-decode-kv`                      | `workloads/transformer_decode_kv_pipeline.mlir`             | Decode step @valid_len=2048：8 个 KV block L2 ping/pong 流水 + online softmax state + fixed-position KV append；baseline 对照 `_baseline`；生成器在 `generators/`                                   |
| `transformer-prefill-attention-multicontext` | `workloads/transformer_prefill_attention_multicontext.mlir` | 单次 Prefill 四 query-row grids，默认每 tile 两 UCE contexts，共享 Q/K/V 与 WO                                                                                                                      |
| `transformer-decode-kv-multicontext`         | `workloads/transformer_decode_kv_multicontext.mlir`         | 单请求 split-KV 四分区 + stable merge；默认两 UCE contexts，显式 HBM channel-aware packet packing                                                                                                   |

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
  Group execution slot pin（`nest.context context = 0..3`）与 UCE context pin
  （dispatch `context = 0..3`）分别约束不同层级；本例 `--context-mode 4`
  是验证配置，硅片 context 数由 PPA 冻结。
- **block-packed 全局布局**：`A[2,4,2,256,32]`/`B[2,2,32,256]`/`C[2,2,4,256,256]`
  把 tiling 维全放前导维，所有 subview/DMA 都是连续 row-major 区间。

```bash
bash examples/run.sh matmul-2048x512-boa256 --trace-json /tmp/matmul-boa256.json --json
```

`reduce-sum-*` 是一组 Reduce-Sum 示例（`Y = sum_k X[:, k]`，沿 **reduce axis**
分块经 L2 流水；单 context 的 512 KiB K 切片超出本 program 的 per-task L1
Arena contract 135168 B，故必须分块）：

- **`reduce-sum-single-context`**（256x4096 bf16，X 2 MiB）：1 个 dispatch、
  4 task 各拥有 64x4096 K 切片；切成 8 x 64 KiB chunk，`buf0/buf1` 双缓冲
  让 MFE load 与 EVU `reduce_sum` 流水重叠，`%acc[64]` f32 常驻 L1 承担
  跨 chunk 循环携带依赖；`input_released` 在最后一个 chunk load await 之后。
- **`reduce-sum-splitk-multicontext`**（256x16384 bf16，X 8 MiB）：split-K
  跨 context——8 个 producer context 各拥有 256x2048 K 分区（1 MiB），
  部分和经 HBM 写入 `Y_part[8,4,64]`，combine context 用 8 个独立
  `[4,64]` 输入逐份读回（DMA 只做线性拷贝、不转置）后合并出 `Y[256]`。
- **`reduce-sum-multiuce`**（512x4096）：非 reduce axis 的 multiUCE——
  M 维两级分工：**UCE supertile**（dim0，4 个静态变体 dispatch，UCE pin
  0..3）× **task 行块**（placement=15 的 4 task，`task_dim = 1`），4 uce ×
  4 task × 32 行 = 512 行无重复覆盖，全部 4 个 tile 参与执行且每 tile
  同时驻留 4 个 UCE context（`--context-mode 4`）。每 context 规约完整
  K=4096（context 内 8×32 KiB chunk 双缓冲），输出 4 个私有 `[4,32]`
  f32 buffer 分别 HBM store，无合并步骤（对照
  `reduce-sum-splitk-multiuce` 的 reduce-axis 切分 + 合并）。
- **`reduce-sum-gpu-tree`**（256x4096）：GPU reduce-tree 两段式算法——
  4 个 leaf dispatch（placement=15、4 task、UCE pin 0..3）各规约一个
  K 配对分区（k_step {2d, 2d+1}，8 个 k_step 恰覆盖一次），partial 经
  HBM store 写入 scratch `S`（对应 CUDA 把 block partial 写回 global
  memory）；合并 dispatch 在 store 完成后从 scratch prefetch 读回，
  二叉树两级合并出 `Y[4,64]`。需 `--context-mode 4`。
- **`reduce-sum-splitk-multiuce`**（256x4096）：**reduce axis** 的
  multiUCE——X 一次 HBM->L2 prefetch 驻留，4 个 leaf dispatch
  （placement=15、4 task、UCE pin 0..3，每 tile 同时驻留 4 个 UCE
  context）按静态变体规约 k_step 配对 {2d, 2d+1}（每 task 经
  `task_dim = 0` 取自己的 64 行块），产出 4 份 `[4,64]` context-local
  partial；扁平 4 路合并 dispatch 同 context 直读（`ins` 绑定同一 L2
  buffer），等待 writer `output_ready` 与 reader `input_released` 后释放，
  partial 全程 L2 驻留，无 HBM 往返。需 `--context-mode 4`。
- **共性**：输入 chunk 为 bf16，累加器/输出为 f32（BF16 reduction 使用
  FP32 accumulate，见
  [ELENOR_EVU_Design.md](../design/elenor_evu/ELENOR_EVU_Design.md)）。
  validator 为时间模型：`tile.evu.async` 无 operand、无 accumulate 语义、
  不执行数值运算；本组示例验证 tiling 结构、搬运字节数、依赖/生命周期
  与时序。

```bash
bash examples/run.sh reduce-sum-single-context --trace-json /tmp/rs1.json --json
bash examples/run.sh reduce-sum-splitk-multicontext --trace-json /tmp/rs2.json --json
bash examples/run.sh reduce-sum-multiuce --trace-json /tmp/rs3.json --json
bash examples/run.sh reduce-sum-gpu-tree --trace-json /tmp/rs4.json --json
bash examples/run.sh reduce-sum-splitk-multiuce --trace-json /tmp/rs5.json --json
```

`matmul-splitk-pipeline` 展示 matmul 的 **HBM→L2→L1 三级流水**（每级都有显式 IR）：

- **三级结构**：第 1 级 HBM→L2 一次整块 prefetch；第 2 级 L2→L1 双缓冲 load（下一 k_sub 的
  load 与当前 BOA compute 重叠）；第 3 级 BOA 在 L1 逐步 `accumulate` 进常驻 f32 C acc。
- **`matmul-splitk-pipeline`**（A[256,512] × B[512,256] → C[256,256]）：
  reduce axis split-K——4 个 leaf dispatch（4 task，UCE pin 0..3，
  `--context-mode 4`）各规约 K quarter {2d, 2d+1}（每 task 2 个
  k_sub×64 的 BOA accumulate），partial 经 HBM scratch `S` 写回，
  combine dispatch 从 scratch prefetch 读回 4 份并顺序累加出
  C[256,256]。leaf L1 contract 147456 B、combine **131072 B**（两个
  `[64,256]xf32` buffer 各 65536 B）；L2 contract 2883584 B。
  multicontext 变体的 combine contract 也是 131072 B。原输入文件头的
  196608 B 说法与实际 `tile.resources` / 编译产物不一致，不作为合同。

- **数值边界**：validator 为时间模型，`tile.boa.async`/`tile.evu.async`
  无 operand、不执行数值运算，`accumulate` 仅为意图标注；输入 bf16、
  C 累加器/输出 f32（FP32 accumulate，见
  [ELENOR_EVU_Design.md](../design/elenor_evu/ELENOR_EVU_Design.md)）。

```bash
bash examples/run.sh matmul-splitk-pipeline --trace-json /tmp/mm.json --json
```

新增的 `matmul-splitk-multicontext-pipeline` 保留上述 4 leaf → partial → combine
结构，将 C[512,512] 按 M/N 的 2×2 象限分给 4 个独立的 `nest.context`。
每个 context 的 4 个 task 沿 M 再分成 4 个 64 行块；4 个 leaf 各处理
两个 K=64 tile，在 f32 L1 acc 中累加。combine 在本 context 直接读取
4 份 `sharing = "context-local"` L2 partial，写回自己负责的 C 象限；
combine 按 32 行半区做双缓冲 software pipeline：q0 半区直接 load 进
acc 半区，q1/q2/q3 的 L2→L1 load 与前一个 EVU add 重叠，L1 contract
仍为 131072 B（4 x 32 KiB 缓冲）。与原例的 HBM scratch 中转不同，
A/B 输入在每个 context 独立 prefetch。
每 context L2 contract 1,835,008 B，4 个 context 合计 7 MiB。
该示例仍是时间模型，不检查 tensor 数值结果。

```bash
bash examples/run.sh matmul-splitk-multicontext-pipeline \
  --trace-json /tmp/mm-splitk-multicontext.json --json
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

| 示例                        | 结构                                                                                                                             | 验证中的相对次序条件                                                                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `matmul-pow-parallel`       | 2 个 matmul + 2 个独立 Y 的 pow Context，连续 submit、末尾统一 await；默认 `DeviceConfig.issue_width=1`，不在同一 cycle 全部提交 | 检查 Context 生命周期；BOA/EVU 引擎区间是否重叠须读取本次 trace。                                                                          |
| `matmul-pow-free-slot`      | 4 个 matmul 后连续提交 2 个无数据依赖的 pow；首先占满 CPU outstanding 额度，不是必然占满 Group 物理槽位                          | 看 `device_outstanding_full` 及 request 的 `active_cycle`/`completion_cycle`；Group 准入另看 `device_admission_wait` 和 port wait reason。 |
| `matmul-pow-data-dep`       | pow 读取 matmul 写回 C；`@pow_np_c0` 只依赖 m0 行两个生产者、`@pow_np_c1` 只依赖 m1 行两个生产者                                 | 消费者准入不早于各自生产者完成，不必等无关行结束。                                                                                         |
| `matmul17-pow-tail-overlap` | 4 × `placement=15` 加 1 × `placement=1` 的尾 Context；两份 pow 消费前 16 块                                                      | 尾部只占 Tile0 相应 UCE context；pow 可在尾 Context 尚未完成时准入，不保证所有 fidelity 下 EVU/BOA 引擎窗重叠。                            |

设计说明：

- ex2/ex3 是同一拓扑的对照组：pow 输入从独立 Y 换成 matmul 输出 C 后，启动
  约束从"slot 空闲"变成"生产者完成"。
- ex4 的 17 个 tile Task：4 个满 placement Context（各 4 task）加 1 个
  placement=1 的尾 Task；尾 Task 只占 Tile0 的 1 个 UCE context。其他 Task
  仍须满足 pin、资源与依赖条件才能补位，并非立即抢占。
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

## Transformer workload 组（timing-only）

`workloads/transformer_*.mlir` 由 `generators/generate_transformer_prefill.py`
/ `generate_transformer_decode.py` 生成（生成器确定性；不要手改生成的
MLIR，改参数后重新生成）。文件顶部注明：validator 是 timing/resource
模型，`tile.boa.async`/`tile.evu.async` 不携带 tensor operand，不验证
数值正确性。

- **Prefill**（seq=512, GQA 16:4, BF16）：单 root。QKV 投影按 K=128 分 8 个
  chunk，L2 ping/pong prefetch 与 BOA accumulate 流水；attention 每 tile
  4 query block × 4 KV block，L1 KV ping/pong（per-head BOA QK + EVU
  online softmax + BOA PV）；Wo prefetch 门控在最后 chunk 的
  `input_released` 上与 attention 重叠；输出投影 N-split（Wo N-packed）。
  baseline 变体逐 chunk `prefetch → nest.await → dispatch → nest.await`
  串行。`analyze_transformer.py` 输出 useful BOA utilization 等指标。
- **Decode**（valid_len=2048, KV_BLOCK=256）：8 个 KV block 静态展开，
  L2 ping/pong：`prefetch(b+2)` 门控 `input_released(b)`，`dispatch(b+1)`
  另需 `output_ready(b)`（online softmax state 经 context-local L2 state
  buffer 循环携带）；`kv_append_tile` 把当前 token 的 K/V 写入静态
  append 位置（K_APPEND/V_APPEND）。`actual_II`、`pipeline_efficiency`
  见 `analyze_transformer.py --decode-trace`。

```bash
bash examples/run.sh transformer-decode-kv \
  --memory-trace --trace-json /tmp/dec.json --json
python examples/generators/analyze_transformer.py \
  --decode /tmp/pipe.report.json /tmp/base.report.json \
  --decode-trace /tmp/dec.json
```

同一请求内部的 multicontext 优化（不是增加 `num_requests`）：

```bash
bash examples/run.sh transformer-prefill-attention-multicontext --memory-trace --json
bash examples/run.sh transformer-decode-kv-multicontext --memory-trace --json
```

两个入口默认 `--context-mode 2`，没有增大硬件带宽或 Group queue capacity。
生成器、R1/R2/R4 消融、物理 packing/binding 变化、性能归因与 Perfetto
并发检查见 [generator README](generators/README.md#单请求内-multicontext-优化)。
Decode 的 packet/channel 放置收益必须与 UCE multicontext 收益分开；
旧串行 KV-block II 分析公式不能直接用于 split-KV completion 顺序。

## PagedAttention decode（full_memory）

```bash
bash examples/run.sh paged-attention-decode --json --report /tmp/pa.pipeline.json
bash examples/run.sh paged-attention-decode-baseline --json --report /tmp/pa.baseline.json
# 输出完整 trace 时另加 --trace-json /tmp/pa.trace.json
```

两个入口使用 `workloads/paged_attention_decode_scenario.json` 和
`generators/run_paged_attention.py`，默认启用 memory trace、4 个 Tile hardware
contexts 和 8 个 device contexts。`full_memory` 保留真实地址、ByteStore、
逐 cache-line Gather/Scatter、cache/MSHR、NoC 和多段 transfer；
BOA/EVU 仍是 timing/resource 模型，不计算 attention 的数值结果。

Runner 使用 scenario 声明的 `hbm_fixed_latency_cycles`、`num_dma_channels`、
`group_policy`、`contexts_per_tile` 和 `max_cycles`，不再把记录的场景参数
悄悄替换为硬件/仿真默认值。默认场景的 HBM latency 为 10 cycles，
Global DMA channels 为 2。

- 不指定 `--hw-config`：scenario 提供上述两个硬件 timing 参数；
  `--hw-override KEY=VALUE` 可以覆盖它们。
- 指定 `--hw-config`：该文件选择硬件 target（未写字段使用 bundled defaults），
  替代 scenario 的硬件 timing 参数；`--hw-override` 最后生效。
- 仿真配置：scenario → `--sim-override` → 专用选项
  `--group-policy`、`--context-mode`、`--max-cycles`。
  `run.sh` 自带的 context 选项可由后续同名命令行选项覆盖。
- L1/L2 reset mode 与生成的 IR contract 一致；`--max-cycles` 只限制
  simulation loop，不限制 seed/parse/compile/load 的墙钟时间。

运行时间包含逐 cycle 控制与有限 lookup/HBM/DMA 资源的排队，不能用
小尺寸 micro 测试的完成时间推断默认尺寸的运行时间。memory trace 的
cache/MSHR 统计使用增量计数；饱和 transfer stage 的重复拒绝使用
resource-free epoch 跳过重复 issue 计算，但每 cycle 的 generation/liveness
检查、wait 计数、仲裁顺序和 trace 事件保持不变。

本机实测的实现层 A/B（同一默认尺寸，固定旧硬件参数
`--hw-override hbm_fixed_latency_cycles=200 --hw-override num_dma_channels=4`）：

| 验证运行                                  | 原版墙钟 | 优化后墙钟 | 加速比 |
| ----------------------------------------- | -------- | ---------- | ------ |
| 完整 pipeline，关闭 memory trace          | 786.7 s  | 462.7 s    | 1.70×  |
| pipeline 前 30k cycles，开启 memory trace | 98.3 s   | 29.4 s     | 3.35×  |

前 30k cycles 的对照有意触发 cycle cap，不是完成时间。
完整无 trace 运行均为 516,209 cycles，JSON report 逐字节一致；
另一个 60k-cycle 对照的 JSON report 与完整 Chrome trace 也逐字节一致。
墙钟数值仅供本机参考，不代表硬件性能。

配置修复后的实际 `run.sh paged-attention-decode`（默认 memory trace、
scenario 的 HBM latency=10 / DMA channels=2）在本机完成于
374,048 cycles / 695.1 s，所有检查 PASS。这一 cycle 变化来自
应用场景参数，与上表保持模拟结果不变的实现加速分开计算。

## 协议场景

| 名称                                | 编辑文件                                           | 主要路径                |
| ----------------------------------- | -------------------------------------------------- | ----------------------- |
| `ready-action-branch`               | `scenarios/ready_action_branch.mlir`               | S0/S1 有界扫描对照      |
| `device-dependency-submit`          | `scenarios/device_dependency_submit.mlir`          | 跨 root 提交依赖        |
| `l2-admission-wait`                 | `scenarios/l2_admission_wait.mlir`                 | L2 容量准入等待         |
| `sequential-release-counterexample` | `scenarios/sequential_release_counterexample.mlir` | 显式等待造成串行        |
| `profile-reconfiguration`           | `scenarios/profile_reconfiguration.mlir`           | Profile 配置事务        |
| `l2-profile-switch-load-ordering`   | `scenarios/l2_profile_switch_load_ordering.mlir`   | L2 切档与首笔 load 顺序 |
| `l2-admission-profile-switch`       | `scenarios/l2_admission_profile_switch.mlir`       | L2 准入与切档           |
| `l2-shared-weight`                  | `scenarios/l2_shared_weight.mlir`                  | readonly 权重复用       |
| `l2-shared-fanout`                  | `scenarios/l2_shared_fanout.mlir`                  | L2 产出后的只读 fanout  |
| `l2-private-weight`                 | `scenarios/l2_private_weight.mlir`                 | 私有 prefetch 对照      |

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
2. Dispatch 的 `globals` 与 `bindings` 分别按位置对齐 global/L2 formals；
   必填 `ins`/`outs` 只声明真实读/写 actual 集合，允许 bindings alias 和 unused formal。
3. 每个 async event tag 在所属 body 内必须唯一。
4. Gather profile 的 request bytes 总和必须等于 `result_bytes`。
5. `tile.await` 决定 engine 顺序；input_released/output_ready 前必须 await
   对应全部 L2 load/store，发出该 phase 后不得再进行同方向访问。
6. 输出路径需要同时声明 `tile.store.async`、`output_ready`、L2 `role="out"` 和
   `nest.dma.store.async`。
7. 每次 HBM Store 等此前真实 writer.output_ready，最后一次覆盖全部写者，不等纯读者计算尾部。
8. 每个 Buffer 恰好 release 一次，依赖精确列齐全部 reader.input_released、
   prefetch completion、Store completion；所有 role 一致，不得省略并行搬运。
   纯读 pin 按真实访问解除，不按 alloc.role；readwrite 必须独立满足读写两阶段。
9. Tile L1 scratch 可在最后一次实际使用完成后 `tile.free %buffer`，无需等
   Task 退休；必须 await 使用该 buffer 的 load/store/Gather 和不透明引擎事件。
   此操作使 view 失效，不返还 owner Arena 容量或 R lease；后续 `tile.alloc`
   只按编译期 lifetime 选定的 slot/offset 再绑定，不进行运行期空闲池搜索。
   只有需要跨 Task 持久化的结果才须完成 store。

以下 [2026-09-11 历史 tile.free 审计](artifacts/tile_free/run-20260911-final/verification.json)
仅对应当时的输入、合同和模型，不代表当前 corpus 的容量、生命周期或验收。

当前 `tile.boa.async` 和 `tile.evu.async` 是 timing descriptor，没有显式 L1
operand/result。组合示例沿用 Pow 的隐式 L1 原地约定：`%matmul_dst` 作为最终工作
buffer，并在最后一个 engine event 完成后显式 Store。该路径是 output lifecycle /
timing accurate，不是 tensor value accurate。

## 全量 MLIR 的 free 历史审查（2026-09-12）

以下数量和产物仅对应当时的输入快照，不是当前 corpus 规模或本次同步验收。
当时逐份审查 132 份 `.mlir`，其中 128 份当前方言输入有 581 个 Tile Program、
1116 个 L1 allocation 声明；历史 reference 另有 1 个。

| 文件结论                 | 数量 | 处理                                                   |
| ------------------------ | ---: | ------------------------------------------------------ |
| 存在有意义的提前释放窗口 |  108 | 共补534个 `tile.free`                                  |
| 最后使用后只剩收尾       |   15 | 保留 terminal 自动回收                                 |
| 已有显式 free 足够       |    1 | 保留 Gather 示例的2个 free                             |
| 没有 L1 allocation       |    4 | 不伪造 free                                            |
| 历史方言文件             |    4 | 逐份人工检查后保留；当前 parser 不支持，不声称运行通过 |

放置规则不是“最后一次 load 后就 free”：当前 BOA/EVU/Pow 不声明 L1 operands，
因此自动审查将程序中的所有此类计算视为潜在使用者，等全部相关事件已由现有
`tile.await` 等待后才考虑释放。`matmul_gather_add` 的 `gather_dst` 保留到
后续 `add_done`；多 chunk matmul 的共享输入、权重保留到最后一个 chunk。
仅当此后仍有实质搬运或异步等待时添加 free；只剩 signal/return 则避免额外 UCE
指令。已有 free 不移动；L2 `nest.release`、所有 await/依赖、shape/bytes、
计算 repeat、placement 和 Context 切分均不改。

- [逐文件结论](artifacts/tile_free/full_mlir_audit-20260912-110535Z/file_decisions.json)
- [逐 allocation 的使用、await 与释放决定](artifacts/tile_free/full_mlir_audit-20260912-110535Z/allocation_audit.json)
- [历史文件审查及 parser 限制](artifacts/tile_free/full_mlir_audit-20260912-110535Z/historical_audit.json)
- [迁移 hash 与 roundtrip](artifacts/tile_free/full_mlir_audit-20260912-110535Z/roundtrip.json)
- [本轮完整配置、命令与运行结果](artifacts/tile_free/full_mlir_audit-20260912-110535Z/execution/summary.json)
- [L1 allocation/release 核对](artifacts/tile_free/full_mlir_audit-20260912-110535Z/execution/l1_lifecycle.json)

128 份当前输入均经过 parse → print → parse 的结构等价检查，其中两份指定 N08
在前后两次 verify 中均保持原有范围错误，其余126份通过。除新增 free 外的源码
字节保持不变；历史 trace/hash 只对应各自当时的输入，不作为本次修改后的证据。

该次历史记录实跑239次：两种 fidelity 各108 completed、2个预期 verifier 拒绝、
2个预期容量 fault，另15个 workload/protocol 入口全部完成；235份 trace
结构检查通过。旧 trace/hash 只对应当时的输入和模型，不作为当前运行的证据。

## Fixtures 与 artifacts

- `fixtures/pow_single_context.mlir`：测试使用的单 context IR，不出现在 `run.sh list`。
- `artifacts/`：历史生成的 trace/report。运行新实验时建议输出到 `/tmp`，避免把大 JSON
  与可编辑 MLIR 混在一起。
