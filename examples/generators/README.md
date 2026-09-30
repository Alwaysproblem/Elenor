# Transformer Workload Generators

为 `Pipeline Validator` 确定性生成两个 Transformer 推理 workload（以及各自
的 baseline 对照），并用 `analyze_transformer.py` 从 report/trace 汇总
guide 要求的指标表。

## 文件

| 文件                                           | 作用                                                                                                                                                                                                           |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `transformer_common.py`                        | 共享 emit 辅助：Arena contract 计算（`conservative_arena_bytes` + 逐 mode 放行）、`tile.*` op 构造器封装、`write_workload`（header 注释 + `print_workload_ir` + 100 列换行 `scripts.format_mlir.format_text`） |
| `generate_transformer_prefill.py`              | 生成 `workloads/transformer_prefill_attention_pipeline.mlir`（流水版）与 `..._baseline.mlir`（逐 chunk 串行对照）                                                                                              |
| `generate_transformer_decode.py`               | 生成 `workloads/transformer_decode_kv_pipeline.mlir`（ping/pong 流水版）与 `..._baseline.mlir`；支持 `--kv-block` / `--num-requests` 变体（写到任意输出路径）                                                  |
| `analyze_transformer.py`                       | 读 `--report` JSON 与 `--trace-json`，输出 Prefill / Decode 指标表、KV_BLOCK II/efficiency、多请求 tokens/s                                                                                                    |
| `generate_transformer_prefill_multicontext.py` | 单次 Prefill 的 query-row 分块多 UCE context 调度；默认 R=4，可生成 R=1/2 消融对照                                                                                                                             |
| `generate_transformer_decode_multicontext.py`  | 单请求 split-KV 四分区与稳定 softmax merge；默认 R=4，显式 KV packet padding/channel 分布                                                                                                                      |
| `analyze_multicontext.sql`                     | Perfetto：逐 tile Task lease 峰值、跨 context MFE/compute 重叠、BOA FLOPs 与 trace 导入审计                                                                                                                    |

## 使用方法

```bash
# 重新生成提交在仓库里的 4 个 .mlir（在仓库根执行）：
PYTHONPATH=. python examples/generators/generate_transformer_prefill.py
PYTHONPATH=. python examples/generators/generate_transformer_decode.py

# 变体（不会覆盖提交文件）：
PYTHONPATH=. python examples/generators/generate_transformer_decode.py \
  --kv-block 128 --output /tmp/dec_128.mlir \
  --baseline-output /tmp/dec_128_base.mlir
PYTHONPATH=. python examples/generators/generate_transformer_decode.py \
  --num-requests 4 --output /tmp/dec_r4.mlir \
  --baseline-output /tmp/dec_r4_base.mlir

# 运行 + 指标汇总：
bash examples/run.sh transformer-decode-kv \
  --memory-trace --trace-json /tmp/dec.json --report /tmp/dec.json.report --json
python examples/generators/analyze_transformer.py \
  --decode /tmp/dec.json.report /tmp/dec_base.report \
  --decode-trace /tmp/dec.json
```

约定：

- **提交的 `.mlir` 是测试 fixture，不要手改**；改参数后用生成器重新生成并
  提交（生成是确定性的，同参数 byte-identical）。
- 生成器在 `num_requests > 1` 时用字母后缀 `_ra/_rb/...`（xDSL printer 会
  把 `K_CACHE_0` 这类纯数字尾缀当作可重编号的 SSA 名，parse 后 entry 名会
  漂移）；对应 binding 名也要用 `_ra..`。
- Prefill 编译较慢（40 logical task 的资源证明，~2 分钟/次）；sweep 中
  `context_count`/`device_context_count` 在 target fingerprint 内，改这些
  参数必须重编译，`--group-policy` 可对已编译 artifact 重放。

## 当前结论（2026-09-29 基线，fidelity=full_memory，dma_channels=2）

数值口径：PMU stall / engine busy 是按 tile/resource **聚合** cycle 计数，
不可直接除 wall-clock makespan 当百分比；下表仅作相对比较。

### Prefill（seq=512, GQA 16:4, KV_BLOCK=128）

| Metric                           |           Baseline |           Optimized |
| -------------------------------- | -----------------: | ------------------: |
| total cycles                     |            616,032 |             543,941 |
| useful BOA utilization           |              74.5% |               84.3% |
| BOA busy (aggregate tile-cycles) |          1,837,504 |           1,837,504 |
| HBM bytes                        |  7,340,032（精确） |           7,340,032 |
| memory / dependency stall        | 77,882 / 1,846,964 | 101,000 / 1,830,375 |

- **能保持矩阵供给**：优化形态 useful BOA utilization 84.3%，BOA busy ≈
  理想算量 + launch 开销；观测到的限制因素是 BOA 计算本身。
- 旧示例的 `context_count`（1/2/4）、device outstanding（1/2/4）无差，
  S1 vs S0 仅差 3 cycles。其原因是单 root 的依赖链与 `R=1` 没有暴露
  可并发 Task；这不是 multicontext 无效或硬件资源已饱和的证明。

### Decode（valid_len=2048, KV_BLOCK=256）

| Metric                         |        Baseline |       Pipelined |
| ------------------------------ | --------------: | --------------: |
| token latency（整步 makespan） |          43,915 |          32,548 |
| KV block II                    |               – |         4,091.7 |
| reference II                   |               – |             512 |
| pipeline efficiency            |               – |           0.125 |
| effective KV BW                |               – |    64.1 B/cycle |
| memory / dependency stall      | 16,266 / 23,672 | 74,570 / 23,652 |

- **能构成稳态流水**。actual II 在 KV_BLOCK 64→512 全部点恒为解析下界的
  ~8 倍且随 block bytes 线性缩放——**trace 与 II scaling 指向**（不构成
  硬件保证）tile 本地搬运路径（L2 bank 读 → 单 MFE load channel → L1 写
  串行 legs）的带宽类限制，而非 per-block 延迟或 Group 调度。
- 调度轴中仅 S1 有感（S0 慢 21%）。

### KV_BLOCK sweep（pipelined）

| KV_BLOCK | blocks | cycles |      II | ref II | efficiency | KV B/cyc |   L1/tile |
| -------: | -----: | -----: | ------: | -----: | ---------: | -------: | --------: |
|       64 |     32 | 33,035 | 1,023.5 |    128 |      0.125 |     64.0 |  28,672 B |
|      128 |     16 | 31,977 | 1,986.6 |    256 |      0.129 |     66.0 |  45,056 B |
|      256 |      8 | 32,548 | 4,091.7 |    512 |      0.125 |     64.1 |  77,824 B |
|      512 |      4 | 33,105 | 8,575.7 |  1,024 |      0.119 |     61.1 | 143,360 B |

**容量注记**：KV_BLOCK=64（33 条 live Grid routes）与 KV_BLOCK=128（17 条）
都超过默认 `group.dispatch_capacity=16`；这两点使用
`--sim-override group.dispatch_capacity=64 --sim-override
group.context_action_quota=64 --sim-override group.action_capacity=64`
运行，其余硬件带宽参数不变。KV_BLOCK=128 为 sweet spot（eff 最高、总周期
最低）；512 受 L1 双缓冲压力，64 受 dispatch 容量与每 block 固定开销。

逐点复现（64 为例；128/512 同理换 `--kv-block` 与文件名；256 用提交的
workload，无需 override）：

```bash
PYTHONPATH=. python examples/generators/generate_transformer_decode.py \
  --kv-block 64 --output /tmp/dec_64.mlir --baseline-output /tmp/dec_64_base.mlir
python -m pipeline_validator --ir-file /tmp/dec_64.mlir \
  --hw-override num_dma_channels=2 --hw-override hbm_fixed_latency_cycles=10 \
  --sim-override group.dispatch_capacity=64 --sim-override group.context_action_quota=64 \
  --sim-override group.action_capacity=64 \
  --memory-trace --trace-json /tmp/dec_64.trace.json --report /tmp/dec_64.report --json \
  --input-binding K_CACHE=0x1000000:1048576:r --input-binding V_CACHE=0x2000000:1048576:r \
  --input-binding Q_IN=0x3000000:2048:r --input-binding S_INIT=0x3001000:4224:rw \
  --input-binding OUT=0x3010000:4096:w --input-binding H_T=0x3011000:2048:r \
  --input-binding WK_A=0x3012000:524288:r --input-binding WV_A=0x3092000:524288:r \
  --input-binding K_APPEND=0x3200000:512:w --input-binding V_APPEND=0x3201000:512:w
python examples/generators/analyze_transformer.py \
  --decode /tmp/dec_64.report /tmp/dec_64.report \
  --decode-trace /tmp/dec_64.trace.json --kv-block 64
```

### 多请求（`--num-requests`）

| num_requests | batch makespan (cycles) | aggregate tokens/s |
| -----------: | ----------------------: | -----------------: |
|            1 |                  32,548 |              30.7k |
|            2 |                  63,293 |              31.6k |
|            4 |                 126,608 |              31.6k |

single-request latency（独立 num_requests=1 运行实测）= 32,548 cycles；
2/4 列是并发 batch makespan，不是单请求延迟。Group context peak=4 仅证明
多个 root 并发准入，不等于每 tile 的 UCE slots 都在并发执行有效工作。
这里每个 root 的 `R=1` 只约束该 root 在每 tile 的 Task 数，不会禁止
其他 root 同时使用该 tile。各 Context 的 `l2_spm_bytes` 共同占同一 Group
L2 pool（mode 0 总 user 容量 16×(524288−4096) B），而不是每 root 独享 8 MiB。

## 单请求内 multicontext 优化

新增的两个 `*_multicontext.mlir` 都只有 **一个 `nexus.submit_context.async`**：
不靠复制完整 Prefill/Decode 请求增加工作量，而是把一次请求中的独立工作交给
不同 Tile UCE contexts。当前 committed examples、生成器和 `run.sh` 均默认
`R=4` / `--context-mode 4`；R1/R2 仍可通过生成器和入口的显式参数运行。
这是示例运行配置，不修改架构 V1.x 的两-context cutline。
Device outstanding 是另一个层级；本例只有一个 root，调大它不会增加 Tile 并发。

### Prefill：按 query rows 拆开 attention 与输出投影

- QKV 仍只投影一次；为满足 `R × max(child L1 Arena)` 约束，将 QKV 内部
  row tile 改为 64。不是复制四份 QKV。
- 512 query tokens 切成四个 128-row grids，每个 grid 仍遍历完整的 512 KV
  tokens。各 grid 的 O/OUT 使用独立的 context-local L2 allocations，
  避免 compiler 的整 buffer RAW/WAW 依赖把不相交的输出强制串行化。
- 每个 query grid 的输出投影只等自己的 attention `output_ready` 与 WO
  prefetch，不等其他 query grids 完成。先注册全部独立 attention grids，
  再注册依赖它们的 projection/store；去掉阶段间的 QKV `grid_done` fence，
  仍保留数据 `output_ready` 依赖和末尾的完整退休等待。
- QKV 的 Q/K/V accumulator 不共用：Q store 可与 K BOA、K store 可与 V BOA
  重叠；每个 row 的三次 store 全部完成后才复用 accumulator。
- Attention/outproj 的独立最终 row stores 一起发出，再等待共同的完成
  frontier；仍在所有 stores 完成后才发 `output_ready`。
- L2 root contract 仍为 6,815,744 B；attention child Arena 为 245,760 B，
  QKV/outproj 各 163,840 B。当前 R4 的 attention 四-context L1 包络为
  983,040 B/tile，只允许 L1 mode 0；L2 root Arena 和每 Task buffers 不变。
  R2 消融的 attention 可以使用 L1 modes 0/1/2。
- R4 提供四个 UCE slots，但不复制 BOA/EVU datapath；
  Task lease 的驻留数不等于同时 compute 的引擎数。
- HBM 输入与输出字节数不变；OUT 的物理布局改为
  `[qblock,head_block,128,256]`，对应旧逻辑
  `OUT[head_block, qblock*128+row, col]`。输出 head/token 顺序不能直接
  当作旧文件的连续布局使用。
- 代价：四个输出 Tasks 从共享 L2 各自读取 WO，较旧软件流水版多
  **6,291,456 B L2→L1**；BOA 有用 FLOPs 仍严格为 3,758,096,384。
  更多小 BOA/EVU launches 也带来开销，需通过下述 R1/R2 对照衡量净收益。

### Decode：split-KV、分区状态与全局 merge

四个分区各处理两个 256-token blocks：`p` 负责 `2p` 与 `2p+1`。
先提交所有分区首块，再提交其第二块，避免尚未就绪的 continuation
占住 ready-action window。分区间没有 softmax recurrence 依赖；
每个分区内部仍必须保留自己的 `output_ready` 链。
首块 pin 为 `p % R`，continuation 改为 `(p+1) % R`，通过 L2 state/output
和显式 `output_ready` 完成跨 UCE context handoff；不是复制分区状态。

每个分区有独立的 FP32 `(m,l)` 和未归一化 `o[64]` 状态。
`S_INIT_STATE_mc[4,4,4,2]` 必须提供每分区的 `m=-inf,l=0`，
`S_INIT_OUT_mc[4,4,4,64]` 必须为零；不能把一份已有非空历史状态重复计入四次。
所有分区 `output_ready` 后执行有真实 L2 loads/stores 的 EVU merge：

```text
m       = max_p(m_p)
alpha_p = exp(m_p - m)
l       = sum_p(alpha_p * l_p)
o       = sum_p(alpha_p * o_p)
OUT     = o / l
```

每分区只保留一个 block 大小的 K/V staging set；第二块 prefetch 等第一块
`input_released`，第二块 dispatch 另等第一块 `output_ready`。没有整 KV cache
一次性 prefetch 后的 barrier。共 8 个 attention grids + 1 merge + 2 append
grids，最多 11 routes；44 个逻辑 Tasks（32 attention + 8 append + 4 merge），
无需调高默认 `dispatch_capacity=16`。L2 Arena 为 2,150,400 B，
attention child L1 为 86,016 B，两个 append children 各 139,264 B。
合并 EVU 额外计 8,480 ops，attention 的 output recurrence 也显式计 EVU 工作；
不省略 merge 成本来制造 speedup。

K/V append 现在有独立的数据依赖：K 只等 `H_T/WK`，V 只等 `H_T/WV`；
R2 下分别 pin 到 contexts 0/1。K 不再被尚未完成的 WV prefetch 卡住，
V 则可与换到 context 0 的末块 attention 重叠。代价是每 tile 多读一次
2,048 B 的 H_T：L2 read / local DMA / L1 write 各增加 **8,192 B**，
HBM/global DMA、其他接口字节量和所有 BOA/EVU ops 均不变。

**HBM channel 放置必须与 multicontext 收益分开。**
当前 `memory/transfer.py::_requests_for_leg` 把一整个 HBM transaction 分配给
`(start_address // hbm_burst_bytes) % hbm_channels`，不是自动跨八 channel
条带化。旧例的 128 KiB KV packets 和 1 MiB-aligned base 全落 channel 0；
仅拆依赖链不会让输入更早 ready。

新例使用 `[8,65568]xbf16` 的物理 packets，每个 packet 前 65536 个元素为
原 `[4,256,64]` payload，最后 32 个元素为 **64 B 不传输的间隔**。
K/V globals 各绑定 1,049,088 B，实际 history DMA 仍各 1 MiB。
`run.sh` 把 V base 设为 `0x2000040`，K base 为 `0x1000000`：
四个分区首块的 K/V 分布在不同 HBM channels，带宽配置不变。
这属于显式物理布局/绑定优化，不是增加 HBM 带宽。`--kv-padding-bytes 0`
可生成无间隔对照。新旧 S_INIT 组织差异使 HBM reads 增加 12,672 B；
attention+append 的 BOA FLOPs 仍严格为 9,437,184，HBM writes 都为 5,120 B。

### 生成、运行与 R 消融

以下命令均从仓库根执行；两个新生成器的默认输出路径也按仓库根解析。
`--contexts-per-tile` 修改的是 source contract（及 Decode partition pin），
`--context-mode` 修改硬件 slot 数；不能拿 R4 文件直接用 context_count=1 跑。

```bash
# 生成默认 R4 committed examples。
PYTHONPATH=. conda run -n elenor-validator python \
  examples/generators/generate_transformer_prefill_multicontext.py
PYTHONPATH=. conda run -n elenor-validator python \
  examples/generators/generate_transformer_decode_multicontext.py

mkdir -p examples/artifacts/transformer_multicontext/reproduce
bash examples/run.sh transformer-prefill-attention-multicontext \
  --memory-trace --trace-json examples/artifacts/transformer_multicontext/reproduce/prefill.trace.json \
  --report examples/artifacts/transformer_multicontext/reproduce/prefill.report.json --json
bash examples/run.sh transformer-decode-kv-multicontext \
  --memory-trace --trace-json examples/artifacts/transformer_multicontext/reproduce/decode.trace.json \
  --report examples/artifacts/transformer_multicontext/reproduce/decode.report.json --json

# 对同一新图分别重新生成 R1/R2/R4；逻辑工作、buffer 大小、HBM 放置不变。
for r in 1 2 4; do
  for kind in prefill decode; do
    stem=transformer-prefill-attention-multicontext
    [ "$kind" = decode ] && stem=transformer-decode-kv-multicontext
    output="examples/artifacts/transformer_multicontext/reproduce/${kind}_r${r}.mlir"
    PYTHONPATH=. conda run -n elenor-validator python \
      "examples/generators/generate_transformer_${kind}_multicontext.py" \
      --contexts-per-tile "$r" --output "$output"
    bash examples/run.sh "$stem" --ir-file "$output" --context-mode "$r" \
      --memory-trace \
      --trace-json "examples/artifacts/transformer_multicontext/reproduce/${kind}_r${r}.trace.json" \
      --report "examples/artifacts/transformer_multicontext/reproduce/${kind}_r${r}.report.json" --json
  done
done

# Perfetto SQL：先看 parse health，再看每 tile Task 峰值和跨 context engine overlap。
examples/artifacts/tools/trace_processor_shell \
  examples/artifacts/transformer_multicontext/reproduce/decode.trace.json \
  -q examples/generators/analyze_multicontext.sql
```

对 split-KV 图不要套用旧 `analyze_transformer.py` 的串行 KV-block II 公式：
不同分区 completion 可能交错，且最后还有 merge。此处比较整请求 makespan、
真实并发、FLOPs/流量和 R1 消融，而非把并行 partial completion 当成串行 recurrence。

### 上一版实测（2026-09-29）：R2 默认，R4 为 what-if

所有运行均为 full_memory、S1、1 GHz、4 tiles、DMA channels=2、
HBM fixed latency=10，硬件 YAML 的其他项不变。没有增大队列或带宽。
基准和新例都实际执行 compile→load→run；原始 report/trace 位于
`examples/artifacts/transformer_multicontext/`。这些是上一版 IR 的历史结果；
当前生成器已更新，以下本轮实测才对应当前 committed examples。

| 同一逻辑 workload         | 旧 software pipeline | 新图 R1 | 新图 R2（默认） | 新图 R4 |
| ------------------------- | -------------------: | ------: | --------------: | ------: |
| Prefill makespan / cycles |              543,941 | 557,553 |         515,222 | 516,241 |
| Decode makespan / cycles  |               32,548 |  18,508 |          17,536 |  17,536 |

Prefill 的 R1→R2 缩短约 **7.59%**，较旧软件流水版缩短约 **5.28%**；
R4 多占资源却略慢，因此上一版默认使用 R2。Decode 的 R1→R2 缩短约 **5.25%**，
这是相同 packet 布局、相同字节量与计算图下的 multicontext 净收益；
旧版→17,536 的总缩短不能全部记为 multicontext 收益，包含 packet/channel
放置、ready-action 顺序和 split-KV handoff 的变化。额外对照仅将旧版 V base
偏移 64 B、不改变算法或 packet 格式，旧版变为 23,234 cycles。

Perfetto 核验默认 R2：两个 workload 在所有四个 tiles 的 Task lease 峰值均为 2；
Prefill 每 tile 有 125 对跨 UCE context MFE/compute 重叠，Decode 每 tile 有
29 对。它们不是只在 Group 层多提交 root。R4 Prefill 的峰值为 4，但并不更快。
所有报告的 credit/arena zero-leak 检查通过；trace 无 unfinished slices，
raw `X+i` 数与 Perfetto slice 数相等。`slice_spill_overlapping_complete_event`
是同 track 重叠 complete events 的展示 spill，不是 data loss。SQL 中的
pairwise overlap cycles 也不能当作去重后的 wall-clock busy cycles。

### IR-only 优化实测（2026-09-30，R2 对照）

以下是切换默认 R4 前的 IR-only 优化对照；当时 `pipeline_validator/`
源码、硬件 YAML 和 `run.sh` 配置均未修改，使用 full_memory、S1、4 tiles、
R2、DMA channels=2、HBM fixed latency=10。原始 IR 快照和优化 IR
均实际执行 compile→load→run。

| Workload | 原 IR R2 / cycles | 优化 IR R2 / cycles |  缩短 |
| -------- | ----------------: | ------------------: | ----: |
| Prefill  |           515,222 |             502,049 | 2.56% |
| Decode   |            17,536 |              17,007 | 3.02% |

当前图的同-workload R 消融（每个 R 分别重新编译，不增加带宽/队列）：

| Workload | R1 / cycles | R2 / cycles | R4 / cycles |
| -------- | ----------: | ----------: | ----------: |
| Prefill  |     544,216 |     502,049 |     506,126 |
| Decode   |      17,877 |      17,007 |      17,007 |

R1→R2 分别缩短 7.75% / 4.87%；R4 没有净收益。随后按运行配置要求将默认切为 R4。
Prefill R4 的每 tile lease 峰值为 4，Decode R4 为 3（不是四路同时 compute）。

Prefill：tile 0 的首个 attention Task 入驻从 244,622 提前到 231,301。
QKV BOA 与 result store 的重叠从零增加到每 tile **13,056 cycles**
（init 1,632 + accum 11,424）；所有接口 traffic 和 L1/L2 reservations/peaks
均未增加。attention 原有的 BOA/EVU overlap 保留：
每 tile EVU service 25,216 cycles，其中 25,084 cycles 已与 BOA 重叠
（99.48%）。R2 的两波 attention residence 不是缺失四路 pipeline 的证据。
额外 score 双缓冲仅节省 270 cycles、额外 weight 双缓冲无进一步收益，
因此都不纳入最终 IR。

Decode：K append 的 Task 在 10,830 入驻，tile 0 的 BOA slice 在 11,419..11,487
运行，此时 WV 的 HBM read 尚未完成；原版本 append Task 要到 15,960
才入驻。最终 V append 与 block 7 在不同 contexts 的 Task leases 重叠
**370..402 cycles/tile**（tile 0 为 370）。这是分阶段 overlap；不能声称所有聚合 overlap
指标都上升（tile 0 的跨-context MFE/compute pairwise overlap 从
772 降为 735 cycles）。末块 KV 仍受 HBM channel 竞争限制，
本轮没有通过修改 simulator 或增加带宽掩盖这一点。

原始快照、完整 trace/report 与独立 Perfetto 审计位于
`examples/artifacts/transformer_multicontext_ir_optimization/`：
`prefill_before` / `prefill_final_r2`、`decode_before` / `decode_after`，
以及 `final_evidence.json` 的原始报告指标、Perfetto SQL/raw 对照和源码 hash。
两个最终报告均通过完成、credit invariant 和 arena zero-leak 检查；
每 tile 峰值为两个 Task，结束时无残留 leases/arenas/backings/views。
Perfetto 的 `X+i` / counter 数量与 raw JSON 完全匹配，无 unfinished slices；
仅有同 track overlapping-slice spill，无 data loss。

### 当前默认 R4

两个 IR 的 `requested_contexts_per_tile`、生成器 API/CLI 默认值和
`run.sh --context-mode` 均已切为 4。Decode 的 partition pins 使用 contexts
0/1/2/3；Prefill 的 attention L1 allowed modes 由四-context 包络重算为 `[0]`。
Device 层仍只提交一个 root，未修改 validator 的编译器或模拟器实现。
现有回归测试从 IR resource contract 读取 context 数和并发上限，不再写死 R2。
上面的 R2 指标保留为历史对照；当前默认运行产物使用 `*_default_r4` 命名。

本次默认入口实际运行：Prefill **506,126 cycles**，Decode **17,007 cycles**；
完成、credit invariant 和 arena zero-leak 检查均通过。Prefill 每 tile 的 Task lease
峰值为 4，Decode 为 3；后者不意味着配置仍为 R3，而是 R4 容量未全部同时入驻。
回归测试为 **7 passed, 1 skipped**；完整 trace/report 和
`default_r4_evidence.json` 位于上述产物目录。

重建当前默认 R4 结果：

```bash
mkdir -p examples/artifacts/transformer_multicontext_ir_optimization
bash examples/run.sh transformer-prefill-attention-multicontext \
  --memory-trace \
  --trace-json examples/artifacts/transformer_multicontext_ir_optimization/prefill_default_r4.trace.json \
  --report examples/artifacts/transformer_multicontext_ir_optimization/prefill_default_r4.report.json --json
bash examples/run.sh transformer-decode-kv-multicontext \
  --memory-trace \
  --trace-json examples/artifacts/transformer_multicontext_ir_optimization/decode_default_r4.trace.json \
  --report examples/artifacts/transformer_multicontext_ir_optimization/decode_default_r4.report.json --json
examples/artifacts/tools/trace_processor_shell \
  examples/artifacts/transformer_multicontext_ir_optimization/prefill_default_r4.trace.json \
  -q examples/generators/analyze_multicontext.sql
```

## Dispatch 并行对照：`transformer_prefill_attention_dispatchparallel_multicontext.mlir`

问题：multicontext 基线的三个 producer 程序（`qkv_chunk_init` /
`qkv_chunk_accum` / `prefill_outproj_tile`）靠 **tile 程序内 software
pipeline** 拿吞吐——Q store 与下一条 K BOA co-issue、`input_released` 在最后
一条 BOA 之前提前发出让根级提前换装。本对照把这套 pipeline 从 producer 里
**全部拆掉**，改用 dispatch 数量换并行：

- producer 内不再流水：任意时刻在飞的 L1 load ≤2（权重 fill + 首个 X fill，
  然后 X + 一份 partial）；每条 store 在下一条 BOA 之前 drain 完；
  `input_released` 在最后一个 L2 load 之后才发。
- 并行改由 dispatch 提供：Q 投影（写 `q_l2`）与 K/V 投影（写 `k_l2`/`v_l2`）
  是独立分配，每个 input-K chunk 拆成两个互不争用的 Grid——16 个 QKV
  dispatch 组成两条独立的 8 步链；每个 query block 的输出投影按 64 行
  低/高两半各发一个 Grid（各自的输出 buffer），8 个互不争用的 outproj
  dispatch；OUT 物理布局相应改为 `[qblock, half, head, row, feat]`
  （每 (qblock,half) 一条整 buffer store，字节数与基线一致）。
- attention 程序与基线逐 op 相同（归一化 SSA 后零 diff，有测试锁定）。
- 28 个 Grid（基线 20），`GRID_WINDOW=8` 退休节流保持在 16 条 route 表内。

### 实测（2026-09-30，R=4，同一套 flag）

| 指标                           |    软件流水基线 |   dispatch 并行 |      变化 |
| ------------------------------ | --------------: | --------------: | --------: |
| cycles                         |         506,126 |         488,827 | **−3.4%** |
| utilization                    |          100.0% |          100.0% |      持平 |
| BOA active cycles（有用 MACs） |       1,844,224 |       1,844,224 |        0% |
| EVU active cycles              |         100,864 |         100,864 |        0% |
| MFE active cycles              |         245,386 |         305,512 |      +24% |
| hbm_read / hbm_write_bytes     | 6.29 / 1.05 MiB | 6.29 / 1.05 MiB |      持平 |
| dependency_stall_cycles        |       1,959,475 |       1,889,727 |     −3.6% |
| memory_stall_cycles            |          97,550 |          60,866 |    −37.6% |
| l2_read_bytes                  |      35,127,296 |      47,710,208 |    +35.8% |

trace 证据：Q 链（Grid 4→199µs）与 K/V 链（5→214µs）重叠 193µs——两条链
确实并行；`prefill_outproj_lo/hi`（345→478 / 345→486µs）同样重叠；每 tile
Task lease 峰值 4。producer 拆成 Q/KV 两条链后，X staging 每 chunk 被
两个 Grid 各读一次，MFE 的 L2→L1 读因此多约 2 MiB（+24%），但换来的是
BOA 在 100% utilization 下更少的依赖停顿。

结论：**在这份时序模型里，dispatch 级并行（多 Grid、独立 buffer）比
producer 内 software pipeline 更有效**——把 pipeline 从 producer 拆掉、
用 Q/KV 双链 + outproj 分半补回来，净收益 3.4%。前提是拆出来的 Grid
必须写独立分配（整 buffer WAW 会把行/头维度的拆分重新串行化）。

## 边界

Timing model only：`tile.boa.async` / `tile.evu.async` 不携带 tensor
operand、无数值执行。这些 workload 验证调度、生命周期与流量，不验证
Transformer 数值正确性；上述性能数字是 validator 时序模型的输出，不是
硬件性能承诺。
