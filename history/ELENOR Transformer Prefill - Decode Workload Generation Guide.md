# ELENOR Nexus-V3 Transformer Workload Generation Guide

## 0. 目标

基于当前 `nexus-v3` 分支，为 `Pipeline Validator` 新增两个代表 Transformer 推理行为的 workload：

1. `transformer_prefill_attention_pipeline.mlir`
2. `transformer_decode_kv_pipeline.mlir`

目的不是验证 Transformer 数值正确性，而是验证 ELENOR 当前调度模型在 Transformer workload 下的：

- Group ready-action scheduling；
- Tile multi-context；
- HBM → L2 → L1 数据流水；
- BOA / EVU / MFE overlap；
- L1/L2 buffer 生命周期；
- async event dependency；
- input_released / output_ready；
- L2 SRAM 容量与带宽；
- steady-state pipeline efficiency。

当前 validator 是 timing/resource model。

`tile.boa.async`、`tile.evu.async` 不携带真实 tensor operand，也不进行数值运算，因此：

**不要声称 workload 验证了 Transformer numerical correctness。**

必须在文件顶部注释这一点。

---

# 1. 必须遵守当前 Nexus-V3 的约束

不要新增 IR semantic。

必须严格使用当前已有的：

```text
nexus.*
nest.*
tile.*
```

主要约束：

### 1.1 Task 数量

当前：

```text
task.range count == popcount(placement)
```

因此如果：

```mlir
placement = 15
```

代表 4 Tile，则使用：

```mlir
%tasks = nest.task.range from = 0 to = 4
```

不要生成一个 dispatch 内 8/16/32 task。

更大的计算必须：

- 在 Tile Program 内静态展开；
- 或拆成多个 dispatch/context。

---

### 1.2 Global memory 必须使用 block-packed layout

V1 `nest.subview` 只支持：

```text
unit stride
contiguous row-major region
```

因此禁止直接使用普通逻辑布局后生成 stride view，例如不要试图从：

```text
K[seq, head, dim]
```

任意抽出：

```text
K[:, head, :]
```

而应该提前使用物理 block-packed layout，例如：

```text
K[kv_block][tile][token][head_dim]
```

让一个 KV block：

```mlir
offsets = [block_id, 0, 0, 0]
sizes   = [1, 4, BLOCK_TOKENS, HEAD_DIM]
```

成为连续区域。

---

### 1.3 使用现有资源合同

每个：

```mlir
nest.context
```

必须声明：

```mlir
#nest.context_resources<
    l2_mode = ...,
    allowed_profiles = [...],
    logical_tasks = ...,
    l2_spm_bytes = ...,
    requested_contexts_per_tile = ...
>
```

每个：

```mlir
tile.program
```

必须声明：

```mlir
#tile.resources<
    allowed_profiles = [...],
    tile_l1_spm_bytes_per_context = ...
>
```

资源值必须通过当前 compiler resource calculation 得到。

**不要手工随便填一个足够大的数字。**

---

# 2. Example A：Transformer Prefill

文件：

```text
examples/workloads/transformer_prefill_attention_pipeline.mlir
```

---

# 3. Prefill 的设计目标

这个 workload 要回答：

> 对于规则、计算密集、静态 Transformer Prefill，当前 ELENOR 是否能够通过 Group/Tile 异步执行保持 BOA 有足够工作，同时让 DMA/MFE 隐藏在计算之后？

不要一开始生成整个 LLM。

只生成：

```text
one Transformer Attention block
```

建议包含：

```text
Input X
   │
   ▼
Q/K/V projection
   │
   ├──── Q
   ├──── K
   └──── V
        │
        ▼
Blocked Attention
   QKᵀ
    ↓
 online softmax
    ↓
   PV
    │
    ▼
Attention Output
    │
    ▼
Output Projection
```

FFN 暂时不要放进第一个版本。

因为：

```text
Linear → activation → Linear
```

本质已经可以由现有 Matmul workload 覆盖。

第一版应该集中验证 Attention 数据流。

---

# 4. Prefill 推荐 shape

使用：

```text
batch          = 1
sequence       = 512
hidden_dim     = 1024

query_heads    = 16
kv_heads       = 4
head_dim       = 64

dtype          = BF16
accumulator    = FP32 where appropriate
```

即 GQA：

```text
4 query heads
      ↓
share
      ↓
1 KV head
```

4 个 KV head 正好映射到：

```text
Tile 0 → KV head 0 → Q heads 0..3
Tile 1 → KV head 1 → Q heads 4..7
Tile 2 → KV head 2 → Q heads 8..11
Tile 3 → KV head 3 → Q heads 12..15
```

这个映射非常适合作为当前 4-Tile Group 的 Transformer workload。

---

# 5. Prefill Attention tiling

推荐：

```text
QUERY_BLOCK = 128 tokens
KV_BLOCK    = 128 tokens
HEAD_DIM    = 64
```

所以：

```text
sequence = 512

query blocks = 4
KV blocks    = 4
```

每个 Tile：

```text
4 query heads
×
128 query tokens
×
128 KV tokens
```

完成一个 attention subproblem。

不要物化完整：

```text
512 × 512 attention matrix
```

必须采用 blocked/online attention。

---

# 6. Prefill 的 L1 pipeline

每个 Tile Program 建议：

```text
Q block             resident
running softmax m   resident
running softmax l   resident
output accumulator  resident

KV buffer 0         ping
KV buffer 1         pong
```

逻辑：

```text
load KV0 → buffer0
load KV1 → buffer1

await KV0

QK(KV0)
online-softmax-update
PV(KV0)

load KV2 → buffer0

QK(KV1)
online-softmax-update
PV(KV1)

load KV3 → buffer1

...
```

必须体现：

```text
MFE load(KV i+1)
       ||
BOA/EVU(KV i)
```

---

# 7. Prefill Group-level 生命周期

这个 workload 最值得验证的不是单个 GEMM，而是 L2 资源生命周期。

推荐组织：

```text
Stage 0:
    prefetch X
    prefetch Wq/Wk/Wv

Stage 1:
    dispatch QKV projection

Stage 2:
    Q/K/V ready

    release:
        X
        Wq
        Wk
        Wv

Stage 3:
    attention dispatches

    meanwhile:
        prefetch Wo

Stage 4:
    attention output ready

    release:
        Q
        K
        V

Stage 5:
    output projection

Stage 6:
    store final output
```

关键要求：

**不要等待整个 QKV Context 完全结束以后才开始下一阶段的所有内存工作。**

例如：

```text
QKV output ready
       │
       ├── attention can start
       │
       └── QKV weights can release
                 │
                 └── Wo prefetch can start
```

利用：

```text
input_released
output_ready
grid_done
```

的不同语义。

---

# 8. Prefill Tile Program

可以参考现有：

```text
matmul_2048x512x64_boa256x256x32.mlir
```

的 K-tiling 方式。

Q/K/V projection 与 output projection 都应使用类似：

```mlir
%a_loaded = tile.load.async ...
%b_loaded = tile.load.async ...

tile.await %a_loaded, %b_loaded

%compute = tile.boa.async "matmul"
    m = ...
    n = ...
    k = ...
    ops = ...

tile.await %compute
```

如果 K 被分块：

```text
K0
K1
K2
...
```

第一步：

```text
accumulate = false
```

后续：

```text
accumulate = true
```

---

# 9. Prefill Attention Tile Program

建议单独定义：

```text
@prefill_attention_tile
```

每个 Tile：

```text
task → one KV head
```

内部静态展开：

```text
for query_block = 0..3
    for kv_block = 0..3
```

由于当前没有 loop IR：

**必须静态展开。**

但生成代码时应使用 Python builder/generator，不要人工复制粘贴几百行。

概念：

```text
load Q(query block)

load K0/V0
load K1/V1

QK0
softmax_update0
PV0

load K2/V2
QK1
softmax_update1
PV1

load K3/V3
QK2
...

QK3
softmax_update3
PV3

store attention output
```

注意：

```text
QK 和 PV 都使用 BOA
```

不能假定：

```text
QK(i+1)
||
PV(i)
```

能在同一个 Tile BOA 上并发。

计算 resource II 时必须使用：

```text
BOA_time_per_KV_block =
    QK_time + PV_time
```

---

# 10. Prefill 必须采集的指标

不要把当前 aggregate `utilization` 作为主要结果。

至少输出：

```text
total_cycles

BOA active cycles
EVU active cycles
MFE active cycles

HBM bytes
L2 read bytes
L2 write bytes
L1 transfer bytes

active_context_peak

admission stall cycles
dependency stall cycles
engine queue stall cycles
memory stall cycles
```

另外计算：

```text
useful_matrix_utilization

= useful_BF16_matrix_flops
  /
  (theoretical_BOA_peak × total_time)
```

Prefill 主要观察这个指标。

---

# 11. Prefill 必须做的 configuration sweep

至少运行：

```text
Tile context_count:
    1
    2
    4

Device outstanding:
    1
    2
    4

Group scheduler:
    S0
    S1
```

其他参数保持不变。

输出：

```text
config
total_cycles
speedup
boa_busy
effective_matrix_util
mfe_busy
memory_stall
dependency_stall
```

目标不是强行得到 80%。

目标是回答：

```text
Context 增加后什么时候收益饱和？

S1 相比 S0 是否真正减少等待？

Prefill 最终受 BOA、MFE、L2 还是调度限制？
```

---

# 12. Example B：Decode + KV Cache

文件：

```text
examples/workloads/transformer_decode_kv_pipeline.mlir
```

这个例子比 Prefill 更重要。

主要目标不是 BOA utilization。

目标是：

> 测量 KV cache block pipeline 的 steady-state initiation interval。

---

# 13. Decode 推荐 shape

使用：

```text
batch             = 1
hidden_dim        = 1024

query_heads       = 16
kv_heads          = 4
head_dim          = 64

valid_sequence    = 2048
KV_BLOCK          = 256

num_kv_blocks     = 8
```

因此：

```text
2048 / 256 = 8 blocks
```

仍采用 GQA：

```text
Tile 0 → KV head 0 → 4 Q heads
Tile 1 → KV head 1 → 4 Q heads
Tile 2 → KV head 2 → 4 Q heads
Tile 3 → KV head 3 → 4 Q heads
```

这是推荐的 mapping。

---

# 14. KV Cache global layout

不要使用逻辑 layout：

```text
[sequence][head][dim]
```

建议使用：

```text
K_CACHE[8][4][256][64]
V_CACHE[8][4][256][64]
```

含义：

```text
[kv_block][tile/kv_head][token][head_dim]
```

这样一个 HBM→L2 block 可以直接：

```mlir
offsets = [block_id, 0, 0, 0]
sizes   = [1, 4, 256, 64]
```

成为连续传输。

Tile 内：

```mlir
tile.subview task_dim = 0
```

得到：

```text
[1][256][64]
```

即该 Tile 对应 KV head 的 block。

---

# 15. Decode 不要一次把整个 KV cache prefetch 到 L2

禁止：

```text
prefetch all 2048-token K/V
await
dispatch attention
```

这无法验证真正的 KV pipeline。

必须采用：

```text
L2 ping/pong buffer
```

例如：

```text
K0
V0

K1
V1
```

两个 buffer set。

---

# 16. Decode Group Pipeline

当前 V3 没有通用 loop，所以 **8 个 KV block 静态展开**。

推荐：

```text
prefetch KV block 0 → ping
prefetch KV block 1 → pong

dispatch block 0
    depends_on(prefetch0)

prefetch block 2 → ping
    depends_on(block0.input_released)

dispatch block 1
    depends_on(prefetch1,
               block0.output_ready)

prefetch block 3 → pong
    depends_on(block1.input_released)

dispatch block 2
    depends_on(prefetch2,
               block1.output_ready)

...
```

核心结构：

```text
time ------------------------------------------------>

HBM:
      KV0      KV1      KV2      KV3      KV4

L2:
      ping     pong     ping     pong

Tile:
               ATT0     ATT1     ATT2     ATT3
```

其中：

```text
prefetch KV(i+2)
```

只依赖：

```text
input_released(i)
```

因为 Tile 已经把 block i 搬入 L1 后：

```text
L2 ping/pong buffer
```

就可以安全覆盖。

但：

```text
dispatch(i+1)
```

同时必须依赖：

```text
output_ready(i)
```

因为 online softmax state 存在 loop-carried dependency。

---

# 17. Decode Running State

需要一个很小的 context-local L2 state buffer。

每个 Tile 保存：

```text
4 query heads
```

每个 query head：

```text
m               FP32
l               FP32
output[64]      FP32
```

因此逻辑上类似：

```text
state[4 tiles][4 q_heads][66] f32
```

大小非常小。

每个 KV block dispatch：

```text
load state
load K block
load V block
load Q

signal input_released

QK
online-softmax-update
PV

store updated state

signal output_ready
```

注意：

```text
input_released
```

应该在：

```text
K/V/Q/state input
```

都已经进入 Tile-local storage 后发出。

这样上一轮 ping/pong L2 buffer 可以被下一轮 prefetch 覆盖。

---

# 18. Decode Tile Program

定义：

```text
@decode_attention_block
```

一个 dispatch 只处理：

```text
一个 KV block
```

Tile program 内：

```text
load Q
load K block
load V block
load running state

await loads

signal input_released

BOA:
    Q × Kᵀ

EVU:
    online softmax state update

BOA:
    P × V

store running output/state

await store

signal output_ready
```

不要把 QK 与 PV 当作并行 BOA。

---

# 19. KV Cache append

该 workload 是：

```text
fixed decode step
```

例如：

```text
current_position = 2048
```

QKV projection 产生：

```text
K_new
V_new
```

并写入固定 append location。

因为当前 IR 没有动态地址：

```text
current_position
```

必须静态固定。

文件注释明确说明：

```text
This example represents one decoder iteration at
valid_sequence_length = 2048.

Future device-side loop support may turn the append offset into
a runtime loop-carried value.
```

不要因此修改当前 V3 IR。

---

# 20. Decode 外层 token loop

**本 workload 不实现 token generation loop。**

当前只表示：

```text
one decode step
```

例如：

```mlir
nexus.program @decode_one_token(...) {
    %done =
        nexus.submit_context.async @decode_step(...)

    nexus.await %done
    nexus.return
}
```

未来：

```text
repeat decode_step
```

属于另外的 device-side loop proposal。

不要在这个 PR 中加入。

---

# 21. Decode Pipeline Efficiency

Decode 的主要性能指标：

```text
KV block initiation interval
```

假设第一个 block 完成时间：

```text
T0
```

最后一个：

```text
T7
```

则：

```text
actual_II =
    (T7 - T0) / 7
```

再计算：

```text
ideal_resource_II =
    max(
        HBM_service_time_per_block,
        DMA_service_time_per_block,
        L2_service_time_per_block,
        MFE_service_time_per_block,
        BOA_QK_time + BOA_PV_time,
        EVU_time,
        recurrence_minimum
    )
```

pipeline efficiency：

```text
pipeline_efficiency =
    ideal_resource_II / actual_II
```

这是 Decode workload 最重要的指标。

---

# 22. Decode 同时输出以下指标

必须输出：

```text
kv_block_actual_ii
kv_block_reference_ii
pipeline_efficiency

effective_kv_bandwidth

HBM bytes
L2 bytes
L1 bytes

MFE busy
BOA busy
EVU busy

memory exposed stall
dependency stall
buffer reuse stall
admission stall
```

其中：

```text
effective_kv_bandwidth =
    useful_KV_bytes /
    steady_state_time
```

不要用：

```text
BOA utilization
```

作为 Decode 优劣的主要评价。

Decode 很可能：

```text
pipeline efficiency high
BOA utilization moderate/low
```

这是完全合理的。

---

# 23. Decode 还需要一个非常重要的参数 Sweep

至少：

```text
KV_BLOCK =
    64
    128
    256
    512
```

观察：

```text
HBM effective bandwidth
KV block II
pipeline efficiency
L1 footprint
```

这样可以找到：

```text
transaction latency 太大
        ↓
需要更大的 block

block 太大
        ↓
L1 双缓冲压力过高
```

之间的 sweet spot。

---

# 24. Decode 多请求扩展

默认：

```text
num_requests = 1
```

额外支持可选 benchmark：

```text
num_requests =
    1
    2
    4
```

不同 request 之间没有依赖。

例如：

```text
decode request 0
decode request 1
decode request 2
decode request 3
```

通过：

```text
nexus.submit_context.async
```

连续提交，中间不要 `await`。

目的：

> 验证单请求 Decode memory-bound 时，多 Context 是否可以利用其他请求填补 MFE / BOA / EVU 空泡。

这里测的是：

```text
aggregate token throughput
```

不是单请求 latency。

必须同时报告：

```text
single-request latency
aggregate tokens/s
```

避免吞吐与 latency 混淆。

---

# 25. 必须建立 baseline

两个 workload 都至少要有：

```text
baseline
optimized
```

### Prefill baseline

```text
所有 load 完成
→ compute
→ 所有 store

context_count = 1
scheduler = S0
```

### Prefill optimized

```text
async prefetch
ready-action
double buffering
context_count > 1 where useful
scheduler = S1
```

---

### Decode baseline

禁止 KV prefetch/compute overlap：

```text
prefetch block i
await
compute block i
await
prefetch block i+1
```

### Decode optimized

```text
ping/pong prefetch
+
online state dependency
+
next-block DMA overlap
```

这样才能回答：

> ELENOR pipeline 调度到底带来了多少收益。

---

# 26. 不允许做的事情

不要：

1. 修改 simulator 让 benchmark 看起来更快；
2. 新增未设计好的 loop IR；
3. 新增 arbitrary CFG；
4. 假设非连续 subview 合法；
5. 把完整 attention score matrix 全部物化；
6. 为了提高 BOA utilization 做没有意义的重复计算；
7. 用 aggregate PMU utilization 作为核心 KPI；
8. 声称时间模型验证 numerical correctness；
9. 把整个 KV cache 一次性搬入 L2，然后称为 KV streaming pipeline；
10. 让 QK 和 PV 同时占用同一个 Tile 的 BOA；
11. 在 baseline 与 optimized 之间偷偷改变硬件带宽；
12. 在不同测试间改变 tensor shape 后直接比较 speedup。

---

# 27. 推荐参考当前已有 workload

实现时优先模仿：

### BOA tiling / block-packed layout

```text
examples/workloads/
matmul_2048x512x64_boa256x256x32.mlir
```

重点参考：

```text
K tiling
BOA accumulate
block-packed HBM layout
4-task → 4-Tile mapping
```

### L1 double buffering

```text
examples/workloads/
reduce_sum_ktiled_single_context.mlir
```

重点参考：

```text
ping/pong L1 buffer
static unrolled K loop
MFE + EVU overlap
loop-carried accumulator intent
```

### L2 readonly reuse

```text
examples/scenarios/
l2_shared_weight.mlir
```

重点参考：

```text
L2 physical backing lifetime
readonly sharing
input_released
cross-context reuse
```

---

# 28. 建议实现方式

不要手写几千行 MLIR。

建议新增 Python generator/builder，例如：

```text
examples/generators/
    generate_transformer_prefill.py
    generate_transformer_decode.py
```

或者扩展：

```text
pipeline_validator/workload_builders.py
```

实现：

```python
make_prefill_attention(...)
make_decode_kv(...)
```

参数：

```python
@dataclass
class PrefillConfig:
    seq_len: int = 512
    hidden_dim: int = 1024
    q_heads: int = 16
    kv_heads: int = 4
    head_dim: int = 64
    q_block: int = 128
    kv_block: int = 128


@dataclass
class DecodeConfig:
    seq_len: int = 2048
    hidden_dim: int = 1024
    q_heads: int = 16
    kv_heads: int = 4
    head_dim: int = 64
    kv_block: int = 256
```

生成后的 `.mlir` 应提交进 repository，使测试 deterministic。

---

# 29. 测试

增加：

```text
pipeline_validator/tests/
test_transformer_workloads.py
```

至少验证：

```python
def test_prefill_compiles_and_runs():
    ...

def test_decode_compiles_and_runs():
    ...

def test_decode_kv_pingpong_overlap():
    ...

def test_decode_no_buffer_overwrite_before_input_released():
    ...

def test_prefill_l2_lifetime():
    ...

def test_decode_block_count():
    ...

def test_decode_hbm_bytes_match_expected_kv_bytes():
    ...
```

对于 Decode：

```text
seq_len = 2048
kv_heads = 4
head_dim = 64
BF16
```

理论历史 KV 数据量：

K：

```text
2048 × 4 × 64 × 2
= 1 MiB
```

V：

```text
1 MiB
```

因此一轮完整 KV scan 的 minimum useful traffic：

```text
K + V = 2 MiB
```

HBM trace 必须至少能够解释这 2 MiB。

如果明显高于：

```text
2 MiB
```

需要说明额外流量是什么。

---

# 30. 最终报告格式

Agent 完成后输出两个表。

## Prefill

| Metric                 | Baseline | Optimized |
| ---------------------- | -------: | --------: |
| total cycles           |          |           |
| speedup                |      1.0 |           |
| useful BOA utilization |          |           |
| BOA busy               |          |           |
| MFE busy               |          |           |
| HBM bytes              |          |           |
| L2 bytes               |          |           |
| memory stall           |          |           |
| dependency stall       |          |           |
| context peak           |          |           |

## Decode

| Metric                 | Baseline | Pipelined |
| ---------------------- | -------: | --------: |
| token latency          |          |           |
| KV block II            |          |           |
| pipeline efficiency    |          |           |
| effective KV bandwidth |          |           |
| HBM bytes              |          |           |
| L2 bytes               |          |           |
| MFE busy               |          |           |
| BOA busy               |          |           |
| exposed memory stall   |          |           |

---

# 31. 最重要的研究问题

最终不要只回答：

```text
快了多少？
```

而必须回答：

### Prefill

```text
ELENOR 能否在 Transformer 的 compute-heavy phase
保持连续的矩阵计算供给？

瓶颈是 BOA、L2、MFE 还是 scheduler？
```

### Decode

```text
ELENOR 能否把：

HBM KV fetch
L2 movement
L1 movement
QK
softmax
PV

组成接近稳态的 KV-block pipeline？

实际 II 距离资源下界还有多少？
空泡来自哪里？
```

如果这两个问题能够通过 trace 和 PMU 明确回答，这两个 workload 就达到了目的。
