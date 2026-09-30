#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash examples/run.sh list
  bash examples/run.sh <name> [extra pipeline_validator args]
  bash examples/run.sh file <path.mlir> [pipeline_validator args]

Examples:
  bash examples/run.sh gather
  bash examples/run.sh gather-matmul --trace-json /tmp/gather-matmul.json --json
  bash examples/run.sh file examples/workloads/my_model.mlir \
    --input-binding input=0x100000:4096:r
EOF
}

list_examples() {
  cat <<'EOF'
Runnable workloads:
  gather                         workloads/gather_profiled.mlir
  gather-matmul                  workloads/gather_matmul.mlir
  matmul-gather-add              workloads/matmul_gather_add.mlir
  gather-matmul-4tiles-2contexts
                                 workloads/gather_matmul_4tiles_2contexts.mlir
  matmul-2048x512-boa256          workloads/matmul_2048x512x64_boa256x256x32.mlir
                                 (2048x512x64 matmul, BOA 256x256x32, K tile 内
                                 展开, 4 context x placement=15 x 4 task)
  matmul-gather-add-4tiles-2contexts
                                 workloads/matmul_gather_add_4tiles_2contexts.mlir
  matmul-pow-parallel          workloads/matmul_pow_parallel.mlir
                                 (1024x512x64 matmul x2 context + pow(Y) x2
                                 context, 全部 placement=15 并发 submit,
                                 验证 BOA matmul 与 EVU pow 并行)
  matmul-pow-free-slot          workloads/matmul_pow_free_slot.mlir
                                 (4 个 matmul context 占满 slot 后 submit
                                 pow; 验证首个 matmul 提前结束即释放 slot
                                 给 pow 提前调度)
  matmul-pow-data-dep           workloads/matmul_pow_data_dep.mlir
                                 (pow 消费 matmul 输出 C; 验证 data 依赖下
                                 pow 只等它的生产者、不提前也不全串行)
  matmul17-pow-tail-overlap     workloads/matmul17_pow_tail_overlap.mlir
                                 (M=4352 → 17 个 tile context = 4 x
                                 placement15 + 1 x placement1 尾块；验证
                                 尾块独占 tile0 时 pow 提前占用其余资源)
   pow-dual-context               workloads/pow_dual_context.mlir
   pow-dual-context-mixed-shapes  workloads/pow_dual_context_mixed_shapes.mlir
   pow-sequential-contexts        workloads/pow_sequential_contexts.mlir
  reduce-sum-single-context       workloads/reduce_sum_ktiled_single_context.mlir
                                 (256x4096 bf16 Reduce-Sum，单 context：1 个
                                 dispatch 4 task，8x64KiB chunk 双缓冲 +
                                 f32 acc；时间模型，无数值执行)
  reduce-sum-splitk-multicontext  workloads/reduce_sum_splitk_multicontext.mlir
                                 (256x16384 bf16 Reduce-Sum，split-K 跨
                                 context：8 producer context + HBM 部分和
                                 combine)
  reduce-sum-multiuce             workloads/reduce_sum_multiuce.mlir
                                 (512x4096 bf16 Reduce-Sum，UCE supertile x
                                 task 行块两级 M 分工，全 tile 4 UCE context
                                 并发；--context-mode 4)
  reduce-sum-gpu-tree             workloads/reduce_sum_gpu_tree.mlir
                                 (256x4096 bf16 Reduce-Sum，GPU reduce-tree：
                                 4 个 leaf dispatch 按 K 配对分区 + 两级
                                 合并树；--context-mode 4)
  reduce-sum-splitk-multiuce      workloads/reduce_sum_splitk_multiuce.mlir
                                 (256x4096 bf16 Reduce-Sum，全 tile 4 UCE
                                 context 沿 reduce axis 切分（k_step 配对）
                                 + context-local L2 扁平合并；--context-mode 4)
  matmul-splitk-pipeline          workloads/matmul_splitk_pipeline.mlir
                                 (256x256x512 matmul，reduce-K tiling 三级
                                 流水：4 个 split-K leaf + scratch 合并
                                 dispatch；--context-mode 4)
  matmul-splitk-multicontext-pipeline
                                 workloads/matmul_splitk_multicontext_pipeline.mlir
                                 (512x512x512 matmul，M/N 2x2 四 context，
                                 每个 context 的 split-K partial 在 L2 本地合并；
                                 --context-mode 4 --device-context-mode 4)

  transformer-prefill-attention
                                 workloads/transformer_prefill_attention_pipeline.mlir
                                 (512 token prefill attention block，单 root：
                                 QKV K-chunk 流水 + blocked attention
                                 (4x4, L1 KV ping/pong) + Wo prefetch overlap +
                                 N-split output projection；timing-only)
  transformer-prefill-attention-baseline
                                 workloads/transformer_prefill_attention_baseline.mlir
                                 (同一 workload 的逐 chunk 串行对照：
                                 prefetch -> await -> dispatch -> await)
  transformer-decode-kv          workloads/transformer_decode_kv_pipeline.mlir
                                 (decode step @valid_len=2048，8 个 KV block
                                 L2 ping/pong 流水 + online softmax state +
                                 fixed-position KV append；测 KV block II)
  transformer-decode-kv-baseline workloads/transformer_decode_kv_baseline.mlir
                                 (逐 block prefetch->await->dispatch->await
                                 串行对照)
  transformer-prefill-attention-multicontext
                                 workloads/transformer_prefill_attention_multicontext.mlir
                                 (同一 prefill 的 4 个独立 query-block grids；
                                 R=4，非复制 4 个请求；--context-mode 4)
  transformer-prefill-attention-dispatchparallel
                                 workloads/transformer_prefill_attention_dispatchparallel_multicontext.mlir
                                 (producer 程序去掉 software pipeline（在飞 load ≤2、
                                 store 即时 drain、input_released 在最后一个 load 后），
                                 并行改由 dispatch 数提供：Q 链 ∥ K/V 链 16 个 QKV
                                 dispatch + 每块 outproj lo/hi 共 8 个；attention tail
                                 与基线逐 op 相同)
  transformer-decode-kv-multicontext
                                 workloads/transformer_decode_kv_multicontext.mlir
                                 (单请求 split-KV 4 partitions + stable softmax
                                 merge；KV packet 64B gap 分散 HBM channel；
                                 --context-mode 4)

Protocol scenarios:
  ready-action-branch           scenarios/ready_action_branch.mlir
  device-dependency-submit      scenarios/device_dependency_submit.mlir
  l2-admission-wait              scenarios/l2_admission_wait.mlir
  sequential-release-counterexample
                                 scenarios/sequential_release_counterexample.mlir
  profile-reconfiguration         scenarios/profile_reconfiguration.mlir
  l2-profile-switch-load-ordering
                                 scenarios/l2_profile_switch_load_ordering.mlir
  l2-admission-profile-switch    scenarios/l2_admission_profile_switch.mlir
                                 (A mode0 完整 store/pow 后，compiler 自动等
                                 完整 root completion frontier 再切 L2
                                 mode1，B 在切档命令完成后才准许/加载)
  l2-shared-weight               scenarios/l2_shared_weight.mlir
                                 (loader 一次 HBM->L2 prefetch W 并 publish,
                                 两个 reader 各 4 tile 从同一 backing 读,
                                 共享只 8192 B; 私有对照为 l2-private-weight)
  l2-shared-fanout               scenarios/l2_shared_fanout.mlir
                                 (A 用 tile.load/store 在 L2 造 X 并 publish,
                                 无 X 的 HBM binding; B/C 借用同一 backing)
  l2-private-weight              scenarios/l2_private_weight.mlir
                                 (对照: 两个 reader 各自私有 prefetch W,
                                 HBM->L2 流量 16384 B vs 共享 8192 B)
EOF
  printf '\nNEST subgraphs (timing/lifetime, not tensor numerics):\n'
  local model stem
  for model in "$ROOT_DIR"/examples/scenarios/nest_subgraphs/*.mlir; do
    [[ -f "$model" ]] || continue
    stem="${model##*/}"
    stem="${stem%.mlir}"
    printf '  nest-%-36s scenarios/nest_subgraphs/%s.mlir\n' "${stem//_/-}" "$stem"
  done
}

name="${1:-list}"
if [[ "$name" == "list" ]]; then
  list_examples
  exit 0
fi
if [[ "$name" == "help" || "$name" == "--help" || "$name" == "-h" ]]; then
  usage
  exit 0
fi
shift

case "$name" in
  gather)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/gather_profiled.mlir" \
      --hw-override num_dma_channels=2 \
      --input-binding table=0x200000:8388608:r \
      --input-binding indices=0xA00000:4096:r \
      --input-binding output=0xB00000:256:w \
      --max-cycles 200000 \
      "$@"
    ;;
  gather-matmul)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/gather_matmul.mlir" \
      --hw-override num_dma_channels=2 \
      --input-binding lhs=0x100000:16384:r \
      --input-binding rhs=0x110000:16384:r \
      --input-binding table=0x200000:8388608:r \
      --input-binding indices=0xA00000:4096:r \
      --input-binding output=0xB00000:32768:w \
      --max-cycles 200000 \
      "$@"
    ;;
  matmul-gather-add)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/matmul_gather_add.mlir" \
      --hw-override num_dma_channels=2 \
      --input-binding lhs=0x100000:16384:r \
      --input-binding rhs=0x110000:16384:r \
      --input-binding table=0x200000:8388608:r \
      --input-binding indices=0xA00000:4096:r \
      --input-binding output=0xB00000:32768:w \
      --max-cycles 200000 \
      "$@"
    ;;
  gather-matmul-4tiles-2contexts)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/gather_matmul_4tiles_2contexts.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 2 \
      --device-context-mode 2 \
      --input-binding table=0x200000:8388608:r \
      --input-binding lhs0=0x100000:65536:r \
      --input-binding rhs0=0x120000:65536:r \
      --input-binding indices0=0x140000:256:r \
      --input-binding output0=0xB00000:131072:w \
      --input-binding lhs1=0x150000:65536:r \
      --input-binding rhs1=0x170000:65536:r \
      --input-binding indices1=0x190000:256:r \
      --input-binding output1=0xD00000:131072:w \
      --max-cycles 500000 \
      "$@"
    ;;
  matmul-2048x512-boa256)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/matmul_2048x512x64_boa256x256x32.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --device-context-mode 4 \
      --input-binding A=0x100000:262144:r \
      --input-binding B=0x150000:65536:r \
      --input-binding C=0x200000:2097152:w \
      --max-cycles 500000 \
      "$@"
    ;;
  matmul-pow-parallel)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/matmul_pow_parallel.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --device-context-mode 4 \
      --input-binding A=0x100000:131072:r \
      --input-binding B=0x120000:65536:r \
      --input-binding C=0x200000:1048576:w \
      --input-binding Y0=0x400000:262144:rw \
      --input-binding Y1=0x440000:262144:rw \
      --max-cycles 500000 \
      "$@"
    ;;
  matmul-pow-free-slot)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/matmul_pow_free_slot.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --device-context-mode 4 \
      --input-binding A=0x100000:262144:r \
      --input-binding B=0x150000:65536:r \
      --input-binding C=0x200000:2097152:w \
      --input-binding Y0=0x400000:262144:rw \
      --input-binding Y1=0x440000:262144:rw \
      --max-cycles 500000 \
      "$@"
    ;;
  matmul-pow-data-dep)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/matmul_pow_data_dep.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --device-context-mode 4 \
      --input-binding A=0x100000:262144:r \
      --input-binding B=0x150000:65536:r \
      --input-binding C=0x200000:2097152:rw \
      --max-cycles 500000 \
      "$@"
    ;;
  matmul17-pow-tail-overlap)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/matmul17_pow_tail_overlap.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 5 \
      --device-context-mode 5 \
      --input-binding A=0x100000:655360:r \
      --input-binding B=0x1A0000:32768:r \
      --input-binding C=0x200000:2621440:rw \
      --max-cycles 500000 \
      "$@"
    ;;
  matmul-gather-add-4tiles-2contexts)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/matmul_gather_add_4tiles_2contexts.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 2 \
      --device-context-mode 2 \
      --input-binding table=0x200000:8388608:r \
      --input-binding lhs0=0x100000:65536:r \
      --input-binding rhs0=0x120000:65536:r \
      --input-binding indices0=0x140000:256:r \
      --input-binding output0=0xB00000:131072:w \
      --input-binding lhs1=0x150000:65536:r \
      --input-binding rhs1=0x170000:65536:r \
      --input-binding indices1=0x190000:256:r \
      --input-binding output1=0xD00000:131072:w \
      --max-cycles 500000 \
      "$@"
    ;;
  pow-dual-context)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/pow_dual_context.mlir" \
      --hw-override num_dma_channels=2 \
      --device-context-mode 2 \
      --context-mode 2 \
      --input-binding Y0=0x100000:131072:rw \
      --input-binding Y1=0x200000:131072:rw \
      --hw-override hbm_fixed_latency_cycles=10 \
      --max-cycles 200000 \
      "$@"
    ;;
  pow-dual-context-mixed-shapes)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/pow_dual_context_mixed_shapes.mlir" \
      --hw-override num_dma_channels=2 \
      --device-context-mode 2 \
      --context-mode 2 \
      --input-binding Y0=0x100000:131072:rw \
      --input-binding Y1=0x200000:262144:rw \
      --hw-override hbm_fixed_latency_cycles=10 \
      --max-cycles 200000 \
      "$@"
    ;;
  pow-sequential-contexts)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/pow_sequential_contexts.mlir" \
      --hw-override num_dma_channels=2 \
      --input-binding Y0=0x100000:131072:rw \
      --input-binding Y1=0x200000:131072:rw \
      --hw-override hbm_fixed_latency_cycles=10 \
      --max-cycles 200000 \
      "$@"
    ;;
  reduce-sum-single-context)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/reduce_sum_ktiled_single_context.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --input-binding X=0x100000:2097152:r \
      --input-binding Y=0x1100000:1024:w \
      --max-cycles 500000 \
      "$@"
    ;;
  reduce-sum-splitk-multicontext)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/reduce_sum_splitk_multicontext.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --device-context-mode 4 \
      --input-binding X=0x100000:8388608:r \
      --input-binding Y_part=0x1000000:8192:rw \
      --input-binding Y=0x1100000:1024:w \
      --max-cycles 500000 \
      "$@"
    ;;
  reduce-sum-multiuce)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/reduce_sum_multiuce.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --input-binding X=0x100000:4194304:r \
      --input-binding Y=0x1100000:2048:w \
      --max-cycles 500000 \
      "$@"
    ;;
  reduce-sum-gpu-tree)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/reduce_sum_gpu_tree.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --input-binding X=0x100000:2097152:r \
      --input-binding Y=0x1100000:1024:w \
      --input-binding S=0x1200000:8192:rw \
      --max-cycles 500000 \
      "$@"
    ;;
  reduce-sum-splitk-multiuce)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/reduce_sum_splitk_multiuce.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --input-binding X=0x100000:2097152:r \
      --input-binding Y=0x1100000:1024:w \
      --max-cycles 500000 \
      "$@"
    ;;
  matmul-splitk-pipeline)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/matmul_splitk_pipeline.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --input-binding A=0x100000:262144:r \
      --input-binding B=0x1000000:262144:r \
      --input-binding S=0x1400000:1048576:rw \
      --input-binding C=0x1800000:262144:w \
      --max-cycles 500000 \
      "$@"
    ;;
  matmul-splitk-multicontext-pipeline)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/matmul_splitk_multicontext_pipeline.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --device-context-mode 4 \
      --input-binding A=0x100000:524288:r \
      --input-binding B=0x200000:524288:r \
      --input-binding C=0x300000:1048576:w \
      --max-cycles 500000 \
      "$@"
    ;;
  l2-admission-wait)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/l2_admission_wait.mlir" \
      --hw-config "$ROOT_DIR/examples/configs/profile_l2_256k.yaml" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 2 \
      --device-context-mode 2 \
      --input-binding A_IN=0x100000:131072:rw \
      --input-binding A_OUT=0x200000:131072:rw \
      --input-binding B_IN=0x300000:131072:rw \
      --max-cycles 500000 \
      "$@"
    ;;
  sequential-release-counterexample)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/sequential_release_counterexample.mlir" \
      --hw-config "$ROOT_DIR/examples/configs/profile_l2_256k.yaml" \
      --hw-override num_dma_channels=2 \
      --input-binding YA_in=0x100000:131072:rw \
      --input-binding YA_out=0x200000:131072:rw \
      --input-binding YB=0x300000:131072:rw \
      --hw-override hbm_fixed_latency_cycles=10 \
      --max-cycles 200000 \
      "$@"
    ;;
  nest-*)
    stem="${name#nest-}"
    if [[ ! "$stem" =~ ^[a-z0-9_-]+$ ]]; then
      echo "error: unknown example '$name'" >&2
      exit 2
    fi
    stem="${stem//-/_}"
    model_path="$ROOT_DIR/examples/scenarios/nest_subgraphs/$stem.mlir"
    if [[ ! -f "$model_path" ]]; then
      echo "error: unknown example '$name'" >&2
      exit 2
    fi
    nest_args=(
      --ir-file "$model_path"
      --hw-override num_dma_channels=2
      --hw-override hbm_fixed_latency_cycles=10
      --context-mode 4
      --device-context-mode 4
      --max-cycles 2000000
    )
    case "$stem" in
      *_single|*_single_*) nest_args+=(--device-context-mode 1) ;;
    esac
    case "$stem" in
      t01_single_one_uce)
        nest_args+=(--context-mode 1 --device-context-mode 1) ;;
      n03_wait_capacity|n09_impossible)
        nest_args+=(--hw-config "$ROOT_DIR/examples/configs/profile_l2_64k.yaml") ;;
      n04_fifo_hol|s09_single_exact)
        nest_args+=(--hw-config "$ROOT_DIR/examples/configs/profile_l2_192k.yaml") ;;
      s09_single_short)
        nest_args+=(--hw-config "$ROOT_DIR/examples/configs/profile_l2_short.yaml") ;;
      s07_node_c100|s12_node_v100_evu)
        nest_args+=(--max-cycles 8000000) ;;
    esac
    case "$stem" in
      t04_*)
        nest_args+=(
          --input-binding R0=0x1000000:8388608:rw
          --input-binding R1=0x2000000:8388608:rw
          --input-binding R2=0x3000000:8388608:rw
          --input-binding R3=0x4000000:8388608:rw
          --input-binding W=0x5000000:32768:r
        ) ;;
      *) nest_args+=(--input-binding arena=0x1000000:8388608:rw) ;;
    esac
    set -- "${nest_args[@]}" "$@"
    ;;
  ready-action-branch)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/ready_action_branch.mlir" \
      --context-mode 2 \
      --input-binding arena=0x100000:8192:rw \
      --hw-override hbm_fixed_latency_cycles=10 \
      --max-cycles 200000 \
      "$@"
    ;;
  device-dependency-submit)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/device_dependency_submit.mlir" \
      --context-mode 2 \
      --device-context-mode 2 \
      --input-binding src=0x100000:2048:r \
      --input-binding a=0x110000:2048:rw \
      --input-binding b=0x120000:2048:w \
      --input-binding c=0x130000:2048:w \
      --hw-override hbm_fixed_latency_cycles=10 \
      --max-cycles 200000 \
      "$@"
    ;;
  profile-reconfiguration)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/profile_reconfiguration.mlir" \
      --context-mode 1 \
      --device-context-mode 1 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --max-cycles 500000 \
      "$@"
    ;;

  l2-admission-profile-switch)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/l2_admission_profile_switch.mlir" \
      --hw-config "$ROOT_DIR/examples/configs/profile_l2_256k_switch.yaml" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 2 \
      --device-context-mode 2 \
      --input-binding A_IN=0x100000:131072:rw \
      --input-binding A_OUT=0x200000:131072:rw \
      --input-binding B_IN=0x300000:131072:rw \
      --max-cycles 500000 \
      "$@"
    ;;
  l2-shared-weight)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/l2_shared_weight.mlir" \
      --hw-config "$ROOT_DIR/examples/configs/profile_l2_256k.yaml" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 2 \
      --device-context-mode 2 \
      --input-binding W=0x100000:8192:r \
      --input-binding B_OUT=0x200000:32768:w \
      --input-binding C_OUT=0x300000:32768:w \
      --max-cycles 500000 \
      "$@"
    ;;
  l2-shared-fanout)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/l2_shared_fanout.mlir" \
      --hw-config "$ROOT_DIR/examples/configs/profile_l2_256k.yaml" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 2 \
      --device-context-mode 2 \
      --input-binding A_IN=0x100000:8192:r \
      --input-binding B_OUT=0x200000:32768:w \
      --input-binding C_OUT=0x300000:32768:w \
      --max-cycles 500000 \
      "$@"
    ;;
  l2-private-weight)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/l2_private_weight.mlir" \
      --hw-config "$ROOT_DIR/examples/configs/profile_l2_256k.yaml" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 2 \
      --device-context-mode 2 \
      --input-binding W=0x100000:8192:r \
      --input-binding B_OUT=0x200000:32768:w \
      --input-binding C_OUT=0x300000:32768:w \
      --max-cycles 500000 \
      "$@"
    ;;
  l2-profile-switch-load-ordering)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/l2_profile_switch_load_ordering.mlir" \
      --hw-config "$ROOT_DIR/examples/configs/profile_l2_switch.yaml" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 2 \
      --device-context-mode 2 \
      --input-binding A_IN=0x100000:131072:r \
      --input-binding A_OUT=0x200000:131072:rw \
      --input-binding B_IN=0x300000:131072:r \
      --max-cycles 300000 \
      "$@"
    ;;
  transformer-prefill-attention)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/transformer_prefill_attention_pipeline.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --input-binding X=0x1000000:1048576:r \
      --input-binding WQ=0x2000000:2097152:r \
      --input-binding WK=0x3000000:524288:r \
      --input-binding WV=0x3100000:524288:r \
      --input-binding WO=0x4000000:2097152:r \
      --input-binding OUT=0x5000000:1048576:w \
      --max-cycles 2000000 \
      "$@"
    ;;
  transformer-prefill-attention-baseline)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/transformer_prefill_attention_baseline.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --group-policy s0 \
      --input-binding X=0x1000000:1048576:r \
      --input-binding WQ=0x2000000:2097152:r \
      --input-binding WK=0x3000000:524288:r \
      --input-binding WV=0x3100000:524288:r \
      --input-binding WO=0x4000000:2097152:r \
      --input-binding OUT=0x5000000:1048576:w \
      --max-cycles 2000000 \
      "$@"
    ;;
  transformer-decode-kv)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/transformer_decode_kv_pipeline.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --input-binding K_CACHE=0x1000000:1048576:r \
      --input-binding V_CACHE=0x2000000:1048576:r \
      --input-binding Q_IN=0x3000000:2048:r \
      --input-binding S_INIT=0x3001000:4224:rw \
      --input-binding OUT=0x3010000:4096:w \
      --input-binding H_T=0x3011000:2048:r \
      --input-binding WK_A=0x3012000:524288:r \
      --input-binding WV_A=0x3092000:524288:r \
      --input-binding K_APPEND=0x3200000:512:w \
      --input-binding V_APPEND=0x3201000:512:w \
      --max-cycles 400000 \
      "$@"
    ;;
  transformer-decode-kv-baseline)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/transformer_decode_kv_baseline.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --input-binding K_CACHE=0x1000000:1048576:r \
      --input-binding V_CACHE=0x2000000:1048576:r \
      --input-binding Q_IN=0x3000000:2048:r \
      --input-binding S_INIT=0x3001000:4224:rw \
      --input-binding OUT=0x3010000:4096:w \
      --input-binding H_T=0x3011000:2048:r \
      --input-binding WK_A=0x3012000:524288:r \
      --input-binding WV_A=0x3092000:524288:r \
      --input-binding K_APPEND=0x3200000:512:w \
      --input-binding V_APPEND=0x3201000:512:w \
      --max-cycles 400000 \
      "$@"
    ;;
  transformer-prefill-attention-multicontext)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/transformer_prefill_attention_multicontext.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --group-policy s1 \
      --input-binding X=0x1000000:1048576:r \
      --input-binding WQ=0x2000000:2097152:r \
      --input-binding WK=0x3000000:524288:r \
      --input-binding WV=0x3100000:524288:r \
      --input-binding WO=0x4000000:2097152:r \
      --input-binding OUT=0x5000000:1048576:w \
      --max-cycles 2000000 \
      "$@"
    ;;
  transformer-prefill-attention-dispatchparallel)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/transformer_prefill_attention_dispatchparallel_multicontext.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --group-policy s1 \
      --input-binding X=0x1000000:1048576:r \
      --input-binding WQ=0x2000000:2097152:r \
      --input-binding WK=0x3000000:524288:r \
      --input-binding WV=0x3100000:524288:r \
      --input-binding WO=0x4000000:2097152:r \
      --input-binding OUT=0x5000000:1048576:w \
      --max-cycles 2000000 \
      "$@"
    ;;
  transformer-decode-kv-multicontext)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/transformer_decode_kv_multicontext.mlir" \
      --hw-override num_dma_channels=2 \
      --hw-override hbm_fixed_latency_cycles=10 \
      --context-mode 4 \
      --group-policy s1 \
      --input-binding K_CACHE_mc=0x1000000:1049088:r \
      --input-binding V_CACHE_mc=0x2000040:1049088:r \
      --input-binding Q_IN_mc=0x3000000:2048:r \
      --input-binding S_INIT_STATE_mc=0x3001000:512:r \
      --input-binding S_INIT_OUT_mc=0x3002000:16384:r \
      --input-binding OUT_mc=0x3010000:4096:w \
      --input-binding H_T_mc=0x3011000:2048:r \
      --input-binding WK_A_mc=0x3012000:524288:r \
      --input-binding WV_A_mc=0x3092000:524288:r \
      --input-binding K_APPEND_mc=0x3200000:512:w \
      --input-binding V_APPEND_mc=0x3201000:512:w \
      --max-cycles 400000 \
      "$@"
    ;;
  file)
    if [[ $# -lt 1 ]]; then
      echo "error: file mode requires a .mlir path" >&2
      usage >&2
      exit 2
    fi
    model_path="$1"
    shift
    set -- --ir-file "$model_path" "$@"
    ;;
  *)
    echo "error: unknown example '$name'" >&2
    list_examples >&2
    exit 2
    ;;
esac

exec conda run -n elenor-validator python -m pipeline_validator "$@"
