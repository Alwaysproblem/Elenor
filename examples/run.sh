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

Protocol scenarios:
  l2-admission-wait              scenarios/l2_admission_wait.mlir
  sequential-release-counterexample
                                 scenarios/sequential_release_counterexample.mlir
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
      --sim-override fidelity=full_memory \
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
      --sim-override fidelity=full_memory \
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
      --sim-override fidelity=full_memory \
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
      --sim-override fidelity=full_memory \
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
      --sim-override fidelity=full_memory \
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
      --sim-override fidelity=full_memory \
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
      --sim-override fidelity=full_memory \
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
      --sim-override fidelity=full_memory \
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
      --sim-override fidelity=full_memory \
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
      --sim-override fidelity=full_memory \
      --max-cycles 500000 \
      "$@"
    ;;
  pow-dual-context)
    set -- \
      --ir-file "$ROOT_DIR/examples/workloads/pow_dual_context.mlir" \
      --hw-override num_dma_channels=2 \
      --device-context-mode 2 \
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
  l2-admission-wait)
    set -- \
      --ir-file "$ROOT_DIR/examples/scenarios/l2_admission_wait.mlir" \
      --hw-override num_dma_channels=2 \
      --sim-override fidelity=full_memory \
      --hw-override group_sram_bytes=262144 \
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
      --sim-override fidelity=full_memory
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
        nest_args+=(--hw-override group_sram_bytes=65536) ;;
      n04_fifo_hol|s09_single_exact)
        nest_args+=(--hw-override group_sram_bytes=196608) ;;
      s09_single_short)
        nest_args+=(--hw-override group_sram_bytes=196352) ;;
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
