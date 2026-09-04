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
  matmul-gather-add-4tiles-2contexts
                                 workloads/matmul_gather_add_4tiles_2contexts.mlir
  pow-dual-context               workloads/pow_dual_context.mlir
  pow-dual-context-mixed-shapes  workloads/pow_dual_context_mixed_shapes.mlir
  pow-sequential-contexts        workloads/pow_sequential_contexts.mlir

Protocol scenarios:
  l2-admission-wait              scenarios/l2_admission_wait.mlir
  sequential-release-counterexample
                                 scenarios/sequential_release_counterexample.mlir
EOF
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
