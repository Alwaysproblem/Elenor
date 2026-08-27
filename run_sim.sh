#!/bin/bash

# conda run -n elenor-validator python -m pipeline_validator.cli --workload tiled_matmul --sim-override fidelity=runtime --trace-json tiled_matmul_runtime.json --hw-override num_dma_channels=1 --print-ir
  # --sim-override fidelity=full_memory \

conda run -n elenor-validator \
  python -m pipeline_validator.cli \
  --trace-json examples/pow.json \
  --sim-override fidelity=runtime \
  --hw-override num_dma_channels=2 \
  --context-mode 4 \
  --device-context-mode 4 \
  --ir-file examples/example.mlir \
  --input-binding Y0=0x100000:262144:rw \
  --input-binding Y1=0x200000:262144:rw
