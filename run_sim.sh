#!/bin/bash

conda run -n elenor-validator python -m pipeline_validator.cli --workload tiled_matmul --sim-override fidelity=runtime --trace-json tiled_matmul_runtime.json --hw-override num_dma_channels=1 --print-ir
