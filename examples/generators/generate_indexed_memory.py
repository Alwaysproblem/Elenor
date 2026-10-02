"""Generate the standalone indexed gather/scatter scenario (plan section 6).

Writes ``examples/workloads/indexed_gather_scatter.mlir`` plus the ByteStore
input files ``indexed_memory_data/{data,gather_indices,scatter_indices}.bin``:

- DATA is a ``[4, 4]i32`` table with row values ``[10..13] / [20..23] /
  [30..33] / [40..43]`` (i32 little-endian);
- GATHER_IDX ``= [3, 0, 3]`` gathers rows 3, 0 and 3 again (48 B payload,
  row 0 read twice);
- SCATTER_IDX ``= [1, 1, 2]`` writes rows 1, 1 and 2 (row 1 written twice:
  ordinal 1 overwrites ordinal 0, so the surviving row-1 value comes from the
  second scatter segment);
- OUT is left unseeded: only the scattered rows are read back afterwards.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKLOADS = REPO / "examples" / "workloads"

DATA_ROWS = (
  (10, 11, 12, 13),
  (20, 21, 22, 23),
  (30, 31, 32, 33),
  (40, 41, 42, 43),
)
GATHER_INDICES = (3, 0, 3)
SCATTER_INDICES = (1, 1, 2)

TEMPLATE = REPO / "examples" / "generators" / "templates" / "indexed_gather_scatter.mlir"
MLIR_TEMPLATE = TEMPLATE.read_text(encoding="utf-8")


def _display(path: Path) -> str:
  """Repo-relative path when the output lives inside the repo, else absolute."""
  try:
    return str(path.relative_to(REPO))
  except ValueError:
    return str(path)


def _data_bytes() -> bytes:
  return b"".join(struct.pack("<4i", *row) for row in DATA_ROWS)


def _indices_bytes(indices) -> bytes:
  return b"".join(struct.pack("<i", value) for value in indices)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir",
    default=str(WORKLOADS),
    help="directory receiving the workload IR and data files",
  )
  args = parser.parse_args()

  output_root = Path(args.output_dir)
  output_root.mkdir(parents=True, exist_ok=True)

  mlir_path = output_root / "indexed_gather_scatter.mlir"
  mlir_path.write_text(MLIR_TEMPLATE)
  print(f"wrote {_display(mlir_path)} ({len(MLIR_TEMPLATE)} bytes)")

  data_dir = output_root / "indexed_memory_data"
  data_dir.mkdir(parents=True, exist_ok=True)
  payloads = (
    ("data.bin", _data_bytes()),
    ("gather_indices.bin", _indices_bytes(GATHER_INDICES)),
    ("scatter_indices.bin", _indices_bytes(SCATTER_INDICES)),
  )
  for name, payload in payloads:
    target = data_dir / name
    target.write_bytes(payload)
    print(f"wrote {_display(target)} ({len(payload)} bytes)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
