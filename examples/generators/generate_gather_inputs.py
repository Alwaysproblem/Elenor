"""Generate ByteStore input files for the v3 indexed-gather fixtures.

For every gather fixture the entry (nexus.program) formals enumerate the HBM
bindings; each binding gets one ``<binding>.bin`` under
``<output-dir>/<fixture-stem>/``.  Seeding rules (plan §3 item 4):

- ``indices*`` bindings are filled entirely with the repeating ``[3, 0, 2, 1]``
  i32-LE sequence (prefetch reads the whole binding, so every byte must be
  initialised);
- ``table`` bindings get only their first 4096 bytes seeded, byte value
  ``offset % 251`` (the gather row addresses stay inside that prefix);
- ``lhs``/``rhs`` bindings are filled entirely with ``offset % 251``;
- ``acc_init`` bindings are all zero (the accumulator initialiser);
- output bindings are write-only and are not generated.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKLOADS = REPO / "examples" / "workloads"

GATHER_FIXTURES = (
    "gather_indexed",
    "gather_matmul",
    "matmul_gather_add",
    "gather_matmul_4tiles_2contexts",
    "matmul_gather_add_4tiles_2contexts",
)

INDEX_PATTERN = (3, 0, 2, 1)
TABLE_PREFIX_BYTES = 4096
SEED_MODULUS = 251

DTYPE_BYTES = {
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
}


def _binding_size(memref_type) -> int:
  """Return the byte size of one global memref formal."""
  dtype = memref_type.dtype.data
  dims = [int(dim.value.data) for dim in memref_type.dims]
  return prod(dims) * DTYPE_BYTES[dtype]


def prod(values) -> int:
  result = 1
  for value in values:
    result *= value
  return result


def _seed_indices(size: int) -> bytes:
  return b"".join(
    struct.pack("<i", INDEX_PATTERN[i % len(INDEX_PATTERN)]) for i in range(size // 4)
  )


def _seed_modulo(size: int, limit: int | None = None) -> bytes:
  span = size if limit is None else min(size, limit)
  return bytes(offset % SEED_MODULUS for offset in range(span))


def _binding_payload(name: str, size: int) -> bytes | None:
  """Return the input bytes for one binding, or ``None`` when unseeded."""
  if "output" in name or name == "out":
    return None
  if "indices" in name:
    return _seed_indices(size)
  if name.startswith("table"):
    return _seed_modulo(size, limit=TABLE_PREFIX_BYTES)
  if name.startswith("lhs") or name.startswith("rhs"):
    return _seed_modulo(size)
  if "acc_init" in name:
    return b"\x00" * size
  raise ValueError(f"no seeding rule for gather fixture binding '{name}'")


def _entry_bindings(path: Path) -> list[tuple[str, int]]:
  """Enumerate ``(binding_name, byte_size)`` pairs of one fixture.

  CLI binding names resolve against the nexus.program entry formals (the
  model's top-level HBM globals).
  """
  from pipeline_validator.dialects.elenor import NestContextOp, NestGlobalMemref, NexusProgramOp
  from pipeline_validator.workload_ir import load_workload_ir

  module = load_workload_ir(str(path))
  entry = next(
    (op for op in module.body.block.ops if isinstance(op, NexusProgramOp)), None
  ) or next(
    (op for op in module.body.block.ops if isinstance(op, NestContextOp)), None
  )
  if entry is None:
    raise ValueError(f"{path}: no nexus.program or nest.context entry")
  bindings: dict[str, int] = {}
  for arg in entry.body.block.args:
    if isinstance(arg.type, NestGlobalMemref):
      name = arg.name_hint or ""
      bindings[name] = _binding_size(arg.type)
  return sorted(bindings.items())


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir",
    default=str(WORKLOADS / "gather_inputs"),
    help="directory that receives <fixture-stem>/<binding>.bin files",
  )
  args = parser.parse_args()

  output_root = Path(args.output_dir)
  written = 0
  for stem in GATHER_FIXTURES:
    fixture = WORKLOADS / f"{stem}.mlir"
    for name, size in _entry_bindings(fixture):
      payload = _binding_payload(name, size)
      if payload is None:
        continue
      target = output_root / stem / f"{name}.bin"
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_bytes(payload)
      written += 1
      print(f"{target.relative_to(output_root)} ({len(payload)} of {size} bytes)")
  print(f"wrote {written} input files under {output_root}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
