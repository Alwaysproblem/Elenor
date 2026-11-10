"""Numeric probe driver (C011): pack directed vectors into tile runs and compare to the reference.

Split in two halves so the comparison is testable without a container:

- `build_groups` packs `known_vectors.json` cell vectors into tile runs. Vector i of a group is
  placed at tile cell (i, i); the A operand of that cell is a[k][i] and the B operand b[k][i], so
  two vectors never share a row or a column. The other 16*16 - n cells are cross-checks: they pair
  unrelated operands and are compared against the same model.
- `compare` checks a probe record (real RTL output) against `numeric_ref`, cell by cell, and names
  the first differing cell. `tamper_selftest` proves the comparison notices a one-bit change.

The container half (`run_container.sh` equivalent) lives in the `numeric-probe` handler; it only
executes the overlay bench and writes the record. It never computes an expected value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

import numeric_ref as nr  # noqa: E402

DIM = 16
CELL_KINDS = ("cell", "ablation", "tk_sequence")
# The C006 bench registers these two BF16 kernels; both compute the same tile semantics and differ
# only in the accumulator base (mt0 / mt8).
IMPLS = ("vtfmm_bf16_mt0", "vtfmm_bf16_mt8")


class ProbeError(Exception):
  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code


def build_groups(vectors: dict[str, Any], impl: str = IMPLS[0]) -> list[dict[str, Any]]:
  """Pack cell vectors into groups of at most 16, one group per tk value per 16 vectors."""
  if impl not in IMPLS:
    raise ProbeError("IMPL_UNKNOWN", impl)
  cells = [v for v in vectors["vectors"] if v["kind"] in CELL_KINDS]
  groups: list[dict[str, Any]] = []
  for tk in (1, 2):
    chosen = [v for v in cells if v["tk"] == tk]
    for start in range(0, len(chosen), DIM):
      chunk = chosen[start : start + DIM]
      a_rows = [[0] * DIM for _ in range(nr_rows())]
      b_rows = [[0] * DIM for _ in range(nr_rows())]
      c_init = [0] * (DIM * DIM)
      placed: list[dict[str, Any]] = []
      for slot, vec in enumerate(chunk):
        pairs = vec["pairs_bf16_bits"]
        for k in range(nr_rows()):
          a_rows[k][slot], b_rows[k][slot] = pairs[k]
        c_init[slot * DIM + slot] = vec["old_fp32_bits"]
        placed.append({"vector": vec["id"], "cell": [slot, slot]})
      groups.append(
        {
          "group": len(groups),
          "tk": tk,
          "impl": impl,
          "a_bf16_bits": a_rows[0] + a_rows[1],
          "b_bf16_bits": b_rows[0] + b_rows[1],
          "c_init_u32": c_init,
          "placed": placed,
        }
      )
  return groups


def nr_rows() -> int:
  return 2


def model_group(group: dict[str, Any]) -> list[int]:
  a = [group["a_bf16_bits"][:DIM], group["a_bf16_bits"][DIM:]]
  b = [group["b_bf16_bits"][:DIM], group["b_bf16_bits"][DIM:]]
  return nr.vtfmm_tile(a, b, group["c_init_u32"], DIM, DIM, group["tk"])


def compare(probe: dict[str, Any], groups: list[dict[str, Any]]) -> list[str]:
  """Compare a probe record to the model. Inputs recorded by the bench must equal what was asked."""
  errors: list[str] = []
  if probe.get("schema") != "cn_p0_numeric_probe/v1":
    return [f"probe schema {probe.get('schema')!r}"]
  recorded = {g["group"]: g for g in probe["groups"]}
  for group in groups:
    rec = recorded.get(group["group"])
    if rec is None:
      errors.append(f"group {group['group']}: missing from the probe record")
      continue
    for key in ("tk", "impl", "a_bf16_bits", "b_bf16_bits", "c_init_u32"):
      if rec[key] != group[key]:
        errors.append(f"group {group['group']}: recorded {key} differs from what the driver sent")
    want = model_group(group)
    for index, (w, got) in enumerate(zip(want, rec["actual_u32"], strict=True)):
      if w != got:
        errors.append(f"group {group['group']} cell {index}: model {w:#010x} != real RTL {got:#010x}")
        break
  extra = set(recorded) - {g["group"] for g in groups}
  errors.extend(f"group {g}: not requested" for g in sorted(extra))
  return errors


def vector_results(
  probe: dict[str, Any], groups: list[dict[str, Any]], vectors: dict[str, Any]
) -> list[str]:
  """Check each placed vector's expected bits against the REAL RTL cell it landed in."""
  by_id = {v["id"]: v for v in vectors["vectors"]}
  recorded = {g["group"]: g for g in probe["groups"]}
  problems: list[str] = []
  for group in groups:
    rec = recorded.get(group["group"])
    if rec is None:
      problems.append(f"group {group['group']}: missing")
      continue
    for place in group["placed"]:
      vec = by_id[place["vector"]]
      row, col = place["cell"]
      got = rec["actual_u32"][row * DIM + col]
      if got != vec["expected_fp32_bits"]:
        problems.append(
          f"vector {vec['id']} at cell {place['cell']}: real RTL {got:#010x} != hand-derived "
          f"{vec['expected_fp32_bits']:#010x}"
        )
  return problems


def tamper_selftest(probe: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
  """Flip one result bit; the comparison must name that cell."""
  bad = json.loads(json.dumps(probe))
  bad["groups"][0]["actual_u32"][5] ^= 1
  errors = compare(bad, groups)
  hit = any("group 0 cell 5" in e for e in errors)
  return {"detected": hit, "location": [0, 5]}
