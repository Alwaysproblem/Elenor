# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""P0 BF16 KAT capture (C006).

Runs the four exactly-precise bf16 matmul KATs through the unmodified
upstream VmeCoreMiniAxi Verilator model and writes
$P0_BF16_KAT_OUT/coralnpu_kat.json with full case records: input bits,
accumulator preload, actual result bits, per-case cycle_count and the ELF
hash. Operands are fixed quarter-integers, so replay is bit- and
cycle-identical regardless of seed; the reference is recomputed here in
exact fp64 arithmetic. SPIKE-independent by construction: the C047 oracle
contract is untouched.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

MM_DIM = 16
# bf16 matmul stages A/B as 2 register rows each (v8/v12, v16/v20); row slots
# >= tk are masked off by the hardware.
MM_BF16_ROWS = 2

_SYMBOLS = [
  "mm_a",
  "mm_b",
  "mm_c_init",
  "mm_out",
  "mm_tm",
  "mm_tn",
  "mm_tk",
  "mm_init_mode",
  "mm_impl",
  "vtfmm_bf16_mt0",
  "vtfmm_bf16_mt8",
]

CASES = [
  {"impl": "vtfmm_bf16_mt0", "init": 0, "tm": 16, "tn": 16, "tk": 2},
  {"impl": "vtfmm_bf16_mt8", "init": 1, "tm": 16, "tn": 16, "tk": 2},
  {"impl": "vtfmm_bf16_mt0", "init": 1, "tm": 16, "tn": 16, "tk": 1},
  {"impl": "vtfmm_bf16_mt8", "init": 1, "tm": 9, "tn": 11, "tk": 2},
]

# Deterministic operand tables (NOT random): quarter-integers in [-4, 4],
# exact in bf16 (mantissa fits) so no rounding is contracted.
_A_ROWS = [
  [((-1) ** ((i + j) % 2)) * ((7 * i + 3 * j) % 9 - 4) for j in range(MM_DIM)]
  for i in range(MM_BF16_ROWS)
]
_B_ROWS = [
  [((-1) ** ((2 * i + j) % 3)) * ((5 * i + 2 * j) % 9 - 4) for j in range(MM_DIM)]
  for i in range(MM_BF16_ROWS)
]


def _fp_ref(a_f32, b_f32, c_init_f32, tm, tn, tk):
  """C[:tm,:tn] += A[:tk,:tm]^T @ B[:tk,:tn]; exact in fp64 quarters."""
  a_rows = a_f32.reshape(MM_BF16_ROWS, MM_DIM)
  b_rows = b_f32.reshape(MM_BF16_ROWS, MM_DIM)
  ref = c_init_f32.astype(np.float64).reshape(MM_DIM, MM_DIM).copy()
  prod = a_rows[:tk, :tm].astype(np.float64).T @ b_rows[:tk, :tn].astype(
      np.float64
  )
  ref[:tm, :tn] += prod
  assert np.all(ref[:tm, :tn] == np.rint(ref[:tm, :tn])), (
    "reference left the exact domain; operand tables changed"
  )
  return ref.astype(np.float32).view(np.uint32)


async def _load_fixture(fixture):
  # ELFs stay inside the upstream package (its visibility is intentionally
  # restricted); the driver exports the bazel-bin path instead of a data dep.
  env_path = os.environ.get("P0_BF16_KAT_ELF", "")
  if not env_path:
    raise RuntimeError(
      "P0_BF16_KAT_ELF not set: the capture driver must export the "
      "absolutely-resolved upstream elf path (cross-package data deps are "
      "rejected by bazel 9 default-private visibility)."
    )
  elf = env_path
  await fixture.load_elf_and_lookup_symbols(elf, _SYMBOLS)
  return elf


async def _run_case(fixture, case, a_u16, b_u16, c_init_u32):
  await fixture.write("mm_a", a_u16)
  await fixture.write("mm_b", b_u16)
  await fixture.write("mm_c_init", c_init_u32)
  await fixture.write_word("mm_tm", case["tm"])
  await fixture.write_word("mm_tn", case["tn"])
  await fixture.write_word("mm_tk", case["tk"])
  await fixture.write_word("mm_init_mode", case["init"])
  # Clear the output buffer so stale data from a previous case cannot pass.
  await fixture.write("mm_out", np.zeros(MM_DIM * MM_DIM, dtype=np.uint32))
  await fixture.write_ptr("mm_impl", case["impl"])
  cycles = await fixture.run_to_halt(timeout_cycles=400000)
  raw = await fixture.read("mm_out", MM_DIM * MM_DIM * 4)
  actual = np.frombuffer(raw, dtype=np.uint32).reshape(MM_DIM, MM_DIM)
  return actual, cycles


async def _capture(fixture):
  elf = await _load_fixture(fixture)
  a_f32 = np.array(_A_ROWS, dtype=np.float32).reshape(MM_BF16_ROWS * MM_DIM)
  b_f32 = np.array(_B_ROWS, dtype=np.float32).reshape(MM_BF16_ROWS * MM_DIM)
  # Round-trip proof: the staged f32 values must survive f32->bf16->f32.
  np.testing.assert_array_equal(
    (a_f32.view(np.uint32) & 0xFFFF0000).view(np.float32), a_f32
  )
  np.testing.assert_array_equal(
    (b_f32.view(np.uint32) & 0xFFFF0000).view(np.float32), b_f32
  )
  a_bits = (a_f32.view(np.uint32) >> 16).astype(np.uint16)
  b_bits = (b_f32.view(np.uint32) >> 16).astype(np.uint16)
  c_init = (np.arange(MM_DIM * MM_DIM, dtype=np.float32) + 3.0).view(np.uint32)

  records = []
  for i, case in enumerate(CASES):
    actual, cycles = await _run_case(fixture, case, a_bits, b_bits, c_init)
    preload = c_init if case["init"] else np.zeros_like(c_init)
    expected = _fp_ref(
      a_f32, b_f32, preload.view(np.float32), case["tm"], case["tn"], case["tk"]
    )
    mismatches = np.argwhere(actual != expected)
    assert mismatches.size == 0, (
      f"case {i} ({case['impl']}): {len(mismatches)} mismatching elements"
    )
    records.append({
      "case": i,
      "impl": case["impl"],
      "tm": case["tm"], "tn": case["tn"], "tk": case["tk"],
      "init_mode": case["init"],
      "a_bf16_bits": [int(x) for x in a_bits],
      "b_bf16_bits": [int(x) for x in b_bits],
      "c_init_u32": [int(x) for x in c_init],
      "actual_u32": [int(x) for x in actual.reshape(-1)],
      "cycles": int(cycles),
    })

  out = os.environ.get("P0_BF16_KAT_OUT", "")
  if out:
    Path(out).write_text(
      json.dumps({
        "schema": "cn_p0_bf16_kat/v1",
        "elf_sha256": hashlib.sha256(Path(elf).read_bytes()).hexdigest(),
        "cases": records,
      }, indent=2) + "\n"
    )
    print(f"[P0_KAT] written {out}")


@coralflow_test
async def p0_bf16_kat_test(fixture):
  """Capture the four exactly-precise bf16 KAT cases with full records."""
  await _capture(fixture)


if __name__ == "__main__":
  coralflow_main()
