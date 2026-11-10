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

"""C011 numeric probe (DUT contract check, NOT the independent oracle).

Reads directed vectors from $P0_NUMERIC_VECTORS (JSON written by the driver), places vector i at
tile cell (i, i), runs the unmodified upstream BF16 matmul kernel once per (tk, init) group and
writes $P0_NUMERIC_OUT with the input bits and the ACTUAL result bits of every cell.
No expected value is computed or asserted here: comparison is done by the driver against the
reference model, so a wrong reference cannot be 'fixed' by editing this bench.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from coralnpu_test_utils.coralflow.runner import coralflow_main, coralflow_test

MM_DIM = 16
MM_BF16_ROWS = 2

_SYMBOLS = [
  "mm_a", "mm_b", "mm_c_init", "mm_out", "mm_tm", "mm_tn", "mm_tk",
  "mm_init_mode", "mm_impl", "vtfmm_bf16_mt0", "vtfmm_bf16_mt8",
]


async def _run(fixture, tk, a_u16, b_u16, c_init_u32, impl):
  await fixture.write("mm_a", a_u16)
  await fixture.write("mm_b", b_u16)
  await fixture.write("mm_c_init", c_init_u32)
  await fixture.write_word("mm_tm", MM_DIM)
  await fixture.write_word("mm_tn", MM_DIM)
  await fixture.write_word("mm_tk", tk)
  await fixture.write_word("mm_init_mode", 1)  # preload mm_c_init through vtmv.t.v
  await fixture.write("mm_out", np.zeros(MM_DIM * MM_DIM, dtype=np.uint32))
  await fixture.write_ptr("mm_impl", impl)
  cycles = await fixture.run_to_halt(timeout_cycles=400000)
  raw = await fixture.read("mm_out", MM_DIM * MM_DIM * 4)
  return np.frombuffer(raw, dtype=np.uint32).copy(), cycles


@coralflow_test
async def p0_numeric_probe_test(fixture):
  vectors = json.loads(Path(os.environ["P0_NUMERIC_VECTORS"]).read_text())
  elf = os.environ["P0_NUMERIC_ELF"]
  await fixture.load_elf_and_lookup_symbols(elf, _SYMBOLS)
  records = []
  for group in vectors["groups"]:
    a = np.array(group["a_bf16_bits"], dtype=np.uint16)
    b = np.array(group["b_bf16_bits"], dtype=np.uint16)
    c = np.array(group["c_init_u32"], dtype=np.uint32)
    actual, cycles = await _run(fixture, group["tk"], a, b, c, group["impl"])
    records.append({
      "group": group["group"], "tk": group["tk"], "impl": group["impl"],
      "a_bf16_bits": [int(x) for x in a], "b_bf16_bits": [int(x) for x in b],
      "c_init_u32": [int(x) for x in c], "actual_u32": [int(x) for x in actual],
      "cycles": int(cycles),
    })
  out = os.environ.get("P0_NUMERIC_OUT", "")
  if out:
    Path(out).write_text(json.dumps({
      "schema": "cn_p0_numeric_probe/v1",
      "elf_sha256": hashlib.sha256(Path(elf).read_bytes()).hexdigest(),
      "groups": records,
    }, indent=2) + "\n")
    print("[P0_NUMERIC] written " + out)


if __name__ == "__main__":
  coralflow_main()
