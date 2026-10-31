"""P08 soft-Conv gate: byte-true im2col + typed BOA Matmul vs the P03 conv reference.

Every correctness assertion runs the real production path
``parse_workload_ir -> compile_program -> serialize/parse artifact ->
load_program -> Simulator(fidelity="full_memory", ByteStore)`` on the
modules produced by ``examples/generators/generate_soft_conv.py`` (the
committed fixtures are proven byte-identical to that generator first).
The im2col expansion moves bytes only through real services:

* every expanded element is read by ``tile.gather.global.async`` from the
  X HBM binding; padding positions read the explicit trailing +0 BF16 word
  of that binding (never an out-of-bounds address, never default-zero L1);
* the reduced matmul is one typed ``tile.boa.async`` AM chain whose per-step
  K depths mirror the P03 ``tile_k`` contribution blocks (each step is a
  single block at the default ``boa_numeric_tile_k=8`` >= case tile_k);
* no EVU launch exists anywhere in the compiled artifact, so no timing-only
  EVU "copy" can masquerade as the expansion path.

Golden authority (tests never import ``hardware/boa/reference``):
``hardware/boa/reference/vectors/conv.json`` (P03) for the seven P03 cases
and ``examples/workloads/soft_conv_inputs/pad1_stride1/golden.json`` for the
generated pad=1/stride=1 alias case (produced by the generator through the
unchanged P03 ``conv2d_reference``).  ``conv-i8-out`` runs the real P08
FINAL epilogue path (``inv_scale``/``zero_point`` on the AM allocation with
an i8 output binding); the epilogue syntax is available, so no case is
xfail-ed.

Counter <-> analytic mapping (single task, ``l1_mode = 0``/``l2_mode = 0``
so both caches are disabled and every gathered element is exactly one
direct HBM read; ``E = M*K``, ``P = E-V`` padding elements, ``S`` steps,
``W = M*O*out_width``)::

  analytic quantity              | counters that report it
  -------------------------------+---------------------------------------------
  V valid patch reads (elements) | the in-bounds portion of ``gather_requests``
  P padding zero elements        | no counter splits V from P: P is proven on
                                 | the IDX stream itself (zero-word index
                                 | count, test_indices_stay_in_bounds...);
                                 | with both caches off EVERY gather request
                                 | is charged to ``gather_cache_bypass_requests``
                                 | (== E == ``gather_requests``), V and P alike
  valid patch bytes 2V + padding | ``2E`` of ``hbm_read_bytes`` (gather direct
  bytes 2P (= 2V + 2P = 2E)      | response legs, == 2V + 2P)
  im2col writes 2E               | PMU ``gather_bytes`` and the ``l1_write_bytes``
                                 | gather destination-write legs (both == 2E)
  matmul A read 2E               | no byte counter exists: BOA operand fetch is
                                 | a timing-only roofline (``BOAEngine.breakdown``
                                 | charges ``m*k*2 + k*n*2 + m*n*4``; engines
                                 | never move bytes).  Byte truth of the A operand
                                 | is asserted directly: the debug variant stores
                                 | each ``%a_s`` L1 buffer to HBM between the
                                 | awaited BOA step and ``tile.free`` -- the only
                                 | prior writer is the awaited gather, so the
                                 | captured bytes are exactly the operand bytes at
                                 | matmul execution -- and they equal P03 ``a_hex``.
  matmul B read 2*K*O            | ``hbm_read_bytes`` (B prefetch leg) plus the
                                 | ``l2_read_bytes``/``local_dma_bytes``/
                                 | ``l1_write_bytes`` B tile-load legs
  C write M*O*width              | the C ``l1_read``/``l2_write``/``local_dma``
                                 | (tile.store) legs plus the C ``l2_read``/
                                 | ``noc_request``/``global_dma_bytes``/
                                 | ``hbm_write_bytes`` (global store) legs
  index stream 4*E bytes         | ``l1_read_bytes`` (INDEX_READ legs) plus the
                                 | 4E IDX prefetch/tile-load legs
  op multiset                    | ``issued_by_op``: 2 prefetch + 2S tile_load +
                                 | E (index_read + gather_direct_response +
                                 | gather_dest_write) + 1 tile_store +
                                 | 1 global_store (debug A capture adds S
                                 | tile_store + 1 global_store and 2E per moved
                                 | leg, accounted in the same table)

Standalone CLI, the engine behind the ``hardware/scripts/check soft-conv``
container target (layer F; see ``hardware/scripts/soft_conv_cases.py``)::

  python -m pipeline_validator.tests.test_soft_conv \\
      --case {int8-chain, param-reject, stage-matrix, conv-pad-stride-dilation,
              conv-traffic} --out DIR --seed N [--config PATH]

``--config`` is an optional strict JSON document (unknown keys rejected) with
``{"schema": 1, "profile": "bf16_fp32_mac_rne_v1", "epilogue": PATH,
"matmul": PATH}``; the conv golden paths are fixture identity (the committed
generator inputs) and are not overridable.  Reports contain only
deterministic content; wall-clock time lands in ``<case>.timing.json``
(``deterministic: false``).  Result kinds are kept separate: ``production-path``
(Simulator end to end), ``epilogue-function parity`` (production
``boa_numeric.apply_epilogue`` on the committed AM), ``blocked-unreachable-am``
(explicit per-vector record with proof class) and ``negative``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import struct
import sys
import time
from fractions import Fraction as _Fraction
from pathlib import Path

import pytest
from xdsl.utils.exceptions import ParseError, VerifyException

from pipeline_validator.boa_numeric import BoaEpilogueSpec, apply_epilogue
from pipeline_validator.compiled_program import parse_compiled_program, serialize_compiled_program
from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import ExecTileOp, GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import load_workload_ir, parse_workload_ir, print_workload_ir

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ROOT / "examples" / "workloads"
INPUT_ROOT = WORKLOADS / "soft_conv_inputs"
GENERATOR = ROOT / "examples" / "generators" / "generate_soft_conv.py"

L2_BASE = 0x100000
ALIGN = 256
L1_CONTRACT_BYTES = 32768
SEED = 20261031
DTYPE_WIDTH = {"f32": 4, "bf16": 2, "i8": 1}


def _load_generator():
  generators = ROOT / "examples" / "generators"
  if str(generators) not in sys.path:
    sys.path.insert(0, str(generators))
  spec = importlib.util.spec_from_file_location("generate_soft_conv", GENERATOR)
  module = importlib.util.module_from_spec(spec)
  sys.modules["generate_soft_conv"] = module
  spec.loader.exec_module(module)
  return module


GEN = _load_generator()
CASES = {case.source_id: case for case in GEN.all_cases()}


# ---------------------------------------------------------------------------
# Analytic traffic, expected counters, golden cross-check
# ---------------------------------------------------------------------------


def _analytic(case) -> dict:
  geo = case.geometry
  valid = GEN.valid_patch_elements(geo)
  expanded = geo.m * geo.k
  return {
    "M": geo.m,
    "O": geo.o,
    "K": geo.k,
    "steps": tuple(case.steps),
    "V": valid,
    "E": expanded,
    "P": expanded - valid,
    "valid_patch_read_bytes": 2 * valid,
    "padding_zero_bytes": 2 * (expanded - valid),
    "im2col_write_bytes": 2 * expanded,
    "matmul_a_read_bytes": 2 * expanded,
    "matmul_b_read_bytes": 2 * geo.k * geo.o,
    "output_bytes": geo.m * geo.o * DTYPE_WIDTH[case.out_dtype],
    "idx_bytes": 4 * expanded,
    "x_binding_bytes": 2 * (geo.x_elements + 1),
  }


def _expected_counters(case, *, debug_a: bool) -> tuple[dict, dict, dict]:
  """Return ``(issued_by_op, gather PMU events, byte counters)``."""
  analytic = _analytic(case)
  steps = len(analytic["steps"])
  expanded = analytic["E"]
  idx_bytes = analytic["idx_bytes"]
  b_bytes = analytic["matmul_b_read_bytes"]
  out_bytes = analytic["output_bytes"]
  gather_bytes = 2 * expanded
  debug_bytes = gather_bytes if debug_a else 0
  issued_by_op = {
    "prefetch": 2,
    "tile_load": 2 * steps,
    "index_read": expanded,
    "gather_direct_response": expanded,
    "gather_dest_write": expanded,
    "tile_store": 1 + (steps if debug_a else 0),
    "global_store": 1 + (1 if debug_a else 0),
  }
  gather_events = {
    "gather_requests": expanded,
    "gather_cache_bypass_requests": expanded,
    "gather_index_reads": expanded,
    "gather_bytes": gather_bytes,
    "gather_l1_hits": 0,
    "gather_l2_hits": 0,
    "gather_hbm_misses": 0,
  }
  byte_counters = {
    "hbm_read_bytes": gather_bytes + idx_bytes + b_bytes,
    "global_dma_bytes": idx_bytes + b_bytes + out_bytes + debug_bytes,
    "noc_response_bytes": gather_bytes + idx_bytes + b_bytes,
    "noc_request_bytes": out_bytes + debug_bytes,
    "l2_write_bytes": idx_bytes + b_bytes + out_bytes + debug_bytes,
    "l2_read_bytes": idx_bytes + b_bytes + out_bytes + debug_bytes,
    "local_dma_bytes": gather_bytes + idx_bytes + b_bytes + out_bytes + debug_bytes,
    "l1_write_bytes": gather_bytes + idx_bytes + b_bytes,
    "l1_read_bytes": idx_bytes + out_bytes + debug_bytes,
    "hbm_write_bytes": out_bytes + debug_bytes,
  }
  return issued_by_op, gather_events, byte_counters


def _assert_golden_traffic(case, analytic: dict) -> None:
  traffic = case.expected["traffic"]
  assert analytic["V"] == traffic["valid_patch_elements"]
  assert analytic["P"] == traffic["padding_zero_elements"]
  assert analytic["E"] == traffic["expanded_elements"]
  for key in (
    "valid_patch_read_bytes",
    "padding_zero_bytes",
    "im2col_write_bytes",
    "matmul_a_read_bytes",
    "matmul_b_read_bytes",
    "output_bytes",
  ):
    assert analytic[key] == traffic[key], key


# ---------------------------------------------------------------------------
# Production-path runner
# ---------------------------------------------------------------------------


def _run_conv_case(case, *, debug_a: bool, source_text: str | None = None, source_name: str = "soft-conv"):
  geo = case.geometry
  if source_text is None:
    source_text = GEN.build_source(case, debug_a=debug_a)
  module = parse_workload_ir(print_workload_ir(parse_workload_ir(source_text)))
  hw = HardwareConfig()
  sim = SimConfig(fidelity="full_memory", context_count=1, seed=SEED)
  bindings: dict[str, GlobalBinding] = {}
  seeds: list[tuple[int, bytes]] = []
  cursor = L2_BASE

  def reserve(name: str, size: int, permission: str, data: bytes = b"") -> None:
    nonlocal cursor
    cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
    bindings[name] = GlobalBinding(name, cursor, size, permission)
    if data:
      seeds.append((cursor, data))
    cursor += size

  inputs = INPUT_ROOT / case.case_id
  x_bytes = (inputs / "X.bin").read_bytes()
  b_bytes = (inputs / "B.bin").read_bytes()
  idx_bytes = (inputs / "IDX.bin").read_bytes()
  assert len(x_bytes) == 2 * (geo.x_elements + 1)
  assert len(b_bytes) == 2 * geo.k * geo.o
  assert len(idx_bytes) == 4 * geo.m * geo.k
  reserve("X", len(x_bytes), "r", x_bytes)
  reserve("B", len(b_bytes), "r", b_bytes)
  reserve("IDX", len(idx_bytes), "r", idx_bytes)
  debug_size = 2 * geo.m * geo.k if debug_a else 0
  if debug_a:
    reserve("AD", debug_size, "w")
  reserve("C", geo.m * geo.o * DTYPE_WIDTH[case.out_dtype], "w")

  artifact = compile_program(module, hw, sim, binding_assumptions=bindings, source_name=source_name)
  replay = parse_compiled_program(serialize_compiled_program(artifact))
  loaded = load_program(replay, hw, sim, actual_bindings=bindings)
  store = ByteStore()
  for address, data in seeds:
    store.seed_hbm(address, data)
  result = Simulator(hw, sim, byte_store=store).run(loaded)
  return result, store, bindings, debug_size, loaded


def _tile_programs(loaded) -> list:
  entry = loaded.compiled.entry
  tasks = entry.tasks.values() if hasattr(entry, "tasks") else [entry]
  return [role.tile_program for task in tasks for role in task.role_bindings.values()]


def _assert_no_evu(loaded) -> None:
  for program in _tile_programs(loaded):
    ops = [inst.op for inst in program.insts]
    assert ExecTileOp.LAUNCH_EVU not in ops
    assert all(desc.kind != "EVU" for desc in program.descriptors.values())


def _assert_am_pool(result, steps: int) -> None:
  pools = result.group_snapshot["shared_am_per_tile"]
  assert sum(pool["acquire_count"] for pool in pools) == 1
  assert sum(pool["commit_count"] for pool in pools) == steps
  assert sum(pool["final_count"] for pool in pools) == 1
  assert sum(pool["slot_free_count"] for pool in pools) == 1
  assert all(
    pool["fault_count"] == pool["live_leases"] == pool["pending_requests"] == pool["reserved_slots"] == 0
    and pool["free_slots"] == pool["total_slots"]
    for pool in pools
  )


def _reconstruct_a(case, ad_words: list[int]) -> bytes:
  """Rebuild the [M, K] im2col matrix from the per-step flat A captures."""
  geo = case.geometry
  flat: list[int] = []
  start = 0
  for depth in case.steps:
    flat.extend(ad_words[start : start + geo.m * depth])
    start += geo.m * depth
  words: list[int] = []
  for row in range(geo.m):
    offset = 0
    for depth in case.steps:
      words.extend(flat[offset + row * depth : offset + (row + 1) * depth])
      offset += geo.m * depth
  return b"".join(word.to_bytes(2, "little") for word in words)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_committed_fixtures_and_inputs_match_generator() -> None:
  """Regeneration fidelity: the committed files are exactly the generator output."""
  for case in CASES.values():
    fixture = WORKLOADS / f"soft_conv_{case.case_id}.mlir"
    committed = print_workload_ir(load_workload_ir(fixture))
    rebuilt = print_workload_ir(parse_workload_ir(GEN.build_source(case)))
    assert committed == rebuilt, fixture.name
    inputs = INPUT_ROOT / case.case_id
    assert (inputs / "X.bin").read_bytes() == GEN.x_binding_bytes(case)
    assert (inputs / "B.bin").read_bytes() == GEN.b_binding_bytes(case)
    assert (inputs / "IDX.bin").read_bytes() == GEN.idx_binding_bytes(case)
    assert json.loads((INPUT_ROOT / "pad1_stride1" / "golden.json").read_text())["schema"] == 1


@pytest.mark.parametrize("source_id", sorted(CASES))
def test_indices_stay_in_bounds_and_padding_is_explicit(source_id: str) -> None:
  """E-P08-8: padding reads exactly the +0 tail word, never out of bounds."""
  case = CASES[source_id]
  analytic = _analytic(case)
  _assert_golden_traffic(case, analytic)
  geo = case.geometry
  raw = GEN.idx_binding_bytes(case)
  indices = [int.from_bytes(raw[i * 4 : i * 4 + 4], "little") for i in range(len(raw) // 4)]
  assert len(indices) == analytic["E"]
  assert min(indices) >= 0
  assert max(indices) <= geo.x_elements  # the binding holds x_elements + the +0 word
  zero_refs = sum(1 for index in indices if index == geo.x_elements)
  assert zero_refs == analytic["P"]
  # The step-major index stream decodes back to the reference im2col matrix:
  # step s contributes the column band [k0, k0+depth) of every row.
  reference = GEN.im2col_indices(geo)
  per_step: list[list[int]] = []
  cursor = 0
  for depth in case.steps:
    per_step.append(indices[cursor : cursor + geo.m * depth])
    cursor += geo.m * depth
  rebuilt: list[list[int]] = [[] for _ in range(geo.m)]
  for row in range(geo.m):
    for block, depth in zip(per_step, case.steps, strict=True):
      rebuilt[row].extend(block[row * depth : (row + 1) * depth])
  assert rebuilt == reference


@pytest.mark.parametrize("source_id", sorted(CASES))
def test_soft_conv_output_and_im2col_bytes_match_p03(source_id: str) -> None:
  """E-P08-7: C bytes == P03 golden; A operand bytes == P03 im2col a_hex."""
  case = CASES[source_id]
  analytic = _analytic(case)
  result, store, bindings, debug_size, loaded = _run_conv_case(case, debug_a=True)
  assert result.completed, result.reason
  _assert_am_pool(result, len(analytic["steps"]))
  _assert_no_evu(loaded)

  c_bytes = store.read_hbm(bindings["C"].base_iova, analytic["output_bytes"])
  assert c_bytes == bytes.fromhex(case.expected["output_hex"])

  ad_bytes = store.read_hbm(bindings["AD"].base_iova, debug_size)
  ad_words = [int.from_bytes(ad_bytes[i * 2 : i * 2 + 2], "little") for i in range(debug_size // 2)]
  assert _reconstruct_a(case, ad_words) == bytes.fromhex(case.expected["a_hex"])


@pytest.mark.parametrize("source_id", sorted(CASES))
def test_soft_conv_stage_traffic_counters(source_id: str) -> None:
  """E-P08-8: per-stage bytes equal the independent analytic counts."""
  case = CASES[source_id]
  analytic = _analytic(case)
  fixture = WORKLOADS / f"soft_conv_{case.case_id}.mlir"
  result, store, bindings, _, loaded = _run_conv_case(
    case, debug_a=False, source_text=fixture.read_text(encoding="utf-8"), source_name=fixture.name
  )
  assert result.completed, result.reason
  _assert_am_pool(result, len(analytic["steps"]))
  _assert_no_evu(loaded)
  assert store.read_hbm(bindings["C"].base_iova, analytic["output_bytes"]) == bytes.fromhex(
    case.expected["output_hex"]
  )

  issued_by_op, gather_events, byte_counters = _expected_counters(case, debug_a=False)
  transfers = result.group_snapshot["memory"]["transfers"]
  assert transfers["issued_by_op"] == issued_by_op
  assert transfers["byte_counters"] == byte_counters
  events = result.pmu.events
  for key, value in gather_events.items():
    assert events.get(key, 0) == value, key

  # The debug-A capture variant adds exactly its own legs to the same table.
  debug_result, _, _, _, _ = _run_conv_case(case, debug_a=True)
  assert debug_result.completed, debug_result.reason
  debug_ops, debug_gather, debug_bytes = _expected_counters(case, debug_a=True)
  debug_transfers = debug_result.group_snapshot["memory"]["transfers"]
  assert debug_transfers["issued_by_op"] == debug_ops
  assert debug_transfers["byte_counters"] == debug_bytes
  for key, value in debug_gather.items():
    assert debug_result.pmu.events.get(key, 0) == value, key




# ---------------------------------------------------------------------------
# hardware/scripts/check soft-conv CLI (P08 layer F)
# ---------------------------------------------------------------------------
#
# Five cases mirroring the P06 numeric-profile module CLI:
#
#   int8-chain              i8-output AM chains end to end (production-path):
#                           the matmul.json i8 chains, conv-i8-out through the
#                           gather-im2col conv path, and every epilogue.json i8
#                           vector whose golden AM is producible by a legal
#                           bf16 chain (shared-factor product, exact bf16 sum
#                           split, or -0 underflow touch -- see the blocked
#                           notes below).  Each synthesized vector runs twice:
#                           a raw f32 no-epilogue FINAL whose C bytes must
#                           equal the committed golden AM bit-exactly (every
#                           product factor / summand preserved -- quantization
#                           must never mask a wrong operand), then the
#                           vector's own contribution chain + k=0 FINAL whose
#                           output must equal the committed output_hex.
#                           AM words that no chain can produce get explicit
#                           blocked records with the proof class; nothing is
#                           dropped silently.
#   param-reject            every epilogue.json illegal-parameter vector at its
#                           real boundary (source parse, or compiled-artifact
#                           decode for the fields source text cannot encode),
#                           plus codec tampers on a fresh v6 artifact.
#   stage-matrix            every bias/scale/relu/clip/inv_scale combination in
#                           epilogue.json for f32/bf16/i8 output, full bytes at
#                           the boundary that owns the contract (production
#                           boa_numeric.apply_epilogue on the committed AM --
#                           the exact function the runtime FINAL path calls),
#                           labeled "epilogue-function parity"; representative
#                           chains run the whole Simulator path and are labeled
#                           "production-path".
#   conv-pad-stride-dilation all eight conv fixtures (pad/stride/dilation
#                           coverage incl. asymmetric and pad1-stride1): C bytes
#                           and the captured im2col A bytes equal the goldens.
#   conv-traffic            the exact counter<->analytic mapping of the module
#                           docstring, per case, for the committed fixture run
#                           and the debug-A capture variant.
#
# Why blocked records exist: the production AM only ever holds values computed
# by bf16 FMA chains -- there is no AM-seeding op.  numeric_freeze section 2
# fixes every arithmetic NaN output at 0x7fc00000 without keeping payload or
# sign, so sNaN/payload-NaN golden words are unproducible, full stop.
# Everything else is attempted constructively before any blocked record is
# written: a shared-factor product step (the only shape that preserves a
# sub-normal factor pair such as 0x00000001 = 0x0001 * 0x3780), an exact
# bf16 sum decomposition over INIT/ADD steps (FLT_MAX 0x7f7fffff =
# 0x7f7f + 0x7b7f + 0x777f), and the tiny-negative underflow touch frozen by
# the P03 mm-add-k0-negzero vector for -0 (fma(0x8001, 0x0001, +0) rounds to
# 0x80000000 and a k=0 FINAL keeps the bits).  A missing construction is a
# gate failure, not proof that the word is unreachable.  Only the frozen
# arithmetic NaN rule justifies a blocked record.

CLI_CASES = (
  "int8-chain",
  "param-reject",
  "stage-matrix",
  "conv-pad-stride-dilation",
  "conv-traffic",
)
DEFAULT_EPILOGUE_VECTORS = ROOT / "hardware" / "boa/reference/vectors/epilogue.json"
DEFAULT_MATMUL_VECTORS = ROOT / "hardware" / "boa/reference/vectors/matmul.json"
LABEL_PRODUCTION = "production-path"
LABEL_FUNCTION = "epilogue-function parity"
LABEL_BLOCKED = "blocked-unreachable-am"
LABEL_NEGATIVE = "negative"


def load_config(path: str | None) -> dict:
  config: dict = {
    "profile": "bf16_fp32_mac_rne_v1",
    "epilogue": DEFAULT_EPILOGUE_VECTORS,
    "matmul": DEFAULT_MATMUL_VECTORS,
  }
  if path is None:
    return config
  raw = json.loads(Path(path).read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError("soft-conv config must be a JSON object")
  unknown = sorted(set(raw) - {"schema", "profile", "epilogue", "matmul"})
  if unknown:
    raise ValueError(f"unknown soft-conv config keys: {unknown}")
  if "schema" in raw and (type(raw["schema"]) is not int or raw["schema"] != 1):
    raise ValueError(f"soft-conv config schema must be 1, got {raw['schema']!r}")
  if "profile" in raw and raw["profile"] != config["profile"]:
    raise ValueError(f"soft-conv config profile must be {config['profile']!r}")
  for key in ("epilogue", "matmul"):
    if key in raw:
      if not isinstance(raw[key], str):
        raise ValueError(f"soft-conv config '{key}' must be a path string")
      candidate = Path(raw[key])
      config[key] = candidate if candidate.is_absolute() else ROOT / candidate
  for key in ("epilogue", "matmul"):
    if not Path(config[key]).is_file():
      raise ValueError(f"soft-conv config '{key}' file missing: {config[key]}")
  return config


def _load_vector_document(path) -> dict:
  document = json.loads(Path(path).read_text(encoding="utf-8"))
  if document.get("profile") != "bf16_fp32_mac_rne_v1":
    raise ValueError(f"{path} profile must be bf16_fp32_mac_rne_v1")
  return document


# ---------------------------------------------------------------------------
# AM reachability: which golden AM words can a legal bf16 chain produce?
# ---------------------------------------------------------------------------

_SPECIAL_BF16 = {0x7FC00000: 0x7FC0, 0x7F800000: 0x7F80, 0xFF800000: 0xFF80, 0x00000000: 0x0000}


def _bf16_factor(word: int) -> tuple[int, int] | None:
  """Exact finite bf16 (a, b) with a*b == word, or None.

  The exact product may overflow to Inf or underflow and round to another
  f32 word.  Fraction equality proves that this particular product is
  exactly representable, without assuming every bf16 product is exact.
  """
  target = _Fraction(struct.unpack("<f", struct.pack("<I", word))[0])
  if not target:
    return None
  candidates = []
  for bits in range(0x10000):
    if (bits & 0x7F80) == 0x7F80:
      continue
    value = _Fraction(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    if value:
      candidates.append((bits, value))
  for bits, value in candidates:
    quotient = target / value
    if quotient in _BF16_VALUES:
      return (bits, _BF16_WORD[quotient])
  return None


_BF16_FRACTIONS: dict = {}


def _build_bf16_table() -> None:
  for bits in range(0x10000):
    if (bits & 0x7F80) == 0x7F80:
      continue
    value = _Fraction(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    _BF16_FRACTIONS[value] = bits


_build_bf16_table()
_BF16_VALUES = frozenset(_BF16_FRACTIONS)
_BF16_WORD = _BF16_FRACTIONS


def _am_word_class(word: int) -> str | None:
  """None when no rule makes this AM word outright unreachable, else the class.

  Only one family is unreachable per se: numeric_freeze section 2 fixes every
  arithmetic NaN output at ``0x7fc00000`` without preserving payload or sign,
  so sNaN/payload-NaN golden words can never appear in the AM.  Everything
  else -- finite words (single product or exact bf16 sum, see
  ``_chain_operand_plan``) and ``-0`` (tiny-negative FMA underflow, frozen by
  the P03 ``mm-add-k0-negzero`` vector) -- is attempted constructively before
  any blocked record is written.
  """
  if word in _SPECIAL_BF16:
    return None
  if word & 0x7F800000 == 0x7F800000:
    return "snan-or-nan-payload"
  return None


def _word_value(word: int) -> _Fraction:
  return _Fraction(struct.unpack("<f", struct.pack("<I", word))[0])


def _word_special(word: int) -> bool:
  """qNaN / +-Inf / +0: producible with any finite nonzero A factor."""
  return word in _SPECIAL_BF16


def _common_factor_a(words: list[int]) -> int | None:
  """One bf16 A word with am_j / A exactly bf16 for every finite am word.

  A single k=1 row chain computes AM[0][j] = A * B[0][j], so the left factor
  is shared by all columns; both factors must be preserved or a subnormal
  golden word (e.g. 0x00000001 = 2^-149 = 0x0001 * 0x3780) silently rounds
  to the wrong AM.  Candidates are 1.0 plus every factor of the first
  finite word, in deterministic order.
  """
  candidates = [0x3F80]
  finite = [word for word in words if not _word_special(word)]
  if finite:
    factors = _bf16_factor(finite[0])
    if factors is not None:
      candidates.extend(factors)
  for candidate in candidates:
    a_value = _word_value(candidate << 16)  # candidates are 16-bit bf16 words
    if not a_value:
      continue
    if all(_word_value(word) / a_value in _BF16_VALUES for word in finite):
      return candidate
  return None


def _bf16_split(word: int, max_terms: int = 8) -> list[int] | None:
  """Exact bf16 summand decomposition of one finite non-special AM word.

  The AM of a legal chain is a sequential single-rounded f32 sum of bf16
  products, so a word decomposes chain-reachably when it splits into bf16
  summands whose running f32 partial sums are exact and finite.  Greedy
  largest-first truncation lands every committed finite golden word in at
  most three summands (e.g. FLT_MAX 0x7f7fffff = 0x7f7f + 0x7b7f + 0x777f);
  a residual that is no longer f32-exact or underflows below the bf16
  subnormal minimum returns None (the caller then falls back to / rejects
  the vector).
  """
  if (word & 0x7F800000) == 0x7F800000 or word == 0x80000000:
    return None
  target = _word_value(word)
  if not target:
    return [0x0000]
  summands: list[int] = []
  partial = _Fraction(0)
  for _ in range(max_terms):
    remaining = target - partial
    if not remaining:
      return summands
    remaining_word = struct.unpack("<I", struct.pack("<f", float(remaining)))[0]
    if _word_value(remaining_word) != remaining:
      return None  # residual not f32-exact: this constructor does not apply
    head = (remaining_word & 0xFFFF0000) >> 16
    if (head & 0x7F80) == 0x7F80:
      return None  # unreachable for a finite residual, guarded anyway
    if _word_value(head << 16) == 0:
      return None  # residual below the smallest bf16 subnormal summand
    new_partial = partial + _word_value(head << 16)
    partial_word = struct.unpack("<I", struct.pack("<f", float(new_partial)))[0]
    if partial_word in (0x7F800000, 0xFF800000):
      return None
    if _word_value(partial_word) != new_partial:
      return None
    summands.append(head)
    partial = new_partial
  return None


# fma(0x8001, 0x0001, +0): exact -2^-266 rounds (single RNE) to -0, and a
# following k=0 FINAL keeps the bits -- the construction the P03
# mm-add-k0-negzero vector freezes for a -0 golden AM word.
_NEGZERO_TOUCH_A = 0x8001
_NEGZERO_TOUCH_B = 0x0001


def _split_steps(words: list[int]) -> list[dict] | None:
  """Contribution steps accumulating every column of one row onto the AM.

  Without ``-0`` words this is one k=1 step per summand index (A=[1.0],
  B_j = the summand or +0 padding): every ADD block partial is exact, so
  AM + partial stays exact.

  A ``-0`` word needs different plumbing: ADD blocks compute their partial
  from p=+0 and then do AM += partial, and +0 + (-0) = +0 would erase it.
  So when any word is ``0x80000000`` the whole row goes into ONE multi-lane
  INIT block whose sequential lane accumulation ends in the underflow touch:
  lane s has A=1.0 with the summands, and the last lane has A=0x8001 with
  B_j=0x0001 for the -0 columns (fma(-2^-133, 2^-133, +0) single-rounds to
  -0, the arithmetic the P03 mm-add-k0-negzero vector freezes) and B_j=0x0000
  elsewhere (the -0 product passes any lane accumulator through unchanged).
  The INIT overwrite then commits the block result verbatim; the caller's
  k=0 FINAL keeps the bits.
  """
  columns: list[list[int]] = []
  for word in words:
    if word in _SPECIAL_BF16:
      columns.append([_SPECIAL_BF16[word]])
    elif word == 0x80000000:
      columns.append([0x0000])  # +0 padding; the touch lane makes it -0
    else:
      split = _bf16_split(word)
      if split is None:
        return None
      columns.append(split)
  depth = max(len(column) for column in columns)
  if not any(word == 0x80000000 for word in words):
    return [
      {
        "k": 1,
        "a": ["0x3f80"],
        "b": [
          f"0x{column[s]:04x}" if s < len(column) else "0x0000" for column in columns
        ],
      }
      for s in range(depth)
    ]
  # One [1 x (depth+1)] A row: 1.0 per summand lane, the touch A last; the B
  # matrix is lane-major [k x n] to match the builder's flat packing.
  a_words = ["0x3f80"] * depth + [f"0x{_NEGZERO_TOUCH_A:04x}"]
  b_words: list[str] = []
  for lane in range(depth + 1):
    for word, column in zip(words, columns):
      if lane < depth:
        b_words.append(f"0x{column[lane]:04x}" if lane < len(column) else "0x0000")
      else:
        b_words.append(f"0x{_NEGZERO_TOUCH_B:04x}" if word == 0x80000000 else "0x0000")
  return [{"k": depth + 1, "a": a_words, "b": b_words}]


def _chain_operand_plan(am: list[str], m: int) -> list[dict] | None:
  """Contribution steps (each one ``tile.boa.async`` INIT/ADD block) whose
  accumulation lands exactly on the golden AM; the caller appends the k=0
  FINAL step.

  m == 1 first tries one shared-factor product step (B[0][j] = am[0][j] / A,
  both factors preserved exactly -- the only shape that reaches sub-normal
  golden words) and otherwise the exact sum decomposition of
  ``_split_steps``.  m > 1 needs the identity construction (A = I_m,
  B = am), which is only safe when every word is finite and not ``-0``:
  off-diagonal 0 x Inf / 0 x NaN would poison the row sums and a 1.0
  multiplier cannot produce a signed zero.
  """
  words = [int(word, 16) for word in am]
  if m == 1:
    a_word = _common_factor_a(words)
    # A quotient-built B word (e.g. 0x8000 for a -0 golden word) cannot
    # produce -0 through the +0-seeded FMA, so -0 words need the split path
    # with its underflow touch step instead.
    if a_word is not None and 0x80000000 not in words:
      a_value = _word_value(a_word << 16)

      def b_word(word: int) -> int:
        if word in _SPECIAL_BF16:
          return _SPECIAL_BF16[word]
        return _BF16_WORD[_word_value(word) / a_value]

      return [
        {
          "k": 1,
          "a": [f"0x{a_word:04x}"],
          "b": [f"0x{b_word(word):04x}" for word in words],
        }
      ]
    return _split_steps(words)
  if all(
    (word & 0x7F800000) != 0x7F800000 and (word & 0xFFFF) == 0 and word != 0x80000000
    for word in words
  ):
    identity = ["0x0000"] * (m * m)
    for index in range(m):
      identity[index * m + index] = "0x3f80"
    return [
      {"k": m, "a": identity, "b": [f"0x{word >> 16:04x}" for word in words]}
    ]
  return None


# ---------------------------------------------------------------------------
# Minimal AM-chain module builder (faithful epilogue keyword emitter)
# ---------------------------------------------------------------------------


def _epilogue_keywords(epilogue: dict | None) -> str:
  """Emit every present epilogue field verbatim, independent of output dtype.

  Unlike the numeric-profile emitter this refuses to reinterpret illegal
  goldens: ``zero_point``/``inv_scale`` on a non-i8 output and negative word
  values are printed exactly as given so the source parser rejects them.
  """
  if not epilogue:
    return ""
  parts: list[str] = []
  if epilogue.get("bias") is not None:
    parts.append("bias = [" + ", ".join(str(word) for word in epilogue["bias"]) + "]")
  if epilogue.get("scale") is not None:
    parts.append(f"scale = {epilogue['scale']}")
  if epilogue.get("relu"):
    parts.append("relu")
  if epilogue.get("clip") is not None:
    parts.append("clip = [" + ", ".join(str(word) for word in epilogue["clip"]) + "]")
  if epilogue.get("inv_scale") is not None:
    parts.append(f"inv_scale = {epilogue['inv_scale']}")
  if epilogue.get("zero_point") is not None and (
    epilogue.get("output_dtype") == "i8" or epilogue["zero_point"] != 0
  ):
    parts.append(f"zero_point = {epilogue['zero_point']}")
  return (" " + " ".join(parts)) if parts else ""


def _am_chain_source(
  name: str, m: int, n: int, steps: list[dict], out_dtype: str, epilogue: dict | None
) -> str:
  """One typed AM chain: step s has A[m,ks]/B[ks,n] bf16 L2 formals and one
  ``tile.boa.async``; the AM allocation carries the epilogue keywords."""
  name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
  lines: list[str] = []
  params = ["%task: !nest.task"]
  for index, step in enumerate(steps):
    ks = step.get("storage_k", step["k"])
    params.append(f"%a{index}_l2: !nest.l2_buffer<{m}x{ks}xbf16>")
    params.append(f"%b{index}_l2: !nest.l2_buffer<{ks}x{n}xbf16>")
  params.append(f"%c_l2: !nest.l2_buffer<{m}x{n}x{out_dtype}>")
  lines.append("builtin.module {")
  lines.append(f"  tile.program @{name}({', '.join(params)})")
  lines.append(
    "      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],\n"
    f"    tile_l1_spm_bytes_per_context = {L1_CONTRACT_BYTES}> {{"
  )
  for index, step in enumerate(steps):
    ks = step.get("storage_k", step["k"])
    lines.append(
      f"    %a{index}_v = tile.subview %a{index}_l2 offsets = [0, 0] sizes = "
      f"[{m}, {ks}] strides = [1, 1]\n"
      f"      : !nest.l2_view<{m}x{ks}xbf16>"
    )
    lines.append(
      f"    %b{index}_v = tile.subview %b{index}_l2 offsets = [0, 0] sizes = "
      f"[{ks}, {n}] strides = [1, 1]\n"
      f"      : !nest.l2_view<{ks}x{n}xbf16>"
    )
  lines.append(
    f"    %c_v = tile.subview %c_l2 offsets = [0, 0] sizes = [{m}, {n}] strides = [1, 1]\n"
    f"      : !nest.l2_view<{m}x{n}x{out_dtype}>"
  )
  for index, step in enumerate(steps):
    ks = step.get("storage_k", step["k"])
    lines.append(
      f"    %a{index} = tile.alloc shape = [{m}, {ks}] dtype = \"bf16\" "
      f"alignment = 256 : !tile.l1_buffer<{m}x{ks}xbf16>"
    )
    lines.append(
      f"    %b{index} = tile.alloc shape = [{ks}, {n}] dtype = \"bf16\" "
      f"alignment = 256 : !tile.l1_buffer<{ks}x{n}xbf16>"
    )
  lines.append(
    f"    %out = tile.alloc memory_space = \"l1\" shape = [{m}, {n}] dtype = \"{out_dtype}\" "
    f"alignment = 256\n      : !tile.l1_buffer<{m}x{n}x{out_dtype}>"
  )
  lines.append(
    f"    %acc = tile.alloc memory_space = \"am\" logical_slot = 0 output(%out) "
    f"shape = [{m}, {n}]\n      dtype = \"f32\" alignment = 256"
    f"{_epilogue_keywords(epilogue)}\n      : !tile.am_buffer<{m}x{n}xf32>"
  )
  events: list[str] = []
  for index in range(len(steps)):
    lines.append(
      f"    %la{index} = tile.load.async %a{index}_v into %a{index} : !tile.event<\"la{index}\">"
    )
    lines.append(
      f"    %lb{index} = tile.load.async %b{index}_v into %b{index} : !tile.event<\"lb{index}\">"
    )
    events.extend((f"%la{index}", f"%lb{index}"))
  if events:
    lines.append(f"    tile.await {', '.join(events)}")
  lines.append("    tile.signal input_released(%task)")
  for index, step in enumerate(steps):
    modifiers = []
    if index > 0:
      modifiers.append("accumulate")
    if index == len(steps) - 1:
      modifiers.append("output")
    tail = (" " + " ".join(modifiers)) if modifiers else ""
    mask_text = ""
    if step.get("mask") is not None:
      mask_text = " k_mask = [" + ", ".join("1" if flag else "0" for flag in step["mask"]) + "]"
    lines.append(
      f"    %k{index} = tile.boa.async \"matmul\" ins(%a{index}, %b{index}) outs(%out) "
      f"acc(%acc)\n        m = {m} n = {n} k = {step['k']} seq = {index}{mask_text}{tail}"
      f" : !tile.event<\"k{index}\">"
    )
    lines.append(f"    tile.await %k{index}")
  for index in range(len(steps)):
    lines.append(f"    tile.free %a{index}")
    lines.append(f"    tile.free %b{index}")
  lines.append("    %stored = tile.store.async %out into %c_v : !tile.event<\"stored\">")
  lines.append("    tile.await %stored")
  lines.append("    tile.signal output_ready(%task)")
  lines.append("    tile.return")
  lines.append("  }")

  ctx_params = [
    part
    for index, step in enumerate(steps)
    for part in (
      f"%A{index}: !nest.global_memref<{m}x{step.get('storage_k', step['k'])}xbf16>",
      f"%B{index}: !nest.global_memref<{step.get('storage_k', step['k'])}x{n}xbf16>",
    )
  ]
  ctx_params.append(f"%C: !nest.global_memref<{m}x{n}x{out_dtype}>")
  l2_bytes = max(
    sum(
      2 * (m * step.get("storage_k", step["k"]) + step.get("storage_k", step["k"]) * n)
      for step in steps
    )
    + m * n * DTYPE_WIDTH[out_dtype],
    ALIGN,
  )
  lines.append(f"  nest.context @{name}_ctx({', '.join(ctx_params)})")
  lines.append(
    "    placement = 1\n"
    "        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],\n"
    f"      logical_tasks = 1, l2_spm_bytes = {l2_bytes}, requested_contexts_per_tile = 1> {{"
  )
  pref_events: list[str] = []
  for index, step in enumerate(steps):
    ks = step.get("storage_k", step["k"])
    for tag, dims in (("a", f"{m}x{ks}"), ("b", f"{ks}x{n}")):
      formal = f"%A{index}" if tag == "a" else f"%B{index}"
      lines.append(
        f"    %{tag}{index}_blk = nest.subview {formal} offsets = [0, 0] sizes = "
        f"[{dims.replace('x', ', ')}] strides = [1, 1]\n"
        f"      : !nest.global_view<{dims}xbf16>"
      )
      lines.append(
        f"    %{tag}{index}_l2_1 = nest.alloc slot = \"{tag}{index}\" role = \"in\" "
        f"shape = [{dims.replace('x', ', ')}] dtype = \"bf16\" alignment = 256\n"
        f"        : !nest.l2_buffer<{dims}xbf16>"
      )
      lines.append(
        f"    %{tag}{index}_pref = nest.dma.prefetch.async %{tag}{index}_blk into %{tag}{index}_l2_1"
        f" : !nest.event<\"{tag}{index}_pref\">"
      )
      pref_events.append(f"%{tag}{index}_pref")
  lines.append(
    f"    %c_blk = nest.subview %C offsets = [0, 0] sizes = [{m}, {n}] strides = [1, 1]\n"
    f"      : !nest.global_view<{m}x{n}x{out_dtype}>"
  )
  lines.append(
    f"    %c_l2_1 = nest.alloc slot = \"c\" role = \"out\" shape = [{m}, {n}] "
    f"dtype = \"{out_dtype}\" alignment = 256 : !nest.l2_buffer<{m}x{n}x{out_dtype}>"
  )
  lines.append("    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range")
  names_a = ", ".join(f"%a{index}_l2_1" for index in range(len(steps)))
  names_b = ", ".join(f"%b{index}_l2_1" for index in range(len(steps)))
  bindings_list = ", ".join(
    name for index in range(len(steps)) for name in (f"%a{index}_l2_1", f"%b{index}_l2_1")
  )
  lines.append(
    f"    %grid, %read, %ready = nest.dispatch.tasks.async @{name} l1_mode = 0 context = 0\n"
    f"      tasks(%tasks) globals() bindings({bindings_list}, %c_l2_1)\n"
    f"      ins({names_a}, {names_b}) outs(%c_l2_1)\n"
    "      signal_policy {\n"
    "        input_released = #nest.aggregate<all_tasks>,\n"
    "        output_ready = #nest.aggregate<all_tasks>\n"
    f"      }} depends_on({', '.join(pref_events)})\n"
    "      : (!nest.event<\"grid\">, !nest.event<\"read\">, !nest.event<\"ready\">)"
  )
  lines.append(
    "    %c_store = nest.dma.store.async %c_l2_1 into %c_blk depends_on(%ready)"
    " : !nest.event<\"c_store\">"
  )
  for index in range(len(steps)):
    for tag in ("a", "b"):
      lines.append(
        f"    nest.release %{tag}{index}_l2_1 depends_on(%read, %{tag}{index}_pref)"
      )
  lines.append("    nest.release %c_l2_1 depends_on(%c_store)")
  lines.append("    nest.await %grid, %c_store")
  lines.append("    nest.return")
  lines.append("  }")
  entry_params = list(ctx_params)
  submit_args = ", ".join(
    name for index in range(len(steps)) for name in (f"%A{index}", f"%B{index}")
  )
  submit_args = f"{submit_args}, %C" if submit_args else "%C"
  lines.append(f"  nexus.program @{name}_entry({', '.join(entry_params)}) {{")
  lines.append(
    f"    %done = nexus.submit_context.async @{name}_ctx({submit_args})\n"
    "        : !nexus.event<\"done\">"
  )
  lines.append("    nexus.await %done")
  lines.append("    nexus.return")
  lines.append("  }")
  lines.append("}")
  return "\n".join(lines) + "\n"


def _chain_layout(m: int, n: int, steps: list[dict], out_dtype: str):
  """Bindings, HBM seeds and the C location for one chain module."""
  bindings: dict[str, GlobalBinding] = {}
  seeds: list[tuple[int, bytes]] = []
  cursor = L2_BASE

  def reserve(name: str, size: int, data: bytes) -> None:
    nonlocal cursor
    cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
    bindings[name] = GlobalBinding(name, cursor, size, "r" if name != "C" else "w")
    if data:
      seeds.append((cursor, data))
    cursor += size

  for index, step in enumerate(steps):
    words = [int(word, 16) for word in step["a"]]
    reserve(f"A{index}", len(words) * 2, struct.pack(f"<{len(words)}H", *words))
    words = [int(word, 16) for word in step["b"]]
    reserve(f"B{index}", len(words) * 2, struct.pack(f"<{len(words)}H", *words))
  c_bytes = m * n * DTYPE_WIDTH[out_dtype]
  reserve("C", c_bytes, b"")
  return bindings, seeds, bindings["C"].base_iova, c_bytes


def _run_chain(
  name: str, m: int, n: int, steps: list[dict], out_dtype: str, epilogue: dict | None, seed: int,
  *, tile_k: int = 8,
):
  """Real production path for one chain: parse -> compile -> serialize ->
  parse -> load -> Simulator(full_memory, ByteStore)."""
  source = _am_chain_source(name, m, n, steps, out_dtype, epilogue)
  module = parse_workload_ir(print_workload_ir(parse_workload_ir(source)))
  hw = HardwareConfig().with_overrides(boa_numeric_tile_k=tile_k)
  sim = SimConfig(fidelity="full_memory", context_count=1, seed=seed)
  bindings, seeds, c_address, c_bytes = _chain_layout(m, n, steps, out_dtype)
  artifact = compile_program(
    module, hw, sim, binding_assumptions=bindings, source_name="soft-conv"
  )
  replay = parse_compiled_program(serialize_compiled_program(artifact))
  loaded = load_program(replay, hw, sim, actual_bindings=bindings)
  store = ByteStore()
  for address, data in seeds:
    store.seed_hbm(address, data)
  result = Simulator(hw, sim, byte_store=store).run(loaded)
  actual = store.read_hbm(c_address, c_bytes) if result.completed and c_bytes else b""
  return result, actual, source


def _vector_record(
  record_id: str,
  label: str,
  passed: bool,
  detail: str,
  *,
  golden_hex: str = "",
  actual_hex: str = "",
  cycles: int | None = None,
  stage: str = "",
  error: str = "",
  blocked_class: str = "",
) -> dict:
  record = {
    "id": record_id,
    "label": label,
    "pass": bool(passed),
    "detail": detail,
    "golden_hex": golden_hex,
    "actual_hex": actual_hex,
  }
  if cycles is not None:
    record["cycles"] = cycles
  if stage:
    record["stage"] = stage
  if error:
    record["error"] = error
  if blocked_class:
    record["blocked_class"] = blocked_class
  return record


def _finish_report(case: str, config: dict, checks: list[dict], vectors: list[dict]) -> dict:
  counts = {
    "vectors": len(vectors),
    "passed": sum(1 for item in vectors if item["pass"]),
    "failed": sum(1 for item in vectors if not item["pass"]),
    "production_path": sum(1 for item in vectors if item["label"] == LABEL_PRODUCTION),
    "epilogue_function_parity": sum(1 for item in vectors if item["label"] == LABEL_FUNCTION),
    "blocked": sum(1 for item in vectors if item["label"] == LABEL_BLOCKED),
    "negative": sum(1 for item in vectors if item["label"] == LABEL_NEGATIVE),
    "cycles_total": sum(item.get("cycles", 0) for item in vectors),
  }
  blocked_by_class: dict[str, int] = {}
  for item in vectors:
    if item["label"] == LABEL_BLOCKED:
      blocked_by_class[item["blocked_class"]] = blocked_by_class.get(item["blocked_class"], 0) + 1
  counts["blocked_by_class"] = blocked_by_class
  status = "PASS" if checks and all(item["pass"] for item in checks) and not counts["failed"] else "FAIL"
  return {
    "schema": 1,
    "case": case,
    "profile": config["profile"],
    "counts": counts,
    "checks": checks,
    "status": status,
  }


def _inputs_echo(config: dict) -> dict:
  echo = {"profile": config["profile"]}
  for key in ("epilogue", "matmul"):
    path = Path(config[key])
    echo[key] = {"path": str(path), "sha256": _sha256(path)}
  return echo


def _sha256(path: Path) -> str:
  import hashlib

  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Case engines
# ---------------------------------------------------------------------------


def _engine_int8_chain(config: dict, seed: int):
  checks: list[dict] = []
  vectors: list[dict] = []
  sources: list[tuple[str, str]] = []

  def record_vector(item: dict) -> None:
    vectors.append(item)
    checks.append({"id": f"vector:{item['id']}", "pass": item["pass"], "detail": item["detail"]})

  matmul_doc = _load_vector_document(config["matmul"])
  for case in matmul_doc["cases"]:
    inp = case.get("input") or {}
    epilogue = inp.get("epilogue") or {}
    if epilogue.get("output_dtype") != "i8":
      continue
    steps = [
      {
        "k": step["k"],
        "storage_k": step.get("storage_k") or step["k"],
        "mask": step.get("mask"),
        "a": list(step["a"]),
        "b": list(step["b"]),
      }
      for step in inp["steps"]
    ]
    golden = case["expected"]["output_hex"]
    try:
      result, actual, source = _run_chain(
        case["id"], inp["m"], inp["n"], steps, "i8", epilogue, seed, tile_k=inp["tile_k"]
      )
    except (ParseError, ValueError, VerifyException) as exc:
      record_vector(
        _vector_record(
          case["id"], LABEL_PRODUCTION, False,
          f"legal matmul.json i8 chain failed construction: {exc}",
          golden_hex=golden, stage="compile", error=str(exc),
        )
      )
      continue
    sources.append((f"{case['id'].replace('-', '_')}.mlir", source))
    record_vector(
      _vector_record(
        case["id"], LABEL_PRODUCTION,
        result.completed and actual == bytes.fromhex(golden),
        "completed Simulator run must equal the P03 golden FINAL bytes",
        golden_hex=golden, actual_hex=actual.hex(), cycles=result.cycles,
      )
    )

  conv_case = CASES["conv-i8-out"]
  result, store, bindings, debug_size, _loaded = _run_conv_case(conv_case, debug_a=True)
  analytic = _analytic(conv_case)
  c_actual = store.read_hbm(bindings["C"].base_iova, analytic["output_bytes"])
  ad_bytes = store.read_hbm(bindings["AD"].base_iova, debug_size)
  ad_words = [int.from_bytes(ad_bytes[i * 2 : i * 2 + 2], "little") for i in range(debug_size // 2)]
  conv_ok = (
    result.completed
    and c_actual == bytes.fromhex(conv_case.expected["output_hex"])
    and _reconstruct_a(conv_case, ad_words) == bytes.fromhex(conv_case.expected["a_hex"])
  )
  record_vector(
    _vector_record(
      "conv-i8-out", LABEL_PRODUCTION, conv_ok,
      "gather-im2col conv path with the AM FINAL epilogue must equal C and A goldens",
      golden_hex=conv_case.expected["output_hex"], actual_hex=c_actual.hex(),
      cycles=result.cycles,
    )
  )

  epilogue_doc = _load_vector_document(config["epilogue"])
  i8_total = 0
  for case in epilogue_doc["cases"]:
    inp = case.get("input") or {}
    expected = case.get("expected") or {}
    if "error" in expected or (inp.get("epilogue") or {}).get("output_dtype") != "i8":
      continue
    i8_total += 1
    m, n = inp["m"], inp["n"]
    am = list(inp["am"])
    classes = [_am_word_class(int(word, 16)) for word in am]
    blocking = next(((word, cls) for word, cls in zip(am, classes) if cls), None)
    plan = None if blocking else _chain_operand_plan(am, m)
    golden = expected["output_hex"]
    if blocking is not None:
      word, cls = blocking
      record_vector(
        _vector_record(
          case["id"], LABEL_BLOCKED, True,
          f"golden AM word {word} ({cls}) cannot be produced by any legal "
          f"bf16 chain (numeric_freeze s2 canonicalizes every arithmetic NaN "
          f"output to 0x7fc00000, discarding payload and sign); recorded, "
          f"not silently skipped",
          golden_hex=golden, blocked_class=cls,
        )
      )
      continue
    if plan is None:
      record_vector(
        _vector_record(
          case["id"], LABEL_PRODUCTION, False,
          "no exact BF16 chain was constructed for this vector; this is a "
          "coverage failure, not proof of unreachable AM",
          golden_hex=golden, stage="construct",
        )
      )
      continue
    steps = [*plan, {"k": 0, "a": [], "b": []}]
    # Run 1 -- raw f32 FINAL with no epilogue writes the pre-epilogue AM
    # bits straight to C, so the chain must land exactly on the committed
    # golden AM before any quantization can hide a wrong operand factor.
    am_bytes = b"".join(int(word, 16).to_bytes(4, "little") for word in am)
    try:
      am_result, am_actual, am_source = _run_chain(
        f"{case['id']}-am", m, n, steps, "f32", None, seed
      )
    except (ParseError, ValueError, VerifyException) as exc:
      record_vector(
        _vector_record(
          f"{case['id']}-am", LABEL_PRODUCTION, False,
          f"synthesized AM-parity chain failed construction: {exc}",
          golden_hex=am_bytes.hex(), stage="compile", error=str(exc),
        )
      )
      am_actual = None
    else:
      sources.append((f"{case['id'].replace('-', '_')}_am.mlir", am_source))
      record_vector(
          _vector_record(
            f"{case['id']}-am", LABEL_PRODUCTION,
            am_result.completed and am_actual == am_bytes,
            "raw f32 FINAL (no epilogue) must write exactly the committed "
            "golden AM bits -- every product factor / summand preserved",
            golden_hex=am_bytes.hex(), actual_hex=am_actual.hex(),
            cycles=am_result.cycles,
          )
        )
    # Run 2 -- the vector's own output dtype and epilogue apply to that AM.
    try:
      result, actual, source = _run_chain(case["id"], m, n, steps, "i8", inp["epilogue"], seed)
    except (ParseError, ValueError, VerifyException) as exc:
      record_vector(
        _vector_record(
          case["id"], LABEL_PRODUCTION, False,
          f"synthesized chain failed construction: {exc}",
          golden_hex=golden, stage="compile", error=str(exc),
        )
      )
      continue
    sources.append((f"{case['id'].replace('-', '_')}.mlir", source))
    record_vector(
      _vector_record(
        case["id"], LABEL_PRODUCTION,
        result.completed and actual == bytes.fromhex(golden),
        "contribution chain + k=0 FINAL applies the golden epilogue to "
        "exactly the golden AM (verified raw above); output must equal golden",
        golden_hex=golden, actual_hex=actual.hex(), cycles=result.cycles,
      )
    )

  covered = {item["id"].removesuffix("-am") for item in vectors if item["id"].startswith("epi-")}
  checks.append(
    {
      "id": "i8-vector-coverage",
      "pass": len(covered) == i8_total,
      "detail": f"{len(covered)} of {i8_total} epilogue.json i8 vectors recorded (run or blocked)",
    }
  )
  report = _finish_report("int8-chain", config, checks, vectors)
  report["inputs"] = _inputs_echo(config)
  return report, vectors, sources


_REJECT_PLAN: dict[str, dict] = {
  "epi-illegal-bias-length": {"kind": "parse", "pattern": "bias has 2 words"},
  "epi-illegal-bias-word-range": {
    "kind": "parse",
    "pattern": "bias\\[2\\] must be a 32-bit FP32 bit pattern",
  },
  "epi-illegal-bias-word-negative": {"kind": "parse", "pattern": "expected in bias"},
  "epi-illegal-scale-word-range": {"kind": "parse", "pattern": "scale"},
  "epi-illegal-clip-nan-bound": {"kind": "parse", "pattern": "clip bounds must not be NaN"},
  "epi-illegal-clip-reversed": {"kind": "parse", "pattern": "unordered"},
  "epi-illegal-clip-triple": {"kind": "parse", "pattern": "clip must be"},
  "epi-illegal-i8-missing-inv-scale": {"kind": "parse", "pattern": "inv_scale is required for i8"},
  "epi-illegal-inv-scale-zero": {"kind": "parse", "pattern": "inv_scale"},
  "epi-illegal-inv-scale-negzero": {"kind": "parse", "pattern": "inv_scale"},
  "epi-illegal-inv-scale-nan": {"kind": "parse", "pattern": "inv_scale"},
  "epi-illegal-inv-scale-inf": {"kind": "parse", "pattern": "inv_scale"},
  "epi-illegal-inv-scale-negative": {"kind": "parse", "pattern": "inv_scale"},
  "epi-illegal-zero-point-high": {"kind": "parse", "pattern": "zero_point 200 is outside"},
  "epi-illegal-zero-point-low": {"kind": "parse", "pattern": "zero_point -129 is outside"},
  "epi-illegal-zero-point-bool": {"kind": "parse", "pattern": "zero_point"},
  "epi-illegal-quant-on-f32-inv": {"kind": "parse", "pattern": "inv_scale is only valid for i8"},
  "epi-illegal-quant-on-f32-zp": {"kind": "parse", "pattern": "zero_point is only valid for i8"},
  "epi-illegal-unknown-dtype": {"kind": "codec-dtype", "pattern": "dtype"},
  "epi-illegal-relu-not-bool": {"kind": "codec-relu", "pattern": "relu must be"},
}

# The dtype and relu tampers already back the mapped vectors above; these
# two add coverage for fields no source text can encode illegally.
_CODEC_TAMPERS: tuple[tuple[str, str, object, str], ...] = (
  ("codec-bias-length", "bias", [0, 0, 0], "bias has"),
  ("codec-zero-point-bool", "zero_point", True, "zero_point"),
)


def _valid_i8_artifact(seed: int) -> str:
  doc = _load_vector_document(DEFAULT_MATMUL_VECTORS)
  case = next(item for item in doc["cases"] if item["id"] == "mm-stages-i8")
  inp = case["input"]
  steps = [{"k": step["k"], "a": list(step["a"]), "b": list(step["b"])} for step in inp["steps"]]
  module = parse_workload_ir(
    print_workload_ir(
      parse_workload_ir(
        _am_chain_source(case["id"], inp["m"], inp["n"], steps, "i8", inp["epilogue"])
      )
    )
  )
  hw = HardwareConfig()
  sim = SimConfig(fidelity="full_memory", context_count=1, seed=seed)
  bindings, _seeds, _addr, _size = _chain_layout(inp["m"], inp["n"], steps, "i8")
  artifact = compile_program(module, hw, sim, binding_assumptions=bindings, source_name="soft-conv")
  return serialize_compiled_program(artifact)


def _engine_param_reject(config: dict, seed: int):
  checks: list[dict] = []
  vectors: list[dict] = []
  sources: list[tuple[str, str]] = []
  epilogue_doc = _load_vector_document(config["epilogue"])
  by_id = {case["id"]: case for case in epilogue_doc["cases"]}

  def reject_record(record_id: str, stage: str, error: Exception | None, pattern: str) -> None:
    passed = error is not None and re.search(pattern, str(error), re.I) is not None
    vectors.append(
      _vector_record(
        record_id, LABEL_NEGATIVE, passed,
        f"illegal parameters must be rejected at {stage}",
        stage=stage, error="" if error is None else str(error),
      )
    )
    checks.append(
      {"id": f"vector:{record_id}", "pass": passed, "detail": str(error)[:160] if error else "accepted"}
    )

  for record_id, plan in _REJECT_PLAN.items():
    case = by_id[record_id]
    inp = case["input"]
    epilogue = dict(inp["epilogue"])
    steps = [
      {
        "k": 1,
        "a": ["0x3f80"] * max(inp["m"], 1),
        "b": ["0x3f80"] * max(inp["n"], 1),
      }
    ]
    if plan["kind"] == "parse":
      source = _am_chain_source(
        record_id, max(inp["m"], 1), max(inp["n"], 1), steps, "f32", epilogue
      )
      if epilogue.get("output_dtype") == "i8":
        source = _am_chain_source(
          record_id, max(inp["m"], 1), max(inp["n"], 1), steps, "i8", epilogue
        )
      sources.append((f"{record_id.replace('-', '_')}.mlir", source))
      try:
        parse_workload_ir(source)
        reject_record(record_id, "parse", None, plan["pattern"])
      except (ParseError, ValueError, VerifyException) as exc:
        reject_record(record_id, "parse", exc, plan["pattern"])
      continue
    artifact_text = _valid_i8_artifact(seed)
    document = json.loads(artifact_text)
    program = document["entry"]["tasks"]["$map"][0][1]["role_bindings"]["$map"][0][1]["tile_program"]
    am_buffer = program["am_buffers"][0]
    if plan["kind"] == "codec-dtype":
      am_buffer["output"]["dtype"] = "f16"
    elif plan["kind"] == "codec-relu":
      am_buffer["epilogue"]["relu"] = 2
    tampered = json.dumps(document, sort_keys=True)
    try:
      parse_compiled_program(tampered)
      reject_record(record_id, "artifact-decode", None, plan["pattern"])
    except ValueError as exc:
      reject_record(record_id, "artifact-decode", exc, plan["pattern"])

  artifact_text = _valid_i8_artifact(seed)
  for record_id, field, value, pattern in _CODEC_TAMPERS:
    document = json.loads(artifact_text)
    program = document["entry"]["tasks"]["$map"][0][1]["role_bindings"]["$map"][0][1]["tile_program"]
    am_buffer = program["am_buffers"][0]
    if field is None:
      am_buffer["output"]["dtype"] = value
    else:
      am_buffer["epilogue"][field] = value
    tampered = json.dumps(document, sort_keys=True)
    try:
      parse_compiled_program(tampered)
      reject_record(record_id, "artifact-decode", None, pattern)
    except ValueError as exc:
      reject_record(record_id, "artifact-decode", exc, pattern)

  error_vectors = sum(1 for case in epilogue_doc["cases"] if "error" in (case.get("expected") or {}))
  checks.append(
    {
      "id": "illegal-vector-coverage",
      "pass": error_vectors == len(_REJECT_PLAN),
      "detail": f"{len(_REJECT_PLAN)} of {error_vectors} epilogue.json error vectors mapped to a boundary",
    }
  )
  report = _finish_report("param-reject", config, checks, vectors)
  report["inputs"] = _inputs_echo(config)
  return report, vectors, sources


def _engine_stage_matrix(config: dict, seed: int):
  checks: list[dict] = []
  vectors: list[dict] = []
  sources: list[tuple[str, str]] = []
  epilogue_doc = _load_vector_document(config["epilogue"])

  def record_vector(item: dict) -> None:
    vectors.append(item)
    checks.append({"id": f"vector:{item['id']}", "pass": item["pass"], "detail": item["detail"]})

  for case in epilogue_doc["cases"]:
    inp = case.get("input") or {}
    expected = case.get("expected") or {}
    if "error" in expected:
      continue
    epilogue = inp["epilogue"]
    spec = BoaEpilogueSpec(
      epilogue.get("output_dtype") or "f32",
      bias=None if epilogue.get("bias") is None else tuple(int(word, 16) for word in epilogue["bias"]),
      scale=None if epilogue.get("scale") is None else int(epilogue["scale"], 16),
      relu=bool(epilogue.get("relu")),
      clip=(
        None
        if epilogue.get("clip") is None
        else (int(epilogue["clip"][0], 16), int(epilogue["clip"][1], 16))
      ),
      inv_scale=None if epilogue.get("inv_scale") is None else int(epilogue["inv_scale"], 16),
      zero_point=int(epilogue.get("zero_point") or 0),
    )
    golden = expected["output_hex"]
    actual = apply_epilogue([int(word, 16) for word in inp["am"]], inp["m"], inp["n"], spec)
    record_vector(
      _vector_record(
        case["id"], LABEL_FUNCTION, actual == bytes.fromhex(golden),
        "production boa_numeric.apply_epilogue (the runtime FINAL path) on the "
        "committed AM must equal the golden output bytes",
        golden_hex=golden, actual_hex=actual.hex(),
      )
    )

  matmul_doc = _load_vector_document(config["matmul"])
  for vector_id in ("mm-stages-all", "mm-stages-bf16", "mm-stages-i8"):
    case = next(item for item in matmul_doc["cases"] if item["id"] == vector_id)
    inp = case["input"]
    epilogue = inp.get("epilogue") or {}
    out_dtype = epilogue.get("output_dtype") or "f32"
    steps = [
      {
        "k": step["k"],
        "storage_k": step.get("storage_k") or step["k"],
        "mask": step.get("mask"),
        "a": list(step["a"]),
        "b": list(step["b"]),
      }
      for step in inp["steps"]
    ]
    golden = case["expected"]["output_hex"]
    result, actual, source = _run_chain(
      vector_id, inp["m"], inp["n"], steps, out_dtype, epilogue, seed, tile_k=inp["tile_k"]
    )
    sources.append((f"{vector_id.replace('-', '_')}.mlir", source))
    record_vector(
      _vector_record(
        vector_id, LABEL_PRODUCTION, result.completed and actual == bytes.fromhex(golden),
        "Simulator wiring proof: source epilogue keywords apply once at FINAL",
        golden_hex=golden, actual_hex=actual.hex(), cycles=result.cycles,
      )
    )

  combos = sum(1 for item in vectors if item["id"].startswith("epi-combo-"))
  checks.append(
    {
      "id": "combo-coverage",
      "pass": combos == 48,
      "detail": f"{combos} of 48 epi-combo vectors (16 stage combinations x 3 dtypes) compared",
    }
  )
  report = _finish_report("stage-matrix", config, checks, vectors)
  report["inputs"] = _inputs_echo(config)
  return report, vectors, sources


def _engine_conv_pad_stride_dilation(config: dict, seed: int):
  checks: list[dict] = []
  vectors: list[dict] = []
  sources: list[tuple[str, str]] = []
  for case in CASES.values():
    result, store, bindings, debug_size, loaded = _run_conv_case(case, debug_a=True)
    analytic = _analytic(case)
    c_actual = store.read_hbm(bindings["C"].base_iova, analytic["output_bytes"])
    ad_bytes = store.read_hbm(bindings["AD"].base_iova, debug_size)
    ad_words = [
      int.from_bytes(ad_bytes[i * 2 : i * 2 + 2], "little") for i in range(debug_size // 2)
    ]
    passed = (
      result.completed
      and c_actual == bytes.fromhex(case.expected["output_hex"])
      and _reconstruct_a(case, ad_words) == bytes.fromhex(case.expected["a_hex"])
    )
    item = _vector_record(
      case.source_id, LABEL_PRODUCTION, passed,
      "C bytes and the captured im2col A operand bytes must equal the P03 goldens",
      golden_hex=case.expected["output_hex"], actual_hex=c_actual.hex(),
      cycles=result.cycles,
    )
    item["conv"] = {
      "shape": list(case.shape),
      "kernel": list(case.kernel),
      "stride": list(case.stride),
      "padding": list(case.padding),
      "dilation": list(case.dilation),
      "steps": list(case.steps),
      "V": analytic["V"],
      "E": analytic["E"],
      "P": analytic["P"],
    }
    vectors.append(item)
    checks.append(
      {"id": f"vector:{case.source_id}", "pass": passed, "detail": item["detail"]}
    )
    _assert_no_evu(loaded)
  report = _finish_report("conv-pad-stride-dilation", config, checks, vectors)
  report["inputs"] = _inputs_echo(config)
  return report, vectors, sources


def _engine_conv_traffic(config: dict, seed: int):
  checks: list[dict] = []
  vectors: list[dict] = []
  sources: list[tuple[str, str]] = []
  for case in CASES.values():
    analytic = _analytic(case)
    _assert_golden_traffic(case, analytic)
    fixture = WORKLOADS / f"soft_conv_{case.case_id}.mlir"
    result, store, bindings, _, _loaded = _run_conv_case(
      case,
      debug_a=False,
      source_text=fixture.read_text(encoding="utf-8"),
      source_name=fixture.name,
    )
    issued_by_op, gather_events, byte_counters = _expected_counters(case, debug_a=False)
    transfers = result.group_snapshot["memory"]["transfers"]
    ok = (
      result.completed
      and transfers["issued_by_op"] == issued_by_op
      and transfers["byte_counters"] == byte_counters
      and all(result.pmu.events.get(key, 0) == value for key, value in gather_events.items())
      and store.read_hbm(bindings["C"].base_iova, analytic["output_bytes"])
      == bytes.fromhex(case.expected["output_hex"])
    )
    detail = (
      f"E={analytic['E']} V={analytic['V']} P={analytic['P']}; ops={transfers['issued_by_op']}"
    )
    vectors.append(
      _vector_record(
        case.source_id, LABEL_PRODUCTION, ok,
        "per-stage transfer/gather bytes equal the independent analytic counts; "
        + detail,
        cycles=result.cycles,
      )
    )
    checks.append({"id": f"vector:{case.source_id}", "pass": ok, "detail": detail})
  report = _finish_report("conv-traffic", config, checks, vectors)
  report["inputs"] = _inputs_echo(config)
  return report, vectors, sources


ENGINES = {
  "int8-chain": _engine_int8_chain,
  "param-reject": _engine_param_reject,
  "stage-matrix": _engine_stage_matrix,
  "conv-pad-stride-dilation": _engine_conv_pad_stride_dilation,
  "conv-traffic": _engine_conv_traffic,
}


def _write_outputs(
  out: Path, case: str, report: dict, vectors: list[dict], sources: list[tuple[str, str]]
) -> None:
  out.mkdir(parents=True, exist_ok=True)
  (out / f"{case}.report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  (out / f"{case}.vectors.json").write_text(
    json.dumps(
      {"schema": 1, "case": case, "profile": report["profile"], "vectors": vectors},
      indent=2,
      sort_keys=True,
    )
      + "\n",
    encoding="utf-8",
  )
  sources_dir = out / "sources"
  sources_dir.mkdir(parents=True, exist_ok=True)
  for name, text in sources:
    (sources_dir / name).write_text(text, encoding="utf-8")


def _run_cli_case(case: str, out: Path, *, seed: int, config: dict | None = None) -> dict:
  if case not in ENGINES:
    raise ValueError(f"unknown soft-conv case {case!r} (registered: {', '.join(CLI_CASES)})")
  resolved = config if config is not None else load_config(None)
  report, vectors, sources = ENGINES[case](resolved, seed)
  _write_outputs(Path(out), case, report, vectors, sources)
  return report


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    prog="python -m pipeline_validator.tests.test_soft_conv",
    description="P08 soft-conv production-path gate (im2col + typed BOA Matmul + FINAL epilogue)",
  )
  parser.add_argument("--case", required=True, choices=CLI_CASES)
  parser.add_argument("--out", required=True, help="output directory for report/vectors/sources")
  parser.add_argument("--seed", type=int, default=1, help="recorded run seed and simulator seed")
  parser.add_argument("--config", default=None, help="optional strict JSON config (see module docstring)")
  args = parser.parse_args(argv)
  try:
    config = load_config(args.config)
  except ValueError as exc:
    print(f"config-error: {exc}", file=sys.stderr)
    return 2
  started = time.monotonic()
  report = _run_cli_case(args.case, Path(args.out), seed=args.seed, config=config)
  elapsed = time.monotonic() - started
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / f"{args.case}.timing.json").write_text(
    json.dumps(
      {
        "schema": 1,
        "case": args.case,
        "seed": args.seed,
        "wall_clock_seconds": elapsed,
        "deterministic": False,
        "note": "elapsed time is environment-dependent; excluded from deterministic reproduction",
      },
      indent=2,
      sort_keys=True,
    )
      + "\n",
    encoding="utf-8",
  )
  failed_checks = [item["id"] for item in report["checks"] if not item["pass"]]
  print(
    json.dumps(
      {
        "case": args.case,
        "status": report["status"],
        "counts": report["counts"],
        "failed_checks": failed_checks,
      },
      sort_keys=True,
    )
  )
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))


# ---------------------------------------------------------------------------
# pytest entry points for the five CLI cases
# ---------------------------------------------------------------------------


def test_int8_chain_matches_committed_goldens(tmp_path):
  report = _run_cli_case("int8-chain", Path(tmp_path), seed=20261028)
  assert report["status"] == "PASS", [c for c in report["checks"] if not c["pass"]]
  counts = report["counts"]
  # 2 matmul.json chains + conv-i8-out + 9 synthesized chains x 2 runs
  # (raw f32 AM parity + the vector's own epilogue output)
  assert counts["production_path"] == 21
  assert counts["blocked"] == 17
  assert set(counts["blocked_by_class"]) == {"snan-or-nan-payload"}
  assert counts["blocked_by_class"]["snan-or-nan-payload"] == 17

  vectors = json.loads((Path(tmp_path) / "int8-chain.vectors.json").read_text())["vectors"]
  by_id = {item["id"]: item for item in vectors}
  committed = {
    case["id"]: case
    for case in _load_vector_document(DEFAULT_EPILOGUE_VECTORS)["cases"]
  }
  # The four vectors once mislabelled unreachable (FLT_MAX golden words are
  # reachable through the exact bf16 sum split, -0 through the tiny-negative
  # underflow touch) now run the production path with byte-exact raw AM and
  # FINAL output -- not merely different aggregate counts.
  for record_id in (
    "epi-i8-saturation",
    "epi-i8-overflow-product",
    "epi-rand-00",
    "epi-rand-05",
  ):
    raw = by_id[f"{record_id}-am"]
    assert raw["label"] == LABEL_PRODUCTION and raw["pass"], raw
    golden_am = b"".join(
      int(word, 16).to_bytes(4, "little") for word in committed[record_id]["input"]["am"]
    ).hex()
    assert raw["golden_hex"] == golden_am, record_id
    assert raw["actual_hex"] == golden_am, record_id
    produced = by_id[record_id]
    assert produced["label"] == LABEL_PRODUCTION and produced["pass"], produced
    assert produced["actual_hex"] == committed[record_id]["expected"]["output_hex"], record_id
  # Every remaining block is a NaN-payload golden word no chain can produce.
  blocked = [item for item in vectors if item["label"] == LABEL_BLOCKED]
  assert {item["id"] for item in blocked} == {
    *(f"epi-combo-{index:02d}-i8" for index in range(16)),
    "epi-i8-nan-zp",
  }
  assert all(
    item["pass"] and item["blocked_class"] == "snan-or-nan-payload" for item in blocked
  )


def test_int8_cli_fails_for_unconstructed_finite_am(tmp_path):
  """A limited constructor must not turn an untested finite vector into PASS."""
  vectors_path = tmp_path / "finite.json"
  vectors_path.write_text(json.dumps({
    "profile": "bf16_fp32_mac_rne_v1",
    "cases": [{
      "id": "epi-custom-finite",
      "input": {
        "m": 2, "n": 1, "am": ["0x3f800001", "0x3f800001"],
        "epilogue": {"output_dtype": "i8", "inv_scale": "0x3f800000", "zero_point": 0},
      },
      "expected": {"output_hex": "0101"},
    }],
  }))
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps({"schema": 1, "epilogue": str(vectors_path)}))
  out = tmp_path / "out"
  assert main([
    "--case", "int8-chain", "--out", str(out), "--config", str(config_path), "--seed", "20261028",
  ]) == 1
  report = json.loads((out / "int8-chain.report.json").read_text())
  assert report["status"] == "FAIL" and report["counts"]["failed"] == 1
  assert report["counts"]["blocked"] == 0
  vectors = json.loads((out / "int8-chain.vectors.json").read_text())["vectors"]
  (failed,) = [item for item in vectors if item["id"] == "epi-custom-finite"]
  assert failed["pass"] is False and failed["label"] == LABEL_PRODUCTION


def test_param_rejects_at_real_boundaries(tmp_path):
  report = _run_cli_case("param-reject", Path(tmp_path), seed=20261028)
  assert report["status"] == "PASS", [c for c in report["checks"] if not c["pass"]]
  counts = report["counts"]
  assert counts["negative"] == len(_REJECT_PLAN) + len(_CODEC_TAMPERS)
  assert counts["vectors"] == counts["negative"]


def test_stage_matrix_full_bytes(tmp_path):
  report = _run_cli_case("stage-matrix", Path(tmp_path), seed=20261028)
  assert report["status"] == "PASS", [c for c in report["checks"] if not c["pass"]]
  counts = report["counts"]
  assert counts["epilogue_function_parity"] == 68
  assert counts["production_path"] == 3


def test_conv_pad_stride_dilation_case(tmp_path):
  report = _run_cli_case("conv-pad-stride-dilation", Path(tmp_path), seed=20261028)
  assert report["status"] == "PASS", [c for c in report["checks"] if not c["pass"]]
  assert report["counts"]["production_path"] == len(CASES)


def test_conv_traffic_case(tmp_path):
  report = _run_cli_case("conv-traffic", Path(tmp_path), seed=20261028)
  assert report["status"] == "PASS", [c for c in report["checks"] if not c["pass"]]
  assert report["counts"]["production_path"] == len(CASES)


def test_cli_rejects_unknown_case(tmp_path):
  with pytest.raises(SystemExit) as excinfo:
    main(["--case", "bogus", "--out", str(tmp_path), "--seed", "1"])
  assert excinfo.value.code == 2


def test_cli_rejects_unknown_config_key(tmp_path):
  config_path = Path(tmp_path) / "bad.json"
  config_path.write_text(json.dumps({"schema": 1, "bogus": True}), encoding="utf-8")
  assert main(
    ["--case", "conv-traffic", "--out", str(tmp_path / "o"), "--seed", "1", "--config", str(config_path)]
  ) == 2
