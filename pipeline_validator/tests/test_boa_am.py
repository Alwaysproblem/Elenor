"""Software gate for software-managed AM and typed BOA Matmul.

Covers the plan's IR01-IR17 items that are in scope before hardware:
source semantics, AM/L1 separation, capacity, four modes, numeric bytes,
lease quota/affinity, bounded commit timing, current-format replay,
wrong-target rejection, and refusal of legacy v4 artifacts.  Concrete
low-level retry CFG and RTL are intentionally absent.
"""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError, VerifyException

from pipeline_validator.compiled_program import (
  parse_compiled_program,
  seal_program,
  serialize_compiled_program,
)
from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import ExecTileOp, GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.memory.byte_store import ByteStore
from pipeline_validator.runtime.shared_am import AmOwner, SharedAmManager
from pipeline_validator.simulator import Simulator
from pipeline_validator.workload_ir import make_elenor_context, parse_workload_ir

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "examples" / "workloads" / "boa_am_matmul.mlir"
INPUTS = ROOT / "examples" / "workloads" / "boa_am_inputs" / "boa_am_matmul"
# Verbatim pre-P06 capture (never resealed, never upgraded in place); parsing
# it under the current compiler must fail at the version gate.
LEGACY_V4_ARTIFACT = ROOT / "pipeline_validator" / "tests" / "data" / "legacy_v4_boa_am.json"


def _hw(capacity: int = 65536) -> HardwareConfig:
  return HardwareConfig().with_overrides(boa_am_slot_capacity_bytes=capacity)


def _sim() -> SimConfig:
  return SimConfig(fidelity="full_memory", context_count=1)


def _bindings(a_bytes: int, b_bytes: int) -> dict[str, GlobalBinding]:
  return {
    "A": GlobalBinding("A", 0x100000, a_bytes, "r"),
    "B": GlobalBinding("B", 0x200000, b_bytes, "r"),
    "C": GlobalBinding("C", 0x300000, 65536, "w"),
  }


# ---------------------------------------------------------------------------
# Independent exact oracle (bf16_fp32_mac_rne_v1 semantics)
#
# Structurally independent of pipeline_validator.boa_numeric: BF16/FP32 words
# are decoded to exact dyadic rationals (numerator, power-of-two exponent)
# and every rounding is a single round-to-nearest-even over Python integers.
# ---------------------------------------------------------------------------


def _bf16_dyadic(bits: int) -> tuple[int, int]:
  """Exact value of one BF16 word as ``num * 2**exp`` (finite words only)."""
  sign = -1 if bits >> 15 else 1
  biased = (bits >> 7) & 0xFF
  frac = bits & 0x7F
  if biased == 0xFF:
    raise ValueError("oracle fixture must not contain Inf/NaN words")
  if biased == 0:
    return sign * frac, -133
  return sign * (128 + frac), biased - 134


def _bf16_eighths(bits: int) -> int:
  """Exact fixture word value in 1/8 units (``value = k * 2**-3``).

  Normalizes the decoded dyadic to the eighth-unit integer ``k`` and asserts
  the proven fixture bound (zero, or 1 <= |k| <= 56); words finer than 1/8
  are rejected instead of silently mis-scaled.
  """
  num, exp = _bf16_dyadic(bits)
  shift = exp + 3
  if shift >= 0:
    scaled = num << shift
  else:
    den = 1 << -shift
    if num % den:
      raise AssertionError("fixture word is not a multiple of 1/8")
    scaled = num // den
  if scaled and not 1 <= abs(scaled) <= 56:
    raise AssertionError("fixture no longer within the proven exact 1/8-unit bound")
  return scaled


def _f32_dyadic(bits: int) -> tuple[int, int]:
  """Exact value of one finite FP32 word as ``num * 2**exp``."""
  sign = -1 if bits >> 31 else 1
  biased = (bits >> 23) & 0xFF
  frac = bits & 0x007F_FFFF
  if biased == 0xFF:
    raise ValueError("oracle AM left the finite FP32 range")
  if biased == 0:
    return sign * frac, -149
  return sign * ((1 << 23) + frac), biased - 150


def _rne_f32(num: int, exp: int) -> int:
  """Round the exact dyadic value ``num * 2**exp`` to FP32 bits.

  One round-to-nearest-even at the target precision with an unbounded
  exponent range (tininess decided after rounding, per the frozen P03
  rules); overflow goes to same-sign infinity.
  """
  sign = 0x8000_0000 if num < 0 else 0
  if num < 0:
    num = -num
  if num == 0:
    return sign
  bl = num.bit_length()
  unbiased = bl - 1 + exp
  if unbiased > 127:
    return sign | 0x7F80_0000
  if unbiased >= -126:
    # Quantize the significand to 24 precision bits.
    right = bl - 24
    if right <= 0:
      mant = num << -right
    else:
      q, rem = divmod(num, 1 << right)
      half = 1 << (right - 1)
      if rem > half or (rem == half and q & 1):
        q += 1
      mant = q
    if mant == 1 << 24:  # rounding carried into a new binade
      if unbiased == 127:
        return sign | 0x7F80_0000
      mant, unbiased = 1 << 23, unbiased + 1
    return sign | ((unbiased + 127) << 23) | (mant - (1 << 23))
  # Subnormal grid: value = mant * 2**-149, mant in [0, 2**23).
  right = -149 - exp
  if right <= 0:
    mant = num << -right
  else:
    q, rem = divmod(num, 1 << right)
    half = 1 << (right - 1)
    if rem > half or (rem == half and q & 1):
      q += 1
    mant = q
  if mant == 1 << 23:  # tiny result rounded up to the minimum normal
    return sign | (1 << 23)
  return sign | mant


def _add_f32_rne(x_bits: int, y_bits: int) -> int:
  """Single-rounded exact FP32 addition of two raw words."""
  xn, xe = _f32_dyadic(x_bits)
  yn, ye = _f32_dyadic(y_bits)
  common = min(xe, ye)
  return _rne_f32(xn * (1 << (xe - common)) + yn * (1 << (ye - common)), common)


def _oracle(a: bytes, b: bytes, *, tile_k: int = 8) -> list[int]:
  """Independent oracle for the boa_am fixture chain.

  Fixture contract: A is block-packed [2, 64, 64] and B [2, 64, 256] BF16
  for two k=64 steps (step 0 INIT, step 1 ADD), m=64, n=256, storage_k=64
  per step.  Returns row-major FP32 bit patterns.

  Exactness proof (asserted below, not assumed): every fixture word decodes
  to a multiple of 1/8 with |value| <= 7, so every FMA product is a multiple
  of 1/64 with |product| <= 49, and every per-MAC partial sums at most
  128 such products: |partial| <= 401408 in 1/64 units, below 2**24.  Each
  partial is therefore exactly representable in FP32, which makes the
  production per-MAC single-rounded FMA chain equal to the exact block sum
  rounded once, and the INIT-overwrite / subsequent-block ``add_rne``
  block order below is the frozen P03 block semantics.
  """
  if tile_k < 1:
    raise ValueError("tile_k must be positive")
  a_words = [int.from_bytes(a[i : i + 2], "little") for i in range(0, len(a), 2)]
  b_words = [int.from_bytes(b[i : i + 2], "little") for i in range(0, len(b), 2)]
  if len(a_words) != 2 * 64 * 64 or len(b_words) != 2 * 64 * 256:
    raise ValueError("oracle fixture shape changed; re-derive the exactness bound")
  # Eighth-unit integers: value = k * 2**-3, so every product is a multiple
  # of 1/64 (numerator an*bn at exponent -6) and every partial of at most
  # 128 products stays below 2**24 in those units -> exactly representable.
  a_vals = [_bf16_eighths(word) for word in a_words]
  b_vals = [_bf16_eighths(word) for word in b_words]
  result = [0] * (64 * 256)
  for step in range(2):
    a_base = step * 64 * 64
    b_base = step * 64 * 256
    for start in range(0, 64, tile_k):
      overwrite = step == 0 and start == 0  # INIT's first block overwrites AM
      for row in range(64):
        row_base = a_base + row * 64
        out_base = row * 256
        for col in range(256):
          # Exact block sum in 1/64 units (every product exponent is -6).
          num = 0
          for kk in range(start, min(start + tile_k, 64)):
            num += a_vals[row_base + kk] * b_vals[b_base + kk * 256 + col]
          block = _rne_f32(num, -6)
          result[out_base + col] = (
            block if overwrite else _add_f32_rne(result[out_base + col], block)
          )
  return result


def _compile_and_run(
  text: str | None = None,
  *,
  capacity: int = 65536,
  c_bytes: int = 65536,
  block_cycles: int | None = None,
):
  a = (INPUTS / "A.bin").read_bytes()
  b = (INPUTS / "B.bin").read_bytes()
  bindings = _bindings(len(a), len(b))
  bindings["C"] = GlobalBinding("C", 0x300000, c_bytes, "w")
  hw = _hw(capacity)
  sim_cfg = _sim()
  module = parse_workload_ir(text) if text is not None else parse_workload_ir(FIXTURE.read_text())
  artifact = compile_program(
    module,
    hw,
    sim_cfg,
    binding_assumptions=bindings,
    source_name=str(FIXTURE),
  )
  loaded = load_program(artifact, hw, sim_cfg, actual_bindings=bindings)
  store = ByteStore()
  store.seed_hbm(0x100000, a)
  store.seed_hbm(0x200000, b)
  simulator = Simulator(hw, sim_cfg, byte_store=store)
  if block_cycles is not None:
    service = simulator.group.tiles[0].boa_results
    service.set_output_blocked(True)
    original_step = simulator.group.step
    resume_cycle = None

    def step(cycle: int) -> bool:
      nonlocal resume_cycle
      # Port stimulus follows real readiness; no assumed absolute compute time.
      if resume_cycle is None and service.snapshot()["states"]["wait_output"]:
        resume_cycle = cycle + block_cycles
      if resume_cycle is not None and cycle >= resume_cycle:
        service.set_output_blocked(False)
      return original_step(cycle)

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(simulator.group, "step", step)
      result = simulator.run(loaded)
  else:
    result = simulator.run(loaded)
  return artifact, simulator, store, result


def _first_program(artifact):
  task = next(iter(artifact.entry.tasks.values()))
  return next(iter(task.role_bindings.values())).tile_program


def _fixture_text() -> str:
  return FIXTURE.read_text()


def _tile_program_text(body: str) -> str:
  return f'''builtin.module {{
  tile.program @p(%task : !nest.task)
      resource_contract = #tile.resources<allowed_profiles = [0, 1],
          tile_l1_spm_bytes_per_context = 147456> {{
    %a = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %b = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %out = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<64x256xf32>
    %acc = tile.alloc memory_space = "am" logical_slot = 0 output(%out)
        shape = [64, 256] dtype = "f32" alignment = 256
        : !tile.am_buffer<64x256xf32>
{body}
    tile.return
  }}
}}
'''


def _never_woken(lease, cycle):
  raise AssertionError(f"unexpected AM wake for lease {lease.lease_id} at {cycle}")


def _acquire(manager, owner, *, bundle_slots: int = 1, wake=_never_woken, **kwargs):
  """Issue one ALLOC_AM request; ``None`` means parked until ``wake``."""
  return manager.request(owner, bundle_slots=bundle_slots, wake=wake, **kwargs)


def _verify_program_text(text: str):
  module = Parser(make_elenor_context(), text).parse_module()
  from pipeline_validator.dialects.elenor import TileProgramDefOp
  from pipeline_validator.workload_ir import _verify_program

  program = next(op for op in module.body.block.ops if isinstance(op, TileProgramDefOp))
  return _verify_program(program)


# ---------------------------------------------------------------------------
# IR01/IR02: AM is not L1; explicit capacity is enforced
# ---------------------------------------------------------------------------


def test_ir01_am_declaration_is_not_in_l1_arena():
  artifact, simulator, _store, result = _compile_and_run()
  program = _first_program(artifact)
  assert result.completed
  assert [am.name for am in program.am_buffers] == ["am:0"]
  assert all(buffer.name.startswith("l1:") for buffer in program.l1_buffers)
  assert len(program.l1_buffers) == 5
  assert all(item.buffer_id != "am:0" for item in program.layout.buffer_layouts)
  assert ExecTileOp.ALLOC_AM in [inst.op for inst in program.insts]
  assert simulator.group.tiles[0].shared_am.snapshot()["acquire_count"] == 1


def test_ir02_am_capacity_is_explicit_and_not_grown():
  with pytest.raises(ValueError, match="AM buffer"):
    _compile_and_run(capacity=32768)
  artifact, simulator, _store, _result = _compile_and_run(capacity=65536)
  assert _first_program(artifact).am_buffers[0].bytes == 65536
  assert simulator.group.tiles[0].shared_am.snapshot()["slot_capacity_bytes"] == 65536
  large = _hw(262144)
  small = _hw(65536)
  from pipeline_validator.execution_verifier import target_fingerprint

  assert target_fingerprint(large, _sim()) != target_fingerprint(small, _sim())


# ---------------------------------------------------------------------------
# IR03/IR05/IR07/IR14: real numerical bytes, automatic final release
# ---------------------------------------------------------------------------


def test_ir03_05_07_f32_output_matches_independent_oracle():
  _artifact, simulator, store, result = _compile_and_run()
  a = (INPUTS / "A.bin").read_bytes()
  b = (INPUTS / "B.bin").read_bytes()
  expected = struct.pack("<16384I", *_oracle(a, b))
  actual = store.read_hbm(0x300000, 65536)
  assert result.completed
  assert actual == expected
  snapshot = simulator.group.tiles[0].shared_am.snapshot()
  assert snapshot["live_leases"] == 0
  assert snapshot["free_slots"] == snapshot["total_slots"] == 8
  assert snapshot["terminals"][-1]["kind"] == "chain_done"
  assert snapshot["terminals"][-1]["success"] is True


def _bf16_bytes(f32_bits: int) -> bytes:
  """RNE BF16 conversion of one raw FP32 word with canonical positive NaN."""
  if f32_bits & 0x7F800000 == 0x7F800000 and f32_bits & 0x007FFFFF:
    return struct.pack("<H", 0x7FC0)
  rounded = f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)
  return struct.pack("<H", (rounded >> 16) & 0xFFFF)


def test_ir07_bf16_output_is_converted_only_at_final():
  text = (
    _fixture_text()
    .replace(
      '%out = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "f32"',
      '%out = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "bf16"',
    )
    .replace("!tile.l1_buffer<64x256xf32>", "!tile.l1_buffer<64x256xbf16>")
    .replace("!nest.l2_buffer<64x256xf32>", "!nest.l2_buffer<64x256xbf16>")
    .replace("!nest.l2_view<64x256xf32>", "!nest.l2_view<64x256xbf16>")
    .replace("!nest.global_memref<64x256xf32>", "!nest.global_memref<64x256xbf16>")
    .replace("!nest.global_view<64x256xf32>", "!nest.global_view<64x256xbf16>")
    .replace(
      '%c_l2_1 = nest.alloc slot = "c" role = "out" shape = [64, 256] dtype = "f32"',
      '%c_l2_1 = nest.alloc slot = "c" role = "out" shape = [64, 256] dtype = "bf16"',
    )
  )
  _artifact, _simulator, store, result = _compile_and_run(text, c_bytes=32768)
  a = (INPUTS / "A.bin").read_bytes()
  b = (INPUTS / "B.bin").read_bytes()
  expected = b"".join(_bf16_bytes(bits) for bits in _oracle(a, b))
  assert result.completed
  assert store.read_hbm(0x300000, 32768) == expected


# ---------------------------------------------------------------------------
# IR09/IR10: quota, busy without side effect, affinity/engine handles
# ---------------------------------------------------------------------------


def _request_kwargs(name: str, logical_slot: int, cycle: int) -> dict:
  return {
    "buffer_name": name,
    "logical_slot": logical_slot,
    "dims": (64, 256),
    "byte_count": 65536,
    "output_base": f"l1:{name}",
    "cycle": cycle,
  }


def _finish_chain(manager, lease, cycle: int) -> None:
  """Drive one granted chain through INIT+FINAL, commit and slot_free."""
  payload = b"\x00" * (64 * 256 * 4)
  manager.begin_step(lease.lease_id, seq=0, init=True, final=True, cycle=cycle)
  manager.commit_step(lease.lease_id, seq=0, data=payload, final=True, cycle=cycle + 1)
  manager.release_final(lease.lease_id, cycle + 2)


def test_ir09_allocation_beyond_bundle_is_rejected_without_side_effect():
  manager = SharedAmManager(
    slot_capacity_bytes=65536,
    group_count=4,
    slots_per_group=2,
    slots_per_context=2,
  )
  owner = AmOwner("ctx", 1, 0, 0)
  first = _acquire(manager, owner, bundle_slots=2, **_request_kwargs("a", 0, 0))
  second = _acquire(manager, owner, bundle_slots=2, **_request_kwargs("b", 1, 0))
  assert first is not None and second is not None
  before = manager.snapshot()
  # Both chains still need UCE instructions to reach FINAL: blocking a third
  # request on them would self-deadlock, so it is a contract error, not a wait.
  with pytest.raises(ValueError):
    _acquire(manager, owner, bundle_slots=2, **_request_kwargs("c", 0, 1))
  after = manager.snapshot()
  assert after["live_leases"] == before["live_leases"] == 2
  assert after["slots"] == before["slots"]
  assert after["pending_requests"] == 0

  manager.abort_lease(first.lease_id, 2, "test")
  fourth = _acquire(manager, owner, bundle_slots=2, **_request_kwargs("d", 0, 3))
  assert fourth is not None


def test_ir10_affinity_is_fixed_and_one_engine_serves_many_handles():
  manager = SharedAmManager(
    slot_capacity_bytes=65536,
    group_count=4,
    slots_per_group=2,
    slots_per_context=2,
    engine_count=1,
  )
  owner = AmOwner("ctx", 1, 0, 0)
  engines = []
  for index in range(6):
    lease = _acquire(
      manager,
      owner,
      buffer_name=f"a{index}",
      logical_slot=index % 2,
      dims=(64, 256),
      byte_count=65536,
      output_base=f"l1:{index}",
      cycle=index,
    )
    assert lease is not None
    # One engine serves every handle; nothing re-pins a live lease.
    assert lease.bound_engine == 0
    engines.append(lease.bound_engine)
    snapshot = manager.snapshot()
    assert snapshot["engine_handle_counts"] == {0: 1}
    assert sum(snapshot["engine_handle_counts"].values()) == snapshot["live_leases"]
    manager.abort_lease(lease.lease_id, index + 1, "test")
    assert manager.snapshot()["engine_handle_counts"] == {0: 0}
  assert engines == [0] * 6


# ---------------------------------------------------------------------------
# IR11: the completion event is a commit, not raw compute-done
# ---------------------------------------------------------------------------


def test_ir11_typed_boa_commit_is_after_compute_and_drain():
  artifact, simulator, _store, result = _compile_and_run()
  program = _first_program(artifact)
  typed = [
    descriptor
    for descriptor in program.descriptors.values()
    if descriptor.params.get("matmul") is not None
  ]
  assert len(typed) == 2
  service = simulator.group.tiles[0].boa_results.snapshot()
  finals = [record for record in service["records"] if record["final"]]
  assert len(finals) == 1
  record = finals[0]
  assert record["compute_done_cycle"] >= 0
  assert record["commit_cycle"] >= record["compute_done_cycle"]
  assert record["output_cycle"] is not None
  assert record["output_cycle"] >= record["compute_done_cycle"]
  assert record["event_cycle"] >= record["output_cycle"]
  assert service["release_count"] == 1
  assert service["output_write_count"] == 1
  assert service["credits"] == service["credit_capacity"]
  assert result.cycles >= record["event_cycle"]


# ---------------------------------------------------------------------------
# IR12/IR13: source linear lifetime and descriptor snapshot semantics
# ---------------------------------------------------------------------------


def test_ir15_missing_final_is_rejected():
  text = _tile_program_text(
    '''    %k0 = tile.boa.async "matmul" ins(%a, %b) outs(%out) acc(%acc)
        m = 64 n = 256 k = 64 seq = 0 : !tile.event<"k0">
    tile.await %k0'''
  )
  with pytest.raises(VerifyException, match="must terminate with one output"):
    _verify_program_text(text)


def test_ir08_wrong_seq_and_extra_final_are_rejected():
  wrong_seq = _tile_program_text(
    '''    %k0 = tile.boa.async "matmul" ins(%a, %b) outs(%out) acc(%acc)
        m = 64 n = 256 k = 64 seq = 2 : !tile.event<"k0">
    tile.await %k0'''
  )
  with pytest.raises(VerifyException, match="seq must be 0"):
    _verify_program_text(wrong_seq)
  extra_final = _tile_program_text(
    '''    %k0 = tile.boa.async "matmul" ins(%a, %b) outs(%out) acc(%acc)
        m = 64 n = 256 k = 64 seq = 0 output : !tile.event<"k0">
    tile.await %k0
    %k1 = tile.boa.async "matmul" ins(%a, %b) outs(%out) acc(%acc)
        m = 64 n = 256 k = 64 seq = 1 accumulate output : !tile.event<"k1">
    tile.await %k1'''
  )
  with pytest.raises(VerifyException, match=r"not yet finalized|finalized"):
    _verify_program_text(extra_final)


def test_am_cannot_be_freed_or_loaded_as_l1():
  free_am = _tile_program_text(
    '''    %k0 = tile.boa.async "matmul" ins(%a, %b) outs(%out) acc(%acc)
        m = 64 n = 256 k = 64 seq = 0 output : !tile.event<"k0">
    tile.await %k0
    tile.free %acc'''
  )
  with pytest.raises((ParseError, VerifyException)):
    _verify_program_text(free_am)

  am_operand = _tile_program_text(
    '''    %k0 = tile.load.async %a_view into %acc : !tile.event<"k0">
    tile.await %k0'''
  )
  with pytest.raises((ParseError, VerifyException)):
    _verify_program_text(am_operand)


def test_two_independent_chains_are_expressible():
  text = '''builtin.module {
  tile.program @p(%task : !nest.task,
      %a0_l2 : !nest.l2_buffer<64x64xbf16>,
      %b0_l2 : !nest.l2_buffer<64x256xbf16>,
      %a1_l2 : !nest.l2_buffer<64x64xbf16>,
      %b1_l2 : !nest.l2_buffer<64x256xbf16>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1],
          tile_l1_spm_bytes_per_context = 294912> {
    %a0v = tile.subview %a0_l2 offsets = [0, 0] sizes = [64, 64] strides = [1, 1]
        : !nest.l2_view<64x64xbf16>
    %b0v = tile.subview %b0_l2 offsets = [0, 0] sizes = [64, 256] strides = [1, 1]
        : !nest.l2_view<64x256xbf16>
    %a1v = tile.subview %a1_l2 offsets = [0, 0] sizes = [64, 64] strides = [1, 1]
        : !nest.l2_view<64x64xbf16>
    %b1v = tile.subview %b1_l2 offsets = [0, 0] sizes = [64, 256] strides = [1, 1]
        : !nest.l2_view<64x256xbf16>
    %a0 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %b0 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %a1 = tile.alloc shape = [64, 64] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x64xbf16>
    %b1 = tile.alloc shape = [64, 256] dtype = "bf16"
        alignment = 256 : !tile.l1_buffer<64x256xbf16>
    %out0 = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<64x256xf32>
    %out1 = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "f32"
        alignment = 256 : !tile.l1_buffer<64x256xf32>
    %acc0 = tile.alloc memory_space = "am" logical_slot = 0 output(%out0)
        shape = [64, 256] dtype = "f32" alignment = 256 : !tile.am_buffer<64x256xf32>
    %acc1 = tile.alloc memory_space = "am" logical_slot = 1 output(%out1)
        shape = [64, 256] dtype = "f32" alignment = 256 : !tile.am_buffer<64x256xf32>
    %la0 = tile.load.async %a0v into %a0 : !tile.event<"la0">
    %lb0 = tile.load.async %b0v into %b0 : !tile.event<"lb0">
    %la1 = tile.load.async %a1v into %a1 : !tile.event<"la1">
    %lb1 = tile.load.async %b1v into %b1 : !tile.event<"lb1">
    tile.await %la0, %lb0, %la1, %lb1
    tile.signal input_released(%task)
    %k0 = tile.boa.async "matmul" ins(%a0, %b0) outs(%out0) acc(%acc0)
        m = 64 n = 256 k = 64 seq = 0 : !tile.event<"k0">
    %k1 = tile.boa.async "matmul" ins(%a1, %b1) outs(%out1) acc(%acc1)
        m = 64 n = 256 k = 64 seq = 0 : !tile.event<"k1">
    tile.await %k0
    tile.await %k1
    %d0 = tile.boa.async "matmul" ins(%a0, %b0) outs(%out0) acc(%acc0)
        m = 64 n = 256 k = 64 seq = 1 accumulate output : !tile.event<"d0">
    %d1 = tile.boa.async "matmul" ins(%a1, %b1) outs(%out1) acc(%acc1)
        m = 64 n = 256 k = 64 seq = 1 accumulate output : !tile.event<"d1">
    tile.await %d0, %d1
    tile.return
  }
}
'''
  _verify_program_text(text)


# ---------------------------------------------------------------------------
# IR16/P06: current-format replay, legacy-version and wrong-target rejection
# ---------------------------------------------------------------------------


def test_legacy_v4_artifact_is_rejected_at_the_version_gate():
  """The verbatim pre-cutover v4 capture must fail on schema/ABI version at
  parse time; old bytes are never upgraded in place and never reach the
  fingerprint check."""
  text = LEGACY_V4_ARTIFACT.read_text()
  document = json.loads(text)
  assert document["schema_version"] == 4 and document["compiler_abi"] == "v4"
  with pytest.raises(ValueError) as excinfo:
    parse_compiled_program(text)
  message = str(excinfo.value)
  assert "recompile" in message or "unsupported compiled schema" in message
  assert "fingerprint" not in message


def test_ir16_serialized_artifact_runs_and_wrong_target_fingerprint_is_rejected():
  a = (INPUTS / "A.bin").read_bytes()
  b = (INPUTS / "B.bin").read_bytes()
  bindings = _bindings(len(a), len(b))
  hw, sim_cfg = _hw(), _sim()
  compiled = compile_program(
    parse_workload_ir(_fixture_text()), hw, sim_cfg, binding_assumptions=bindings,
    source_name=str(FIXTURE),
  )
  replay = parse_compiled_program(serialize_compiled_program(compiled))
  loaded = load_program(replay, hw, sim_cfg, actual_bindings=bindings)
  store = ByteStore()
  store.seed_hbm(0x100000, a)
  store.seed_hbm(0x200000, b)
  result = Simulator(hw, sim_cfg, byte_store=store).run(loaded)
  assert result.completed, result.reason
  assert store.read_hbm(0x300000, 65536) == struct.pack("<16384I", *_oracle(a, b))
  assert all(
    pool["live_leases"] == pool["pending_requests"] == pool["reserved_slots"] == 0
    and pool["free_slots"] == pool["total_slots"]
    for pool in result.group_snapshot["shared_am_per_tile"]
  )
  # Independent second negative: a freshly compiled current-format artifact
  # loaded against a different static target (the numeric tile_k field is
  # part of the fingerprint) is refused.  This is distinct from the
  # legacy-version rejection above and must not collapse into it.
  other_hw = _hw().with_overrides(boa_numeric_tile_k=16)
  with pytest.raises(ValueError, match="target fingerprint mismatch"):
    load_program(replay, other_hw, sim_cfg, actual_bindings=bindings)


def test_ir16_resealed_tampered_am_binding_is_rejected():
  artifact, _simulator, _store, _result = _compile_and_run()
  task = next(iter(artifact.entry.tasks.values()))
  binding = next(iter(task.role_bindings.values()))
  program = binding.tile_program
  descriptors = dict(program.descriptors)
  name = next(
    key for key, desc in descriptors.items() if desc.params.get("matmul") is not None
  )
  descriptor = descriptors[name]
  matmul = descriptor.params["matmul"]
  from pipeline_validator.immutable import FrozenMap

  descriptors[name] = replace(
    descriptor,
    params=FrozenMap({"matmul": replace(matmul, seq=1, init=False)}),
  )
  tampered_program = replace(
    program,
    descriptors=descriptors,
    program_hash=0,
  )
  from pipeline_validator.compiled_program import program_digest

  tampered_program = replace(
    tampered_program, program_hash=program_digest(tampered_program)
  )
  tampered_binding = replace(binding, tile_program=tampered_program)
  tampered_task = replace(task, role_bindings={binding.role_id: tampered_binding})
  tampered_entry = replace(
    artifact.entry, tasks={task.binding_id: tampered_task}
  )
  tampered = seal_program(replace(artifact, entry=tampered_entry))
  with pytest.raises(ValueError, match=r"AM chain sequence|INIT/ADD"):
    load_program(tampered, _hw(), _sim())

  # An absent AM reference is caught even earlier by the strict codec.
  descriptors[name] = replace(
    descriptor, params=FrozenMap({"matmul": replace(matmul, am="am:missing")})
  )
  bad_program = replace(
    program, descriptors=descriptors, program_hash=0
  )
  bad_program = replace(bad_program, program_hash=program_digest(bad_program))
  bad_entry = replace(
    artifact.entry,
    tasks={
      task.binding_id: replace(
        task,
        role_bindings={
          binding.role_id: replace(binding, tile_program=bad_program)
        },
      )
    },
  )
  with pytest.raises(ValueError, match="absent AM buffer"):
    seal_program(replace(artifact, entry=bad_entry))


def test_ir17_estimate_is_explicit_and_not_an_am_chain():
  from pipeline_validator.dialects.elenor import TileBoaEstimateOp

  op = TileBoaEstimateOp("matmul", 64, 64, 64, 524288, "e")
  assert op.name == "tile.boa.estimate.async"
  assert not hasattr(op, "acc")
  text = _tile_program_text(
    '''    %e = tile.boa.estimate.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        : !tile.event<"e">
    tile.await %e'''
  )
  # The AM declaration in the helper remains unterminated; the estimate is
  # not allowed to stand in for its FINAL.
  with pytest.raises(VerifyException, match="must terminate with one output"):
    _verify_program_text(text)


def test_am_slot_reuse_increments_generation_and_keeps_terminal():
  manager = SharedAmManager(
    slot_capacity_bytes=65536,
    group_count=1,
    slots_per_group=1,
    slots_per_context=1,
  )
  owner = AmOwner("ctx", 1, 0, 0)
  first = _acquire(
    manager,
    owner,
    buffer_name="a",
    logical_slot=0,
    dims=(64, 128),
    byte_count=32768,
    output_base="l1:0",
    cycle=0,
  )
  assert first is not None
  payload = b"\x00" * (64 * 128 * 4)
  manager.begin_step(first.lease_id, seq=0, init=True, final=False, cycle=1)
  manager.commit_step(first.lease_id, seq=0, data=payload, final=True, cycle=2)
  manager.release_final(first.lease_id, 3)
  second = _acquire(
    manager,
    owner,
    buffer_name="b",
    logical_slot=0,
    dims=(64, 128),
    byte_count=32768,
    output_base="l1:1",
    cycle=4,
  )
  assert second is not None
  assert second.generation == first.generation + 1
  snapshot = manager.snapshot()
  assert snapshot["terminals"][0]["kind"] == "chain_done"
  assert snapshot["terminals"][0]["generation"] == first.generation


def test_am_release_before_final_and_fault_cleanup():
  manager = SharedAmManager(
    slot_capacity_bytes=65536,
    group_count=1,
    slots_per_group=2,
    slots_per_context=2,
  )
  owner = AmOwner("ctx", 1, 0, 0)
  lease = _acquire(
    manager,
    owner,
    buffer_name="a",
    logical_slot=0,
    dims=(64, 256),
    byte_count=65536,
    output_base="l1:0",
    cycle=0,
  )
  assert lease is not None
  manager.begin_step(lease.lease_id, seq=0, init=True, final=False, cycle=1)
  with pytest.raises(ValueError, match="not ready for final release"):
    manager.release_final(lease.lease_id, 2)
  manager.abort_owner(owner, 3, "fault")
  snapshot = manager.snapshot()
  assert snapshot["live_leases"] == 0
  assert snapshot["fault_count"] == 1
  assert snapshot["terminals"][-1]["kind"] == "fault"
  assert snapshot["free_slots"] == snapshot["total_slots"]


def test_old_operandless_boa_syntax_is_rejected():
  text = _tile_program_text(
    '''    %k0 = tile.boa.async "matmul" m = 64 n = 256 k = 64 ops = 2097152
        : !tile.event<"k0">
    tile.await %k0'''
  )
  with pytest.raises((ParseError, VerifyException)):
    _verify_program_text(text)


def test_am_i8_output_is_accepted_by_source_but_rejected_numerically():
  text = _fixture_text().replace(
    '%out = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "f32"',
    '%out = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "i8"',
  )
  text = text.replace("!tile.l1_buffer<64x256xf32>", "!tile.l1_buffer<64x256xi8>", 1)
  text = (
    text.replace("!nest.l2_buffer<64x256xf32>", "!nest.l2_buffer<64x256xi8>")
    .replace("!nest.l2_view<64x256xf32>", "!nest.l2_view<64x256xi8>")
    .replace("!nest.global_memref<64x256xf32>", "!nest.global_memref<64x256xi8>")
    .replace("!nest.global_view<64x256xf32>", "!nest.global_view<64x256xi8>")
    .replace(
      '%c_l2_1 = nest.alloc slot = "c" role = "out" shape = [64, 256] dtype = "f32"',
      '%c_l2_1 = nest.alloc slot = "c" role = "out" shape = [64, 256] dtype = "i8"',
    )
  )
  _artifact, simulator, _store, result = _compile_and_run(text, c_bytes=16384)
  assert not result.completed
  assert "INT8 AM output conversion policy" in result.reason
  snapshot = simulator.group.tiles[0].shared_am.snapshot()
  assert snapshot["fault_count"] == 1
  assert snapshot["live_leases"] == 0


def test_missing_output_binding_is_rejected():
  text = '''builtin.module {
  tile.program @p(%task : !nest.task)
      resource_contract = #tile.resources<allowed_profiles = [0, 1],
          tile_l1_spm_bytes_per_context = 131072> {
    %acc = tile.alloc memory_space = "am" logical_slot = 0
        shape = [64, 256] dtype = "f32" alignment = 256
        : !tile.am_buffer<64x256xf32>
    tile.return
  }
}
'''
  with pytest.raises((ParseError, VerifyException)):
    _verify_program_text(text)


def test_eight_contexts_with_one_lease_each_fill_all_slots():
  manager = SharedAmManager(
    slot_capacity_bytes=65536,
    group_count=4,
    slots_per_group=2,
    slots_per_context=2,
  )
  leases = []
  for index in range(8):
    lease = _acquire(
    manager,
      AmOwner("ctx", 1, index, 0),
      buffer_name="a",
      logical_slot=0,
      dims=(64, 256),
      byte_count=65536,
      output_base=f"l1:{index}",
      cycle=index,
    )
    assert lease is not None
    leases.append(lease)
  assert manager.snapshot()["free_slots"] == 0
  woken: list = []
  ninth = _acquire(
    manager,
    AmOwner("ctx", 1, 8, 0),
    wake=lambda lease, cycle: woken.append((lease, cycle)),
    **_request_kwargs("a", 0, 9),
  )
  assert ninth is None
  assert manager.snapshot()["live_leases"] == 8
  assert manager.snapshot()["pending_requests"] == 1
  manager.abort_lease(leases[0].lease_id, 10, "test")
  assert len(woken) == 1 and woken[0][1] == 10
  assert woken[0][0].owner == AmOwner("ctx", 1, 8, 0)
  for lease in leases[1:]:
    manager.abort_lease(lease.lease_id, 11, "test")
  manager.abort_lease(woken[0][0].lease_id, 12, "test")
  assert manager.snapshot()["free_slots"] == 8


def _prepared_final(manager, lease_id, *, final: bool, events: list):
  from types import SimpleNamespace
  from typing import cast

  from pipeline_validator.execution_ir import ExecBoaMatmulDesc
  from pipeline_validator.runtime.boa_result import BoaPreparedResult

  payload = b"\x01" * (64 * 256 * 4)
  matmul = cast(
    ExecBoaMatmulDesc,
    SimpleNamespace(am="am:0", final=final, seq=0, init=True),
  )

  def commit(cycle: int) -> None:
    manager.commit_step(lease_id, seq=0, data=payload, final=False, cycle=cycle)
    events.append(("commit", cycle))

  def write_output(cycle: int) -> None:
    events.append(("write", cycle))

  def commit_final(cycle: int) -> None:
    manager.commit_step(lease_id, seq=0, data=payload, final=True, cycle=cycle)
    events.append(("commit_final", cycle))

  def release(cycle: int) -> None:
    manager.release_final(lease_id, cycle)
    events.append(("release", cycle))

  def abort(cycle: int, reason: str) -> None:
    manager.abort_lease(lease_id, cycle, reason)
    events.append(("abort", cycle))

  return BoaPreparedResult(
    matmul, payload, final, commit, commit_final, write_output, release, abort
  )


def _pending_job():
  from pipeline_validator.engines import EngineJob

  job = EngineJob(
    desc=None,  # type: ignore[arg-type]
    start_cycle=0,
    finish_cycle=0,
    event_id="e",
  )
  job.compute_done_cycle = 0
  return job


def test_ir11_nonfinal_result_commits_in_a_later_stage():
  from pipeline_validator.runtime.boa_result import BoaResultService

  manager = SharedAmManager(
    slot_capacity_bytes=65536,
    group_count=1,
    slots_per_group=1,
    slots_per_context=1,
  )
  owner = AmOwner("ctx", 1, 0, 0)
  lease = _acquire(
    manager,
    owner,
    buffer_name="a",
    logical_slot=0,
    dims=(64, 256),
    byte_count=65536,
    output_base="l1:0",
    cycle=0,
  )
  assert lease is not None
  manager.begin_step(lease.lease_id, seq=0, init=True, final=False, cycle=0)
  service = BoaResultService(_hw())
  events: list[tuple[str, int]] = []
  job = _pending_job()
  service.submit(job, _prepared_final(manager, lease.lease_id, final=False, events=events), 0)
  assert service.advance(0) == []
  assert service.advance(1) == []
  assert manager.lease(lease.lease_id).partial is None
  ready = service.advance(2)
  assert ready == [job]
  assert events == [("commit", 2)]
  assert job.event_cycle == 2
  assert manager.lease(lease.lease_id).state.value == "running"
  assert manager.snapshot()["live_leases"] == 1
  assert service.snapshot()["event_count"] == 1


def test_ir14_final_output_backpressure_defers_release_and_event():
  from pipeline_validator.runtime.boa_result import BoaResultService

  manager = SharedAmManager(
    slot_capacity_bytes=65536,
    group_count=1,
    slots_per_group=1,
    slots_per_context=1,
  )
  owner = AmOwner("ctx", 1, 0, 0)
  lease = _acquire(
    manager,
    owner,
    buffer_name="a",
    logical_slot=0,
    dims=(64, 256),
    byte_count=65536,
    output_base="l1:0",
    cycle=0,
  )
  assert lease is not None
  manager.begin_step(lease.lease_id, seq=0, init=True, final=True, cycle=0)
  service = BoaResultService(_hw())
  service.set_output_blocked(True)
  events: list[tuple[str, int]] = []
  job = _pending_job()
  service.submit(job, _prepared_final(manager, lease.lease_id, final=True, events=events), 0)
  assert service.advance(1) == []
  assert service.advance(2) == []
  assert service.advance(50) == []
  snapshot = service.snapshot()
  assert snapshot["states"]["wait_output"] == 1
  assert snapshot["blocked_cycles"] > 0
  assert manager.snapshot()["live_leases"] == 1
  assert manager.snapshot()["free_slots"] == 0
  assert events == []
  assert job.event_cycle is None

  service.set_output_blocked(False)
  ready = []
  for cycle in range(51, 70):
    ready.extend(service.advance(cycle))
  assert ready == [job]
  assert [name for name, _ in events] == ["write", "commit_final", "release"]
  assert events[0][1] >= 51
  assert events[0][1] <= events[1][1] <= events[2][1]
  assert job.event_cycle == events[2][1]
  assert manager.snapshot()["live_leases"] == 0
  assert manager.snapshot()["terminals"][-1]["kind"] == "chain_done"
  record = service.snapshot()["records"][-1]
  assert record["compute_done_cycle"] == 0
  assert record["output_cycle"] is not None
  assert record["event_cycle"] >= record["output_cycle"]


def test_ir14_runtime_final_output_backpressure_recovers_without_polling():
  hold_cycles = 11
  _artifact, simulator, store, result = _compile_and_run(block_cycles=hold_cycles)
  assert result.completed
  service = simulator.group.tiles[0].boa_results.snapshot()
  finals = [record for record in service["records"] if record["final"]]
  assert len(finals) == 1
  record = finals[0]
  assert service["blocked_cycles"] >= hold_cycles
  assert record["output_cycle"] - record["compute_done_cycle"] >= hold_cycles
  assert record["event_cycle"] >= record["output_cycle"]
  assert simulator.group.tiles[0].shared_am.snapshot()["live_leases"] == 0
  assert simulator.group.tiles[0].shared_am.snapshot()["terminals"][-1]["kind"] == "chain_done"
  a = (INPUTS / "A.bin").read_bytes()
  b = (INPUTS / "B.bin").read_bytes()
  assert store.read_hbm(0x300000, 65536) == struct.pack("<16384I", *_oracle(a, b))


def test_am_request_parks_once_and_wakes_on_slot_free():
  manager = SharedAmManager(
    slot_capacity_bytes=65536, group_count=1, slots_per_group=1, slots_per_context=1
  )
  holder = AmOwner("holder", 1, 0, 0)
  waiter = AmOwner("waiter", 1, 1, 0)
  first = _acquire(manager, holder, **_request_kwargs("a", 0, 0))
  assert first is not None
  wakes: list = []
  parked = _acquire(
    manager,
    waiter,
    wake=lambda lease, cycle: wakes.append((lease.owner, lease.generation, cycle)),
    **_request_kwargs("a", 0, 1),
  )
  assert parked is None
  before = manager.snapshot()
  assert before["pending_requests"] == 1 and before["parked_count"] == 1
  assert before["slots"][0]["generation"] == first.generation
  # Nothing re-issues the request while it waits; the parked count stays 1.
  _finish_chain(manager, first, 100)
  assert wakes == [(waiter, first.generation + 1, 102)]
  after = manager.snapshot()
  assert after["pending_requests"] == 0
  assert after["parked_count"] == 1 and after["request_count"] == 2
  assert manager.live_for_owner(waiter) == 1


def test_am_bundle_reserves_continuation_and_prevents_hold_and_wait():
  manager = SharedAmManager(
    slot_capacity_bytes=65536, group_count=2, slots_per_group=1, slots_per_context=2
  )
  owner_a = AmOwner("a", 1, 0, 0)
  owner_b = AmOwner("b", 1, 1, 0)
  owner_c = AmOwner("c", 1, 2, 0)
  first_a = _acquire(manager, owner_a, bundle_slots=2, **_request_kwargs("a0", 0, 0))
  assert first_a is not None
  assert manager.snapshot()["reserved_slots"] == 1
  b_woken: list = []
  # The second physical slot is free but reserved for A's continuation, so
  # B parks instead of creating an A-holds-1/B-holds-1 hold-and-wait cycle.
  assert (
    _acquire(
      manager, owner_b, wake=lambda lease, cycle: b_woken.append(lease), **_request_kwargs("b0", 0, 1)
    )
    is None
  )
  second_a = _acquire(manager, owner_a, bundle_slots=2, **_request_kwargs("a1", 1, 2))
  assert second_a is not None
  assert manager.snapshot()["reserved_slots"] == 0
  # A fresh request may not overtake the FIFO head.
  assert _acquire(manager, owner_c, wake=lambda lease, cycle: None, **_request_kwargs("c0", 0, 3)) is None
  _finish_chain(manager, second_a, 10)
  # A still holds a0 and keeps its bundle: the freed slot returns to A's
  # reservation, not to B, until A has no live lease.
  assert b_woken == []
  assert manager.snapshot()["reserved_slots"] == 1
  _finish_chain(manager, first_a, 20)
  assert len(b_woken) == 1
  assert manager.snapshot()["pending_requests"] == 0
  assert manager.live_for_owner(owner_c) == 1


def test_am_continuation_waits_only_for_own_draining_final():
  manager = SharedAmManager(
    slot_capacity_bytes=65536, group_count=2, slots_per_group=1, slots_per_context=2
  )
  owner = AmOwner("a", 1, 0, 0)
  other = AmOwner("b", 1, 1, 0)
  first = _acquire(manager, owner, bundle_slots=1, **_request_kwargs("a0", 0, 0))
  assert first is not None
  other_lease = _acquire(manager, other, **_request_kwargs("b0", 0, 0))
  assert other_lease is not None
  with pytest.raises(ValueError, match="bundle"):
    _acquire(manager, owner, bundle_slots=1, **_request_kwargs("a1", 1, 1))
  payload = b"\x00" * (64 * 256 * 4)
  manager.begin_step(first.lease_id, seq=0, init=True, final=True, cycle=2)
  woken: list = []
  # FINAL accepted: the chain drains autonomously, so the next chain may
  # block on this owner's own slot without a self-deadlock.
  assert (
    _acquire(
      manager,
      owner,
      bundle_slots=1,
      wake=lambda lease, cycle: woken.append(cycle),
      **_request_kwargs("a1", 1, 3),
    )
    is None
  )
  manager.commit_step(first.lease_id, seq=0, data=payload, final=True, cycle=4)
  manager.release_final(first.lease_id, 5)
  assert woken == [5]
  assert manager.live_for_owner(owner) == 1


@pytest.mark.parametrize("quota", [1, 2])
def test_am_same_slot_reuse_waits_for_final_release(quota: int):
  manager = SharedAmManager(
    slot_capacity_bytes=65536, group_count=1, slots_per_group=quota, slots_per_context=quota
  )
  owner = AmOwner("reuse", 1, 0, 0)
  first = _acquire(manager, owner, bundle_slots=quota, **_request_kwargs("old", 0, 0))
  assert first is not None
  manager.begin_step(first.lease_id, seq=0, init=True, final=True, cycle=1)
  before = manager.snapshot()
  wakes: list = []
  assert (
    _acquire(
      manager,
      owner,
      bundle_slots=quota,
      wake=lambda lease, cycle: wakes.append((lease, cycle)),
      **_request_kwargs("new", 0, 2),
    )
    is None
  )
  parked = manager.snapshot()
  assert parked["slots"] == before["slots"]
  assert parked["live_leases"] == 1
  assert parked["parked_count"] == 1
  assert parked["pending_requests"] == 1
  assert wakes == []
  payload = b"\x00" * first.byte_count
  manager.commit_step(first.lease_id, seq=0, data=payload, final=True, cycle=3)
  assert wakes == []
  manager.release_final(first.lease_id, 4)
  assert len(wakes) == 1
  replacement, grant_cycle = wakes[0]
  assert grant_cycle == 4
  assert replacement.owner == owner and replacement.logical_slot == 0
  assert replacement.lease_id != first.lease_id
  assert replacement.generation == first.generation + 1
  assert manager.live_for_owner(owner) == 1
  assert manager.snapshot()["pending_requests"] == 0
  _finish_chain(manager, replacement, 5)
  snapshot = manager.snapshot()
  assert len(wakes) == 1
  assert snapshot["wake_count"] == 1
  assert snapshot["live_leases"] == snapshot["reserved_slots"] == 0
  assert snapshot["free_slots"] == quota


def test_am_cancelled_request_is_never_woken():
  manager = SharedAmManager(
    slot_capacity_bytes=65536, group_count=1, slots_per_group=1, slots_per_context=1
  )
  holder = AmOwner("holder", 1, 0, 0)
  waiter = AmOwner("waiter", 1, 1, 0)
  lease = _acquire(manager, holder, **_request_kwargs("a", 0, 0))
  assert lease is not None
  assert _acquire(manager, waiter, **_request_kwargs("a", 0, 1)) is None
  assert manager.cancel_requests(waiter, 2, "context fault") == 1
  _finish_chain(manager, lease, 3)
  snapshot = manager.snapshot()
  assert snapshot["cancelled_request_count"] == 1
  assert snapshot["free_slots"] == 1 and snapshot["bundles"] == []


def test_runtime_alloc_am_blocks_once_and_wakes_on_slot_free():
  hw = _hw().with_overrides(boa_am_group_count=1, boa_am_slots_per_group=1, boa_am_slots_per_context=1)
  a = (INPUTS / "A.bin").read_bytes()
  b = (INPUTS / "B.bin").read_bytes()
  bindings = _bindings(len(a), len(b))
  sim_cfg = replace(_sim(), context_count=2)
  artifact = compile_program(
    parse_workload_ir(FIXTURE.read_text()), hw, sim_cfg, binding_assumptions=bindings,
    source_name=str(FIXTURE),
  )
  loaded = load_program(artifact, hw, sim_cfg, actual_bindings=bindings)
  store = ByteStore()
  store.seed_hbm(0x100000, a)
  store.seed_hbm(0x200000, b)
  simulator = Simulator(hw, sim_cfg, byte_store=store)
  manager = simulator.group.tiles[0].shared_am
  blocker = manager.request(
    AmOwner("blocker", 1, 1, 0), bundle_slots=1, wake=_never_woken, **_request_kwargs("x", 0, 0)
  )
  assert blocker is not None
  release_cycle = 2000
  original_step = simulator.group.step

  def step(cycle: int) -> bool:
    if cycle == release_cycle:
      manager.abort_lease(blocker.lease_id, cycle, "test release")
    return original_step(cycle)

  simulator.group.step = step  # type: ignore[method-assign]
  result = simulator.run(loaded)
  assert result.completed
  snapshot = manager.snapshot()
  assert snapshot["parked_count"] == 1
  assert snapshot["wake_count"] == 1
  assert snapshot["request_count"] == 2
  assert snapshot["live_leases"] == 0
  assert store.read_hbm(0x300000, 65536) == struct.pack("<16384I", *_oracle(a, b))
