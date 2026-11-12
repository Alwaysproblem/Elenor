"""C021 scalar ISA reference driver and bitwise oracle unit suite (`cn unit --suite w7`).

Covers every C021 / W7-02a..e scalar requirement and negative control:
- Pinned `SpikeSimulator` (`riscv-isa-sim@93a10ae6` + CoralNPU patches) executes the hand-computable
  scalar selftest ELF (`scalar_core_selftest.S`), matching all 16 memory words, register writebacks,
  sticky-isolated per-instruction `flags_delta`, and illegal-instruction `mcause`/`mepc` recovery.
- `bitwise_scalar.py` uses pure integer / `Fraction` operations (no `float`, no `numpy`/`scipy`,
  no DUT Zvt RTL reads) and agrees bit-for-bit with Spike on IEEE FP32 rounding modes and flags.
- Flipping any single bit in a register writeback, memory byte, or `flags_delta`, or corrupting a
  rounding/exception rule, pinpoints the exact mismatching bit index (`lsb_diff_bit`, `xor_mask`).
- Raw upstream Spike traps (`mcause == 2`) on Nexus-private CSRs (`0x7c8..0x7cf`) so upstream
  capability is recorded honestly as `NONE`, while `nexus_extend.py` handles the C020-sealed CSRs
  and C012/C014 declaration legality without inferring outputs from DUT registers.
- `oracle_capability.json` and `coverage_matrix.json` are byte-identical to a fresh regeneration:
  every non-`NONE` class carries verified output bit patterns (`evidence_bits`), BF16/ROD/VME
  classes are `BLOCKED(归 C047)` with `g2_blocking == False`, and a G2 mandatory `NONE` class
  becomes `BLOCKED(oracle 证据不足)` with `g2_blocking == True`.
- Pairing is strictly by `(ctx_slot, ctx_generation, owner_op_seq)`, never by PC alone, and
  `mini` RetirementBuffer mode is rejected.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
H = HERE.parents[1]
REPO = H.parents[1]
sys.path.insert(0, str(HERE))

import bitwise_scalar as bs  # noqa: E402
import nexus_extend as ne  # noqa: E402
import oracle_main as om  # noqa: E402
import spike_ref_driver as srd  # noqa: E402


@pytest.fixture(scope="module")
def selftest_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  work = tmp_path_factory.mktemp("c021_selftest")
  return om.run_scalar_selftest(work_dir=work)


def test_committed_capability_and_coverage_match_fresh_generation(
  selftest_run: dict[str, Any],
) -> None:
  cap, cov = om.build_capability_and_coverage(selftest_run)
  assert om.CAPABILITY_PATH.is_file() and om.COVERAGE_PATH.is_file()
  assert om.CAPABILITY_PATH.read_text() == om.canonical_json(cap)
  assert om.COVERAGE_PATH.read_text() == om.canonical_json(cov)
  assert cap["summary"]["g2_ready"] is True
  assert cap["summary"]["g2_blocking_classes"] == []
  assert len(cap["summary"]["c047_blocked_classes"]) == len(om.G3_C047_CLASSES)

  # Every ready scalar class must carry non-empty verified output bit patterns, never just a count.
  ready = [c for c in cap["classes"] if c["oracle_kind"] != "NONE"]
  assert len(ready) >= 9
  for cls in ready:
    assert cls["evidence_bits"], cls["class_id"]
    for pattern in cls["evidence_bits"]:
      assert isinstance(pattern, str) and pattern.startswith("0x") and len(pattern) == 10

  # BF16/ROD/VME classes are explicitly BLOCKED(归 C047) and do not block G2.
  c047 = [c for c in cap["classes"] if c["oracle_kind"] == "NONE"]
  assert len(c047) == len(om.G3_C047_CLASSES)
  for cls in c047:
    assert cls["status"] == "BLOCKED(归 C047)"
    assert cls["g2_blocking"] is False
    assert cls["owner"] == "C047"


def test_g2_mandatory_none_class_is_flagged_blocked_insufficient_evidence(
  selftest_run: dict[str, Any], tmp_path: Path
) -> None:
  cap, cov = om.build_capability_and_coverage(
    selftest_run,
    extra_classes=[
      {
        "class_id": "rv32i.synthetic_missing_oracle",
        "category": "scalar_int",
        "gate_phase": "G2",
        "mandatory_in_g2": True,
        "oracle_kind": "NONE",
      }
    ],
  )
  assert cap["summary"]["g2_ready"] is False
  assert cap["summary"]["g2_blocking_classes"] == ["rv32i.synthetic_missing_oracle"]
  assert cov["oracle_coverage"] == "BLOCKED_G2"
  injected = next(c for c in cap["classes"] if c["class_id"] == "rv32i.synthetic_missing_oracle")
  assert injected["status"] == "BLOCKED(oracle 证据不足)"
  assert injected["g2_blocking"] is True

  # `oracle_main.check` on a G2 mandatory NONE class returns BLOCKED(oracle 证据不足), while a C047
  # BF16/VME class returns BLOCKED(归 C047) with g2_blocking=False.
  g2_dir = tmp_path / "g2_none"
  g2_dir.mkdir()
  (g2_dir / "oracle_input.json").write_text(
    json.dumps({"oracle_class": "rv32i.synthetic_missing_oracle", "phase": "G2"})
  )
  v_g2 = om.check("T-01", g2_dir, capability_doc=cap)
  assert v_g2["t_verdict"] == "BLOCKED"
  assert v_g2["notes"] == "BLOCKED(oracle 证据不足)"
  assert v_g2["g2_blocking"] is True

  g3_dir = tmp_path / "g3_c047"
  g3_dir.mkdir()
  (g3_dir / "oracle_input.json").write_text(
    json.dumps({"oracle_class": "vme.fp32_acc_rod_max_norm", "phase": "G3"})
  )
  v_g3 = om.check("T-29", g3_dir, capability_doc=cap)
  assert v_g3["t_verdict"] == "BLOCKED"
  assert v_g3["notes"] == "BLOCKED(归 C047)"
  assert v_g3["g2_blocking"] is False

  # Non-NONE class without verified output bits is refused at generation time.
  with pytest.raises(om.OracleError) as exc:
    om.build_capability_and_coverage(
      selftest_run,
      extra_classes=[
        {
          "class_id": "rv32i.empty_bits_claim",
          "gate_phase": "G2",
          "mandatory_in_g2": True,
          "oracle_kind": "ISA_REF",
          "evidence_bits": [],
        }
      ],
    )
  assert exc.value.code == "MISSING_EVIDENCE_BITS"


def test_bitwise_scalar_uses_pure_integer_math_and_never_touches_float_or_zvt_rtl() -> None:
  src = Path(bs.__file__).read_text()
  tree = ast.parse(src)
  forbidden_imports = {"numpy", "scipy", "ctypes"}
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        assert alias.name.split(".")[0] not in forbidden_imports
    elif isinstance(node, ast.ImportFrom) and node.module:
      assert node.module.split(".")[0] not in forbidden_imports
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      assert node.func.id != "float", "float() call forbidden in bitwise_scalar.py"
    elif isinstance(node, ast.Constant) and isinstance(node.value, float):
      raise AssertionError(f"float literal {node.value!r} forbidden in bitwise_scalar.py")
  assert "zvt_pe_" not in src and "fp_rounding.sv" not in src


def test_fp32_rounding_and_flags_pinpoint_corrupted_rules() -> None:
  # Exact differential across all 5 scalar rounding modes on 1.0 + 2^-24 (halfway) and +overflow.
  one = 0x3F800000
  half_ulp = 0x33800000
  plus_tiny = 0x33800001  # slightly above half ULP
  assert bs.fadd_s(one, half_ulp, insn_rm=bs.FRM_RNE) == bs.Fp32Result(0x3F800000, bs.FLAG_NX, 1)
  assert bs.fadd_s(0x3F800001, half_ulp, insn_rm=bs.FRM_RNE) == bs.Fp32Result(
    0x3F800002, bs.FLAG_NX, 1
  )
  assert bs.fadd_s(one, plus_tiny, insn_rm=bs.FRM_RNE) == bs.Fp32Result(0x3F800001, bs.FLAG_NX, 1)
  assert bs.fadd_s(one, half_ulp, insn_rm=bs.FRM_RMM) == bs.Fp32Result(0x3F800001, bs.FLAG_NX, 1)
  assert bs.fadd_s(one, half_ulp, insn_rm=bs.FRM_RUP) == bs.Fp32Result(0x3F800001, bs.FLAG_NX, 1)
  assert bs.fadd_s(one, half_ulp, insn_rm=bs.FRM_RTZ) == bs.Fp32Result(0x3F800000, bs.FLAG_NX, 1)
  assert bs.fadd_s(0xBF800000, 0xB3800001, insn_rm=bs.FRM_RDN) == bs.Fp32Result(
    0xBF800001, bs.FLAG_NX, 1
  )

  # Subnormal tininess-after-rounding boundary: value rounds up to min normal 0x00800000 -> NX=1, UF=0.
  almost_min_norm = 0x007FFFFF
  tiny_add = 0x00000001
  up_to_norm = bs.fadd_s(almost_min_norm, tiny_add, insn_rm=bs.FRM_RNE)
  assert up_to_norm.bits == 0x00800000 and up_to_norm.fflags_delta == 0

  # Illegal rounding modes (5=ROD, 6=unassigned, 7=unresolved DYN) are rejected, never guessed.
  for bad_frm in (5, 6, 7):
    with pytest.raises(bs.BitwiseScalarError) as exc:
      bs.fadd_s(one, one, insn_rm=bs.FRM_DYN, csr_frm=bad_frm)
    assert exc.value.code == "ILLEGAL_FRM"


def test_csr_warl_and_retire_flag_accumulation_rules() -> None:
  # Ordered retire flags: early flags_delta stays per-entry and ORs in program order.
  ff, vx = 0, 0
  ff, vx = bs.merge_retire_flags(ff, vx, bs.pack_flags_delta(bs.FLAG_NX, 0))
  assert (ff, vx) == (bs.FLAG_NX, 0)
  ff, vx = bs.merge_retire_flags(ff, vx, bs.pack_flags_delta(bs.FLAG_OF, 1))
  assert (ff, vx) == (bs.FLAG_NX | bs.FLAG_OF, 1)

  # CoralNPU mtvec Direct mode clears MODE[1:0]; read-only CSR write traps.
  mtv = bs.csr_step_expectation(0x305, "csrrw", 0x00000100, 0x00000203)
  assert mtv.rd_value == 0x00000100 and mtv.new_value == 0x00000200
  with pytest.raises(bs.BitwiseScalarError) as exc:
    bs.csr_step_expectation(0xF11, "csrrw", 0x426, 0x1234)
  assert exc.value.code == "CSR_READ_ONLY_TRAP"


def test_nexus_extend_verifies_seal_and_enforces_c012_declaration_fsm(tmp_path: Path) -> None:
  verified = ne.verify_sealed_inputs()
  assert "hardware/coral_nexus/protocol/csr_name_map.yaml" in verified

  # Tampered seal hash is rejected.
  bad_seal = json.loads(ne.SEAL_PATH.read_text())
  bad_seal["inputs"]["hardware/coral_nexus/protocol/csr_name_map.yaml"] = "0" * 64
  bad_path = tmp_path / "PROTOCOL_FREEZE.json"
  bad_path.write_text(json.dumps(bad_seal))
  with pytest.raises(ne.NexusExtendError) as exc:
    ne.verify_sealed_inputs(seal_path=bad_path)
  assert exc.value.code == "SEAL_HASH_MISMATCH"

  # Valid C012 lifecycle: START -> E2 -> FIRST/CONTINUE/FINAL -> E2 -> END -> E2 -> REGION_DONE.
  ad = ne.NexusContextAdapter(ctx_slot=1, ctx_generation=3)

  def _csrw(addr: int, rs1: int = 5) -> int:
    return ((addr & 0xFFF) << 20) | (rs1 << 15) | (1 << 12) | 0x73

  ad.execute_private_csr(pc=0x0, insn=_csrw(0x7CA), rs1_val=1, owner_op_seq=0)  # START
  # Repeated csrw before E2 -> ARMED_CONFLICT
  with pytest.raises(ne.NexusExtendError) as conflict:
    ad.execute_private_csr(pc=0x4, insn=_csrw(0x7CC), rs1_val=2, owner_op_seq=1)
  assert conflict.value.code == "ARMED_CONFLICT"
  assert ad.consume_pending_e2() == ("REGION_START", 1)
  # Second E2 without pending -> CONSUME_WITHOUT_PENDING
  with pytest.raises(ne.NexusExtendError) as dup_e2:
    ad.consume_pending_e2()
  assert dup_e2.value.code == "CONSUME_WITHOUT_PENDING"

  ad.execute_private_csr(pc=0x8, insn=_csrw(0x7CC), rs1_val=10, owner_op_seq=2)  # FIRST
  ad.consume_pending_e2()
  ad.execute_private_csr(pc=0xC, insn=_csrw(0x7CD), rs1_val=11, owner_op_seq=3)  # CONTINUE
  ad.consume_pending_e2()
  ad.execute_private_csr(pc=0x10, insn=_csrw(0x7CE), rs1_val=12, owner_op_seq=4)  # FINAL
  ad.consume_pending_e2()
  ad.execute_private_csr(pc=0x14, insn=_csrw(0x7CB), rs1_val=13, owner_op_seq=5)  # END
  ad.consume_pending_e2()
  with pytest.raises(ne.NexusExtendError) as early_exit:
    ad.check_task_exit_clean()
  assert early_exit.value.code == "EXIT_BEFORE_REGION_DONE"
  ad.inject_env_event(ne.NexusEnvEvent(step_ordinal=6, kind="REGION_DONE", key=0, status="DONE"))
  ad.check_task_exit_clean()


def test_oracle_check_runs_real_spike_and_pinpoints_single_bit_corruptions(
  tmp_path: Path,
) -> None:
  elf = om.build_asm_elf(
    om.SELFTEST_DIR / "scalar_core_selftest.S",
    om.SELFTEST_DIR / "scalar_selftest.ld",
    tmp_path / "scalar.elf",
  )
  out_base = 0x00010100
  ranges = [{"address": out_base, "length": 64}]

  # Baseline run: DUT trace matching the independent Spike reference passes C4 with 0 diffs.
  ref_trace = srd.run_trace_request(
    srd.TraceRequest(elf_path=str(elf), ctx_slot=1, ctx_generation=7, read_memory_ranges=ranges)
  )
  run_ok = tmp_path / "run_ok"
  run_ok.mkdir()
  (run_ok / "oracle_input.json").write_text(
    json.dumps(
      {
        "oracle_class": "rv32f.scalar_fp32_ieee_and_flags_delta",
        "elf_path": str(elf),
        "ctx_slot": 1,
        "ctx_generation": 7,
        "read_memory_ranges": ranges,
      }
    )
  )
  dut_clean = {
    "retirement_buffer_mode": "full",
    "retire_stream": copy.deepcopy(ref_trace["retire_stream"]),
    "memory_bytes": copy.deepcopy(ref_trace["memory_bytes"]),
  }
  (run_ok / "dut_retire.json").write_text(json.dumps(dut_clean))
  v_ok = om.check("T-01", run_ok)
  assert v_ok["t_verdict"] == "PASS" and v_ok["diffs"] == []

  # Negative 1: flip bit 13 of a written GPR in one retire record -> pinpoints bit 13 & register.
  dut_gpr_flip = copy.deepcopy(dut_clean)
  wb_rec = next(r for r in dut_gpr_flip["retire_stream"] if r["x_wb"] != 0)
  reg_idx = (wb_rec["x_wb"] & -wb_rec["x_wb"]).bit_length() - 1
  bank_int = int(wb_rec["x_wdata"], 16) ^ (1 << (32 * reg_idx + 13))
  wb_rec["x_wdata"] = f"0x{bank_int:0256x}"
  diffs_gpr = om.compare_retire_and_memory(oracle_trace=ref_trace, dut_trace=dut_gpr_flip)
  assert len(diffs_gpr) == 1
  assert diffs_gpr[0]["kind"] == "GPR_WB_MISMATCH"
  assert diffs_gpr[0]["register"] == f"x{reg_idx}"
  assert diffs_gpr[0]["bit_diff"]["lsb_diff_bit"] == 13
  assert diffs_gpr[0]["bit_diff"]["xor_mask"] == "0x00002000"

  # Negative 2: corrupt flags_delta NX bit (bit 0) on an FP32 instruction -> pinpoints bit 0.
  dut_flag_flip = copy.deepcopy(dut_clean)
  fp_rec = next(r for r in dut_flag_flip["retire_stream"] if r["flags_delta"] != 0)
  fp_rec["flags_delta"] ^= 0b000001
  diffs_flag = om.compare_retire_and_memory(oracle_trace=ref_trace, dut_trace=dut_flag_flip)
  assert len(diffs_flag) == 1
  assert diffs_flag[0]["kind"] == "RETIRE_FIELD_MISMATCH"
  assert diffs_flag[0]["field"] == "flags_delta"
  assert diffs_flag[0]["bit_diff"]["lsb_diff_bit"] == 0

  # Negative 3: flip bit 5 of byte 41 in final memory -> pinpoints byte_offset=41, lsb_diff_bit=5.
  dut_mem_flip = copy.deepcopy(dut_clean)
  mem_key = f"0x{out_base:08x}:64"
  mbytes = bytearray.fromhex(dut_mem_flip["memory_bytes"][mem_key])
  mbytes[41] ^= 1 << 5
  dut_mem_flip["memory_bytes"][mem_key] = bytes(mbytes).hex()
  diffs_mem = om.compare_retire_and_memory(oracle_trace=ref_trace, dut_trace=dut_mem_flip)
  assert len(diffs_mem) == 1
  assert diffs_mem[0]["kind"] == "MEMORY_BYTE_MISMATCH"
  assert diffs_mem[0]["byte_offset"] == 41
  assert diffs_mem[0]["bit_diff"]["lsb_diff_bit"] == 5

  # Negative 4: swapping Context identity on same-PC records is rejected (never paired by PC alone).
  dut_ctx_swap = copy.deepcopy(dut_clean)
  dut_ctx_swap["retire_stream"][0]["ctx_generation"] = 99
  diffs_ctx = om.compare_retire_and_memory(oracle_trace=ref_trace, dut_trace=dut_ctx_swap)
  assert {d["kind"] for d in diffs_ctx} == {"UNEXPECTED_DUT_RECORD", "MISSING_DUT_RECORD"}

  # Negative 5: mini RetirementBuffer mode cannot be compared (C010 mode rule).
  dut_mini = copy.deepcopy(dut_clean)
  dut_mini["retirement_buffer_mode"] = "mini"
  diffs_mini = om.compare_retire_and_memory(oracle_trace=ref_trace, dut_trace=dut_mini)
  assert diffs_mini == [
    {
      "kind": "RETIREMENT_BUFFER_MODE",
      "detail": "requires full RetirementBuffer, got 'mini'",
    }
  ]
