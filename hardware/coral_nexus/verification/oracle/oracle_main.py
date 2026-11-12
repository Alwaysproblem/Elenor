"""Main scalar/CSR oracle coordinator, capability generator, and `check()` entry (C021 / W7-02).

Responsibilities:
1. Build and execute the hand-computable scalar selftest ELF (`selftest/scalar_core_selftest.S`)
   and Nexus-private CSR probe ELF (`selftest/nexus_private_probe.S`) against the pinned Spike
   library (`spike_ref_driver.py`) and cross-check FP32/CSR bit patterns against `bitwise_scalar.py`.
2. Emit deterministic, hash-stable `oracle_capability.json` and `coverage_matrix.json`:
   - Every covered scalar/FP32/Zbb/CSR/trap/helper class carries actual executed output bit patterns
     (`evidence_bits`), never just a vector count or file-non-empty check.
   - Raw upstream Spike capability for Nexus-private CSRs is recorded honestly as `NONE`, with the
     explicit `nexus_extend.py` adapter status recorded alongside.
   - BF16/ROD/VME matrix classes are recorded as `oracle_kind: "NONE"` with
     `status: "BLOCKED(归 C047)"`, `g2_blocking: false`.
   - Any G2 mandatory class whose `oracle_kind == "NONE"` (and is not owned by C047/G3+) produces
     `BLOCKED(oracle 证据不足)` with `g2_blocking: true`.
3. Expose `check(t_id, run_dir, ...)` returning a `cn_verdict/1`-shaped verdict for C4 / C041:
   - Compares retire streams by `(ctx_slot, ctx_generation, owner_op_seq)` (`oracle_requirements.yaml`),
     never by PC alone.
   - Checks bitwise equality of register writebacks, `flags_delta`, traps, and final memory bytes,
     pinpointing exact bit diffs (`xor_mask`, `first_diff_bit`) on any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import bitwise_scalar as bs
import nexus_extend as ne
import spike_ref_driver as srd
import yaml

HERE = Path(__file__).resolve().parent
H = HERE.parents[1]
REPO = H.parents[1]

SELFTEST_DIR = HERE / "selftest"
VECTORS_PATH = SELFTEST_DIR / "scalar_vectors.json"
DEVIATIONS_PATH = HERE / "known_oracle_deviation.md"
CAPABILITY_PATH = HERE / "oracle_capability.json"
COVERAGE_PATH = HERE / "coverage_matrix.json"
TMATRIX_PATH = H / "verification/matrix/t_matrix.yaml"
ORACLE_REQ_PATH = H / "rtl/scalar/doc/oracle_requirements.yaml"

TOOLCHAIN_BIN = REPO / "out/coral_nexus/cache/toolchain/extracted/toolchain_coralnpu_v2/bin"
MARCH = "rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma"

# Classes owned by C047 (CN-G3 BF16/ROD/VME bitwise oracle) -- explicitly BLOCKED(归 C047) in C021.
G3_C047_CLASSES: tuple[dict[str, str], ...] = (
  {
    "class_id": "vme.bf16_mul_rod_jam",
    "category": "bf16_vme",
    "rule_refs": "MUL-01,JAM-01,OVF-MUL-01",
    "owner": "C047",
    "known_deviation": "NONE",
    "reason": "BF16 batched multiply tree (26-bit aligned grid + jam + widening Inf) belongs to C047",
  },
  {
    "class_id": "vme.fp32_acc_rod_max_norm",
    "category": "bf16_vme",
    "rule_refs": "ACC-01,OVF-ACC-01",
    "owner": "C047",
    "known_deviation": "KD-01",
    "reason": (
      "Accumulate-stage ROD overflow saturates to max-normal (0x7f7fffff); raw Spike yields Inf (KD-01)"
    ),
  },
  {
    "class_id": "vme.lane_mask_before_special",
    "category": "bf16_vme",
    "rule_refs": "MSK-01,SPC-01",
    "owner": "C047",
    "known_deviation": "KD-02",
    "reason": (
      "Raw Spike vtfmm lacks lane-mask input; inactive lanes must mask before NaN/Inf classification"
    ),
  },
  {
    "class_id": "vme.tk_grouping_and_cancellation",
    "category": "bf16_vme",
    "rule_refs": "TKG-01,CAN-01,SUB-01,FLG-01",
    "owner": "C047",
    "known_deviation": "NONE",
    "reason": (
      "TK1/TK2 grouping, tree +0 cancellation, subnormal grid, and OF/NV-only exposure belong to C047"
    ),
  },
  {
    "class_id": "rvv.bf16_narrow_rne_and_widen_fma",
    "category": "bf16_rvv",
    "rule_refs": "NAR-01,FRM-01",
    "owner": "C047",
    "known_deviation": "NONE",
    "reason": "vfncvtbf16.f.f.w RNE narrowing and vfwmaccbf16 vector bitwise oracle belong to C047",
  },
  {
    "class_id": "vme.matrix_tile_state_ops",
    "category": "bf16_vme",
    "rule_refs": "vtle,vtse,vtzero,vtmv,msetmtype,msettk",
    "owner": "C047",
    "known_deviation": "NONE",
    "reason": "Matrix tile layout / state instructions belong to G3 C047",
  },
)


class OracleError(Exception):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


def canonical_json(doc: Any) -> str:
  return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def first_diff_bit(actual: int, expected: int, width: int = 32) -> dict[str, Any]:
  mask = (1 << width) - 1
  xor = (actual ^ expected) & mask
  if xor == 0:
    return {"xor_mask": "0x0", "diff_bits": [], "lsb_diff_bit": None}
  diff_bits = [b for b in range(width) if (xor >> b) & 1]
  return {
    "xor_mask": f"0x{xor:0{(width + 3) // 4}x}",
    "diff_bits": diff_bits,
    "lsb_diff_bit": diff_bits[0],
  }


def build_asm_elf(asm_path: Path, ld_path: Path, out_elf: Path) -> Path:
  """Compile and link a freestanding RV32 selftest `.S` with the pinned C004 toolchain."""
  if not TOOLCHAIN_BIN.is_dir():
    raise OracleError("MISSING_TOOLCHAIN", str(TOOLCHAIN_BIN))
  out_elf.parent.mkdir(parents=True, exist_ok=True)
  obj_path = out_elf.with_suffix(".o")
  cc_cmd = [
    str(TOOLCHAIN_BIN / "clang"),
    "--target=riscv32-unknown-elf",
    f"-march={MARCH}",
    "-mabi=ilp32",
    "-mcmodel=medany",
    "-ffreestanding",
    "-fno-builtin",
    "-c",
    str(asm_path),
    "-o",
    str(obj_path),
  ]
  proc = subprocess.run(cc_cmd, capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    raise OracleError("ASM_COMPILE_FAILED", proc.stderr.strip())
  ld_cmd = [
    str(TOOLCHAIN_BIN / "riscv64-unknown-elf-ld.bfd"),
    "-m",
    "elf32lriscv",
    "-T",
    str(ld_path),
    "-o",
    str(out_elf),
    str(obj_path),
  ]
  proc2 = subprocess.run(ld_cmd, capture_output=True, text=True, check=False)
  if proc2.returncode != 0:
    raise OracleError("ELF_LINK_FAILED", proc2.stderr.strip())
  return out_elf


def _eval_bitwise_vector(vec: dict[str, Any]) -> bs.Fp32Result:
  op = str(vec["op"])
  rm = int(vec["rm"])
  a = int(str(vec["a"]), 16)
  if op == "fadd_s":
    return bs.fadd_s(a, int(str(vec["b"]), 16), insn_rm=rm, csr_frm=rm)
  if op == "fsub_s":
    return bs.fsub_s(a, int(str(vec["b"]), 16), insn_rm=rm, csr_frm=rm)
  if op == "fmul_s":
    return bs.fmul_s(a, int(str(vec["b"]), 16), insn_rm=rm, csr_frm=rm)
  if op == "fdiv_s":
    return bs.fdiv_s(a, int(str(vec["b"]), 16), insn_rm=rm, csr_frm=rm)
  if op == "fmadd_s":
    return bs.fmadd_s(a, int(str(vec["b"]), 16), int(str(vec["c"]), 16), insn_rm=rm, csr_frm=rm)
  if op == "fcvt_w_s":
    return bs.fcvt_w_s(a, signed_out=True, insn_rm=rm, csr_frm=rm)
  raise OracleError("UNKNOWN_VECTOR_OP", op)


def run_scalar_selftest(work_dir: Path | None = None) -> dict[str, Any]:
  """Build & execute selftest ELFs, verify every bit pattern, and return structured evidence."""
  sealed_hashes = ne.verify_sealed_inputs()
  vectors_doc = json.loads(VECTORS_PATH.read_text())
  stage = work_dir or (REPO / "out/coral_nexus/C021/selftest_build")
  stage.mkdir(parents=True, exist_ok=True)

  ld_path = SELFTEST_DIR / "scalar_selftest.ld"
  scalar_elf = build_asm_elf(
    SELFTEST_DIR / "scalar_core_selftest.S", ld_path, stage / "scalar_core_selftest.elf"
  )
  nexus_elf = build_asm_elf(
    SELFTEST_DIR / "nexus_private_probe.S", ld_path, stage / "nexus_private_probe.elf"
  )

  out_base = int(str(vectors_doc["scalar_elf"]["output_base"]), 16)
  word_specs = list(vectors_doc["scalar_elf"]["output_words_u32"])
  byte_len = 4 * len(word_specs)

  scalar_trace = srd.run_trace_request(
    srd.TraceRequest(
      elf_path=str(scalar_elf),
      ctx_slot=0,
      ctx_generation=1,
      read_memory_ranges=[{"address": out_base, "length": byte_len}],
    )
  )
  if not scalar_trace["halted"]:
    raise OracleError("SELFTEST_NOT_HALTED", f"steps={scalar_trace['steps_executed']}")

  raw_hex = scalar_trace["memory_bytes"][f"0x{out_base:08x}:{byte_len}"]
  raw_bytes = bytes.fromhex(raw_hex)
  actual_words = [
    int.from_bytes(raw_bytes[4 * i : 4 * (i + 1)], "little") for i in range(len(word_specs))
  ]

  mismatches: list[dict[str, Any]] = []
  elf_word_evidence: dict[str, dict[str, Any]] = {}
  for spec, actual_u32 in zip(word_specs, actual_words):
    expected_u32 = int(str(spec["expected_u32"]), 16)
    diff = first_diff_bit(actual_u32, expected_u32, 32)
    entry = {
      "index": int(spec["index"]),
      "insn_family": str(spec["insn_family"]),
      "expected_u32": f"0x{expected_u32:08x}",
      "actual_u32": f"0x{actual_u32:08x}",
      "match": actual_u32 == expected_u32,
      "bit_diff": diff,
    }
    elf_word_evidence[str(spec["class_id"])] = entry
    if actual_u32 != expected_u32:
      mismatches.append({"class_id": spec["class_id"], **entry})

  # Verify per-instruction retire records carry C010 required fields and isolate flags_delta.
  req_fields = yaml.safe_load(ORACLE_REQ_PATH.read_text())["retire_fields_compared"]["fields"]
  for rec in scalar_trace["retire_stream"]:
    missing = [f for f in req_fields if f not in rec]
    if missing:
      raise OracleError("RETIRE_SCHEMA_MISSING", f"missing={missing}")

  nonzero_flag_deltas = [
    {"pc": f"0x{r['pc_rdata']:08x}", "insn": f"0x{r['insn']:08x}", "flags_delta": r["flags_delta"]}
    for r in scalar_trace["retire_stream"]
    if r["flags_delta"] != 0
  ]
  trapped_records = [
    {"pc": f"0x{r['pc_rdata']:08x}", "insn": f"0x{r['insn']:08x}", "mcause": r["mcause"]}
    for r in scalar_trace["retire_stream"]
    if r["trap"] == 1
  ]
  if not nonzero_flag_deltas or not trapped_records:
    raise OracleError("SELFTEST_TRACE_INCOMPLETE", "expected non-zero flags_delta and trap records")

  # Evaluate independent bitwise FP32 vectors.
  fp32_evidence: list[dict[str, Any]] = []
  for vec in vectors_doc["fp32_bitwise_vectors"]:
    res = _eval_bitwise_vector(vec)
    exp_bits = int(str(vec["expected_bits"]), 16)
    exp_flags = int(vec["expected_fflags"])
    bdiff = first_diff_bit(res.bits, exp_bits, 32)
    fdiff = first_diff_bit(res.fflags_delta, exp_flags, 5)
    ok = (res.bits == exp_bits) and (res.fflags_delta == exp_flags)
    rec = {
      "id": str(vec["id"]),
      "op": str(vec["op"]),
      "rm": int(vec["rm"]),
      "expected_bits": f"0x{exp_bits:08x}",
      "actual_bits": f"0x{res.bits:08x}",
      "expected_fflags": f"0x{exp_flags:02x}",
      "actual_fflags": f"0x{res.fflags_delta:02x}",
      "flags_delta_6b": f"0x{res.flags_delta:02x}",
      "match": ok,
      "bits_diff": bdiff,
      "fflags_diff": fdiff,
    }
    fp32_evidence.append(rec)
    if not ok:
      mismatches.append(rec)

  # Probe Nexus-private CSRs BOTH ways: raw upstream Spike MUST trap (oracle_kind NONE on upstream),
  # and nexus_extend MUST execute cleanly and match expected CSR read/write values.
  raw_nexus = srd.run_trace_request(
    srd.TraceRequest(
      elf_path=str(nexus_elf),
      enable_nexus_extend=False,
      read_memory_ranges=[{"address": out_base, "length": 16}],
    )
  )
  raw_nexus_mem = bytes.fromhex(raw_nexus["memory_bytes"][f"0x{out_base:08x}:16"])
  raw_mcause = int.from_bytes(raw_nexus_mem[12:16], "little")
  if raw_mcause != 2:
    raise OracleError("NEXUS_RAW_PROBE_UNEXPECTED", f"expected mcause=2, got {raw_mcause}")

  ext_nexus = srd.run_trace_request(
    srd.TraceRequest(
      elf_path=str(nexus_elf),
      enable_nexus_extend=True,
      read_memory_ranges=[{"address": out_base, "length": 16}],
    )
  )
  ext_mem = bytes.fromhex(ext_nexus["memory_bytes"][f"0x{out_base:08x}:16"])
  ext_words = [int.from_bytes(ext_mem[4 * i : 4 * (i + 1)], "little") for i in range(4)]
  expected_ext = [0x00000000, 0x13572468, 0x13572479, 0x00000000]
  for idx, (act, exp) in enumerate(zip(ext_words, expected_ext)):
    if act != exp:
      mismatches.append(
        {
          "class_id": f"nexus_extend.word_{idx}",
          "actual": f"0x{act:08x}",
          "expected": f"0x{exp:08x}",
          "bit_diff": first_diff_bit(act, exp, 32),
        }
      )

  if mismatches:
    raise OracleError("SELFTEST_BIT_MISMATCH", json.dumps(mismatches[0], sort_keys=True))

  return {
    "sealed_hashes": sealed_hashes,
    "spike_seed_sha256": scalar_trace["build_meta"]["seed_sha256"],
    "riscv_isa_sim_commit": scalar_trace["build_meta"]["riscv_isa_sim_commit"],
    "elf_word_evidence": elf_word_evidence,
    "fp32_evidence": fp32_evidence,
    "nonzero_flag_deltas": nonzero_flag_deltas,
    "trapped_records": trapped_records,
    "nexus_probe": {
      "raw_upstream_mcause": raw_mcause,
      "extended_words_hex": [f"0x{w:08x}" for w in ext_words],
    },
    "retire_count": len(scalar_trace["retire_stream"]),
  }


def build_capability_and_coverage(
  selftest_evidence: dict[str, Any],
  *,
  extra_classes: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Construct deterministic `oracle_capability.json` and `coverage_matrix.json` from real runs."""
  words = selftest_evidence["elf_word_evidence"]
  fp_by_id = {e["id"]: e for e in selftest_evidence["fp32_evidence"]}

  csr_exp_mtvec = bs.csr_step_expectation(0x305, "csrrw", 0, 0x00000103)
  csr_exp_mvendor = bs.csr_step_expectation(0xF11, "csrrs", 0x426, 0, rs1_is_x0=True)

  scalar_entries: list[dict[str, Any]] = [
    {
      "class_id": "rv32i.alu_and_shifts",
      "category": "scalar_int",
      "gate_phase": "G2",
      "mandatory_in_g2": True,
      "oracle_kind": "ISA_REF",
      "status": "READY",
      "g2_blocking": False,
      "evidence_bits": [
        words["rv32i.alu_wrap"]["actual_u32"],
        words["rv32i.compare_shift"]["actual_u32"],
      ],
    },
    {
      "class_id": "rv32m.mul_div_rem",
      "category": "scalar_int",
      "gate_phase": "G2",
      "mandatory_in_g2": True,
      "oracle_kind": "ISA_REF",
      "status": "READY",
      "g2_blocking": False,
      "evidence_bits": [
        words["rv32m.div_overflow_zero"]["actual_u32"],
        words["rv32m.mulhsu_remu"]["actual_u32"],
      ],
    },
    {
      "class_id": "zbb.scalar_bitmanip",
      "category": "scalar_int",
      "gate_phase": "G2",
      "mandatory_in_g2": True,
      "oracle_kind": "ISA_REF",
      "status": "READY",
      "g2_blocking": False,
      "evidence_bits": [words["zbb.bitmanip"]["actual_u32"]],
    },
    {
      "class_id": "rv32i.branch_jump",
      "category": "scalar_control",
      "gate_phase": "G2",
      "mandatory_in_g2": True,
      "oracle_kind": "ISA_REF",
      "status": "READY",
      "g2_blocking": False,
      "evidence_bits": [words["rv32i.branch_jal_jalr"]["actual_u32"]],
    },
    {
      "class_id": "rv32i.l1_load_store_ext",
      "category": "scalar_memory",
      "gate_phase": "G2",
      "mandatory_in_g2": True,
      "oracle_kind": "ISA_REF",
      "status": "READY",
      "g2_blocking": False,
      "evidence_bits": [words["rv32i.load_store_ext"]["actual_u32"]],
    },
    {
      "class_id": "rv32i.illegal_insn_trap_mret",
      "category": "scalar_trap",
      "gate_phase": "G2",
      "mandatory_in_g2": True,
      "oracle_kind": "ISA_REF",
      "status": "READY",
      "g2_blocking": False,
      "evidence_bits": [
        words["trap.illegal_insn_mcause"]["actual_u32"],
        words["trap.mepc_recovery"]["actual_u32"],
      ],
    },
    {
      "class_id": "rv32f.scalar_fp32_ieee_and_flags_delta",
      "category": "scalar_fp32",
      "gate_phase": "G2",
      "mandatory_in_g2": True,
      "oracle_kind": "BOTH",
      "status": "READY",
      "g2_blocking": False,
      "evidence_bits": [
        words["rv32f.fadd_rne_bits"]["actual_u32"],
        words["rv32f.fadd_rne_fflags"]["actual_u32"],
        words["rv32f.fmul_rtz_overflow_bits"]["actual_u32"],
        words["rv32f.fmul_rtz_overflow_fflags"]["actual_u32"],
        words["rv32f.fdiv_dz_nv"]["actual_u32"],
        words["rv32f.fcvt_w_s_sat"]["actual_u32"],
        fp_by_id["fp32_rup_overflow_pos_inf"]["actual_bits"],
        fp_by_id["fp32_subnormal_underflow_nx_uf"]["actual_bits"],
        fp_by_id["fp32_fmadd_single_rounding"]["actual_bits"],
      ],
    },
    {
      "class_id": "zicsr.arch_warl_and_retire_flags",
      "category": "scalar_csr",
      "gate_phase": "G2",
      "mandatory_in_g2": True,
      "oracle_kind": "BOTH",
      "status": "READY",
      "g2_blocking": False,
      "evidence_bits": [
        words["zicsr.mtvec_mvendorid"]["actual_u32"],
        f"0x{csr_exp_mtvec.new_value:08x}",
        f"0x{csr_exp_mvendor.rd_value:08x}",
      ],
    },
    {
      "class_id": "nexus.private_csrs_and_helpers",
      "category": "nexus_private",
      "gate_phase": "G2",
      "mandatory_in_g2": True,
      "upstream_spike_oracle_kind": "NONE",
      "oracle_kind": "BIT_ORACLE",
      "adapter": "nexus_extend.py",
      "status": "READY_VIA_NEXUS_EXTEND",
      "g2_blocking": False,
      "evidence_bits": list(selftest_evidence["nexus_probe"]["extended_words_hex"]),
      "upstream_trap_mcause": selftest_evidence["nexus_probe"]["raw_upstream_mcause"],
    },
  ]

  for g3 in G3_C047_CLASSES:
    scalar_entries.append(
      {
        "class_id": g3["class_id"],
        "category": g3["category"],
        "gate_phase": "G3",
        "mandatory_in_g2": False,
        "oracle_kind": "NONE",
        "owner": g3["owner"],
        "known_deviation": g3["known_deviation"],
        "status": "BLOCKED(归 C047)",
        "g2_blocking": False,
        "reason": g3["reason"],
        "evidence_bits": [],
      }
    )

  for raw_extra in extra_classes or []:
    entry = dict(raw_extra)
    kind = str(entry.get("oracle_kind", "NONE"))
    owner = str(entry.get("owner", ""))
    mandatory_g2 = bool(entry.get("mandatory_in_g2", entry.get("gate_phase") == "G2"))
    if kind == "NONE":
      if owner == "C047" or entry.get("gate_phase") == "G3":
        entry["status"] = "BLOCKED(归 C047)"
        entry["g2_blocking"] = False
      elif mandatory_g2:
        entry["status"] = "BLOCKED(oracle 证据不足)"
        entry["g2_blocking"] = True
      else:
        entry["status"] = "BLOCKED(oracle 证据不足)"
        entry["g2_blocking"] = False
    else:
      bits = entry.get("evidence_bits") or []
      if not bits:
        raise OracleError(
          "MISSING_EVIDENCE_BITS",
          f"{entry.get('class_id')}: non-NONE oracle class must carry verified output bits",
        )
      entry.setdefault("status", "READY")
      entry["g2_blocking"] = False
    scalar_entries.append(entry)

  # Enforce C021 gate rule: every non-NONE class MUST have actual verified output bits.
  for item in scalar_entries:
    if item["oracle_kind"] != "NONE" and not item.get("evidence_bits"):
      raise OracleError("EMPTY_EVIDENCE_BITS", str(item["class_id"]))

  g2_blocked = [e["class_id"] for e in scalar_entries if e.get("g2_blocking") is True]
  c047_deferred = [e["class_id"] for e in scalar_entries if e.get("status") == "BLOCKED(归 C047)"]

  # Cross-reference T-matrix rows that require BIT_ORACLE / G2 scalar coverage.
  tmatrix = yaml.safe_load(TMATRIX_PATH.read_text())
  t_row_mapping: list[dict[str, Any]] = []
  for row in tmatrix["rows"]:
    tid = str(row["t_id"])
    gate = (row.get("scenario_source") or {}).get("gate")
    okind = (row.get("oracle") or {}).get("kind")
    if gate == "G2" or okind in ("bit_oracle", "scenario_oracle"):
      if tid in ("T-28", "T-29", "T-39", "T-61"):
        row_state = "BLOCKED(归 C047)"
        g2_blk = False
      elif tid in ("T-16", "T-50", "T-65"):
        row_state = "DEFERRED_TO_W4_04_ORACLE" if gate != "G2" else "SCALAR_ADDR_READY_INDEXED_C055"
        g2_blk = False
      elif g2_blocked and gate == "G2":
        row_state = "BLOCKED(oracle 证据不足)"
        g2_blk = True
      else:
        row_state = "SCALAR_ORACLE_READY"
        g2_blk = False
      t_row_mapping.append(
        {
          "t_id": tid,
          "gate": gate,
          "matrix_oracle_kind": okind,
          "c021_oracle_state": row_state,
          "g2_blocking": g2_blk,
        }
      )

  capability_doc = {
    "schema": "cn_oracle_capability/v1",
    "card": "C021",
    "scope": "scalar_isa_and_csr_core",
    "pinned_sources": {
      "riscv_isa_sim_commit": selftest_evidence["riscv_isa_sim_commit"],
      "spike_seed_sha256": selftest_evidence["spike_seed_sha256"],
      "sealed_contracts": selftest_evidence["sealed_hashes"],
      "known_deviations_sha256": sha256_bytes(DEVIATIONS_PATH.read_bytes()),
    },
    "summary": {
      "ready_scalar_classes": sum(1 for e in scalar_entries if e["oracle_kind"] != "NONE"),
      "c047_blocked_classes": c047_deferred,
      "g2_blocking_classes": g2_blocked,
      "g2_ready": len(g2_blocked) == 0,
    },
    "classes": scalar_entries,
    "t_row_oracle_status": t_row_mapping,
  }

  coverage_doc = {
    "schema": "cn_oracle_coverage_matrix/v1",
    "card": "C021",
    "oracle_coverage": "SUFFICIENT_SCALAR_G2" if not g2_blocked else "BLOCKED_G2",
    "scalar_elf_selftest": {
      "retire_records_verified": selftest_evidence["retire_count"],
      "nonzero_flags_delta_records": selftest_evidence["nonzero_flag_deltas"],
      "trap_records": selftest_evidence["trapped_records"],
      "verified_words": selftest_evidence["elf_word_evidence"],
    },
    "fp32_bitwise_vectors": selftest_evidence["fp32_evidence"],
    "nexus_private_csr_probe": selftest_evidence["nexus_probe"],
    "g3_c047_pending_matrix": [
      {
        "class_id": g["class_id"],
        "rule_refs": g["rule_refs"],
        "status": "BLOCKED(归 C047)",
        "known_deviation": g["known_deviation"],
        "vector_count_in_c021": 0,
        "owner": "C047",
      }
      for g in G3_C047_CLASSES
    ],
  }
  return capability_doc, coverage_doc


def write_generated_oracle_artifacts(work_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
  evidence = run_scalar_selftest(work_dir=work_dir)
  cap, cov = build_capability_and_coverage(evidence)
  CAPABILITY_PATH.write_text(canonical_json(cap))
  COVERAGE_PATH.write_text(canonical_json(cov))
  return cap, cov


def compare_retire_and_memory(
  *,
  oracle_trace: dict[str, Any],
  dut_trace: dict[str, Any],
) -> list[dict[str, Any]]:
  """Bitwise comparison of DUT retire records and memory bytes against the independent oracle.

  Rules (`rtl/scalar/doc/oracle_requirements.yaml`):
  - Paired strictly by `(ctx_slot, ctx_generation, owner_op_seq)`, NEVER by PC alone.
  - Compares `pc_rdata`, `insn`, `trap`, `flags_delta`, `x_wb`, `x_wdata`, optional `f_wb`/`f_wdata`,
    and final `memory_bytes`.
  """
  diffs: list[dict[str, Any]] = []
  if dut_trace.get("retirement_buffer_mode", "full") != "full":
    diffs.append(
      {
        "kind": "RETIREMENT_BUFFER_MODE",
        "detail": f"requires full RetirementBuffer, got {dut_trace.get('retirement_buffer_mode')!r}",
      }
    )
    return diffs

  def _key(rec: dict[str, Any]) -> tuple[int, int, int]:
    return (int(rec["ctx_slot"]), int(rec["ctx_generation"]), int(rec["owner_op_seq"]))

  oracle_by_key = {_key(r): r for r in oracle_trace.get("retire_stream") or []}
  dut_records = list(dut_trace.get("retire_stream") or [])
  seen_keys: set[tuple[int, int, int]] = set()
  for drec in dut_records:
    k = _key(drec)
    if k in seen_keys:
      diffs.append({"kind": "DUPLICATE_DUT_KEY", "key": list(k)})
      continue
    seen_keys.add(k)
    orec = oracle_by_key.get(k)
    if orec is None:
      diffs.append({"kind": "UNEXPECTED_DUT_RECORD", "key": list(k), "pc": drec.get("pc_rdata")})
      continue
    for field_name, width in (
      ("valid", 1),
      ("pc_rdata", 32),
      ("insn", 32),
      ("trap", 1),
      ("flags_delta", 6),
      ("x_wb", 32),
    ):
      dv = int(drec[field_name])
      ov = int(orec[field_name])
      if dv != ov:
        diffs.append(
          {
            "kind": "RETIRE_FIELD_MISMATCH",
            "key": list(k),
            "field": field_name,
            "dut": f"0x{dv:x}",
            "oracle": f"0x{ov:x}",
            "bit_diff": first_diff_bit(dv, ov, width),
          }
        )
    dx_wdata = int(str(drec["x_wdata"]), 16) if isinstance(drec["x_wdata"], str) else int(drec["x_wdata"])
    ox_wdata = int(str(orec["x_wdata"]), 16) if isinstance(orec["x_wdata"], str) else int(orec["x_wdata"])
    # Compare every written GPR bank slot (and full 1024-bit bank if x_wb matches).
    wb_mask = int(orec["x_wb"])
    for reg_idx in range(1, 32):
      if (wb_mask >> reg_idx) & 1:
        d_word = (dx_wdata >> (32 * reg_idx)) & 0xFFFFFFFF
        o_word = (ox_wdata >> (32 * reg_idx)) & 0xFFFFFFFF
        if d_word != o_word:
          diffs.append(
            {
              "kind": "GPR_WB_MISMATCH",
              "key": list(k),
              "register": f"x{reg_idx}",
              "dut": f"0x{d_word:08x}",
              "oracle": f"0x{o_word:08x}",
              "bit_diff": first_diff_bit(d_word, o_word, 32),
            }
          )
    if "f_wb" in drec and orec.get("f_wb", 0) != 0:
      if int(drec["f_wb"]) != int(orec["f_wb"]):
        diffs.append(
          {
            "kind": "FPR_WB_MASK_MISMATCH",
            "key": list(k),
            "dut": f"0x{int(drec['f_wb']):08x}",
            "oracle": f"0x{int(orec['f_wb']):08x}",
            "bit_diff": first_diff_bit(int(drec["f_wb"]), int(orec["f_wb"]), 32),
          }
        )
      else:
        df_w = int(str(drec["f_wdata"]), 16) if isinstance(drec["f_wdata"], str) else int(drec["f_wdata"])
        of_w = int(str(orec["f_wdata"]), 16) if isinstance(orec["f_wdata"], str) else int(orec["f_wdata"])
        fmask = int(orec["f_wb"])
        for f_idx in range(32):
          if (fmask >> f_idx) & 1:
            dw = (df_w >> (32 * f_idx)) & 0xFFFFFFFF
            ow = (of_w >> (32 * f_idx)) & 0xFFFFFFFF
            if dw != ow:
              diffs.append(
                {
                  "kind": "FPR_WB_MISMATCH",
                  "key": list(k),
                  "register": f"f{f_idx}",
                  "dut": f"0x{dw:08x}",
                  "oracle": f"0x{ow:08x}",
                  "bit_diff": first_diff_bit(dw, ow, 32),
                }
              )

  for k in sorted(set(oracle_by_key) - seen_keys):
    diffs.append({"kind": "MISSING_DUT_RECORD", "key": list(k), "pc": oracle_by_key[k]["pc_rdata"]})

  dut_mem = dut_trace.get("memory_bytes") or {}
  for rng_key, o_hex in (oracle_trace.get("memory_bytes") or {}).items():
    if rng_key not in dut_mem:
      diffs.append({"kind": "MISSING_MEMORY_DUMP", "range": rng_key})
      continue
    o_bytes = bytes.fromhex(o_hex)
    d_bytes = bytes.fromhex(str(dut_mem[rng_key]))
    if len(o_bytes) != len(d_bytes):
      diffs.append(
        {
          "kind": "MEMORY_LEN_MISMATCH",
          "range": rng_key,
          "dut_len": len(d_bytes),
          "oracle_len": len(o_bytes),
        }
      )
      continue
    for byte_off, (db, ob) in enumerate(zip(d_bytes, o_bytes)):
      if db != ob:
        diffs.append(
          {
            "kind": "MEMORY_BYTE_MISMATCH",
            "range": rng_key,
            "byte_offset": byte_off,
            "dut": f"0x{db:02x}",
            "oracle": f"0x{ob:02x}",
            "bit_diff": first_diff_bit(db, ob, 8),
          }
        )
        break
  return diffs


def check(
  t_id: str, run_dir: Path | str, *, capability_doc: dict[str, Any] | None = None
) -> dict[str, Any]:
  """C4 / C041 entry point `oracle.check(t_id, run_dir) -> verdict` (`cn_verdict/1` schema).

  Expects `run_dir / "oracle_input.json"` and `run_dir / "dut_retire.json"`:
  - If the requested `oracle_class` has `oracle_kind == "NONE"`, returns `t_verdict = "BLOCKED"`
    with explicit `blocked_by` (`归 C047` for BF16/VME or `oracle 证据不足` for G2 mandatory).
  - Otherwise runs the independent Spike + `bitwise_scalar` + `nexus_extend` reference on the
    declared ELF and environment inputs (never reading `dut_retire.json` to form expectations),
    then performs bitwise comparison.
  """
  rdir = Path(run_dir)
  cap = capability_doc or json.loads(CAPABILITY_PATH.read_text())
  classes_by_id = {c["class_id"]: c for c in cap.get("classes") or []}

  inp_path = rdir / "oracle_input.json"
  dut_path = rdir / "dut_retire.json"
  if not inp_path.is_file():
    raise OracleError("MISSING_ORACLE_INPUT", str(inp_path))
  inp = json.loads(inp_path.read_text())

  req_class = str(inp.get("oracle_class", "rv32i.alu_and_shifts"))
  cls_entry = classes_by_id.get(req_class)
  run_id = str(inp.get("run_id", "oracle-run"))
  scenario_id = str(inp.get("scenario_id", "oracle-scenario"))
  phase = str(inp.get("phase", "G2"))
  t_scope = str(inp.get("t_scope", "mandatory"))

  if cls_entry is None or cls_entry.get("oracle_kind") == "NONE":
    status_note = (
      cls_entry.get("status", "BLOCKED(oracle 证据不足)")
      if cls_entry is not None
      else "BLOCKED(oracle 证据不足)"
    )
    is_c047 = status_note == "BLOCKED(归 C047)"
    return {
      "schema": "cn_verdict/1",
      "run_id": run_id,
      "scenario_id": scenario_id,
      "t_id": t_id,
      "t_verdict": "BLOCKED",
      "phase": phase,
      "checker": [
        {
          "name": "C4",
          "status": "BLOCKED",
          "reason": status_note,
          "evidence": req_class,
        }
      ],
      "evidence": [],
      "t_scope": t_scope,
      "notes": status_note,
      "g2_blocking": not is_c047 and phase == "G2",
      "blocked_by": {
        "kind": "unfrozen_spec" if is_c047 else "environment_assumption",
        "ref": "C047" if is_c047 else f"{req_class}:oracle_kind=NONE",
      },
    }

  if not dut_path.is_file():
    raise OracleError("MISSING_DUT_RETIRE", str(dut_path))

  contexts = inp.get("contexts")
  if not contexts:
    contexts = [
      {
        "elf_path": inp["elf_path"],
        "entry_pc": inp.get("entry_pc"),
        "ctx_slot": int(inp.get("ctx_slot", 0)),
        "ctx_generation": int(inp.get("ctx_generation", 1)),
        "initial_gpr": inp.get("initial_gpr") or {},
        "initial_fpr": inp.get("initial_fpr") or {},
        "initial_csr": inp.get("initial_csr") or {},
        "initial_memory": inp.get("initial_memory") or [],
        "read_memory_ranges": inp.get("read_memory_ranges") or [],
        "enable_nexus_extend": bool(inp.get("enable_nexus_extend", False)),
        "env_events": inp.get("env_events") or [],
      }
    ]

  combined_retire: list[dict[str, Any]] = []
  combined_mem: dict[str, str] = {}
  for ctx in contexts:
    elf_p = Path(ctx["elf_path"])
    if not elf_p.is_absolute():
      elf_p = (REPO / elf_p) if (REPO / elf_p).is_file() else (rdir / elf_p)
    trace = srd.run_trace_request(
      srd.TraceRequest(
        elf_path=str(elf_p),
        entry_pc=ctx.get("entry_pc"),
        ctx_slot=int(ctx.get("ctx_slot", 0)),
        ctx_generation=int(ctx.get("ctx_generation", 1)),
        max_steps=int(ctx.get("max_steps", 4096)),
        initial_gpr={str(k): int(v) for k, v in (ctx.get("initial_gpr") or {}).items()},
        initial_fpr={str(k): int(v) for k, v in (ctx.get("initial_fpr") or {}).items()},
        initial_csr={str(k): int(v) for k, v in (ctx.get("initial_csr") or {}).items()},
        initial_memory=list(ctx.get("initial_memory") or []),
        read_memory_ranges=list(ctx.get("read_memory_ranges") or []),
        enable_nexus_extend=bool(ctx.get("enable_nexus_extend", False)),
        env_events=list(ctx.get("env_events") or []),
      )
    )
    combined_retire.extend(trace["retire_stream"])
    combined_mem.update(trace["memory_bytes"])

  oracle_out = {"retire_stream": combined_retire, "memory_bytes": combined_mem}
  oracle_ref_path = rdir / "oracle_reference.json"
  oracle_bytes = canonical_json(oracle_out).encode()
  oracle_ref_path.write_bytes(oracle_bytes)

  dut_doc = json.loads(dut_path.read_text())
  diffs = compare_retire_and_memory(oracle_trace=oracle_out, dut_trace=dut_doc)
  passed = len(diffs) == 0
  verdict: dict[str, Any] = {
    "schema": "cn_verdict/1",
    "run_id": run_id,
    "scenario_id": scenario_id,
    "t_id": t_id,
    "t_verdict": "PASS" if passed else "FAIL",
    "phase": phase,
    "checker": [
      {
        "name": "C4",
        "status": "PASS" if passed else "FAIL",
        "reason": "BITWISE_MATCH" if passed else f"NUMERIC_FAIL:{diffs[0]['kind']}",
        "evidence": str(oracle_ref_path),
      }
    ],
    "evidence": [
      {
        "kind": "BIT_ORACLE",
        "path": str(oracle_ref_path),
        "sha256": sha256_bytes(oracle_bytes),
        "t_id": t_id,
      }
    ],
    "t_scope": t_scope,
    "notes": "" if passed else json.dumps(diffs[0], sort_keys=True),
    "diffs": diffs,
  }
  if not passed:
    verdict["repro"] = {
      "window_id": str(inp.get("window_id", "w0")),
      "seed": int(inp.get("seed", 1)),
      "inject_plan_ref": str(inp.get("inject_plan_ref", "none")),
    }
  return verdict


def main() -> int:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="cmd", required=True)
  gen = sub.add_parser("generate")
  gen.add_argument("--work-dir", default="")
  chk = sub.add_parser("check")
  chk.add_argument("--t-id", required=True)
  chk.add_argument("--run-dir", required=True)
  args = parser.parse_args()

  if args.cmd == "generate":
    wdir = Path(args.work_dir) if args.work_dir else None
    cap, _cov = write_generated_oracle_artifacts(work_dir=wdir)
    print(
      json.dumps(
        {
          "capability_sha256": sha256_bytes(CAPABILITY_PATH.read_bytes()),
          "coverage_sha256": sha256_bytes(COVERAGE_PATH.read_bytes()),
          "g2_ready": cap["summary"]["g2_ready"],
          "ready_scalar_classes": cap["summary"]["ready_scalar_classes"],
          "c047_blocked": len(cap["summary"]["c047_blocked_classes"]),
        },
        indent=2,
        sort_keys=True,
      )
    )
    return 0

  verdict = check(args.t_id, Path(args.run_dir))
  print(canonical_json(verdict), end="")
  return 0 if verdict["t_verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
