"""C038 scalar control helper oracle tests (W7-02 / W5-02).

Two layers, both real:

  * The helper/CSR semantic model (`nexus_extend` + sealed `scalar_helper_abi_v1.json`): carrier
    legality, operand domains, WAIT park/resolve, descriptor transfer settlement through real
    memory, and the sealed retire chain ending at task_ret. These run always.
  * The linked-package acceptance (`scalar_ctl_run.load_case` + `run_case`): the sealed C037
    Loader authenticates the package (codec + semantic load_program + ELF cross-check + linked_slot /
    slot_freeze commitment), then the real linked `image.elf` executes on the pinned Spike
    reference with the `scalar_env` armed from the binding's `launch_descriptors`. The verdict is
    carried by independent golden output (input words + 7, never DUT-derived), both MFE transfer
    directions in the final memory, the retire stream against the published ELF bytes and the
    committed freeze, the fflags path against `bitwise_scalar`, and once-only env-event delivery.
    Nothing here skips or xfails: a build, loader or trace defect is a plain failure.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
H = HERE.parents[1]
REPO = H.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(H / "scripts" / "lib"))

import nexus_extend as ne  # noqa: E402

BINDING = H / "kernels" / "bindings" / "scalar_ctl_v1.yaml"
ABI = H / "protocol" / "scalar_helper_abi_v1.json"


class _DictMem:
  """Minimal ReadMemory/WriteMemory endpoint for the settle path."""

  def __init__(self) -> None:
    self.mem: dict[int, bytes] = {}

  def ReadMemory(self, addr: int, length: int) -> bytes:
    data = self.mem.get(addr)
    if data is None or len(data) < length:
      return bytes(data or b"") + bytes(max(0, length - len(data or b"")))
    return data[:length]

  def WriteMemory(self, addr: int, data: bytes) -> None:
    self.mem[addr] = bytes(data)


def _env(descriptors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
  return {
    "descriptors": descriptors
    or [
      {
        "index": 0,
        "direction": "l2_to_frame",
        "handle": "input",
        "bytes": 16,
        "slot_offset": 0,
        "frame_offset": 0,
      },
      {
        "index": 1,
        "direction": "frame_to_l2",
        "handle": "output",
        "bytes": 16,
        "slot_offset": 0,
        "frame_offset": 0,
      },
    ],
    "bindings": {"input": {"base": 0x18000, "length": 64}, "output": {"base": 0x1C000, "length": 64}},
    "frame_base": 0x14000,
    "frame_length": 64,
  }


def _scalar_csr_addr(name: str) -> int:
  """Resolve a scalar helper CSR address by logical name from the sealed catalog."""
  catalog = ne.load_nexus_csr_catalog()
  for addr, spec in catalog.items():
    if spec.name == name:
      return addr
  raise AssertionError(f"{name} not in generated layout")


CSR_MFE = _scalar_csr_addr("nexus.mfe.submit")
CSR_WAIT = _scalar_csr_addr("nexus.event.wait")
CSR_PHASE = _scalar_csr_addr("nexus.phase.signal")
CSR_TASK = _scalar_csr_addr("nexus.task.ret")


LOADER_ERROR_CODES = {
  "BAD_JSON",
  "MISSING_FIELD",
  "UNREADABLE",
  "SCHEMA_MISMATCH",
  "IMAGE_NOT_ELF",
  "PACKAGE_REJECTED",
  "CONTROL_PLANE_INVALID",
  "CONTROL_PLANE_REJECTED",
  "SLOT_FREEZE_MISMATCH",
  "SLOT_FREEZE_MISSING",
  "SLOT_MEASUREMENT_MISMATCH",
  "SLOT_IDENTITY_MISMATCH",
  "IMAGE_SHA_MISMATCH",
  "RESOURCE_MISMATCH",
  "BINDING_MISMATCH",
  "TARGET_MISMATCH",
}


def _armed(env: dict[str, Any] | None = None) -> tuple[ne.NexusContextAdapter, _DictMem]:
  ad = ne.NexusContextAdapter(ctx_slot=0, ctx_generation=1)
  mem = _DictMem()
  ad.arm_scalar(ne.scalar_env_from_dict(env or _env()), mem=mem)
  return ad, mem


def _csrw(csr: int, rs1: int = 0) -> int:
  return (rs1 << 15) | (1 << 12) | (csr << 20) | 0x73


def _csrwi(csr: int, uimm: int = 0) -> int:
  return (uimm << 15) | (5 << 12) | (csr << 20) | 0x73


# ---------------------------------------------------------------------------
# Sealed ABI / CSR mapping honesty.
# ---------------------------------------------------------------------------


def test_scalar_abi_loads_and_maps_the_four_csrs() -> None:
  abi = ne.load_scalar_helper_abi()
  kinds = set(abi.get("_helper_by_kind") or {})
  assert kinds == {"MFE_SUBMIT", "WAIT", "SIGNAL_PHASE", "TASK_RET"}
  catalog = ne.load_nexus_csr_catalog()
  by_name = {s.name: s for s in catalog.values()}
  for csr in ("nexus.mfe.submit", "nexus.event.wait", "nexus.phase.signal", "nexus.task.ret"):
    assert csr in by_name, f"{csr} not in generated layout"
    assert by_name[csr].decl_kind in kinds


def test_scalar_env_validates_descriptor_contract() -> None:
  env = ne.scalar_env_from_dict(_env())
  assert [d["index"] for d in env.descriptors] == [0, 1]
  assert len(env.wait_events) == 2  # defaulted to descriptor indices
  with pytest.raises(ne.NexusExtendError, match="DESCRIPTOR_FIELD_MISSING"):
    ne.scalar_env_from_dict(_env([{"index": 0, "direction": "l2_to_frame"}]))
  empty_env = _env()
  empty_env["descriptors"] = []
  with pytest.raises(ne.NexusExtendError, match="MISSING_DESCRIPTORS"):
    ne.scalar_env_from_dict(empty_env)
  with pytest.raises(ne.NexusExtendError, match="FRAME_RANGE"):
    ne.scalar_env_from_dict({**_env(), "frame_base": 0})


# ---------------------------------------------------------------------------
# Carrier legality + operand domains (sealed forms only).
# ---------------------------------------------------------------------------


def test_scalar_csr_carrier_forms_enforced() -> None:
  ad, _ = _armed()
  csr = CSR_MFE
  ad.execute_private_csr(pc=0, insn=_csrw(csr), rs1_val=0, owner_op_seq=0)
  # A read form / a set-form is a protocol fault, never modeled.
  csrrs = (0 << 15) | (2 << 12) | (csr << 20) | 0x73
  with pytest.raises(ne.NexusExtendError, match="HELPER_ENCODING_INVALID"):
    ad.execute_private_csr(pc=4, insn=csrrs, rs1_val=0, owner_op_seq=1)
  csrr_rd = (0 << 15) | (2 << 12) | (csr << 20) | (1 << 7) | 0x73  # rd != x0
  with pytest.raises(ne.NexusExtendError):
    ad.execute_private_csr(pc=8, insn=csrr_rd, rs1_val=0, owner_op_seq=2)


def test_mfe_submit_descriptor_index_domain() -> None:
  ad, _ = _armed()
  ad.execute_private_csr(pc=0, insn=_csrwi(CSR_MFE, 0), rs1_val=0, owner_op_seq=0)
  with pytest.raises(ne.NexusExtendError, match="DESCRIPTOR_INDEX_RANGE"):
    ad.execute_private_csr(pc=4, insn=_csrwi(CSR_MFE, 9), rs1_val=0, owner_op_seq=1)


def test_wait_operand_domain_and_park_resolve() -> None:
  ad, _ = _armed()
  ad.execute_private_csr(pc=0, insn=_csrwi(CSR_MFE, 0), rs1_val=0, owner_op_seq=0)
  with pytest.raises(ne.NexusExtendError, match="OPERAND_OUT_OF_DOMAIN"):
    ad.execute_private_csr(pc=4, insn=_csrwi(CSR_WAIT, 9), rs1_val=0, owner_op_seq=1)
  rec = ad.execute_private_csr(pc=8, insn=_csrwi(CSR_WAIT, 0), rs1_val=0, owner_op_seq=2)
  assert rec["effect"]["parked"] is True and rec["effect"]["resolved"] is False
  assert ad.try_resolve_wait() == "parked"
  ad.inject_env_event(ne.NexusEnvEvent(9, "EVENT_DONE", 0, "DONE"))
  assert ad.try_resolve_wait() == "resolved"


def test_signal_phase_and_task_ret_domains() -> None:
  ad, _ = _armed()
  with pytest.raises(ne.NexusExtendError, match="SIGNAL_PHASE_RANGE"):
    ad.execute_private_csr(pc=0, insn=_csrwi(CSR_PHASE, 9), rs1_val=0, owner_op_seq=0)
  ad.execute_private_csr(pc=4, insn=_csrwi(CSR_PHASE, 0), rs1_val=0, owner_op_seq=1)
  with pytest.raises(ne.NexusExtendError, match="TASK_RET_CODE"):
    ad.execute_private_csr(pc=8, insn=_csrwi(CSR_TASK, 7), rs1_val=0, owner_op_seq=2)


# ---------------------------------------------------------------------------
# Real transfer settlement through memory + the retire chain.
# ---------------------------------------------------------------------------


def test_l2_to_frame_and_frame_to_l2_transfers_move_real_bytes() -> None:
  env = _env()
  mem = _DictMem()
  words = (11, 22, 33, 44)
  mem.mem[0x18000] = b"".join(w.to_bytes(4, "little") for w in words)
  ad = ne.NexusContextAdapter(ctx_slot=0, ctx_generation=1)
  ad.arm_scalar(ne.scalar_env_from_dict(env), mem=mem)
  ad.inject_env_event(ne.NexusEnvEvent(0, "EVENT_DONE", 0, "DONE"))
  ad.execute_private_csr(pc=0, insn=_csrwi(CSR_MFE, 0), rs1_val=0, owner_op_seq=0)
  ad.execute_private_csr(pc=4, insn=_csrwi(CSR_WAIT, 0), rs1_val=0, owner_op_seq=1)
  assert mem.mem[0x14000] == b"".join(w.to_bytes(4, "little") for w in words)
  # Write the frame, submit the outbound store, wait -> real bytes land at the L2 slot.
  mem.mem[0x14000] = b"".join((w + 7).to_bytes(4, "little") for w in words)
  ad.inject_env_event(ne.NexusEnvEvent(1, "EVENT_DONE", 1, "DONE"))
  ad.execute_private_csr(pc=8, insn=_csrwi(CSR_MFE, 1), rs1_val=0, owner_op_seq=2)
  ad.execute_private_csr(pc=12, insn=_csrwi(CSR_WAIT, 1), rs1_val=0, owner_op_seq=3)
  assert mem.mem[0x1C000] == b"".join((w + 7).to_bytes(4, "little") for w in words)
  assert ad.settled_transfer_indices == {0, 1}


def test_task_ret_last_and_clean_exit() -> None:
  ad, mem = _armed()
  mem.mem[0x18000] = bytes(16)
  for key in (0, 1):
    ad.inject_env_event(ne.NexusEnvEvent(key, "EVENT_DONE", key, "DONE"))
  for pc, (csr, op) in enumerate(
    [(CSR_MFE, 0), (CSR_WAIT, 0), (CSR_MFE, 1), (CSR_WAIT, 1), (CSR_PHASE, 0), (CSR_PHASE, 1)]
  ):
    ad.execute_private_csr(pc=4 * pc, insn=_csrwi(csr, op), rs1_val=0, owner_op_seq=pc)
  # task_ret while a transfer is unsettled is refused.
  ad2, _ = _armed()
  ad2.execute_private_csr(pc=0, insn=_csrwi(CSR_MFE, 0), rs1_val=0, owner_op_seq=0)
  with pytest.raises(ne.NexusExtendError, match="TASK_RET_UNSETTLED"):
    ad2.execute_private_csr(pc=4, insn=_csrwi(CSR_TASK, 0), rs1_val=0, owner_op_seq=1)
  # A clean chain retires at task_ret and accepts nothing after it.
  ad.execute_private_csr(pc=24, insn=_csrwi(CSR_TASK, 0), rs1_val=0, owner_op_seq=6)
  assert ad.task_finished
  with pytest.raises(ne.NexusExtendError, match="TASK_AFTER_RET"):
    ad.execute_private_csr(pc=28, insn=_csrwi(CSR_PHASE, 0), rs1_val=0, owner_op_seq=7)
  ad.check_task_exit_clean()


# ---------------------------------------------------------------------------
# Linked-package acceptance (real Loader + real Spike run + independent golden).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  """Build the real package in-process, then run it on the pinned Spike through the sealed Loader."""
  import package_build
  import scalar_ctl_run as sc

  pkg_out = tmp_path_factory.mktemp("c038-pkg")
  package_build.build(BINDING, H / "configs" / "cs1_c2.yaml", pkg_out)
  case = sc.load_case(
    binding_path=BINDING,
    freeze_path=pkg_out / "slot_freeze.json",
    slot="A",
    input_bytes=b"".join(w.to_bytes(4, "little") for w in range(1, 9)),
    expect_addend=7,
  )
  report, trace = sc.run_case(case)
  return {"sc": sc, "case": case, "report": report, "trace": trace}


def test_scalar_ctl_package_runs_real_spike_with_independent_golden(real_run: dict[str, Any]) -> None:
  """Real C038 acceptance: sealed Loader + published ELF + independent golden and flags on Spike."""
  case, report, trace = (real_run[k] for k in ("case", "report", "trace"))
  assert report["verdict"] == "PASS", json.dumps(report["diffs"], indent=1)
  for section in ("helper_stream", "helper_map", "retire", "fflags", "transfer", "env_events"):
    assert report[section]["ok"] is True, f"{section}: {json.dumps(report[section])[:400]}"
  # Independence: the oracle-computed golden (input + 7) equals the committed kernel golden.
  assert report["golden"]["golden_file_match"] is True
  kinds = [h["kind"] for h in trace["scalar_helpers"]["helpers"]]
  assert kinds[:4] == ["MFE_SUBMIT", "WAIT", "MFE_SUBMIT", "WAIT"]
  assert kinds[-1] == "TASK_RET"
  assert trace["task_exit"]["clean"] is True
  # Retire: one record per instruction of the published image, matching the freeze's own counts,
  # and the file that was graded is the sealed image.
  retire = report["retire"]
  assert retire["records"] == retire["instruction_count"] == retire["retire_count"]
  assert retire["records"] == len(trace["retire_stream"])
  assert retire["image_sha256_checked"] == retire["image_sha256"] == report["image_sha256"]
  # Helper mapping: the committed freeze's helpers, in binding order, each on its generated CSR.
  helpers = report["helper_map"]["helpers"]
  assert [row["helper"] for row in helpers] == [h["name"] for h in case.kernel["helpers"]]
  # Transfers: the Frame output stripe equals the L2 output; nothing outside descriptors moved.
  assert report["transfer"]["frame_untouched_outside_descriptors"] is True
  # fflags: the kernel squares one power of two. 2^-k * 2^-k = 2^-2k is nonzero yet below the
  # smallest subnormal (2^-149), so RNE delivers +0 and raises exactly underflow and inexact. That
  # expectation is derived here from first principles, independent of Spike and of bitwise_scalar.
  op = report["fflags"]["fp_ops"][0]
  m = re.fullmatch(r"2\^-(\d+)", op["fs1"]["value"])
  assert m is not None and op["fs2"]["value"] == op["fs1"]["value"], op
  assert 2 * int(m.group(1)) > 149
  assert op["retired_result"] == "0x00000000"
  assert op["retired_flags_delta"] == op["expected_fflags"] == 0b00011  # fflags: NX=bit0, UF=bit1
  assert op["expected_fflags_names"] == ["UF", "NX"]
  # Software read the raised flags back through the kernel's own csrr, never through the driver.
  read = report["fflags"]["reads"][0]
  assert report["fflags"]["raise_observed_by_software"] is True
  assert read["fflags"] == op["expected_fflags"]
  assert trace["retire_stream"][read["record"]]["x_rd_val"] == op["expected_fflags"]
  # Every scheduled env completion was delivered exactly once.
  assert report["env_events"]["scheduled"] == report["env_events"]["logged"]
  assert len(set(report["env_events"]["logged"])) == len(case.env_events)


def test_scalar_ctl_grading_fails_closed_on_altered_evidence(real_run: dict[str, Any]) -> None:
  """Every verdict-bearing section fails on the specific defect it guards, using the real trace."""
  import copy

  sc, case, trace = (real_run[k] for k in ("sc", "case", "trace"))
  fflags = real_run["report"]["fflags"]
  fmul = fflags["fp_ops"][0]["record"]
  csrr = fflags["reads"][0]["record"]
  plain = next(
    i for i, r in enumerate(trace["retire_stream"]) if r["source"] == "spike_isa_ref" and i < fmul - 2
  )
  frame_key = f"0x{case.frame_base:08x}:{case.frame_length}"
  out_key = f"0x{case.out_range['address']:08x}:{case.out_range['length']}"
  in_desc = next(d for d in case.descriptors if d["direction"] == "l2_to_frame")
  in_base = case.bindings[in_desc["handle"]]["base"] + in_desc["slot_offset"]
  in_key = f"0x{in_base:08x}:{in_desc['bytes']}"
  out_offset = next(d["frame_offset"] for d in case.descriptors if d["direction"] == "frame_to_l2")

  def altered(edit: Callable[[dict[str, Any]], None]) -> tuple[Any, dict[str, Any]]:
    t = copy.deepcopy(trace)
    edit(t)
    return case, t

  def flip_bytes(t: dict[str, Any], key: str, offset: int) -> None:
    raw = bytearray.fromhex(t["memory_bytes"][key])
    raw[offset] ^= 0xFF
    t["memory_bytes"][key] = bytes(raw).hex()

  def retire_edit(index: int, **fields: Any) -> Callable[[dict[str, Any]], None]:
    def edit(t: dict[str, Any]) -> None:
      t["retire_stream"][index].update(fields)

    return edit

  def refrozen(list_key: str, index: int, field: str, value: Any) -> tuple[Any, dict[str, Any]]:
    c = copy.deepcopy(case)
    c.freeze[list_key][index][field] = value
    return c, copy.deepcopy(trace)

  def drop_last_record(t: dict[str, Any]) -> None:
    t["retire_stream"].pop()
    t["arch_fflags_stream"].pop()

  def vacuous_fflags_path(t: dict[str, Any]) -> None:
    # 1.0 * 1.0 raises nothing, and every read-back is made consistent with that: still no flag
    # is ever raised or observed, which the acceptance must refuse to call an fflags path.
    t["retire_stream"][fmul - 2]["x_rd_val"] = 0x3F800000
    t["retire_stream"][fmul - 1]["f_rd_val"] = 0x3F800000
    t["retire_stream"][fmul].update(f_rd_val=0x3F800000, flags_delta=0)
    t["retire_stream"][csrr]["x_rd_val"] = 0
    for i in range(fmul, len(t["arch_fflags_stream"])):
      t["arch_fflags_stream"][i] = 0

  def zero_stream_at_fmul(t: dict[str, Any]) -> None:
    t["arch_fflags_stream"][fmul] = 0

  def set_final_fflags(t: dict[str, Any]) -> None:
    t["final_csr"]["fflags"] = 3

  def drop_stream(t: dict[str, Any]) -> None:
    del t["arch_fflags_stream"]

  def flip_frame_last(t: dict[str, Any]) -> None:
    flip_bytes(t, frame_key, case.frame_length - 1)

  def flip_frame_stripe(t: dict[str, Any]) -> None:
    flip_bytes(t, frame_key, out_offset)

  def flip_output(t: dict[str, Any]) -> None:
    flip_bytes(t, out_key, 0)

  def clobber_input(t: dict[str, Any]) -> None:
    # The L2 input range and its Frame stripe still agree with each other, but not with the input.
    flip_bytes(t, in_key, 0)
    flip_bytes(t, frame_key, in_desc["frame_offset"])

  def duplicate_event(t: dict[str, Any]) -> None:
    t["scalar_helpers"]["env_log"].append(dict(t["scalar_helpers"]["env_log"][0]))

  flipped_insn = trace["retire_stream"][plain]["insn"] ^ (1 << 20)
  mutations: list[tuple[str, str, tuple[Any, dict[str, Any]]]] = [
    ("dropped retire record", "RETIRE_STREAM", altered(drop_last_record)),
    (
      "retired word differs from the image",
      "RETIRE_INSN_MISMATCH",
      altered(retire_edit(plain, insn=flipped_insn)),
    ),
    ("freeze retire_count", "RETIRE_COUNT_MISMATCH", refrozen("slots", 0, "retire_count", 35)),
    (
      "freeze instruction_count far beyond the text span",
      "RETIRE_FREEZE_ROW_INVALID",
      refrozen("slots", 0, "instruction_count", 10**12),
    ),
    (
      "freeze retire_count not positive",
      "RETIRE_FREEZE_ROW_INVALID",
      refrozen("slots", 0, "retire_count", 0),
    ),
    (
      "freeze retire_count a float",
      "RETIRE_FREEZE_ROW_INVALID",
      refrozen("slots", 0, "retire_count", 36.0),
    ),
    ("freeze helper offset", "HELPER_MAP", refrozen("helper_sequence", 5, "offset", 137)),
    (
      "unmodeled FP instruction (fsub.s)",
      "FFLAGS_MODEL_UNSUPPORTED",
      altered(retire_edit(plain, insn=0x08008053)),
    ),
    ("fmul.s flags_delta zeroed", "FFLAGS_DELTA_MISMATCH", altered(retire_edit(fmul, flags_delta=0))),
    ("fmul.s result altered", "FFLAGS_RESULT_MISMATCH", altered(retire_edit(fmul, f_rd_val=1))),
    (
      "csrr fflags read-back altered",
      "FFLAGS_CSR_READBACK_MISMATCH",
      altered(retire_edit(csrr, x_rd_val=0)),
    ),
    ("per-record architectural fflags", "FFLAGS_ARCH_STREAM_MISMATCH", altered(zero_stream_at_fmul)),
    ("final fflags", "FFLAGS_FINAL_MISMATCH", altered(set_final_fflags)),
    ("fflags evidence removed", "FFLAGS_EVIDENCE_MISSING", altered(drop_stream)),
    ("vacuous fflags path", "FFLAGS_PATH_VACUOUS", altered(vacuous_fflags_path)),
    ("Frame written outside descriptors", "FRAME_TOUCHED_OUTSIDE_DESCRIPTORS", altered(flip_frame_last)),
    ("Frame output stripe differs from L2", "TRANSFER_MISMATCH", altered(flip_frame_stripe)),
    ("L2 input overwritten", "TRANSFER_MISMATCH", altered(clobber_input)),
    ("env completion delivered twice", "ENV_EVENT_DELIVERY", altered(duplicate_event)),
  ]
  # Control: an untouched copy still passes, so each failure below comes from its mutation alone.
  assert sc.evaluate(*altered(lambda t: None))["verdict"] == "PASS"
  for name, kind, (c, t) in mutations:
    rep = sc.evaluate(c, t)
    assert rep["verdict"] == "FAIL", f"{name}: not rejected"
    assert kind in {d["kind"] for d in rep["diffs"]}, f"{name}: {rep['diffs']}"
  # The output defect stays first in `diffs` (consumers read it as the primary verdict).
  rep = sc.evaluate(*altered(flip_output))
  assert rep["verdict"] == "FAIL" and rep["diffs"][0]["kind"] == "GOLDEN_MISMATCH"


def test_bogus_package_rejected_by_loader(tmp_path: Path) -> None:
  """A malformed package.json is refused by the sealed Loader's codec, never trusted."""
  import scalar_ctl_run as sc

  fake_dir = tmp_path / "A"
  fake_dir.mkdir()
  (fake_dir / "package.json").write_text("{}")
  (fake_dir / "compiled_program.json").write_text("{}")
  (fake_dir / "image.elf").write_bytes(b"\x7fELF")
  with pytest.raises(sc.ScalarCaseError) as exc:
    sc.load_case(binding_path=BINDING, package_dir=fake_dir, input_bytes=bytes(32), expect_addend=7)
  # The refusal came from the Loader's codec, not from pre-Loader plumbing.
  assert exc.value.code not in {"CONTROL_PLANE_FAILED", "LOADER_UNAVAILABLE", "PACKAGE_MISSING"}
  assert exc.value.code in LOADER_ERROR_CODES, exc.value.code
