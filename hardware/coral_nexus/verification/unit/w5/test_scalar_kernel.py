"""C038 scalar control kernel: real link, static audit, slot freeze and refusals.

What is real: the two images are linked by the pinned C004 toolchain from the committed kernel sources
into the two fixed slots, the audit reads the executable PT_LOAD bytes of those ELFs, and the freeze and
the golden bytes are the files ``cn package`` publishes. Every refusal must be a coded
``measure_kernel.MeasureError``: a link script that does not state its slot, overlapping slot images, a
Frame that cannot fit one bank, a kernel that overflows its slot or carries undeclared bytes, and every
helper-ABI or Frame-access violation on the executed bytes. The audit refusals run on hand-encoded words
(public RV32 base fields), so they need no toolchain; the hand encoding is compared with the real image
whenever the toolchain is present.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
H = REPO_ROOT / "hardware/coral_nexus"
sys.path.insert(0, str(H / "scripts/lib"))

import measure_kernel as mk  # noqa: E402
import scalar_kernel_build as skb  # noqa: E402

HAVE_TOOLCHAIN = mk.TOOLCHAIN.is_dir()
BINDING = skb.BINDING
CONFIG = H / "configs/cs1_c2.yaml"
CLOBBER = ["t0", "t1", "ft0", "ft1"]
X0, RA, SP, T0, T1, S0, A0 = 0, 1, 2, 5, 6, 8, 10
FT0, FT1 = 0, 1
FFLAGS = 0x001
CSRRW, CSRRS, CSRRWI = 1, 2, 5
OUT = 1024  # the output buffer: one stripe round above the input, as the compiled control plane places it
# instruction positions in kernels/library/scalar_ctl_v1/scalar_ctl_v1.S
SUBMIT0, WAIT0, FIRST_LOAD, FIRST_STORE = 0, 1, 2, 4
LUI, FMUL = 26, 28
SUBMIT1, WAIT1, PHASE0, PHASE1, RET = 31, 32, 33, 34, 35

needs_toolchain = pytest.mark.skipif(
  not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted under out/coral_nexus/cache"
)


# ---------------------------------------------------------------------------------------------------
# inputs and the hand encoding of the kernel
# ---------------------------------------------------------------------------------------------------
def binding() -> dict[str, Any]:
  return yaml.safe_load(BINDING.read_text())


def config() -> dict[str, Any]:
  return yaml.safe_load(CONFIG.read_text())


def helpers() -> dict[str, dict[str, Any]]:
  return skb.helper_csrs()["helpers"]


def nexus_address(name: str) -> int:
  return {logical: number for number, logical in skb.helper_csrs()["nexus"].items()}[name]


def addresses() -> dict[str, int]:
  return {name: helper["address"] for name, helper in helpers().items()}


def refuse(code: str, fn: Any, *args: Any, **kw: Any) -> mk.MeasureError:
  with pytest.raises(mk.MeasureError) as caught:
    fn(*args, **kw)
  error = caught.value
  assert error.code == code, error
  assert not error.detail.startswith(code), "the code is never repeated inside the detail"
  return error


def _i(f3: int, rd: int, rs1: int, imm: int, opcode: int) -> int:
  return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode


def lw(rd: int, rs1: int, imm: int) -> int:
  return _i(2, rd, rs1, imm, 0x03)


def addi(rd: int, rs1: int, imm: int) -> int:
  return _i(0, rd, rs1, imm, 0x13)


def lui(rd: int, imm20: int) -> int:
  return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | 0x37


def sw(rs2: int, rs1: int, imm: int) -> int:
  imm &= 0xFFF
  return (((imm >> 5) & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | (2 << 12) | ((imm & 31) << 7) | 0x23


def csr(f3: int, rd: int, rs1: int, csr: int) -> int:
  return ((csr & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x73


def fmv(rd: int, rs: int) -> int:
  return (0x78 << 25) | (rs << 15) | (rd << 7) | 0x53


def fmul(rd: int, rs1: int, rs2: int, rm: int = 0) -> int:
  return (0x08 << 25) | (rs2 << 20) | (rs1 << 15) | (rm << 12) | (rd << 7) | 0x53


def program(h: dict[str, dict[str, Any]] | None = None) -> list[int]:
  """The 36 words of the committed kernel, encoded from the public RV32 base fields."""
  h = helpers() if h is None else h
  submit, wait, phase, ret = (
    h[name]["address"] for name in ("mfe_submit", "wait", "signal_phase", "task_ret")
  )
  words = [csr(CSRRWI, X0, 0, submit), csr(CSRRWI, X0, 0, wait)]
  for index in range(8):
    words += [lw(T0, A0, 4 * index), addi(T0, T0, 7), sw(T0, A0, OUT + 4 * index)]
  words += [
    lui(T0, 0x800),  # 0x00800000 = 2^-126, the smallest normal
    fmv(FT0, T0),
    fmul(FT1, FT0, FT0),
    csr(CSRRS, T1, X0, FFLAGS),  # csrr t1, fflags
    csr(CSRRW, X0, X0, FFLAGS),  # csrrw x0, fflags, x0
  ]
  words += [
    csr(CSRRWI, X0, 1, submit),
    csr(CSRRWI, X0, 1, wait),
    csr(CSRRWI, X0, 0, phase),
    csr(CSRRWI, X0, 1, phase),
    csr(CSRRWI, X0, 0, ret),
  ]
  return words


def packed(words: list[int]) -> bytes:
  return b"".join(struct.pack("<I", word) for word in words)


def audited(words: list[int], **overrides: Any) -> dict[str, Any]:
  descriptors, expected = skb.check_binding(binding(), config(), skb.scalar_spec())
  return skb.audit(
    packed(words),
    0,
    overrides.get("descriptors", descriptors),
    overrides.get("expected", expected),
    skb.helper_csrs(),
    overrides.get("clobber", CLOBBER),
  )


def kernel_copy(tmp_path: Path, **edits: tuple[str, str]) -> Path:
  """A copy of the committed kernel directory with text replacements applied per file."""
  kernel_dir = tmp_path / skb.KERNEL
  shutil.copytree(skb.SOURCE, kernel_dir)
  for name, (old, new) in edits.items():
    path = kernel_dir / name
    text = path.read_text()
    assert old in text, (name, old)
    path.write_text(text.replace(old, new))
  return kernel_dir


def oversized_frame(doc: dict[str, Any], elements: int, first_bytes: int) -> None:
  """Widen both views and split the Frame envelope over the two descriptors."""
  entry = doc["manifest_kernel"]
  for view in entry["ins"] + entry["outs"]:
    view["shape"] = [elements]
  first, second = entry["launch_descriptors"]
  first["bytes"] = first_bytes
  second["frame_offset"], second["bytes"] = first_bytes, 4 * elements - first_bytes


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  out = tmp_path_factory.mktemp("scalar_ctl_v1") / "stage"
  images = skb.measure(BINDING, CONFIG, out)
  return {"out": out, "images": images, "freeze": json.loads((out / skb.FREEZE).read_text())}


# ---------------------------------------------------------------------------------------------------
# the committed binding and the sealed inputs (no toolchain)
# ---------------------------------------------------------------------------------------------------
def test_the_binding_is_authored_and_layout_is_exactly_the_two_contract_slots() -> None:
  doc = binding()
  entry = doc["manifest_kernel"]
  spec = skb.scalar_spec()
  assert doc["kernel"] == skb.KERNEL and doc["layout"] == ["A", "B"]
  assert "linked_slot" not in entry, "linked_slot is written by the build, never authored"
  assert entry["kernel_id"] == skb.KERNEL_ID and entry["kernel_version"] == skb.KERNEL_VERSION
  assert entry["initial"] == {"method": "none"} and entry["mode_table"] == "none"
  assert entry["region"] == {"start": False, "end": False, "chains": []}
  descriptors, expected = skb.check_binding(doc, config(), spec)
  assert [(d["index"], d["direction"], d["frame_range"]) for d in descriptors] == [
    (0, "l2_to_frame", [0, 32]),
    (1, "frame_to_l2", [OUT, OUT + 32]),
  ]
  assert expected == [
    ("mfe_submit", 0),
    ("wait", 0),
    ("mfe_submit", 1),
    ("wait", 1),
    ("signal_phase", 0),
    ("signal_phase", 1),
    ("task_ret", 0),
  ]
  assert spec["linked_slot"]["slot_names"] == ["A", "B"]


def test_slot_geometry_comes_from_the_target_and_identities_from_the_contract() -> None:
  constraints = config()["constraints"]
  spec = skb.scalar_spec()
  slots = {slot["name"]: slot for slot in skb.slot_table(config(), spec)}
  assert list(slots) == spec["linked_slot"]["slot_names"]
  text = constraints["itcm_bytes"] // constraints["program_slots"]
  data = constraints["dtcm_bytes"] // constraints["program_slots"]
  assert text == 4096 and data == 16384
  assert (slots["A"]["text_base"], slots["B"]["text_base"]) == (0, text)
  assert (slots["A"]["data_base"], slots["B"]["data_base"]) == (0x10000, 0x10000 + data)
  assert slots["A"]["itcm_region"] == [0, text] and slots["B"]["itcm_region"] == [text, 2 * text]
  assert slots["A"]["dtcm_region"] == [0x10000, 0x10000 + data]
  assert {name: slot["program_id"] for name, slot in slots.items()} == {"A": 1, "B": 2}


def test_program_identities_follow_the_contract_mapping(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  text = skb.CONTRACT.read_text()
  needle = "slot_program_id: { A: 1, B: 2 }"
  assert needle in text
  remapped = tmp_path / "contract.yaml"
  remapped.write_text(text.replace(needle, "slot_program_id: { A: 7, B: 3 }"))
  monkeypatch.setattr(skb, "CONTRACT", remapped)
  slots = {slot["name"]: slot["program_id"] for slot in skb.slot_table(config(), skb.scalar_spec())}
  assert slots == {"A": 7, "B": 3}
  remapped.write_text(text.replace(needle, "slot_program_id: { A: 1, B: 1 }"))
  refuse("BAD_CONTRACT", skb.slot_table, config(), skb.scalar_spec())


BINDING_REFUSALS = [
  ("layout-string", lambda b: b.__setitem__("layout", "A"), "BAD_LAYOUT"),
  ("layout-one-slot", lambda b: b.__setitem__("layout", ["A"]), "BAD_LAYOUT"),
  ("layout-reordered", lambda b: b.__setitem__("layout", ["B", "A"]), "BAD_LAYOUT"),
  ("other-kernel", lambda b: b.__setitem__("kernel", "bootstrap"), "BAD_BINDING"),
  ("other-version", lambda b: b["manifest_kernel"].__setitem__("kernel_version", 2), "BAD_BINDING"),
  ("other-kernel-id", lambda b: b["manifest_kernel"].__setitem__("kernel_id", "K-OTHER"), "BAD_BINDING"),
  ("scratch", lambda b: b["manifest_kernel"].__setitem__("scratch", [{"name": "s"}]), "BAD_BINDING"),
  ("target-mismatch", lambda b: b["target"].__setitem__("context_count", 1), "TARGET_MISMATCH"),
  (
    "no-descriptors",
    lambda b: b["manifest_kernel"].__setitem__("launch_descriptors", []),
    "BAD_DESCRIPTOR",
  ),
  (
    "descriptor-index-not-dense",
    lambda b: b["manifest_kernel"]["launch_descriptors"].reverse(),
    "BAD_DESCRIPTOR",
  ),
  (
    "descriptor-field-missing",
    lambda b: b["manifest_kernel"]["launch_descriptors"][0].pop("frame_offset"),
    "BAD_DESCRIPTOR",
  ),
  (
    "descriptor-wrong-side",
    lambda b: b["manifest_kernel"]["launch_descriptors"][0].__setitem__("direction", "frame_to_l2"),
    "BAD_DESCRIPTOR",
  ),
  (
    "descriptor-past-the-view",
    lambda b: b["manifest_kernel"]["launch_descriptors"][0].__setitem__("bytes", 64),
    "BAD_DESCRIPTOR",
  ),
  (
    "descriptor-odd-frame-offset",
    lambda b: b["manifest_kernel"]["launch_descriptors"][1].__setitem__("frame_offset", OUT + 2),
    "BAD_DESCRIPTOR",
  ),
  (
    "descriptor-odd-length",
    lambda b: b["manifest_kernel"]["launch_descriptors"][1].__setitem__("bytes", 30),
    "BAD_DESCRIPTOR",
  ),
  (
    "helper-operand-not-integer",
    lambda b: b["manifest_kernel"]["helpers"][0]["params"].__setitem__("index", "0"),
    "BAD_BINDING",
  ),
  (
    "helper-operand-field-swapped",
    lambda b: b["manifest_kernel"]["helpers"][4].__setitem__("params", {"code": 0}),
    "BAD_BINDING",
  ),
  ("no-helpers", lambda b: b["manifest_kernel"].__setitem__("helpers", []), "BAD_BINDING"),
]


@pytest.mark.parametrize(
  ("edit", "code"), [case[1:] for case in BINDING_REFUSALS], ids=[c[0] for c in BINDING_REFUSALS]
)
def test_binding_inputs_that_break_the_link_or_the_audit_are_refused(edit: Any, code: str) -> None:
  doc = binding()
  edit(doc)
  refuse(code, skb.plan, doc, CONFIG)


@pytest.mark.parametrize(
  ("key", "value"),
  [("program_slots", 1), ("itcm_bytes", "8192"), ("user_spm_per_bank", 0), ("context_count", 2.0)],
)
def test_a_target_that_cannot_hold_the_declared_slots_is_refused(key: str, value: Any) -> None:
  cfg = config()
  cfg["constraints"][key] = value
  refuse("BAD_CONFIG", skb.plan, BINDING, cfg)


def test_a_target_whose_bank_split_disagrees_with_the_l1_geometry_is_refused() -> None:
  cfg = config()
  cfg["constraints"]["user_spm_per_bank"] -= 1
  refuse("BANK_GEOMETRY_MISMATCH", skb.plan, BINDING, cfg)


# ---------------------------------------------------------------------------------------------------
# link-stage refusals: the two scripts are independent inputs (no toolchain)
# ---------------------------------------------------------------------------------------------------
def test_the_committed_link_scripts_state_exactly_their_slots() -> None:
  for slot in skb.plan(BINDING, CONFIG)["slots"]:
    script = skb.SOURCE / f"slot_{slot['name'].lower()}.ld"
    stated = skb.link_regions(script.read_text())
    assert stated["ITCM"] == (slot["text_base"], slot["text_bytes_per_slot"])
    assert stated["DTCM"] == (slot["data_base"], slot["data_bytes_per_slot"])
    skb.check_link_script(script.read_text(), slot, script)


SCRIPT_REFUSALS = [
  ("slot-b-inside-slot-a", {"slot_b.ld": ("ORIGIN = 0x00001000", "ORIGIN = 0x00000080")}),
  ("slot-a-claims-the-whole-itcm", {"slot_a.ld": ("LENGTH = 0x00001000", "LENGTH = 0x00002000")}),
  ("slot-b-data-on-slot-a-data", {"slot_b.ld": ("ORIGIN = 0x00014000", "ORIGIN = 0x00010000")}),
  (
    "region-declared-twice",
    {
      "slot_a.ld": (
        "SECTIONS {",
        "MEMORY { ITCM (rx) : ORIGIN = 0x00000000, LENGTH = 0x00001000 }\nSECTIONS {",
      )
    },
  ),
  ("slot-script-without-itcm", {"slot_a.ld": ("  ITCM (rx)  : ORIGIN", "  XTCM (rx)  : ORIGIN")}),
]


@pytest.mark.parametrize(
  ("edits",), [case[1:] for case in SCRIPT_REFUSALS], ids=[c[0] for c in SCRIPT_REFUSALS]
)
def test_a_link_script_that_does_not_state_its_slot_is_refused_before_linking(
  tmp_path: Path, edits: dict[str, tuple[str, str]]
) -> None:
  refuse("LINK_SCRIPT_MISMATCH", skb.plan, BINDING, CONFIG, kernel_copy(tmp_path, **edits))


def test_a_missing_link_script_or_source_is_a_refusal(tmp_path: Path) -> None:
  kernel_dir = kernel_copy(tmp_path)
  (kernel_dir / "slot_b.ld").unlink()
  refuse("LINK_SCRIPT_MISSING", skb.plan, BINDING, CONFIG, kernel_dir)
  shutil.rmtree(kernel_dir)
  refuse("SOURCE_MISSING", skb.plan, BINDING, CONFIG, kernel_dir)


def test_a_helper_csr_named_by_a_numeric_literal_in_the_source_is_refused_before_the_build(
  tmp_path: Path,
) -> None:
  # The substituted source assembles to the same bytes, so only the source scan can see it, and
  # measure() reaches that scan before any tool runs.
  for index, (name, helper) in enumerate(helpers().items()):
    kernel_dir = kernel_copy(tmp_path / f"literal-{name}")
    source = kernel_dir / f"{skb.KERNEL}.S"
    text = source.read_text()
    assert helper["macro"] in text, name
    literal = str(helper["address"]) if index % 2 else f"{helper['address']:#x}"
    source.write_text(text.replace(helper["macro"], literal))
    out = kernel_dir.parent / "out"
    error = refuse("HAND_WRITTEN_ADDRESS", skb.measure, BINDING, CONFIG, out, source_dir=kernel_dir)
    assert error.detail.startswith(f"{skb.KERNEL}.S:") and helper["csr"] in error.detail
    assert not out.exists()


def test_the_source_scan_reads_the_csr_operand_of_every_csr_form_and_nothing_else() -> None:
  nexus = skb.helper_csrs()["nexus"]
  address = next(iter(nexus))
  path = Path(f"{skb.KERNEL}.S")
  refused = [
    (1, f"  csrrwi x0, {address:#x}, 0\n"),  # three-operand form
    (1, f"  csrwi {address:#x}, 0\n"),  # two-operand alias
    (1, f"  csrrw x0, {address:#05x}, t0\n"),
    (1, f"  csrs {address}, t1\n"),  # decimal spelling
    (2, f"  /* a comment */\n  csrw {address:#x}, t0\n"),
  ]
  for line, text in refused:
    error = refuse("HAND_WRITTEN_ADDRESS", skb.check_source, text, nexus, path)
    assert error.detail.startswith(f"{path.name}:{line}:")
  accepted = [
    f"  csrrwi x0, CN_CSR_NEXUS_MFE_SUBMIT, {address & 31}\n",  # the immediate is a number, the CSR is not
    "  csrr t1, fflags\n  csrrw x0, fflags, x0\n",  # the standard flags register keeps its symbolic name
    f"  /* csrw {address:#x}, t0: a commented-out literal */\n  lui t0, 0x800\n",
    "  csrw 0x001, x0\n",  # not a generated Nexus address: the audit of the linked bytes decides
  ]
  for text in accepted:
    skb.check_source(text, nexus, path)
  skb.check_source((skb.SOURCE / f"{skb.KERNEL}.S").read_text(), nexus, path)


def test_the_scripts_follow_a_changed_target_never_the_other_way(tmp_path: Path) -> None:
  cfg = config()
  cfg["constraints"]["program_slots"] = 4  # ITCM slots shrink to 2048 B: the committed scripts are stale
  refuse("LINK_SCRIPT_MISMATCH", skb.plan, BINDING, cfg, skb.SOURCE)
  halved = {
    "slot_a.ld": (
      "ITCM (rx)  : ORIGIN = 0x00000000, LENGTH = 0x00001000\n"
      "  DTCM (rwx) : ORIGIN = 0x00010000, LENGTH = 0x00004000",
      "ITCM (rx)  : ORIGIN = 0x00000000, LENGTH = 0x00000800\n"
      "  DTCM (rwx) : ORIGIN = 0x00010000, LENGTH = 0x00002000",
    ),
    "slot_b.ld": (
      "ITCM (rx)  : ORIGIN = 0x00001000, LENGTH = 0x00001000\n"
      "  DTCM (rwx) : ORIGIN = 0x00014000, LENGTH = 0x00004000",
      "ITCM (rx)  : ORIGIN = 0x00000800, LENGTH = 0x00000800\n"
      "  DTCM (rwx) : ORIGIN = 0x00012000, LENGTH = 0x00002000",
    ),
  }
  prepared = skb.plan(BINDING, cfg, kernel_copy(tmp_path, **halved))
  assert [(slot["text_base"], slot["data_base"]) for slot in prepared["slots"]] == [
    (0, 0x10000),
    (0x800, 0x12000),
  ]


def test_overlapping_slot_images_are_refused_and_adjacent_ones_coexist() -> None:
  first = {"name": "A", "text_base": 0, "text_bytes": 144, "data_base": 0x10000, "data_bytes": 0}
  second = dict(first, name="B", text_base=0x1000, data_base=0x14000)
  skb.check_coexist([first, second])
  skb.check_coexist([first, dict(second, text_base=144)])  # B right after A's text
  skb.check_coexist([dict(first, text_bytes=0x1000), second])  # A fills its slot up to B
  skb.check_coexist([dict(second, data_base=0x10000)])  # empty data ranges cannot overlap
  skb.check_coexist(
    [dict(first, data_bytes=0x4000), dict(second, data_base=0x14000)]  # A fills its DTCM slot
  )
  refuse("IMAGES_OVERLAP", skb.check_coexist, [first, dict(second, text_base=0x80)])
  refuse("IMAGES_OVERLAP", skb.check_coexist, [dict(first, text_bytes=0x1004), second])
  refuse(
    "IMAGES_OVERLAP",
    skb.check_coexist,
    [dict(first, data_bytes=0x4000), dict(second, data_base=0x13FFC, data_bytes=16)],
  )


# ---------------------------------------------------------------------------------------------------
# per-bank capacity (no toolchain)
# ---------------------------------------------------------------------------------------------------
def test_a_frame_that_cannot_fit_one_bank_is_refused() -> None:
  doc = binding()
  oversized_frame(doc, 16000, 63968)  # a 64000 B live envelope per Context: two exceed one bank
  error = refuse("BANK_CAPACITY", skb.plan, doc, CONFIG)
  assert "129024" in error.detail  # 2 Contexts x 64512 B (63 stripe rounds)


def test_the_per_bank_bound_counts_every_contexts_reservation_on_one_bank() -> None:
  doc = binding()
  oversized_frame(doc, 4000, 8000)  # 16000 B live, 16384 B reserved per Context
  banks = skb.plan(doc, CONFIG)["banks"]
  assert (banks["frame_live_bytes_per_context"], banks["frame_reservation_bytes_per_context"]) == (
    16000,
    16384,
  )
  assert (banks["worst_case_bytes_on_one_bank"], banks["per_bank_bytes_striped"]) == (32768, 2048)
  banks = skb.plan(BINDING, CONFIG)["banks"]
  assert banks["stripe_round_bytes"] == 1024
  assert banks["frame_live_bytes_per_context"] == OUT + 32
  assert banks["frame_reservation_bytes_per_context"] == 2048
  assert (banks["worst_case_bytes_on_one_bank"], banks["per_bank_bytes_striped"]) == (4096, 256)
  assert banks["user_bytes_left_per_bank"] == banks["user_bytes_per_bank"] - 4096
  assert banks["user_bytes_per_bank"] + banks["system_reserved_bytes_per_bank"] == banks["bytes_per_bank"]


# ---------------------------------------------------------------------------------------------------
# the audit on hand-encoded words (no toolchain)
# ---------------------------------------------------------------------------------------------------
HELPER_CALLS = [
  ("mfe_submit", 0),
  ("wait", 0),
  ("mfe_submit", 1),
  ("wait", 1),
  ("signal_phase", 0),
  ("signal_phase", 1),
  ("task_ret", 0),
]


def test_the_kernel_words_pass_every_rule_and_prove_the_bounds() -> None:
  result = audited(program())
  assert result["instruction_count"] == 36
  assert (result["stack_bytes"], result["spill_bytes"]) == (0, 0)
  assert result["clobber"] == CLOBBER
  assert [(call["helper"], call["operand"]) for call in result["helpers"]] == HELPER_CALLS
  assert [call["offset"] for call in result["helpers"]] == [0, 4, 124, 128, 132, 136, 140]
  assert {call["form"] for call in result["helpers"]} == {"csrrwi x0,csr,imm"}


def test_the_register_form_of_a_helper_needs_a_statically_known_operand() -> None:
  wait = helpers()["wait"]["address"]
  words = program()
  words[WAIT0] = csr(CSRRW, X0, T1, wait)  # t1 holds nothing proven here
  refuse("OPERAND_NOT_STATIC", audited, words)
  words[WAIT0 : WAIT0 + 2] = [lui(T1, 0), addi(T1, T1, 0), csr(CSRRW, X0, T1, wait)]
  call = audited(words)["helpers"][1]
  assert (call["helper"], call["operand"], call["form"]) == ("wait", 0, "csrrw x0,csr,rs1")
  words[WAIT0 + 1] = addi(T1, T1, 1)  # the tracked value is used, not assumed: wait(1) before submit(1)
  refuse("WAIT_WITHOUT_SUBMIT", audited, words)


NOT_SCALAR = [
  0x02000457,  # a V-extension opcode
  0x00000063,  # beq
  0x0000006F,  # jal
  0x00008067,  # ret (jalr)
  0x00000033,  # add: the OP opcode is not part of the frozen subset
  0x0000001B,  # mul (RV32M)
  0x0000000B,  # custom-0
  0x0000005B,  # custom-1
  0x0000002F,  # atomic
  0x00000003,  # lb: only lw is in the subset
  0x00000073,  # ecall
  0x30200073,  # mret
  0x10500073,  # wfi
  0x000000FF,
  0x0001,  # a 16-bit instruction
  0xFFFFFFFF,
]


@pytest.mark.parametrize("word", NOT_SCALAR, ids=lambda word: f"{word:08x}")
def test_any_instruction_outside_the_frozen_scalar_subset_is_refused(word: int) -> None:
  words = program()
  words[LUI] = word
  refuse("INSTRUCTION_NOT_ALLOWED", audited, words)


AUDIT_REFUSALS = [
  ("helper-with-rd", SUBMIT0, lambda a: csr(CSRRWI, T0, 0, a["mfe_submit"]), "HELPER_ACCESS_FORM"),
  ("helper-read", WAIT0, lambda a: csr(CSRRS, T0, 0, a["wait"]), "HELPER_ACCESS_FORM"),
  ("helper-csrrs-write", PHASE0, lambda a: csr(CSRRS, X0, T0, a["signal_phase"]), "HELPER_ACCESS_FORM"),
  (
    "undeclared-nexus-csr",
    SUBMIT0,
    lambda a: csr(CSRRWI, X0, 0, nexus_address("nexus.mregion.start")),
    "UNKNOWN_HELPER",
  ),
  ("foreign-csr", SUBMIT0, lambda a: csr(CSRRWI, X0, 0, 0x300), "CSR_NOT_ALLOWED"),
  ("fflags-immediate-form", PHASE1, lambda a: csr(CSRRWI, T0, 1, FFLAGS), "CSR_NOT_ALLOWED"),
  ("submit-out-of-domain", SUBMIT0, lambda a: csr(CSRRWI, X0, 2, a["mfe_submit"]), "OPERAND_OUT_OF_DOMAIN"),
  ("wait-out-of-domain", WAIT0, lambda a: csr(CSRRWI, X0, 5, a["wait"]), "OPERAND_OUT_OF_DOMAIN"),
  ("phase-out-of-domain", PHASE1, lambda a: csr(CSRRWI, X0, 5, a["signal_phase"]), "OPERAND_OUT_OF_DOMAIN"),
  ("ret-code-nonzero", RET, lambda a: csr(CSRRWI, X0, 1, a["task_ret"]), "OPERAND_OUT_OF_DOMAIN"),
  ("phase-repeated", PHASE1, lambda a: csr(CSRRWI, X0, 0, a["signal_phase"]), "PHASE_REPEATED"),
  ("wait-before-submit", WAIT0, lambda a: csr(CSRRWI, X0, 1, a["wait"]), "WAIT_WITHOUT_SUBMIT"),
  (
    "descriptor-resubmitted",
    WAIT1,
    lambda a: csr(CSRRWI, X0, 0, a["mfe_submit"]),
    "DESCRIPTOR_RESUBMITTED",
  ),
  ("output-ready-early", WAIT1, lambda a: addi(X0, X0, 0), "OUTPUT_READY_EARLY"),
  ("load-before-wait", WAIT0, lambda a: addi(X0, X0, 0), "FRAME_ACCESS_VIOLATION"),
  ("store-outside-descriptors", FIRST_STORE, lambda a: sw(T0, A0, OUT + 32), "FRAME_ACCESS_VIOLATION"),
  ("store-into-the-input-buffer", FIRST_STORE, lambda a: sw(T0, A0, 0), "FRAME_ACCESS_VIOLATION"),
  ("store-not-frame-relative", FIRST_STORE, lambda a: sw(T0, T1, OUT), "FRAME_ACCESS_VIOLATION"),
  ("load-not-frame-relative", FIRST_LOAD, lambda a: lw(T0, T1, 0), "FRAME_ACCESS_VIOLATION"),
  ("load-misaligned", FIRST_LOAD, lambda a: lw(T0, A0, 2), "FRAME_ACCESS_VIOLATION"),
  ("stack-used", LUI, lambda a: addi(SP, SP, -16), "STACK_USED"),
  ("stack-read", LUI, lambda a: addi(T0, SP, 0), "STACK_USED"),
  ("return-address-used", LUI, lambda a: addi(RA, X0, 0), "STACK_USED"),
  ("frame-base-clobbered", LUI, lambda a: addi(A0, A0, 4), "FRAME_BASE_CLOBBERED"),
  ("frame-base-loaded", LUI, lambda a: lw(A0, A0, 0), "FRAME_BASE_CLOBBERED"),
  ("clobber-undeclared", LUI, lambda a: addi(S0, X0, 0), "CLOBBER_MISMATCH"),
  ("fmul-dynamic-rounding", FMUL, lambda a: fmul(FT1, FT0, FT0, rm=7), "FRM_DYN"),
]


@pytest.mark.parametrize(
  ("index", "edit", "code"), [case[1:] for case in AUDIT_REFUSALS], ids=[case[0] for case in AUDIT_REFUSALS]
)
def test_helper_abi_and_frame_violations_on_the_executed_bytes_are_refused(
  index: int, edit: Any, code: str
) -> None:
  words = program()
  words[index] = edit(addresses())
  refuse(code, audited, words)


def _swap(words: list[int], first: int, second: int) -> None:
  words[first], words[second] = words[second], words[first]


STRUCTURAL_REFUSALS = [
  ("ret-not-last", lambda w: _swap(w, PHASE1, RET), "RET_NOT_LAST"),
  ("ret-duplicated", lambda w: w.insert(RET, w[RET]), "RET_NOT_LAST"),
  ("ret-missing", lambda w: w.pop(), "RET_MISSING"),
  ("phases-swapped", lambda w: _swap(w, PHASE0, PHASE1), "HELPER_SEQUENCE"),
  ("store-after-submit", lambda w: w.insert(SUBMIT1 + 1, sw(T0, A0, OUT)), "FRAME_ACCESS_VIOLATION"),
  ("load-after-release", lambda w: w.insert(PHASE1, lw(T0, A0, 0)), "LOAD_AFTER_RELEASE"),
]


@pytest.mark.parametrize(
  ("edit", "code"),
  [case[1:] for case in STRUCTURAL_REFUSALS],
  ids=[case[0] for case in STRUCTURAL_REFUSALS],
)
def test_structural_violations_of_the_task_chain_are_refused(edit: Any, code: str) -> None:
  words = program()
  edit(words)
  refuse(code, audited, words)


def test_the_binding_sequence_cannot_authorize_a_different_image() -> None:
  # the image signals output_ready first. No ABI rule is broken, so only the comparison with the binding's
  # own sequence refuses it; a binding that declares that other order accepts it.
  descriptors, expected = skb.check_binding(binding(), config(), skb.scalar_spec())
  words = program()
  _swap(words, PHASE0, PHASE1)
  body = packed(words)
  refuse("HELPER_SEQUENCE", skb.audit, body, 0, descriptors, expected, skb.helper_csrs(), CLOBBER)
  other = [*expected[:4], expected[5], expected[4], *expected[6:]]
  skb.audit(body, 0, descriptors, other, skb.helper_csrs(), CLOBBER)


def test_the_clobber_declaration_is_exactly_what_the_image_writes() -> None:
  words = program()
  words[LUI] = addi(S0, X0, 0)
  audited(words, clobber=[*CLOBBER, "s0"])
  refuse("CLOBBER_MISMATCH", audited, words)  # s0 is written but not declared
  refuse("CLOBBER_MISMATCH", audited, program(), clobber=["t0"])  # declared, but not all that is written


def test_helper_names_come_from_the_abi_and_the_sources_type_no_csr_number() -> None:
  assert {helper["csr"] for helper in helpers().values()} == {
    "nexus.mfe.submit",
    "nexus.event.wait",
    "nexus.phase.signal",
    "nexus.task.ret",
  }
  for helper in helpers().values():
    assert helper["macro"] == f"CN_CSR_{skb._ident(helper['csr'])}"
  for name in (f"{skb.KERNEL}.S", "slot_a.ld", "slot_b.ld"):
    assert not re.search(r"0x7[c-f][0-9a-f]+\b", (skb.SOURCE / name).read_text(), re.I), name


# ---------------------------------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------------------------------
def test_the_section_and_elf_readers_refuse_what_they_cannot_read() -> None:
  report = (
    "  [Nr] Name              Type            Address          Off      Size   ES Flg Lk Inf Al\n"
    "  [ 1] .text             PROGBITS        00001000 001000 000090 00  AX  0   0 16\n"
    "  [ 2] .text._start.long PROGBITS        00001090 001090 000004 00  AX  0   0  4\n"
    "  [ 3] .riscv.attributes RISCV_ATTRIBUTES 00000000 001094 000049 00     0   0  1\n"
  )
  assert skb.section_table(report) == {
    ".text": {"addr": 0x1000, "size": 144},
    ".text._start.long": {"addr": 0x1090, "size": 4},
    ".riscv.attributes": {"addr": 0, "size": 73},
  }
  assert skb.section_table("There are 2 section headers, starting at offset 0x0:") == {}
  for blob in (b"", b"\x7fELF" + bytes(47), bytes(60), b"\x7fELF\x01\x01" + bytes(70)):
    refuse("NOT_ELF", skb.elf_segments, blob)


# ---------------------------------------------------------------------------------------------------
# the real build (pinned toolchain)
# ---------------------------------------------------------------------------------------------------
@needs_toolchain
def test_both_fixed_slots_are_real_images_at_the_sealed_addresses(built: dict[str, Any]) -> None:
  assert [layout["name"] for _, layout, _ in built["images"]] == ["A", "B"]
  rows = {row["name"]: row for row in built["freeze"]["slots"]}
  program_ids = skb.scalar_spec()["linked_slot"]["slot_program_id"]
  for record, layout, elf_path in built["images"]:
    assert elf_path == built["out"] / layout["name"] / skb.IMAGE and elf_path.is_file()
    header = subprocess.run(
      [str(mk.TOOLCHAIN / "riscv64-unknown-elf-readelf"), "-h", str(elf_path)],
      capture_output=True,
      text=True,
      check=True,
    ).stdout
    assert "ELF32" in header and "RISC-V" in header
    segments = skb.elf_segments(elf_path.read_bytes())
    assert segments["entry"] == layout["entry"] == layout["text_base"]
    assert [segment["vaddr"] for segment in segments["text"]] == [layout["text_base"]]
    assert segments["data"] == []
    assert {key: record[key] for key in skb.ROW_KEYS} == rows[layout["name"]]
    assert all(record[key] == layout[key] for key in layout)
    assert layout["program_id"] == program_ids[layout["name"]]
    assert layout["text_bytes"] == len(segments["text"][0]["data"]) == 144
    assert layout["elf"] == f"{layout['name']}/{skb.IMAGE}"
  layouts = [layout for _, layout, _ in built["images"]]
  assert [layout["text_base"] for layout in layouts] == [0, 0x1000]
  assert [layout["entry"] for layout in layouts] == [0, 0x1000]
  assert [layout["data_base"] for layout in layouts] == [0x10000, 0x14000]
  assert [layout["data_bytes"] for layout in layouts] == [0, 0]


@needs_toolchain
def test_resources_come_from_the_linked_image_and_its_bytes(built: dict[str, Any]) -> None:
  for record, layout, elf_path in built["images"]:
    report = (elf_path.parent / f"{skb.KERNEL}.sections").read_text()
    assert skb.section_table(report)[".text"] == {"addr": layout["text_base"], "size": 144}
    assert record["instruction_count"] == record["retire_count"] == 36
    assert (record["stack_bytes"], record["scratch_bytes"], record["spill_bytes"]) == (0, 0, 0)
    resource = record["resource_manifest"]
    assert resource == {
      "bound_source": "static_analysis",
      "scratch_bytes": 0,
      "stack_bytes": 0,
      "static_data_bytes": 0,
      "text_bytes": 144,
    }
    assert record["provenance"]["source_sha256"] == {
      path.name: hashlib.sha256(path.read_bytes()).hexdigest()
      for path in sorted(skb.SOURCE.iterdir())
      if path.suffix in (".S", ".ld")
    }
    assert record["provenance"]["toolchain_lock_sha256"] == mk._lock_sha()
  assert built["freeze"]["resource_manifest"] == {
    "bound_source": "static_analysis",
    "frame_live_bytes": OUT + 32,
    "frame_reservation_bytes": 2048,
    "scratch_bytes": 0,
    "spill_bytes": 0,
    "stack_bytes": 0,
    "static_data_bytes": 0,
    "text_bytes": 144,
  }


@needs_toolchain
def test_the_hand_encoding_is_the_linked_image(built: dict[str, Any]) -> None:
  for _, _, elf_path in built["images"]:
    assert skb.elf_segments(elf_path.read_bytes())["text"][0]["data"] == packed(program())
  # the fflags path squares the smallest normal: `lui t0, 0x800` loads 0x00800000 = 2^-126
  assert program()[LUI] == 0x008002B7


@needs_toolchain
def test_the_slots_do_not_overlap_and_carry_identical_text(built: dict[str, Any]) -> None:
  freeze = built["freeze"]
  skb.check_coexist(freeze["slots"])
  first, second = (skb.elf_segments(elf.read_bytes())["text"][0]["data"] for _, _, elf in built["images"])
  assert first == second
  assert freeze["instruction_storage"] == {
    "free_bytes": 8192 - 2 * 144,
    "itcm_bytes": 8192,
    "used_by_frozen_slots": 2 * 144,
  }
  assert "checks" not in freeze and "fits" not in freeze["instruction_storage"]


@needs_toolchain
def test_the_linked_image_touches_only_the_helper_csrs_and_fflags(built: dict[str, Any]) -> None:
  helper_addresses = {helper["address"] for helper in helpers().values()}
  for _, _, elf_path in built["images"]:
    body = skb.elf_segments(elf_path.read_bytes())["text"][0]["data"]
    used = {word >> 20 for (word,) in struct.iter_unpack("<I", body) if word & 0x7F == 0x73}
    assert used == helper_addresses | {FFLAGS}


@needs_toolchain
def test_the_published_golden_bytes_are_the_hand_computable_case(built: dict[str, Any]) -> None:
  out, freeze = built["out"], built["freeze"]
  values = list(range(1, 9))
  assert (out / skb.GOLDEN["input"]).read_bytes() == struct.pack("<8i", *values)
  assert (out / skb.GOLDEN["output"]).read_bytes() == struct.pack("<8i", *(v + 7 for v in values))
  for side in ("input", "output"):
    entry = freeze["golden"][side]
    assert (entry["file"], entry["view"], entry["slot_offset"]) == (skb.GOLDEN[side], side, 0)
    assert entry["sha256"] == hashlib.sha256((out / entry["file"]).read_bytes()).hexdigest()
    assert (entry["bytes"], entry["values"]) == (32, values if side == "input" else [v + 7 for v in values])
  assert freeze["golden"]["output"]["values"] == list(range(8, 16))
  assert (freeze["golden"]["dtype"], freeze["golden"]["elements"]) == ("i32", 8)


@needs_toolchain
def test_the_freeze_and_the_images_do_not_depend_on_where_they_were_built(
  built: dict[str, Any], tmp_path: Path
) -> None:
  elsewhere = tmp_path / "a much deeper" / "stage"
  images = skb.measure(BINDING, CONFIG, elsewhere)
  assert (elsewhere / skb.FREEZE).read_bytes() == (built["out"] / skb.FREEZE).read_bytes()
  for (_, layout, _elf), (_, again, elf_again) in zip(built["images"], images, strict=True):
    assert layout == again
    blob = elf_again.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == layout["image_sha256"]
    assert str(elsewhere).encode() not in blob and str(built["out"]).encode() not in blob
  refuse("PATH_IN_IMAGE", skb.check_relocatable, b"prefix" + str(elsewhere).encode(), elsewhere)


@needs_toolchain
def test_mapping_inputs_give_the_same_freeze_as_paths(built: dict[str, Any], tmp_path: Path) -> None:
  out = tmp_path / "from-documents"
  skb.measure(binding(), config(), out)
  assert (out / skb.FREEZE).read_bytes() == (built["out"] / skb.FREEZE).read_bytes()


@needs_toolchain
def test_a_kernel_that_overflows_its_slot_is_refused(tmp_path: Path) -> None:
  kernel_dir = kernel_copy(tmp_path)
  source = kernel_dir / f"{skb.KERNEL}.S"
  ret = "  csrrwi x0, CN_CSR_NEXUS_TASK_RET, 0\n"
  text = source.read_text()
  assert ret in text
  source.write_text(text.replace(ret, "  .rept 1100\n  nop\n  .endr\n" + ret))  # 4544 B > 4096 B
  out = tmp_path / "overflow"
  error = refuse("TEXT_EXCEEDS_SLOT", skb.measure, BINDING, CONFIG, out, source_dir=kernel_dir)
  assert "4544" in error.detail
  assert not (out / skb.FREEZE).exists() and not (out / skb.GOLDEN["output"]).exists()


@needs_toolchain
def test_a_kernel_with_undeclared_bytes_is_refused(tmp_path: Path) -> None:
  section = '  .section .text._start,"ax",@progbits'
  kernel_dir = kernel_copy(
    tmp_path,
    **{f"{skb.KERNEL}.S": (section, '  .section .rodata,"a",@progbits\n  .word 0x2a\n\n' + section)},
  )
  out = tmp_path / "rodata"
  error = refuse("UNDECLARED_DATA", skb.measure, BINDING, CONFIG, out, source_dir=kernel_dir)
  assert ".rodata" in error.detail
  assert not (out / skb.FREEZE).exists()


@needs_toolchain
def test_no_toolchain_is_a_refusal_not_a_fake_measurement(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(mk, "TOOLCHAIN", tmp_path / "nowhere")
  refuse("NO_TOOLCHAIN", skb.measure, BINDING, CONFIG, tmp_path / "out")
  assert not (tmp_path / "out" / skb.FREEZE).exists()
