"""Build, audit and measure the real scalar control kernel images (C038 / W5-02).

``measure(binding, config, out)`` links K-SCALAR-ADD7 (``kernels/library/scalar_ctl_v1``) into both fixed
pre-linked slots with the pinned C004 toolchain and returns ``[(record, layout, elf_path), ...]`` for
``cn package`` to seal. Every number comes from the linked ELF or from a proof over the bytes that will
execute; nothing is estimated:

* text / entry: the executable PT_LOAD of the real ELF (its program headers, the bytes the Loader reads);
* stack / spill: 0 by proof. The bytes are decoded with the public RV32 base fields and every instruction
  must belong to the frozen scalar subset: no call, branch, jump, vector or matrix instruction, no
  compressed instruction, no access to ra or sp, and every load or store is 4-byte aligned relative to the
  launch Frame register a0 and inside a descriptor the binding declares (read after its WAIT, written
  before its submit);
* helper CSRs: named by ``protocol/scalar_helper_abi_v1.json`` and numbered only by the C017-generated
  ``cn_layout.json`` / ``cn_messages.h``. A helper CSR written in any other form, an undeclared Nexus CSR,
  an operand outside its domain, or a helper CSR named by a direct numeric literal in the kernel source
  (ABI fault HAND_WRITTEN_ADDRESS: the CSR operand of every CSR instruction is scanned before the build,
  which is not an assembler-level analysis) is a build refusal;
* frame / per bank: the live Frame envelope is the declared descriptor extent; its reservation is that
  envelope in whole stripe rounds (one cache line per bank), the number the control plane reserves too.
  Under the all-bank striped Arena each bank holds reservation / banks per Context; no sealed input fixes
  the control plane's buffer alignments or a non-striped placement, so the per-bank refusal is the bound
  that holds for any placement: every Context's whole reservation on ONE bank, against the user SPM of a
  bank.

The slot geometry is derived from the sealed target (program_slots, itcm_bytes, dtcm_bytes over the pinned
upstream origins) and each slot identity (name, program_id) from the contract, never from a directory.
The two committed link scripts are independent inputs: each is read back and must state exactly its slot
before anything is linked, and each linked image is then checked again against its own slot.

Every refusal is a ``measure_kernel.MeasureError(code, detail)``. ``slot_freeze.json`` and the golden bytes
are written only when every check passed, and the freeze holds paths relative to ``out``. The image bytes
never depend on the directory they were built in (all tools run on relative names inside the slot's build
directory), so a renamed or published stage stays valid. On a refusal the slot directories that were
already built stay under ``out``; ``cn package`` stages in a temporary directory and publishes nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import yaml

LIB = Path(__file__).resolve().parent
H = LIB.parents[1]
sys.path.insert(0, str(LIB))

import measure_kernel as mk  # noqa: E402

MeasureError = mk.MeasureError

KERNEL = "scalar_ctl_v1"
KERNEL_ID = "K-SCALAR-ADD7"
KERNEL_VERSION = 1
SOURCE = H / "kernels/library" / KERNEL
BINDING = H / "kernels/bindings" / f"{KERNEL}.yaml"
GENERATED = H / "protocol/generated"
HEADER = GENERATED / "cn_messages.h"
LAYOUT = GENERATED / "cn_layout.json"
HELPER_ABI = H / "protocol/scalar_helper_abi_v1.json"
CONTRACT = H / "protocol/w5_package.contract.yaml"
W4_CONSTRAINTS = H / "configs/w4_memory_constraints.yaml"
# The card's ISA set with no V or Zve* extension: nothing RVV or VME can be assembled at all.
MARCH = "rv32imf_zicsr_zifencei_zbb"
MABI = "ilp32"
IMAGE = f"{KERNEL}.elf"
FREEZE = "slot_freeze.json"
GOLDEN = {"input": "golden_input.bin", "output": "golden_output.bin"}
ELEMENTS = 8  # the frozen golden case of the card: 8 x i32 in, x + 7 out
DIRECTIONS = ("l2_to_frame", "frame_to_l2")
DESCRIPTOR_FIELDS = ("index", "direction", "handle", "bytes", "slot_offset", "frame_offset")
DATA_SECTIONS = (".data", ".sdata", ".rodata", ".srodata", ".bss", ".sbss", ".stack")
ROW_KEYS = (
  "name",
  "program_id",
  "entry",
  "text_base",
  "text_bytes",
  "data_base",
  "data_bytes",
  "image_sha256",
  "stack_bytes",
  "scratch_bytes",
  "spill_bytes",
  "instruction_count",
  "retire_count",
  "elf",
  "itcm_region",
  "dtcm_region",
)

# Register numbers of the public RV32 ABI and the standard fflags CSR number (Zicsr). Helper CSR numbers
# are never written here: they come from helper_csrs().
_X0, _RA, _SP, _A0 = 0, 1, 2, 10
_FFLAGS = 0x001
_X_NAMES = (
  "zero ra sp gp tp t0 t1 t2 s0 s1 a0 a1 a2 a3 a4 a5 a6 a7 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 t3 t4 t5 t6"
).split()
_F_NAMES = (
  "ft0 ft1 ft2 ft3 ft4 ft5 ft6 ft7 fs0 fs1 fa0 fa1 fa2 fa3 fa4 fa5 fa6 fa7 "
  "fs2 fs3 fs4 fs5 fs6 fs7 fs8 fs9 fs10 fs11 ft8 ft9 ft10 ft11"
).split()
_CSR_FORMS = {1: "csrrw", 2: "csrrs", 3: "csrrc", 5: "csrrwi", 6: "csrrsi", 7: "csrrci"}
_RE_CSR_DEFINE = re.compile(r"^#define (CN_CSR_NEXUS_[A-Z0-9_]+) (0x[0-9a-f]{3})$", re.M)
_RE_REGION = re.compile(
  r"\b(ITCM|DTCM)\s*\([^)]*\)\s*:\s*ORIGIN\s*=\s*(0x[0-9a-fA-F]+)\s*,\s*LENGTH\s*=\s*(0x[0-9a-fA-F]+)"
)
_RE_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_RE_CSR_INSN = re.compile(
  r"\b(csrrw|csrrs|csrrc|csrrwi|csrrsi|csrrci|csrw|csrs|csrc|csrwi|csrsi|csrci|csrr)\s+([^;\n]*)"
)
_RE_NUMBER = re.compile(r"^(0[xX][0-9a-fA-F]+|\d+)$")
# one line of `readelf -SW`: [Nr] Name Type Address Off Size ...
_RE_SECTION = re.compile(r"^\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", re.M)


def _sha(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _signed(value: int, bits: int) -> int:
  return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _ident(name: str) -> str:
  """The C017 generator's macro rule (gen_protocol.py ``_ident``): a macro name is never typed by hand."""
  return re.sub(r"[^0-9A-Za-z]", "_", name).upper()


def _doc(value: Any, what: str) -> dict[str, Any]:
  """A parsed document, given as a mapping or as the path of a YAML or JSON file."""
  if isinstance(value, dict):
    return value
  try:
    doc = yaml.safe_load(Path(value).read_text())
  except (OSError, TypeError, yaml.YAMLError) as exc:
    raise MeasureError(f"BAD_{what.upper()}", f"{value}: unreadable ({exc})") from exc
  if not isinstance(doc, dict):
    raise MeasureError(f"BAD_{what.upper()}", f"{value}: not a mapping")
  return doc


# ---------------------------------------------------------------------------------------------------
# generated inputs
# ---------------------------------------------------------------------------------------------------
def helper_csrs() -> dict[str, Any]:
  """The scalar helper CSRs by logical name, and every Nexus CSR the C017 allocator issued.

  Three generated artifacts must agree before an address is used: the ABI names each helper's CSR, the
  generated layout holds its address, and the generated header states the same address as a macro.
  """
  try:
    abi = json.loads(HELPER_ABI.read_text())
    layout = {str(entry["name"]): int(entry["address"]) for entry in json.loads(LAYOUT.read_text())["csr"]}
    macros = dict(_RE_CSR_DEFINE.findall(HEADER.read_text()))
  except (OSError, KeyError, TypeError, ValueError) as exc:
    raise MeasureError(
      "GENERATED_CSR_MISSING", f"the generated CSR artifacts are unreadable ({exc})"
    ) from exc
  for name, address in layout.items():
    macro = f"CN_CSR_{_ident(name)}"
    if macro not in macros or int(macros[macro], 16) != address:
      raise MeasureError(
        "HELPER_CSR_MISMATCH",
        f"{macro} is {macros.get(macro)} in cn_messages.h but {address:#05x} in cn_layout.json",
      )
  helpers: dict[str, dict[str, Any]] = {}
  for helper in abi["helpers"]:
    csr = str(helper["csr"])
    if csr not in layout:
      raise MeasureError("GENERATED_CSR_MISSING", f"{helper['name']}: {csr} is not allocated by C017")
    helpers[str(helper["name"])] = {
      "csr": csr,
      "macro": f"CN_CSR_{_ident(csr)}",
      "address": layout[csr],
      "operand": str(helper["operand"]["kind"]),
    }
  return {"helpers": helpers, "nexus": {address: name for name, address in layout.items()}}


def scalar_spec() -> dict[str, Any]:
  """The scalar kernel class rules of the W5 contract: the consumer-side authority."""
  try:
    spec = yaml.safe_load(CONTRACT.read_text())["binding_manifest"]["scalar"]
  except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
    raise MeasureError("BAD_CONTRACT", f"{CONTRACT}: no binding_manifest.scalar block ({exc})") from exc
  for key in ("helper_operand", "launch_descriptors", "linked_slot"):
    if not isinstance(spec, dict) or not isinstance(spec.get(key), dict):
      raise MeasureError("BAD_CONTRACT", f"{CONTRACT}: binding_manifest.scalar.{key} is missing")
  return spec


def _cut_csrs(build: Path) -> None:
  """cn_csrs.inc: every generated Nexus CSR macro, copied verbatim from cn_messages.h for the assembler."""
  lines = [match.group(0) for match in _RE_CSR_DEFINE.finditer(HEADER.read_text())]
  if not lines:
    raise MeasureError("GENERATED_CSR_MISSING", f"{HEADER} defines no CN_CSR_NEXUS_* macro")
  (build / "cn_csrs.inc").write_text(
    "/* GENERATED CSR names, cut verbatim from protocol/generated/cn_messages.h "
    "by scalar_kernel_build.py. */\n" + "\n".join(lines) + "\n"
  )


# ---------------------------------------------------------------------------------------------------
# static inputs: slots, binding, link scripts, per-bank capacity (no toolchain needed)
# ---------------------------------------------------------------------------------------------------
def slot_table(config: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
  """Fixed slot geometry from the sealed target: slot k owns the k-th equal part of ITCM and of DTCM."""
  constraints = config.get("constraints")
  if config.get("kind") != "cs1_target" or not isinstance(constraints, dict):
    raise MeasureError("BAD_CONFIG", "a cs1_target config with constraints is required")
  for key in (
    "context_count",
    "program_slots",
    "itcm_bytes",
    "dtcm_bytes",
    "user_spm_per_bank",
    "system_reserved_spm_per_bank",
  ):
    value = constraints.get(key)
    if type(value) is not int or value <= 0:
      raise MeasureError("BAD_CONFIG", f"constraints.{key} must be a positive integer, got {value!r}")
  linked = spec["linked_slot"]
  names, ids = linked["slot_names"], linked["slot_program_id"]
  if not isinstance(names, list) or not isinstance(ids, dict):
    raise MeasureError("BAD_CONTRACT", "linked_slot needs slot_names and slot_program_id")
  if len(names) < 2 or len(set(names)) != len(names) or constraints["program_slots"] < len(names):
    raise MeasureError(
      "BAD_CONFIG", f"{len(names)} contract slots do not fit program_slots {constraints['program_slots']}"
    )
  program_ids = [ids.get(name) for name in names]
  if any(type(pid) is not int or pid <= 0 for pid in program_ids):
    raise MeasureError("BAD_CONTRACT", f"slot_program_id must be positive for {names}, got {program_ids!r}")
  if len(set(program_ids)) != len(names):
    raise MeasureError("BAD_CONTRACT", f"slot_program_id is not distinct: {program_ids!r}")
  mk._verify_upstream_regions()  # the pinned upstream DTCM origin, read back: never a local guess
  text_slot = constraints["itcm_bytes"] // constraints["program_slots"]
  data_slot = constraints["dtcm_bytes"] // constraints["program_slots"]
  slots = []
  for index, name in enumerate(names):
    text_base = mk.ITCM_ORIGIN + index * text_slot
    data_base = mk.DTCM_ORIGIN + index * data_slot
    slots.append(
      {
        "name": name,
        "program_id": program_ids[index],
        "text_base": text_base,
        "data_base": data_base,
        "text_bytes_per_slot": text_slot,
        "data_bytes_per_slot": data_slot,
        "itcm_region": [text_base, text_base + text_slot],
        "dtcm_region": [data_base, data_base + data_slot],
      }
    )
  return slots


def check_binding(
  binding: dict[str, Any], config: dict[str, Any], spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
  """The launch descriptor table with its Frame byte ranges, and the helper sequence the image must show.

  Only what the link and the audit consume is checked here. The sealed contract is validated again by
  ``cn package`` (binding_ir) when each package is sealed.
  """
  entry = binding.get("manifest_kernel")
  if binding.get("kernel") != KERNEL or not isinstance(entry, dict):
    raise MeasureError("BAD_BINDING", f"kernel must be {KERNEL} with a manifest_kernel entry")
  if binding.get("layout") != spec["linked_slot"]["slot_names"]:
    raise MeasureError("BAD_LAYOUT", f"layout must be exactly {spec['linked_slot']['slot_names']}")
  if (entry.get("kernel_id"), entry.get("kernel_version")) != (KERNEL_ID, KERNEL_VERSION):
    raise MeasureError("BAD_BINDING", f"the sealed kernel is {KERNEL_ID} v{KERNEL_VERSION}")
  if entry.get("scratch") != []:
    raise MeasureError("BAD_BINDING", f"{KERNEL_ID} owns no scratch")
  target = binding.get("target")
  if not isinstance(target, dict) or target.get("context_count") != config["constraints"]["context_count"]:
    raise MeasureError("TARGET_MISMATCH", "the binding's context_count differs from the config")
  views: dict[str, tuple[str, dict[str, Any]]] = {}
  for side in ("ins", "outs"):
    for view in entry.get(side) or []:
      if isinstance(view, dict) and isinstance(view.get("handle"), str):
        views[view["handle"]] = (side, view)
  directions = spec["launch_descriptors"]["directions"]
  if tuple(directions) != DIRECTIONS:
    raise MeasureError("BAD_CONTRACT", f"launch_descriptors.directions must be {DIRECTIONS}")
  table = entry.get("launch_descriptors")
  if not isinstance(table, list) or not table:
    raise MeasureError("BAD_DESCRIPTOR", "the binding declares no launch descriptors")
  descriptors = []
  for index, desc in enumerate(table):
    path = f"launch_descriptors[{index}]"
    if not isinstance(desc, dict) or any(key not in desc for key in DESCRIPTOR_FIELDS):
      raise MeasureError("BAD_DESCRIPTOR", f"{path}: needs {DESCRIPTOR_FIELDS}")
    if desc["index"] != index or desc["direction"] not in directions:
      raise MeasureError(
        "BAD_DESCRIPTOR", f"{path}: index must sit at its position and direction be one of {directions}"
      )
    view = views.get(desc["handle"]) if isinstance(desc["handle"], str) else None
    if view is None or view[0] != ("ins" if desc["direction"] == directions[0] else "outs"):
      raise MeasureError(
        "BAD_DESCRIPTOR", f"{path}.handle: {desc['handle']!r} must name a view of the matching side"
      )
    shape = view[1].get("shape")
    if (
      view[1].get("dtype") != "i32" or not isinstance(shape, list) or not all(type(n) is int for n in shape)
    ):
      raise MeasureError("BAD_DESCRIPTOR", f"{path}.handle: scalar views are i32 with an integer shape")
    extent = 4 * math.prod(shape)
    for key in ("bytes", "slot_offset", "frame_offset"):
      if type(desc[key]) is not int or desc[key] < 0 or desc[key] % 4:
        raise MeasureError("BAD_DESCRIPTOR", f"{path}.{key}: a non-negative whole number of i32")
    if desc["bytes"] == 0 or desc["slot_offset"] + desc["bytes"] > extent:
      raise MeasureError("BAD_DESCRIPTOR", f"{path}: empty, or slot_offset+bytes exceeds the view")
    descriptors.append(
      {**desc, "frame_range": [desc["frame_offset"], desc["frame_offset"] + desc["bytes"]]}
    )
  operand = spec["helper_operand"]
  expected = []
  for helper in entry.get("helpers") or []:
    rule = operand.get(helper.get("name")) if isinstance(helper, dict) else None
    if rule is None or not isinstance(helper.get("params"), dict) or rule["field"] not in helper["params"]:
      raise MeasureError("BAD_BINDING", "helpers must follow binding_manifest.scalar.helper_operand")
    value = helper["params"][rule["field"]]
    if type(value) is not int:
      raise MeasureError("BAD_BINDING", f"{helper['name']}: the operand must be an integer")
    expected.append((helper["name"], value))
  if not expected:
    raise MeasureError("BAD_BINDING", "the binding declares no helpers")
  return descriptors, expected


def bank_proof(config: dict[str, Any], frame_bytes: int) -> dict[str, Any]:
  """Per-bank capacity of the L1 Arena for the per-Context Frames.

  ``frame_bytes`` is the live envelope of one Context's Frame, the highest end of a descriptor. Every
  Arena placement consumes whole all-bank stripe rounds (compiler/resources.py ``layout_buffers``,
  memory/arena.py), so the reservation is that envelope rounded up to a stripe round, one cache line
  per bank, the smallest round the sealed geometry allows; under that striped layout each bank holds
  reservation / banks per Context. Neither the control plane's buffer alignments (the stripe is their
  LCM with the profile alignment) nor a non-striped placement is fixed by a sealed input, so the refusal
  is the bound that holds for any placement: every Context's whole reservation on ONE bank must fit.
  """
  constraints = config["constraints"]
  try:
    inputs = {item["name"]: item["value"] for item in yaml.safe_load(W4_CONSTRAINTS.read_text())["inputs"]}
    banks, capacity, line = (
      inputs["tile_l1_banks"],
      inputs["tile_l1_capacity_bytes"],
      inputs["cache_line_bytes"],
    )
  except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
    raise MeasureError("BAD_CONSTRAINTS", f"{W4_CONSTRAINTS}: unreadable ({exc})") from exc
  if any(type(value) is not int or value <= 0 for value in (banks, capacity, line)):
    raise MeasureError(
      "BAD_CONSTRAINTS", "tile_l1_banks, tile_l1_capacity_bytes and cache_line_bytes must be positive"
    )
  reserved, user = constraints["system_reserved_spm_per_bank"], constraints["user_spm_per_bank"]
  bank_bytes = capacity // banks
  if user + reserved != bank_bytes:
    raise MeasureError(
      "BANK_GEOMETRY_MISMATCH",
      f"user {user} + reserved {reserved} != {bank_bytes} B per bank ({capacity} B L1 / {banks} banks)",
    )
  stripe = banks * line
  reservation = -(-frame_bytes // stripe) * stripe
  contexts = constraints["context_count"]
  worst = contexts * reservation
  proof = {
    "banks": banks,
    "bytes_per_bank": bank_bytes,
    "system_reserved_bytes_per_bank": reserved,
    "user_bytes_per_bank": user,
    "context_count": contexts,
    "stripe_round_bytes": stripe,
    "frame_live_bytes_per_context": frame_bytes,
    "frame_reservation_bytes_per_context": reservation,
    "per_bank_bytes_striped": contexts * reservation // banks,
    "worst_case_bytes_on_one_bank": worst,
    "user_bytes_left_per_bank": user - worst,
    "rule": (
      "the reservation is the live envelope rounded up to a stripe round (tile_l1_banks x "
      "cache_line_bytes); under the all-bank striped Arena each bank holds reservation / banks per "
      "Context (per_bank_bytes_striped); the refusal uses the bound that holds for any placement: "
      "context_count * frame_reservation_bytes_per_context <= user_bytes_per_bank"
    ),
  }
  if worst > user:
    raise MeasureError(
      "BANK_CAPACITY",
      f"{contexts} Contexts x {reservation} B Frame reservation = {worst} B on one bank "
      f"> {user} B user SPM per bank",
    )
  return proof


def link_regions(text: str) -> dict[str, tuple[int, int]]:
  """The MEMORY regions a link script states, read back from its text with comments removed. A region
  is found wherever it is declared, on its own line or inside a one-line MEMORY block."""
  regions: dict[str, tuple[int, int]] = {}
  for match in _RE_REGION.finditer(_RE_COMMENT.sub("", text)):
    if match.group(1) in regions:
      raise MeasureError("LINK_SCRIPT_MISMATCH", f"{match.group(1)} is declared twice")
    regions[match.group(1)] = (int(match.group(2), 16), int(match.group(3), 16))
  return regions


def check_link_script(text: str, slot: dict[str, Any], path: Path) -> None:
  """A slot's link script is an independent input: it must state exactly that slot's two regions."""
  want = {
    "ITCM": (slot["text_base"], slot["text_bytes_per_slot"]),
    "DTCM": (slot["data_base"], slot["data_bytes_per_slot"]),
  }
  stated = link_regions(text)
  for region, expected in want.items():
    if stated.get(region) != expected:
      raise MeasureError(
        "LINK_SCRIPT_MISMATCH",
        f"{path.name}: {region} {stated.get(region)} != slot {slot['name']} {expected}, "
        "the slot the target derives (program_slots, itcm_bytes, dtcm_bytes)",
      )


def check_source(text: str, nexus: dict[int, str], path: Path) -> None:
  """A helper CSR must not be named by a direct numeric literal in the kernel source.

  This is the ABI's static build fault HAND_WRITTEN_ADDRESS. The scan reads the CSR operand of every CSR
  instruction with comments removed; it is not an assembler-level analysis: a number that only appears
  through an alias, a macro or an expression is not seen here and still has to pass the audit of the
  linked bytes. A standard register keeps its symbolic name (``csrr t1, fflags``), and a literal that is
  not a generated Nexus address is left to that audit.
  """
  body = _RE_COMMENT.sub(lambda match: "\n" * match.group(0).count("\n"), text)
  for match in _RE_CSR_INSN.finditer(body):
    operands = [operand.strip() for operand in match.group(2).split(",")]
    csr = operands[1] if len(operands) == 3 else operands[0]
    if _RE_NUMBER.match(csr) and int(csr, 0) in nexus:
      line = body.count("\n", 0, match.start()) + 1
      raise MeasureError(
        "HAND_WRITTEN_ADDRESS",
        f"{path.name}:{line}: {match.group(1)} names {nexus[int(csr, 0)]} by the literal {csr}; "
        "use its generated CN_CSR_NEXUS_* macro",
      )


def plan(binding: Any, config: Any, source_dir: Any = None) -> dict[str, Any]:
  """Validate every static input and derive the slot table. No toolchain is needed for this step."""
  binding, config = _doc(binding, "binding"), _doc(config, "config")
  spec = scalar_spec()
  csrs = helper_csrs()
  slots = slot_table(config, spec)
  descriptors, expected = check_binding(binding, config, spec)
  frame_bytes = max(desc["frame_offset"] + desc["bytes"] for desc in descriptors)
  banks = bank_proof(config, frame_bytes)
  source = SOURCE if source_dir is None else Path(source_dir)
  kernel_source = source / f"{KERNEL}.S"
  if not kernel_source.is_file():
    raise MeasureError("SOURCE_MISSING", f"{kernel_source}: the kernel source")
  check_source(kernel_source.read_text(), csrs["nexus"], kernel_source)
  for slot in slots:
    script = source / f"slot_{slot['name'].lower()}.ld"
    if not script.is_file():
      raise MeasureError("LINK_SCRIPT_MISSING", f"{script}: slot {slot['name']} has no link script")
    check_link_script(script.read_text(), slot, script)
  return {
    "binding": binding,
    "config": config,
    "spec": spec,
    "csrs": csrs,
    "slots": slots,
    "descriptors": descriptors,
    "expected": expected,
    "frame_bytes": frame_bytes,
    "banks": banks,
    "source": source,
  }


# ---------------------------------------------------------------------------------------------------
# the real build
# ---------------------------------------------------------------------------------------------------
def section_table(readelf_sw: str) -> dict[str, dict[str, int]]:
  """Sections of an object or image from ``readelf -SW``: name -> address and size in memory."""
  return {
    match.group(1): {"addr": int(match.group(3), 16), "size": int(match.group(5), 16)}
    for match in _RE_SECTION.finditer(readelf_sw)
  }


def text_bytes_of(table: dict[str, dict[str, int]], slot: dict[str, Any], where: str) -> int:
  """The real text size of an object or image, refused when it carries undeclared bytes or overflows."""
  undeclared = sorted(name for name, section in table.items() if name in DATA_SECTIONS and section["size"])
  if undeclared:
    raise MeasureError(
      "UNDECLARED_DATA", f"{where}: {undeclared} carry bytes; {KERNEL_ID} owns no static data"
    )
  text = sum(section["size"] for name, section in table.items() if name.startswith(".text"))
  if text > slot["text_bytes_per_slot"]:
    raise MeasureError(
      "TEXT_EXCEEDS_SLOT",
      f"{where}: {text} B of text exceed slot {slot['name']} "
      f"({slot['text_bytes_per_slot']} B = itcm_bytes / program_slots)",
    )
  return text


def _build_slot(slot: dict[str, Any], source: Path, out: Path) -> tuple[Path, str, str]:
  """Assemble and link one slot with the pinned toolchain.

  Every tool runs inside the slot's build directory on relative names, so nothing about the build
  directory can reach the image bytes. Returns the ELF path and the two ``readelf -SW`` reports
  (object, linked image).
  """
  build = out / slot["name"]
  build.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(source / f"{KERNEL}.S", build / f"{KERNEL}.S")
  shutil.copyfile(source / f"slot_{slot['name'].lower()}.ld", build / "slot.ld")
  _cut_csrs(build)
  tool = mk.TOOLCHAIN
  obj = f"{KERNEL}.o"
  mk._run(
    [
      str(tool / "clang"),
      "--target=riscv32-unknown-elf",
      f"-march={MARCH}",
      f"-mabi={MABI}",
      "-mcmodel=medany",
      "-ffreestanding",
      "-fno-builtin",
      "-Wa,--no-warn",
      "-c",
      f"{KERNEL}.S",
      "-o",
      obj,
    ],
    cwd=build,
  )
  readelf = str(tool / "riscv64-unknown-elf-readelf")
  object_report = mk._run([readelf, "-SW", obj], cwd=build)
  text_bytes_of(section_table(object_report), slot, f"slot {slot['name']} object")
  try:
    mk._run(
      [
        str(tool / "riscv64-unknown-elf-ld.bfd"),
        "-m",
        "elf32lriscv",
        "-T",
        "slot.ld",
        "-Map",
        f"{KERNEL}.map",
        "-o",
        IMAGE,
        obj,
      ],
      cwd=build,
    )
  except mk.MeasureError as exc:  # ld refused: a coded link failure with ld's own message
    raise MeasureError("LINK_FAILED", f"slot {slot['name']}: {exc.detail}") from exc
  reports = (
    ("dis", [str(tool / "llvm-objdump"), "-d", IMAGE]),
    ("size", [str(tool / "llvm-size"), "-A", IMAGE]),
    ("sections", [readelf, "-SW", "-l", IMAGE]),
  )
  image_report = ""
  for suffix, command in reports:
    report = mk._run(command, cwd=build)
    (build / f"{KERNEL}.{suffix}").write_text(report)
    image_report = report if suffix == "sections" else image_report
  return build / IMAGE, object_report, image_report


def elf_segments(elf: bytes) -> dict[str, Any]:
  """Entry point, executable PT_LOAD and writable PT_LOAD segments of a linked RV32 ELF, from its program
  headers: the bytes that are really installed and executed."""
  if len(elf) < 52 or elf[:4] != b"\x7fELF" or elf[4] != 1 or elf[5] != 1:
    raise MeasureError("NOT_ELF", "not a 32-bit little-endian ELF file")
  if struct.unpack_from("<H", elf, 18)[0] != 243:
    raise MeasureError("NOT_ELF", "not a RISC-V ELF")
  entry, phoff = struct.unpack_from("<II", elf, 24)
  phentsize, phnum = struct.unpack_from("<HH", elf, 42)
  if phentsize < 32 or phoff + phnum * phentsize > len(elf):
    raise MeasureError("NOT_ELF", "the program header table is outside the file")
  segments = []
  for index in range(phnum):
    kind, offset, vaddr, _paddr, filesz, memsz, flags, _align = struct.unpack_from(
      "<IIIIIIII", elf, phoff + index * phentsize
    )
    if kind != 1 or memsz == 0:
      continue
    if filesz > memsz or (filesz and offset + filesz > len(elf)):
      raise MeasureError("NOT_ELF", f"PT_LOAD {index}: file range and memory size disagree")
    segments.append({"vaddr": vaddr, "memsz": memsz, "flags": flags, "data": elf[offset : offset + filesz]})
  return {
    "entry": entry,
    "text": [segment for segment in segments if segment["flags"] & 1],
    "data": [segment for segment in segments if not segment["flags"] & 1 and segment["flags"] & 2],
  }


def check_image(elf: bytes, slot: dict[str, Any], sections_report: str) -> bytes:
  """One linked image against its own slot, independently of the other slot."""
  where = f"slot {slot['name']}"
  segments = elf_segments(elf)
  if len(segments["text"]) != 1:
    raise MeasureError(
      "ELF_LAYOUT_MISMATCH", f"{where}: expected one executable PT_LOAD, found {len(segments['text'])}"
    )
  load = segments["text"][0]
  body = load["data"]
  if segments["entry"] != slot["text_base"]:
    raise MeasureError(
      "ENTRY_MISMATCH", f"{where}: e_entry {segments['entry']:#x} != text base {slot['text_base']:#x}"
    )
  if load["vaddr"] != slot["text_base"] or load["memsz"] != len(body):
    raise MeasureError(
      "VMA_MISMATCH",
      f"{where}: executable segment at {load['vaddr']:#x} is {load['memsz']} B in memory "
      f"and {len(body)} B in the file, text base {slot['text_base']:#x}",
    )
  if segments["data"]:
    raise MeasureError(
      "UNDECLARED_DATA", f"{where}: the image loads a writable segment; {KERNEL_ID} owns no static data"
    )
  if load["vaddr"] + load["memsz"] > slot["itcm_region"][1]:
    raise MeasureError(
      "TEXT_EXCEEDS_SLOT",
      f"{where}: text ends at {load['vaddr'] + load['memsz']:#x}, past the slot's ITCM "
      f"{slot['itcm_region'][1]:#x}",
    )
  table = section_table(sections_report)
  text = text_bytes_of(table, slot, where)
  if table.get(".text", {}).get("addr") != slot["text_base"]:
    raise MeasureError("VMA_MISMATCH", f"{where}: .text is not placed at the slot's text base")
  if text != len(body) or len(body) % 4:
    raise MeasureError(
      "ELF_LAYOUT_MISMATCH", f"{where}: .text is {text} B, the executable segment holds {len(body)} B"
    )
  return body


def check_relocatable(elf: bytes, out: Path) -> None:
  """The image must not depend on the directory it was built in."""
  for needle in {str(out), str(out.resolve())}:
    if needle.encode() in elf:
      raise MeasureError("PATH_IN_IMAGE", f"the image embeds its build directory {needle}")


def check_coexist(rows: list[dict[str, Any]]) -> None:
  """Two fixed pre-linked images may be resident together only if their address ranges are disjoint."""
  for index, first in enumerate(rows):
    for second in rows[index + 1 :]:
      for a_base, a_len in (("text_base", "text_bytes"), ("data_base", "data_bytes")):
        for b_base, b_len in (("text_base", "text_bytes"), ("data_base", "data_bytes")):
          overlap = (
            first[a_base] < second[b_base] + second[b_len] and second[b_base] < first[a_base] + first[a_len]
          )
          if overlap:
            raise MeasureError(
              "IMAGES_OVERLAP",
              f"slot {first['name']} [{first[a_base]:#x}, +{first[a_len]}) and slot {second['name']} "
              f"[{second[b_base]:#x}, +{second[b_len]}) overlap",
            )


# ---------------------------------------------------------------------------------------------------
# the audit: the bytes that will execute
# ---------------------------------------------------------------------------------------------------
def decode(word: int, pc: int) -> dict[str, Any]:
  """Decode one 32-bit instruction of the frozen scalar subset with the public RV32 base fields.

  Anything else (vector, matrix, custom, branch, jump, compressed, any other base op) is refused.
  """
  if word & 0x3 != 0x3:
    raise MeasureError("INSTRUCTION_NOT_ALLOWED", f"{pc:#x}: {word & 0xFFFF:#06x} is a 16-bit instruction")
  opcode = word & 0x7F
  rd, f3 = (word >> 7) & 31, (word >> 12) & 7
  rs1, f7 = (word >> 15) & 31, (word >> 25) & 127
  rs2 = (word >> 20) & 31
  insn: dict[str, Any] = {
    "kind": "",
    "rd": rd,
    "rs1": rs1,
    "rs2": rs2,
    "f3": f3,
    "imm": 0,
    "csr": None,
    "xreads": (),
    "xwrites": (),
    "fwrites": (),
  }
  if opcode == 0x37:
    insn.update(kind="lui", imm=word & 0xFFFFF000, xwrites=(rd,))
  elif opcode == 0x13 and f3 == 0:
    insn.update(kind="addi", imm=_signed(word >> 20, 12), xreads=(rs1,), xwrites=(rd,))
  elif opcode == 0x03 and f3 == 2:
    insn.update(kind="lw", imm=_signed(word >> 20, 12), xreads=(rs1,), xwrites=(rd,))
  elif opcode == 0x23 and f3 == 2:
    imm = _signed(((word >> 25) << 5) | ((word >> 7) & 31), 12)
    insn.update(kind="sw", imm=imm, xreads=(rs1, rs2))
  elif opcode == 0x73 and f3 in _CSR_FORMS:
    insn.update(kind="csr", csr=word >> 20, xwrites=(rd,), xreads=(rs1,) if f3 < 5 else ())
  elif opcode == 0x53 and f7 == 0x08 and f3 == 7:
    # bits 14..12 are the rounding mode of an FP arithmetic op; 111 selects the frm CSR at run time
    raise MeasureError(
      "FRM_DYN", f"{pc:#x}: {word:#010x} is an FP op with a dynamic rounding mode (frm=DYN)"
    )
  elif opcode == 0x53 and f7 == 0x08:
    insn.update(kind="fmul.s", fwrites=(rd,))
  elif opcode == 0x53 and f3 == 0 and f7 == 0x78 and rs2 == 0:
    insn.update(kind="fmv.w.x", xreads=(rs1,), fwrites=(rd,))
  else:
    raise MeasureError(
      "INSTRUCTION_NOT_ALLOWED",
      f"{pc:#x}: {word:#010x} is outside the frozen scalar subset "
      "(no vector, matrix, branch, jump or other base instruction)",
    )
  return insn


def audit(
  body: bytes,
  base: int,
  descriptors: list[dict[str, Any]],
  expected: list[tuple[str, int]],
  csrs: dict[str, Any],
  clobber: list[str],
) -> dict[str, Any]:
  """Prove the resource bounds and the helper protocol of one image from the bytes that will execute.

  ``body`` is the executable segment at address ``base``. The helper calls must be exactly the binding's
  ``expected`` ``(helper, operand)`` sequence, and every rule of the helper ABI holds on its own: the two
  write forms only, operands inside their domain, each descriptor submitted once and waited before any of
  its bytes is read, each phase signalled once, no input read after input_released, output_ready only
  after every store completion was waited, and task_ret last with nothing left in flight.
  """
  helpers = csrs["helpers"]
  by_address = {helper["address"]: (name, helper) for name, helper in helpers.items()}
  if len(by_address) != len(helpers):
    raise MeasureError("BAD_CONTRACT", "two helpers share one generated CSR address")
  nexus = csrs["nexus"]
  by_index = {desc["index"]: desc for desc in descriptors}
  known: dict[int, int] = {_X0: 0}  # integer registers whose value is a proven constant here
  written_x: set[int] = set()
  written_f: set[int] = set()
  calls: list[dict[str, Any]] = []
  submitted: set[int] = set()
  waited: set[int] = set()
  phases: set[int] = set()
  released = retired = False
  count = len(body) // 4
  for position in range(count):
    pc = base + 4 * position
    if retired:
      raise MeasureError("RET_NOT_LAST", f"{pc:#x}: an instruction retires after task_ret")
    insn = decode(struct.unpack_from("<I", body, 4 * position)[0], pc)
    kind, rd, rs1 = insn["kind"], insn["rd"], insn["rs1"]
    if _SP in insn["xreads"] + insn["xwrites"] or _RA in insn["xreads"] + insn["xwrites"]:
      raise MeasureError("STACK_USED", f"{pc:#x}: {kind} uses sp or ra; this kernel has no stack")
    if _A0 in insn["xwrites"]:
      raise MeasureError("FRAME_BASE_CLOBBERED", f"{pc:#x}: {kind} overwrites the launch Frame base a0")
    written_x.update(reg for reg in insn["xwrites"] if reg)
    written_f.update(insn["fwrites"])
    if rd:
      if kind == "lui":
        known[rd] = insn["imm"]
      elif kind == "addi" and rs1 in known:
        known[rd] = (known[rs1] + insn["imm"]) & 0xFFFFFFFF
      else:
        known.pop(rd, None)
    if kind in ("lw", "sw"):
      low, high = insn["imm"], insn["imm"] + 4
      if rs1 != _A0 or low % 4:
        raise MeasureError(
          "FRAME_ACCESS_VIOLATION",
          f"{pc:#x}: {kind} is not a 4-byte-aligned access relative to the Frame base a0",
        )
      covering = [d for d in descriptors if d["frame_range"][0] <= low and high <= d["frame_range"][1]]
      if kind == "lw":
        fills = [d for d in covering if d["direction"] == DIRECTIONS[0]]
        if not fills or all(d["index"] not in waited for d in fills):
          raise MeasureError(
            "FRAME_ACCESS_VIOLATION",
            f"{pc:#x}: load of Frame [{low}, {high}) that no waited l2_to_frame descriptor fills",
          )
        if released:
          raise MeasureError("LOAD_AFTER_RELEASE", f"{pc:#x}: Frame input is read after input_released")
      else:
        drains = [d for d in covering if d["direction"] == DIRECTIONS[1]]
        if not drains or all(d["index"] in submitted for d in drains):
          raise MeasureError(
            "FRAME_ACCESS_VIOLATION",
            f"{pc:#x}: store to Frame [{low}, {high}) that no unsubmitted frame_to_l2 descriptor drains",
          )
      continue
    if kind != "csr":
      continue
    csr, f3 = insn["csr"], insn["f3"]
    if csr in by_address:
      name, helper = by_address[csr]
      if rd != _X0 or f3 not in (1, 5):
        raise MeasureError(
          "HELPER_ACCESS_FORM",
          f"{pc:#x}: {_CSR_FORMS[f3]} against {helper['csr']} is neither CSRRW x0,csr,rs1 "
          "nor CSRRWI x0,csr,imm",
        )
      if f3 == 5:
        operand, form = rs1, "csrrwi x0,csr,imm"
      elif rs1 in known:
        operand, form = known[rs1], "csrrw x0,csr,rs1"
      else:
        raise MeasureError(
          "OPERAND_NOT_STATIC",
          f"{pc:#x}: csrrw x0,{helper['csr']},{_X_NAMES[rs1]} carries an operand no static value proves",
        )
      calls.append(
        {"offset": pc - base, "helper": name, "csr": helper["csr"], "form": form, "operand": operand}
      )
      if helper["operand"] == "descriptor_index":
        if operand not in by_index:
          raise MeasureError(
            "OPERAND_OUT_OF_DOMAIN", f"{pc:#x}: mfe_submit {operand} names no launch descriptor"
          )
        if operand in submitted:
          raise MeasureError(
            "DESCRIPTOR_RESUBMITTED", f"{pc:#x}: launch descriptor {operand} is submitted twice"
          )
        submitted.add(operand)
      elif helper["operand"] == "event_index":
        if operand not in by_index:
          raise MeasureError("OPERAND_OUT_OF_DOMAIN", f"{pc:#x}: wait {operand} names no launch descriptor")
        if operand not in submitted:
          raise MeasureError(
            "WAIT_WITHOUT_SUBMIT",
            f"{pc:#x}: wait {operand} before launch descriptor {operand} is submitted",
          )
        waited.add(operand)
      elif helper["operand"] == "phase_id":
        if operand not in (0, 1):
          raise MeasureError("OPERAND_OUT_OF_DOMAIN", f"{pc:#x}: signal_phase {operand} is not 0 or 1")
        if operand in phases:
          raise MeasureError("PHASE_REPEATED", f"{pc:#x}: phase {operand} is signalled twice")
        phases.add(operand)
        if operand == 0:
          released = True
        elif any(d["direction"] == DIRECTIONS[1] and d["index"] not in waited for d in descriptors):
          raise MeasureError(
            "OUTPUT_READY_EARLY", f"{pc:#x}: output_ready before every frame_to_l2 completion was waited"
          )
      elif helper["operand"] == "code":
        if operand != 0:
          raise MeasureError(
            "OPERAND_OUT_OF_DOMAIN", f"{pc:#x}: task_ret {operand} is not the normal retire"
          )
        if position != count - 1:
          raise MeasureError("RET_NOT_LAST", f"{pc:#x}: task_ret is not the last instruction")
        if set(by_index) - waited:
          raise MeasureError("PENDING_AT_RET", f"{pc:#x}: task_ret with an unwaited launch descriptor")
        retired = True
      else:
        raise MeasureError("BAD_CONTRACT", f"{helper['csr']}: unknown operand kind {helper['operand']!r}")
    elif csr in nexus:
      raise MeasureError("UNKNOWN_HELPER", f"{pc:#x}: {nexus[csr]} is not a helper this kernel declares")
    elif csr == _FFLAGS and (f3 == 1 or (f3 == 2 and rs1 == _X0)):
      pass  # the fflags path: csrr rd, fflags reads, csrrw rd, fflags, rs1 writes
    else:
      raise MeasureError(
        "CSR_NOT_ALLOWED", f"{pc:#x}: CSR {csr:#05x} is neither a declared helper nor fflags"
      )
  if not retired:
    raise MeasureError("RET_MISSING", "the image never reaches task_ret")
  shown = [(call["helper"], call["operand"]) for call in calls]
  # every ABI rule held for the linked calls; the binding's own order is the last thing compared
  if shown != expected:
    first = next(
      (i for i, (a, b) in enumerate(zip(shown, expected)) if a != b), min(len(shown), len(expected))
    )
    raise MeasureError(
      "HELPER_SEQUENCE",
      f"the linked helper calls {shown} differ from the binding's at call {first}: {expected}",
    )
  clobbered = sorted(_X_NAMES[reg] for reg in written_x) + sorted(_F_NAMES[reg] for reg in written_f)
  if sorted(clobbered) != sorted(clobber):
    raise MeasureError(
      "CLOBBER_MISMATCH", f"the image clobbers {clobbered}, the binding declares {sorted(clobber)}"
    )
  return {
    "instruction_count": count,
    "stack_bytes": 0,  # no instruction reads or writes sp or ra, and nothing can call
    "spill_bytes": 0,  # every load and store is a proven Frame access inside a descriptor
    "helpers": calls,
    "clobber": clobbered,
  }


# ---------------------------------------------------------------------------------------------------
# golden bytes and the freeze
# ---------------------------------------------------------------------------------------------------
def golden(binding: dict[str, Any], descriptors: list[dict[str, Any]]) -> tuple[bytes, bytes]:
  """The hand-computable case of the card: 8 x i32 little-endian 1..8 in, x + 7 = 8..15 out."""
  entry = binding["manifest_kernel"]
  views = {view["handle"]: view for side in ("ins", "outs") for view in entry[side]}
  for desc in descriptors:
    if views[desc["handle"]]["shape"] != [ELEMENTS] or desc["bytes"] != 4 * ELEMENTS or desc["slot_offset"]:
      raise MeasureError(
        "BAD_BINDING",
        f"the golden case of {KERNEL_ID} is {ELEMENTS} x i32 moved whole (descriptor {desc['index']})",
      )
  values = list(range(1, ELEMENTS + 1))
  return struct.pack(f"<{ELEMENTS}i", *values), struct.pack(
    f"<{ELEMENTS}i", *(value + 7 for value in values)
  )


def measure(
  binding: Any, config: Any, out: Any, *, source_dir: Any = None
) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
  """Link, audit and measure both fixed slots; write slot_freeze.json and the golden bytes under ``out``.

  Returns one ``(record, layout, elf_path)`` per slot, in slot order, for ``cn package`` to seal:
  ``record`` carries ``entry``, ``resource_manifest`` and ``provenance``; ``layout`` carries the slot's
  ``name`` and its ``text_*`` / ``data_*`` link layout.
  """
  prepared = plan(binding, config, source_dir)
  if not mk.TOOLCHAIN.is_dir():
    raise MeasureError("NO_TOOLCHAIN", f"{mk.TOOLCHAIN} missing (cn env seed pulls the locked archive)")
  golden_input, golden_output = golden(prepared["binding"], prepared["descriptors"])
  binding, config, csrs, slots = (
    prepared["binding"],
    prepared["config"],
    prepared["csrs"],
    prepared["slots"],
  )
  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  provenance = {
    "toolchain": "toolchain_coralnpu_v2",
    "toolchain_lock_sha256": mk._lock_sha(),
    "source_sha256": {
      path.name: _sha(path.read_bytes())
      for path in sorted(prepared["source"].iterdir())
      if path.is_file() and path.suffix in (".S", ".ld")
    },
    "generated_sha256": {path.name: _sha(path.read_bytes()) for path in (HEADER, LAYOUT, HELPER_ABI)},
  }
  images: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
  bodies: dict[str, bytes] = {}
  bounds: dict[str, Any] = {}
  for slot in slots:
    elf_path, _, sections_report = _build_slot(slot, prepared["source"], out)
    elf = elf_path.read_bytes()
    check_relocatable(elf, out)
    body = check_image(elf, slot, sections_report)
    name = slot["name"]
    bounds = audit(
      body,
      slot["text_base"],
      prepared["descriptors"],
      prepared["expected"],
      csrs,
      binding["manifest_kernel"]["clobber"],
    )
    bodies[name] = body
    layout = {
      "name": name,
      "program_id": slot["program_id"],
      "entry": slot["text_base"],
      "text_base": slot["text_base"],
      "text_bytes": len(body),
      "data_base": slot["data_base"],
      "data_bytes": 0,
      "stack_bytes": bounds["stack_bytes"],
      "scratch_bytes": 0,
      "spill_bytes": bounds["spill_bytes"],
      "elf": f"{name}/{IMAGE}",
      "image_sha256": _sha(elf),
    }
    # the DTCM side of this slot: static data + the proven stack + scratch must fit the slot's own range
    if layout["data_bytes"] + layout["stack_bytes"] + layout["scratch_bytes"] > slot["data_bytes_per_slot"]:
      raise MeasureError(
        "DATA_EXCEEDS_SLOT",
        f"slot {name}: static data + stack + scratch exceed {slot['data_bytes_per_slot']} B "
        "(dtcm_bytes / program_slots)",
      )
    record = {
      "schema": "cn_kernel_measurement/v1",
      "kernel": KERNEL,
      "kernel_id": binding["manifest_kernel"]["kernel_id"],
      "kernel_version": binding["manifest_kernel"]["kernel_version"],
      **layout,
      "instruction_count": bounds["instruction_count"],
      "retire_count": bounds["instruction_count"],  # straight-line: every instruction retires once
      "itcm_region": slot["itcm_region"],
      "dtcm_region": slot["dtcm_region"],
      "resource_manifest": {
        "text_bytes": len(body),
        "static_data_bytes": 0,
        "stack_bytes": bounds["stack_bytes"],
        "scratch_bytes": 0,
        "bound_source": "static_analysis",
      },
      "provenance": provenance,
    }
    images.append((record, layout, elf_path))
  rows = [{key: record[key] for key in ROW_KEYS} for record, _, _ in images]
  check_coexist(rows)
  if any(bodies[slot["name"]] != bodies[slots[0]["name"]] for slot in slots):
    raise MeasureError(
      "IMAGE_DIFFERS_BETWEEN_SLOTS",
      "the slots' text differs: the image holds an absolute address, not Frame-relative code",
    )
  constraints = config["constraints"]
  geometry = slots[0]
  used = sum(row["text_bytes"] for row in rows)
  freeze = {
    "schema": "cn_slot_freeze/v1",
    "kernel": KERNEL,
    "kernel_id": binding["manifest_kernel"]["kernel_id"],
    "kernel_version": binding["manifest_kernel"]["kernel_version"],
    "isa": {
      "march": MARCH,
      "mabi": MABI,
      "mcmodel": "medany",
      "note": "the card's scalar ISA set; without a V or Zve* extension no RVV or VME instruction exists",
    },
    "frame": {
      "base_register": "a0",
      "live_bytes_per_context": prepared["frame_bytes"],
      "stripe_round_bytes": prepared["banks"]["stripe_round_bytes"],
      "reservation_bytes_per_context": prepared["banks"]["frame_reservation_bytes_per_context"],
      "descriptors": prepared["descriptors"],
    },
    "helper_sequence": bounds["helpers"],
    "clobber": bounds["clobber"],
    "resource_manifest": {
      "text_bytes": rows[0]["text_bytes"],
      "static_data_bytes": 0,
      "stack_bytes": bounds["stack_bytes"],
      "scratch_bytes": 0,
      "spill_bytes": bounds["spill_bytes"],
      "frame_live_bytes": prepared["frame_bytes"],
      "frame_reservation_bytes": prepared["banks"]["frame_reservation_bytes_per_context"],
      "bound_source": "static_analysis",
    },
    "slots": rows,
    "slot_geometry": {
      "program_slots": constraints["program_slots"],
      "itcm_bytes": constraints["itcm_bytes"],
      "dtcm_bytes": constraints["dtcm_bytes"],
      "itcm_origin": mk.ITCM_ORIGIN,
      "dtcm_origin": mk.DTCM_ORIGIN,
      "text_bytes_per_slot": geometry["text_bytes_per_slot"],
      "data_bytes_per_slot": geometry["data_bytes_per_slot"],
      "rule": (
        "slot k owns [origin + k * bytes / program_slots, origin + (k + 1) * bytes / program_slots) "
        "of ITCM and of DTCM; the committed link script of each slot must state exactly these regions"
      ),
    },
    "instruction_storage": {
      "itcm_bytes": constraints["itcm_bytes"],
      "used_by_frozen_slots": used,
      "free_bytes": constraints["itcm_bytes"] - used,
    },
    "bank_capacity": prepared["banks"],
    "golden": {
      "dtype": "i32",
      "little_endian": True,
      "elements": ELEMENTS,
      "operation": "x + 7",
      "input": {
        "file": GOLDEN["input"],
        "bytes": len(golden_input),
        "sha256": _sha(golden_input),
        "view": next(d["handle"] for d in prepared["descriptors"] if d["direction"] == DIRECTIONS[0]),
        "slot_offset": 0,
        "values": list(range(1, ELEMENTS + 1)),
      },
      "output": {
        "file": GOLDEN["output"],
        "bytes": len(golden_output),
        "sha256": _sha(golden_output),
        "view": next(d["handle"] for d in prepared["descriptors"] if d["direction"] == DIRECTIONS[1]),
        "slot_offset": 0,
        "values": list(range(8, ELEMENTS + 8)),
      },
    },
    "provenance": provenance,
  }
  (out / FREEZE).write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
  (out / GOLDEN["input"]).write_bytes(golden_input)
  (out / GOLDEN["output"]).write_bytes(golden_output)
  return images


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
  parser.add_argument("--binding", default=str(BINDING))
  parser.add_argument("--config", required=True)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  try:
    images = measure(Path(args.binding), Path(args.config), Path(args.out))
  except MeasureError as exc:
    print(f"scalar_kernel_build: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
  print(
    json.dumps(
      {
        record["name"]: {key: record[key] for key in ("program_id", "entry", "text_bytes", "image_sha256")}
        for record, _, _ in images
      },
      indent=2,
      sort_keys=True,
    )
  )


if __name__ == "__main__":
  main()
