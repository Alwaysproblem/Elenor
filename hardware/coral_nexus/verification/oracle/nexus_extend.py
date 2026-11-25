"""Nexus-private CSR / helper / region-declaration oracle extension (C021 / W7-02c).

Pinned upstream Spike does NOT natively implement Nexus-private CSRs (`0x7c8..0x7cf`) or
Nexus control helpers (`WAIT`, `SIGNAL_PHASE`, `REGION_*`, `CHAIN_*`). Per C021:
- We NEVER pretend upstream Spike supports Nexus-private surfaces: the raw upstream probe reports
  `oracle_kind == "NONE"` for Nexus-private CSRs.
- Explicit environment inputs (`NexusEnvInput`) drive helper completions/events; the oracle NEVER
  infers reference outputs from DUT destination registers.
- Reads CSR logical names and allocated addresses from the C020-sealed protocol artifacts
  (`PROTOCOL_FREEZE.json`, `cn_layout.json`, `w3_vme_decl_abi_v1.json`, `w5_package.contract.yaml`),
  checking that the seal hash matches before trusting any address.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
  import yaml  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - inside minimal container python3
  yaml = None  # type: ignore[assignment]

HERE = Path(__file__).resolve().parent
H = HERE.parents[1]
REPO = H.parents[1]

SEAL_PATH = H / "protocol/PROTOCOL_FREEZE.json"
LAYOUT_PATH = H / "protocol/generated/cn_layout.json"
CSR_MAP_PATH = H / "protocol/csr_name_map.yaml"
VME_DECL_PATH = H / "protocol/w3_vme_decl_abi_v1.json"
W5_CONTRACT_PATH = H / "protocol/w5_package.contract.yaml"
SCALAR_ABI_PATH = H / "protocol/scalar_helper_abi_v1.json"
# v5-sealed input: the scalar helper path requires it (see `verify_sealed_inputs(extra_inputs=...)`).
SCALAR_ABI_SEALED_INPUT = "hardware/coral_nexus/protocol/scalar_helper_abi_v1.json"

SCALAR_HELPER_KINDS = ("MFE_SUBMIT", "WAIT", "SIGNAL_PHASE", "TASK_RET")
NUMERIC_PROFILE_PATH = H / "protocol/w3_numeric_profile_v1.yaml"

SEALED_ORACLE_INPUTS = (
  "hardware/coral_nexus/protocol/cn_protocol_v1.yaml",
  "hardware/coral_nexus/protocol/csr_name_map.yaml",
  "hardware/coral_nexus/protocol/w3_numeric_profile_v1.yaml",
  "hardware/coral_nexus/protocol/w3_vme_decl_abi_v1.json",
  "hardware/coral_nexus/protocol/w5_package.contract.yaml",
)


class NexusExtendError(Exception):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


def _sha256_file(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sealed_inputs(
  seal_path: Path = SEAL_PATH,
  repo_root: Path = REPO,
  extra_inputs: tuple[str, ...] = (),
) -> dict[str, str]:
  """Verify the C020 joint seal hashes for every contract consumed by the oracle.

  `extra_inputs` (e.g. `SCALAR_ABI_SEALED_INPUT` for the C038 scalar helper path) must also be
  listed under the seal's `inputs` and match on disk; the caller fails closed otherwise.
  """
  if not seal_path.is_file():
    raise NexusExtendError("MISSING_SEAL", str(seal_path))
  seal = json.loads(seal_path.read_text())
  if seal.get("schema") != "cn_protocol_freeze/v1" or seal.get("state") != "frozen":
    raise NexusExtendError("UNFROZEN_SEAL", f"state={seal.get('state')!r}")
  inputs = seal.get("inputs") or {}
  generated = seal.get("generated") or {}
  verified: dict[str, str] = {}
  for rel in SEALED_ORACLE_INPUTS:
    expected = inputs.get(rel)
    if not expected:
      raise NexusExtendError("UNSEALED_INPUT", rel)
    actual = _sha256_file(repo_root / rel)
    if actual != expected:
      raise NexusExtendError("SEAL_HASH_MISMATCH", f"{rel}: {actual} != {expected}")
    verified[rel] = actual
  for rel in extra_inputs:
    expected = inputs.get(rel)
    if not expected:
      raise NexusExtendError("UNSEALED_INPUT", rel)
    actual = _sha256_file(repo_root / rel)
    if actual != expected:
      raise NexusExtendError("SEAL_HASH_MISMATCH", f"{rel}: {actual} != {expected}")
    verified[rel] = actual
  layout_rel = "hardware/coral_nexus/protocol/generated/cn_layout.json"
  expected_layout = generated.get(layout_rel)
  if not expected_layout:
    raise NexusExtendError("UNSEALED_GENERATED", layout_rel)
  actual_layout = _sha256_file(repo_root / layout_rel)
  if actual_layout != expected_layout:
    raise NexusExtendError("SEAL_HASH_MISMATCH", f"{layout_rel}: {actual_layout} != {expected_layout}")
  verified[layout_rel] = actual_layout
  return verified


@dataclass(frozen=True)
class NexusCsrSpec:
  name: str
  address: int
  owner: str
  scope: str
  helper_name: str | None
  decl_kind: str | None


def load_nexus_csr_catalog(
  layout_path: Path = LAYOUT_PATH,
  csr_map_path: Path = CSR_MAP_PATH,
  w5_path: Path = W5_CONTRACT_PATH,
) -> dict[int, NexusCsrSpec]:
  """Build `{csr_address: NexusCsrSpec}` from the C017 generated layout and C009/C014 contracts."""
  layout = json.loads(layout_path.read_text())
  vme_decl = json.loads(VME_DECL_PATH.read_text())
  decl_csr_names = list((vme_decl.get("carrier") or {}).get("csr_name_map_entries") or [])
  decl_kinds = list((vme_decl.get("declaration") or {}).get("kinds") or [])
  fallback_helper_by_csr = {
    csr_name: (kind.lower(), kind) for csr_name, kind in zip(decl_csr_names, decl_kinds)
  }
  # In-container fallback (no PyYAML): the C038 scalar helper kinds also come from the sealed
  # scalar_helper_abi_v1.json (pure JSON) so csrw/csrrwi to scalar helper CSRs still model helpers.
  for entry in _scalar_helper_entries(abi_path=SCALAR_ABI_PATH, missing_ok=True):
    csr_name = entry.get("csr")
    if csr_name and csr_name not in fallback_helper_by_csr:
      fallback_helper_by_csr[csr_name] = (str(entry["name"]), str(entry["kind"]))
  if yaml is not None:
    csr_map = yaml.safe_load(csr_map_path.read_text())
    w5 = yaml.safe_load(w5_path.read_text())
    meta_by_name = {entry["name"]: entry for entry in csr_map.get("declared") or []}
    helper_by_csr = {
      entry["csr"]: (entry["name"], entry["kind"])
      for entry in (w5.get("helpers") or {}).get("registered") or []
    }
  else:
    meta_by_name = {}
    helper_by_csr = fallback_helper_by_csr
  out: dict[int, NexusCsrSpec] = {}
  for entry in layout.get("csr") or []:
    name = str(entry["name"])
    addr = int(entry["address"])
    if yaml is not None:
      meta = meta_by_name.get(name)
      if meta is None:
        raise NexusExtendError("UNDECLARED_CSR", name)
      owner = str(meta["owner"])
      scope = str(meta["scope"])
    else:
      owner = str(entry.get("owner", "W2"))
      scope = str(entry.get("scope", "per_context"))
    helper_name, decl_kind = helper_by_csr.get(name, (None, None))
    out[addr] = NexusCsrSpec(
      name=name,
      address=addr,
      owner=owner,
      scope=scope,
      helper_name=helper_name,
      decl_kind=decl_kind,
    )
  return out


@dataclass(frozen=True)
class NexusEnvEvent:
  """Explicit environment input for helper/WAIT/event completion (never inferred from DUT rd)."""

  step_ordinal: int
  kind: str  # "EVENT_DONE" | "REGION_GRANTED" | "REGION_DONE"
  key: int
  status: str = "DONE"


@dataclass
class NexusContextAdapter:
  """Per-Context adapter for Nexus-private CSR writes and C012/C014 helper state."""

  ctx_slot: int = 0
  ctx_generation: int = 0
  catalog: dict[int, NexusCsrSpec] = field(default_factory=load_nexus_csr_catalog)
  csr_values: dict[int, int] = field(default_factory=dict)
  pending_decl: tuple[str, int] | None = None
  region_active: bool = False
  region_ending: bool = False
  chain_open: bool = False
  emitted_helpers: list[dict[str, Any]] = field(default_factory=list)
  satisfied_events: dict[int, str] = field(default_factory=dict)
  env_log: list[dict[str, Any]] = field(default_factory=list)
  # C038 scalar-control helper state (armed only when `arm_scalar` is called with an env).
  scalar_env: ScalarEnv | None = None
  scalar_abi: dict[str, Any] | None = None
  mem: Any = None  # object with ReadMemory(addr, length)/WriteMemory(addr, data); sim or dict shim
  transfers: list[dict[str, Any]] = field(default_factory=list)
  settled_transfer_indices: set[int] = field(default_factory=set)
  parked_wait: dict[str, Any] | None = None
  signal_phases: list[int] = field(default_factory=list)
  task_finished: bool = False
  scalar_kind_by_csr: dict[int, str] = field(default_factory=dict)

  def inject_env_event(self, ev: NexusEnvEvent, *, at_step: int | None = None) -> None:
    if ev.status not in ("DONE", "ERROR", "TIMEOUT", "RESET"):
      raise NexusExtendError("BAD_EVENT_STATUS", ev.status)
    self.satisfied_events[ev.key] = ev.status
    self.env_log.append(
      {
        "kind": ev.kind,
        "key": ev.key,
        "status": ev.status,
        "step_ordinal": ev.step_ordinal,
        "injected_at": ev.step_ordinal if at_step is None else at_step,
      }
    )
    if ev.kind == "REGION_DONE" and ev.status == "DONE" and self.region_ending:
      self.region_active = False
      self.region_ending = False

  def is_nexus_csr(self, csr_addr: int) -> bool:
    return csr_addr in self.catalog

  @staticmethod
  def decode_zicsr(insn: int) -> tuple[str, int, int, int, int] | None:
    """Return `(mnemonic, rd, rs1_or_uimm, csr_addr, funct3)` if `insn` is a Zicsr op."""
    if (insn & 0x7F) != 0x73:
      return None
    funct3 = (insn >> 12) & 0x7
    if funct3 not in (1, 2, 3, 5, 6, 7):
      return None
    rd = (insn >> 7) & 0x1F
    rs1 = (insn >> 15) & 0x1F
    csr_addr = (insn >> 20) & 0xFFF
    names = {
      1: "csrrw",
      2: "csrrs",
      3: "csrrc",
      5: "csrrwi",
      6: "csrrsi",
      7: "csrrci",
    }
    return (names[funct3], rd, rs1, csr_addr, funct3)

  def execute_private_csr(
    self,
    *,
    pc: int,
    insn: int,
    rs1_val: int,
    owner_op_seq: int,
  ) -> dict[str, Any]:
    """Execute a Nexus-private CSR instruction that upstream Spike does not implement."""
    decoded = self.decode_zicsr(insn)
    if decoded is None:
      raise NexusExtendError("NOT_ZICSR", f"insn={insn:#010x}")
    mnemonic, rd, rs1_field, csr_addr, funct3 = decoded
    spec = self.catalog.get(csr_addr)
    if spec is None:
      raise NexusExtendError("UNKNOWN_NEXUS_CSR", f"csr={csr_addr:#05x}")

    operand = rs1_field if funct3 >= 5 else (rs1_val & 0xFFFFFFFF)
    is_write = not (mnemonic in ("csrrs", "csrrc", "csrrsi", "csrrci") and rs1_field == 0)

    scalar_kind = self.scalar_kind_by_csr.get(csr_addr)
    if scalar_kind is None and spec.decl_kind in SCALAR_HELPER_KINDS:
      scalar_kind = spec.decl_kind
    if scalar_kind is not None:
      # Sealed carrier: only `csrrw x0,csr,rs1` or `csrrwi x0,csr,imm` reaches a scalar helper;
      # every other access form is a protocol fault, never modeled.
      if mnemonic not in ("csrrw", "csrrwi") or rd != 0 or not is_write:
        raise NexusExtendError(
          "HELPER_ENCODING_INVALID",
          f"{spec.name}: {mnemonic} rd=x{rd} funct3={funct3}; "
          "only csrrw x0,csr,rs1 | csrrwi x0,csr,imm is legal",
        )
      if self.scalar_env is None or self.scalar_abi is None:
        raise NexusExtendError(
          "SCALAR_ENV_MISSING", f"{spec.name}: no scalar_env was armed for this Context"
        )
      effect = self._apply_scalar_helper(
        scalar_kind, operand, pc=pc, insn=insn, owner_op_seq=owner_op_seq
      )

    old_val = self.csr_values.get(csr_addr, 0)
    if mnemonic in ("csrrw", "csrrwi"):
      new_val = operand
    elif mnemonic in ("csrrs", "csrrsi"):
      new_val = (old_val | operand) if is_write else old_val
    else:
      new_val = (old_val & (~operand & 0xFFFFFFFF)) if is_write else old_val

    if is_write and spec.decl_kind is not None and scalar_kind is None:
      self._apply_declaration(spec.decl_kind, new_val)

    if is_write:
      self.csr_values[csr_addr] = new_val & 0xFFFFFFFF

    rec = {
      "pc": pc & 0xFFFFFFFF,
      "next_pc": (pc + 4) & 0xFFFFFFFF,
      "insn": insn & 0xFFFFFFFF,
      "csr_name": spec.name,
      "csr_address": csr_addr,
      "helper": spec.helper_name,
      "decl_kind": spec.decl_kind,
      "rd": rd,
      "rd_value": 0 if rd == 0 else old_val,
      "written_value": new_val & 0xFFFFFFFF if is_write else None,
      "owner_op_seq": owner_op_seq,
    }
    if scalar_kind is not None:
      rec["scalar_kind"] = scalar_kind
      rec["effect"] = effect
    self.emitted_helpers.append(rec)
    return rec

  def _apply_declaration(self, kind: str, operand: int) -> None:
    """Enforce C012 region/chain declaration legality (`w3_vme_decl_abi_v1.json`)."""
    if self.pending_decl is not None:
      raise NexusExtendError("ARMED_CONFLICT", f"{kind} while {self.pending_decl[0]} is still armed")
    if kind == "REGION_START":
      if self.region_active or self.region_ending:
        raise NexusExtendError("NESTED_START", "REGION_START while region already active/ending")
      self.pending_decl = (kind, operand)
      return
    if kind == "REGION_END":
      if not self.region_active or self.region_ending:
        raise NexusExtendError("END_WITHOUT_ACTIVE", "REGION_END without active region")
      if self.chain_open:
        raise NexusExtendError("END_WITH_OPEN_CHAIN", "REGION_END before CHAIN_FINAL")
      self.pending_decl = (kind, operand)
      return
    if kind == "CHAIN_FIRST":
      if not self.region_active or self.region_ending:
        raise NexusExtendError("TOUCH_WITHOUT_BINDING", "CHAIN_FIRST outside active region")
      if self.chain_open:
        raise NexusExtendError("CHAIN_ALREADY_OPEN", "CHAIN_FIRST while previous chain is open")
      self.pending_decl = (kind, operand)
      return
    if kind == "CHAIN_CONTINUE":
      if not self.region_active or not self.chain_open:
        raise NexusExtendError("CONTINUE_WITHOUT_FIRST", "CHAIN_CONTINUE without open chain")
      self.pending_decl = (kind, operand)
      return
    if kind == "CHAIN_FINAL":
      if not self.region_active or not self.chain_open:
        raise NexusExtendError("FINAL_WITHOUT_FIRST", "CHAIN_FINAL without open chain")
      self.pending_decl = (kind, operand)
      return
    raise NexusExtendError("UNKNOWN_KIND", kind)

  def consume_pending_e2(self) -> tuple[str, int]:
    """Simulate E2 single consumption when a granted target instruction is accepted."""
    if self.pending_decl is None:
      raise NexusExtendError("CONSUME_WITHOUT_PENDING", "target reached E2 with no armed declaration")
    kind, operand = self.pending_decl
    self.pending_decl = None
    if kind == "REGION_START":
      self.region_active = True
    elif kind == "REGION_END":
      self.region_ending = True
    elif kind == "CHAIN_FIRST":
      self.chain_open = True
    elif kind == "CHAIN_FINAL":
      self.chain_open = False
    return kind, operand

  def arm_scalar(
    self,
    env: ScalarEnv,
    *,
    abi: dict[str, Any] | None = None,
    mem: Any = None,
  ) -> None:
    """Arm the C038 scalar control helper model for this Context.

    `env` supplies the sealed launch descriptors, resolved L2 slot bases and the writable task
    frame; `mem` is the reference-memory endpoint the modeled transfers copy through (the Spike
    session or an equivalent ReadMemory/WriteMemory object). Nothing is inferred from the DUT.
    """
    self.scalar_env = env
    self.scalar_abi = abi if abi is not None else load_scalar_helper_abi()
    self.mem = mem
    self.scalar_kind_by_csr = {}
    by_csr = self.scalar_abi.get("_kind_by_csr") or {}
    for addr, spec in self.catalog.items():
      kind = by_csr.get(spec.name)
      if kind is None and spec.decl_kind in SCALAR_HELPER_KINDS:
        kind = spec.decl_kind
      if kind in SCALAR_HELPER_KINDS:
        self.scalar_kind_by_csr[addr] = kind

  def _apply_scalar_helper(
    self, kind: str, operand: int, *, pc: int, insn: int, owner_op_seq: int
  ) -> dict[str, Any]:
    """Sealed scalar helper semantics from `scalar_helper_abi_v1.json` (C038)."""
    assert self.scalar_env is not None and self.scalar_abi is not None
    if self.task_finished:
      raise NexusExtendError("TASK_AFTER_RET", f"{kind} after task_ret")
    if kind == "MFE_SUBMIT":
      if operand >= len(self.scalar_env.descriptors):
        raise NexusExtendError(
          "DESCRIPTOR_INDEX_RANGE",
          f"mfe_submit index {operand} outside 0..{len(self.scalar_env.descriptors) - 1}",
        )
      self.transfers.append({"index": operand, "state": "submitted", "pc": pc})
      return {"op": "mfe_submit", "index": operand, "state": "in_flight"}
    if kind == "WAIT":
      if self.parked_wait is not None:
        raise NexusExtendError("NESTED_WAIT", "wait while another wait is parked")
      wait_domain = len(self.scalar_env.wait_events) or len(self.scalar_env.descriptors)
      if operand >= wait_domain:
        raise NexusExtendError(
          "OPERAND_OUT_OF_DOMAIN", f"wait event index {operand} outside 0..{wait_domain - 1}"
        )
      status = self.satisfied_events.get(operand)
      if status is None:
        # Park: the record stays unretired until the environment event arrives (completion_first
        # on the same cycle means an already-satisfied event hits directly instead).
        self.parked_wait = {"operand": operand, "pc": pc, "insn": insn, "owner_op_seq": owner_op_seq}
        return {"op": "wait", "index": operand, "parked": True, "resolved": False}
      if status != "DONE":
        raise NexusExtendError("WAIT_FAILED_TERMINAL", f"wait({operand}) -> {status}")
      self._complete_event(operand, pc=pc)
      return {"op": "wait", "index": operand, "parked": False, "resolved": True, "status": status}
    if kind == "SIGNAL_PHASE":
      phase_ids = self.scalar_abi.get("_phase_ids") or {0, 1}
      if operand not in phase_ids:
        raise NexusExtendError("SIGNAL_PHASE_RANGE", f"phase {operand} not in {sorted(phase_ids)}")
      self.signal_phases.append(operand)
      return {"op": "signal_phase", "phase": operand}
    if kind == "TASK_RET":
      codes = self.scalar_abi.get("_task_ret_codes") or {0}
      if operand not in codes:
        raise NexusExtendError("TASK_RET_CODE", f"code {operand} not in {sorted(codes)}")
      if self.parked_wait is not None:
        raise NexusExtendError("TASK_RET_PARKED", "task_ret while a wait is parked")
      unsettled = sorted(
        t["index"] for t in self.transfers if t["index"] not in self.settled_transfer_indices
      )
      if unsettled:
        raise NexusExtendError("TASK_RET_UNSETTLED", f"unsettled transfers {unsettled}")
      self.task_finished = True
      return {"op": "task_ret", "code": operand}
    raise NexusExtendError("UNKNOWN_SCALAR_KIND", kind)

  def _complete_event(self, operand: int, *, pc: int) -> None:
    """Complete a wait operand: when it names a submitted descriptor, model the sealed transfer."""
    assert self.scalar_env is not None
    if 0 <= operand < len(self.scalar_env.descriptors):
      d = self.scalar_env.descriptors[operand]
      if operand not in self.settled_transfer_indices and any(
        t["index"] == operand for t in self.transfers
      ):
        self._settle_transfer(d, pc=pc)

  def _settle_transfer(self, d: dict[str, Any], *, pc: int) -> None:
    """Copy `d["bytes"]` between the bound L2 slot and the task frame through `self.mem`."""
    assert self.scalar_env is not None
    if self.mem is None:
      raise NexusExtendError("MEM_MISSING", "arm_scalar requires a memory endpoint")
    rng = self.scalar_env.bindings[d["handle"]]
    slot_addr = rng["base"] + d["slot_offset"]
    frame_addr = self.scalar_env.frame_base + d["frame_offset"]
    if d["direction"] == "l2_to_frame":
      src, dst = slot_addr, frame_addr
    elif d["direction"] == "frame_to_l2":
      src, dst = frame_addr, slot_addr
    else:
      raise NexusExtendError("DESCRIPTOR_DIRECTION", str(d["direction"]))
    data = self.mem.ReadMemory(src, d["bytes"])
    self.mem.WriteMemory(dst, data)
    self.settled_transfer_indices.add(int(d["index"]))
    for t in self.transfers:
      if t["index"] == d["index"] and t["state"] == "submitted":
        t["state"] = "settled"
        t["settled_pc"] = pc

  def try_resolve_wait(self) -> str:
    """Driver hook: while a WAIT is parked, resolve it when its env event becomes terminal."""
    if self.parked_wait is None:
      return "hit"
    operand = int(self.parked_wait["operand"])
    status = self.satisfied_events.get(operand)
    if status is None:
      return "parked"
    if status != "DONE":
      raise NexusExtendError("WAIT_FAILED_TERMINAL", f"wait({operand}) -> {status}")
    self._complete_event(operand, pc=int(self.parked_wait["pc"]))
    self.parked_wait = None
    for rec in reversed(self.emitted_helpers):
      if rec.get("effect", {}).get("op") == "wait" and rec["effect"].get("index") == operand:
        rec["effect"].update({"parked": True, "resolved": True, "status": status})
        break
    return "resolved"

  def scalar_report(self) -> dict[str, Any]:
    """Post-run report of the modeled scalar helper activity (oracle evidence, never DUT-derived)."""
    helper_ops = [
      {"pc": r["pc"], "csr_name": r["csr_name"], "kind": r.get("scalar_kind"), "effect": r.get("effect")}
      for r in self.emitted_helpers
      if r.get("scalar_kind") is not None
    ]
    return {
      "helpers": helper_ops,
      "transfers": [dict(t) for t in self.transfers],
      "settled": sorted(self.settled_transfer_indices),
      "signal_phases": list(self.signal_phases),
      "env_log": [dict(e) for e in self.env_log],
      "task_finished": self.task_finished,
      "parked_wait": self.parked_wait is not None,
    }

  def check_task_exit_clean(self) -> None:
    if self.pending_decl is not None:
      raise NexusExtendError("UNCONSUMED_DECLARATION", self.pending_decl[0])
    if self.chain_open:
      raise NexusExtendError("EXIT_WITH_OPEN_CHAIN", "chain still open at task exit")
    if self.region_ending:
      raise NexusExtendError("EXIT_BEFORE_REGION_DONE", "region_end accepted but region_done not observed")
    if self.region_active:
      raise NexusExtendError("EXIT_WITHOUT_END", "active region never closed by REGION_END")
    if self.scalar_env is not None:
      if self.parked_wait is not None:
        raise NexusExtendError("EXIT_WITH_PARKED_WAIT", f"wait({self.parked_wait['operand']})")
      unsettled = sorted(
        t["index"] for t in self.transfers if t["index"] not in self.settled_transfer_indices
      )
      if unsettled:
        raise NexusExtendError("EXIT_WITH_UNSETTLED_TRANSFERS", f"{unsettled}")
      if not self.task_finished:
        raise NexusExtendError("EXIT_WITHOUT_TASK_RET", "no task_ret in the retire stream")


# ---------------------------------------------------------------------------
# C038 scalar control helper ABI (`scalar_helper_abi_v1.json`, sealed inside the v5 freeze).
# ---------------------------------------------------------------------------


def _scalar_helper_entries(
  *, abi_path: Path = SCALAR_ABI_PATH, missing_ok: bool = False
) -> list[dict[str, Any]]:
  """Return the `[{name, csr, kind}]` helper entries declared by the sealed scalar ABI.

  With `missing_ok=True` a not-yet-sealed ABI degrades to an empty list (fallback path only);
  the strict path always requires the file.
  """
  if not abi_path.is_file():
    if missing_ok:
      return []
    raise NexusExtendError("MISSING_SCALAR_ABI", str(abi_path))
  doc = json.loads(abi_path.read_text())
  helpers = doc.get("helpers") or doc.get("helper_kinds") or []
  if isinstance(helpers, dict):
    helpers = [{"name": k, **(v if isinstance(v, dict) else {"kind": v})} for k, v in helpers.items()]
  out: list[dict[str, Any]] = []
  for entry in helpers:
    if not isinstance(entry, dict) or not entry.get("name"):
      continue
    out_entry = dict(entry)
    out_entry["name"] = str(entry["name"])
    out_entry["csr"] = str(entry["csr"])
    out_entry["kind"] = str(entry["kind"])
    out.append(out_entry)
  return out


def load_scalar_helper_abi(abi_path: Path = SCALAR_ABI_PATH) -> dict[str, Any]:
  """Load and sanity-check the sealed scalar helper ABI document."""
  if not abi_path.is_file():
    raise NexusExtendError("MISSING_SCALAR_ABI", str(abi_path))
  doc = json.loads(abi_path.read_text())
  entries = _scalar_helper_entries(abi_path=abi_path)
  kinds = {e["kind"] for e in entries}
  missing = set(SCALAR_HELPER_KINDS) - kinds
  if missing:
    raise NexusExtendError("SCALAR_ABI_MISSING_KINDS", f"{sorted(missing)}")
  # Carrier legality: read the declared forms (`"CSRRW x0,csr,rs1"`, `"CSRRWI x0,csr,imm"`); a
  # sealed ABI may only declare the two write forms with rd == x0.
  carrier = doc.get("carrier") or {}
  forms = carrier.get("forms") or ["CSRRW x0,csr,rs1", "CSRRWI x0,csr,imm"]
  mnemonics = {str(f).split()[0].strip().lower() for f in forms}
  if not mnemonics <= {"csrrw", "csrrwi"}:
    raise NexusExtendError("SCALAR_ABI_CARRIER_FORMS", f"{sorted(mnemonics)}")
  if carrier.get("rd_must_be_x0") is False:
    raise NexusExtendError("SCALAR_ABI_CARRIER_RD", "rd_must_be_x0 must hold for scalar helpers")
  doc["_mnemonics"] = mnemonics
  doc["_helper_by_kind"] = {e["kind"]: e for e in entries}
  doc["_kind_by_csr"] = {e["csr"]: e["kind"] for e in entries}

  def _operand_domain(kind: str) -> set[int]:
    entry = doc["_helper_by_kind"].get(kind) or {}
    op = entry.get("operand") or {}
    domain = op.get("domain")
    if isinstance(domain, dict):
      out = set()
      for key in domain:
        try:
          out.add(int(key))
        except (TypeError, ValueError):
          continue
      return out
    if isinstance(domain, (list, tuple)):
      return {int(v) for v in domain}
    return set()

  doc["_phase_ids"] = _operand_domain("SIGNAL_PHASE") or {0, 1}
  doc["_task_ret_codes"] = _operand_domain("TASK_RET") or {0}
  record = doc.get("descriptor_record") or {}
  doc["_descriptor_required_fields"] = list(record.get("required_fields") or [])
  doc["_descriptor_directions"] = list(record.get("directions") or ("l2_to_frame", "frame_to_l2"))
  return doc


def verify_scalar_abi_seal(repo_root: Path = REPO) -> dict[str, str]:
  """The scalar helper path additionally requires the ABI file to be inside the current seal."""
  return verify_sealed_inputs(repo_root=repo_root, extra_inputs=(SCALAR_ABI_SEALED_INPUT,))


@dataclass
class ScalarEnv:
  """Environment inputs for the scalar control helper model (never inferred from DUT rd).

  `descriptors`: sealed `launch_descriptors` entries (slot-local immutable).
  `bindings`: `handle -> {"base": int, "length": int}` L2 slot resolution.
  `frame_base`/`frame_length`: the writable task-frame window (a0 base per the kernel ABI).
  """

  descriptors: list[dict[str, Any]] = field(default_factory=list)
  wait_events: list[Any] = field(default_factory=list)
  bindings: dict[str, dict[str, int]] = field(default_factory=dict)
  frame_base: int = 0
  frame_length: int = 0


def scalar_env_from_dict(raw: dict[str, Any], abi: dict[str, Any] | None = None) -> ScalarEnv:
  """Parse `scalar_env` from an oracle_input/runner document, sealing field names via the ABI."""
  abi = abi or load_scalar_helper_abi()
  descs_in = list(raw.get("descriptors") or [])
  if not descs_in:
    raise NexusExtendError("MISSING_DESCRIPTORS", "scalar_env.descriptors is empty")
  directions = set(abi.get("_descriptor_directions") or ("l2_to_frame", "frame_to_l2"))
  required = set(abi.get("_descriptor_required_fields") or ())
  descs: list[dict[str, Any]] = []
  for d in descs_in:
    if required and not required <= set(d):
      raise NexusExtendError(
        "DESCRIPTOR_FIELD_MISSING",
        f"{json.dumps(d, sort_keys=True)} lacks {sorted(required - set(d))}",
      )
    try:
      index = int(d["index"])
      direction = str(d["direction"])
      handle = str(d["handle"])
      nbytes = int(d["bytes"])
      slot_offset = int(d.get("slot_offset", 0))
      frame_offset = int(d.get("frame_offset", 0))
    except (KeyError, TypeError, ValueError) as exc:
      raise NexusExtendError("DESCRIPTOR_FIELD_MISSING", json.dumps(d, sort_keys=True)) from exc
    if direction not in directions:
      raise NexusExtendError("DESCRIPTOR_DIRECTION", f"{direction!r} not in {sorted(directions)}")
    if nbytes <= 0 or slot_offset < 0 or frame_offset < 0:
      raise NexusExtendError("DESCRIPTOR_RANGE", json.dumps(d, sort_keys=True))
    descs.append(
      {
        "index": index,
        "direction": direction,
        "handle": handle,
        "bytes": nbytes,
        "slot_offset": slot_offset,
        "frame_offset": frame_offset,
      }
    )
  descs.sort(key=lambda d: d["index"])
  if [d["index"] for d in descs] != list(range(len(descs))):
    raise NexusExtendError(
      "DESCRIPTOR_INDEX_GAP", f"indices={[d['index'] for d in descs]} must be dense from 0"
    )
  bindings: dict[str, dict[str, int]] = {}
  for handle, rng in (raw.get("bindings") or {}).items():
    if not isinstance(rng, dict) or "base" not in rng or "length" not in rng:
      raise NexusExtendError("BINDING_RANGE", f"{handle}: {rng!r}")
    bindings[str(handle)] = {"base": int(rng["base"]), "length": int(rng["length"])}
  for d in descs:
    rng = bindings.get(d["handle"])
    if rng is None:
      raise NexusExtendError("BINDING_MISSING", f"descriptor handle {d['handle']!r} has no L2 range")
    if d["slot_offset"] + d["bytes"] > rng["length"]:
      raise NexusExtendError(
        "BINDING_RANGE",
        f"descriptor {d['index']}: slot_offset {d['slot_offset']} + bytes {d['bytes']} "
        f"> length {rng['length']}",
      )
  frame_base = int(raw.get("frame_base") or 0)
  frame_length = int(raw.get("frame_length") or 0)
  if frame_base <= 0 or frame_length <= 0:
    raise NexusExtendError(
      "FRAME_RANGE", f"frame_base={frame_base:#x} frame_length={frame_length}"
    )
  for d in descs:
    if d["frame_offset"] + d["bytes"] > frame_length:
      raise NexusExtendError(
        "FRAME_RANGE",
        f"descriptor {d['index']}: frame_offset {d['frame_offset']} + bytes {d['bytes']} "
        f"> frame_length {frame_length}",
      )
  wait_events = list(raw.get("wait_events") or [])
  if not wait_events:
    # Per the ABI wait_events[i] is the completion event of launch_descriptors[i]; a binding that
    # omits the table means descriptor index i waits on event key i.
    wait_events = [{"event": d["index"], "key": d["index"]} for d in descs]
  return ScalarEnv(
    descriptors=descs,
    wait_events=wait_events,
    bindings=bindings,
    frame_base=frame_base,
    frame_length=frame_length,
  )
