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


def verify_sealed_inputs(seal_path: Path = SEAL_PATH, repo_root: Path = REPO) -> dict[str, str]:
  """Verify the C020 joint seal hashes for every contract consumed by the C021 oracle."""
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

  def inject_env_event(self, ev: NexusEnvEvent) -> None:
    if ev.status not in ("DONE", "ERROR", "TIMEOUT", "RESET"):
      raise NexusExtendError("BAD_EVENT_STATUS", ev.status)
    self.satisfied_events[ev.key] = ev.status
    if ev.kind == "REGION_DONE" and ev.status == "DONE" and self.region_ending:
      self.region_active = False
      self.region_ending = False

  def is_nexus_csr(self, csr_addr: int) -> bool:
    return csr_addr in self.catalog

  def decode_zicsr(self, insn: int) -> tuple[str, int, int, int, int] | None:
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
    old_val = self.csr_values.get(csr_addr, 0)
    if mnemonic in ("csrrw", "csrrwi"):
      new_val = operand
    elif mnemonic in ("csrrs", "csrrsi"):
      new_val = (old_val | operand) if is_write else old_val
    else:
      new_val = (old_val & (~operand & 0xFFFFFFFF)) if is_write else old_val

    if is_write and spec.decl_kind is not None:
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

  def check_task_exit_clean(self) -> None:
    if self.pending_decl is not None:
      raise NexusExtendError("UNCONSUMED_DECLARATION", self.pending_decl[0])
    if self.chain_open:
      raise NexusExtendError("EXIT_WITH_OPEN_CHAIN", "chain still open at task exit")
    if self.region_ending:
      raise NexusExtendError("EXIT_BEFORE_REGION_DONE", "region_end accepted but region_done not observed")
    if self.region_active:
      raise NexusExtendError("EXIT_WITHOUT_END", "active region never closed by REGION_END")
