"""Pinned Spike ISA reference driver (C021 / W7-02a).

Drives the unmodified pinned `coralnpu::sim::SpikeSimulator` (`riscv-isa-sim@93a10ae6` +
CoralNPU's 9 pinned patches) via `libcn_spike_ref.so` inside `coral-nexus-builder:c002`:
- Exposes `LoadProgram`, `Step`, `IsHalted`, `Reset`, `ReadRegister`, `WriteRegister`,
  `ReadMemory`, `WriteMemory`, `ReadVectorRegister`, `WriteVectorRegister`,
  `ReadCurrentMatrixTile`, plus `trace_elf` which executes step-by-step and emits C010-normalized
  per-instruction retire records (`retire_record.yaml`).
- Upstream Spike executes ONLY upstream ISA semantics. Nexus-private CSRs (`0x7c8..0x7cf`) trap
  in raw upstream Spike (`oracle_kind == "NONE"` on the raw probe) and are routed ONLY when
  `enable_nexus_extend=True` via `nexus_extend.NexusContextAdapter`.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bitwise_scalar import FFLAGS_MASK, pack_flags_delta
from nexus_extend import NexusContextAdapter, NexusEnvEvent

HERE = Path(__file__).resolve().parent
H = HERE.parents[1]
REPO = H.parents[1]
BUILD_SCRIPT = HERE / "build_spike_ref.sh"
DOCKER_RUN = H / "scripts/lib/docker_run.sh"
IMAGE_LOCK = H / "env/image.lock"

DEFAULT_CACHE_DIR = REPO / "out/coral_nexus/cache/spike_ref"


class SpikeRefError(Exception):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


class _CMemRegion(ctypes.Structure):
  _fields_ = [
    ("start_address", ctypes.c_uint64),
    ("length", ctypes.c_uint64),
  ]


def _load_c_lib(so_path: Path) -> ctypes.CDLL:
  lib = ctypes.CDLL(str(so_path))
  lib.cn_spike_create.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(_CMemRegion),
    ctypes.c_size_t,
  ]
  lib.cn_spike_create.restype = ctypes.c_void_p
  lib.cn_spike_destroy.argtypes = [ctypes.c_void_p]
  lib.cn_spike_destroy.restype = None
  lib.cn_spike_reset.argtypes = [ctypes.c_void_p]
  lib.cn_spike_reset.restype = ctypes.c_int
  lib.cn_spike_load_program.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_uint32,
  ]
  lib.cn_spike_load_program.restype = ctypes.c_int
  lib.cn_spike_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.cn_spike_step.restype = ctypes.c_int
  lib.cn_spike_is_halted.argtypes = [ctypes.c_void_p]
  lib.cn_spike_is_halted.restype = ctypes.c_int
  lib.cn_spike_get_pc.argtypes = [ctypes.c_void_p]
  lib.cn_spike_get_pc.restype = ctypes.c_uint64
  lib.cn_spike_get_cycle_count.argtypes = [ctypes.c_void_p]
  lib.cn_spike_get_cycle_count.restype = ctypes.c_uint64
  lib.cn_spike_read_register.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint64),
  ]
  lib.cn_spike_read_register.restype = ctypes.c_int
  lib.cn_spike_write_register.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_uint64,
  ]
  lib.cn_spike_write_register.restype = ctypes.c_int
  lib.cn_spike_read_vector_register.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint32),
  ]
  lib.cn_spike_read_vector_register.restype = ctypes.c_int
  lib.cn_spike_write_vector_register.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint32),
  ]
  lib.cn_spike_write_vector_register.restype = ctypes.c_int
  lib.cn_spike_read_matrix_tile.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
  ]
  lib.cn_spike_read_matrix_tile.restype = ctypes.c_int
  lib.cn_spike_read_memory.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_size_t,
  ]
  lib.cn_spike_read_memory.restype = ctypes.c_int
  lib.cn_spike_write_memory.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_size_t,
  ]
  lib.cn_spike_write_memory.restype = ctypes.c_int
  lib.cn_spike_last_error.argtypes = [ctypes.c_void_p]
  lib.cn_spike_last_error.restype = ctypes.c_char_p
  return lib


def ensure_spike_library(cache_dir: Path = DEFAULT_CACHE_DIR) -> tuple[Path, dict[str, Any]]:
  """Build or reuse `libcn_spike_ref.so` from the pinned W0-01 seed archive."""
  proc = subprocess.run(
    ["bash", str(BUILD_SCRIPT), "--cache-dir", str(cache_dir)],
    capture_output=True,
    text=True,
    check=False,
  )
  if proc.returncode != 0:
    raise SpikeRefError("BUILD_FAILED", proc.stderr.strip() or proc.stdout.strip())
  meta_path = Path(proc.stdout.strip().splitlines()[-1])
  meta = json.loads(meta_path.read_text())
  return cache_dir / str(meta["library"]), meta


class InProcessSpikeSession:
  """Direct ctypes binding to `coralnpu::sim::SpikeSimulator` (runs inside the locked container)."""

  def __init__(
    self,
    so_path: Path,
    *,
    isa: str = "",
    priv: str = "m",
    memory_regions: list[tuple[int, int]] | None = None,
  ) -> None:
    self._lib = _load_c_lib(so_path)
    reg_ptr = None
    reg_len = 0
    if memory_regions:
      arr_type = _CMemRegion * len(memory_regions)
      arr = arr_type(*(_CMemRegion(start, length) for start, length in memory_regions))
      reg_ptr = ctypes.cast(arr, ctypes.POINTER(_CMemRegion))
      reg_len = len(memory_regions)
      self._keepalive = arr
    self._handle = self._lib.cn_spike_create(
      isa.encode() if isa else None,
      priv.encode() if priv else None,
      reg_ptr,
      reg_len,
    )
    if not self._handle:
      raise SpikeRefError("INIT_FAILED", "cn_spike_create returned null")

  def close(self) -> None:
    if getattr(self, "_handle", None):
      self._lib.cn_spike_destroy(self._handle)
      self._handle = None

  def __del__(self) -> None:
    self.close()

  def LoadProgram(self, elf_path: str | Path, entry_point: int | None = None) -> bool:
    has_entry = 1 if entry_point is not None else 0
    rc = self._lib.cn_spike_load_program(
      self._handle, str(elf_path).encode(), has_entry, int(entry_point or 0) & 0xFFFFFFFF
    )
    return rc == 0

  def Step(self, num_steps: int = 1) -> int:
    rc = self._lib.cn_spike_step(self._handle, int(num_steps))
    if rc == -2:
      err = (self._lib.cn_spike_last_error(self._handle) or b"").decode(errors="replace")
      raise SpikeRefError("STEP_EXCEPTION", err)
    return int(rc)

  def Reset(self) -> None:
    if self._lib.cn_spike_reset(self._handle) != 0:
      raise SpikeRefError("RESET_FAILED")

  def IsHalted(self) -> bool:
    return bool(self._lib.cn_spike_is_halted(self._handle))

  def GetPC(self) -> int:
    return int(self._lib.cn_spike_get_pc(self._handle)) & 0xFFFFFFFF

  def GetCycleCount(self) -> int:
    return int(self._lib.cn_spike_get_cycle_count(self._handle))

  def ReadRegister(self, name: str) -> int:
    out = ctypes.c_uint64(0)
    if self._lib.cn_spike_read_register(self._handle, name.encode(), ctypes.byref(out)) != 0:
      raise SpikeRefError("READ_REG_FAILED", name)
    return int(out.value) & 0xFFFFFFFF

  def WriteRegister(self, name: str, value: int) -> None:
    if self._lib.cn_spike_write_register(self._handle, name.encode(), int(value) & 0xFFFFFFFF) != 0:
      raise SpikeRefError("WRITE_REG_FAILED", name)

  def ReadVectorRegister(self, v_idx: int) -> list[int]:
    words = (ctypes.c_uint32 * 4)()
    if self._lib.cn_spike_read_vector_register(self._handle, int(v_idx), words) != 0:
      raise SpikeRefError("READ_VREG_FAILED", f"v{v_idx}")
    return [int(words[i]) for i in range(4)]

  def WriteVectorRegister(self, v_idx: int, words_in: list[int]) -> None:
    if len(words_in) != 4:
      raise SpikeRefError("WRITE_VREG_LEN", str(len(words_in)))
    words = (ctypes.c_uint32 * 4)(*(int(w) & 0xFFFFFFFF for w in words_in))
    if self._lib.cn_spike_write_vector_register(self._handle, int(v_idx), words) != 0:
      raise SpikeRefError("WRITE_VREG_FAILED", f"v{v_idx}")

  def ReadCurrentMatrixTile(self, tile: int) -> tuple[int, list[int]]:
    tew = ctypes.c_uint32(0)
    actual = ctypes.c_size_t(0)
    buf = (ctypes.c_uint32 * 256)()
    rc = self._lib.cn_spike_read_matrix_tile(
      self._handle, int(tile), ctypes.byref(tew), buf, 256, ctypes.byref(actual)
    )
    if rc != 0:
      raise SpikeRefError("READ_MT_FAILED", f"tile={tile}")
    return int(tew.value), [int(buf[i]) for i in range(int(actual.value))]

  def ReadMemory(self, address: int, length: int) -> bytes:
    buf = ctypes.create_string_buffer(length)
    if self._lib.cn_spike_read_memory(self._handle, int(address), buf, int(length)) != 0:
      raise SpikeRefError("READ_MEM_FAILED", f"addr={address:#x} len={length}")
    return buf.raw

  def WriteMemory(self, address: int, data: bytes) -> None:
    buf = ctypes.create_string_buffer(bytes(data), len(data))
    if self._lib.cn_spike_write_memory(self._handle, int(address), buf, len(data)) != 0:
      raise SpikeRefError("WRITE_MEM_FAILED", f"addr={address:#x} len={len(data)}")


@dataclass(frozen=True)
class TraceRequest:
  elf_path: str
  entry_pc: int | None = None
  ctx_slot: int = 0
  ctx_generation: int = 1
  max_steps: int = 4096
  initial_gpr: dict[str, int] = field(default_factory=dict)
  initial_fpr: dict[str, int] = field(default_factory=dict)
  initial_csr: dict[str, int] = field(default_factory=dict)
  initial_memory: list[dict[str, Any]] = field(default_factory=list)
  read_memory_ranges: list[dict[str, int]] = field(default_factory=list)
  enable_nexus_extend: bool = False
  env_events: list[dict[str, Any]] = field(default_factory=list)
  memory_regions: list[tuple[int, int]] | None = None
  isa: str = ""


def _pack_reg_bank(values: list[int]) -> str:
  """Pack 32 x 32-bit register words as 256-hex-char little-reg-0-low integer hex."""
  acc = 0
  for idx, val in enumerate(values):
    acc |= (int(val) & 0xFFFFFFFF) << (32 * idx)
  return f"0x{acc:0256x}"


def _execute_trace_in_process(so_path: Path, req: TraceRequest) -> dict[str, Any]:
  sim = InProcessSpikeSession(
    so_path,
    isa=req.isa,
    memory_regions=req.memory_regions,
  )
  try:
    if not sim.LoadProgram(req.elf_path, req.entry_pc):
      raise SpikeRefError("LOAD_ELF_FAILED", req.elf_path)

    for name, val in req.initial_csr.items():
      sim.WriteRegister(name, val)
    for name, val in req.initial_gpr.items():
      sim.WriteRegister(name, val)
    for name, val in req.initial_fpr.items():
      sim.WriteRegister(name, val)
    for seg in req.initial_memory:
      sim.WriteMemory(int(seg["address"]), bytes.fromhex(str(seg["hex"])))

    adapter = NexusContextAdapter(ctx_slot=req.ctx_slot, ctx_generation=req.ctx_generation)
    events_by_step: dict[int, list[NexusEnvEvent]] = {}
    for raw_ev in req.env_events:
      ev = NexusEnvEvent(
        step_ordinal=int(raw_ev["step_ordinal"]),
        kind=str(raw_ev["kind"]),
        key=int(raw_ev["key"]),
        status=str(raw_ev.get("status", "DONE")),
      )
      events_by_step.setdefault(ev.step_ordinal, []).append(ev)

    retire_stream: list[dict[str, Any]] = []
    helper_trace: list[dict[str, Any]] = []

    prev_x = [sim.ReadRegister(f"x{i}") for i in range(32)]
    prev_f = [sim.ReadRegister(f"f{i}") for i in range(32)]

    step_idx = 0
    while step_idx < req.max_steps and not sim.IsHalted():
      for ev in events_by_step.get(step_idx, []):
        adapter.inject_env_event(ev)

      pc = sim.GetPC()
      try:
        insn_bytes = sim.ReadMemory(pc, 4)
        insn_word = int.from_bytes(insn_bytes, "little")
      except SpikeRefError:
        insn_word = 0

      zicsr = adapter.decode_zicsr(insn_word)
      if req.enable_nexus_extend and zicsr is not None and adapter.is_nexus_csr(zicsr[3]):
        rs1_idx = zicsr[2]
        rs1_val = 0 if rs1_idx == 0 else prev_x[rs1_idx]
        hrec = adapter.execute_private_csr(
          pc=pc,
          insn=insn_word,
          rs1_val=rs1_val,
          owner_op_seq=step_idx,
        )
        helper_trace.append(hrec)
        rd_idx = int(hrec["rd"])
        cur_x = list(prev_x)
        x_wb = 0
        if rd_idx != 0:
          cur_x[rd_idx] = int(hrec["rd_value"]) & 0xFFFFFFFF
          sim.WriteRegister(f"x{rd_idx}", cur_x[rd_idx])
          x_wb = 1 << rd_idx
        sim.WriteRegister("pc", int(hrec["next_pc"]))
        retire_stream.append(
          {
            "ctx_slot": req.ctx_slot,
            "ctx_generation": req.ctx_generation,
            "owner_op_seq": step_idx,
            "rob_tag": step_idx & 0xF,
            "flags_delta": 0,
            "valid": 1,
            "order": step_idx,
            "insn": insn_word,
            "trap": 0,
            "pc_rdata": pc,
            "next_pc": int(hrec["next_pc"]),
            "x_wb": x_wb,
            "x_rd": rd_idx if x_wb else None,
            "x_rd_val": cur_x[rd_idx] if x_wb else 0,
            "x_wdata": _pack_reg_bank(cur_x),
            "f_wb": 0,
            "f_rd": None,
            "f_rd_val": 0,
            "f_wdata": _pack_reg_bank(prev_f),
            "source": "nexus_extend",
          }
        )
        prev_x = cur_x
        step_idx += 1
        continue

      # Snapshot CSRs before single-stepping so per-instruction flags_delta is exact even when
      # prior sticky bits were already set in fflags/vxsat: clear sticky shadow, step, restore OR'd.
      arch_fflags_before = sim.ReadRegister("fflags") & FFLAGS_MASK
      arch_vxsat_before = sim.ReadRegister("vxsat") & 0x1
      mtvec_target = sim.ReadRegister("mtvec") & ~0x3
      prev_mcause = sim.ReadRegister("mcause")
      prev_mepc = sim.ReadRegister("mepc")
      prev_minstret = sim.ReadRegister("minstret")

      is_explicit_fflags_csr = zicsr is not None and zicsr[3] in (0x001, 0x002, 0x003, 0x009)
      if not is_explicit_fflags_csr:
        sim.WriteRegister("fflags", 0)
        sim.WriteRegister("vxsat", 0)
      executed = sim.Step(1)
      if executed == 0 and sim.IsHalted():
        if not is_explicit_fflags_csr:
          sim.WriteRegister("fflags", arch_fflags_before)
          sim.WriteRegister("vxsat", arch_vxsat_before)
        break

      new_pc = sim.GetPC()
      mcause_after = sim.ReadRegister("mcause")
      mepc_after = sim.ReadRegister("mepc")
      minstret_after = sim.ReadRegister("minstret")
      halted_now = sim.IsHalted()

      # A synchronous exception in Spike does not advance minstret, sets mepc=pc, and jumps to mtvec.
      # Never overwrite mcause before stepping, or a trap handler reading `csrr mcause` would be altered.
      is_mret = insn_word == 0x30200073
      trapped = int(
        (not halted_now)
        and (not is_mret)
        and minstret_after == prev_minstret
        and mepc_after == pc
        and new_pc == mtvec_target
        and (mcause_after != 0 or prev_mcause != 0 or prev_mepc != pc)
      )
      if trapped:
        sim.WriteRegister("fflags", arch_fflags_before)
        sim.WriteRegister("vxsat", arch_vxsat_before)
        fflags_delta = 0
        vxsat_delta = 0
      elif is_explicit_fflags_csr:
        fflags_delta = 0
        vxsat_delta = 0
      else:
        fflags_delta = sim.ReadRegister("fflags") & FFLAGS_MASK
        vxsat_delta = sim.ReadRegister("vxsat") & 0x1
        sim.WriteRegister("fflags", arch_fflags_before | fflags_delta)
        sim.WriteRegister("vxsat", arch_vxsat_before | vxsat_delta)

      _ = prev_mepc

      cur_x = [sim.ReadRegister(f"x{i}") for i in range(32)]
      cur_f = [sim.ReadRegister(f"f{i}") for i in range(32)]

      x_wb = 0
      x_rd: int | None = None
      x_rd_val = 0
      rd_cand = (insn_word >> 7) & 0x1F
      for idx in range(1, 32):
        if cur_x[idx] != prev_x[idx]:
          x_wb |= 1 << idx
          x_rd = idx
          x_rd_val = cur_x[idx]
      # Writes of the identical value to rd still architecturally write rd on non-trap ALU/load/JAL/CSR ops.
      opcode = insn_word & 0x7F
      writes_gpr_opcodes = {0x37, 0x17, 0x6F, 0x67, 0x03, 0x13, 0x33}
      if not trapped and not halted_now and x_wb == 0 and rd_cand != 0:
        if opcode in writes_gpr_opcodes or (opcode == 0x73 and zicsr is not None) or (
          opcode == 0x53 and ((insn_word >> 25) & 0x7F) in (0x50, 0x60, 0x70)
        ):
          x_wb = 1 << rd_cand
          x_rd = rd_cand
          x_rd_val = cur_x[rd_cand]

      f_wb = 0
      f_rd: int | None = None
      f_rd_val = 0
      for idx in range(32):
        if cur_f[idx] != prev_f[idx]:
          f_wb |= 1 << idx
          f_rd = idx
          f_rd_val = cur_f[idx]
      writes_fpr_opcodes = {0x07, 0x43, 0x47, 0x4B, 0x4F}
      if not trapped and not halted_now and f_wb == 0:
        if opcode in writes_fpr_opcodes or (
          opcode == 0x53 and ((insn_word >> 25) & 0x7F) not in (0x50, 0x60, 0x70)
        ):
          f_wb = 1 << rd_cand
          f_rd = rd_cand
          f_rd_val = cur_f[rd_cand]

      retire_stream.append(
        {
          "ctx_slot": req.ctx_slot,
          "ctx_generation": req.ctx_generation,
          "owner_op_seq": step_idx,
          "rob_tag": step_idx & 0xF,
          "flags_delta": pack_flags_delta(fflags_delta, vxsat_delta),
          "valid": 1,
          "order": step_idx,
          "insn": insn_word,
          "trap": trapped,
          "mcause": mcause_after if trapped else 0,
          "pc_rdata": pc,
          "next_pc": new_pc,
          "x_wb": x_wb,
          "x_rd": x_rd,
          "x_rd_val": x_rd_val,
          "x_wdata": _pack_reg_bank(cur_x),
          "f_wb": f_wb,
          "f_rd": f_rd,
          "f_rd_val": f_rd_val,
          "f_wdata": _pack_reg_bank(cur_f),
          "source": "spike_isa_ref",
        }
      )
      prev_x = cur_x
      prev_f = cur_f
      step_idx += 1

    mem_dumps: dict[str, str] = {}
    for rng in req.read_memory_ranges:
      addr = int(rng["address"])
      length = int(rng["length"])
      mem_dumps[f"0x{addr:08x}:{length}"] = sim.ReadMemory(addr, length).hex()

    return {
      "halted": sim.IsHalted(),
      "steps_executed": step_idx,
      "final_pc": sim.GetPC(),
      "final_gpr": {f"x{i}": prev_x[i] for i in range(32)},
      "final_fpr": {f"f{i}": prev_f[i] for i in range(32)},
      "final_csr": {
        "fflags": sim.ReadRegister("fflags") & FFLAGS_MASK,
        "frm": sim.ReadRegister("frm") & 0x7,
        "fcsr": sim.ReadRegister("fcsr") & 0xFF,
        "vxsat": sim.ReadRegister("vxsat") & 0x1,
        "mstatus": sim.ReadRegister("mstatus"),
        "misa": sim.ReadRegister("misa"),
        "mtvec": sim.ReadRegister("mtvec"),
        "mscratch": sim.ReadRegister("mscratch"),
        "mepc": sim.ReadRegister("mepc"),
        "mcause": sim.ReadRegister("mcause"),
        "mtval": sim.ReadRegister("mtval"),
        "vl": sim.ReadRegister("vl"),
        "vtype": sim.ReadRegister("vtype"),
        "vlenb": sim.ReadRegister("vlenb"),
        "mtype": sim.ReadRegister("mtype"),
      },
      "memory_bytes": mem_dumps,
      "retire_stream": retire_stream,
      "helper_trace": helper_trace,
      "nexus_csr_values": {f"0x{k:03x}": v for k, v in sorted(adapter.csr_values.items())},
    }
  finally:
    sim.close()


def run_trace_request(req: TraceRequest, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any]:
  """Execute `req` against `libcn_spike_ref.so` (directly if compatible, else in the locked container)."""
  so_path, build_meta = ensure_spike_library(cache_dir)
  try:
    result = _execute_trace_in_process(so_path, req)
    result["build_meta"] = build_meta
    return result
  except OSError:
    # Host glibc is older than Debian trixie inside coral-nexus-builder:c002; execute inside the
    # locked container with /workspace read-only and a dedicated writable staging directory.
    return _run_trace_in_container(req, cache_dir=cache_dir, build_meta=build_meta)


def _to_container_elf_path(elf_path: Path, stage_dir: Path) -> str:
  resolved = elf_path.resolve()
  try:
    _ = resolved.relative_to(REPO.resolve())
    # Copy into stage_dir (/out) so external/tmp_path ELFs are visible as well.
  except ValueError:
    pass
  staged = stage_dir / "input.elf"
  staged.write_bytes(resolved.read_bytes())
  return "/out/input.elf"


def _run_trace_in_container(
  req: TraceRequest, *, cache_dir: Path, build_meta: dict[str, Any]
) -> dict[str, Any]:
  stage_dir = cache_dir / f"run-{os.getpid()}"
  stage_dir.mkdir(parents=True, exist_ok=True)
  container_elf = _to_container_elf_path(Path(req.elf_path), stage_dir)
  payload = {
    "elf_path": container_elf,
    "entry_pc": req.entry_pc,
    "ctx_slot": req.ctx_slot,
    "ctx_generation": req.ctx_generation,
    "max_steps": req.max_steps,
    "initial_gpr": req.initial_gpr,
    "initial_fpr": req.initial_fpr,
    "initial_csr": req.initial_csr,
    "initial_memory": req.initial_memory,
    "read_memory_ranges": req.read_memory_ranges,
    "enable_nexus_extend": req.enable_nexus_extend,
    "env_events": req.env_events,
    "memory_regions": req.memory_regions,
    "isa": req.isa,
  }
  (stage_dir / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
  image = json.loads(IMAGE_LOCK.read_text())["tag"]
  # Mount both the read-only repo, the cached Spike shared library dir, and the per-run stage dir.
  cmd = [
    "docker",
    "run",
    "--rm",
    "--user",
    f"{os.getuid()}:{os.getgid()}",
    "--env",
    "HOME=/tmp",
    "-v",
    f"{REPO}:/workspace:ro",
    "-v",
    f"{cache_dir.resolve()}:/spike_cache:ro",
    "-v",
    f"{stage_dir.resolve()}:/out",
    "-w",
    "/workspace",
    image,
    "python3",
    "/workspace/hardware/coral_nexus/verification/oracle/spike_ref_driver.py",
    "--in-container-so",
    "/spike_cache/libcn_spike_ref.so",
    "--request",
    "/out/request.json",
    "--response",
    "/out/response.json",
  ]
  proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    raise SpikeRefError("CONTAINER_RUN_FAILED", proc.stderr.strip() or proc.stdout.strip())
  resp = json.loads((stage_dir / "response.json").read_text())
  resp["build_meta"] = build_meta
  return resp


def _cli_in_container() -> int:
  import argparse

  parser = argparse.ArgumentParser(description="In-container SpikeSimulator runner")
  parser.add_argument("--in-container-so", required=True)
  parser.add_argument("--request", required=True)
  parser.add_argument("--response", required=True)
  args = parser.parse_args()

  raw = json.loads(Path(args.request).read_text())
  regions = [tuple(x) for x in raw["memory_regions"]] if raw.get("memory_regions") else None
  req = TraceRequest(
    elf_path=raw["elf_path"],
    entry_pc=raw.get("entry_pc"),
    ctx_slot=int(raw.get("ctx_slot", 0)),
    ctx_generation=int(raw.get("ctx_generation", 1)),
    max_steps=int(raw.get("max_steps", 4096)),
    initial_gpr={str(k): int(v) for k, v in (raw.get("initial_gpr") or {}).items()},
    initial_fpr={str(k): int(v) for k, v in (raw.get("initial_fpr") or {}).items()},
    initial_csr={str(k): int(v) for k, v in (raw.get("initial_csr") or {}).items()},
    initial_memory=list(raw.get("initial_memory") or []),
    read_memory_ranges=list(raw.get("read_memory_ranges") or []),
    enable_nexus_extend=bool(raw.get("enable_nexus_extend", False)),
    env_events=list(raw.get("env_events") or []),
    memory_regions=regions,  # type: ignore[arg-type]
    isa=str(raw.get("isa") or ""),
  )
  out = _execute_trace_in_process(Path(args.in_container_so), req)
  Path(args.response).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  sys.exit(_cli_in_container())
