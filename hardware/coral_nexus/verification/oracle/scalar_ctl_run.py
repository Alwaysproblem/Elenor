"""C038 scalar control kernel oracle runner and acceptance CLI (W7-02 / W5-02).

One callable entry point for the real-run acceptance:
  * `load_case(...)` resolves a sealed package (`package_dir` or `slot_freeze.json` + slot), the
    binding's `launch_descriptors`, the L2 slot bindings and the writable task frame, then
    `run_case` drives the real linked `image.elf` on the pinned Spike reference
    (`spike_ref_driver` + `nexus_extend` scalar helper model) and checks:
      - independent golden output bytes (never inferred from the DUT/package),
      - the committed `golden_output.bin` produced by the kernel build (when present),
      - helper retire stream order (`mfe_submit/wait` pairs, `signal_phase`, `task_ret` last),
      - a clean retire-chain exit (`task_exit`), and
      - an optional DUT retire/memory comparison (`--dut-retire`).
  * Package authentication is delegated to the sealed C037 read-only Loader (`load_file`): the
    codec, the semantic `load_program`, the image ELF cross-check and the `linked_slot` /
    `slot_freeze` commitment all run there. Nothing here re-derives or trusts those fields.

CLI:
  python scalar_ctl_run.py --binding H/kernels/bindings/scalar_ctl_v1.yaml \
      --package-dir out/package/A --input-file out/golden_input.bin --out report.json
  python scalar_ctl_run.py --binding ... --freeze out/slot_freeze.json --slot B --out report.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nexus_extend as ne
import oracle_main as om
import spike_ref_driver as srd
import yaml

HERE = Path(__file__).resolve().parent
H = HERE.parents[1]
REPO = H.parents[1]

for _p in (str(REPO), str(H / "scripts" / "lib")):
  if _p not in sys.path:
    sys.path.insert(0, _p)

try:  # the sealed Loader + control plane are host-side only; degrade to a clear error at use.
  import package_build

  from pipeline_validator.config import HardwareConfig, SimConfig  # noqa: F401
  from pipeline_validator.package import PackageError, load_file
except ImportError:  # pragma: no cover
  package_build = None  # type: ignore[assignment]
  PackageError = Exception  # type: ignore[assignment,misc]
  load_file = None  # type: ignore[assignment]

DEFAULT_FRAME_BASE = 0x00040000
# C038 task frame reservation: descriptor frame_offsets reach 1024+bytes (output stripe), so the
# default is the declared 2048 reservation, not the older 32-word guess. --frame-length still wins.
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_OUTPUT_FILL = 0xEE


class ScalarCaseError(Exception):
  def __init__(self, code: str, detail: str = "") -> None:
    super().__init__(f"{code}: {detail}" if detail else code)
    self.code = code
    self.detail = detail


def _sha(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


@dataclass
class ScalarCase:
  """One fully-resolved oracle run for a scalar control package."""

  slot: str
  elf_path: Path
  entry_pc: int
  package: dict[str, Any] | None
  binding: dict[str, Any]
  kernel: dict[str, Any]
  descriptors: list[dict[str, Any]]
  bindings: dict[str, dict[str, int]]
  frame_base: int
  frame_length: int
  input_bytes: bytes
  expected_bytes: bytes
  golden_file_bytes: bytes | None
  out_range: dict[str, int]
  env_events: list[dict[str, Any]]
  l2_source: str
  freeze_dir: Path | None = None
  verified: dict[str, Any] = field(default_factory=dict)
  wait_events: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Resolution helpers (sealed/binding documents only; nothing guessed).
# ---------------------------------------------------------------------------


def _kernel_doc(binding: dict[str, Any], package: dict[str, Any] | None) -> dict[str, Any]:
  if package is not None:
    kernels = ((package.get("binding_manifest") or {}).get("kernels")) or []
    if len(kernels) != 1:
      raise ScalarCaseError("PACKAGE_KERNELS", f"expected exactly one kernel, got {len(kernels)}")
    return kernels[0]
  manifest = binding.get("manifest_kernel")
  if isinstance(manifest, dict):
    return manifest
  raise ScalarCaseError("BINDING_KERNEL", "no manifest_kernel in binding and no package given")


def _launch_descriptors(kernel: dict[str, Any], binding: dict[str, Any]) -> list[dict[str, Any]]:
  descs = kernel.get("launch_descriptors") or binding.get("launch_descriptors") or []
  if not descs:
    raise ScalarCaseError(
      "DESCRIPTORS_MISSING", "no launch_descriptors in binding_manifest kernel or binding"
    )
  return [dict(d) for d in descs]


def _view_handles(kernel: dict[str, Any], binding: dict[str, Any]) -> dict[str, str]:
  """`{handle: permission}` from the canonical ins/outs views."""
  out: dict[str, str] = {}
  for side, perm in (("ins", "r"), ("outs", "w")):
    for v in kernel.get(side) or binding.get(side) or []:
      handle = str(v.get("handle"))
      out[handle] = str(v.get("permission", perm))
  return out


def _ranges_from(doc: Any) -> dict[str, dict[str, Any]]:
  """Collect `name -> {base,length}` from dict-/list-shaped tables or GlobalBinding objects."""
  out: dict[str, dict[str, Any]] = {}
  items: Any
  if isinstance(doc, dict):
    items = doc.items()
  elif isinstance(doc, list):
    items = enumerate(doc)
  else:
    return out
  for key, rng in items:
    if isinstance(rng, dict):
      name = rng.get("name") or rng.get("handle") or rng.get("global") or rng.get("id") or key
      base = rng.get("base", rng.get("address", rng.get("base_iova")))
      length = rng.get("length", rng.get("bytes", rng.get("size", rng.get("size_bytes"))))
      access = str(rng.get("access", rng.get("permission", rng.get("permissions", ""))))
    else:  # GlobalBinding-like object
      name = getattr(rng, "name", key)
      base = getattr(rng, "base_iova", getattr(rng, "base", None))
      length = getattr(rng, "size_bytes", getattr(rng, "length", None))
      access = str(getattr(rng, "permissions", getattr(rng, "access", "")))
    if name is None or base is None or length is None:
      continue
    out[str(name)] = {"base": int(base), "length": int(length), "access": access}
  return out


def _resolve_l2_slots(
  *,
  descriptors: list[dict[str, Any]],
  kernel: dict[str, Any],
  binding: dict[str, Any],
  package: dict[str, Any] | None,
  compiled: dict[str, Any] | None,
  actual_bindings: dict[str, Any] | None,
  overrides: dict[str, int],
) -> tuple[dict[str, dict[str, int]], str]:
  """Resolve `handle -> {base,length}` for every descriptor handle.

  Order: explicit overrides > binding globals/slots > view_map > compiled/control-plane
  bindings mapped through `actual_view` (`l2.<global>`) > permission-ordered fallback.
  """
  needed = {d["handle"] for d in descriptors}
  out: dict[str, dict[str, int]] = {}

  for handle in needed:
    if handle in overrides:
      out[handle] = {"base": overrides[handle], "length": overrides.get(f"{handle}:length", 0)}

  source = "cli-override"

  if set(out) != needed:
    source = ""
    for cand_name, cand in (
      ("binding.globals", binding.get("globals")),
      ("binding.slots", binding.get("slots")),
      ("binding.l2_slots", binding.get("l2_slots")),
      ("kernel.view_map", kernel.get("view_map")),
      ("binding.view_map", binding.get("view_map")),
      ("binding.launch_map", binding.get("launch_map")),
    ):
      ranges = _ranges_from(cand)
      hits = {h: r for h, r in ranges.items() if h in needed}
      if hits:
        out.update(hits)
        source = cand_name
        break

    # Map each handle's `actual_view` ("l2.<global>") onto the control-plane bindings.
    global_ranges = (
      _ranges_from(actual_bindings)
      or _ranges_from((compiled or {}).get("globals"))
      or _ranges_from((package or {}).get("globals"))
    )
    for side in ("ins", "outs"):
      for v in kernel.get(side) or binding.get(side) or []:
        handle = str(v.get("handle"))
        if handle in out or handle not in needed:
          continue
        view = str(v.get("actual_view") or "")
        gname = view.split(".", 1)[-1] if "." in view else view
        if gname in global_ranges:
          out[handle] = global_ranges[gname]
          source = source or "control_plane.bindings(actual_view)"

    if set(out) != needed:
      # Permission-ordered fallback: ins handles take readable globals, outs writable ones.
      handles = _view_handles(kernel, binding)
      readable = [n for n, r in global_ranges.items() if "r" in str(r.get("access", ""))]
      writable = [n for n, r in global_ranges.items() if "w" in str(r.get("access", ""))]
      r_i = w_i = 0
      for d in descriptors:
        h = d["handle"]
        if h in out:
          continue
        perm = handles.get(h, "")
        if perm == "r" and r_i < len(readable):
          out[h] = global_ranges[readable[r_i]]
          r_i += 1
        elif perm == "w" and w_i < len(writable):
          out[h] = global_ranges[writable[w_i]]
          w_i += 1
      source = source or "control_plane.bindings(permission-ordered)"

  if set(out) != needed:
    raise ScalarCaseError(
      "SLOTS_UNRESOLVED", f"no L2 range for handles {sorted(needed - set(out))}; pass --l2 <handle>=<base>"
    )
  for handle, rng in out.items():
    if rng["length"] <= 0:
      for d in descriptors:
        if d["handle"] == handle:
          rng["length"] = max(rng["length"], d["slot_offset"] + d["bytes"])
  return out, source


def _find_golden_file(package_dir: Path | None, freeze_dir: Path | None, name: str) -> bytes | None:
  for base in (package_dir, freeze_dir):
    if base is None:
      continue
    for cand in (base / name, base.parent / name):
      if cand.is_file():
        return cand.read_bytes()
  return None


# `linked_slot` authentication is NOT re-implemented here: `load_file` already runs
# `slot_freeze.check_measurements` (image sha/entry/layout/freeze cross-hash) and the semantic
# loader over the wrapped CompiledProgram. Duplicating it would create a second convention.

# ---------------------------------------------------------------------------
# Case loading / execution.
# ---------------------------------------------------------------------------


def load_case(
  *,
  binding_path: Path,
  package_dir: Path | None = None,
  freeze_path: Path | None = None,
  slot: str | None = None,
  config_path: Path | None = None,
  frame_base: int = DEFAULT_FRAME_BASE,
  frame_length: int = DEFAULT_FRAME_LENGTH,
  l2_overrides: dict[str, int] | None = None,
  input_bytes: bytes | None = None,
  output_fill: int = DEFAULT_OUTPUT_FILL,
  expect_hex: bytes | None = None,
  expect_addend: int | None = None,
  event_ordinals: tuple[int, int] = (1, 4095),
  max_steps: int = 4096,
) -> ScalarCase:
  """Resolve everything the real run needs from sealed/binding documents."""
  if not binding_path.is_file():
    raise ScalarCaseError("BINDING_MISSING", str(binding_path))
  binding = yaml.safe_load(binding_path.read_text())

  if package_build is None or load_file is None:
    raise ScalarCaseError("LOADER_UNAVAILABLE", "pipeline_validator/package_build import failed")
  try:
    _ctrl_fields, _ctrl_text, (hw, sim, actual_bindings) = package_build.control_plane(
      str(binding.get("kernel"))
    )
  except Exception as exc:  # BuildError(AUTHORED) for unknown kernels
    raise ScalarCaseError("CONTROL_PLANE_FAILED", f"{type(exc).__name__}: {exc}") from exc

  cfg_path = config_path or (H / "configs" / "cs1_c2.yaml")
  config = yaml.safe_load(cfg_path.read_text()) if cfg_path.is_file() else {}
  target = package_build.independent_target(binding.get("target") or {}, config)

  slot_name = slot or "A"
  freeze_doc: dict[str, Any] | None = None
  freeze_dir: Path | None = None
  package_dir_r: Path | None = package_dir

  if freeze_path is not None:
    if slot_name not in ("A", "B"):
      raise ScalarCaseError("SLOT_UNKNOWN", f"slot must be A or B, got {slot_name!r}")
    freeze_doc = json.loads(freeze_path.read_text())
    freeze_dir = freeze_path.parent
    # The freeze's packages mapping only records identity; the package itself always lives at
    # the known `<freeze_dir>/<slot>/package.json` (`cn package --out dir` layout). A path taken
    # from the unverified index would let a forged index point anywhere.
    package_dir_r = freeze_dir / slot_name
    pkgs = freeze_doc.get("packages") or {}
    if slot_name not in pkgs:
      raise ScalarCaseError("SLOT_MISSING", f"{slot_name} not in slot_freeze.packages")
  elif package_dir_r is not None:
    # Auto-discover the shared freeze the way `cn package --out dir` lays it out.
    for cand in (package_dir_r / "slot_freeze.json", package_dir_r.parent / "slot_freeze.json"):
      if cand.is_file():
        freeze_doc = json.loads(cand.read_text())
        freeze_dir = cand.parent
        break

  if package_dir_r is None or not (package_dir_r / "package.json").is_file():
    raise ScalarCaseError("PACKAGE_MISSING", "pass --package-dir or --freeze")

  # The sealed read-only Loader is the ONLY authentication path (C037): compiled program codec +
  # semantic load_program + image ELF cross-check + slot_freeze commitment. Fails closed.
  package_path = package_dir_r / "package.json"
  compiled_path = package_dir_r / "compiled_program.json"
  image_path = package_dir_r / "image.elf"
  try:
    loaded = load_file(
      package_path,
      compiled_path=compiled_path,
      hw=hw,
      sim=sim,
      actual_bindings=actual_bindings,
      image_path=image_path,
      target=target,
      slot_freeze=freeze_doc,
    )
  except PackageError as exc:
    code = getattr(exc, "code", "PACKAGE_REJECTED")
    if code == "SLOT_FREEZE_MISMATCH" and freeze_doc is None:
      raise ScalarCaseError(
        "SLOT_FREEZE_MISSING", "linked_slot requires slot_freeze.json next to the package (or --freeze)"
      ) from exc
    raise ScalarCaseError(code, f"{getattr(exc, 'field', '$')}: {getattr(exc, 'detail', exc)}") from exc
  package = loaded.to_dict()

  kernel = _kernel_doc(binding, package)
  linked = kernel.get("linked_slot") or {}
  if slot is not None and linked.get("slot") and linked["slot"] != slot_name:
    raise ScalarCaseError(
      "SLOT_MISMATCH", f"requested {slot_name} but package linked_slot={linked['slot']}"
    )
  if freeze_doc is not None:
    pkg_entry = (freeze_doc.get("packages") or {}).get(str(linked.get("slot")))
    if pkg_entry and pkg_entry.get("package_hash") != loaded.package_hash:
      raise ScalarCaseError(
        "FREEZE_IDENTITY_MISMATCH",
        f"freeze packages.{linked.get('slot')}.package_hash != loaded {loaded.package_hash}",
      )
  verified: dict[str, Any] = {
    "loader": {"ok": True, "package_hash": loaded.package_hash},
    "linked_slot": {
      "ok": True,
      "slot": linked.get("slot"),
      "program_id": linked.get("program_id"),
      "freeze_sha256": linked.get("freeze_sha256"),
    },
  }

  descriptors = _launch_descriptors(kernel, binding)
  compiled: dict[str, Any] | None = None
  if compiled_path.is_file():
    compiled = json.loads(compiled_path.read_text())

  bindings, l2_source = _resolve_l2_slots(
    descriptors=descriptors,
    kernel=kernel,
    binding=binding,
    package=package,
    compiled=compiled,
    actual_bindings=actual_bindings,
    overrides=l2_overrides or {},
  )

  wait_events: list[dict[str, Any]] = kernel.get("wait_events") or binding.get("wait_events") or []
  env = {
    "descriptors": descriptors,
    "wait_events": wait_events,
    "bindings": bindings,
    "frame_base": frame_base,
    "frame_length": frame_length,
  }
  ne.scalar_env_from_dict(env)  # fail closed on a malformed env before touching Spike

  if input_bytes is None:
    gi = _find_golden_file(package_dir_r, freeze_dir, "golden_input.bin")
    if gi is None:
      raise ScalarCaseError("INPUT_MISSING", "pass --input-hex/--input-file or ship golden_input.bin")
    input_bytes = gi
  in_words = len(input_bytes) // 4
  in_desc = next((d for d in descriptors if d["direction"] == "l2_to_frame"), None)
  if in_desc is not None and len(input_bytes) != in_desc["bytes"]:
    raise ScalarCaseError(
      "INPUT_LENGTH", f"input {len(input_bytes)}B != descriptor bytes {in_desc['bytes']}"
    )

  out_desc = next((d for d in descriptors if d["direction"] == "frame_to_l2"), None)
  if out_desc is None:
    raise ScalarCaseError("OUTPUT_DESCRIPTOR", "no frame_to_l2 descriptor")
  out_rng = bindings[out_desc["handle"]]
  out_base = out_rng["base"] + out_desc["slot_offset"]
  out_range = {"address": out_base, "length": out_desc["bytes"]}

  golden_file = _find_golden_file(package_dir_r, freeze_dir, "golden_output.bin")
  if expect_hex is not None:
    expected = expect_hex
  elif expect_addend is not None:
    expected = b"".join(
      ((int.from_bytes(input_bytes[4 * i : 4 * (i + 1)], "little") + expect_addend) & 0xFFFFFFFF).to_bytes(
        4, "little"
      )
      for i in range(in_words)
    )
  elif golden_file is not None:
    expected = golden_file
  else:
    raise ScalarCaseError(
      "GOLDEN_MISSING",
      "no expectation: pass --expect-hex/--expect-file/--expect-addend or ship golden_output.bin",
    )
  if len(expected) != out_range["length"]:
    raise ScalarCaseError(
      "GOLDEN_LENGTH", f"expected {len(expected)}B != output range {out_range['length']}B"
    )

  n_desc = len(descriptors)
  env_events: list[dict[str, Any]] = []
  for d in descriptors:
    ordinal = (
      event_ordinals[0]
      if d["index"] == 0
      else (event_ordinals[1] if d["index"] == n_desc - 1 else 2 + d["index"])
    )
    env_events.append(
      {"step_ordinal": ordinal, "kind": "EVENT_DONE", "key": int(d["index"]), "status": "DONE"}
    )

  entry = int((package.get("image") or {}).get("entry", 0))
  return ScalarCase(
    slot=slot_name,
    elf_path=package_dir_r / "image.elf",
    entry_pc=entry,
    package=package,
    binding=binding,
    kernel=kernel,
    descriptors=descriptors,
    bindings=bindings,
    frame_base=frame_base,
    frame_length=frame_length,
    input_bytes=input_bytes,
    expected_bytes=expected,
    golden_file_bytes=golden_file,
    out_range=out_range,
    env_events=env_events,
    l2_source=l2_source,
    freeze_dir=freeze_dir,
    verified=verified,
    wait_events=wait_events,
  )


def _spike_memory_regions(case: ScalarCase) -> list[tuple[int, int]]:
  """Page-aligned, merged backing regions covering the ELF spans, the task frame and L2 slots.

  Spike's default map (spike_simulator.cc) does not back 0x00040000 (frame) or the L2 launch
  windows, so they are declared explicitly — never relying on the ELF's link layout alone.
  """
  spans: list[tuple[int, int]] = [(case.frame_base, case.frame_length)]
  for rng in case.bindings.values():
    spans.append((rng["base"], max(rng["length"], 4)))
  layout = ((case.package or {}).get("image") or {}).get("link_layout") or {}
  for key_base, key_len in (("text_base", "text_bytes"), ("data_base", "data_bytes")):
    base, length = int(layout.get(key_base, 0)), int(layout.get(key_len, 0))
    # base may legitimately be 0 (slot A text); only length decides whether a span exists.
    if length:
      spans.append((base, length))
  page = 0x1000
  merged: list[tuple[int, int]] = []
  for base, length in sorted(spans):
    lo = base & ~(page - 1)
    hi = (base + length + page - 1) & ~(page - 1)
    if merged and lo <= merged[-1][0] + merged[-1][1]:
      start, span_len = merged[-1]
      merged[-1] = (start, max(span_len, hi - start))
    else:
      merged.append((lo, hi - lo))
  return merged


def build_request(
  case: ScalarCase,
  *,
  ctx_slot: int = 0,
  ctx_generation: int = 1,
  max_steps: int = 4096,
  output_fill: int = DEFAULT_OUTPUT_FILL,
) -> srd.TraceRequest:
  """Assemble the TraceRequest: ELF + entry + env + seeded memory, all from the case."""
  in_desc = next(d for d in case.descriptors if d["direction"] == "l2_to_frame")
  in_base = case.bindings[in_desc["handle"]]["base"] + in_desc["slot_offset"]
  out_desc = next(d for d in case.descriptors if d["direction"] == "frame_to_l2")
  out_base = case.bindings[out_desc["handle"]]["base"] + out_desc["slot_offset"]
  initial_memory = [
    {"address": in_base, "hex": case.input_bytes.hex()},
    {"address": out_base, "hex": (bytes([output_fill]) * out_desc["bytes"]).hex()},
    {"address": case.frame_base, "hex": (bytes([0xCD]) * case.frame_length).hex()},
  ]
  read_ranges = [
    {"address": out_base, "length": out_desc["bytes"]},
    {"address": case.frame_base, "length": case.frame_length},
    {"address": in_base, "length": in_desc["bytes"]},
  ]
  return srd.TraceRequest(
    elf_path=str(case.elf_path),
    entry_pc=case.entry_pc,
    ctx_slot=ctx_slot,
    ctx_generation=ctx_generation,
    max_steps=max_steps,
    initial_gpr={"a0": case.frame_base},  # sealed ABI: frame_base arrives at a0
    initial_memory=initial_memory,
    read_memory_ranges=read_ranges,
    enable_nexus_extend=True,
    memory_regions=_spike_memory_regions(case),
    env_events=case.env_events,
    scalar_env={
      "descriptors": case.descriptors,
      "wait_events": case.wait_events,
      "bindings": case.bindings,
      "frame_base": case.frame_base,
      "frame_length": case.frame_length,
    },
  )


def evaluate(
  case: ScalarCase, trace: dict[str, Any], *, dut_doc: dict[str, Any] | None = None
) -> dict[str, Any]:
  """Grade the real run: golden bytes, helper order, exit cleanliness, optional DUT compare."""
  diffs: list[dict[str, Any]] = []
  out_key = f"0x{case.out_range['address']:08x}:{case.out_range['length']}"
  actual_hex = (trace.get("memory_bytes") or {}).get(out_key)
  if actual_hex is None:
    diffs.append({"kind": "OUTPUT_RANGE_NOT_DUMPED", "range": out_key})
    actual = b""
  else:
    actual = bytes.fromhex(actual_hex)
  golden_diffs: list[dict[str, Any]] = []
  for i, (ab, eb) in enumerate(zip(actual, case.expected_bytes)):
    if ab != eb:
      golden_diffs.append(
        {
          "byte_offset": i,
          "actual": f"0x{ab:02x}",
          "expected": f"0x{eb:02x}",
          "bit_diff": om.first_diff_bit(ab, eb, 8),
        }
      )
  if len(actual) != len(case.expected_bytes):
    golden_diffs.append({"length_actual": len(actual), "length_expected": len(case.expected_bytes)})
  golden_match = not golden_diffs

  golden_file_match: bool | None = None
  if case.golden_file_bytes is not None:
    golden_file_match = actual == case.golden_file_bytes

  helpers = ((trace.get("scalar_helpers") or {}).get("helpers")) or []
  kinds = [h["kind"] for h in helpers]
  n_desc = len(case.descriptors)
  expected = [(helper["name"], helper["params"]) for helper in case.kernel["helpers"]]
  actual_helpers = [
    (h["kind"].lower(), {k: v for k, v in h["effect"].items() if k in {"index", "phase", "code"}})
    for h in helpers
  ]
  helper_stream_ok = actual_helpers == expected and kinds[-1:] == ["TASK_RET"]
  settled = sorted((trace.get("scalar_helpers") or {}).get("settled") or [])
  if settled != list(range(n_desc)):
    helper_stream_ok = False
  if not (trace.get("scalar_helpers") or {}).get("task_finished"):
    helper_stream_ok = False
  exit_clean = bool((trace.get("task_exit") or {}).get("clean"))

  waits = [h for h in helpers if h["kind"] == "WAIT"]
  parked_witness = any((h.get("effect") or {}).get("parked") for h in waits)
  direct_witness = any((h.get("effect") or {}).get("parked") is False for h in waits)

  retire = trace.get("retire_stream") or []
  last = retire[-1] if retire else {}
  last_is_task_ret = (
    last.get("source") == "nexus_extend"
    and (trace.get("scalar_helpers") or {}).get("task_finished") is True
  )

  dut_diffs: list[dict[str, Any]] | None = None
  if dut_doc is not None:
    dut_diffs = om.compare_retire_and_memory(
      oracle_trace={"retire_stream": retire, "memory_bytes": trace.get("memory_bytes") or {}},
      dut_trace=dut_doc,
    )

  if not golden_match:
    diffs.append({"kind": "GOLDEN_MISMATCH", "first": golden_diffs[0] if golden_diffs else None})
  if golden_file_match is False:
    diffs.append({"kind": "GOLDEN_FILE_MISMATCH"})
  if not helper_stream_ok:
    diffs.append({"kind": "HELPER_STREAM", "actual": actual_helpers, "expected": expected})
  if not exit_clean:
    diffs.append({"kind": "TASK_EXIT_UNCLEAN", "task_exit": trace.get("task_exit")})
  if not last_is_task_ret:
    diffs.append({"kind": "LAST_RETIRE_NOT_TASK_RET", "last": last.get("insn")})
  if dut_diffs:
    diffs.append({"kind": "DUT_MISMATCH", "first": dut_diffs[0]})

  verdict = "PASS" if not diffs else "FAIL"
  return {
    "schema": "cn_scalar_ctl_run/v1",
    "verdict": verdict,
    "slot": case.slot,
    "package_hash": (case.package or {}).get("package_hash"),
    "image_sha256": (case.package or {}).get("image", {}).get("bytes_sha256"),
    "entry": (case.package or {}).get("image", {}).get("entry"),
    "l2_source": case.l2_source,
    "verified": case.verified,
    "golden": {
      "match": golden_match,
      "golden_file_match": golden_file_match,
      "out_range": out_key,
      "expected_sha256": _sha(case.expected_bytes),
      "actual_sha256": _sha(actual) if actual else None,
      "diffs": golden_diffs[:8],
    },
    "helper_stream": {
      "ok": helper_stream_ok,
      "kinds": kinds,
      "settled": settled,
      "signal_phases": (trace.get("scalar_helpers") or {}).get("signal_phases"),
      "parked_witness": parked_witness,
      "direct_witness": direct_witness,
    },
    "task_exit": trace.get("task_exit"),
    "steps_executed": trace.get("steps_executed"),
    "halted": trace.get("halted"),
    "retire_records": len(retire),
    "dut_diffs": dut_diffs,
    "diffs": diffs,
    "build_meta": trace.get("build_meta"),
    "nexus_csr_values": trace.get("nexus_csr_values"),
  }


def run_case(case: ScalarCase, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
  """Execute the case on the pinned Spike reference and grade it."""
  req = build_request(
    case,
    **{k: v for k, v in kwargs.items() if k in ("ctx_slot", "ctx_generation", "max_steps", "output_fill")},
  )
  trace = srd.run_trace_request(req)
  report = evaluate(case, trace, dut_doc=kwargs.get("dut_doc"))
  return report, trace


def _parse_int(text: str) -> int:
  return int(text, 0)


def main() -> int:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--binding", required=True, help="binding yaml (scalar_ctl_v1.yaml)")
  src = parser.add_mutually_exclusive_group(required=True)
  src.add_argument("--package-dir", help="dir containing package.json/image.elf/compiled_program.json")
  src.add_argument("--freeze", help="slot_freeze.json; requires --slot")
  parser.add_argument("--slot", default="A", help="slot name inside slot_freeze.json (A|B)")
  parser.add_argument("--config", type=Path, help="cs1_target config yaml for the Loader target")
  parser.add_argument("--frame-base", type=_parse_int, default=DEFAULT_FRAME_BASE)
  parser.add_argument("--frame-length", type=_parse_int, default=DEFAULT_FRAME_LENGTH)
  parser.add_argument(
    "--l2",
    action="append",
    default=[],
    metavar="HANDLE=HEX",
    help="override L2 slot base, repeatable (e.g. --l2 input=0x100000 --l2 output=0x200000)",
  )
  parser.add_argument("--l2-length", action="append", default=[], metavar="HANDLE=INT")
  parser.add_argument("--input-hex", help="input bytes as hex (else golden_input.bin)")
  parser.add_argument("--input-file", type=Path)
  parser.add_argument("--output-fill", type=_parse_int, default=DEFAULT_OUTPUT_FILL)
  exp = parser.add_mutually_exclusive_group()
  exp.add_argument("--expect-hex", help="expected output bytes as hex")
  exp.add_argument("--expect-file", type=Path)
  exp.add_argument("--expect-addend", type=int, help="expected = input words + addend (i32 LE)")
  parser.add_argument("--dut-retire", type=Path, help="optional DUT retire/memory json to compare")
  parser.add_argument("--ctx-slot", type=int, default=0)
  parser.add_argument("--ctx-generation", type=int, default=1)
  parser.add_argument("--steps", type=int, default=4096)
  parser.add_argument("--out", required=True, type=Path, help="write the JSON report here")
  parser.add_argument("--trace-out", type=Path, help="optionally dump the raw trace json")
  args = parser.parse_args()

  overrides: dict[str, int] = {}
  for item in args.l2:
    name, _, val = item.partition("=")
    overrides[name] = _parse_int(val)
  for item in args.l2_length:
    name, _, val = item.partition("=")
    overrides[f"{name}:length"] = int(val, 0)

  input_bytes: bytes | None = None
  if args.input_hex:
    input_bytes = bytes.fromhex(args.input_hex)
  elif args.input_file:
    input_bytes = args.input_file.read_bytes()

  expect_hex: bytes | None = None
  if args.expect_hex:
    expect_hex = bytes.fromhex(args.expect_hex)
  elif args.expect_file:
    expect_hex = args.expect_file.read_bytes()

  case = load_case(
    binding_path=Path(args.binding),
    package_dir=Path(args.package_dir) if args.package_dir else None,
    freeze_path=Path(args.freeze) if args.freeze else None,
    slot=args.slot,
    config_path=args.config,
    frame_base=args.frame_base,
    frame_length=args.frame_length,
    l2_overrides=overrides,
    input_bytes=input_bytes,
    output_fill=args.output_fill,
    expect_hex=expect_hex,
    expect_addend=args.expect_addend,
    max_steps=args.steps,
  )
  dut_doc = json.loads(args.dut_retire.read_text()) if args.dut_retire else None
  report, trace = run_case(
    case,
    ctx_slot=args.ctx_slot,
    ctx_generation=args.ctx_generation,
    max_steps=args.steps,
    output_fill=args.output_fill,
    dut_doc=dut_doc,
  )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  if args.trace_out:
    args.trace_out.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": report["verdict"], "out": str(args.out)}, sort_keys=True))
  return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
