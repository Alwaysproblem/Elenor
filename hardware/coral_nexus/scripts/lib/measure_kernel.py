"""Measure bounded resources of the bootstrap kernel image (C018 / W5-02).

What is measured, from what:

* text_bytes / static_data_bytes -- the real section sizes of the actually linked ELF (``llvm-size -A``),
  never the abstract-IR estimate the W5 contract rejects;
* stack_bytes -- a PROVABLE bound from the per-function frame sizes of the real build (``-fstack-usage``)
  closed over the real call graph of the real disassembly, with every construct that would leave the call
  graph open refused (indirect calls, recursion, VLAs/alloca, stack-adjusting sequences the parser does not
  model). A measured peak is evidence, not a bound: nothing dynamic is summed here;
* scratch_bytes -- 0 for this image; the bootstrap has no scratch section. A later kernel with scratch
  declares it with ``bound_source: static_analysis`` per the W5 contract, and outgrowing this candidate
  means a NEW target hash, never widening this one.

The report is written for the C014 package consumer: ``bound_source`` is always ``static_analysis``, and the
two candidate link layouts are proven coexistent by ``package_contract.layouts_coexist`` in the tests. The
record carries the toolchain lock sha256 and the source digests; capacity or overlap failures refuse the
record instead of trimming it (a failure goes back to the build step, never to a Loader patch).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

H = Path(__file__).resolve().parents[2]
REPO = H.parents[1]
BOOTSTRAP = H / "kernels/bootstrap"
TOOLCHAIN = REPO / "out/coral_nexus/cache/toolchain/extracted/toolchain_coralnpu_v2/bin"
OUT_ROOT = REPO / "out/coral_nexus/C018"
STACK_BYTES = 128  # upstream DEFAULT stack_size_bytes; a candidate target may differ -> new target hash
ALIGN = 16  # W5 contract: fixed pre-link images; both candidate bases stay 16-byte aligned
ITCM_ORIGIN = 0x00000000
DTCM_ORIGIN = 0x00010000  # pinned upstream default (rules/linker.bzl dtcm_origin_default), never guessed

# Slot offsets INSIDE the pinned regions: A at the start of ITCM/DTCM, B one 4 KiB slot up. The slot
# size is a candidate planning bound, not a hardware constant; C019 re-checks capacity from the record.
_LAYOUTS: tuple[dict[str, Any], ...] = (
  {"name": "A", "text_base": 0x0000, "data_base": 0x0000},
  {"name": "B", "text_base": 0x1000, "data_base": 0x1000},
)

_RE_CALL = re.compile(r"^\s*[0-9a-f]+:.*\b(jal|j)\s+0x([0-9a-f]+)\s+<([^>+]+)")
_RE_TAIL = re.compile(r"^\s*[0-9a-f]+:.*\bjr\s+\S+\(\w+\)\s+<([^>+]+)>")
_RE_INDIRECT = re.compile(r"^\s*[0-9a-f]+:.*\b(jalr|jr)\s")
_RE_FRAME = re.compile(r"^\s*[0-9a-f]+ <([^>+]+)>:")
_RE_ADDISP = re.compile(r"\baddi\s+sp,\s*sp,\s*-0x([0-9a-f]+)")
_RE_ADDISP_POS = re.compile(r"\baddi\s+sp,\s*sp,\s*0x([0-9a-f]+)")
# Every way an instruction can WRITE sp. The frame model understands only `addi sp, sp, +/-imm`; anything
# else that targets sp (sub/add/mv/li/lw/auipc/csrr ... sp) would silently move the stack, so it is refused.
_RE_SP_WRITE = re.compile(
  r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(?!sw\b|sh\b|sb\b|c\.sw|c\.swsp)(\w[\w.]*)\s+sp\s*,"
)
_RE_CSRW = re.compile(r"\bcsrw\s+(csr\s+)?(0x[0-9a-f]+|\d+)\b")


class MeasureError(Exception):
  """Carries a coded reason so tests can tell the classes apart."""

  def __init__(self, code: str, detail: str) -> None:
    super().__init__(f"{code}: {detail}")
    self.code = code
    self.detail = detail


def _run(cmd: list[str], **kw: Any) -> str:
  proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
  if proc.returncode != 0:
    raise MeasureError("TOOL_FAILED", f"{' '.join(cmd[:3])}: {proc.stderr[-300:]}")
  return proc.stdout


def _parse_disassembly(dis: str) -> tuple[dict[str, list[str]], dict[str, int], list[int]]:
  """Call graph (jal/j targets), per-function max frame, and every csrw immediate.

  A positive sp adjust is the epilogue of the SAME function only while a negative adjust of at least the
  same size is outstanding; anything else (sp restored past its own frame, sp used as scratch) leaves the
  frame model open and is refused instead of guessed. `_start` initializes sp, so its initial positive
  setup is legal.
  """
  calls: dict[str, list[str]] = {}
  frames: dict[str, int] = {}
  csrws: list[int] = []
  depth = 0
  current = "?"
  for line in dis.splitlines():
    head = _RE_FRAME.match(line)
    if head:
      current = head.group(1)
      frames.setdefault(current, 0)
      depth = 0
      continue
    call = _RE_CALL.match(line)
    tail = _RE_TAIL.search(line)
    if call:
      calls.setdefault(current, []).append(call.group(3))
    elif tail:
      # a relaxed tail call names its target; a jr with no annotation is an open indirect call
      calls.setdefault(current, []).append(tail.group(1))
    elif _RE_INDIRECT.search(line):
      raise MeasureError("UNRESOLVED_INDIRECT", f"{current}: an indirect call the bound cannot close")
    grow = _RE_ADDISP.search(line)
    shrink = None if grow else _RE_ADDISP_POS.search(line)
    if not grow and not shrink:
      written = _RE_SP_WRITE.match(line)
      # the ONLY legal non-frame sp write: _start's `la sp, __stack_end` (auipc sp + addi sp, sp, +imm)
      if written and not (current == "_start" and written.group(1) == "auipc"):
        raise MeasureError(
          "UNMODELED_STACK", f"{current}: an sp update the frame model does not cover: {line.strip()}"
        )
    if grow:
      depth += int(grow.group(1), 16)
      frames[current] = max(frames[current], depth)
    elif shrink:
      release = int(shrink.group(1), 16)
      if release > depth and not (current == "_start" and depth == 0):
        raise MeasureError(
          "UNMODELED_STACK", f"{current}: restores {release} with only {depth} outstanding"
        )
      depth -= release
    csr = _RE_CSRW.search(line)
    if csr:
      csrws.append(int(csr.group(2), 0))
  return calls, frames, csrws


def _parse_su(su_text: str) -> dict[str, int]:
  """Compiler frame sizes from -fstack-usage (`file:line:func<TAB>bytes<TAB>static`); else refused."""
  sizes: dict[str, int] = {}
  for su_line in su_text.splitlines():
    if not su_line.strip():
      continue
    parts = su_line.split("\t")
    if len(parts) != 3 or not parts[1].isdigit():
      raise MeasureError("BAD_STACK_USAGE", f"unparseable .su line {su_line!r}")
    if parts[2] != "static":
      raise MeasureError("UNBOUNDED_FRAME", f"{parts[0]}: {parts[2]} stack usage has no bound")
    func = parts[0].rsplit(":", 1)[-1]
    if func in sizes:
      raise MeasureError("BAD_STACK_USAGE", f"{func} appears twice in the .su file")
    sizes[func] = int(parts[1])
  return sizes


def _stack_bound(su_text: str, calls: dict[str, list[str]], frames: dict[str, int]) -> int:
  """Sum the COMPILER's frame sizes along the longest acyclic call path from _start.

  The .su value is the authority; the parsed disassembly frame must not exceed it (a larger parsed frame
  means the compiler under-reported), and a function reachable from _start with no .su frame and a non-zero
  parsed frame, or a call target the disassembly never defined, is refused instead of counted as zero.
  Assembly entry stubs (no .su record) are legal only when their parsed frame is 0.
  """
  sizes = _parse_su(su_text)
  memo: dict[str, int] = {}
  visiting: set[str] = set()

  def frame_of(fn: str) -> int:
    if fn not in frames:
      raise MeasureError("UNKNOWN_FUNCTION", f"{fn} is called but never defined in the image")
    parsed = frames[fn]
    if fn in sizes:
      if parsed > sizes[fn]:
        raise MeasureError(
          "FRAME_MISMATCH", f"{fn}: disassembly frame {parsed} exceeds the compiler's {sizes[fn]}"
        )
      return sizes[fn]
    if parsed != 0:
      raise MeasureError("FRAME_MISMATCH", f"{fn}: {parsed}-byte frame has no .su record to check it")
    return 0

  def depth(fn: str) -> int:
    if fn in memo:
      return memo[fn]
    if fn in visiting:
      raise MeasureError("RECURSION", fn)
    visiting.add(fn)
    total = frame_of(fn) + max((depth(t) for t in calls.get(fn, ())), default=0)
    visiting.discard(fn)
    memo[fn] = total
    return total

  return depth("_start")


_RE_SECTION = re.compile(r"^\s*\[\s*\d+\]\s+(\.[\w.]+)\s+PROGBITS\s+([0-9a-f]+)")


def _section_vmas(readelf_sw: str) -> dict[str, int]:
  """Actual section VMAs from readelf -SW: the ELF is the authority, not the .ld intent."""
  matches = (_RE_SECTION.match(line) for line in readelf_sw.splitlines())
  return {m.group(1): int(m.group(2), 16) for m in matches if m}


def _verify_upstream_regions() -> None:
  """The region bases are the pinned upstream generator's, read back -- never a local guess."""
  linker = H / "upstream/coralnpu/rules/linker.bzl"
  text = linker.read_text()
  if 'dtcm_origin_default = "0x00010000"' not in text:
    raise MeasureError("REGION_DRIFT", "upstream DTCM origin default changed; re-derive the layout")


def build_layout(layout: dict[str, Any], out: Path) -> dict[str, int]:
  """Fill the layout template and build the real ELF with the pinned toolchain."""
  tpl = (BOOTSTRAP / "bootstrap_layout.ld.tpl").read_text()
  script = (
    tpl.replace("@@TEXT_BASE@@", f"0x{layout['text_base']:08x}")
    .replace("@@DATA_BASE@@", f"0x{layout['data_base']:08x}")
    .replace("@@STACK_BYTES@@", str(STACK_BYTES))
  )
  _verify_upstream_regions()
  build_dir = out / layout["name"]
  build_dir.mkdir(parents=True, exist_ok=True)
  (build_dir / "bootstrap.ld").write_text(script)
  _run(
    [
      "bash",
      str(H / "scripts/lib/build_bootstrap.sh"),
      str(TOOLCHAIN),
      str(BOOTSTRAP),
      str(H / "protocol/generated"),
      str(build_dir),
      "bootstrap",
    ]
  )
  sizes = _section_sizes((build_dir / "bootstrap.size").read_text())
  sections = _run([str(TOOLCHAIN / "riscv64-unknown-elf-readelf"), "-SW", str(build_dir / "bootstrap.elf")])
  for line in sections.splitlines():
    parts = line.split()
    if len(parts) >= 5 and parts[1].lstrip(".").rstrip("]").isdigit() is False and "PROGBITS" in line:
      pass
  vmas = _section_vmas(sections)
  expected_text = ITCM_ORIGIN + int(layout["text_base"])
  expected_data = DTCM_ORIGIN + int(layout["data_base"])
  if vmas.get(".text", -1) != expected_text:
    raise MeasureError("VMA_MISMATCH", f".text at {vmas.get('.text', -1):#x}, expected {expected_text:#x}")
  if vmas.get(".data", -1) not in (expected_data, -1):
    raise MeasureError("VMA_MISMATCH", f".data at {vmas.get('.data')!r}, expected near {expected_data:#x}")
  stack_at = vmas.get(".stack")
  if stack_at is not None and stack_at < expected_data:
    raise MeasureError("VMA_MISMATCH", f".stack at {stack_at:#x} is below the data base {expected_data:#x}")
  return sizes


def _section_sizes(size_text: str) -> dict[str, int]:
  text = data = 0
  for line in size_text.splitlines():
    parts = line.split()
    if len(parts) == 3 and parts[0].startswith("."):
      name, size = parts[0], int(parts[1])
      if name.startswith((".text", ".rodata", ".init")):
        text += size
      if name.startswith((".data", ".sdata", ".bss", ".sbss")):
        data += size
  if text <= 0:
    raise MeasureError("NO_TEXT", "the linked image has no text")
  return {"text": text, "data": data}


def measure(out: Path = OUT_ROOT) -> dict[str, Any]:
  """Build both candidate layouts, bound their resources, and write the C018 measurement record."""
  if not TOOLCHAIN.is_dir():
    raise MeasureError("NO_TOOLCHAIN", f"{TOOLCHAIN} missing (cn env seed pulls the locked archive)")
  out.mkdir(parents=True, exist_ok=True)
  sizes_a = build_layout(_LAYOUTS[0], out)
  build_a = out / "A"
  dis = (build_a / "bootstrap.dis").read_text()
  su = (build_a / "bootstrap.su").read_text() if (build_a / "bootstrap.su").exists() else ""
  if not su:
    su = _stack_usage_fallback(build_a)
  calls, frames, csrws = _parse_disassembly(dis)
  stack = _stack_bound(su, calls, frames)
  _check_helper_csrs(csrws, _loads((H / "protocol/generated/cn_layout.json").read_text()))
  elf = (build_a / "bootstrap.elf").read_bytes()
  entry = _entry_address(dis)
  record_layouts = []
  for layout in _LAYOUTS:
    if layout["name"] != "A":
      _ = build_layout(layout, out)
      size_path = out / str(layout["name"]) / "bootstrap.size"
      sec = _section_sizes(size_path.read_text())
      text, data = sec["text"], sec["data"]
    else:
      text, data = sizes_a["text"], sizes_a["data"]
    record_layouts.append(
      {
        "name": layout["name"],
        "text_base": ITCM_ORIGIN + layout["text_base"],
        "text_bytes": text,
        "data_base": DTCM_ORIGIN + layout["data_base"],
        "data_bytes": data,
        "stack_bytes": STACK_BYTES,
        "align": ALIGN,
      }
    )
  record: dict[str, Any] = {
    "schema": "cn_kernel_measurement/v1",
    "kernel": "bootstrap",
    "purpose": "measure-only; not a loadable CS1 package",
    "resource_manifest": {
      "text_bytes": sizes_a["text"],
      "static_data_bytes": sizes_a["data"],
      "stack_bytes": stack,
      "scratch_bytes": 0,
      "bound_source": "static_analysis",
    },
    "candidate_layouts": record_layouts,
    "capacities": {
      "itcm_bytes": 8192,
      "dtcm_bytes": 32768,
      "source": "w5_package.contract.yaml image.capacities",
    },
    "entry": entry,
    "csr_immediates": {"distinct": len(set(csrws)), "max": max(csrws)},
    "provenance": {
      "toolchain": "toolchain_coralnpu_v2",
      "toolchain_lock_sha256": _lock_sha(),
      "source_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(BOOTSTRAP.iterdir())
        if p.is_file()
      },
      "elf_sha256_layout_A": hashlib.sha256(elf).hexdigest(),
    },
    "checks": _checks(record_layouts, stack),
  }
  errors = [c for c in record["checks"] if not c["ok"]]
  if errors:
    raise MeasureError("CAPACITY", "; ".join(f"{c['name']}: {c['detail']}" for c in errors))
  record_text = _json(record)
  (out / "kernel_measurement.json").write_text(record_text)
  # the committed evidence record C019 consumes; regeneration must be byte-identical
  committed = H / "verification/kernel_bootstrap/kernel_measurement.json"
  committed.parent.mkdir(parents=True, exist_ok=True)
  committed.write_text(record_text)
  return record


def _check_helper_csrs(csrws: list[int], generated: dict[str, Any]) -> None:
  """Every helper csrw immediate must be a C017-generated mregion address, nothing else."""
  expected = set(csrws)
  want = {r["address"] for r in generated["csr"] if r["name"].startswith("nexus.mregion.")}
  if expected != want:
    raise MeasureError(
      "HELPER_CSR_MISMATCH",
      f"csrw immediates {sorted(hex(x) for x in expected)} != C017 mregion allocation "
      f"{sorted(hex(x) for x in want)}",
    )


def _stack_usage_fallback(build: Path) -> str:
  """Recover .su output: the builder passes -fstack-usage, so the file sits next to the object."""
  su = build / "bootstrap.su"
  if su.exists():
    return su.read_text()
  raise MeasureError("NO_STACK_USAGE", str(su))


def _entry_address(dis: str) -> int:
  for line in dis.splitlines():
    head = _RE_FRAME.match(line)
    if head and head.group(1) == "_start":
      return int(line.split("<")[0].strip().rstrip(":"), 16)
  raise MeasureError("NO_ENTRY", "_start not found in the disassembly")


def _lock_sha() -> str:
  lock = H / "env/tools.lock"
  data = __import__("json").loads(lock.read_text())
  return str(data["rv32_toolchain"]["sha256"])


def _checks(layouts: list[dict[str, Any]], stack: int) -> list[dict[str, Any]]:
  itcm, dtcm = 8192, 32768
  out = []
  for a in layouts:
    text_range = (a["text_base"], a["text_base"] + a["text_bytes"])
    data_range = (a["data_base"], a["data_base"] + a["data_bytes"] + a["stack_bytes"])
    clash = text_range[0] < data_range[1] and data_range[0] < text_range[1]
    out.append(
      {
        "name": f"regions_disjoint_{a['name']}",
        "ok": not clash,
        "detail": f"text {text_range} vs data+stack {data_range}",
      }
    )
    inside = (
      ITCM_ORIGIN <= a["text_base"]
      and text_range[1] <= ITCM_ORIGIN + itcm
      and DTCM_ORIGIN <= a["data_base"]
      and data_range[1] <= DTCM_ORIGIN + dtcm
    )
    out.append(
      {
        "name": f"regions_fit_{a['name']}",
        "ok": inside,
        "detail": f"slot inside ITCM[{ITCM_ORIGIN:#x},+{itcm}) DTCM[{DTCM_ORIGIN:#x},+{dtcm})",
      }
    )
  for a in layouts:
    for b in layouts:
      if a["name"] < b["name"]:
        overlap = (
          a["text_base"] < b["text_base"] + b["text_bytes"]
          and b["text_base"] < a["text_base"] + a["text_bytes"]
        )
        # data ranges include their stack reservation: a stack may not collide with the
        # other image's data any more than data may collide with data
        overlap = overlap or (
          a["data_base"] < b["data_base"] + b["data_bytes"] + max(a["stack_bytes"], b["stack_bytes"])
          and b["data_base"] < a["data_base"] + a["data_bytes"] + max(a["stack_bytes"], b["stack_bytes"])
        )
        out.append(
          {
            "name": f"coexist_{a['name']}_{b['name']}",
            "ok": not overlap,
            "detail": f"text/data ranges of {a['name']} and {b['name']} overlap" if overlap else "disjoint",
          }
        )
  text_end = max(x["text_base"] + x["text_bytes"] for x in layouts)
  itcm_end = ITCM_ORIGIN + itcm
  out.append(
    {
      "name": "itcm_capacity",
      "ok": text_end <= itcm_end,
      "detail": f"text ends at {text_end} of {itcm_end}",
    }
  )
  data_end = max(x["data_base"] + x["data_bytes"] + STACK_BYTES for x in layouts)
  dtcm_end = DTCM_ORIGIN + dtcm
  out.append(
    {
      "name": "dtcm_capacity",
      "ok": data_end <= dtcm_end,
      "detail": f"data+stack ends at {data_end} of {dtcm_end}",
    }
  )
  reserved = min(x["stack_bytes"] for x in layouts)
  out.append(
    {
      "name": "stack_reservation_covers_bound",
      "ok": stack <= reserved,
      "detail": f"call-graph bound {stack} vs reserved {reserved}",
    }
  )
  return out


def _loads(text: str) -> Any:
  import json

  return json.loads(text)


def _json(record: dict[str, Any]) -> str:
  import json

  return json.dumps(record, indent=2, sort_keys=False) + "\n"


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", default=str(OUT_ROOT))
  args = parser.parse_args()
  try:
    record = measure(Path(args.out))
  except MeasureError as exc:
    print(f"measure_kernel: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
  print(_json(record))


if __name__ == "__main__":
  main()
