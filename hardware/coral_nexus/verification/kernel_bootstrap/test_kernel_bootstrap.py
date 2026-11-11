"""C018 bootstrap kernel build + static resource measurement tests.

What is proven here:

* a REAL ELF is built by the pinned C004 toolchain (clang 20, rv32imf_zve32f.../ilp32, ``-nostdlib``) from
  the helper sources, and it is ELF32 RISC-V;
* the measured ``resource_manifest`` comes from the linked image itself (``llvm-size -A``) and a provable
  stack bound (per-function ``-fstack-usage`` frames closed over the real call graph of the real
  disassembly) -- never a dynamic peak;
* the helper CSR writes in the emitted code are exactly the C017-generated ``nexus.mregion.*`` addresses,
  in the declared control-sequence order (no handwritten numeric address anywhere);
* two candidate link layouts coexist: disjoint ranges, each inside ITCM/DTCM capacity, and the C014
  package consumer's own ``layouts_coexist`` agrees;
* everything that would break the proof is refused: capacity overrun, overlapping layouts, recursion,
  dynamic stack usage, an unmodeled sp adjust, an indirect call the bound cannot close, a bound not from
  static analysis, no text at all, and a missing toolchain.

The whole suite is skipped (with the reason stated) when the pinned toolchain has not been extracted, so a
fresh clone without ``cn env`` stays green without faking a measurement.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
H = REPO_ROOT / "hardware/coral_nexus"
sys.path.insert(0, str(H / "scripts/lib"))

import measure_kernel as mk  # noqa: E402
import package_contract as pc  # noqa: E402

HAVE_TOOLCHAIN = mk.TOOLCHAIN.is_dir()
TOOLCHAIN = mk.TOOLCHAIN


# ---------------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
  out = REPO_ROOT / "out/coral_nexus/C018/test_measure"
  return mk.measure(out)


@pytest.fixture(scope="module")
def build_dir(record: dict[str, Any]) -> Path:
  return REPO_ROOT / "out/coral_nexus/C018/test_measure/A"


def _layout_pkg(layout: dict[str, Any]) -> dict[str, Any]:
  return {
    "image": {
      "link_layout": {
        "text_base": layout["text_base"],
        "text_bytes": layout["text_bytes"],
        "data_base": layout["data_base"],
        "data_bytes": layout["data_bytes"],
      }
    }
  }


# ---------------------------------------------------------------------------------------------------
# the real build
# ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(
  not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted under out/coral_nexus/cache"
)
def test_the_image_is_a_real_elf32_riscv_binary(build_dir: Path) -> None:
  header = subprocess.run(
    [str(TOOLCHAIN / "riscv64-unknown-elf-readelf"), "-h", str(build_dir / "bootstrap.elf")],
    capture_output=True,
    text=True,
    check=True,
  ).stdout
  assert "Class:                             ELF32" in header
  assert "Machine:" in header and "RISC-V" in header


@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted")
def test_text_and_data_come_from_the_linked_image_not_an_estimate(record: dict[str, Any]) -> None:
  size = (REPO_ROOT / "out/coral_nexus/C018/test_measure/A/bootstrap.size").read_text()
  text = sum(int(ln.split()[1]) for ln in size.splitlines() if ln.split()[:1] == [".text"])
  rm = record["resource_manifest"]
  assert rm["text_bytes"] == text > 0
  assert rm["static_data_bytes"] >= 0
  assert rm["scratch_bytes"] == 0
  assert rm["bound_source"] == "static_analysis"


@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted")
def test_the_stack_bound_is_provable_not_measured(build_dir: Path, record: dict[str, Any]) -> None:
  su = (build_dir / "bootstrap.su").read_text()
  frames = {ln.split("\t")[0].rsplit(":", 1)[-1]: int(ln.split("\t")[1]) for ln in su.splitlines()[1:]}
  assert all(ln.split("\t")[2] == "static" for ln in su.splitlines()[1:])
  bound = record["resource_manifest"]["stack_bytes"]
  assert bound == max(frames.values())  # the deepest single frame IS the bound: every helper is a leaf
  assert bound <= record["candidate_layouts"][0]["stack_bytes"] <= record["capacities"]["dtcm_bytes"]


@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted")
def test_every_helper_csr_write_uses_a_generated_address(build_dir: Path) -> None:
  import json

  layout = json.loads((H / "protocol/generated/cn_layout.json").read_text())
  want = {r["address"]: r["name"] for r in layout["csr"] if r["name"].startswith("nexus.mregion.")}
  dis = (build_dir / "bootstrap.dis").read_text()
  written = [int(x, 16) for x in __import__("re").findall(r"csrw\s+0x([0-9a-f]+)", dis)]
  assert sorted(set(written)) == sorted(want), (sorted(hex(x) for x in set(written)), want)
  # execution order = the call sequence inside cn_bootstrap_entry (helpers are noinline leaves)
  entry = dis.split("<cn_bootstrap_entry>:", 1)[1]
  called = __import__("re").findall(r"\b(?:jal|j)\s+0x[0-9a-f]+\s+<([^>+]+)>", entry)
  assert called == [
    "cn_region_start",
    "cn_chain_first",
    "cn_chain_continue",
    "cn_chain_final",
    "cn_region_end",
  ]
  for path in (H / "kernels/bootstrap").iterdir():
    if path.suffix in (".c", ".h", ".S"):
      assert "0x7c" not in path.read_text(), f"{path.name}: a handwritten CSR address is forbidden"


@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted")
def test_two_candidate_layouts_coexist_inside_capacity(record: dict[str, Any]) -> None:
  a, b = record["candidate_layouts"]
  assert a is not b
  # the C014 package consumer's own coexistence proof agrees with the record
  pc.layouts_coexist(_layout_pkg(a), _layout_pkg(b))
  itcm, dtcm = mk.ITCM_ORIGIN, mk.DTCM_ORIGIN
  for layout in (a, b):
    assert layout["text_base"] % layout["align"] == 0
    assert layout["data_base"] % layout["align"] == 0
    assert itcm <= layout["text_base"]
    assert layout["text_base"] + layout["text_bytes"] <= itcm + record["capacities"]["itcm_bytes"]
    assert dtcm <= layout["data_base"]
    assert (
      layout["data_base"] + layout["data_bytes"] + layout["stack_bytes"]
      <= dtcm + record["capacities"]["dtcm_bytes"]
    )
    # within one image, text and data+stack live in different regions and never overlap
    assert (
      layout["text_base"] + layout["text_bytes"] <= layout["data_base"]
      or layout["data_base"] + layout["data_bytes"] + layout["stack_bytes"] <= layout["text_base"]
    )
  assert all(check["ok"] for check in record["checks"])


@pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="pinned C004 toolchain not extracted")
def test_the_record_carries_its_provenance(record: dict[str, Any]) -> None:
  prov = record["provenance"]
  import json

  lock = json.loads((H / "env/tools.lock").read_text())
  assert prov["toolchain"] == "toolchain_coralnpu_v2"
  assert prov["toolchain_lock_sha256"] == lock["rv32_toolchain"]["sha256"]
  assert set(prov["source_sha256"]) == {
    "bootstrap_control.c",
    "bootstrap_helpers.h",
    "bootstrap_layout.ld.tpl",
    "bootstrap_start.S",
  }
  import hashlib

  for name, digest in prov["source_sha256"].items():
    assert hashlib.sha256((H / "kernels/bootstrap" / name).read_bytes()).hexdigest() == digest


# ---------------------------------------------------------------------------------------------------
# refusals: everything that would break the bound or the capacity proof
# ---------------------------------------------------------------------------------------------------
def _expect(code: str, fn: Any, *args: Any) -> None:
  with pytest.raises(mk.MeasureError) as exc:
    fn(*args)
  assert exc.value.code == code


def test_capacity_and_overlap_failures_are_refused() -> None:
  layouts = [
    {
      "name": "A",
      "text_base": 0,
      "text_bytes": 6000,
      "data_base": 0x10000,
      "data_bytes": 0,
      "stack_bytes": 128,
    },
    {
      "name": "B",
      "text_base": 4096,
      "text_bytes": 6000,
      "data_base": 0x11000,
      "data_bytes": 0,
      "stack_bytes": 128,
    },
  ]
  errors = [c for c in mk._checks(layouts, 16) if not c["ok"]]
  assert {c["name"] for c in errors} == {"coexist_A_B", "itcm_capacity", "regions_fit_B"}
  # a data base below the DTCM origin is refused even when the sizes are tiny
  below_dtcm = [
    {
      "name": "A",
      "text_base": 0,
      "text_bytes": 8,
      "data_base": 0x8000,
      "data_bytes": 64,
      "stack_bytes": 16,
    },
    {
      "name": "B",
      "text_base": 4096,
      "text_bytes": 8,
      "data_base": 0x10000,
      "data_bytes": 0,
      "stack_bytes": 16,
    },
  ]
  assert {c["name"] for c in mk._checks(below_dtcm, 16) if not c["ok"]} == {"regions_fit_A"}
  tiny_stack = [
    {"name": "A", "text_base": 0, "text_bytes": 8, "data_base": 0x10000, "data_bytes": 0, "stack_bytes": 8},
    {
      "name": "B",
      "text_base": 4096,
      "text_bytes": 8,
      "data_base": 0x11000,
      "data_bytes": 0,
      "stack_bytes": 8,
    },
  ]
  assert {c["name"] for c in mk._checks(tiny_stack, 16) if not c["ok"]} == {
    "stack_reservation_covers_bound"
  }
  # a stack colliding with the other image's data is an overlap, not just a tight fit
  stack_clash = [
    {
      "name": "A",
      "text_base": 0,
      "text_bytes": 8,
      "data_base": 0x10000,
      "data_bytes": 0,
      "stack_bytes": 0x1080,
    },
    {
      "name": "B",
      "text_base": 4096,
      "text_bytes": 8,
      "data_base": 0x10400,
      "data_bytes": 8,
      "stack_bytes": 16,
    },
  ]
  assert {c["name"] for c in mk._checks(stack_clash, 16) if not c["ok"]} == {"coexist_A_B"}
  ok = [
    {
      "name": "A",
      "text_base": 0,
      "text_bytes": 160,
      "data_base": 0x10000,
      "data_bytes": 0,
      "stack_bytes": 128,
    },
    {
      "name": "B",
      "text_base": 4096,
      "text_bytes": 160,
      "data_base": 0x11000,
      "data_bytes": 0,
      "stack_bytes": 128,
    },
  ]
  assert all(c["ok"] for c in mk._checks(ok, 16))


def test_recursion_dynamic_stack_and_open_indirects_are_refused() -> None:
  su = "x.c:1:a\t8\tstatic\nx.c:2:b\t8\tstatic\n"
  calls = {"_start": ["a"], "a": ["b"], "b": ["a"]}
  _expect("RECURSION", mk._stack_bound, su, calls, {"_start": 0, "a": 8, "b": 8})
  _expect("UNBOUNDED_FRAME", mk._stack_bound, "x.c:1:f\t16\tdynamic\n", {}, {"f": 16})
  _expect("UNRESOLVED_INDIRECT", mk._parse_disassembly, "   4: 000000e7 \tjr\tt0\n\n00000000 <_start>:\n")
  _, frames, _ = mk._parse_disassembly(
    "00000000 <f>:\n   0: ff010113 \taddi\tsp, sp, -0x10\n   4: 01010113 \taddi\tsp, sp, 0x10\n"
  )
  assert frames["f"] == 0x10
  _expect("UNMODELED_STACK", mk._parse_disassembly, "00000000 <f>:\n   0: 01010113 \taddi\tsp, sp, 0x20\n")


def test_the_compilers_frame_is_the_authority_and_the_parse_must_agree() -> None:
  su = "x.c:1:f\t128\tstatic\n"
  graph = {"_start": ["f"]}
  # the compiler says 128, the parse saw 0: the bound is 128, never the under-parsed 0
  assert mk._stack_bound(su, graph, {"_start": 0, "f": 0}) == 128
  # the parse saw MORE than the compiler reported: the .su is wrong, refuse
  _expect("FRAME_MISMATCH", mk._stack_bound, "x.c:1:f\t16\tstatic\n", graph, {"_start": 0, "f": 32})
  # a frame the .su never recorded cannot be counted as zero
  _expect("FRAME_MISMATCH", mk._stack_bound, "", graph, {"_start": 0, "f": 8})
  # a call target the image never defined is unknown, not a free leaf
  _expect("UNKNOWN_FUNCTION", mk._stack_bound, "", {"_start": ["ghost"]}, {"_start": 0})
  # a malformed or duplicated .su record is refused rather than skipped
  _expect("BAD_STACK_USAGE", mk._stack_bound, "x.c:1:f\tmany\tstatic\n", graph, {"_start": 0, "f": 0})
  _expect("BAD_STACK_USAGE", mk._stack_bound, su + su, graph, {"_start": 0, "f": 0})
  # the deepest chain wins: _start -> a (8) -> b (16) beats _start -> c (4)
  deep = "x.c:1:a\t8\tstatic\nx.c:2:b\t16\tstatic\nx.c:3:c\t4\tstatic\n"
  frames = {"_start": 0, "a": 8, "b": 16, "c": 4}
  assert mk._stack_bound(deep, {"_start": ["a", "c"], "a": ["b"]}, frames) == 24


def test_every_sp_write_the_frame_model_does_not_cover_is_refused() -> None:
  for insn in (
    "sub\tsp, sp, a0",
    "mv\tsp, a0",
    "li\tsp, 0",
    "lw\tsp, 0(a0)",
    "addi\tsp, a0, 4",
    "auipc\tsp, 0x10",
  ):
    _expect("UNMODELED_STACK", mk._parse_disassembly, f"00000000 <f>:\n   0: 00000000 \t{insn}\n")
  # stores THROUGH sp and loads INTO other registers are not sp writes
  _, frames, _ = mk._parse_disassembly(
    "00000000 <f>:\n   0: 00000000 \tsw\ts0, 8(sp)\n   4: 00000000 \tlw\tra, 0xc(sp)\n"
    "   8: 00000000 \tsw\tsp, 0(a0)\n   c: 00000000 \tc.swsp\tsp, 4(a0)\n"
  )
  assert frames == {"f": 0}
  # _start alone may initialise sp (auipc + addi), nothing else may
  start = "00000000 <_start>:\n   8: 00010117 \tauipc\tsp, 0x10\n   c: 07810113 \taddi\tsp, sp, 0x78\n"
  assert mk._parse_disassembly(start)[1] == {"_start": 0}
  for insn in ("mv\tsp, a0", "li\tsp, 0", "sub\tsp, sp, a0", "lw\tsp, 0(a0)"):
    _expect("UNMODELED_STACK", mk._parse_disassembly, f"00000000 <_start>:\n   0: 00000000 \t{insn}\n")


def test_a_size_report_without_text_is_refused() -> None:
  _expect("NO_TEXT", mk._section_sizes, "section size addr\n.data 8 0\n")
  assert mk._section_sizes("section size addr\n.text 160 0\n.data 8 0\n.bss 4 0\n") == {
    "text": 160,
    "data": 12,
  }


def test_the_package_consumer_accepts_the_measured_manifest_and_rejects_a_peak(
  record: dict[str, Any],
) -> None:
  contract = pc.load_contract()
  package = {
    "image": {"text_bytes": record["resource_manifest"]["text_bytes"]},
    "target_fingerprint": {
      "tensor_input_bytes": 2,
      "accumulator_bytes": 4,
      "tensor_output_bytes": 2,
      "rvv_vlen": 128,
      "context_count": 1,
    },
    "resource_manifest": dict(record["resource_manifest"]),
  }
  pc._check_resources(package, contract)
  import copy

  peak = copy.deepcopy(package)
  peak["resource_manifest"]["bound_source"] = "measured_peak"
  with pytest.raises(pc.PackageError) as exc:
    pc._check_resources(peak, contract)
  assert exc.value.code == "BOUND_NOT_STATIC"
  drifting = copy.deepcopy(package)
  drifting["resource_manifest"]["text_bytes"] += 16
  with pytest.raises(pc.PackageError) as exc:
    pc._check_resources(drifting, contract)
  assert exc.value.code == "TEXT_BYTES_DISAGREE"
  overflowing = copy.deepcopy(package)
  overflowing["resource_manifest"]["stack_bytes"] = (
    contract["image"]["capacities"]["dtcm_bytes"]["value"] + 1
  )
  with pytest.raises(pc.PackageError) as exc:
    pc._check_resources(overflowing, contract)
  assert exc.value.code == "STACK_EXCEEDS_CAPACITY"


# ---------------------------------------------------------------------------------------------------
# entry / no-toolchain behaviour
# ---------------------------------------------------------------------------------------------------
def test_helper_csrs_that_do_not_match_the_c017_allocation_are_refused() -> None:
  import json

  layout = json.loads((H / "protocol/generated/cn_layout.json").read_text())
  want = sorted(r["address"] for r in layout["csr"] if r["name"].startswith("nexus.mregion."))
  mk._check_helper_csrs(want, layout)
  _expect("HELPER_CSR_MISMATCH", mk._check_helper_csrs, [*want[:-1], want[-1] + 1], layout)
  _expect("HELPER_CSR_MISMATCH", mk._check_helper_csrs, [*want, 0x7FF], layout)
  _expect("HELPER_CSR_MISMATCH", mk._check_helper_csrs, want[:4], layout)


def test_no_toolchain_is_a_refusal_not_a_fake_measurement(tmp_path: Path, monkeypatch: Any) -> None:
  monkeypatch.setattr(mk, "TOOLCHAIN", tmp_path / "nowhere")
  _expect("NO_TOOLCHAIN", mk.measure, tmp_path / "out")
  assert shutil.which("true") is not None  # the suite ran to here; skips above only hide real builds
