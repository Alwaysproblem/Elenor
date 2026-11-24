"""C039: Tile install service against the real sealed wire and the C030 port.

Unit scope: `tb_install_service` instantiates the real
`rtl/w5/install_service` together with the real `rtl/memory/l1_program_port`
(C030) it writes through. The bench drives the sealed v5 messages as raw
64-bit words (PROGRAM_INSTALL / PROGRAM_READ_REQ / PROGRAM_READ_RESP /
PROGRAM_INSTALL_ACK / PROGRAM_READY / PROGRAM_INVALIDATE / INVALIDATE_ACK)
and probes residency by fetching the installed text back through the C030
fetch channel - there is no shadow copy of the text anywhere.

Trusted setup and measured digest are REAL and reproducible: the runner
rebuilds the C038 release with `cn package --binding
kernels/bindings/scalar_ctl_v1.yaml --config configs/cs1_c2.yaml` into a
private tmpdir, then calls `program_prepare.prepare_programs` on the sealed
t58 scenario with the C2 target config. The Loader verifies every package
(image bytes, linked_slot, target fingerprint); the bench's staged inputs
are `ProgramImage.text` / `package_hash` / `program_epoch` and the freeze's
`linked_slot` text_base. The third identity ("C") is the t58 `fixtures.C`
rule: B's verified text under a distinct fixture program_id/package token -
the measured digest stays honest because it is computed over the same
streamed bytes.

The wire constants come from the GENERATED C017 header only:
`install_service.sv` includes `cn_offsets.svh`, staged into /out like every
other W5 slice; the run fails closed at build time if the layout is not
regenerated.

Each green run must emit exactly one `C039_PASS <top> contexts=N checks=M
installs=K warm=W inv=V` record. Mutants die by an executed assertion after
a clean build, per the shared harness rule.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

W5 = Path(__file__).resolve().parent
W4 = W5.parent / "w4"
H = W5.parents[2]
REPO = W5.parents[4]
sys.path.insert(0, str(W4))
sys.path.insert(0, str(H / "scripts/lib"))
sys.path.insert(0, str(REPO))

import gen_memory_params as memory_params  # noqa: E402
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice, selected_target  # noqa: E402

from pipeline_validator.cosim.program_prepare import prepare_programs  # noqa: E402

SVH = H / "protocol/generated/cn_offsets.svh"
CONTRACT = H / "protocol/w5_package.contract.yaml"
SCENARIO = H / "verification/scenarios/t58_cold_warm.yaml"
BINDING = H / "kernels/bindings/scalar_ctl_v1.yaml"
CONFIG = H / "configs/cs1_c2.yaml"
CN = H / "scripts/cn.sh"

PASS_RE = re.compile(
  r"^C039_PASS (\S+) contexts=(\d+) checks=(\d+) installs=(\d+) warm=(\d+) inv=(\d+)$",
  re.M,
)

BENCH = Bench(
  top="tb_install_service",
  sources=(
    "memory/c029_params_pkg.sv",
    "memory/l1_program_port.sv",
    "w6/c033_event_pkg.sv",
    "w6/c034_reset_pkg.sv",
    "w5/install_service.sv",
    "w5/tb/tb_install_service.sv",
  ),
  contexts=(1, 2),
  # cn_offsets.svh is staged into /out by the service (the sealed copy).
  args_by_context={1: "+incdir+/out", 2: "+incdir+/out", 4: "+incdir+/out"},
  timeout_s=300,
)

HOST_SOURCES = (
  Path(__file__),
  W4 / "w4_rtl_harness.py",
  SVH,
  CONTRACT,
  SCENARIO,
  BINDING,
  CONFIG,
  Path(prepare_programs.__code__.co_filename),
)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
  @contextmanager
  def serve(out: Path, _labels: Sequence[str]) -> Iterator[None]:
    (out / "cn_offsets.svh").write_bytes(SVH.read_bytes())

    # Rebuild the C038 release byte-for-byte; the Loader inside
    # prepare_programs verifies the sealed artifacts again at read time.
    root = out / "c038pkg"
    root.mkdir()
    build = subprocess.run(
      [
        "bash",
        str(CN),
        "package",
        "--binding",
        str(BINDING),
        "--config",
        str(CONFIG),
        "--out",
        str(root),
      ],
      capture_output=True,
      text=True,
      check=False,
    )
    (out / "c039-package-build.log").write_text(build.stdout + build.stderr)
    assert build.returncode == 0, f"cn package failed:\n{build.stdout}\n{build.stderr}"

    run = prepare_programs(SCENARIO, root, config_path=CONFIG)

    # The bench's third identity is the t58 fixtures.C rule: B's verified
    # text under a distinct program_id/package token.
    images: list[dict[str, Any]] = []
    for slot in ("A", "B"):
      im = run.images[slot]
      linked = run.packages[slot].view["binding_manifest"]["kernels"][0]["linked_slot"]
      text_len = len(im.text)
      images.append(
        {
          "program_id": im.program_id,
          "version": im.version,
          "program_epoch": im.program_epoch,
          "slot": int(linked["text_base"]),
          "text_base": int(linked["text_base"]),
          "text_len": text_len,
          "package_hash": im.package_hash.hex(),
          "text_sha256": hashlib.sha256(im.text).hexdigest(),
          "staged": im.text + bytes(4096 - text_len),  # zero tail pads
        }
      )
    images.append(
      {
        "program_id": run.scenario["fixtures"]["C"]["program_id"],
        "version": 1,
        "program_epoch": 1,
        "slot": images[1]["slot"],
        "text_base": images[1]["text_base"],
        "text_len": images[1]["text_len"],
        "package_hash": hashlib.sha256(b"cn-c039-imageC-fixture").hexdigest(),
        "text_sha256": images[1]["text_sha256"],
        "staged": images[1]["staged"],
        "fixture": "t58 fixtures.C (text_from: B)",
      }
    )

    receipt: dict[str, Any] = {
      "generated/cn_offsets.svh": hashlib.sha256(SVH.read_bytes()).hexdigest(),
      "protocol/w5_package.contract.yaml": hashlib.sha256(
        CONTRACT.read_bytes()
      ).hexdigest(),
      "scenario": str(SCENARIO.relative_to(REPO)),
      "binding": str(BINDING.relative_to(REPO)),
      "config": str(CONFIG.relative_to(REPO)),
      "slot_freeze_sha256": hashlib.sha256((root / "slot_freeze.json").read_bytes()).hexdigest(),
      "images": {},
    }
    names = ("a", "b", "c")
    meta = [0] * 42
    for i, img in enumerate(images):
      (out / f"c039_img_{names[i]}.hex").write_text(
        "".join(f"{b:02x}\n" for b in img["staged"])
      )
      receipt["images"][names[i].upper()] = {
        k: img[k]
        for k in (
          "program_id",
          "version",
          "program_epoch",
          "slot",
          "text_base",
          "text_len",
          "package_hash",
          "text_sha256",
        )
      }
      pkg = bytes.fromhex(img["package_hash"])
      dig = bytes.fromhex(img["text_sha256"])
      base = i * 14
      meta[base + 0] = img["program_id"]
      meta[base + 1] = img["version"]
      for w in range(4):
        meta[base + 2 + w] = int.from_bytes(pkg[w * 8 : w * 8 + 8], "little")
      meta[base + 6] = img["program_epoch"]
      meta[base + 7] = img["slot"]
      meta[base + 8] = img["text_base"]
      meta[base + 9] = img["text_len"]
      for w in range(4):
        meta[base + 10 + w] = int.from_bytes(dig[w * 8 : w * 8 + 8], "little")
    (out / "c039_meta.hex").write_text("".join(f"{w:016x}\n" for w in meta))
    (out / "c039-sealed-inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
    yield

  return run_slice(
    Slice(
      name="c039install",
      benches=(BENCH,),
      mutant_globs=("install-*.patch",),
      mutant_bench=BENCH,
      min_mutants=7,
      assert_markers=("C039_ASSERT", "C030_ASSERT"),
      service=serve,
      host_sources=HOST_SOURCES,
      rtl_dir=H / "rtl",
      mutants_dir=W5 / "mutants",
    ),
    tmp_path_factory,
  )


def test_install_service(evidence: dict[str, Any]) -> None:
  check_verdicts(evidence, pass_marker="C039_PASS")
  path = evidence["path"]
  params = memory_params.parameters()
  requested = int(selected_target()["constraints"]["context_count"])
  counts = sorted({int(params["C1_CONTEXTS"]), requested})
  pairs = {(case["top"], case["context_count"]) for case in evidence["positive"]}
  assert pairs == {(BENCH.top, count) for count in counts}
  for case in evidence["positive"]:
    log = (path / f"{case['label']}.log").read_text()
    passes = PASS_RE.findall(log)
    assert len(passes) == 1, f"{case['label']}: expected one C039_PASS record, got {passes}"
    top, contexts, checks, installs, warm, inv = passes[0]
    assert (top, int(contexts)) == (case["top"], case["context_count"]), (
      case["label"],
      passes,
    )
    assert int(checks) >= 120, f"{case['label']}: PASS record ran too few checks"
    # The observable ledger: six INSTALL frames (A cold, A warm hit, C cold,
    # C re-install after invalidate, B cold, deferred A re-install), one warm
    # hit, and real + noop invalidations.
    assert int(installs) >= 4 and int(warm) == 1 and int(inv) >= 1, (
      case["label"],
      passes,
    )
