"""C039 end-to-end PROGRAM-channel installation (Group software ↔ Verilated C030 program SRAM)."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from pipeline_validator.tests import w5_residency_cosim as cosim

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "hardware/coral_nexus"
W4 = H / "verification/unit/w4"
sys.path.insert(0, str(W4))
from w4_rtl_harness import Bench, Slice, check_verdicts, run_slice, selected_target  # noqa: E402


def elf_text(path: Path) -> bytes:
  # Independent fetch oracle: decode the linked executable segment, never the staging buffer or ACK.
  elf = path.read_bytes()
  start = struct.unpack_from("<I", elf, 28)[0]
  stride, count = struct.unpack_from("<HH", elf, 42)
  for index in range(count):
    kind, offset, _, _, length, _, flags, _ = struct.unpack_from("<8I", elf, start + index * stride)
    if kind == 1 and flags & 1:
      return elf[offset : offset + length]
  raise AssertionError("no executable PT_LOAD")


def test_install_pull_residency_and_program_fetch(tmp_path_factory: pytest.TempPathFactory) -> None:
  scenario_path = H / "verification/scenarios/t58_cold_warm.yaml"
  scenario = yaml.safe_load(scenario_path.read_text())
  cosim.validate_scenario(scenario, str(scenario_path))
  config = H / "configs/cs1_c2.yaml"  # T58 pins the released C2 packages.
  root = tmp_path_factory.mktemp("c038-release")
  subprocess.run(
    [
      "bash",
      str(H / "scripts/cn.sh"),
      "package",
      "--binding",
      str(H / "kernels/bindings/scalar_ctl_v1.yaml"),
      "--config",
      str(config),
      "--out",
      str(root),
    ],
    check=True,
    timeout=300,
  )
  texts = {slot: elf_text(root / slot / "image.elf") for slot in ("A", "B")}
  contexts = selected_target()["constraints"]["context_count"]
  assert contexts == 2, "T58 consumes C038 cs1_c2 packages; use the matching configuration"
  header = H / "protocol/generated/cn_offsets.svh"
  bench = Bench(
    "tb_c039_cosim",
    (
      "w6/c033_event_pkg.sv",
      "w6/c034_reset_pkg.sv",
      "memory/c029_params_pkg.sv",
      "memory/l1_program_port.sv",
      "w5/install_service.sv",
      "w5/tb/tb_c039_cosim.sv",
    ),
    contexts=(2,),
    args_by_context={2: "+incdir+/out"},
    run_args=("+PEER=/out/{label}",),
  )

  def serve(evidence, labels):
    shutil.copyfile(header, evidence / header.name)
    return cosim.drive(
      evidence,
      labels,
      scenario=scenario,
      scenario_path=scenario_path,
      package_root=root,
      config_path=config,
      texts=texts,
      default_contexts=contexts,
    )

  evidence = run_slice(
    Slice(
      "c039live",
      [bench],
      ("cosim-*.patch",),
      bench,
      rtl_dir=H / "rtl",
      mutants_dir=H / "verification/unit/w5/mutants",
      service=serve,
      assert_markers=("C039_ASSERT",),
      host_sources=(Path(__file__), Path(cosim.__file__), header, scenario_path),
    ),
    tmp_path_factory,
  )
  reports = json.loads((evidence["path"] / "residency-cosim.json").read_bytes())
  for case in evidence["positive"]:
    run = reports[case["label"]]
    assert run["error"] is None and run["verdict"] is None, run
    assert set(run["sequences"]) == set(scenario["steps"]) and all(
      seq["ok"] for seq in run["sequences"].values()
    )
    assert run["group"] == {"credits_hold": True, "l2_objects": []}
  for case in evidence["negative"]:
    assert reports[case["label"]]["verdict"]["key"] == "probe A+0x0 bytes", reports[case["label"]]
  check_verdicts(evidence, pass_marker="C039_PASS")
