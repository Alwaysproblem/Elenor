"""Report text detail gating (--detailed): six gated headings, JSON/trace invariance."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from pipeline_validator.compiler import compile_program
from pipeline_validator.config import HardwareConfig, SimConfig
from pipeline_validator.execution_ir import GlobalBinding
from pipeline_validator.loader import load_program
from pipeline_validator.report import build_report, report_to_json, report_to_text
from pipeline_validator.simulator import Simulator
from pipeline_validator.workloads import PowWorkload

_GATED_HEADINGS = (
  "Configured/effective resources",
  "Profile controller",
  "Arena pools",
  "Group scheduler",
  "CPU device controller",
  "CPU request timing",
)

_HW = HardwareConfig().with_overrides(hbm_fixed_latency_cycles=10)
_POW_BINDINGS = {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")}


def _pow_report():
  wl = PowWorkload(hw=_HW)
  sim = Simulator(_HW, SimConfig(max_cycles=200_000))
  artifact = compile_program(wl.module, sim.hw, sim.sim, workload_info=wl.info)
  loaded = load_program(artifact, sim.hw, sim.sim, actual_bindings=_POW_BINDINGS)
  result = sim.run(loaded)
  assert result.completed, result.reason
  return build_report(wl.info, result)


class TestReportDetailedText:
  """report_to_text gates exactly the six detail headings; everything else stays."""

  def test_default_text_omits_detail_sections(self):
    rep = _pow_report()
    text = report_to_text(rep)
    for heading in _GATED_HEADINGS:
      assert heading not in text
    # Ungated content must survive: task leases, checks, engine/stall/events.
    assert "Task leases" in text
    assert "Checks:" in text
    assert "Engine active cycles:" in text
    assert "Stall breakdown" in text
    assert "Events:" in text

  def test_memory_peaks_section_is_data_conditional_not_gated(self):
    # pow runs without memory tracing (rep.memory empty), so populate it
    # deterministically: the section must render in BOTH texts when present.
    rep = dataclasses.replace(_pow_report(), memory={"l2_used_bytes_peak": 123})
    assert "Memory peaks:" in report_to_text(rep)
    assert "Memory peaks:" in report_to_text(rep, detailed=True)
    assert "l2_used_bytes_peak" in report_to_text(rep)

  def test_detailed_text_includes_populated_detail_sections(self):
    rep = _pow_report()
    text = report_to_text(rep, detailed=True)
    populated = {
      "Configured/effective resources": True,  # always rendered
      "Profile controller": bool(rep.profile),
      "Arena pools": bool(rep.arenas),
      "Group scheduler": bool(rep.scheduler),
      "CPU device controller": bool(rep.device),
      "CPU request timing": bool(rep.request_timing),
    }
    for heading, expected in populated.items():
      assert (heading in text) == expected, heading
    assert "Task leases" in text
    assert len(text) > len(report_to_text(rep))

  def test_json_serialization_ignores_detailed(self):
    rep = _pow_report()
    # report_to_json has no detailed parameter: the JSON payload is the same object.
    assert json.loads(report_to_json(rep))["scheduler"] == rep.scheduler
    assert report_to_text(rep) != report_to_text(rep, detailed=True)


class TestCliDetailedFlag:
  """CLI --detailed changes only the text report; traces and JSON are byte-identical."""

  def test_trace_and_json_report_identical_with_and_without_detailed(self, tmp_path):
    repo = Path(__file__).resolve().parents[2]
    runs = {}
    for label in ("default", "detailed"):
      report_path = tmp_path / f"report-{label}.json"
      trace_path = tmp_path / f"trace-{label}.json"
      extra = ["--detailed"] if label == "detailed" else []
      proc = subprocess.run(
        [
          "bash",
          str(repo / "examples/run.sh"),
          "gather-matmul",
          "--json",
          "--report",
          str(report_path),
          "--trace-json",
          str(trace_path),
          "--max-cycles",
          "200000",
          *extra,
        ],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=300,
      )
      assert proc.returncode == 0, proc.stderr[-2000:]
      runs[label] = (report_path, trace_path)

    # Traces and JSON reports must not depend on the flag.
    assert runs["default"][1].read_bytes() == runs["detailed"][1].read_bytes()
    assert runs["default"][0].read_bytes() == runs["detailed"][0].read_bytes()

    # The JSON report still carries the detail payloads, and the text report
    # (rendered separately here from the same builder contract) gates headings.
    payload = json.loads(runs["detailed"][0].read_text())
    assert isinstance(payload, list) and payload
    assert payload[0].get("scheduler"), "JSON report must keep scheduler data"

    text_default = subprocess.run(
      ["bash", str(repo / "examples/run.sh"), "gather-matmul", "--max-cycles", "200000"],
      capture_output=True,
      text=True,
      cwd=repo,
      timeout=300,
    )
    text_detailed = subprocess.run(
      ["bash", str(repo / "examples/run.sh"), "gather-matmul", "--detailed", "--max-cycles", "200000"],
      capture_output=True,
      text=True,
      cwd=repo,
      timeout=300,
    )
    assert text_default.returncode == 0 and text_detailed.returncode == 0
    for heading in _GATED_HEADINGS:
      assert heading not in text_default.stdout
      assert heading in text_detailed.stdout


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(pytest.main([__file__]))
