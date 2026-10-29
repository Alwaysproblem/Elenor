"""Probe a user-provided read-only Vivado installation; never install or produce a bitstream."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

OUT = Path("/out")
VIVADO = "/opt/vivado/bin/vivado"
PART = "xcvu9p-flga2104-2L-e"


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  version = subprocess.run([VIVADO, "-version"], capture_output=True, text=True)
  (OUT / "version.log").write_text(version.stdout + version.stderr)
  match = re.search(r"Vivado\s+v([0-9.]+)", version.stdout + version.stderr)
  result: dict = {
    "vivado": {"status": "present", "version": match.group(1) if match else None, "license": "unverified"},
    "fpga_capability": "blocked",
  }
  if version.returncode == 0 and match:
    (OUT / "capability.v").write_text(
      "module capability_probe(input clk, input [15:0] a, b, output reg [15:0] q);\n"
      "always @(posedge clk) q <= a + b;\nendmodule\n"
    )
    (OUT / "capability.tcl").write_text(
      f"create_project -in_memory -part {PART}\n"
      "read_verilog /out/capability.v\n"
      f"synth_design -top capability_probe -part {PART}\n"
      f'if {{[get_property PART [current_design]] ne "{PART}"}} {{error "wrong device"}}\n'
      'puts "P02_VIVADO_LICENSE_SYNTHESIS_OK"\nexit\n'
    )
    probe = subprocess.run(
      [
        VIVADO,
        "-mode",
        "batch",
        "-notrace",
        "-source",
        "/out/capability.tcl",
        "-log",
        "/out/synthesis.log",
        "-journal",
        "/out/synthesis.jou",
      ],
      cwd=OUT,
      capture_output=True,
      text=True,
    )
    (OUT / "synthesis.stdout.log").write_text(probe.stdout + probe.stderr)
    if probe.returncode == 0 and "P02_VIVADO_LICENSE_SYNTHESIS_OK" in probe.stdout:
      result["vivado"]["license"] = "verified"
      result["fpga_capability"] = "ok"
    else:
      result["vivado"]["reason"] = f"device/license synthesis probe failed, exit {probe.returncode}"
  else:
    result["vivado"]["reason"] = f"Vivado version probe failed, exit {version.returncode}"
  (OUT / "capability.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result))


if __name__ == "__main__":
  main()
