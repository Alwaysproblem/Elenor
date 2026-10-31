"""P08 soft-conv case adapters for the existing container runner.

Drives the real production soft-conv gate
(``pipeline_validator.tests.test_soft_conv``: parse -> compile ->
serialize -> parse -> load -> Simulator full_memory ByteStore against
committed P03 goldens, plus the production epilogue function on the
committed epilogue.json AM words) inside the locked container.  No
numerical algorithms live here.

The committed golden vectors pin each chain's own ``boa_numeric_tile_k``
(the matmul.json chains carry 2/4/8/16; the conv fixtures assume the
default 8), so this target deliberately consumes no ``validator``
configuration: ``validator.hw_overrides`` must be empty and is rejected
otherwise (fail closed, never silently discarded).  ``--seed`` is recorded
and feeds the simulator; the goldens are fixed and never change with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _module_config(ctx) -> Path:
  """Strict module config: profile only; paths default inside the module."""
  payload = {"schema": 1, "profile": "bf16_fp32_mac_rne_v1"}
  path = ctx.work / "soft-conv-config.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return path


def _run_soft_conv_case(ctx, case: str) -> bool:
  overrides = ctx.config.get("validator", {}).get("hw_overrides", {})
  if not ctx.check(
    "fixed-vector-mode-no-overrides",
    not overrides,
    "golden vectors pin per-case boa_numeric_tile_k; validator.hw_overrides "
    f"must be empty, got {overrides}",
  ):
    return False

  out = ctx.artifacts_dir()
  argv = [
    sys.executable,
    "-m",
    "pipeline_validator.tests.test_soft_conv",
    "--case",
    case,
    "--out",
    str(out),
    "--seed",
    str(ctx.request["seed"]),
    "--config",
    str(_module_config(ctx)),
  ]
  code = ctx.run(
    argv,
    ctx.repo,
    f"soft-conv-{case}",
    env={"PYTHONPATH": str(ctx.repo)},
    timeout=3600,
  )
  ok = ctx.check(f"{case}-exit", code == 0, f"exit={code}; see command log")

  report = None
  report_path = out / f"{case}.report.json"
  if report_path.is_file():
    try:
      report = json.loads(report_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
      ok &= ctx.check(f"{case}-report-parses", False, f"invalid JSON: {exc}")
  else:
    ok &= ctx.check(f"{case}-report-written", False, f"{report_path} missing")

  if report is not None:
    ok &= ctx.check(
      f"{case}-status",
      report.get("status") == "PASS",
      f"status={report.get('status')!r} counts={report.get('counts')}",
    )
    for check in report.get("checks", []):
      ok &= ctx.check(f"{case}:{check['id']}", check["pass"], check.get("detail", ""))
    counts = report.get("counts", {})
    ctx.metrics[f"{case}_vectors"] = counts.get("vectors")
    ctx.metrics[f"{case}_cycles_total"] = counts.get("cycles_total")
    for label, key in (
      ("production_path", "production_path"),
      ("epilogue_function", "epilogue_function_parity"),
      ("blocked", "blocked"),
      ("negative", "negative"),
    ):
      if counts.get(key) is not None:
        ctx.metrics[f"{case}_{label}"] = counts.get(key)

  # Wall-clock timing is deliberately recorded with deterministic=False so
  # offline reproduction compares outputs, never elapsed time.
  for name, deterministic in (
    (f"{case}.report.json", True),
    (f"{case}.vectors.json", True),
    (f"{case}.timing.json", False),
  ):
    candidate = out / name
    if candidate.is_file():
      ctx.record_artifact(candidate, deterministic=deterministic)
  sources = out / "sources"
  if sources.is_dir():
    for source in sorted(sources.glob("*.mlir")):
      ctx.record_artifact(source, deterministic=True)
  return bool(ok)


def case_int8_chain(ctx) -> bool:
  """i8-output AM chains end to end: matmul.json i8 chains, conv-i8-out via
  gather-im2col, and every chain-reachable epilogue.json i8 vector; each
  synthesized chain first proves raw bit-exact AM parity (f32 FINAL, no
  epilogue) and then reproduces the golden quantized output; unproducible
  golden AM words get explicit blocked records with their proof class."""
  return _run_soft_conv_case(ctx, "int8-chain")


def case_param_reject(ctx) -> bool:
  """Every epilogue.json illegal-parameter vector at its real boundary
  (source parse or v6 artifact decode), plus codec tampers."""
  return _run_soft_conv_case(ctx, "param-reject")


def case_stage_matrix(ctx) -> bool:
  """Every bias/scale/relu/clip/inv_scale combination in epilogue.json for
  f32/bf16/i8 output at full bytes via the production epilogue function,
  plus Simulator wiring chains."""
  return _run_soft_conv_case(ctx, "stage-matrix")


def case_conv_pad_stride_dilation(ctx) -> bool:
  """All eight soft-conv fixtures (pad/stride/dilation coverage): C bytes and
  the captured im2col A operand bytes equal the committed goldens."""
  return _run_soft_conv_case(ctx, "conv-pad-stride-dilation")


def case_conv_traffic(ctx) -> bool:
  """Per-stage transfer/gather bytes equal the independent analytic counts
  (the counter<->analytic mapping documented in the module)."""
  return _run_soft_conv_case(ctx, "conv-traffic")
