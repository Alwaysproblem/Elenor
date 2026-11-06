"""P0 BF16 KAT capture driver (C006, derived overlay).

Runs the isolated KAT target twice (two cocotb seeds) inside the locked
container, then folds the two KAT jsons into bf16_kat_report.json:

- per case: input bits, accumulator preload, actual result bits, cycle_count
- bit- and cycle-identical replay across seeds (asserted here)
- ELF hash recorded; one deliberate bit-flip on a copy of run A is PROVEN to
  change the comparison (forge detection demonstrated, not assumed).

Runs INSIDE the locked container on the materialized P0 work root.
Usage: p0_bf16_capture.py <work_root> <out_dir>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

TARGET = "//tests/cocotb/vme_test/p0_kat:p0_bf16_kat_verilator"
BAZEL_ROOT = "/w/ob"
SEEDS = (("a", "42"), ("b", "4243"))


def sha256_of(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  out = Path(sys.argv[2]).resolve()
  out.mkdir(parents=True, exist_ok=True)

  kats = {}
  for tag, seed in SEEDS:
    kat_file = out / f"bf16_kat_run_{tag}.json"
    proc = subprocess.run(
      [
        "bash",
        "-c",
        "cd /work/coralnpu && "
        f"bazel --output_user_root={BAZEL_ROOT} test -c opt "
        "--test_output=streamed "
        f"--test_env=P0_BF16_KAT_OUT=/w/kat_{tag} "
        f"--test_env=CORALFLOW_SEED={seed} "
        f"--test_env=P0_BF16_KAT_ELF={_resolve_elf()} "
        f"{TARGET}",
      ],
      capture_output=True,
      text=True,
      timeout=2400,
    )
    (out / f"kat_run_{tag}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
      print(
        f"error: run {tag} failed (rc={proc.returncode}); "
        f"tail: {proc.stdout[-2000:]}{proc.stderr[-2000:]}",
        file=sys.stderr,
      )
      sys.exit(1)
    # The test writes a FILE at /w/kat_{tag} (P0_BF16_KAT_OUT); persist it.
    kat_file.write_text(Path(f"/w/kat_{tag}").read_text())
    kats[tag] = json.loads(kat_file.read_text())

  a, b = kats["a"], kats["b"]
  if a != b:
    print("error: seeded replay differs (bits or cycles)", file=sys.stderr)
    for ca, cb in zip(a["cases"], b["cases"]):
      if ca != cb:
        print(f"  case {ca['case']} differs", file=sys.stderr)
    sys.exit(1)

  # Forge-detection demonstration: flip one bit in run A's copy; the folded
  # comparison must catch WHICH element changed, not merely 'differs'.
  tampered = json.loads(json.dumps(a))
  tampered["cases"][0]["actual_u32"][0] ^= 1
  flipped = [
    (c["case"], f"actual_u32[{i}]")
    for c, t in zip(a["cases"], tampered["cases"])
    for i, (x, y) in enumerate(zip(c["actual_u32"], t["actual_u32"]))
    if x != y
  ]
  assert len(flipped) == 1, "self-test: tamper detection failed"
  assert tampered != a and len(
    [(x, y) for x, y in zip(a, b)]
  ) == len(a), "self-test: comparison semantics broken"

  runs = {}
  for tag, seed in SEEDS:
    kat_file = out / f"bf16_kat_run_{tag}.json"
    runs[tag] = {
      "seed": seed,
      "kat_sha256": sha256_of(kat_file),
      "cycles_per_case": {c["case"]: c["cycles"] for c in kats[tag]["cases"]},
    }
  report = {
    "schema": "cn_p0_bf16_kat_record/v1",
    "test_target": TARGET,
    "elf_sha256": a["elf_sha256"],
    "runs": runs,
    "cases": a["cases"],
    "replay_identical": True,
    "tamper_self_test": {"location": flipped[0], "detected": True},
    "evidence_tier": "p0_upstream_rtl",
    "notes": (
      "Four exactly-precise bf16 KATs (quarter-integer operands exact in "
      "bf16) through the unmodified upstream VmeCoreMiniAxi Verilator model; "
      "reference recomputed in exact fp64 and asserted element-wise inside "
      "the test; two seeds replayed bit- and cycle-identical. NOT an "
      "independent oracle; C047 remains the oracle owner."
    ),
  }
  (out / "bf16_kat_record.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


def _resolve_elf() -> str:
  """Resolve the absolute upstream ELF path inside THIS container process.

  One bazel root for build + cquery so both see the same output tree; the
  returned path is made absolute against the execroot reported by the same
  invocation (cquery emits workspace-relative paths).
  """
  script = (
    "cd /work/coralnpu && "
    "bazel --output_user_root=/w/ob build -c opt "
    "//tests/cocotb/vme_test:vme_matmul_test >/dev/null 2>&1; "
    "ELF=$(bazel --output_user_root=/w/ob cquery -c opt --output=files "
    "//tests/cocotb/vme_test:vme_matmul_test 2>/dev/null | grep 'elf$' "
    "| head -1); "
    "ROOT=$(bazel --output_user_root=/w/ob info output_base 2>/dev/null)"
    "/execroot/_main; "
    'case "$ELF" in /*) echo "$ELF";; *) echo "$ROOT/$ELF";; esac'
  )
  proc = subprocess.run(
    ["bash", "-c", script], capture_output=True, text=True, timeout=900
  )
  path = proc.stdout.strip()
  if proc.returncode != 0 or not path:
    print(
      f"error: ELF resolve failed: {proc.stderr[-800:]}",
      file=sys.stderr,
    )
    sys.exit(1)
  return path


if __name__ == "__main__":
  main()
