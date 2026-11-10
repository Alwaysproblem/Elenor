"""C011 numeric probe capture driver. Runs INSIDE the locked container on a materialized work root.

Packs known_vectors.json into tile runs (numeric_probe.build_groups), runs the overlay bench
//tests/cocotb/vme_test/p0_numeric:p0_numeric_probe_verilator twice (two cocotb seeds) on the
UNMODIFIED upstream VmeCoreMiniAxi model, checks the two runs are identical, and writes the record.
It computes no expected value; the host compares the record to the reference model.

Usage: p0_numeric_capture.py <work_root> <out_dir>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/hardware/coral_nexus/scripts/lib")
import numeric_probe as probe

TARGET = "//tests/cocotb/vme_test/p0_numeric:p0_numeric_probe_verilator"
BAZEL_ROOT = "/w/ob"
SEEDS = (("a", "42"), ("b", "4243"))
VECTORS = "/workspace/hardware/coral_nexus/verification/numerics/known_vectors.json"


def resolve_elf() -> str:
  script = (
    "cd /work/coralnpu && "
    f"bazel --output_user_root={BAZEL_ROOT} build -c opt //tests/cocotb/vme_test:vme_matmul_test "
    ">/dev/null 2>&1; "
    f"ELF=$(bazel --output_user_root={BAZEL_ROOT} cquery -c opt --output=files "
    "//tests/cocotb/vme_test:vme_matmul_test 2>/dev/null | grep 'elf$' | head -1); "
    f"ROOT=$(bazel --output_user_root={BAZEL_ROOT} info output_base 2>/dev/null)/execroot/_main; "
    'case "$ELF" in /*) echo "$ELF";; *) echo "$ROOT/$ELF";; esac'
  )
  proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=900)
  path = proc.stdout.strip()
  if proc.returncode != 0 or not path:
    print(f"error: ELF resolve failed: {proc.stderr[-800:]}", file=sys.stderr)
    sys.exit(1)
  return path


def main() -> None:
  out = Path(sys.argv[2]).resolve()
  out.mkdir(parents=True, exist_ok=True)
  vectors = json.loads(Path(VECTORS).read_text())
  groups = probe.build_groups(vectors)
  Path("/w/numeric_request.json").write_text(json.dumps({"groups": groups}))
  elf = resolve_elf()

  runs = {}
  for tag, seed in SEEDS:
    target_out = f"/w/numeric_{tag}"
    Path(target_out).unlink(missing_ok=True)
    proc = subprocess.run(
      [
        "bash",
        "-c",
        f"cd /work/coralnpu && bazel --output_user_root={BAZEL_ROOT} test -c opt --test_output=streamed "
        "--test_env=P0_NUMERIC_VECTORS=/w/numeric_request.json "
        f"--test_env=P0_NUMERIC_OUT={target_out} --test_env=CORALFLOW_SEED={seed} "
        f"--test_env=P0_NUMERIC_ELF={elf} {TARGET}",
      ],
      capture_output=True,
      text=True,
      timeout=2400,
    )
    (out / f"numeric_run_{tag}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
      print(
        f"error: run {tag} failed rc={proc.returncode}: {proc.stdout[-1500:]}{proc.stderr[-1500:]}",
        file=sys.stderr,
      )
      sys.exit(1)
    runs[tag] = json.loads(Path(target_out).read_text())
    (out / f"numeric_probe_run_{tag}.json").write_text(
      json.dumps(runs[tag], indent=2, sort_keys=True) + "\n"
    )

  if runs["a"] != runs["b"]:
    print("error: seeded replay differs", file=sys.stderr)
    sys.exit(1)
  record = {
    "schema": "cn_p0_numeric_probe/v1",
    "elf_sha256": runs["a"]["elf_sha256"],
    "groups": runs["a"]["groups"],
    "replay_identical": True,
    "seeds": [seed for _, seed in SEEDS],
    "request_sha256": hashlib.sha256(Path("/w/numeric_request.json").read_bytes()).hexdigest(),
    "evidence_tier": "p0_upstream_rtl",
    "notes": (
      "directed BF16 vectors on the unmodified upstream VmeCoreMiniAxi; "
      "DUT contract check, NOT an independent oracle"
    ),
  }
  errors = probe.compare(record, groups)
  record["compared_to_reference_in_container"] = {"errors": errors}
  (out / "numeric_probe_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: v for k, v in record.items() if k != "groups"}, indent=2))
  if errors:
    sys.exit(1)


if __name__ == "__main__":
  main()
