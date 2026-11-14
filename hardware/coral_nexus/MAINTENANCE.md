# Coral Nexus hardware workspace maintenance

Operational handbook for the `hardware/coral_nexus` workspace (CN-G0 onward).
Verification evidence lives under `out/coral_nexus/<CARD>/attempt-NNN/`;
`out/` is gitignored and never constitutes an archive by itself.

## Fresh-machine bootstrap (full sequence)

Every step below is a verified entry on a new clone. The G0 gate only turns
green after the evidence producers have actually run on THIS machine (or
their archived outputs were restored and re-pinned — the pin verifies bytes,
it does not create them).

```bash
git clone --recurse-submodules <repo> && cd coral-npu-nexus
alias cn="$PWD/hardware/coral_nexus/scripts/cn.sh"

cn env build        # or: docker load < archived image, then cn env verify
cn sources verify   # 19 checks; the bare form defaults --lock/--out

# Evidence producers (fresh attempts under out/coral_nexus/...):
cn baseline --config hardware/coral_nexus/configs/p0_coral.yaml \
  --out out/coral_nexus/C005/baseline
cn bf16-baseline --out out/coral_nexus/C006/attempt-001  # ~10 min, two-seed KAT

# Offline archive: needs ONE online build first to produce the vendor source
# tree (see Offline seeds), then export and the two proofs:
cn env export --out out/coral_nexus/C004/export-001
python3 hardware/coral_nexus/scripts/lib/sources_tool.py materialize \
  --lock hardware/coral_nexus/env/sources.lock \
  --out out/coral_nexus/C004/offline/work
cn env offline-rebuild --export out/coral_nexus/C004/export-001 \
  --work out/coral_nexus/C004/offline/work \
  --out out/coral_nexus/C004/offline/rebuild
cn env devcontainer-probe --out out/coral_nexus/C004/devcontainer

cn gate --stage G0 --out out/coral_nexus/C007/gate  # 31 checks, exit 0
```

Known gap on a new machine: the ~7 GB vendor archive bytes are NOT in git
(only the pinned digest is). Rebuilding them requires one online C005-style
build followed by `bazel vendor`, or fetching the archive from an artifact
store that does not exist yet.

## Two dispatchers, one registry

| Where     | Entry point                                  | Behaviour                                                                                    |
| --------- | -------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Host      | `hardware/coral_nexus/scripts/cn.sh`         | runs host commands (`env`, `sources`, `gate`, `unit`, `baseline`, `bf16-baseline`, `verify`) |
| Container | `/workspace/hardware/coral_nexus/scripts/cn` | refuses any command whose `executor` is not `container` (exit 3)                             |

Both read the same registry (`scripts/commands.yaml`); there is no shadow
list. For convenience on the host:

```bash
alias cn="$PWD/hardware/coral_nexus/scripts/cn.sh"   # or: export PATH
```

Exit codes are part of the contract: `0` success, `1` checker/assertion
failure, `2` usage error, `3` unknown command or suite, `4` registered but
not implemented.

## Environment

- One locked build image: `coral-nexus-builder:c002` built from
  `env/Dockerfile` (base digest, tool versions and deviations recorded in
  `env/image.lock`). Build and verify:

  ```bash
  cn env build
  cn env verify --out out/coral_nexus/C002/verify
  ```

- Jobs always run as the caller UID/GID; no docker socket, no
  `--privileged`, repository mounted read-only.
- Python split: validator in conda env `elenor-validator`
  (`pipeline_validator/environment.yml`, hash locked in
  `env/requirements.lock`); container python is distro-managed only.

## Sources

- Upstreams are pristine submodules pinned in `env/sources.lock`
  (gitlink + git-archive sha256 per source; compiler-embedded pins recorded,
  never aligned). Verify: `cn sources verify --out <dir>`.
- Derived builds materialize into fresh work roots under
  `out/coral_nexus/work/` via `scripts/lib/materialize.sh`; overlay entries
  (`env/overlay.yaml`) and the patch series (`patches/series`,
  single-owner enforced by `patches/file_owners.yaml`) apply strictly there.

## Focused C027 acceptance

The `w2` suite contains both C026 and C027. To verify C027 without rerunning
an already-accepted C026 run, use its independent pytest entry:

```bash
EVIDENCE=out/coral_nexus/C027/attempt-NNN
mkdir -p "$EVIDENCE"
conda run --no-capture-output -n elenor-validator python -m pytest \
  hardware/coral_nexus/verification/unit/w2/test_w2_c027.py \
  -n0 -v --basetemp="$EVIDENCE/pytest"
```

Requires the locked Docker builder, the populated `cache/w2_bazel` offline
caches, and the C021 pinned Spike/toolchain cache. The runner materializes
strict patches, builds the hermetic Bazel Verilator, executes native SV
cases and Scala RTL scenarios, then compares cancel-time retirement records
with independent Spike runs. It preserves logs, JUnit, native results and
oracle comparison output under the attempt. A mutation is accepted only
for a real behavioral assertion failure; build errors and skipped cases fail.
`PROTOCOL_FREEZE.json` must remain byte-identical.

Patch files are excluded from whitespace/EOF fixers: removing a trailing
blank context line can corrupt a unified diff. Regenerate patch artifacts
from the derived tree rather than formatting them.

ChiselSim test titles become build-directory names: use shell-safe words,
not semicolons, pipes or ampersands. The simulation launcher adds ChiselSim's
`support-artifacts` include directory and parses macro headers before SV
consumers; FPnew definitions must not depend on resource enumeration order.
Native-resource simulations retain warnings under the upstream
`fpga/coralnpu_soc.core` `-Wno-fatal` policy; this is functional acceptance,
not a warning-free lint claim. Chisel-only cases keep fatal warnings.

## Offline seeds (C004)

- Inventory: `env/seed_targets.json` — every seed has URL/path + sha256 +
  license row in `env/notices.md`. `bazel_vendor_cache` is a `local-dir`
  seed whose sha256 is the tree digest of the **archived** copy.
- Directory seeds are archived by copying the tree and re-hashing it. In-tree
  relative symlinks are preserved (bazel uses them for interpreters and tool
  aliases); absolute or escaping links are dropped and listed under
  `dropped_links`, because they point at a producer-local path and would make
  the archive non-self-contained. A missing source or a digest mismatch is a
  hard failure that names the seed — the exporter never refills silently.
- Export into a fresh versioned directory (refuses a non-empty one):

  ```bash
  cn env export --out out/coral_nexus/C004/export-<stamp>
  ```

  Export reuses `out/coral_nexus/cache/seeds/`, verifies every sha256, and
  fails if any seed is missing or mismatched. Report: `export_manifest.json`.

### Offline rebuild proof (C004)

Rebuild the P0 targets with no network and empty build caches, using only the
exported vendor archive:

```bash
python3 hardware/coral_nexus/scripts/lib/sources_tool.py materialize \
  --lock hardware/coral_nexus/env/sources.lock --out out/coral_nexus/C004/offline/work
cn env offline-rebuild --export out/coral_nexus/C004/export-<stamp> \
  --work out/coral_nexus/C004/offline/work --out out/coral_nexus/C004/offline/rebuild
```

The probe runs `docker --network none`, asserts the bazel output root, disk
cache and repository cache are empty, re-verifies the archive digest, then
builds `-c opt`. It copies the two ELFs into `artifacts/` and records their
hashes; the G0 gate refuses an offline rebuild whose ELF bytes differ from the
recorded baseline (`offline_rebuild.matches_baseline`).

### Devcontainer proof (C004)

```bash
cn env devcontainer-probe --out out/coral_nexus/C004/devcontainer
```

Launches the container through `@devcontainers/cli` (fresh, from the current
JSON), then probes the non-root user, tool versions, the `/workspace` mount
and a real `bazel build` inside it. A static JSON check is not acceptance:
when the CLI is missing the command records a blocker and exits 3.

## Gates (C007 / C008)

```bash
cn gate --stage G0 --out out/coral_nexus/C007/gate
```

G0 verifies five evidence sets (`sources_verify`, `environment_verify`,
`seed_export`, `baseline_run`, `bf16_baseline`) against
**`verification/expected_evidence.json`** — a committed pin holding the
sha256 of every required report _and_ artifact plus the expected verdict.
The pin is the root of trust: re-hashing against a manifest rebuilt from
the same directory proves nothing, and a directory containing only
`{"pass": true}` fails. A missing or incomplete pin fails closed
(`pin.present` / `pin.coverage`).

Exit codes: `0` pass, `1` evidence mismatch, `2` usage, `4` stage not
implemented. `--evidence-index DIR` and `--pin FILE` override the
defaults (used by the negative tests in
`verification/contracts/test_g0_gate.py`).

### Refreshing the pin

After a legitimate re-run of any evidence producer, the recorded hashes
change and the gate fails until the pin is updated **in a reviewed
commit**:

```bash
cn gate --stage G0 --out out/coral_nexus/C007/gate            # see which hashes moved
python3 hardware/coral_nexus/verification/gates.py \
  --evidence-index out/coral_nexus/C007/evidence_index \
  --pin hardware/coral_nexus/verification/expected_evidence.json --write-pin
```

Never refresh the pin merely to make a red gate green: an unchanged
verdict with changed bytes means the producer changed, and that is a
review event.

## CN-G1 gate and the capacity decisions (C020)

`cn gate --stage G1 --out <dir>` re-runs every producer validator (C009-C016), re-renders every generated
artifact, checks the measured resource record and verifies the joint seal. Exit codes: 11 contracts,
12 codec artifacts, 13 CSR bindings, 14 open capacity budget, 15 measured resources, 16 targets, 17 seal,
18 gate spec drift.

Two capacities have no derivable bound and are therefore **decisions**, recorded in
`configs/w4_capacity_decisions.yaml` (read by `freeze_check.open_budgets`, sealed by hash):

- `per_ctx_outstanding_credit_cap`: per-Context CreditManager ceiling, in entries.
- `response_record_budget`: waiter response records per MSHR refill entry, in entries.

A budget is closed only when `value` is a positive integer in `entries` and `decided_by`, `rationale` and
`evidence` are non-empty (for the response budget the rationale must say the hardware stalls when full).
`null`, 0, a bool, a float or a blank field keeps it open. Recording a decision does not edit
`w4_memory_constraints.yaml`: both stay `derivable: false` there. After filling the file: run
`python hardware/coral_nexus/protocol/freeze_check.py seal`, then `cn gate --stage G1`. Changing a recorded
value later changes the file hash, so the seal stops verifying until `seal --bump`.

## Devcontainer

`.devcontainer/coral_nexus/devcontainer.json` builds the same locked image
(same Dockerfile, same digest) and runs as the non-root `builder` user with
the repository mounted; it mounts no docker socket and adds no privileges.
Static policy violations (privileged, socket mount, `build` section) are
rejected by `test_environment.py`.

## Upgrade path

1. Change the pin/hash in the relevant lock (`sources.lock`,
   `tools.lock`, `image.lock`, `seed_targets.json`); add/adjust the
   notices.md row in the same commit.
2. Rebuild the image and rerun `cn env verify`; re-materialize work roots
   and rerun affected suites — old PASS never carries over.
3. Bazel vendor seeds: after a green in-container build, refresh
   `bazel_vendor_cache` in `seed_targets.json` and re-export; offline
   rebuild verification is C073's exec-repro gate.
