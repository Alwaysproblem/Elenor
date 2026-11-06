# Coral Nexus hardware workspace maintenance

Operational handbook for the `hardware/coral_nexus` workspace (CN-G0 onward).
Verification evidence lives under `out/coral_nexus/<CARD>/attempt-NNN/`;
`out/` is gitignored and never constitutes an archive by itself.

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
