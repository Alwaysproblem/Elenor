# Coral Nexus hardware workspace maintenance

Operational handbook for the `hardware/coral_nexus` workspace (CN-G0 onward).
Verification evidence lives under `out/coral_nexus/<CARD>/attempt-NNN/`;
`out/` is gitignored and never constitutes an archive by itself.

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
  license row in `env/notices.md`. `bazel_vendor_cache` is explicitly
  `pending-on-C005` until the first in-container build exists.
- Export into a fresh versioned directory:

  ```bash
  cn env export --out out/coral_nexus/C004/export-<stamp>
  ```

  Export reuses the shared download cache
  (`out/coral_nexus/cache/seeds/`), verifies every sha256, and fails if any
  seed is missing or mismatched. The manifest is `export_manifest.json`.

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
