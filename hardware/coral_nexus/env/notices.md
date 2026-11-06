# Third-party notices (CN-G0, C004)

License index for everything the locked environment downloads or embeds.
License texts are fetched from the pinned sources listed in
`seed_targets.json` / `sources.lock`; nothing here is vendored wholesale.

| Component                                         | Version / pin                            | License                                             | Source of truth                                                       |
| ------------------------------------------------- | ---------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------- |
| coralnpu (RTL, sim, toolchain defs)               | f79522157fc0f1d7311f44e209fc36c9c8b7a039 | Apache-2.0                                          | `upstream/coralnpu/LICENSE`                                           |
| coralnpu-compiler                                 | d715556c3853719e414182c0bf5f4a10f248f1d9 | Apache-2.0                                          | `upstream/coralnpu-compiler/LICENSE`                                  |
| rv32_toolchain — toolchain_coralnpu_v2 (RV32 GCC) | 2026-06-29 archive                       | GPL-3.0-or-later with GCC runtime library exception | `rules/repo_defs.bzl` @ f795221; full notices ship inside the archive |
| verilator                                         | v5.052                                   | LGPL-3.0-only OR Artistic-2.0                       | archive `COPYING`                                                     |
| svdpi_header — svdpi.h                            | v5.028 snapshot                          | LGPL-3.0-only OR Artistic-2.0                       | verilator source tree                                                 |
| buildifier                                        | 7.3.1                                    | Apache-2.0                                          | buildtools release                                                    |
| verible                                           | v0.0-4051-g9fdb4057                      | Apache-2.0                                          | verible release                                                       |
| scalafmt                                          | 3.10.7                                   | Apache-2.0                                          | scalameta release                                                     |
| riscv_isa_sim — riscv-isa-sim (Spike)             | 93a10ae6                                 | BSD-3-Clause                                        | archive `LICENSE`                                                     |
| Perfetto trace_processor_shell                    | repo artifact (v58.x)                    | Apache-2.0                                          | `examples/artifacts/tools/`                                           |
| bazel_vendor_cache (vendored externals)           | produced by bazel vendor @ C005          | aggregated from vendored externals (see above)      | `out/coral_nexus/C005/work-coralnpu/vendor-cache/`                    |
| Debian trixie base + apt packages                 | digest sha256:bfc36ae...                 | various (apt `/usr/share/doc/*/copyright` in-image) | debian.org                                                            |
| Bazel                                             | 9.1.0 (bazel-apt)                        | Apache-2.0                                          | bazel-apt repo                                                        |

Maintenance rules:

- Every new download added to `tools.lock` or `seed_targets.json` must add a
  row here in the same commit.
- `env.sh verify` checks that each `seed_targets.json` entry with a license
  row is present; a seed without a license row is a build failure, not a
  warning.
