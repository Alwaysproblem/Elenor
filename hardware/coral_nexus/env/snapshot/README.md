# snapshot/

Offline seed inventory root for CN-G0 (filled by C004). `bazel vendor`
caches, python wheels, the RV32 toolchain archive and other pinned externals
are exported here by `env.sh export` using `env/tools.lock` as the single
list of URLs and sha256s. This directory contains no committed binaries; it
only holds the manifests describing what a later export must produce.
