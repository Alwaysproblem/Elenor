# syntax=docker/dockerfile:1
# Coral Nexus CS1 locked build container (C002).
#
# Derived from the upstream coralnpu.dockerfile at pin f795221 with three
# deliberate deviations, all recorded in image.lock:
#   1. Base image pinned by amd64 digest (image.lock: base.digest), not :trixie.
#   2. No sudo/sudoers: jobs run as the caller's UID/GID and never need root.
#   3. Only bazel-9.1.0 from the bazel-apt repo (no 8.6.0, no meta package).
# Lint-only tools from the upstream list (markdownlint, shfmt, shellcheck,
# yapf3) are dropped; every build-relevant package is kept.
#
# Build (host side, via scripts/handlers/env.sh):
#   docker build -f hardware/coral_nexus/env/Dockerfile \
#     --build-arg _UID=$(id -u) --build-arg _GID=$(id -g) ...

FROM debian@sha256:bfc36aecdece0387418759dba474aa856c026ca8087da3606b41b90ab0509c99

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ARG _UID=1000
ARG _GID=1000
ARG _USERNAME=builder
ENV HOME=/home/${_USERNAME}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux -o pipefail; \
    ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime; \
    echo "${TZ}" > /etc/timezone; \
    echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/90assumeyes; \
    apt-get update; \
    apt-get install -y -qq \
      apt-transport-https autoconf build-essential ca-certificates ccache \
      cmake curl device-tree-compiler gawk git gnupg libelf-dev liblz4-dev \
      libmpfr-dev ninja-build openjdk-21-jdk-headless openjdk-21-jre-headless \
      python-is-python3 python3 python3-numpy python3-pip python3-setuptools \
      python3-venv srecord tzdata unzip xxd zip \
      flex bison gperf perl zlib1g-dev libfl2 libfl-dev file; \
    update-ca-certificates; \
    apt-get install -y -qq clang-19 clang-tools-19 lld-19 \
      clang-format-19 clang-tidy-19; \
    ln -s /usr/bin/clang-19 /usr/bin/clang; \
    ln -s /usr/bin/clang++-19 /usr/bin/clang++; \
    ln -s /usr/bin/clang-cpp-19 /usr/bin/clang-cpp; \
    ln -s /usr/bin/ld.lld-19 /usr/bin/ld.lld; \
    ln -s /usr/bin/llvm-symbolizer-19 /usr/bin/llvm-symbolizer; \
    curl -fsSL https://releases.bazel.build/bazel-release.pub.gpg \
      | gpg --dearmor > /usr/share/keyrings/bazel-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg-architecture -q DEB_HOST_ARCH) signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
      > /etc/apt/sources.list.d/bazel.list; \
    apt-get update; \
    apt-get install -y -qq bazel-9.1.0; \
    ln -s /usr/bin/bazel-9.1.0 /usr/local/bin/bazel; \
    bazel --version; \
    curl -fL -o /tmp/buildifier \
      https://github.com/bazelbuild/buildtools/releases/download/v7.3.1/buildifier-linux-amd64; \
    echo "5474cc5128a74e806783d54081f581662c4be8ae65022f557e9281ed5dc88009  /tmp/buildifier" \
      | sha256sum --check -; \
    install /tmp/buildifier /usr/local/bin; \
    curl -fL -o /tmp/verible.tar.gz \
      https://github.com/chipsalliance/verible/releases/download/v0.0-4051-g9fdb4057/verible-v0.0-4051-g9fdb4057-linux-static-x86_64.tar.gz; \
    echo "f52e5920ef63f70620a6086e09dea8bd778147cd7a9ff827bb7de5d6316b1754  /tmp/verible.tar.gz" \
      | sha256sum --check -; \
    mkdir -p /tmp/verible; \
    tar -xf /tmp/verible.tar.gz --strip-components=1 -C /tmp/verible; \
    install /tmp/verible/bin/verible-verilog-* /usr/local/bin; \
    curl -fL -o /tmp/scalafmt.zip \
      https://github.com/scalameta/scalafmt/releases/download/v3.10.7/scalafmt-x86_64-pc-linux.zip; \
    echo "a238a4e73141538a2a2e4e834f80154906b8a3f654ff14b6752fc3a3feeb5be1  /tmp/scalafmt.zip" \
      | sha256sum --check -; \
    unzip /tmp/scalafmt.zip -d /tmp/scalafmt; \
    install /tmp/scalafmt/scalafmt /usr/local/bin; \
    groupadd --gid "${_GID}" "${_USERNAME}"; \
    useradd --home-dir "${HOME}" --comment "" --uid "${_UID}" \
      --gid "${_GID}" "${_USERNAME}"; \
    mkdir -p "${HOME}"; \
    chown "${_USERNAME}:${_USERNAME}" "${HOME}"; \
    ln -sf /lib/x86_64-linux-gnu/libmpfr.so.6.2.0 /lib/x86_64-linux-gnu/libmpfr.so.4; \
    rm -f /tmp/buildifier /tmp/verible.tar.gz /tmp/scalafmt.zip

COPY verify_inside.sh /opt/coral_nexus/verify_inside.sh
COPY host_probe.c rv32_probe.S /opt/coral_nexus/
RUN chmod 0755 /opt/coral_nexus/verify_inside.sh

USER ${_USERNAME}
WORKDIR /home/${_USERNAME}/
