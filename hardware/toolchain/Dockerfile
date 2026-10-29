FROM ubuntu@sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982 AS dependencies

ADD --checksum=sha256:28bc73d1cc37d6db2c5eb3757bc4c79b31803b7bed505bc67ec8141e7e4e2c4e https://github.com/conda-forge/miniforge/releases/download/25.11.0-0/Miniforge3-25.11.0-0-Linux-x86_64.sh /opt/downloads/miniforge.sh
RUN bash /opt/downloads/miniforge.sh -b -p /opt/bootstrap

COPY pipeline_validator/environment.lock.explicit.txt pipeline_validator/environment.lock.yml pipeline_validator/environment.lock.pip.txt /opt/locks/
COPY hardware/toolchain/native.explicit.txt hardware/toolchain/chipyard-native.lock.yml.txt /opt/locks/
COPY hardware/toolchain/bootstrap_envs.py /opt/acquire/bootstrap_envs.py
COPY hardware/toolchain/package-seed /opt/package-seed
RUN --mount=type=cache,id=nexus-p02-archives,target=/opt/acquisition-cache /opt/bootstrap/bin/python /opt/acquire/bootstrap_envs.py

ENV PATH="/opt/validator/bin:/opt/native/bin:/opt/native/riscv-tools/bin:/opt/circt/bin:/opt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    CC="/opt/native/bin/x86_64-conda-linux-gnu-gcc" \
    CXX="/opt/native/bin/x86_64-conda-linux-gnu-g++" \
    AR="/opt/native/bin/x86_64-conda-linux-gnu-ar" \
    NEXUS_DEPS="/opt/deps/chipyard" \
    FIRTOOL_BIN="/opt/circt/bin/firtool" \
    PYTHONDONTWRITEBYTECODE="1" \
    SOURCE_DATE_EPOCH="0"

COPY hardware/toolchain/sources.json /opt/sources.json
COPY hardware/toolchain/install_sources.py /opt/acquire/install_sources.py
RUN --mount=type=cache,id=nexus-p02-archives,target=/opt/acquisition-cache /opt/validator/bin/python /opt/acquire/install_sources.py

FROM dependencies AS toolchain
COPY hardware/toolchain/sbt /opt/bin/sbt
COPY hardware/toolchain/sbt-repositories /opt/sbt-repositories
COPY hardware/toolchain/overlay /opt/overlay
RUN chmod 755 /opt/bin/sbt
ENV SBT_BOOT_DIRECTORY="/opt/sbt-bundle/boot" \
    SBT_IVY_HOME="/opt/sbt-bundle/ivy" \
    COURSIER_CACHE="/opt/sbt-bundle/coursier"
WORKDIR /opt/overlay
RUN SBT_OFFLINE=false NEXUS_GEMMINI=/opt/gemmini/upstream sbt 'p02/compile'
COPY hardware/toolchain/smoke /opt/smoke
COPY hardware/toolchain/freeze_image.py /opt/acquire/freeze_image.py
RUN mkdir -p /opt/softfloat/bin && "$CC" -std=c11 -I/opt/softfloat/include -O2 -o /opt/softfloat/bin/softfloat_abi_probe /opt/smoke/softfloat_abi_probe.c
RUN /opt/validator/bin/python /opt/acquire/freeze_image.py && rm -rf /tmp/.sbt /tmp/nexus-sbt-global /root/.cache/nexus-sbt-tmp /root/.cache/nexus-sbt-global
WORKDIR /workspace
CMD ["/bin/bash"]
