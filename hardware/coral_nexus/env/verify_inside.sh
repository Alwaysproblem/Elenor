#!/usr/bin/env bash
# Runs INSIDE the locked container as the caller's UID/GID (env.sh verify).
# Checks every image.lock/tools.lock invariant that is observable from a
# container shell and writes a JSON report into $REPORT_DIR.
#
# Environment provided by env.sh:
#   REPORT_DIR      writable host directory for the JSON report
#   TOOLCHAIN_DIR   writable host cache directory for the RV32 toolchain
#   H_ROOT          read-only mount of hardware/coral_nexus
set -uo pipefail

REPORT_DIR="${REPORT_DIR:?REPORT_DIR must be set}"
TOOLCHAIN_DIR="${TOOLCHAIN_DIR:?TOOLCHAIN_DIR must be set}"
H_ROOT="${H_ROOT:?H_ROOT must be set}"

CHECKS_JSON="${REPORT_DIR}/environment_checks.json"
TOOLCHAIN_URL="$(python3 -c "import json;print(json.load(open('${H_ROOT}/env/tools.lock'))['rv32_toolchain']['url'])")"
TOOLCHAIN_SHA="$(python3 -c "import json;print(json.load(open('${H_ROOT}/env/tools.lock'))['rv32_toolchain']['sha256'])")"
RV32_MARCH="rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma"
RV32_MABI="ilp32"

pass=0
fail=0
declare -a failed_names

note() { printf '%s\n' "$*" >> "${REPORT_DIR}/verify_inside.log"; }

check() {  # check <name> <ok:0|1> <detail>
  local name="$1" ok="$2" detail="$3"
  printf '  %-34s %s | %s\n' "$name" "$([ "$ok" = 0 ] && printf PASS || printf FAIL)" "$detail" \
    >> "${REPORT_DIR}/verify_inside.log"
  if [ "$ok" = 0 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); failed_names+=("$name"); fi
}

expect_eq() {  # expect_eq <name> <actual> <expected>
  if [ "$2" = "$3" ]; then check "$1" 0 "$2"; else check "$1" 1 "actual '$2' != expected '$3'"; fi
}
expect_match() {  # expect_match <name> <actual> <regex>
  if printf '%s' "$2" | grep -Eq "$3"; then check "$1" 0 "matched $3"; else check "$1" 1 "'$2' !~ $3"; fi
}

: > "${REPORT_DIR}/verify_inside.log"

# --- tool versions ---------------------------------------------------------
bazel_version="$(bazel --version 2>/dev/null | awk '{print $2}')"
expect_eq bazel_version "$bazel_version" "9.1.0"

clang_version="$(clang --version 2>/dev/null | head -1)"
expect_match clang_19 "$clang_version" '^Debian clang version 19\.'

java_version="$(java -version 2>&1 | head -1)"
expect_match openjdk_21 "$java_version" 'version "21'

expect_eq buildifier "$(buildifier --version 2>/dev/null | head -1 | awk '{print $NF}')" "7.3.1"
expect_match verible_lint "$(verible-verilog-lint --version 2>/dev/null | head -1)" 'v0.0-4051'
expect_match scalafmt "$(scalafmt --version 2>/dev/null | head -1)" '3\.10\.7'

# --- host program: compile + run + exact output ----------------------------
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
clang "${H_ROOT}/env/host_probe.c" -o "$work/host_probe"
host_out="$("$work/host_probe")"
expect_eq host_probe_output "$host_out" "host-probe-ok 44"

# --- RV32 toolchain: sha256-locked download + real cross compile ------------
cross="${TOOLCHAIN_DIR}/extracted/toolchain_coralnpu_v2/bin/riscv64-unknown-elf-gcc"
toolchain_tar="${TOOLCHAIN_DIR}/$(basename "${TOOLCHAIN_URL}")"
mkdir -p "${TOOLCHAIN_DIR}"
if [ ! -f "${toolchain_tar}" ]; then
  if curl -fsSL -o "${toolchain_tar}.part" "${TOOLCHAIN_URL}"; then
    mv "${toolchain_tar}.part" "${toolchain_tar}"
  else
    rm -f "${toolchain_tar}.part"
    check toolchain_download 1 "curl failed for ${TOOLCHAIN_URL}"
  fi
fi
if [ -f "${toolchain_tar}" ]; then
  actual_sha="$(sha256sum "${toolchain_tar}" | awk '{print $1}')"
  if [ "$actual_sha" = "$TOOLCHAIN_SHA" ]; then
    check toolchain_sha256 0 "$actual_sha"
  else
    check toolchain_sha256 1 "actual ${actual_sha} != locked ${TOOLCHAIN_SHA}"
  fi
  if [ ! -x "$cross" ]; then
    rm -rf "${TOOLCHAIN_DIR}/extracted"
    mkdir -p "${TOOLCHAIN_DIR}/extracted"
    tar -xJf "${toolchain_tar}" -C "${TOOLCHAIN_DIR}/extracted"
  fi
  if [ -x "$cross" ]; then
    "${cross}" -march="${RV32_MARCH}" -mabi="${RV32_MABI}" -mcmodel=medany \
      -nostdlib "${H_ROOT}/env/rv32_probe.S" -o "$work/rv32_probe.elf"
    elf_header="$(readelf -h "$work/rv32_probe.elf")"
    expect_match rv32_elf_class "$elf_header" 'Class:.*ELF32'
    expect_match rv32_elf_machine "$elf_header" 'Machine:.*RISC-V'
  else
    check rv32_cross_compile 1 "cross gcc missing after extraction"
  fi
else
  check rv32_cross_compile 1 "toolchain archive unavailable"
fi

# --- identity: caller must not be root, no docker socket -------------------
if [ "$(id -un)" = "root" ]; then
  check non_root_user 1 "container must run as the caller, not root"
else
  check non_root_user 0 "$(id -un)"
fi
if [ -S /var/run/docker.sock ]; then
  check no_docker_socket 1 "docker socket visible inside container"
else
  check no_docker_socket 0 "not mounted"
fi

python3 - "$pass" "$fail" ${failed_names[@]+"${failed_names[@]}"} > "${CHECKS_JSON}.tmp" <<'PY'
import json
import sys

pass_count, fail_count = int(sys.argv[1]), int(sys.argv[2])
failed = sys.argv[3:] if fail_count else []
report = {
  "schema": "cn_environment_report/v1",
  "pass": fail_count == 0,
  "passed": pass_count,
  "failed": fail_count,
  "failed_checks": failed,
}
print(json.dumps(report, indent=2, sort_keys=True))
PY
mv "${CHECKS_JSON}.tmp" "${CHECKS_JSON}"
cat "${CHECKS_JSON}"
[ "$fail" = 0 ]
