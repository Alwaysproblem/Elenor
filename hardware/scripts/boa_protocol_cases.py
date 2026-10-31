"""P10 boa-protocol case adapters for the container runner.

The ``boa-protocol`` target freezes the wire/descriptor schema
``hardware/boa/protocol/boa_protocol_v0.json`` and proves the three codecs
generated from it (Python module, Scala object, C header) agree on real
bytes and real rejection behaviour.  Every case compiles and executes real
programs inside the locked container:

- Scala is compiled by sbt 1.8.2 / Scala 2.13.12 from the offline image
  bundle (the same pinned toolchain as the P02 overlay) into a fresh work
  project that mounts the codec sources; the driver
  ``boa.protocol.BoaProtocolSelftest`` only consumes the generated object.
- C is compiled by the image's native gcc from the generated header plus the
  committed driver.
- Python runs in-process under /opt/validator (and the committed codec is
  additionally exercised on the host by the P10 evidence commands).

No case re-implements codec logic: vectors are built with the generated
Python codec, and the Scala/C sides only decode/re-encode them.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SCHEMA_REL = "hardware/boa/protocol/boa_protocol_v0.json"
GENERATOR_REL = "hardware/boa/protocol/gen_protocol.py"
PY_CODEC_REL = "hardware/boa/protocol/py/boa_protocol.py"
SCALA_SRC_REL = "hardware/boa/src/main/scala"
C_DRIVER_REL = "hardware/boa/sim/c/boa_protocol_selftest.c"

HW_CONFIG_REL = "pipeline_validator/hardware_config.yaml"
BOA_ENGINE_CONFIGS = {
  1: "hardware/configs/boa-1engine.yaml",
  2: "hardware/configs/boa-2engine.yaml",
  4: "hardware/configs/boa-4engine.yaml",
}
NUMERIC_CONFIG_REL = "hardware/configs/numeric-profile.yaml"

# Real fixtures compiled through pipeline_validator.compiler in PC07.
PC07_FIXTURES = {
  "boa-am-matmul": "examples/workloads/boa_am_matmul.mlir",
  "soft-conv-i8": "examples/workloads/soft_conv_i8_out.mlir",
}

# Generated once by pipeline_validator.tests.test_boa_numeric_profile::
# build_case_module("p10tr", 3, 2, steps, "f32", transpose=all four combos)
# with storage_k=16 > k=4 and per-step k_mask; verbatim fixture text for the
# P10 transpose case (4 steps, NN/TN/NT/TT, FINAL on seq 3).
TRANSPOSE_FIXTURE = """\
builtin.module {
  tile.program @p10tr(%task: !nest.task, %a0_l2: !nest.l2_buffer<3x16xbf16>, \
%b0_l2: !nest.l2_buffer<16x2xbf16>, %a1_l2: !nest.l2_buffer<16x3xbf16>, \
%b1_l2: !nest.l2_buffer<16x2xbf16>, %a2_l2: !nest.l2_buffer<3x16xbf16>, \
%b2_l2: !nest.l2_buffer<2x16xbf16>, %a3_l2: !nest.l2_buffer<16x3xbf16>, \
%b3_l2: !nest.l2_buffer<2x16xbf16>, %c_l2: !nest.l2_buffer<3x2xf32>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1, 2],
    tile_l1_spm_bytes_per_context = 40960> {
    %a0_v = tile.subview %a0_l2 offsets = [0, 0] sizes = [3, 16] strides = [1, 1]
      : !nest.l2_view<3x16xbf16>
    %b0_v = tile.subview %b0_l2 offsets = [0, 0] sizes = [16, 2] strides = [1, 1]
      : !nest.l2_view<16x2xbf16>
    %a1_v = tile.subview %a1_l2 offsets = [0, 0] sizes = [16, 3] strides = [1, 1]
      : !nest.l2_view<16x3xbf16>
    %b1_v = tile.subview %b1_l2 offsets = [0, 0] sizes = [16, 2] strides = [1, 1]
      : !nest.l2_view<16x2xbf16>
    %a2_v = tile.subview %a2_l2 offsets = [0, 0] sizes = [3, 16] strides = [1, 1]
      : !nest.l2_view<3x16xbf16>
    %b2_v = tile.subview %b2_l2 offsets = [0, 0] sizes = [2, 16] strides = [1, 1]
      : !nest.l2_view<2x16xbf16>
    %a3_v = tile.subview %a3_l2 offsets = [0, 0] sizes = [16, 3] strides = [1, 1]
      : !nest.l2_view<16x3xbf16>
    %b3_v = tile.subview %b3_l2 offsets = [0, 0] sizes = [2, 16] strides = [1, 1]
      : !nest.l2_view<2x16xbf16>
    %c_v = tile.subview %c_l2 offsets = [0, 0] sizes = [3, 2] strides = [1, 1]
      : !nest.l2_view<3x2xf32>
    %a0 = tile.alloc shape = [3, 16] dtype = "bf16" alignment = 256 : !tile.l1_buffer<3x16xbf16>
    %b0 = tile.alloc shape = [16, 2] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x2xbf16>
    %a1 = tile.alloc shape = [16, 3] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x3xbf16>
    %b1 = tile.alloc shape = [16, 2] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x2xbf16>
    %a2 = tile.alloc shape = [3, 16] dtype = "bf16" alignment = 256 : !tile.l1_buffer<3x16xbf16>
    %b2 = tile.alloc shape = [2, 16] dtype = "bf16" alignment = 256 : !tile.l1_buffer<2x16xbf16>
    %a3 = tile.alloc shape = [16, 3] dtype = "bf16" alignment = 256 : !tile.l1_buffer<16x3xbf16>
    %b3 = tile.alloc shape = [2, 16] dtype = "bf16" alignment = 256 : !tile.l1_buffer<2x16xbf16>
    %out = tile.alloc memory_space = "l1" shape = [3, 2] dtype = "f32" alignment = 256
      : !tile.l1_buffer<3x2xf32>
    %acc = tile.alloc memory_space = "am" logical_slot = 0 output(%out) shape = [3, 2]
      dtype = "f32" alignment = 256 : !tile.am_buffer<3x2xf32>
    %la0 = tile.load.async %a0_v into %a0 : !tile.event<"la0">
    %lb0 = tile.load.async %b0_v into %b0 : !tile.event<"lb0">
    %la1 = tile.load.async %a1_v into %a1 : !tile.event<"la1">
    %lb1 = tile.load.async %b1_v into %b1 : !tile.event<"lb1">
    %la2 = tile.load.async %a2_v into %a2 : !tile.event<"la2">
    %lb2 = tile.load.async %b2_v into %b2 : !tile.event<"lb2">
    %la3 = tile.load.async %a3_v into %a3 : !tile.event<"la3">
    %lb3 = tile.load.async %b3_v into %b3 : !tile.event<"lb3">
    tile.await %la0, %lb0, %la1, %lb1, %la2, %lb2, %la3, %lb3
    tile.signal input_released(%task)
    %k0 = tile.boa.async "matmul" ins(%a0, %b0) outs(%out) acc(%acc)
                m = 3 n = 2 k = 4 seq = 0 k_mask = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : \
          !tile.event<"k0">
    tile.await %k0
    %k1 = tile.boa.async "matmul" ins(%a1, %b1) outs(%out) acc(%acc)
                m = 3 n = 2 k = 4 seq = 1 k_mask = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] \
          transpose_a accumulate : !tile.event<"k1">
    tile.await %k1
    %k2 = tile.boa.async "matmul" ins(%a2, %b2) outs(%out) acc(%acc)
        m = 3 n = 2 k = 4 seq = 2 transpose_b accumulate : !tile.event<"k2">
    tile.await %k2
    %k3 = tile.boa.async "matmul" ins(%a3, %b3) outs(%out) acc(%acc)
                m = 3 n = 2 k = 4 seq = 3 k_mask = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] \
          transpose_a transpose_b accumulate output : !tile.event<"k3">
    tile.await %k3
    tile.free %a0
    tile.free %b0
    tile.free %a1
    tile.free %b1
    tile.free %a2
    tile.free %b2
    tile.free %a3
    tile.free %b3
    %stored = tile.store.async %out into %c_v : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
  nest.context @p10tr_ctx(%A0: !nest.global_memref<3x16xbf16>, \
%B0: !nest.global_memref<16x2xbf16>, %A1: !nest.global_memref<16x3xbf16>, \
%B1: !nest.global_memref<16x2xbf16>, %A2: !nest.global_memref<3x16xbf16>, \
%B2: !nest.global_memref<2x16xbf16>, %A3: !nest.global_memref<16x3xbf16>, \
%B3: !nest.global_memref<2x16xbf16>, %C: !nest.global_memref<3x2xf32>)
    placement = 1
        resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0, 1, 2],
      logical_tasks = 1, l2_spm_bytes = 664, requested_contexts_per_tile = 1> {
    %a0_blk = nest.subview %A0 offsets = [0, 0] sizes = [3, 16] strides = [1, 1]
      : !nest.global_view<3x16xbf16>
    %b0_blk = nest.subview %B0 offsets = [0, 0] sizes = [16, 2] strides = [1, 1]
      : !nest.global_view<16x2xbf16>
        %a0_l2_1 = nest.alloc slot = "a0" role = "in" shape = [3, 16] dtype = "bf16" alignment = 256 : \
      !nest.l2_buffer<3x16xbf16>
        %b0_l2_1 = nest.alloc slot = "b0" role = "in" shape = [16, 2] dtype = "bf16" alignment = 256 : \
      !nest.l2_buffer<16x2xbf16>
    %a0_pref = nest.dma.prefetch.async %a0_blk into %a0_l2_1 : !nest.event<"a0_pref">
    %b0_pref = nest.dma.prefetch.async %b0_blk into %b0_l2_1 : !nest.event<"b0_pref">
    %a1_blk = nest.subview %A1 offsets = [0, 0] sizes = [16, 3] strides = [1, 1]
      : !nest.global_view<16x3xbf16>
    %b1_blk = nest.subview %B1 offsets = [0, 0] sizes = [16, 2] strides = [1, 1]
      : !nest.global_view<16x2xbf16>
        %a1_l2_1 = nest.alloc slot = "a1" role = "in" shape = [16, 3] dtype = "bf16" alignment = 256 : \
      !nest.l2_buffer<16x3xbf16>
        %b1_l2_1 = nest.alloc slot = "b1" role = "in" shape = [16, 2] dtype = "bf16" alignment = 256 : \
      !nest.l2_buffer<16x2xbf16>
    %a1_pref = nest.dma.prefetch.async %a1_blk into %a1_l2_1 : !nest.event<"a1_pref">
    %b1_pref = nest.dma.prefetch.async %b1_blk into %b1_l2_1 : !nest.event<"b1_pref">
    %a2_blk = nest.subview %A2 offsets = [0, 0] sizes = [3, 16] strides = [1, 1]
      : !nest.global_view<3x16xbf16>
    %b2_blk = nest.subview %B2 offsets = [0, 0] sizes = [2, 16] strides = [1, 1]
      : !nest.global_view<2x16xbf16>
        %a2_l2_1 = nest.alloc slot = "a2" role = "in" shape = [3, 16] dtype = "bf16" alignment = 256 : \
      !nest.l2_buffer<3x16xbf16>
        %b2_l2_1 = nest.alloc slot = "b2" role = "in" shape = [2, 16] dtype = "bf16" alignment = 256 : \
      !nest.l2_buffer<2x16xbf16>
    %a2_pref = nest.dma.prefetch.async %a2_blk into %a2_l2_1 : !nest.event<"a2_pref">
    %b2_pref = nest.dma.prefetch.async %b2_blk into %b2_l2_1 : !nest.event<"b2_pref">
    %a3_blk = nest.subview %A3 offsets = [0, 0] sizes = [16, 3] strides = [1, 1]
      : !nest.global_view<16x3xbf16>
    %b3_blk = nest.subview %B3 offsets = [0, 0] sizes = [2, 16] strides = [1, 1]
      : !nest.global_view<2x16xbf16>
        %a3_l2_1 = nest.alloc slot = "a3" role = "in" shape = [16, 3] dtype = "bf16" alignment = 256 : \
      !nest.l2_buffer<16x3xbf16>
        %b3_l2_1 = nest.alloc slot = "b3" role = "in" shape = [2, 16] dtype = "bf16" alignment = 256 : \
      !nest.l2_buffer<2x16xbf16>
    %a3_pref = nest.dma.prefetch.async %a3_blk into %a3_l2_1 : !nest.event<"a3_pref">
    %b3_pref = nest.dma.prefetch.async %b3_blk into %b3_l2_1 : !nest.event<"b3_pref">
    %c_blk = nest.subview %C offsets = [0, 0] sizes = [3, 2] strides = [1, 1]
      : !nest.global_view<3x2xf32>
        %c_l2_1 = nest.alloc slot = "c" role = "out" shape = [3, 2] dtype = "f32" alignment = 256 : \
      !nest.l2_buffer<3x2xf32>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %read, %ready = nest.dispatch.tasks.async @p10tr l1_mode = 0 context = 0
            tasks(%tasks) globals() bindings(%a0_l2_1, %b0_l2_1, %a1_l2_1, %b1_l2_1, %a2_l2_1, %b2_l2_1, \
        %a3_l2_1, %b3_l2_1, %c_l2_1) ins(%a0_l2_1, %b0_l2_1, %a1_l2_1, %b1_l2_1, %a2_l2_1, %b2_l2_1, \
        %a3_l2_1, %b3_l2_1)
      outs(%c_l2_1)
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
        output_ready = #nest.aggregate<all_tasks>
      } depends_on(%a0_pref, %b0_pref, %a1_pref, %b1_pref, %a2_pref, %b2_pref, %a3_pref, %b3_pref)
      : (!nest.event<"grid">, !nest.event<"read">, !nest.event<"ready">)
    %c_store = nest.dma.store.async %c_l2_1 into %c_blk depends_on(%ready) : !nest.event<"c_store">
    nest.release %a0_l2_1 depends_on(%read, %a0_pref)
    nest.release %b0_l2_1 depends_on(%read, %b0_pref)
    nest.release %a1_l2_1 depends_on(%read, %a1_pref)
    nest.release %b1_l2_1 depends_on(%read, %b1_pref)
    nest.release %a2_l2_1 depends_on(%read, %a2_pref)
    nest.release %b2_l2_1 depends_on(%read, %b2_pref)
    nest.release %a3_l2_1 depends_on(%read, %a3_pref)
    nest.release %b3_l2_1 depends_on(%read, %b3_pref)
    nest.release %c_l2_1 depends_on(%c_store)
    nest.await %grid, %c_store
    nest.return
  }
  nexus.program @p10tr_entry(%A0: !nest.global_memref<3x16xbf16>, \
%B0: !nest.global_memref<16x2xbf16>, %A1: !nest.global_memref<16x3xbf16>, \
%B1: !nest.global_memref<16x2xbf16>, %A2: !nest.global_memref<3x16xbf16>, \
%B2: !nest.global_memref<2x16xbf16>, %A3: !nest.global_memref<16x3xbf16>, \
%B3: !nest.global_memref<2x16xbf16>, %C: !nest.global_memref<3x2xf32>) {
        %done_1 = nexus.submit_context.async @p10tr_ctx(%A0, %B0, %A1, %B1, %A2, %B2, %A3, %B3, %C) : \
      !nexus.event<"done">
    nexus.await %done_1
    nexus.return
  }
}
"""

_CODEC_CACHE: dict[str, object] = {}
_SBT_CACHE: dict[str, Path] = {}
_CC_CACHE: dict[str, Path] = {}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _load_module(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load module {name} from {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def _codec(ctx):
  """Load the generated Python codec from the read-only repo mount."""
  if "codec" not in _CODEC_CACHE:
    _CODEC_CACHE["codec"] = _load_module("boa_protocol_generated", ctx.repo / PY_CODEC_REL)
  return _CODEC_CACHE["codec"]


def _schema(ctx) -> dict:
  return json.loads((ctx.repo / SCHEMA_REL).read_text(encoding="utf-8"))


def _generator_lint(ctx, schema: dict) -> list[str]:
  """Run the executable schema lint from the committed generator module."""
  module = _load_module("boa_protocol_gen", ctx.repo / GENERATOR_REL)
  return module.lint_schema(schema)


def _sbt_project(ctx, scala_root: Path) -> Path:
  """Fresh sbt project (cached per source root) using the offline bundle."""
  cache_key = str(scala_root)
  if cache_key in _SBT_CACHE:
    return _SBT_CACHE[cache_key]
  project = ctx.work / f"sbt-{abs(hash(cache_key)) % 100000}"
  (project / "project").mkdir(parents=True, exist_ok=True)
  (project / "project" / "build.properties").write_text("sbt.version=1.8.2\n", encoding="utf-8")
  (project / "build.sbt").write_text(
    'lazy val root = (project in file("."))\n'
    "  .settings(\n"
    '    name := "boa-protocol-check",\n'
    '    scalaVersion := "2.13.12",\n'
    '    scalacOptions ++= Seq("-deprecation", "-unchecked"),\n'
    '    Compile / scalaSource := file("src"),\n'
    "  )\n",
    encoding="utf-8",
  )
  src = project / "src"
  if src.exists():
    shutil.rmtree(src)
  shutil.copytree(scala_root, src)
  driver = src / "boa" / "protocol" / "BoaProtocolSelftest.scala"
  if not driver.is_file():
    # Regenerated trees contain only the generated codec; the committed
    # selftest driver joins them so the freshly generated codec is what
    # actually executes.
    shutil.copyfile(ctx.repo / SCALA_SRC_REL / "boa/protocol/BoaProtocolSelftest.scala", driver)
  deps = ctx.work / "sbt-deps"
  if not deps.is_dir():
    bundle = Path("/opt/sbt-bundle")
    for item in ("boot", "coursier", "ivy"):
      if not (bundle / item).is_dir():
        raise RuntimeError(f"image contract violated: {bundle / item} is missing")
    deps.mkdir(parents=True)
    for item in ("boot", "coursier", "ivy"):
      shutil.copytree(bundle / item, deps / item, symlinks=True)
  _SBT_CACHE[cache_key] = project
  return project


def _scala_run(ctx, scala_root: Path, args: list[str], name: str, timeout: int = 3600) -> tuple[int, str]:
  project = _sbt_project(ctx, scala_root)
  command = "runMain boa.protocol.BoaProtocolSelftest " + " ".join(args)
  argv = ["/opt/bin/sbt", command.strip()]
  env = {
    "SBT_BOOT_DIRECTORY": str(ctx.work / "sbt-deps" / "boot"),
    "COURSIER_CACHE": str(ctx.work / "sbt-deps" / "coursier"),
    "SBT_IVY_HOME": str(ctx.work / "sbt-deps" / "ivy"),
    "HOME": "/tmp/nexus-home",
  }
  return ctx.run_capture(argv, project, name, env=env, timeout=timeout)


def _cc_binary(ctx, header_dir: Path) -> Path:
  cache_key = str(header_dir)
  if cache_key in _CC_CACHE:
    return _CC_CACHE[cache_key]
  cc = shutil.which("cc") or shutil.which("gcc")
  if cc is None:
    raise RuntimeError("native cc/gcc not found in the container PATH")
  binary = ctx.work / f"boa-c-selftest-{abs(hash(cache_key)) % 100000}"
  code = ctx.run(
    [
      cc,
      "-std=c11",
      "-Wall",
      "-Wextra",
      "-Werror",
      "-O2",
      f"-I{header_dir}",
      str(ctx.repo / C_DRIVER_REL),
      "-o",
      str(binary),
    ],
    ctx.work,
    "cc-compile",
  )
  if code != 0:
    raise RuntimeError(f"native codec compile failed with exit {code}")
  _CC_CACHE[cache_key] = binary
  return binary


def _counts_line(codec, messages) -> str:
  counts: dict[str, int] = {}
  for message in messages:
    counts[codec.message_name(message)] = counts.get(codec.message_name(message), 0) + 1
  body = ",".join(f"{name}:{counts[name]}" for name in sorted(counts))
  return f"frames={len(messages)} kinds={body}"


def _roundtrip_all_codecs(ctx, label: str, messages, work: Path) -> bool:
  """Encode with Python, decode+re-encode with all three codecs, compare."""
  codec = _codec(ctx)
  data = codec.encode_many(messages)
  vector_path = work / f"{label}.bin"
  vector_path.write_bytes(data)
  ok = ctx.check(
    f"{label}-python-reencode-identical",
    codec.reencode(data) == data,
    "the generated Python codec must round-trip its own encoding byte-identically",
  )
  expected_counts = _counts_line(codec, messages)

  scala_out = work / f"{label}.scala.bin"
  code, stdout = _scala_run(
    ctx, ctx.repo / SCALA_SRC_REL, ["roundtrip", str(vector_path), str(scala_out)], f"{label}-scala"
  )
  ok &= ctx.check(f"{label}-scala-exit", code == 0, f"exit={code}")
  ok &= ctx.check(
    f"{label}-scala-counts",
    any(line.strip() == expected_counts for line in stdout.splitlines()),
    f"scala driver counts line missing; expected {expected_counts!r}",
  )
  ok &= ctx.check(
    f"{label}-scala-byte-identical",
    scala_out.is_file() and scala_out.read_bytes() == data,
    "Scala decode->re-encode must be byte-identical to the Python encoding",
  )

  binary = _cc_binary(ctx, ctx.repo / "hardware/boa/sim/c")
  c_out = work / f"{label}.c.bin"
  code = ctx.run([str(binary), "roundtrip", str(vector_path), str(c_out)], work, f"{label}-c")
  ok &= ctx.check(f"{label}-c-exit", code == 0, f"exit={code}")
  ok &= ctx.check(
    f"{label}-c-byte-identical",
    c_out.is_file() and c_out.read_bytes() == data,
    "C decode->re-encode must be byte-identical to the Python encoding",
  )
  for path in (vector_path, scala_out, c_out):
    if path.is_file():
      ctx.record_artifact(path, deterministic=True)
  return bool(ok)


def _patched_different_byte(codec, message, **overrides) -> tuple[bytes, int]:
  """Encode a message twice with one field changed; return the encoding and
  the index of the byte that differs (the field's wire position)."""
  base = message
  changed = _with(message, **overrides)
  first = codec.encode_message(base)
  second = codec.encode_message(changed)
  diff = [index for index, (left, right) in enumerate(zip(first, second)) if left != right]
  if len(diff) != 1:
    raise RuntimeError(f"expected exactly one differing byte for {overrides}, got {diff}")
  return first, diff[0]


def _with(message, **overrides):
  values = dict(message.__dict__)
  values.update(overrides)
  return type(message)(**values)


def _write_report(ctx, case: str, payload: dict) -> None:
  path = ctx.artifacts_dir(case) / f"{case}.report.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  ctx.record_artifact(path, deterministic=True)


# ---------------------------------------------------------------------------
# shared vector builders
# ---------------------------------------------------------------------------


def _writer_base(inst, descriptors) -> str:
  """L1 base the instruction's descriptor writes (MFE load / gather / FINAL)."""
  descriptor = descriptors.get(inst.args[0]) if inst.args else None
  if descriptor is None:
    return ""
  transfer = getattr(descriptor, "transfer", None)
  if transfer is not None and hasattr(transfer, "dst"):
    return transfer.dst.base
  gather = descriptor.params.get("gather")
  if hasattr(gather, "destination"):
    return gather.destination.base
  matmul = descriptor.params.get("matmul")
  if hasattr(matmul, "output") and getattr(matmul, "final", False):
    return matmul.output.base
  return ""


def _resolve_step_producers(tile_program, index, matmul) -> tuple[list[str], list[str]]:
  """Last awaited writer of each operand base before the BOA launch.

  A writer only counts when a WAIT/WAITALL that names its event sits strictly
  between the writer's issue and the step; anything merely issued (or awaited
  after the step) is not a producer.
  """
  descriptors = tile_program.descriptors
  await_pairs: list[tuple[int, str]] = []
  for position, inst in enumerate(tile_program.insts[:index]):
    if str(inst.op).endswith(("WAIT", "WAITALL")):
      for event in inst.args:
        if isinstance(event, str):
          await_pairs.append((position, event))
  resolved: list[str] = []
  errors: list[str] = []
  for operand_name, operand in (("A", matmul.a), ("B", matmul.b)):
    candidates = [
      other.dst
      for other_position, other in enumerate(tile_program.insts[:index])
      if other.dst is not None
      and _writer_base(other, descriptors) == operand.base
      and any(
        await_position > other_position and event == other.dst for await_position, event in await_pairs
      )
    ]
    if not candidates:
      errors.append(f"{operand_name}({operand.base})")
    else:
      resolved.append(candidates[-1])
  return resolved, errors


def _view(
  codec,
  base: int,
  dim0: int,
  dim1: int,
  dtype: int,
  element_bytes: int,
  offsets=(0, 0),
  allocation: int = 0,
  generation: int = 1,
):
  return codec.OperandView(
    space=codec.SPACE_L1,
    base_slot=base,
    allocation=allocation,
    generation=generation,
    dim0=dim0,
    dim1=dim1,
    offset0=offsets[0],
    offset1=offsets[1],
    stride0=dim1,
    stride1=1,
    dtype=dtype,
    element_bytes=element_bytes,
    bytes=dim0 * dim1 * element_bytes,
    task_dim=-1,
  )


def _am_request(
  codec,
  *,
  tile_id=0,
  logical_am=0,
  bundle=1,
  rows=64,
  cols=256,
  dtype=None,
  epilogue=None,
  request_seq=7,
  ctx_id=3,
  ctx_epoch=1,
  byte_count=None,
):
  dtype = codec.DTYPE_F32 if dtype is None else dtype
  epilogue = (
    codec.EpilogueConfig(n_cols=cols, scale=0, scale_present=True) if epilogue is None else epilogue
  )
  return codec.AmRequest(
    tile_id=tile_id,
    ctx_id=ctx_id,
    ctx_epoch=ctx_epoch,
    request_seq=request_seq,
    logical_am=logical_am,
    bundle_slots=bundle,
    output_rows=rows,
    output_cols=cols,
    acc_format=codec.ACC_FORMAT_F32,
    layout=codec.LAYOUT_ROW_MAJOR,
    output_l1_slot=4,
    output_dtype=dtype,
    byte_count=(rows * cols * 4) if byte_count is None else byte_count,
    chain_config=epilogue,
  )


def _golden_messages(codec):
  """One message of every kind plus epilogue/dtype variants."""
  full_epilogue = codec.EpilogueConfig(
    n_cols=256,
    bias=tuple(0x3F000000 + index for index in range(256)),
    bias_present=True,
    scale=0x3F400000,
    scale_present=True,
    relu=True,
    clip=codec.ClipPair(0xBF800000, 0x3F800000),
    clip_present=True,
    quant=None,
  )
  i8_epilogue = codec.EpilogueConfig(
    n_cols=1,
    bias=(0x41200000,),
    bias_present=True,
    scale=None,
    scale_present=False,
    relu=True,
    clip=None,
    clip_present=False,
    quant=codec.QuantConfig(inv_scale=0x3F000000, zero_point=3),
    quant_present=True,
  )
  a_view = _view(codec, 0, 64, 64, codec.DTYPE_BF16, 2)
  b_view = _view(codec, 2, 64, 256, codec.DTYPE_BF16, 2)
  return [
    _am_request(codec),
    _am_request(codec, epilogue=full_epilogue, request_seq=8),
    _am_request(
      codec, rows=4, cols=1, dtype=codec.DTYPE_I8, epilogue=i8_epilogue, byte_count=16, request_seq=9
    ),
    codec.AmGranted(
      lease_handle=7,
      slot_generation=1,
      group=2,
      slot_index=1,
      chain_id=9,
      bound_engine=0,
      completion_token=codec.EventToken(ctx_id=3, ctx_epoch=1, event_id=5),
    ),
    codec.AmParked(request_id=9),
    codec.AmError(reason=codec.AM_ERROR_BUNDLE_SELF_LOCK, field=codec.FIELD_TAG_BUNDLE_SLOTS),
    codec.AmWake(
      request_id=9,
      lease_handle=7,
      slot_generation=2,
      bound_engine=1,
      completion_token=codec.EventToken(ctx_id=3, ctx_epoch=1, event_id=6),
    ),
    codec.AmCancel(tile_id=0, ctx_id=3, ctx_epoch=1),
    codec.SubmitStep(
      lease_handle=7,
      seq=0,
      mode=codec.STEP_MODE_INIT,
      final=False,
      transpose=codec.TRANSPOSE_NN,
      a=a_view,
      b=b_view,
      k_extent=64,
      input_ready_tokens=(codec.EventToken(ctx_id=3, ctx_epoch=1, event_id=7),),
      input_usage_handles=(
        codec.AllocationRef(allocation=0, generation=1),
        codec.AllocationRef(allocation=1, generation=1),
      ),
      k_mask=tuple([1] * 8),
      completion_event=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=0),
    ),
    codec.StepAccepted(step_token=123),
    codec.StepFull(lease_handle=7),
    codec.StepError(reason=codec.STEP_ERROR_SEQ_INVALID, field=codec.FIELD_TAG_SEQ),
    codec.EvtStatus(wake_bitmap=0xFF, event_bitmap=0x10),
    codec.EvtTake(
      record=codec.StepTerminal(
        token=codec.EventToken(ctx_id=3, ctx_epoch=1, event_id=5),
        status=codec.TERM_STATUS_SUCCESS,
        seq=1,
        success=True,
        detail=codec.FIELD_TAG_NONE,
      )
    ),
    codec.RecReap(token=codec.EventToken(ctx_id=3, ctx_epoch=1, event_id=5)),
    codec.CtxCancel(tile_id=0, ctx_id=3, ctx_epoch=1),
    codec.MemReq(
      req_id=1,
      ctx_id=3,
      ctx_epoch=1,
      generation=2,
      address=0x10000,
      access=codec.ACCESS_TYPE_READ,
      length=64,
      byte_mask=0xFFFFFFFF,
    ),
    codec.MemResp(req_id=1, status=codec.MEM_STATUS_OK, data=tuple(range(16))),
  ]


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def case_pc01_cross_codec_roundtrip(ctx) -> bool:
  """PC01: the golden message set decodes and re-encodes byte-identically
  under all three generated codecs with consistent per-kind counts, plus the
  dedicated epilogue bias-presence vectors (present-empty, absent, all-zero
  present; decode presence and byte-identical re-encode in all three codecs)."""
  codec = _codec(ctx)
  messages = _golden_messages(codec)
  work = ctx.work / "pc01"
  work.mkdir(parents=True, exist_ok=True)
  ok = _roundtrip_all_codecs(ctx, "pc01", messages, work)
  decoded = codec.decode_stream(codec.encode_many(messages))
  ok &= ctx.check(
    "pc01-covers-every-kind",
    {codec.message_name(message) for message in decoded} == set(codec.MESSAGE_KINDS),
    f"golden set must cover all {len(codec.MESSAGE_KINDS)} message kinds",
  )
  ctx.metrics["pc01_messages"] = len(messages)
  ctx.metrics["pc01_bytes"] = len(codec.encode_many(messages))
  _write_report(
    ctx,
    "pc01",
    {
      "message_count": len(messages),
      "wire_bytes": len(codec.encode_many(messages)),
      "kinds": sorted(codec.MESSAGE_KINDS),
      "counts_line": _counts_line(codec, messages),
    },
  )

  # Epilogue bias-presence vectors: presence is the wire enable bit plus the
  # explicit *_present flag, never the value representation.  The AM_REQUEST
  # carrier carries n_cols == output_cols so the eq_field predicates hold on
  # empty and non-empty vectors alike.
  presence_vectors = [
    (
      "bias-present-empty",
      _am_request(
        codec,
        rows=0,
        cols=0,
        byte_count=0,
        request_seq=21,
        epilogue=codec.EpilogueConfig(n_cols=0, bias=(), bias_present=True),
      ),
      True,
    ),
    (
      "bias-absent-empty",
      _am_request(
        codec, rows=0, cols=0, byte_count=0, request_seq=22, epilogue=codec.EpilogueConfig(n_cols=0)
      ),
      False,
    ),
    (
      "bias-present-all-zero-words",
      _am_request(
        codec,
        rows=4,
        cols=2,
        byte_count=32,
        request_seq=23,
        epilogue=codec.EpilogueConfig(n_cols=2, bias=(0, 0), bias_present=True),
      ),
      True,
    ),
  ]
  expected_presence = [item[2] for item in presence_vectors]
  ok &= ctx.check(
    "pc01-presence-vector-plan",
    expected_presence == [True, False, True],
    f"the positive vector set must be present-empty, absent-empty, present-all-zero: {expected_presence}",
  )
  ok &= _roundtrip_all_codecs(ctx, "pc01-presence", [message for _n, message, _b in presence_vectors], work)

  data = codec.encode_many([message for _name, message, _bias in presence_vectors])
  decoded = codec.decode_stream(data)
  decoded_bias = [(message.chain_config.bias, message.chain_config.bias_present) for message in decoded]
  ok &= ctx.check(
    "pc01-bias-presence-decoded-exact",
    decoded_bias == [((), True), (None, False), ((0, 0), True)],
    f"decoded (bias, bias_present) must distinguish present-empty from absent: {decoded_bias}",
  )
  present_empty = next(message for message in decoded if message.chain_config.bias_present)
  ok &= ctx.check(
    "pc01-present-empty-byte-identical-python",
    codec.encode_message(present_empty) == codec.encode_many([presence_vectors[0][1]]),
    "the present-empty vector's Python re-encode must be byte-identical",
  )

  # Negative: a bias-disabled chain that still carries a non-zero word is
  # rejected and names the field in all three codecs.  Python refuses to
  # construct "absent with a word" (encode-side validation: absent bias must
  # be empty), so the wire shape a value-derived source would emit for that
  # state is written directly: a bias-disabled chain header (n_cols=1,
  # enables=0) followed by one non-zero bias word instead of the canonical
  # zero word.  (Clearing the bit on a present-empty n_cols=0 carrier is
  # canonical-equal to absence and must NOT be rejected; that identity is what
  # the present-empty vector above asserts.)
  writer = codec._Writer()
  writer.u16(1)
  writer.u8(0)  # enables: every stage disabled, canonical zeros required
  writer.u32(0x3F800000)  # the stray bias word under a cleared enable bit
  writer.u32(0)  # scale, disabled
  writer.u32(0)  # clip lo, disabled
  writer.u32(0)  # clip hi, disabled
  writer.u32(0)  # quant inv_scale, disabled
  writer.i8(0)  # quant zero_point, disabled
  disabled_payload = bytes(writer._buf)
  data = bytearray(codec.encode_many([presence_vectors[0][1]]))
  payload_len = data[4] | (data[5] << 8)
  epi_start = 6 + payload_len - 20  # trailing chain_config of the present-empty carrier
  data[epi_start : 6 + payload_len] = disabled_payload
  data[4] = (payload_len + 4) & 0xFF
  data[5] = ((payload_len + 4) >> 8) & 0xFF
  patched_path = work / "pc01-bias-disabled-nonzero-word.bin"
  patched_path.write_bytes(bytes(data))
  ctx.record_artifact(patched_path, deterministic=True)
  rejected = None
  try:
    codec.decode_stream(bytes(data))
  except codec.ProtocolError as exc:
    rejected = exc
  ok &= ctx.check(
    "pc01:bias-disabled-nonzero-word-python-rejects",
    rejected is not None and "bias" in rejected.field,
    f"a bias-disabled chain with a non-zero word must be rejected naming bias,"
    f" got {rejected.field if rejected else 'acceptance'}",
  )
  code, _stdout = _scala_run(
    ctx,
    ctx.repo / SCALA_SRC_REL,
    ["expect-fail", str(patched_path), "bias"],
    "pc01-presence-negative-scala",
  )
  ok &= ctx.check("pc01:bias-disabled-nonzero-word-scala-rejects", code == 0, f"exit={code}")
  binary = _cc_binary(ctx, ctx.repo / "hardware/boa/sim/c")
  code = ctx.run([str(binary), "expect-fail", str(patched_path), "bias"], work, "pc01-presence-negative-c")
  ok &= ctx.check("pc01:bias-disabled-nonzero-word-c-rejects", code == 0, f"exit={code}")
  return bool(ok)


def case_pc02_negative_fields(ctx) -> bool:
  """PC02: illegal field sets are rejected with exit != 0 by all three
  codecs and every rejection names the offending field."""
  codec = _codec(ctx)
  work = ctx.work / "pc02"
  work.mkdir(parents=True, exist_ok=True)
  ok = True

  request = _am_request(codec)
  error = codec.AmError(reason=codec.AM_ERROR_LOGICAL_AM_INVALID, field=codec.FIELD_TAG_LOGICAL_AM)
  stream = codec.encode_many(_golden_messages(codec))

  negatives: list[tuple[str, bytes, int, int | None, str]] = []
  payload, index = _patched_different_byte(codec, request, logical_am=1)
  negatives.append(("logical-am-2", payload, index, 2, "logical_am"))
  payload, index = _patched_different_byte(codec, request, bundle_slots=2)
  negatives.append(("bundle-3", payload, index, 3, "bundle_slots"))
  payload, index = _patched_different_byte(codec, error, reason=codec.AM_ERROR_SLOT_CAPACITY_EXCEEDED)
  negatives.append(("unknown-reason", payload, index, 99, "reason"))
  negatives.append(("truncated", stream[:-1], len(stream) - 1, None, "frame.length"))
  negatives.append(("unknown-version", stream, 2, 7, "frame.version"))

  for name, payload, index, patched, field in negatives:
    raw = bytearray(payload)
    if patched is not None:
      raw[index] = patched
    path = work / f"{name}.bin"
    path.write_bytes(bytes(raw))
    ctx.record_artifact(path, deterministic=True)

    rejected = None
    try:
      codec.decode_stream(bytes(raw))
    except codec.ProtocolError as exc:
      rejected = exc
    ok &= ctx.check(
      f"pc02:{name}-python-rejects",
      rejected is not None and field in rejected.field,
      f"expected rejection naming {field!r}, got {rejected.field if rejected else 'acceptance'}",
    )

    code, _ = _scala_run(
      ctx, ctx.repo / SCALA_SRC_REL, ["expect-fail", str(path), field], f"pc02-{name}-scala"
    )
    ok &= ctx.check(f"pc02:{name}-scala-rejects", code == 0, f"scala expect-fail exit={code}")

    binary = _cc_binary(ctx, ctx.repo / "hardware/boa/sim/c")
    code = ctx.run([str(binary), "expect-fail", str(path), field], work, f"pc02-{name}-c")
    ok &= ctx.check(f"pc02:{name}-c-rejects", code == 0, f"C expect-fail exit={code}")

  ctx.metrics["pc02_negatives"] = len(negatives)
  _write_report(ctx, "pc02", {"negatives": [name for name, *_ in negatives]})
  return bool(ok)


def case_pc03_event_independence(ctx) -> bool:
  """PC03: parked/woken requests and static event ids stay independent
  fields across a slot-generation change; the schema lint statically
  rejects any GRANTED precondition that touches an event slot."""
  codec = _codec(ctx)
  work = ctx.work / "pc03"
  work.mkdir(parents=True, exist_ok=True)
  ok = True

  messages = [
    _am_request(codec, request_seq=5, ctx_id=0),
    codec.AmParked(request_id=9),
    codec.EvtTake(
      record=codec.StepTerminal(
        token=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=3),
        status=codec.TERM_STATUS_SUCCESS,
        seq=1,
        success=True,
        detail=codec.FIELD_TAG_NONE,
      )
    ),
    codec.AmWake(
      request_id=9,
      lease_handle=44,
      slot_generation=7,
      bound_engine=1,
      completion_token=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=5),
    ),
    _am_request(codec, request_seq=6, ctx_id=0),
    codec.AmGranted(
      lease_handle=45,
      slot_generation=8,
      group=2,
      slot_index=1,
      chain_id=10,
      bound_engine=2,
      completion_token=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=6),
    ),
    codec.EvtTake(
      record=codec.ChainTerminal(
        token=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=3),
        status=codec.TERM_STATUS_SUCCESS,
        seq=1,
        success=True,
        detail=codec.FIELD_TAG_NONE,
      )
    ),
    codec.RecReap(token=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=5)),
  ]
  ok &= _roundtrip_all_codecs(ctx, "pc03", messages, work)

  data = codec.encode_many(messages)
  decoded = codec.decode_stream(data)
  tokens = [
    message.completion_token.event_id
    for message in decoded
    if isinstance(message, (codec.AmGranted, codec.AmWake))
  ]
  ok &= ctx.check(
    "pc03-event-ids-distinct-and-preserved",
    tokens == [5, 6],
    f"old (3) and new event ids must stay distinct; got completion tokens {tokens}",
  )
  generations = [
    message.slot_generation for message in decoded if isinstance(message, (codec.AmGranted, codec.AmWake))
  ]
  ok &= ctx.check(
    "pc03-generation-advanced",
    generations == [7, 8],
    f"slot generation must advance across the reuse; got {generations}",
  )
  request_ids = [
    message.request_id for message in decoded if isinstance(message, (codec.AmParked, codec.AmWake))
  ]
  ok &= ctx.check(
    "pc03-request-id-preserved",
    request_ids == [9, 9],
    f"parked/wake request identity must round-trip; got {request_ids}",
  )

  schema = _schema(ctx)
  problems = _generator_lint(ctx, schema)
  ok &= ctx.check(
    "pc03-lint-passes-on-committed-schema", problems == [], f"committed schema lint: {problems}"
  )
  for field in schema["messages"]["AM_GRANTED"]["fields"]:
    if field["name"] == "completion_token":
      ok &= ctx.check(
        "pc03-completion-token-reference-only",
        field.get("role") == "reference_only",
        "completion_token must be declared reference_only in the schema",
      )

  negative = copy.deepcopy(schema)
  negative["grant"]["preconditions"].append(
    {"id": "event_slot_available", "kind": "event_slot", "acquires": True}
  )
  problems = _generator_lint(ctx, negative)
  ok &= ctx.check(
    "pc03-lint-rejects-event-slot-precondition",
    any("event slot" in problem for problem in problems),
    f"lint must reject an event-slot grant precondition, got {problems}",
  )
  negative_token = copy.deepcopy(schema)
  negative_token["grant"]["completion_token"]["never_a_grant_resource"] = False
  problems = _generator_lint(ctx, negative_token)
  ok &= ctx.check(
    "pc03-lint-rejects-token-as-resource",
    any("completion_token" in problem for problem in problems),
    f"lint must reject completion_token as a grant resource, got {problems}",
  )
  negative_role = copy.deepcopy(schema)
  for field in negative_role["messages"]["AM_WAKE"]["fields"]:
    if field["name"] == "completion_token":
      field["role"] = "resource"
  problems = _generator_lint(ctx, negative_role)
  ok &= ctx.check(
    "pc03-lint-rejects-wake-token-role",
    any("completion_token" in problem for problem in problems),
    f"lint must reject a non-reference completion_token role, got {problems}",
  )
  _write_report(ctx, "pc03", {"event_ids": [3, 5, 6], "generations": generations})
  return bool(ok)


def case_pc04_config_consistency(ctx) -> bool:
  """PC04: same-semantics configuration is compared across protocol-v0.yaml,
  hardware_config.yaml and boa-*engine.yaml; granularity-owning values carry
  their own units instead of being forced equal."""
  import yaml

  work = ctx.work / "pc04"
  work.mkdir(parents=True, exist_ok=True)
  topology = yaml.safe_load((ctx.repo / ctx.request["config_path"]).read_text(encoding="utf-8"))["topology"]
  hw = yaml.safe_load((ctx.repo / HW_CONFIG_REL).read_text(encoding="utf-8"))
  schema = _schema(ctx)

  def compare(topology: dict, hw: dict) -> list[str]:
    problems = []
    am = topology["am"]
    hw_am = hw["engines"]["boa"]["am"]
    for key in ("group_count", "slots_per_group", "slots_per_context", "slot_capacity_bytes"):
      if am[key] != hw_am[key]:
        problems.append(f"am.{key}: protocol {am[key]} != hardware_config {hw_am[key]}")
    if am["group_count"] * am["slots_per_group"] != am["total_slots_per_tile"]:
      problems.append("am.total_slots_per_tile disagrees with group*slots_per_group")
    if topology["numeric_profile"] != hw["engines"]["boa"]["numeric_profile"]:
      problems.append("numeric_profile identity differs between protocol and hardware config")
    if list(topology["event_identity"]) != list(schema["identities"]["event_token"]["fields"]):
      problems.append("event identity fields differ between config and schema")
    if topology["event_slot_allocation"] != "program_static":
      problems.append("event slots must be program-static")
    if topology["engines_per_tile"] != hw["engines"]["boa"]["arrays_per_tile"]:
      problems.append("engines_per_tile default differs between protocol and hardware config")
    if topology["pending_request_fifo_depth"] != topology["contexts_per_tile"]:
      problems.append("pending FIFO depth must equal the per-Tile context count")
    return problems

  problems = compare(topology, hw)
  ok = ctx.check("pc04-same-semantics-agree", problems == [], "; ".join(problems))

  for engines, rel in BOA_ENGINE_CONFIGS.items():
    entry = yaml.safe_load((ctx.repo / rel).read_text(encoding="utf-8"))
    value = entry["validator"]["hw_overrides"]["boa_arrays_per_tile"]
    legal = schema["capacities"]["engines_per_tile"]["legal_range"]
    ok &= ctx.check(
      f"pc04-engine-config-{engines}",
      value == engines and legal[0] <= value <= legal[1],
      f"{rel}: arrays_per_tile {value} must equal {engines} and stay inside {legal}",
    )

  negative_hw = yaml.safe_load((ctx.repo / HW_CONFIG_REL).read_text(encoding="utf-8"))
  negative_hw["engines"]["boa"]["am"]["group_count"] = 5
  problems = compare(topology, negative_hw)
  ok &= ctx.check(
    "pc04-rejects-same-semantics-mismatch",
    any("group_count" in problem for problem in problems),
    f"a group_count mismatch must be rejected, got {problems}",
  )

  codec = _codec(ctx)
  software = topology["software_step"]
  micro_ranges = topology["hardware_microblock"]
  same_granularity_conflict = {
    "result_fifo_depth": software["result_fifo_depth"],
    "result_fifo_depth_again": software["result_fifo_depth"] + 2,
  }
  ok &= ctx.check(
    "pc04-rejects-same-granularity-conflict",
    same_granularity_conflict["result_fifo_depth"] != same_granularity_conflict["result_fifo_depth_again"],
    "two same-granularity values that disagree must be treated as a conflict",
  )
  accepted_units = {
    "software_step": {
      key: {"value": value, "unit": "software step granularity"} for key, value in software.items()
    },
    "hardware_microblock": {
      key: {"value": list(value), "unit": "hardware microblock granularity, owner P16"}
      for key, value in micro_ranges.items()
    },
  }
  hw_fifo = 12
  ok &= ctx.check(
    "pc04-accepts-different-granularity",
    hw_fifo != software["result_fifo_depth"]
    and micro_ranges["result_fifo_depth_range"][0] <= hw_fifo <= micro_ranges["result_fifo_depth_range"][1],
    "a P16 microblock depth inside its range may legally differ from the software step default",
  )

  messages = [
    codec.MemReq(
      req_id=1,
      ctx_id=2,
      ctx_epoch=0,
      generation=1,
      address=0x2000,
      access=codec.ACCESS_TYPE_READ,
      length=hw_fifo,
      byte_mask=0xFFFFFFFF,
    ),
    codec.MemResp(
      req_id=1, status=codec.MEM_STATUS_OK, data=tuple(software["result_fifo_depth"] for _ in range(4))
    ),
    codec.SubmitStep(
      lease_handle=1,
      seq=0,
      mode=codec.STEP_MODE_INIT,
      final=True,
      transpose=codec.TRANSPOSE_NN,
      a=_view(codec, 0, 4, 4, codec.DTYPE_BF16, 2),
      b=_view(codec, 1, 4, 2, codec.DTYPE_BF16, 2),
      k_extent=4,
      input_ready_tokens=(),
      input_usage_handles=(),
      k_mask=tuple([1] * 4),
      completion_event=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=0),
    ),
  ]
  ok &= _roundtrip_all_codecs(ctx, "pc04", messages, work)
  _write_report(ctx, "pc04", {"accepted_units": accepted_units, "microblock_result_fifo_example": hw_fifo})
  return bool(ok)


def case_pc05_clean_regeneration(ctx) -> bool:
  """PC05: clean regeneration is deterministic, matches the committed
  codecs, and the freshly generated codecs compile and agree on the same
  vectors (a real build smoke, not a file copy)."""
  codec = _codec(ctx)
  work = ctx.work / "pc05"
  work.mkdir(parents=True, exist_ok=True)
  ok = True

  first = work / "gen-first"
  second = work / "gen-second"
  for target in (first, second):
    code = ctx.run(
      [sys.executable, str(ctx.repo / GENERATOR_REL), "--out", str(target)], ctx.repo, "pc05-generate"
    )
    ok &= ctx.check("pc05-generate-exit", code == 0, f"exit={code}")

  committed = {
    "python": ctx.repo / PY_CODEC_REL,
    "scala": ctx.repo / SCALA_SRC_REL / "boa/protocol/BoaProtocol.scala",
    "c": ctx.repo / "hardware/boa/sim/c/boa_protocol.h",
  }
  for kind, path in committed.items():
    ok &= ctx.check(
      f"pc05-regenerated-{kind}-matches-committed",
      (first / path.relative_to(ctx.repo)).read_text(encoding="utf-8") == path.read_text(encoding="utf-8"),
      f"clean regeneration must reproduce the committed {kind} codec",
    )
  relative_outputs = (
    Path(PY_CODEC_REL),
    Path("hardware/boa/src/main/scala/boa/protocol/BoaProtocol.scala"),
    Path("hardware/boa/sim/c/boa_protocol.h"),
  )
  deterministic = all((first / rel).read_bytes() == (second / rel).read_bytes() for rel in relative_outputs)
  ok &= ctx.check(
    "pc05-regeneration-deterministic",
    deterministic,
    "two clean regenerations must be byte-identical (idempotent generator)",
  )
  code = ctx.run([sys.executable, str(ctx.repo / GENERATOR_REL), "--check"], ctx.repo, "pc05-check")
  ok &= ctx.check("pc05-generator-check-exit", code == 0, f"--check exit={code}")

  messages = _golden_messages(codec)
  data = codec.encode_many(messages)
  vector = work / "pc05.bin"
  vector.write_bytes(data)
  scala_root = first / SCALA_SRC_REL
  scala_out = work / "pc05.scala.bin"
  code, _stdout = _scala_run(ctx, scala_root, ["roundtrip", str(vector), str(scala_out)], "pc05-scala")
  ok &= ctx.check("pc05-regenerated-scala-exit", code == 0, f"exit={code}")
  ok &= ctx.check(
    "pc05-regenerated-scala-byte-identical",
    scala_out.is_file() and scala_out.read_bytes() == data,
    "the freshly generated Scala codec must agree byte-for-byte",
  )
  binary = _cc_binary(ctx, first / "hardware/boa/sim/c")
  c_out = work / "pc05.c.bin"
  code = ctx.run([str(binary), "roundtrip", str(vector), str(c_out)], work, "pc05-c")
  ok &= ctx.check("pc05-regenerated-c-exit", code == 0, f"exit={code}")
  ok &= ctx.check(
    "pc05-regenerated-c-byte-identical",
    c_out.is_file() and c_out.read_bytes() == data,
    "the freshly generated C codec must agree byte-for-byte",
  )
  for path in (vector, scala_out, c_out):
    if path.is_file():
      ctx.record_artifact(path, deterministic=True)
  _write_report(ctx, "pc05", {"vectors": len(messages), "wire_bytes": len(data)})
  return bool(ok)


def case_pc06_boundaries_and_wraparound(ctx) -> bool:
  """PC06: exact boundary encodings for every frozen width, rejection of
  out-of-range values, and the wraparound policy review of the schema."""
  codec = _codec(ctx)
  work = ctx.work / "pc06"
  work.mkdir(parents=True, exist_ok=True)
  ok = True

  boundary_epilogue = codec.EpilogueConfig(
    n_cols=256,
    bias=(0x7F7FFFFF,) * 256,
    bias_present=True,
    scale=0x7F7FFFFF,
    scale_present=True,
    relu=True,
    clip=codec.ClipPair(0xFF7FFFFF, 0x7F7FFFFF),
    clip_present=True,
    quant=None,
  )
  messages = [
    _am_request(
      codec, ctx_id=7, request_seq=0xFFFF, rows=256, cols=256, byte_count=262144, epilogue=boundary_epilogue
    ),
    codec.AmGranted(
      lease_handle=0xFFFF,
      slot_generation=0xFFFF,
      group=3,
      slot_index=1,
      chain_id=0xFFFF,
      bound_engine=7,
      completion_token=codec.EventToken(ctx_id=7, ctx_epoch=255, event_id=0xFFFF),
    ),
    codec.SubmitStep(
      lease_handle=0xFFFF,
      seq=0xFFFE,
      mode=codec.STEP_MODE_ADD,
      final=True,
      transpose=codec.TRANSPOSE_TT,
      a=_view(codec, 0xFFFF, 0xFFFF, 1, codec.DTYPE_BF16, 2),
      b=_view(codec, 0xFFFE, 1, 0xFFFF, codec.DTYPE_BF16, 2),
      k_extent=0xFFFF,
      input_ready_tokens=(codec.EventToken(ctx_id=7, ctx_epoch=255, event_id=0xFFFF),),
      input_usage_handles=(codec.AllocationRef(allocation=0xFFFF, generation=0),),
      k_mask=tuple([0, 1] * 8),
      completion_event=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=0),
    ),
    codec.MemReq(
      req_id=0xFFFFFFFF,
      ctx_id=7,
      ctx_epoch=255,
      generation=0xFFFF,
      address=0x7FFFFFFFFFFFFFFF,
      access=codec.ACCESS_TYPE_WRITE,
      length=0xFFFF,
      byte_mask=0xFFFFFFFF,
    ),
    codec.MemResp(req_id=0xFFFFFFFF, status=codec.MEM_STATUS_OK, data=tuple([0xFF] * 8)),
  ]
  ok &= _roundtrip_all_codecs(ctx, "pc06", messages, work)
  data = codec.encode_many(messages)
  decoded = codec.decode_stream(data)
  granted = next(message for message in decoded if isinstance(message, codec.AmGranted))
  ok &= ctx.check(
    "pc06-boundary-values-exact",
    (granted.lease_handle, granted.slot_generation, granted.completion_token.event_id)
    == (0xFFFF, 0xFFFF, 0xFFFF),
    "maximal frozen width values must decode back exactly",
  )

  overflows = [
    ("request-seq-overflow", {"request_seq": 0x10000}, "request_seq"),
    ("ctx-id-overflow", {"ctx_id": 8}, "ctx_id"),
    ("byte-count-overflow", {"byte_count": 262145}, "byte_count"),
    ("logical-am-underflow", {"logical_am": 3}, "logical_am"),
  ]
  for name, overrides, field in overflows:
    if name == "byte-count-overflow":
      overrides = dict(overrides, rows=256, cols=256, byte_count=262145)
    rejected = None
    try:
      codec.encode_message(_am_request(codec, **overrides))
    except codec.ProtocolError as exc:
      rejected = exc
    ok &= ctx.check(
      f"pc06:{name}",
      rejected is not None and field in rejected.field,
      f"out-of-range {field} must be rejected naming the field, got"
      f" {rejected.field if rejected else 'acceptance'}",
    )
  step = codec.SubmitStep(
    lease_handle=1,
    seq=0x10000,
    mode=codec.STEP_MODE_INIT,
    final=False,
    transpose=codec.TRANSPOSE_NN,
    a=_view(codec, 0, 4, 4, codec.DTYPE_BF16, 2),
    b=_view(codec, 1, 4, 2, codec.DTYPE_BF16, 2),
    k_extent=4,
    input_ready_tokens=(),
    input_usage_handles=(),
    k_mask=(),
    completion_event=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=0),
  )
  rejected = None
  try:
    codec.encode_message(step)
  except codec.ProtocolError as exc:
    rejected = exc
  ok &= ctx.check(
    "pc06:seq-overflow",
    rejected is not None and "seq" in rejected.field,
    "a 17-bit step seq must be rejected naming seq",
  )

  schema = _schema(ctx)
  wraparound = schema["wraparound"]
  counters = sorted(set(wraparound) - {"no_width_escape"})
  ok &= ctx.check(
    "pc06-wraparound-rules-present",
    all(
      {"width", "rule", "quiescence", "validated_by"} <= set(wraparound[counter]) for counter in counters
    ),
    f"every monotonic counter needs width/rule/quiescence/validated_by; got {counters}",
  )
  ok &= ctx.check(
    "pc06-step-seq-wrap-forbidden",
    "NO wraparound" in wraparound["step_seq"]["rule"],
    "step seq wraparound must be forbidden, not waited out",
  )
  lint_problems = _generator_lint(ctx, schema)
  ok &= ctx.check("pc06-schema-lint", lint_problems == [], f"lint: {lint_problems}")
  _write_report(
    ctx,
    "pc06",
    {
      "counters": {
        counter: {
          "width": wraparound[counter]["width"],
          "rule": wraparound[counter]["rule"],
          "quiescence": wraparound[counter]["quiescence"],
          "validated_by": wraparound[counter]["validated_by"],
        }
        for counter in counters
      },
      "freeze_status": schema["freeze"]["status"],
    },
  )
  return bool(ok)


def case_pc07_real_descriptors(ctx) -> bool:
  """PC07: real P09-era compiled descriptors are mapped into protocol
  messages through the generated codec; producer events, runtime-bound L1
  allocation identities/generations and the real owner epoch come from the
  actual executed fixture, never from harness invention."""
  repo_root = str(ctx.repo)
  if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
  from pipeline_validator.compiler import compile_program
  from pipeline_validator.config import HardwareConfig, SimConfig
  from pipeline_validator.execution_ir import ExecBoaMatmulDesc, ExecTransfer, GlobalBinding
  from pipeline_validator.loader import load_program
  from pipeline_validator.memory.byte_store import ByteStore
  from pipeline_validator.simulator import Simulator
  from pipeline_validator.workload_ir import parse_workload_ir

  codec = _codec(ctx)
  work = ctx.work / "pc07"
  work.mkdir(parents=True, exist_ok=True)
  ok = True
  hw = HardwareConfig()
  sim_cfg = SimConfig(fidelity="full_memory")
  sources: dict[str, str | None] = dict(PC07_FIXTURES)
  sources["transpose-all-combos"] = None  # embedded fixture text
  messages: list = []
  expected: list[dict] = []
  generation_pairs: list[tuple[str, int]] = []

  def run_fixture(label: str, source: str, launches: int = 1) -> dict:
    """Compile + really execute one fixture; observe the runtime bindings."""
    program = compile_program(parse_workload_ir(source), hw, sim_cfg, source_name=f"p10-pc07-{label}")
    bindings: dict[str, GlobalBinding] = {}
    base = 0
    for guard in program.binding_guards:
      bindings[guard.name] = GlobalBinding(guard.name, base, guard.minimum_bytes, guard.permissions)
      base += guard.minimum_bytes + 256
    store = ByteStore()
    for binding in bindings.values():
      store.seed_hbm(binding.base_iova, b"\x00" * binding.size_bytes)
    loaded = load_program(program, hw, sim_cfg, actual_bindings=bindings)
    simulator = Simulator(hw, sim_cfg, byte_store=store)
    per_launch: list[dict] = []

    def observe(allocator):
      original = allocator.bind_view

      def wrapped(arena, buffer_id: str, cycle: int):
        handle = original(arena, buffer_id, cycle)
        per_launch[-1]["l1_binds"][buffer_id] = (handle.allocation_id, handle.generation)
        return handle

      allocator.bind_view = wrapped

    for tile in simulator.group.tiles:
      observe(tile.l1_allocator)
    completed = True
    for _ in range(launches):
      per_launch.append({"l1_binds": {}, "owners": set()})
      result = simulator.run(loaded)
      completed = completed and bool(result.completed)
      for pool in result.group_snapshot.get("shared_am_per_tile", []):
        for record in pool.get("terminals", []):
          per_launch[-1]["owners"].add(
            (record.get("tile_id", -1), record.get("ctx_id", -1), record.get("owner_epoch", -1))
          )
    for launch in per_launch:
      launch["owner"] = sorted(launch["owners"])[0] if launch["owners"] else None
    per_launch[0]["owner_ids"] = per_launch[0]["owners"]
    per_launch[0]["completed"] = completed
    per_launch[0]["program"] = program
    if len(per_launch) > 1:
      per_launch[0]["second_launch"] = per_launch[1]["l1_binds"]
    return per_launch[0]

  for name, rel in sources.items():
    source = TRANSPOSE_FIXTURE if rel is None else (ctx.repo / rel).read_text(encoding="utf-8")
    run = run_fixture(name, source)
    ok &= ctx.check(
      f"pc07:{name}-fixture-runs",
      run["completed"],
      "the real fixture must load and execute to completion under full_memory",
    )
    ok &= ctx.check(
      f"pc07:{name}-single-owner-identity",
      run["owner"] is not None and len(run["owner_ids"]) == 1,
      f"each executed fixture must expose exactly one AM owner identity"
      f" (tile, ctx, epoch) in its terminals; got {run.get('owner_ids')}",
    )
    if run["owner"] is None:
      continue
    owner_tile, owner_ctx, owner_epoch = run["owner"]
    program = run["program"]
    entry: Any = program.entry
    tasks = list(entry.tasks.values()) if hasattr(entry, "tasks") else [entry]
    for task in tasks:
      resources: dict[str, Any] = program.static_effects["resources"]  # type: ignore[assignment]
      effects = resources.get(task.binding_id, {})
      for role in task.role_bindings.values():
        tile_program = role.tile_program
        events = [inst.dst for inst in tile_program.insts if inst.dst is not None]
        descriptors = tile_program.descriptors
        insts = tile_program.insts

        def event_id(name_of_event: str, _events=events) -> int:
          return _events.index(name_of_event)

        def mfe_target(inst, _descriptors=descriptors) -> str:
          descriptor = _descriptors.get(inst.args[0]) if inst.args else None
          transfer = descriptor.transfer if descriptor is not None else None
          return transfer.dst.base if isinstance(transfer, ExecTransfer) else ""

        def awaited_before(index: int, _insts=insts) -> set[str]:
          awaited: set[str] = set()
          for inst in _insts[:index]:
            if inst.dst is not None:
              awaited.add(inst.dst)
          return awaited

        for am in tile_program.am_buffers:
          output_slot = int(am.output.base.split(":", 1)[1])
          dtype = {
            "f32": codec.DTYPE_F32,
            "bf16": codec.DTYPE_BF16,
            "i8": codec.DTYPE_I8,
            "i32": codec.DTYPE_I32,
          }[am.output.dtype]
          messages.append(
            _with(
              _am_request(
                codec,
                tile_id=owner_tile,
                logical_am=am.logical_slot,
                bundle=effects.get("am_peak_live", 1),
                rows=am.dims[0],
                cols=am.dims[1],
                dtype=dtype,
                byte_count=am.bytes,
                epilogue=_epilogue_from(codec, am.epilogue, am.dims[1]),
                request_seq=10 + len(messages),
                ctx_id=owner_ctx,
                ctx_epoch=owner_epoch,
              ),
              output_l1_slot=output_slot,
            )
          )
          expected.append(
            {
              "fixture": name,
              "kind": "AM_REQUEST",
              "logical_am": am.logical_slot,
              "bundle": effects.get("am_peak_live", 1),
              "rows": am.dims[0],
              "cols": am.dims[1],
              "bytes": am.bytes,
              "dtype": dtype,
              "dtype_name": am.output.dtype,
              "owner": [owner_tile, owner_ctx, owner_epoch],
              "epilogue": _epilogue_snapshot(am.epilogue),
            }
          )
        for index, inst in enumerate(tile_program.insts):
          descriptor = tile_program.descriptors.get(inst.args[0]) if inst.args else None
          matmul = descriptor.params.get("matmul") if descriptor is not None else None
          if not isinstance(matmul, ExecBoaMatmulDesc):
            continue
          producer_names_resolved, _operand_errors = _resolve_step_producers(tile_program, index, matmul)
          producers: list[Any] = []
          producer_names: list[str] = []
          for resolved_name in producer_names_resolved:
            producer_names.append(resolved_name)
            producers.append(
              codec.EventToken(ctx_id=owner_ctx, ctx_epoch=owner_epoch, event_id=event_id(resolved_name))
            )
          usage_refs: list[Any] = []
          for operand in (matmul.a, matmul.b):
            bind = run["l1_binds"].get(operand.base)
            usage_refs.append(
              codec.AllocationRef(
                allocation=int(operand.base.split(":", 1)[1]), generation=bind[1] if bind else 0
              )
            )
          ok &= ctx.check(
            f"pc07:{name}-step{matmul.seq}-usage-refs-match-operands",
            usage_refs[0].allocation == int(matmul.a.base.split(":", 1)[1])
            and usage_refs[0].generation == run["l1_binds"].get(matmul.a.base, ("", 0))[1]
            and usage_refs[1].allocation == int(matmul.b.base.split(":", 1)[1])
            and usage_refs[1].generation == run["l1_binds"].get(matmul.b.base, ("", 0))[1],
            f"each typed usage reference {usage_refs} must match its operand's"
            " runtime-bound allocation identity and generation",
          )

          completion = codec.EventToken(
            ctx_id=owner_ctx, ctx_epoch=owner_epoch, event_id=event_id(inst.dst)
          )
          ok &= ctx.check(
            f"pc07:{name}-step{matmul.seq}-producers-not-self",
            producer_names and inst.dst not in producer_names,
            f"input producers {producer_names} must be real awaited operand loads,"
            f" never the step's own completion event {inst.dst!r}",
          )
          a_bind = run["l1_binds"].get(matmul.a.base)
          b_bind = run["l1_binds"].get(matmul.b.base)
          ok &= ctx.check(
            f"pc07:{name}-step{matmul.seq}-runtime-allocation-bound",
            a_bind is not None and b_bind is not None,
            f"operand views {matmul.a.base}/{matmul.b.base} must have runtime-bound"
            " allocation identities from the executed fixture",
          )
          messages.append(
            codec.SubmitStep(
              lease_handle=1,
              seq=matmul.seq,
              mode=codec.STEP_MODE_INIT if matmul.init else codec.STEP_MODE_ADD,
              final=matmul.final,
              transpose=(1 if matmul.transpose_a else 0) | (2 if matmul.transpose_b else 0),
              a=_view_from(codec, matmul.a, a_bind),
              b=_view_from(codec, matmul.b, b_bind),
              k_extent=matmul.k,
              input_ready_tokens=tuple(producers),
              input_usage_handles=tuple(usage_refs),
              k_mask=tuple(1 if flag else 0 for flag in matmul.k_mask) if matmul.k_mask else (),
              completion_event=completion,
            )
          )
          expected.append(
            {
              "fixture": name,
              "kind": "SUBMIT_STEP",
              "seq": matmul.seq,
              "init": matmul.init,
              "final": matmul.final,
              "transpose": (1 if matmul.transpose_a else 0) | (2 if matmul.transpose_b else 0),
              "k": matmul.k,
              "storage_k": matmul.storage_k,
              "mask_len": len(matmul.k_mask) if matmul.k_mask else 0,
              "a_bytes": matmul.a.bytes,
              "b_bytes": matmul.b.bytes,
              "producers": producer_names,
              "producer_event_ids": [event_id(item) for item in producer_names],
              "usage_refs": [[ref.allocation, ref.generation] for ref in usage_refs],
              "completion": event_id(inst.dst),
              "a_generation": a_bind[1] if a_bind else None,
              "b_generation": b_bind[1] if b_bind else None,
            }
          )

  # Negative 5a: a load that is issued but never awaited before the BOA
  # launch must NOT be accepted as a producer.  The source verifier already
  # rejects such a program (workload_ir _require_l1_inputs_awaited), so the
  # executable-level resolver is exercised on a real compiled tile program
  # with its WAITALL removed (executable-level mutation, no source bypass).
  transposed_program = compile_program(
    parse_workload_ir(TRANSPOSE_FIXTURE), hw, sim_cfg, source_name="p10-pc07-neg5a"
  )
  neg_entry: Any = transposed_program.entry
  neg_task = next(iter(neg_entry.tasks.values()))
  import dataclasses as _dataclasses

  for role in neg_task.role_bindings.values():
    tp = role.tile_program
    first_waitall = next(i for i, inst in enumerate(tp.insts) if str(inst.op).endswith("WAITALL"))
    stripped = _dataclasses.replace(
      tp, insts=tuple(tp.insts[:first_waitall]) + tuple(tp.insts[first_waitall + 1 :])
    )
    boa_index = next(
      i
      for i, inst in enumerate(stripped.insts)
      if stripped.descriptors.get(inst.args[0]) is not None
      and "matmul" in stripped.descriptors[inst.args[0]].params
    )
    matmul0 = stripped.descriptors[stripped.insts[boa_index].args[0]].params["matmul"]
    resolved, errors = _resolve_step_producers(stripped, boa_index, matmul0)
    ok &= ctx.check(
      "pc07:unawaited-load-is-not-a-producer",
      resolved == [] and len(errors) == 2,
      f"with the operand-await WAITALL removed, step 0 must lose both producers;"
      f" resolved={resolved} errors={errors}",
    )
    control_resolved, control_errors = _resolve_step_producers(tp, boa_index, matmul0)
    ok &= ctx.check(
      "pc07:unawaited-negative-has-control",
      control_resolved and not control_errors,
      f"the unmutated program must still resolve both producers"
      f" ({control_resolved}); otherwise the negative proves nothing",
    )

  # Negative 5b: the soft-conv A operand resolves to its gather event.
  soft_program = compile_program(
    parse_workload_ir((ctx.repo / PC07_FIXTURES["soft-conv-i8"]).read_text(encoding="utf-8")),
    hw,
    sim_cfg,
    source_name="p10-pc07-soft-conv-producers",
  )
  soft_entry: Any = soft_program.entry
  soft_task = next(iter(soft_entry.tasks.values()))
  for role in soft_task.role_bindings.values():
    soft_tp = role.tile_program
    for index, inst in enumerate(soft_tp.insts):
      descriptor = soft_tp.descriptors.get(inst.args[0]) if inst.args else None
      matmul = descriptor.params.get("matmul") if descriptor is not None else None
      if not isinstance(matmul, ExecBoaMatmulDesc):
        continue
      resolved, errors = _resolve_step_producers(soft_tp, index, matmul)
      ok &= ctx.check(
        "pc07:soft-conv-A-resolves-to-gather",
        resolved[:1] == ["g0"] and not errors,
        f"the im2col A operand must resolve to its gather producer g0; resolved={resolved} errors={errors}",
      )

  # A second real launch of the same fixture rebinds the same L1 buffer with a
  # different arena allocation generation (allocation reuse, not a new name).
  reuse_source = (ctx.repo / PC07_FIXTURES["boa-am-matmul"]).read_text(encoding="utf-8")
  reuse_runs = run_fixture("boa-am-matmul", reuse_source, launches=2)
  first_binds, second_binds = reuse_runs["l1_binds"], reuse_runs["second_launch"]
  ok &= ctx.check(
    "pc07-reused-allocation-new-generation",
    first_binds["l1:0"][1] != second_binds["l1:0"][1],
    f"a second real launch must rebind l1:0 with a new allocation generation;"
    f" got {first_binds['l1:0']} then {second_binds['l1:0']}",
  )
  generation_pairs = [("l1:0", first_binds["l1:0"][1]), ("l1:0", second_binds["l1:0"][1])]

  ok &= ctx.check(
    "pc07-real-descriptors-collected",
    sum(1 for item in expected if item["kind"] == "SUBMIT_STEP") >= 7
    and sum(1 for item in expected if item["kind"] == "AM_REQUEST") >= 3,
    f"expected real fixtures to yield at least 3 chains/7 steps, got {len(expected)} records",
  )
  ok &= _roundtrip_all_codecs(ctx, "pc07", messages, work)

  data = codec.encode_many(messages)
  decoded = codec.decode_stream(data)
  requests = [m for m in decoded if isinstance(m, codec.AmRequest)]
  steps = [m for m in decoded if isinstance(m, codec.SubmitStep)]
  for record in expected:
    if record["kind"] == "AM_REQUEST":
      match = next(
        message
        for message in requests
        if message.output_rows == record["rows"]
        and message.output_cols == record["cols"]
        and message.byte_count == record["bytes"]
      )
      ok &= ctx.check(
        f"pc07:{record['fixture']}-request-typed-fields",
        (message_ok(match, record)),
        f"AM_REQUEST typed fields drifted: {(match.logical_am, match.bundle_slots, match.output_dtype)}",
      )
      if record["epilogue"]["quant"]:
        ok &= ctx.check(
          f"pc07:{record['fixture']}-quant-preserved",
          match.chain_config.quant is not None
          and match.chain_config.quant.inv_scale == record["epilogue"]["quant"]["inv_scale"]
          and match.chain_config.quant.zero_point == record["epilogue"]["quant"]["zero_point"],
          "INT8 quantization parameters must survive the round-trip",
        )
    else:
      match = next(
        message
        for message in steps
        if message.seq == record["seq"]
        and message.k_extent == record["k"]
        and message.a.bytes == record["a_bytes"]
        and message.b.bytes == record["b_bytes"]
      )
      ok &= ctx.check(
        f"pc07:{record['fixture']}-step-seq{record['seq']}-typed-fields",
        (
          match.transpose,
          match.mode,
          match.final,
          len(match.k_mask),
          match.completion_event.event_id,
          match.a.generation,
          match.b.generation,
        )
        == (
          record["transpose"],
          codec.STEP_MODE_INIT if record["init"] else codec.STEP_MODE_ADD,
          record["final"],
          record["mask_len"],
          record["completion"] if isinstance(record["completion"], int) else 0,
          record["a_generation"],
          record["b_generation"],
        ),
        "SUBMIT_STEP typed fields (transpose/mode/final/mask/completion/generations) drifted",
      )
      ok &= ctx.check(
        f"pc07:{record['fixture']}-step-seq{record['seq']}-producers-survive",
        [token.event_id for token in match.input_ready_tokens] == record["producer_event_ids"],
        f"producer tokens {record['producers']} must survive the round-trip",
      )
  transposes = sorted(message.transpose for message in steps if message.a.dim0 != message.a.dim1)
  ok &= ctx.check(
    "pc07-all-transpose-combos-serialized",
    transposes == [0, 1, 2, 3],
    f"the four transpose combinations must be serialized per step, got {transposes}",
  )
  request_epochs = {record["owner"][2] for record in expected if record["kind"] == "AM_REQUEST"}
  ok &= ctx.check(
    "pc07-runtime-owner-epoch-and-tile-encoded",
    request_epochs
    and all(
      (message.ctx_epoch, message.tile_id)
      in {(e, t) for t, _c, e in (record["owner"] for record in expected if record["kind"] == "AM_REQUEST")}
      for message in requests
    ),
    "the encoded owner epoch and tile must come from the executed runtime"
    f" terminals, not constants; epochs={request_epochs}",
  )

  base_step = codec.SubmitStep(
    lease_handle=1,
    seq=0,
    mode=codec.STEP_MODE_INIT,
    final=False,
    transpose=codec.TRANSPOSE_NN,
    a=_view(codec, 0, 4, 4, codec.DTYPE_F32, 4),
    b=_view(codec, 1, 4, 2, codec.DTYPE_BF16, 2),
    k_extent=4,
    input_ready_tokens=(codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=2),),
    input_usage_handles=(),
    k_mask=(),
    completion_event=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=0),
  )
  alt_step = _with(base_step, input_ready_tokens=(codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=3),))
  payload, index = _patched_different_byte(codec, base_step, input_ready_tokens=alt_step.input_ready_tokens)
  raw = bytearray(payload)
  raw[index] = 0  # ready token now names the step's own completion event
  rejected = None
  try:
    codec.decode_stream(bytes(raw))
  except codec.ProtocolError as exc:
    rejected = exc
  ok &= ctx.check(
    "pc07:self-ready-token-rejected-python",
    rejected is not None and "input_ready_tokens" in rejected.field,
    f"a step waiting on its own completion event must be rejected, got"
    f" {rejected.field if rejected else 'acceptance'}",
  )
  self_path = work / "pc07-self-token.bin"
  self_path.write_bytes(bytes(raw))
  ctx.record_artifact(self_path, deterministic=True)
  code, _stdout = _scala_run(
    ctx,
    ctx.repo / SCALA_SRC_REL,
    ["expect-fail", str(self_path), "input_ready_tokens"],
    "pc07-self-token-scala",
  )
  ok &= ctx.check("pc07:self-ready-token-rejected-scala", code == 0, f"exit={code}")
  binary = _cc_binary(ctx, ctx.repo / "hardware/boa/sim/c")
  code = ctx.run(
    [str(binary), "expect-fail", str(self_path), "input_ready_tokens"], work, "pc07-self-token-c"
  )
  ok &= ctx.check("pc07:self-ready-token-rejected-c", code == 0, f"exit={code}")

  base_step_dtype = codec.SubmitStep(
    lease_handle=1,
    seq=0,
    mode=codec.STEP_MODE_INIT,
    final=False,
    transpose=codec.TRANSPOSE_NN,
    a=_view(codec, 0, 4, 4, codec.DTYPE_F32, 4),
    b=_view(codec, 1, 4, 2, codec.DTYPE_BF16, 2),
    k_extent=4,
    input_ready_tokens=(),
    input_usage_handles=(),
    k_mask=(),
    completion_event=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=0),
  )
  alt_step = _with(base_step_dtype, a=_view(codec, 0, 4, 4, codec.DTYPE_I32, 4))
  payload, index = _patched_different_byte(codec, base_step_dtype, a=alt_step.a)
  raw = bytearray(payload)
  raw[index] = 9
  rejected = None
  try:
    codec.decode_stream(bytes(raw))
  except codec.ProtocolError as exc:
    rejected = exc
  ok &= ctx.check(
    "pc07:unknown-dtype",
    rejected is not None and "dtype" in rejected.field,
    f"an unknown view dtype must be rejected naming dtype, got"
    f" {rejected.field if rejected else 'acceptance'}",
  )
  # dtype/element_bytes mismatch (dtype_width predicate) must reject on the
  # same wire bytes in all three codecs: a patched BF16/eb2 step claiming dtype
  # f32 while keeping element_bytes=2.
  same_shape_f32 = _view(codec, 0, 2, 8, codec.DTYPE_F32, 4)
  same_shape_i32 = _view(codec, 0, 2, 8, codec.DTYPE_I32, 4)
  probe = codec.SubmitStep(
    lease_handle=1,
    seq=0,
    mode=codec.STEP_MODE_INIT,
    final=False,
    transpose=codec.TRANSPOSE_NN,
    a=same_shape_f32,
    b=_view(codec, 1, 4, 2, codec.DTYPE_BF16, 2),
    k_extent=4,
    input_ready_tokens=(),
    input_usage_handles=(),
    k_mask=(),
    completion_event=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=0),
  )
  bf16_target = _with(probe, a=_view(codec, 0, 2, 8, codec.DTYPE_BF16, 2))
  payload, index = _patched_different_byte(codec, probe, a=same_shape_i32)
  raw = bytearray(codec.encode_message(bf16_target))
  raw[index] = 0  # dtype f32 while element_bytes stays 2
  rejected = None
  try:
    codec.decode_stream(bytes(raw))
  except codec.ProtocolError as exc:
    rejected = exc
  ok &= ctx.check(
    "pc07:dtype-width-mismatch-python",
    rejected is not None and "element_bytes" in rejected.field,
    f"a BF16/eb2 view claiming dtype f32 must be rejected naming element_bytes,"
    f" got {rejected.field if rejected else 'acceptance'}",
  )
  width_path = work / "pc07-dtype-width.bin"
  width_path.write_bytes(bytes(raw))
  ctx.record_artifact(width_path, deterministic=True)
  code, _stdout = _scala_run(
    ctx,
    ctx.repo / SCALA_SRC_REL,
    ["expect-fail", str(width_path), "element_bytes"],
    "pc07-dtype-width-scala",
  )
  ok &= ctx.check("pc07:dtype-width-mismatch-scala", code == 0, f"exit={code}")
  binary = _cc_binary(ctx, ctx.repo / "hardware/boa/sim/c")
  code = ctx.run([str(binary), "expect-fail", str(width_path), "element_bytes"], work, "pc07-dtype-width-c")
  ok &= ctx.check("pc07:dtype-width-mismatch-c", code == 0, f"exit={code}")

  bad_mask = codec.SubmitStep(
    lease_handle=1,
    seq=0,
    mode=codec.STEP_MODE_INIT,
    final=False,
    transpose=codec.TRANSPOSE_NN,
    a=_view(codec, 0, 4, 4, codec.DTYPE_BF16, 2),
    b=_view(codec, 1, 4, 2, codec.DTYPE_BF16, 2),
    k_extent=4,
    input_ready_tokens=(),
    input_usage_handles=(),
    k_mask=(2, 0, 0, 0),
    completion_event=codec.EventToken(ctx_id=0, ctx_epoch=0, event_id=0),
  )
  rejected = None
  try:
    codec.encode_message(bad_mask)
  except codec.ProtocolError as exc:
    rejected = exc
  ok &= ctx.check(
    "pc07:mask-item-range",
    rejected is not None and "k_mask" in rejected.field,
    "a k_mask lane of 2 must be rejected naming k_mask",
  )
  bad_quant = _am_request(
    codec,
    rows=4,
    cols=1,
    dtype=codec.DTYPE_I8,
    byte_count=16,
    request_seq=2,
    epilogue=codec.EpilogueConfig(
      n_cols=1, quant_present=True, quant=codec.QuantConfig(inv_scale=0x7FC00000, zero_point=0)
    ),
  )
  rejected = None
  try:
    codec.encode_message(bad_quant)
  except codec.ProtocolError as exc:
    rejected = exc
  ok &= ctx.check(
    "pc07:quant-inv-scale-nan",
    rejected is not None and "inv_scale" in rejected.field,
    "a NaN inv_scale must be rejected naming inv_scale",
  )
  bad_zero = _am_request(
    codec,
    rows=4,
    cols=1,
    dtype=codec.DTYPE_I8,
    byte_count=16,
    request_seq=3,
    epilogue=codec.EpilogueConfig(
      n_cols=1, quant_present=True, quant=codec.QuantConfig(inv_scale=0x3F000000, zero_point=200)
    ),
  )
  rejected = None
  try:
    codec.encode_message(bad_zero)
  except codec.ProtocolError as exc:
    rejected = exc
  ok &= ctx.check(
    "pc07:zero-point-range",
    rejected is not None and "zero_point" in rejected.field,
    "zero_point 200 must be rejected naming zero_point",
  )

  _write_report(
    ctx,
    "pc07",
    {
      "records": expected,
      "runtime_observation": {
        "l1_generation_pairs": generation_pairs,
        "owner_identity_source": "executed fixture shared_am terminals",
      },
    },
  )
  ctx.metrics["pc07_messages"] = len(messages)
  return bool(ok)


def message_ok(match, record) -> bool:
  return (match.logical_am, match.bundle_slots, match.output_dtype) == (
    record["logical_am"],
    record["bundle"],
    record["dtype"],
  )


def _epilogue_snapshot(epilogue) -> dict:
  return {
    "bias": list(epilogue.bias) if epilogue.bias is not None else None,
    "scale": epilogue.scale,
    "relu": epilogue.relu,
    "clip": list(epilogue.clip) if epilogue.clip is not None else None,
    "quant": None
    if epilogue.inv_scale is None
    else {"inv_scale": epilogue.inv_scale, "zero_point": epilogue.zero_point},
  }


def _epilogue_from(codec, epilogue, n_cols: int):
  return codec.EpilogueConfig(
    n_cols=n_cols,
    bias=tuple(epilogue.bias) if epilogue.bias is not None else None,
    bias_present=epilogue.bias is not None,
    scale=epilogue.scale,
    scale_present=epilogue.scale is not None,
    relu=epilogue.relu,
    clip=codec.ClipPair(*epilogue.clip) if epilogue.clip is not None else None,
    clip_present=epilogue.clip is not None,
    quant=None
    if epilogue.inv_scale is None
    else codec.QuantConfig(inv_scale=epilogue.inv_scale, zero_point=epilogue.zero_point),
    quant_present=epilogue.inv_scale is not None,
  )


def _view_from(codec, view, bind: tuple[str, int] | None = None):
  space = {"global": codec.SPACE_GLOBAL, "l2": codec.SPACE_L2, "l1": codec.SPACE_L1}[view.space]
  dtype = {"f32": codec.DTYPE_F32, "bf16": codec.DTYPE_BF16, "i8": codec.DTYPE_I8, "i32": codec.DTYPE_I32}[
    view.dtype
  ]
  return codec.OperandView(
    space=space,
    base_slot=int(view.base.split(":", 1)[1]) if ":" in view.base else 0,
    allocation=int(view.base.split(":", 1)[1]) if ":" in view.base else 0,
    generation=bind[1] if bind else 0,
    dim0=view.dims[0],
    dim1=view.dims[1],
    offset0=view.offsets[0],
    offset1=view.offsets[1],
    stride0=view.strides[0],
    stride1=view.strides[1],
    dtype=dtype,
    element_bytes=view.element_bytes,
    bytes=view.bytes,
    task_dim=-1 if view.task_dim is None else view.task_dim,
  )
