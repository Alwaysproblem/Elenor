"""Generate the software im2col Conv fixtures and their input bytes.

Software Conv path (P08 plan section 5 step 7): the im2col expansion of a
conv case is materialised byte-true on the device.  Every expanded element
is moved by ``tile.gather.global.async`` from the X global binding; padding
positions read an explicit trailing +0 BF16 word appended to X, never an
out-of-bounds address and never an implicitly-zero L1 byte.  The reduced
matmul is one typed ``tile.boa.async`` AM chain whose per-step K depth
mirrors the P03 ``tile_k`` block schedule (each step is a single
contribution block at the default ``boa_numeric_tile_k`` >= case tile_k),
with the optional P08 FINAL epilogue carried by the AM allocation.  EVU
carries no im2col bytes; the fixtures contain no EVU op at all.

Layout contract (identical to the P03 ``conv2d_reference``): x is NCHW,
weights are OIHW, K is flattened ``(ic, kh, kw)``, matmul rows are
``(batch, oh, ow)`` and the output binding is ``[N*OH*OW, O]`` row-major.
The B input binding is the host-side ``[K, O]`` repack of the OIHW weights
(``b_hex`` of the P03 vector); the A matrix is never shipped from the host.

Golden authority: ``hardware/boa/reference/vectors/conv.json`` (P03,
unchanged).  The one generated ``pad1_stride1`` case derives its golden
through the unchanged P03 ``conv2d_reference`` (a generator may call the
reference package to produce goldens, never DUT bytes) and commits it to
``soft_conv_inputs/pad1_stride1/golden.json``; tests read committed
goldens only.

The generator is deterministic: regenerating fixtures and inputs is
byte-identical.  Regenerate with
``PYTHONPATH=.:examples/generators python examples/generators/generate_soft_conv.py``
from the repository root.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from transformer_common import contract_bytes, write_workload

from pipeline_validator.config import HardwareConfig
from pipeline_validator.workload_ir import parse_workload_ir

REPO = Path(__file__).resolve().parents[2]
WORKLOADS = REPO / "examples" / "workloads"
INPUT_ROOT = WORKLOADS / "soft_conv_inputs"
VECTORS = REPO / "hardware" / "boa" / "reference" / "vectors" / "conv.json"
REFERENCE_ROOT = REPO / "hardware" / "boa" / "reference"

DTYPE_BYTES = {"bf16": 2, "f32": 4, "i8": 1}
ALIGN = 256
WINDOW_ENTRIES = 16
ZERO_WORD = 0x0000  # explicit +0 BF16 source for every padding element

# P03 vector ids the fixtures cover.  The first six are the f32 no-epilogue
# cases; conv-i8-out exercises the P08 FINAL epilogue (inv_scale/zero_point
# with an i8 output binding).
P03_CASE_IDS = (
  "conv-ones-2x2",
  "conv-asymmetric",
  "conv-pad-poison",
  "conv-k-order",
  "conv-tile4",
  "conv-tile8",
  "conv-i8-out",
)
GENERATED_CASE_ID = "pad1_stride1"


# ---------------------------------------------------------------------------
# Geometry, im2col indices and the host-side B repack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvGeometry:
  n: int
  c: int
  h: int
  w: int
  o: int
  kh: int
  kw: int
  sh: int
  sw: int
  pt: int
  pb: int
  pl: int
  pr: int
  dh: int
  dw: int
  oh: int
  ow: int

  @property
  def m(self) -> int:
    return self.n * self.oh * self.ow

  @property
  def k(self) -> int:
    return self.c * self.kh * self.kw

  @property
  def x_elements(self) -> int:
    return self.n * self.c * self.h * self.w


def geometry(
  shape: tuple[int, int, int, int],
  kernel: tuple[int, int, int, int],
  stride: tuple[int, int],
  padding: tuple[int, int, int, int],
  dilation: tuple[int, int],
) -> ConvGeometry:
  n, c, h, w = shape
  o, kc, kh, kw = kernel
  if kc != c:
    raise ValueError(f"kernel channel count {kc} does not match shape {c}")
  sh, sw = stride
  dh, dw = dilation
  pt, pb, pl, pr = padding
  ekh = (kh - 1) * dh + 1
  ekw = (kw - 1) * dw + 1
  oh = max(0, (h + pt + pb - ekh) // sh + 1)
  ow = max(0, (w + pl + pr - ekw) // sw + 1)
  return ConvGeometry(
    n, c, h, w, o, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, oh, ow
  )


def valid_patch_elements(geo: ConvGeometry) -> int:
  """Analytic V: in-bounds patch reads counted per expanded element."""
  valid_rows = sum(
    1
    for oh_i in range(geo.oh)
    for kh_i in range(geo.kh)
    if 0 <= oh_i * geo.sh - geo.pt + kh_i * geo.dh < geo.h
  )
  valid_cols = sum(
    1
    for ow_i in range(geo.ow)
    for kw_i in range(geo.kw)
    if 0 <= ow_i * geo.sw - geo.pl + kw_i * geo.dw < geo.w
  )
  return geo.n * geo.c * valid_rows * valid_cols


def im2col_indices(geo: ConvGeometry) -> list[list[int]]:
  """Element indices of the im2col A matrix rows, padding -> zero word.

  Row order is ``(batch, oh, ow)`` and column order is ``(ic, kh, kw)``
  exactly as frozen in P03 numeric_freeze section 5.  Out-of-bounds patch
  positions map to ``x_elements`` -- the explicit +0 word appended to the X
  binding -- so no gather ever reads outside the binding.
  """
  xw = geo.x_elements
  matrix: list[list[int]] = []
  for n_i in range(geo.n):
    for oh_i in range(geo.oh):
      base_h = oh_i * geo.sh - geo.pt
      for ow_i in range(geo.ow):
        base_w = ow_i * geo.sw - geo.pl
        row: list[int] = []
        for c_i in range(geo.c):
          for kh_i in range(geo.kh):
            h_in = base_h + kh_i * geo.dh
            row_ok = 0 <= h_in < geo.h
            for kw_i in range(geo.kw):
              w_in = base_w + kw_i * geo.dw
              if row_ok and 0 <= w_in < geo.w:
                row.append(((n_i * geo.c + c_i) * geo.h + h_in) * geo.w + w_in)
              else:
                row.append(xw)
        matrix.append(row)
  return matrix


def weight_matrix(geo: ConvGeometry, weights: list[int]) -> list[int]:
  """Host-side OIHW -> [K, O] repack (P03 ``b_flat`` order)."""
  if len(weights) != geo.o * geo.k:
    raise ValueError("weight word count does not match the kernel shape")
  b_flat = [0] * (geo.k * geo.o)
  for o_i in range(geo.o):
    for k_index in range(geo.k):
      c_i = k_index // (geo.kh * geo.kw)
      rem = k_index % (geo.kh * geo.kw)
      kh_i = rem // geo.kw
      kw_i = rem % geo.kw
      b_flat[k_index * geo.o + o_i] = weights[
        ((o_i * geo.c + c_i) * geo.kh + kh_i) * geo.kw + kw_i
      ]
  return b_flat


def step_depths(k_total: int, tile_k: int) -> list[int]:
  """Per-step K depths mirroring the P03 tile_k contribution blocks."""
  if tile_k < 1:
    raise ValueError("tile_k must be positive")
  depths: list[int] = []
  remaining = k_total
  while remaining > 0:
    depth = min(tile_k, remaining)
    depths.append(depth)
    remaining -= depth
  return depths


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoftConvCase:
  case_id: str
  source_id: str
  shape: tuple[int, int, int, int]
  kernel: tuple[int, int, int, int]
  stride: tuple[int, int]
  padding: tuple[int, int, int, int]
  dilation: tuple[int, int]
  tile_k: int
  out_dtype: str
  epilogue: dict
  x_words: tuple[int, ...]
  w_words: tuple[int, ...]
  expected: dict

  @property
  def geometry(self) -> ConvGeometry:
    return geometry(self.shape, self.kernel, self.stride, self.padding, self.dilation)

  @property
  def steps(self) -> list[int]:
    return step_depths(self.geometry.k, self.tile_k)


def _hex_words(words) -> tuple[int, ...]:
  return tuple(int(word, 16) for word in words)


def _p03_case(record: dict) -> SoftConvCase:
  source_id = record["id"]
  case_id = source_id.removeprefix("conv-").replace("-", "_")
  data = record["input"]
  expected = record["expected"]
  epilogue = data["epilogue"] or {}
  return SoftConvCase(
    case_id=case_id,
    source_id=source_id,
    shape=tuple(data["shape"]),
    kernel=tuple(data["kernel"]),
    stride=tuple(data["stride"]),
    padding=tuple(data["padding"]),
    dilation=tuple(data["dilation"]),
    tile_k=int(data["tile_k"]),
    out_dtype=epilogue.get("output_dtype", "f32") or "f32",
    epilogue=epilogue,
    x_words=_hex_words(data["x"]),
    w_words=_hex_words(data["weights"]),
    expected=expected,
  )


def p03_cases(vectors_path: Path = VECTORS) -> list[SoftConvCase]:
  document = json.loads(Path(vectors_path).read_text(encoding="utf-8"))
  if document.get("profile") != "bf16_fp32_mac_rne_v1":
    raise ValueError("conv vectors profile mismatch")
  by_id = {record["id"]: record for record in document["cases"]}
  missing = [case_id for case_id in P03_CASE_IDS if case_id not in by_id]
  if missing:
    raise ValueError(f"conv vectors are missing cases: {missing}")
  return [_p03_case(by_id[case_id]) for case_id in P03_CASE_IDS]


def _bf16(value: float) -> int:
  bits = struct.unpack("<I", struct.pack("<f", value))[0]
  return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def pad1_stride1_inputs() -> tuple[list[int], list[int]]:
  """Deterministic non-trivial BF16 inputs for the generated alias case."""
  x_words = [_bf16((((index * 31) % 17) - 8) / 7.0) for index in range(16)]
  w_words = [_bf16((((index * 13) % 19) - 9) / 5.0) for index in range(18)]
  return x_words, w_words


def rebuild_pad1_stride1_golden(path: Path) -> None:
  """Recompute the generated-case golden through the unchanged P03 reference.

  Generator-only path: the reference package produces the golden, never DUT
  bytes.  The X/B/IDX fixture inputs are derived from the committed words
  above, not from the reference.
  """
  import sys

  if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))
  from boa_reference.convolution import conv2d_reference

  x_words, w_words = pad1_stride1_inputs()
  result = conv2d_reference(
    x_words,
    w_words,
    (1, 1, 4, 4),
    (2, 1, 3, 3),
    stride=(1, 1),
    padding=(1, 1, 1, 1),
    dilation=(1, 1),
    tile_k=8,
    epilogue=None,
  )
  document = {
    "schema": 1,
    "id": f"soft-conv-{GENERATED_CASE_ID}",
    "source": "boa_reference.conv2d_reference (P03, generator-invoked)",
    "input": {
      "shape": [1, 1, 4, 4],
      "kernel": [2, 1, 3, 3],
      "stride": [1, 1],
      "padding": [1, 1, 1, 1],
      "dilation": [1, 1],
      "tile_k": 8,
      "x": [f"0x{word:04x}" for word in x_words],
      "weights": [f"0x{word:04x}" for word in w_words],
      "epilogue": None,
    },
    "expected": result,
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pad1_stride1_case(golden_path: Path | None = None) -> SoftConvCase:
  golden = golden_path or (INPUT_ROOT / GENERATED_CASE_ID / "golden.json")
  if not golden.is_file():
    rebuild_pad1_stride1_golden(golden)
  document = json.loads(Path(golden).read_text(encoding="utf-8"))
  data = document["input"]
  return SoftConvCase(
    case_id=GENERATED_CASE_ID,
    source_id=document["id"],
    shape=tuple(data["shape"]),
    kernel=tuple(data["kernel"]),
    stride=tuple(data["stride"]),
    padding=tuple(data["padding"]),
    dilation=tuple(data["dilation"]),
    tile_k=int(data["tile_k"]),
    out_dtype="f32",
    epilogue={},
    x_words=_hex_words(data["x"]),
    w_words=_hex_words(data["weights"]),
    expected=document["expected"],
  )


def all_cases() -> list[SoftConvCase]:
  return [*p03_cases(), pad1_stride1_case()]


# ---------------------------------------------------------------------------
# Fixture input bytes
# ---------------------------------------------------------------------------


def x_binding_bytes(case: SoftConvCase) -> bytes:
  """X NCHW words plus the explicit trailing +0 padding source word."""
  geo = case.geometry
  words = [*case.x_words, ZERO_WORD]
  if len(words) != geo.x_elements + 1:
    raise ValueError(f"{case.source_id}: x word count mismatch")
  return b"".join(word.to_bytes(2, "little") for word in words)


def b_binding_bytes(case: SoftConvCase) -> bytes:
  return b"".join(word.to_bytes(2, "little") for word in weight_matrix(case.geometry, list(case.w_words)))


def idx_binding_bytes(case: SoftConvCase) -> bytes:
  """Step-major i32 gather indices: step s covers A columns [k0, k0+ks)."""
  geo = case.geometry
  matrix = im2col_indices(geo)
  words: list[int] = []
  start = 0
  for depth in case.steps:
    for row in matrix:
      words.extend(row[start : start + depth])
    start += depth
  if len(words) != geo.m * geo.k:
    raise ValueError("im2col index count mismatch")
  return b"".join(word.to_bytes(4, "little") for word in words)


# ---------------------------------------------------------------------------
# Source module builder
# ---------------------------------------------------------------------------


def _dims(values) -> str:
  return "x".join(str(value) for value in values)


def _epilogue_text(case: SoftConvCase) -> str:
  keywords: list[str] = []
  epilogue = case.epilogue
  if epilogue.get("bias") is not None:
    words = ", ".join(f"0x{int(word, 16):08x}" for word in epilogue["bias"])
    keywords.append(f"bias = [{words}]")
  if epilogue.get("scale") is not None:
    keywords.append(f"scale = 0x{int(epilogue['scale'], 16):08x}")
  if epilogue.get("relu"):
    keywords.append("relu")
  if epilogue.get("clip") is not None:
    lo, hi = (int(word, 16) for word in epilogue["clip"])
    keywords.append(f"clip = [0x{lo:08x}, 0x{hi:08x}]")
  if epilogue.get("inv_scale") is not None:
    keywords.append(f"inv_scale = 0x{int(epilogue['inv_scale'], 16):08x}")
  if epilogue.get("zero_point") is not None and case.out_dtype == "i8":
    keywords.append(f"zero_point = {int(epilogue['zero_point'])}")
  return (" " + " ".join(keywords)) if keywords else ""


def arena_contracts(
  case: SoftConvCase, *, debug_a: bool
) -> tuple[tuple[int, tuple[int, ...]], tuple[int, tuple[int, ...]]]:
  """Conservative L1/L2 arena reservations plus allowed profiles."""
  hw = HardwareConfig()
  geo = case.geometry
  out_bytes = geo.m * geo.o * DTYPE_BYTES[case.out_dtype]
  l1_pieces = [(out_bytes, ALIGN)]
  l2_pieces = [(out_bytes, ALIGN), (2 * geo.k * geo.o, ALIGN)]
  cursor = 0
  for depth in case.steps:
    l1_pieces.append((4 * geo.m * depth, ALIGN))
    l1_pieces.append((2 * geo.m * depth, ALIGN))
    l1_pieces.append((2 * depth * geo.o, ALIGN))
    l2_pieces.append((4 * geo.m * depth, ALIGN))
    cursor += 4 * geo.m * depth
  if debug_a:
    l2_pieces.append((2 * geo.m * geo.k, ALIGN))
  return (
    contract_bytes(hw, "l1", l1_pieces),
    contract_bytes(hw, "l2", l2_pieces),
  )


def build_source(case: SoftConvCase, *, debug_a: bool = False) -> str:
  """Emit the soft-conv module text for one case.

  ``debug_a`` adds a per-step ``tile.store.async`` of the gathered A step
  matrix into an extra ``[E]`` bf16 L2/global buffer so a test can capture
  the exact matmul A operand bytes; fixtures are generated without it.
  """
  geo = case.geometry
  m, o, k_total = geo.m, geo.o, geo.k
  steps = case.steps
  x_count = geo.x_elements + 1
  idx_total = m * k_total
  out_dtype = case.out_dtype
  (l1_bytes, l1_profiles), (l2_bytes, l2_profiles) = arena_contracts(case, debug_a=debug_a)

  lines: list[str] = []
  lines.append("builtin.module {")
  program_params = [
    "%task : !nest.task",
    f"%x_g : !nest.global_view<{x_count}xbf16>",
    f"%idx_l2 : !nest.l2_buffer<{idx_total}xi32>",
    f"%b_l2 : !nest.l2_buffer<{_dims((k_total, o))}xbf16>",
  ]
  if debug_a:
    program_params.append(f"%ad_l2 : !nest.l2_buffer<{idx_total}xbf16>")
  program_params.append(f"%c_l2 : !nest.l2_buffer<{_dims((m, o))}x{out_dtype}>")
  lines.append(f"  tile.program @soft_conv({', '.join(program_params)})")
  lines.append(
    f"                resource_contract = #tile.resources<allowed_profiles = "
    f"{list(l1_profiles)},\n"
    f"          tile_l1_spm_bytes_per_context = {l1_bytes}> {{"
  )
  idx_cursor = 0
  for index, depth in enumerate(steps):
    lines.append(
      f"    %idx{index}_v = tile.subview %idx_l2 offsets = [{idx_cursor}] "
      f"sizes = [{m * depth}] strides = [1]\n"
      f"        : !nest.l2_view<{m * depth}xi32>"
    )
    lines.append(
      f"    %b{index}_v = tile.subview %b_l2 offsets = [{sum(steps[:index])}, 0] "
      f"sizes = [{depth}, {o}] strides = [1, 1]\n"
      f"        : !nest.l2_view<{_dims((depth, o))}xbf16>"
    )
    if debug_a:
      lines.append(
        f"    %ad{index}_v = tile.subview %ad_l2 offsets = [{idx_cursor}] "
        f"sizes = [{m * depth}] strides = [1]\n"
        f"        : !nest.l2_view<{m * depth}xbf16>"
      )
    idx_cursor += m * depth
  lines.append(
    f"    %c_v = tile.subview %c_l2 offsets = [0, 0] sizes = [{m}, {o}] "
    f"strides = [1, 1]\n        : !nest.l2_view<{_dims((m, o))}x{out_dtype}>"
  )
  for index, depth in enumerate(steps):
    lines.append(
      f"    %idx{index} = tile.alloc shape = [{m}, {depth}] dtype = \"i32\" "
      f"alignment = {ALIGN} : !tile.l1_buffer<{_dims((m, depth))}xi32>"
    )
    lines.append(
      f"    %a{index} = tile.alloc shape = [{m}, {depth}] dtype = \"bf16\" "
      f"alignment = {ALIGN} : !tile.l1_buffer<{_dims((m, depth))}xbf16>"
    )
    lines.append(
      f"    %b{index} = tile.alloc shape = [{depth}, {o}] dtype = \"bf16\" "
      f"alignment = {ALIGN} : !tile.l1_buffer<{_dims((depth, o))}xbf16>"
    )
  lines.append(
    f"    %out = tile.alloc memory_space = \"l1\" shape = [{m}, {o}] "
    f"dtype = \"{out_dtype}\" alignment = {ALIGN}\n"
    f"        : !tile.l1_buffer<{_dims((m, o))}x{out_dtype}>"
  )
  lines.append(
    f"    %acc = tile.alloc memory_space = \"am\" logical_slot = 0 output(%out)"
    f" shape = [{m}, {o}]\n        dtype = \"f32\" alignment = {ALIGN}"
    f"{_epilogue_text(case)}\n        : !tile.am_buffer<{_dims((m, o))}xf32>"
  )
  load_events: list[str] = []
  for index in range(len(steps)):
    lines.append(
      f"    %li{index} = tile.load.async %idx{index}_v into %idx{index}"
      f" : !tile.event<\"li{index}\">"
    )
    lines.append(
      f"    %lb{index} = tile.load.async %b{index}_v into %b{index}"
      f" : !tile.event<\"lb{index}\">"
    )
    load_events.extend((f"%li{index}", f"%lb{index}"))
  lines.append(f"    tile.await {', '.join(load_events)}")
  lines.append("    tile.signal input_released(%task)")
  for index, depth in enumerate(steps):
    lines.append(
      f"    %g{index} = tile.gather.global.async %x_g indices(%idx{index})"
      f" into %a{index}\n"
      "        map = #tile.indexed_map<index_scale = 1 offset = 0 task_stride = 0"
      " repeat = 1\n                                stride = 0 segment = 1>\n"
      f"        window_entries = {WINDOW_ENTRIES} : !tile.event<\"g{index}\">"
    )
    lines.append(f"    tile.await %g{index}")
    modifiers = []
    if index > 0:
      modifiers.append("accumulate")
    if index == len(steps) - 1:
      modifiers.append("output")
    tail = (" " + " ".join(modifiers)) if modifiers else ""
    lines.append(
      f"    %k{index} = tile.boa.async \"matmul\" ins(%a{index}, %b{index})"
      f" outs(%out) acc(%acc)\n"
      f"        m = {m} n = {o} k = {depth} seq = {index}{tail}"
      f" : !tile.event<\"k{index}\">"
    )
    lines.append(f"    tile.await %k{index}")
    if debug_a:
      lines.append(
        f"    %sa{index} = tile.store.async %a{index} into %ad{index}_v"
        f" : !tile.event<\"sa{index}\">"
      )
      lines.append(f"    tile.await %sa{index}")
    lines.append(f"    tile.free %a{index}")
    lines.append(f"    tile.free %b{index}")
    lines.append(f"    tile.free %idx{index}")
  lines.append("    %stored = tile.store.async %out into %c_v : !tile.event<\"stored\">")
  lines.append("    tile.await %stored")
  lines.append("    tile.signal output_ready(%task)")
  lines.append("    tile.return")
  lines.append("  }")

  context_params = [
    f"%X : !nest.global_memref<{x_count}xbf16>",
    f"%B : !nest.global_memref<{_dims((k_total, o))}xbf16>",
    f"%IDX : !nest.global_memref<{idx_total}xi32>",
  ]
  if debug_a:
    context_params.append(f"%AD : !nest.global_memref<{idx_total}xbf16>")
  context_params.append(f"%C : !nest.global_memref<{_dims((m, o))}x{out_dtype}>")
  lines.append(f"  nest.context @soft_conv_ctx({', '.join(context_params)})")
  lines.append(
    "      placement = 1\n"
    f"                resource_contract = #nest.context_resources<l2_mode = 0,\n"
    f"          allowed_profiles = {list(l2_profiles)}, logical_tasks = 1,"
    f" l2_spm_bytes = {l2_bytes},\n          requested_contexts_per_tile = 1> {{"
  )
  lines.append(
    f"    %x_g_1 = nest.subview %X offsets = [0] sizes = [{x_count}] strides = [1]\n"
    f"        : !nest.global_view<{x_count}xbf16>"
  )
  lines.append(
    f"    %b_blk = nest.subview %B offsets = [0, 0] sizes = [{k_total}, {o}]"
    " strides = [1, 1]\n"
    f"        : !nest.global_view<{_dims((k_total, o))}xbf16>"
  )
  lines.append(
    f"    %idx_blk = nest.subview %IDX offsets = [0] sizes = [{idx_total}]"
    " strides = [1]\n"
    f"        : !nest.global_view<{idx_total}xi32>"
  )
  if debug_a:
    lines.append(
      f"    %ad_blk = nest.subview %AD offsets = [0] sizes = [{idx_total}]"
      " strides = [1]\n"
      f"        : !nest.global_view<{idx_total}xbf16>"
    )
  lines.append(
    f"    %c_blk = nest.subview %C offsets = [0, 0] sizes = [{m}, {o}] "
    "strides = [1, 1]\n"
    f"        : !nest.global_view<{_dims((m, o))}x{out_dtype}>"
  )
  lines.append(
    f"    %idx_l2_1 = nest.alloc slot = \"idx\" role = \"in\" shape = [{idx_total}]"
    " dtype = \"i32\"\n"
    f"        alignment = {ALIGN} : !nest.l2_buffer<{idx_total}xi32>"
  )
  lines.append(
    f"    %b_l2_1 = nest.alloc slot = \"b\" role = \"in\""
    f" shape = [{k_total}, {o}] dtype = \"bf16\"\n"
    f"        alignment = {ALIGN} : !nest.l2_buffer<{_dims((k_total, o))}xbf16>"
  )
  if debug_a:
    lines.append(
      f"    %ad_l2_1 = nest.alloc slot = \"ad\" role = \"out\""
      f" shape = [{idx_total}] dtype = \"bf16\"\n"
      f"        alignment = {ALIGN} : !nest.l2_buffer<{idx_total}xbf16>"
    )
  lines.append(
    f"    %c_l2_1 = nest.alloc slot = \"c\" role = \"out\" shape = [{m}, {o}]"
    f" dtype = \"{out_dtype}\"\n"
    f"        alignment = {ALIGN} : !nest.l2_buffer<{_dims((m, o))}x{out_dtype}>"
  )
  lines.append(
    "    %idx_pref = nest.dma.prefetch.async %idx_blk into %idx_l2_1"
    " : !nest.event<\"idx_pref\">"
  )
  lines.append(
    "    %b_pref = nest.dma.prefetch.async %b_blk into %b_l2_1"
    " : !nest.event<\"b_pref\">"
  )
  lines.append("    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range")
  bindings = "%idx_l2_1, %b_l2_1, %c_l2_1"
  ins = "%idx_l2_1, %b_l2_1"
  outs = "%c_l2_1"
  if debug_a:
    bindings = "%idx_l2_1, %b_l2_1, %ad_l2_1, %c_l2_1"
    outs = "%ad_l2_1, %c_l2_1"
  lines.append(
    "    %grid, %read, %ready = nest.dispatch.tasks.async @soft_conv"
    " l1_mode = 0 context = 0\n"
    f"        tasks(%tasks) globals(%x_g_1)\n"
    f"        bindings({bindings}) ins({ins}) outs({outs})\n"
    "        signal_policy {\n"
    "          input_released = #nest.aggregate<all_tasks>,\n"
    "          output_ready = #nest.aggregate<all_tasks>\n"
    "        }\n"
    "        depends_on(%idx_pref, %b_pref)\n"
    "        : (!nest.event<\"grid\">, !nest.event<\"read\">, !nest.event<\"ready\">)"
  )
  stores = [
    "%c_store = nest.dma.store.async %c_l2_1 into %c_blk depends_on(%ready)\n"
    "        : !nest.event<\"c_store\">"
  ]
  releases = [
    "    nest.release %idx_l2_1 depends_on(%read, %idx_pref)",
    "    nest.release %b_l2_1 depends_on(%read, %b_pref)",
    "    nest.release %c_l2_1 depends_on(%c_store)",
  ]
  awaits = "%grid, %c_store"
  if debug_a:
    stores.insert(
      0,
      "%ad_store = nest.dma.store.async %ad_l2_1 into %ad_blk depends_on(%ready)\n"
      "        : !nest.event<\"ad_store\">",
    )
    releases.insert(2, "    nest.release %ad_l2_1 depends_on(%ad_store)")
    awaits = "%grid, %c_store, %ad_store"
  for store in stores:
    lines.append(f"    {store}")
  for release in releases:
    lines.append(release)
  lines.append(f"    nest.await {awaits}")
  lines.append("    nest.return")
  lines.append("  }")

  entry_params = list(context_params)
  submit_args = "%X, %B, %IDX"
  if debug_a:
    submit_args = "%X, %B, %IDX, %AD"
  submit_args = f"{submit_args}, %C"
  lines.append(f"  nexus.program @soft_conv_entry({', '.join(entry_params)}) {{")
  lines.append(
    f"    %done = nexus.submit_context.async @soft_conv_ctx({submit_args})\n"
    "        : !nexus.event<\"done\">"
  )
  lines.append("    nexus.await %done")
  lines.append("    nexus.return")
  lines.append("  }")
  lines.append("}")
  return "\n".join(lines) + "\n"


def case_header(case: SoftConvCase) -> list[str]:
  geo = case.geometry
  steps = case.steps
  return [
    f"Software im2col Conv (P03 {case.source_id}): X{_dims(case.shape)}xbf16 *"
    f" W{_dims(case.kernel)} -> C[{geo.m},{geo.o}]x{case.out_dtype}.",
    "",
    f"stride {_dims(case.stride)}, padding {_dims(case.padding)},"
    f" dilation {_dims(case.dilation)}, tile_k {case.tile_k};"
    f" K={geo.k} flattened (ic,kh,kw),",
    f"M={geo.m} rows (batch,oh,ow), output [M,O] row-major.  im2col A is",
    "materialised by tile.gather.global.async from the X binding (its trailing",
    "+0 word is the explicit source of every padding element); the typed BOA",
    f"AM chain runs {len(steps)} step(s) of depth {'+'.join(str(d) for d in steps)}",
    "mirroring the P03 tile_k block schedule; B is the host-side [K,O] repack",
    "of the OIHW weights.  No EVU op carries im2col bytes.",
    "",
    "Generated by examples/generators/generate_soft_conv.py.",
    f"Run: bash examples/run.sh soft-conv-{case.case_id.replace('_', '-')}",
  ]


# ---------------------------------------------------------------------------
# Fixture writing
# ---------------------------------------------------------------------------


def write_case_fixture(case: SoftConvCase) -> Path:
  fixture = write_workload(
    WORKLOADS / f"soft_conv_{case.case_id}.mlir",
    case_header(case),
    parse_workload_ir(build_source(case)),
  )
  target = INPUT_ROOT / case.case_id
  target.mkdir(parents=True, exist_ok=True)
  (target / "X.bin").write_bytes(x_binding_bytes(case))
  (target / "B.bin").write_bytes(b_binding_bytes(case))
  (target / "IDX.bin").write_bytes(idx_binding_bytes(case))
  return fixture


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--rebuild-golden",
    action="store_true",
    help="recompute the generated pad1_stride1 golden via the P03 reference",
  )
  args = parser.parse_args()

  if args.rebuild_golden:
    rebuild_pad1_stride1_golden(INPUT_ROOT / GENERATED_CASE_ID / "golden.json")
    print(f"rebuilt {INPUT_ROOT.relative_to(REPO)}/{GENERATED_CASE_ID}/golden.json")

  count = 0
  for case in all_cases():
    fixture = write_case_fixture(case)
    print(f"wrote {fixture.relative_to(REPO)}")
    count += 1
  print(f"wrote {count} fixtures and their inputs under {INPUT_ROOT.relative_to(REPO)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
