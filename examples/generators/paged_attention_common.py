"""Shared geometry, naming and host-runtime helpers for the PagedAttention
decode slice (plan §6).

One module serves three consumers:

- ``generate_paged_attention_decode.py`` authors the two workload IRs and
  the scenario JSON from :class:`Scenario`;
- ``run_paged_attention.py`` builds the launch bindings, seeds the
  ``ByteStore`` and assembles the ``HostEnvironment`` the simulator drives;
- ``test_paged_attention.py`` reuses the same math for micro scenarios and
  the reconciliation assertions.

The KV page layout is fixed: one page holds ``page_tokens`` tokens of
``kv_heads * head_dim`` bf16 elements per kind, laid out as
``[K section | V section | padding]`` with the K section at element 0, the
V section at element ``kv_heads * page_tokens * head_dim`` and
``page_padding_bytes`` of unseeded tail padding.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from pipeline_validator.memory.page_pool import HostPagePoolSpec  # noqa: E402
from pipeline_validator.runtime.host_session import (  # noqa: E402
  HostAllocPages,
  HostEnvironment,
  HostFreePages,
  HostWrite,
)

DTYPE_BYTES = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4}

KV_DTYPE = "bf16"
STATE_DTYPE = "f32"

# Partition count of the pipeline variant (plan §6: block b -> partition
# b % PARTITIONS).  The baseline variant uses partition 0 only, but the
# state/accumulator globals keep the same 4-wide partition dim so both
# variants share one binding layout.
PARTITIONS = 4

POOL = "POOL"
BLOCK_TABLE = "BLOCK_TABLE"
LENGTHS = "LENGTHS"
APPEND_IDS = "APPEND_IDS"
Q_IN = "Q_IN"
K_NEW = "K_NEW"
V_NEW = "V_NEW"
S_INIT = "S_INIT"
O_INIT = "O_INIT"
OUT = "OUT"

# nexus.program global order == run.sh --input-binding order (plan §6).
GLOBAL_ORDER = (POOL, BLOCK_TABLE, LENGTHS, APPEND_IDS, Q_IN, K_NEW, V_NEW, S_INIT, O_INIT, OUT)
WRITE_ONLY_GLOBALS = (OUT,)

POOL_NAME = "kv_pages"
FIRST_BINDING_IOVA = 0x1000000
BINDING_ALIGN = 2 * 1024 * 1024


def align_up(value: int, alignment: int) -> int:
  return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Scenario:
  """Frozen PagedAttention scenario description (plan §6 defaults)."""

  num_requests: int = 3
  initial_lengths: tuple[int, ...] = (255, 511, 767)
  steps: int = 4
  page_tokens: int = 16
  kv_heads: int = 4
  heads_per_kv: int = 4
  head_dim: int = 64
  physical_pages: int = 128
  page_padding_bytes: int = 64
  kv_dtype: str = KV_DTYPE
  state_dtype: str = STATE_DTYPE
  placement: int = 15
  contexts_per_tile: int = 4
  l1_mode: int = 1
  l2_mode: int = 1
  group_policy: str = "s1"
  num_dma_channels: int = 2
  hbm_fixed_latency_cycles: int = 10
  max_cycles: int = 2_000_000
  seed: int = 0
  initial_mapping: tuple[tuple[int, ...], ...] | None = None
  variant: str = "pipeline"

  def __post_init__(self) -> None:
    for name in (
      "num_requests",
      "steps",
      "page_tokens",
      "kv_heads",
      "heads_per_kv",
      "head_dim",
      "physical_pages",
    ):
      value = getattr(self, name)
      if type(value) is not int or value <= 0:
        raise ValueError(f"paged attention {name} must be a positive integer")
    for name in (
      "page_padding_bytes",
      "contexts_per_tile",
      "num_dma_channels",
      "hbm_fixed_latency_cycles",
      "max_cycles",
      "seed",
    ):
      value = getattr(self, name)
      if type(value) is not int or value < 0:
        raise ValueError(f"paged attention {name} must be a non-negative integer")
    if len(self.initial_lengths) != self.num_requests:
      raise ValueError("one initial length per request is required")
    for length in self.initial_lengths:
      if type(length) is not int or length < 0:
        raise ValueError("paged attention initial lengths must be non-negative")
    if self.variant not in ("pipeline", "baseline"):
      raise ValueError("scenario variant must be 'pipeline' or 'baseline'")
    if self.kv_dtype not in DTYPE_BYTES or self.state_dtype not in DTYPE_BYTES:
      raise ValueError("unsupported paged attention dtype")
    if self.initial_mapping is not None:
      if len(self.initial_mapping) != self.num_requests:
        raise ValueError("initial mapping needs one page tuple per request")
      seen: set[int] = set()
      for request, pages in enumerate(self.initial_mapping):
        for page in pages:
          if type(page) is not int or not 0 <= page < self.physical_pages:
            raise ValueError(f"initial mapping page {page!r} is out of range")
          if page in seen:
            raise ValueError("initial mapping repeats a page")
          seen.add(page)
        if len(pages) != self.initial_page_count(request):
          raise ValueError(
            f"request {request} needs {self.initial_page_count(request)} initial pages,"
            f" mapping provides {len(pages)}"
          )

  # -- derived geometry --------------------------------------------------

  @property
  def kv_element(self) -> int:
    return self.kv_heads * self.head_dim

  @property
  def page_stride_elements(self) -> int:
    """KV-dtype-element stride between pages (K + V sections + padding)."""
    padding = self.page_padding_bytes // DTYPE_BYTES[self.kv_dtype]
    if self.page_padding_bytes % DTYPE_BYTES[self.kv_dtype]:
      raise ValueError("page padding must be a whole number of KV elements")
    return 2 * self.kv_element * self.page_tokens + padding

  @property
  def page_stride_bytes(self) -> int:
    return self.page_stride_elements * DTYPE_BYTES[self.kv_dtype]

  @property
  def max_pages(self) -> int:
    final = max(self.final_lengths)
    return (final + self.page_tokens - 1) // self.page_tokens

  @property
  def final_lengths(self) -> tuple[int, ...]:
    return tuple(length + self.steps for length in self.initial_lengths)

  def initial_page_count(self, request: int) -> int:
    return (self.initial_lengths[request] + self.page_tokens - 1) // self.page_tokens

  def mapping(self) -> tuple[tuple[int, ...], ...]:
    """Initial page mapping: seeded shuffle, or the explicit override."""
    if self.initial_mapping is not None:
      return self.initial_mapping
    rng = random.Random(self.seed)
    pages = list(range(self.physical_pages))
    rng.shuffle(pages)
    mapping: list[tuple[int, ...]] = []
    cursor = 0
    for request in range(self.num_requests):
      count = self.initial_page_count(request)
      mapping.append(tuple(pages[cursor : cursor + count]))
      cursor += count
    return tuple(mapping)

  def scope(self, request: int) -> str:
    return f"owner_{request}"

  def scopes(self) -> tuple[str, ...]:
    return tuple(self.scope(request) for request in range(self.num_requests))

  def total_tokens(self, request: int) -> int:
    """Token count the request's page table spans (initial + appended)."""
    return self.initial_lengths[request] + self.steps

  # -- per-(request, step) append and attention math ----------------------

  def token(self, request: int, step: int) -> int:
    return self.initial_lengths[request] + step

  def token_in_page(self, request: int, step: int) -> int:
    return self.token(request, step) % self.page_tokens

  def opens_page(self, request: int, step: int) -> bool:
    return self.token_in_page(request, step) == 0

  def block_count(self, request: int, step: int) -> int:
    """Attention blocks after appending the step's token (plan §Verification 8)."""
    length = self.token(request, step) + 1
    return (length + self.page_tokens - 1) // self.page_tokens

  def block_tokens(self, request: int, step: int, block: int) -> int:
    length = self.token(request, step) + 1
    return min(self.page_tokens, length - block * self.page_tokens)

  def total_block_dispatches(self) -> int:
    """Total attention block dispatches across all (request, step) pairs."""
    return sum(
      self.block_count(r, s) for r in range(self.num_requests) for s in range(self.steps)
    )

  def used_table_entries(self, request: int) -> int:
    return (self.total_tokens(request) + self.page_tokens - 1) // self.page_tokens

  # -- global shapes ------------------------------------------------------

  def global_shapes(self) -> dict[str, tuple[int, ...]]:
    r, s = self.num_requests, self.steps
    k, h, d = self.kv_heads, self.heads_per_kv, self.head_dim
    return {
      POOL: (self.physical_pages, self.page_stride_elements),
      BLOCK_TABLE: (r * self.max_pages,),
      LENGTHS: (r,),
      APPEND_IDS: (r * s,),
      Q_IN: (r, s, k, h, d),
      K_NEW: (r, s, k, 1, d),
      V_NEW: (r, s, k, 1, d),
      S_INIT: (r, s, PARTITIONS, k, h, 2),
      O_INIT: (r, s, PARTITIONS, k, h, d),
      OUT: (r, s, k, h, d),
    }

  def global_dtypes(self) -> dict[str, str]:
    return {
      POOL: self.kv_dtype,
      BLOCK_TABLE: "i32",
      LENGTHS: "i32",
      APPEND_IDS: "i32",
      Q_IN: self.kv_dtype,
      K_NEW: self.kv_dtype,
      V_NEW: self.kv_dtype,
      S_INIT: self.state_dtype,
      O_INIT: self.state_dtype,
      OUT: self.state_dtype,
    }

  def global_sizes(self) -> dict[str, int]:
    shapes = self.global_shapes()
    dtypes = self.global_dtypes()
    return {name: _product(shape) * DTYPE_BYTES[dtypes[name]] for name, shape in shapes.items()}

  def bindings(self) -> dict[str, tuple[int, int, str]]:
    """``{name: (base_iova, size_bytes, permissions)}`` starting at
    ``FIRST_BINDING_IOVA`` with ``BINDING_ALIGN`` gaps (plan §6: the first
    four bindings are rw, Q_IN..O_INIT read-only, OUT write-only)."""
    result: dict[str, tuple[int, int, str]] = {}
    cursor = FIRST_BINDING_IOVA
    for name in GLOBAL_ORDER:
      size = self.global_sizes()[name]
      if name in WRITE_ONLY_GLOBALS:
        permission = "w"
      elif name == POOL or name in (BLOCK_TABLE, LENGTHS, APPEND_IDS):
        permission = "rw"
      else:
        permission = "r"
      result[name] = (cursor, size, permission)
      cursor = align_up(cursor + size, BINDING_ALIGN)
    return result

  # -- naming -------------------------------------------------------------

  def context_name(self, request: int, step: int) -> str:
    return f"step_r{request}_s{step}"

  def append_program(self, request: int, tip: int) -> str:
    return f"paged_attention_append_r{request}_tip{tip}"

  def attention_program(self, request: int, tokens: int, final: bool) -> str:
    return f"paged_attention_t{tokens}_{'final' if final else 'step'}_r{request}"

  def merge_program(self, partitions: int = PARTITIONS) -> str:
    return f"paged_attention_merge_p{partitions}"

  def prepare_routine(self, request: int, step: int) -> str:
    return f"prepare_r{request}_s{step}"

  def commit_routine(self, request: int, step: int) -> str:
    return f"commit_r{request}_s{step}"

  def release_routine(self, request: int) -> str:
    return f"release_r{request}"

  # -- host software tables -----------------------------------------------

  def block_table_entries(self, request: int) -> list[int]:
    """Full BLOCK_TABLE row for one request: mapped pages, then -1 filler."""
    entries = [-1] * self.max_pages
    for index, page in enumerate(self.mapping()[request]):
      entries[index] = page
    return entries

  def used_block_entries(self, request: int) -> list[int]:
    """BLOCK_TABLE slots the request's page table actually fills."""
    return list(range(self.used_table_entries(request)))


def _product(shape: tuple[int, ...]) -> int:
  total = 1
  for dim in shape:
    total *= dim
  return total


# ---------------------------------------------------------------------------
# KV byte seeding (plan §6): valid KV byte
# (17*r + 31*token + 7*head + byte + kind_bias) % 251, K bias 0, V bias 113.
# ---------------------------------------------------------------------------

K_BIAS = 0
V_BIAS = 113
MODULUS = 251


def kv_byte(request: int, token: int, head: int, byte: int, kind_bias: int) -> int:
  return (17 * request + 31 * token + 7 * head + byte + kind_bias) % MODULUS


def seed_pool(scenario: Scenario) -> bytes:
  """POOL bytes: initially filled tokens only; free pages, the un-appended
  tail token slots of partially filled pages and the page padding stay
  unwritten (plan §6)."""
  stride = scenario.page_stride_elements
  kv_section = scenario.kv_element * scenario.page_tokens
  elem = DTYPE_BYTES[scenario.kv_dtype]
  pool = bytearray(scenario.physical_pages * stride * elem)
  for request in range(scenario.num_requests):
    pages = scenario.mapping()[request]
    length = scenario.initial_lengths[request]
    for index, page in enumerate(pages):
      base = page * stride * elem
      low = index * scenario.page_tokens
      high = min(length, (index + 1) * scenario.page_tokens)
      for token in range(low, high):
        local = token - low
        for head in range(scenario.kv_heads):
          row = (head * scenario.page_tokens + local) * scenario.head_dim * elem
          for byte in range(scenario.head_dim * elem):
            pool[base + row + byte] = kv_byte(request, token, head, byte, K_BIAS)
          vrow = kv_section * elem + row
          for byte in range(scenario.head_dim * elem):
            pool[base + vrow + byte] = kv_byte(request, token, head, byte, V_BIAS)
  return bytes(pool)


def seed_new_token(scenario: Scenario, request: int, token: int, kind_bias: int) -> bytes:
  """One K or V ``[kv_heads, 1, head_dim]`` row of an appended token."""
  elem = DTYPE_BYTES[scenario.kv_dtype]
  row = bytearray()
  for head in range(scenario.kv_heads):
    for byte in range(scenario.head_dim * elem):
      row.append(kv_byte(request, token, head, byte, kind_bias))
  return bytes(row)


def seed_globals(scenario: Scenario) -> dict[str, bytes]:
  """Seeded binding payloads.  APPEND_IDS, free pages, page padding and the
  un-appended tail of partially filled pages stay unseeded (plan §6)."""
  payloads: dict[str, bytes] = {POOL: seed_pool(scenario)}
  table: list[int] = []
  for request in range(scenario.num_requests):
    table.extend(scenario.block_table_entries(request))
  payloads[BLOCK_TABLE] = struct.pack(f"<{len(table)}i", *table)
  payloads[LENGTHS] = struct.pack(f"<{scenario.num_requests}i", *scenario.initial_lengths)
  payloads[Q_IN] = bytes(
    (11 * r + 5 * s + 3 * head + byte) % MODULUS
    for r in range(scenario.num_requests)
    for s in range(scenario.steps)
    for head in range(scenario.kv_heads * scenario.heads_per_kv)
    for byte in range(scenario.head_dim * DTYPE_BYTES[scenario.kv_dtype])
  )
  k_rows: list[bytes] = []
  v_rows: list[bytes] = []
  for r in range(scenario.num_requests):
    for s in range(scenario.steps):
      token = scenario.token(r, s)
      k_rows.append(seed_new_token(scenario, r, token, K_BIAS))
      v_rows.append(seed_new_token(scenario, r, token, V_BIAS))
  payloads[K_NEW] = b"".join(k_rows)
  payloads[V_NEW] = b"".join(v_rows)
  # Online-softmax start state: m = -inf, l = 0, accumulator = 0.
  start = struct.pack("<f", float("-inf")) + struct.pack("<f", 0.0)
  payloads[S_INIT] = start * (_product(scenario.global_shapes()[S_INIT]) // 2)
  payloads[O_INIT] = bytes(_product(scenario.global_shapes()[O_INIT]) * DTYPE_BYTES[scenario.state_dtype])
  return payloads


# ---------------------------------------------------------------------------
# Host environment (plan §6 routines)
# ---------------------------------------------------------------------------


@dataclass
class HostRunState:
  """Mutable per-request host software state shared by all routines."""

  scenario: Scenario
  page_table: list[list[int]] = field(default_factory=list)
  append_ids: list[list[int]] = field(default_factory=list)

  def __post_init__(self) -> None:
    for request in range(self.scenario.num_requests):
      self.page_table.append(list(self.scenario.mapping()[request]))
      self.append_ids.append([])


def make_host_environment(scenario: Scenario) -> tuple[HostEnvironment, HostRunState]:
  """Handlers + page pool the runner passes to ``Simulator.run(host=...)``.

  Routine contract (plan §6):
  - prepare: allocate one page when the token opens a page, then write the
    page id into ``APPEND_IDS[r*steps+s]`` (4 B, scope ``owner_r``);
  - commit: when the step opened a page, write the new BLOCK_TABLE entry;
    then write the new LENGTHS value (scopes are empty);
  - release: write -1 over every used BLOCK_TABLE entry and LENGTHS, then
    ``HostFreePages`` for the request's scope.
  """
  state = HostRunState(scenario)
  spec = HostPagePoolSpec(
    name=POOL_NAME,
    binding=POOL,
    page_bytes=scenario.page_stride_bytes,
    page_count=scenario.physical_pages,
    initial_owners={scenario.scope(r): scenario.mapping()[r] for r in range(scenario.num_requests)},
  )
  handlers: dict[str, object] = {}
  for r in range(scenario.num_requests):
    for s in range(scenario.steps):
      handlers[scenario.prepare_routine(r, s)] = _prepare_handler(state, r, s)
      handlers[scenario.commit_routine(r, s)] = _commit_handler(state, r, s)
    handlers[scenario.release_routine(r)] = _release_handler(state, r)
  return HostEnvironment(handlers=handlers, pools=(spec,)), state


def _prepare_handler(state: HostRunState, r: int, s: int):
  scenario = state.scenario
  scope = scenario.scope(r)

  def handler(_request):
    if scenario.opens_page(r, s):
      allocation = yield HostAllocPages(POOL_NAME, scope, 1)
      page = allocation.pages[0]
      state.page_table[r].append(page)
    else:
      page = state.page_table[r][-1]
    state.append_ids[r].append(page)
    yield HostWrite(
      APPEND_IDS,
      (r * scenario.steps + s) * 4,
      struct.pack("<i", page),
      scope=scope,
    )

  return handler


def _commit_handler(state: HostRunState, r: int, s: int):
  scenario = state.scenario

  def handler(_request):
    token = scenario.token(r, s)
    if scenario.opens_page(r, s):
      block = token // scenario.page_tokens
      yield HostWrite(BLOCK_TABLE, block * 4, struct.pack("<i", state.page_table[r][block]))
    yield HostWrite(LENGTHS, r * 4, struct.pack("<i", token + 1))

  return handler


def _release_handler(state: HostRunState, r: int):
  scenario = state.scenario

  def handler(_request):
    used = scenario.used_table_entries(r)
    yield HostWrite(BLOCK_TABLE, 0, struct.pack(f"<{used}i", *([-1] * used)))
    yield HostWrite(LENGTHS, r * 4, struct.pack("<i", -1))
    yield HostFreePages(POOL_NAME, scenario.scope(r))

  return handler


# ---------------------------------------------------------------------------
# Scenario JSON (schema_version=1, plan §6)
# ---------------------------------------------------------------------------


def scenario_to_dict(scenario: Scenario, source_hashes: dict[str, str]) -> dict:
  return {
    "schema_version": 1,
    "initial_lengths": list(scenario.initial_lengths),
    "steps": scenario.steps,
    "page_tokens": scenario.page_tokens,
    "physical_pages": scenario.physical_pages,
    "page_padding_bytes": scenario.page_padding_bytes,
    "contexts_per_tile": scenario.contexts_per_tile,
    "l1_mode": scenario.l1_mode,
    "l2_mode": scenario.l2_mode,
    "seed": scenario.seed,
    "initial_mapping": {
      scenario.scope(r): list(pages) for r, pages in enumerate(scenario.mapping())
    },
    "source_hashes": dict(source_hashes),
    "num_requests": scenario.num_requests,
    "kv_heads": scenario.kv_heads,
    "heads_per_kv": scenario.heads_per_kv,
    "head_dim": scenario.head_dim,
    "kv_dtype": scenario.kv_dtype,
    "state_dtype": scenario.state_dtype,
    "placement": scenario.placement,
    "group_policy": scenario.group_policy,
    "num_dma_channels": scenario.num_dma_channels,
    "hbm_fixed_latency_cycles": scenario.hbm_fixed_latency_cycles,
    "max_cycles": scenario.max_cycles,
  }


def scenario_from_dict(data: dict) -> Scenario:
  if data.get("schema_version") != 1:
    raise ValueError(f"unsupported scenario schema_version {data.get('schema_version')!r}")
  mapping = data.get("initial_mapping") or {}
  mapping_tuple = tuple(
    tuple(int(page) for page in mapping[f"owner_{r}"]) for r in range(int(data["num_requests"]))
  )
  return Scenario(
    num_requests=int(data["num_requests"]),
    initial_lengths=tuple(int(v) for v in data["initial_lengths"]),
    steps=int(data["steps"]),
    page_tokens=int(data["page_tokens"]),
    kv_heads=int(data["kv_heads"]),
    heads_per_kv=int(data["heads_per_kv"]),
    head_dim=int(data["head_dim"]),
    physical_pages=int(data["physical_pages"]),
    page_padding_bytes=int(data["page_padding_bytes"]),
    kv_dtype=str(data["kv_dtype"]),
    state_dtype=str(data["state_dtype"]),
    placement=int(data["placement"]),
    contexts_per_tile=int(data["contexts_per_tile"]),
    l1_mode=int(data["l1_mode"]),
    l2_mode=int(data["l2_mode"]),
    group_policy=str(data["group_policy"]),
    num_dma_channels=int(data["num_dma_channels"]),
    hbm_fixed_latency_cycles=int(data["hbm_fixed_latency_cycles"]),
    max_cycles=int(data["max_cycles"]),
    seed=int(data["seed"]),
    initial_mapping=mapping_tuple,
  )


def workload_path(scenario_path: Path, variant: str) -> Path:
  """The workload IR file belonging to one variant of a scenario."""
  return Path(scenario_path).parent / f"paged_attention_decode_{variant}.mlir"


def load_scenario(scenario_path: Path, variant: str) -> Scenario:
  """Load one variant's scenario, verifying the recorded source hash first
  (plan §6: replay checks the scenario source_hash)."""
  data = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
  if data.get("schema_version") != 1:
    raise ValueError(f"unsupported scenario schema_version {data.get('schema_version')!r}")
  hashes = data.get("source_hashes") or {}
  if variant not in hashes:
    raise ValueError(f"scenario source_hashes has no entry for variant '{variant}'")
  ir_path = workload_path(scenario_path, variant)
  digest = hashlib.sha256(ir_path.read_bytes()).hexdigest()
  if digest != hashes[variant]:
    raise ValueError(
      f"scenario source_hash mismatch for '{variant}': {ir_path} hashes to {digest[:16]}...,"
      f" scenario records {hashes[variant][:16]}... (regenerate the workloads)"
    )
  return scenario_from_dict(data)
