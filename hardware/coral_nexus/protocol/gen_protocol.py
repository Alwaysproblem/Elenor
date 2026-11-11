"""Generate the checked boundary-protocol artifacts (C017 / W1-01).

Input : protocol/wire_schema_v1.yaml (the one logical schema) merged with cn_protocol_v1.yaml, the field
        requests of W2..W6, csr_name_map.yaml, upstream_csr_occupied.json and the lifetime/budget files.
Output: protocol/generated/{cn_layout.json,cn_messages.h,cn_offsets.svh} and
        pipeline_validator/cosim/wire.py.

Every artifact is rendered in memory, validated, and only then written (atomic per file), so a failing run
never touches an earlier good output. Output is byte-deterministic: all ordering comes from the catalog and
the schema, never from dict or directory iteration.

A field is accepted only with a provenance: an identity domain it belongs to, an enum/bitset whose members
are read back from the validator or contract they mirror, a `file::needle` source read back from disk, or an
explicit `transport_only` note that is repeated in the report. No address, opcode or silicon width is
produced here; the 64-bit word is a transport container (see wire_schema_v1.yaml).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pprint
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

H = Path(__file__).resolve().parents[1]
REPO = H.parents[1]
DESIGN = REPO / "design/coral_nexus_tile"
sys.path.insert(0, str(H / "scripts/lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import csr_allocator  # noqa: E402
import derive_widths  # noqa: E402
import validate_protocol as vp  # noqa: E402

PROTOCOL_DIR = H / "protocol"
GENERATED_DIR = PROTOCOL_DIR / "generated"
WIRE_PY = REPO / "pipeline_validator/cosim/wire.py"
SCHEMA_FILE = "wire_schema_v1.yaml"
REQUEST_FILES = ("w2.yaml", "w3.yaml", "w4.yaml", "w5.yaml", "w6.yaml")
DIRS = ("G2T", "T2G", "H2G")
KINDS = ("uint", "enum", "bool", "bitset", "digest")
_BANNED_FIELD = re.compile(r"(^|_)(addr|address|opcode|pc)($|_)")
_ONE_WORD_KINDS = {"uint", "enum", "bool", "bitset"}
_INPUTS = (
  "cn_protocol_v1.yaml",
  SCHEMA_FILE,
  "csr_name_map.yaml",
  "upstream_csr_occupied.json",
  "channel_budgets.yaml",
  "lifetime_bounds.yaml",
  *(f"field_requests/{n}" for n in REQUEST_FILES),
)


class GenError(Exception):
  def __init__(self, errors: list[str]) -> None:
    super().__init__("\n".join(errors))
    self.errors = errors


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _yaml(path: Path) -> Any:
  return yaml.safe_load(path.read_text())


# ---------------------------------------------------------------------------------------------------
# source read-back
# ---------------------------------------------------------------------------------------------------
def _resolve(rel: str) -> Path | None:
  for base in (H, DESIGN, REPO):
    candidate = base / rel
    if candidate.is_file():
      return candidate
  return None


def _read(rel: str) -> str | None:
  path = _resolve(rel)
  return path.read_text() if path else None


def _dig(doc: Any, dotted: str) -> Any:
  for part in dotted.split("."):
    doc = doc[part]
  return doc


def _py_enum(file: str, cls: str, mode: str) -> list[str]:
  text = _read(file)
  if text is None:
    raise KeyError(f"{file} not found")
  for node in ast.walk(ast.parse(text)):
    if isinstance(node, ast.ClassDef) and node.name == cls:
      out: list[str] = []
      for stmt in node.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
          if mode == "names":
            out.append(stmt.targets[0].id)
          elif isinstance(stmt.value, ast.Constant):
            out.append(str(stmt.value.value))
      return out
  raise KeyError(f"class {cls} not in {file}")


def _enum_members(name: str, spec: dict[str, Any], errors: list[str]) -> list[str]:
  members = [str(m) for m in spec["members"]]
  src = spec.get("source") or {}
  kind = src.get("kind")
  try:
    if kind == "yaml_list":
      actual = [str(x) for x in _dig(_yaml(_resolve(src["file"]) or Path("/missing")), src["path"])]
    elif kind == "yaml_names":
      actual = [str(x["name"]) for x in _dig(_yaml(_resolve(src["file"]) or Path("/missing")), src["path"])]
    elif kind == "py_enum":
      actual = _py_enum(src["file"], src["class"], src["mode"])
    elif kind == "doc_tokens":
      text = _read(src["file"])
      if text is None:
        raise KeyError(f"{src['file']} not found")
      if src.get("ignore_case"):
        text = text.lower()
      missing = [m for m in members if (m.lower() if src.get("ignore_case") else m) not in text]
      if missing:
        errors.append(f"enum {name}: members {missing} do not occur in {src['file']}")
      return members
    else:
      errors.append(f"enum {name}: unknown source kind {kind!r}")
      return members
  except (KeyError, TypeError, OSError, yaml.YAMLError, SyntaxError) as exc:
    errors.append(f"enum {name}: cannot read source {src.get('file')}: {exc}")
    return members
  if actual != members:
    errors.append(f"enum {name}: members {members} differ from {src.get('file')} {actual}")
  return members


def _check_src(label: str, spec: str, errors: list[str]) -> None:
  rel, sep, needle = spec.partition("::")
  if not sep or not needle:
    errors.append(f"{label}: src {spec!r} is not file::needle")
    return
  text = _read(rel)
  if text is None:
    errors.append(f"{label}: src file {rel} does not exist")
  elif needle not in text:
    errors.append(f"{label}: needle {needle!r} not found in {rel}")


# ---------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------
def _fields_of(name: str, body: dict[str, Any], schema: dict[str, Any], errors: list[str]) -> list[dict]:
  fields: list[dict] = []
  for group in body.get("include") or []:
    if group not in schema["groups"]:
      errors.append(f"message {name}: unknown group {group!r}")
      continue
    fields.extend(dict(f, _group=group) for f in schema["groups"][group])
  fields.extend(dict(f) for f in body.get("fields") or [])
  return fields


def _build(data: dict[str, Any], errors: list[str]) -> dict[str, Any]:
  protocol, schema = data["protocol"], data["schema"]
  catalog = {m["name"]: m for m in protocol["messages"]}
  catalog_order = [m["name"] for m in protocol["messages"]]
  new_by_name = {m["name"]: m for m in schema.get("new_messages") or []}
  channels = [c["name"] for c in protocol["channels"]]
  domains = {n: {f["name"]: f for f in d["fields"]} for n, d in protocol["identity_domains"].items()}
  domains["version_ticket"] = {str(n): {"name": str(n)} for n in protocol["version_ticket"]["fields"]}
  hidden = {
    f["name"] for d in domains.values() for f in d.values() if f.get("field_visibility") == "hidden"
  }

  order = [n for n in catalog_order if n in schema["messages"]]
  order += [m["name"] for m in schema.get("new_messages") or [] if m["name"] in schema["messages"]]
  for n in schema["messages"]:
    if n not in order:
      errors.append(f"message {n}: in the schema but neither in the C009 catalog nor in new_messages")
  for n in catalog_order:
    if n not in schema["messages"]:
      errors.append(f"message {n}: in the C009 catalog but missing from the schema")
  for n in new_by_name:
    if n in catalog:
      errors.append(f"message {n}: listed as new but already in the catalog")

  enums: dict[str, list[str]] = {}
  for ename, spec in schema["enums"].items():
    enums[ename] = _enum_members(ename, spec, errors)
  enums["msg_type"] = list(order)
  bitsets: dict[str, list[str]] = {}
  for bname, spec in schema["bitsets"].items():
    bitsets[bname] = _enum_members(bname, spec, errors)
    if len(bitsets[bname]) > 64:
      errors.append(f"bitset {bname}: more than 64 members cannot ride one word")

  key_aliases = schema["key_aliases"]
  request_aliases = schema["request_aliases"]
  stage_fields = protocol["dedup"].get("stage_fields", {})
  establishes = set((schema.get("stale_generation") or {}).get("establishes_on") or [])
  messages: list[dict[str, Any]] = []
  transport_only: list[dict[str, str]] = []
  for mid, name in enumerate(order):
    body = schema["messages"][name]
    cat = catalog.get(name) or new_by_name.get(name) or {}
    fields = _fields_of(name, body, schema, errors)
    seen: set[str] = set()
    offset = 0
    out_fields: list[dict[str, Any]] = []
    for f in fields:
      fname, kind = f.get("name"), f.get("kind")
      where = f"message {name}.{fname}"
      if not isinstance(fname, str) or kind not in KINDS:
        errors.append(f"{where}: needs a name and a kind in {KINDS}")
        continue
      if fname in seen:
        errors.append(f"{where}: field appears twice")
        continue
      seen.add(fname)
      if _BANNED_FIELD.search(fname):
        errors.append(f"{where}: an address/opcode/PC style field is not allowed on the wire")
      if fname in hidden:
        errors.append(f"{where}: {fname} is field_visibility hidden and must not be on the wire")
      proofs = [k for k in ("ident", "enum", "bitset", "src", "transport_only") if k in f]
      if len(proofs) != 1:
        errors.append(
          f"{where}: needs exactly one provenance of ident/enum/bitset/src/transport_only, has {proofs}"
        )
      if "ident" in f and fname not in domains.get(f["ident"], {}):
        errors.append(f"{where}: not a member of identity domain {f['ident']!r}")
      if "src" in f:
        _check_src(where, f["src"], errors)
      if "transport_only" in f:
        transport_only.append({"message": name, "field": fname, "note": str(f["transport_only"])})
      limit = 0
      if kind == "enum":
        if f.get("enum") not in enums:
          errors.append(f"{where}: unknown enum {f.get('enum')!r}")
        else:
          limit = len(enums[f["enum"]]) + (1 if f.get("optional") else 0)
      elif kind == "bitset":
        if f.get("bitset") not in bitsets:
          errors.append(f"{where}: unknown bitset {f.get('bitset')!r}")
        else:
          limit = (1 << len(bitsets[f["bitset"]])) - 1
      elif kind == "bool":
        limit = 1
      if kind == "digest" and f.get("words") != 4:
        errors.append(f"{where}: a digest is 4 words (sha256)")
      words = 4 if kind == "digest" else 1
      out_fields.append(
        {
          "name": fname,
          "kind": kind,
          "offset": offset,
          "words": words,
          "limit": limit,
          "enum": f.get("enum") or f.get("bitset"),
          "optional": bool(f.get("optional")),
        }
      )
      offset += words
    idx = {f["name"]: f for f in out_fields}
    conditions: list[dict[str, Any]] = []
    for c in schema.get("conditions") or []:
      if c["message"] != name:
        continue
      other_members = enums.get(idx.get(c["other"], {}).get("enum") or "", [])
      if c["field"] not in idx or c["other"] not in idx or c["other_is"] not in other_members:
        errors.append(f"condition on {name}: field/other/other_is do not resolve")
        continue
      conditions.append(
        {
          "kind": "nonzero_iff",
          "a": idx[c["field"]]["offset"],
          "b": idx[c["other"]]["offset"],
          "value": other_members.index(c["other_is"]),
        }
      )
    for domain in protocol["identity_domains"].values():
      for df in domain["fields"]:
        other = df.get("equals")
        if other and df["name"] in idx and other in idx:
          conditions.append(
            {"kind": "equal", "a": idx[df["name"]]["offset"], "b": idx[other]["offset"], "value": 0}
          )
    stage = stage_fields.get(name)
    if stage is not None and stage not in idx:
      errors.append(f"message {name}: dedup stage field {stage!r} is not on the wire")
    if "txn_id" in idx and name in stage_fields and stage_fields[name] == "txn_id":
      errors.append(f"message {name}: stage field must differ from txn_id")
    if cat.get("channel") not in channels:
      errors.append(f"message {name}: channel {cat.get('channel')!r} is not a catalog channel")
    cdir = cat.get("dir")
    if cdir not in DIRS:
      errors.append(f"message {name}: direction {cdir!r} not in {DIRS}")
    host_only = bool(body.get("host_only"))
    if host_only != bool(cat.get("host_only", False)) and name in new_by_name:
      errors.append(f"message {name}: host_only differs between the schema body and new_messages")
    # catalog key fields must be reachable on the wire
    for key in cat.get("key") or []:
      if key in idx or key in key_aliases or key in (body.get("include") or []):
        continue
      errors.append(f"message {name}: catalog key {key!r} is not carried on the wire")
    for key in cat.get("key") or []:
      alias = key_aliases.get(key)
      if alias:
        for need in alias["fields"]:
          if need not in idx:
            errors.append(f"message {name}: key {key!r} needs wire field {need!r}")
    messages.append(
      {
        "name": name,
        "id": mid,
        "channel": cat.get("channel"),
        "channel_index": channels.index(cat["channel"]) if cat.get("channel") in channels else -1,
        "dir": cdir,
        "dir_index": DIRS.index(cdir) if cdir in DIRS else -1,
        "owner": cat.get("owner"),
        "host_only": host_only,
        "payload_words": offset,
        "fields": out_fields,
        "conditions": conditions,
        "txn": idx["txn_id"]["offset"] if "txn_id" in idx else -1,
        "ctx_slot": idx["ctx_slot"]["offset"] if "ctx_slot" in idx else -1,
        "ctx_gen": idx["ctx_generation"]["offset"] if "ctx_generation" in idx else -1,
        "stage": idx[stage]["offset"] if stage in idx else -1,
        "ack_for": idx["ack_for_message_type"]["offset"] if "ack_for_message_type" in idx else -1,
        "establishes": name in establishes,
      }
    )
  for n in establishes:
    if n not in order:
      errors.append(f"stale_generation.establishes_on: {n} is not a message")
  return {
    "messages": messages,
    "enums": enums,
    "bitsets": bitsets,
    "channels": channels,
    "transport_only": transport_only,
    "key_aliases": key_aliases,
    "request_aliases": request_aliases,
  }


def _check_requests(
  model: dict[str, Any], data: dict[str, Any], csr_names: set[str], errors: list[str]
) -> None:
  by_name = {m["name"]: m for m in model["messages"]}
  aliases = model["request_aliases"]
  all_fields = {f["name"] for m in model["messages"] for f in m["fields"]}
  asked_new: dict[str, set[str]] = {}

  def need_field(label: str, msg: str | None, name: str) -> None:
    alias = aliases.get(name)
    if alias and "message" in alias:
      if alias["message"] not in by_name:
        errors.append(f"{label}: alias {name} names unknown message {alias['message']}")
      return
    wire = alias["wire"] if alias else name
    if msg is None:
      if wire not in all_fields:
        errors.append(f"{label}: field {name!r} is carried by no message")
      return
    body = by_name[msg]
    have = {f["name"] for f in body["fields"]}
    if wire in have or wire in model["key_aliases"] or wire in ("context_key", "task_key", "txn_id"):
      if wire in ("context_key", "task_key"):
        # The expectation is the C009 identity domain, never the schema group that is being checked.
        domain = {f["name"] for f in data["protocol"]["identity_domains"][wire]["fields"]}
        if not domain <= have:
          errors.append(f"{label}: {msg} does not carry the full {wire}: missing {sorted(domain - have)}")
      elif wire in model["key_aliases"]:
        if not set(model["key_aliases"][wire]["fields"]) <= have:
          errors.append(f"{label}: {msg} does not carry the fields of key {wire}")
      elif wire not in have:
        errors.append(f"{label}: {msg} does not carry {wire}")
      return
    errors.append(f"{label}: {msg} carries no field {name!r}")

  for fname in REQUEST_FILES:
    req = data["requests"][fname]
    for item in (
      (req.get("requests") or []) + (req.get("messages") or []) + (req.get("new_messages_requested") or [])
    ):
      msg = item.get("message") or item.get("name")
      label = f"{fname} {msg}"
      if msg not in by_name:
        errors.append(f"{label}: requested message is not in the wire schema")
        continue
      if "direction" in item and item["direction"] != by_name[msg]["dir"]:
        errors.append(
          f"{label}: requested direction {item['direction']} differs from the wire {by_name[msg]['dir']}"
        )
      if "dir" in item and item["dir"] != by_name[msg]["dir"]:
        errors.append(f"{label}: requested dir {item['dir']} differs from the wire {by_name[msg]['dir']}")
      if "channel" in item and item["channel"] != by_name[msg]["channel"]:
        errors.append(f"{label}: requested channel differs from the wire")
      for need in item.get("fields_needed") or []:
        need_field(f"{label}.{need['name']}", msg, need["name"])
      if item.get("new") is True or ("name" in item and "channel" in item):
        asked_new.setdefault(msg, set()).add(fname)
    for need in req.get("fields_needed") or []:
      need_field(f"{fname} {need['name']}", None, need["name"])
    for item in (req.get("csr_requests") or []) + (req.get("helpers") or []):
      csr = item.get("csr") or item.get("name")
      if csr not in csr_names:
        errors.append(f"{fname}: requested CSR {csr!r} is not declared in csr_name_map.yaml")
    for event in req.get("internal_events_not_on_wire") or []:
      if event["name"] in by_name or event["name"] in all_fields:
        errors.append(f"{fname}: {event['name']} is declared internal but is on the wire")
  for m in data["schema"].get("new_messages") or []:
    if m["request"] not in asked_new.get(m["name"], set()) and f"{m['request']}" not in asked_new.get(
      m["name"], set()
    ):
      errors.append(f"new message {m['name']}: request {m['request']} does not ask for it")


def _load(directory: Path) -> dict[str, Any]:
  data = vp.load_all(directory)
  data["schema"] = _yaml(directory / SCHEMA_FILE)
  data["requests"] = {n: _yaml(directory / "field_requests" / n) for n in REQUEST_FILES}
  return data


def build_model(directory: Path = PROTOCOL_DIR) -> dict[str, Any]:
  """Validate every input and return the model every renderer consumes."""
  data = _load(directory)
  errors: list[str] = []
  report = vp.validate(directory)
  errors.extend(f"catalog: {e}" for e in report["errors"])
  model = _build(data, errors)
  csr_names = {str(i["name"]) for i in data["csr"]["declared"]}
  _check_requests(model, data, csr_names, errors)
  window = data["csr"].get("allocation_window") or {}
  if window.get("source"):
    _check_src("csr_name_map allocation_window", window["source"], errors)
  csr: list[dict[str, Any]] = []
  try:
    csr = csr_allocator.allocate(data["csr"], data["upstream_csr"])
  except csr_allocator.CsrAllocError as exc:
    errors.append(f"csr: {exc}")
  widths: dict[str, Any] = {"derived": {}, "unresolved": []}
  try:
    result = derive_widths.derive(data["budgets"], data["lifetimes"])
    widths = {"derived": result.derived, "unresolved": result.unresolved}
  except derive_widths.WidthError as exc:
    errors.extend(f"widths: {e}" for e in exc.errors)
  if errors:
    raise GenError(errors)
  model["csr"] = csr
  model["widths"] = widths
  model["header"] = data["schema"]["header"]
  model["errors"] = [str(e["name"]) for e in data["schema"]["errors"]]
  model["consumer_model"] = data["schema"]["consumer_model"]
  model["open_items"] = {
    "transport_only_fields": model["transport_only"],
    "unconfirmed_aliases": sorted(n for n, a in model["request_aliases"].items() if a.get("open")),
    "unresolved_widths": [w["name"] for w in widths["unresolved"]],
  }
  model["inputs"] = {n: _sha(directory / n) for n in _INPUTS}
  model["directory"] = str(directory)
  return model


# ---------------------------------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------------------------------
def _ident(text: str) -> str:
  return re.sub(r"[^0-9A-Za-z]", "_", text).upper()


def render_layout(model: dict[str, Any]) -> str:
  doc = {
    "schema": "cn_layout/v1",
    "status": "draft",
    "protocol_only": True,
    "container": {"word_bytes": 8, "byte_order": "little_endian", "derived_width": False},
    "header_words": model["header"]["header_words"],
    "header": [w["name"] for w in model["header"]["words"]],
    "channels": model["channels"],
    "dirs": list(DIRS),
    "enums": model["enums"],
    "bitsets": model["bitsets"],
    "errors": model["errors"],
    "consumer_model": model["consumer_model"],
    "messages": model["messages"],
    "csr": model["csr"],
    "widths": model["widths"],
    "open_items": model["open_items"],
    "inputs": model["inputs"],
  }
  return json.dumps(doc, indent=2, sort_keys=False) + "\n"


_HEADER_NOTE = (
  "GENERATED by hardware/coral_nexus/protocol/gen_protocol.py from wire_schema_v1.yaml; do not edit.\n"
  "DRAFT transport layout: 64-bit little-endian words, no silicon opcode or width (C017)."
)


def render_c(model: dict[str, Any]) -> str:
  o: list[str] = [
    f"/* {_HEADER_NOTE.replace(chr(10), chr(10) + ' * ')} */",
    "#ifndef CN_MESSAGES_H_",
    "#define CN_MESSAGES_H_",
    "",
  ]
  o += ["#include <stddef.h>", "#include <stdint.h>", ""]
  o += ["#define CN_WORD_BYTES 8", f"#define CN_HEADER_WORDS {model['header']['header_words']}"]
  o += [f"#define CN_CONTEXT_SLOTS {model['consumer_model']['context_slots']}"]
  o += [f"#define CN_DEDUP_ENTRIES {model['consumer_model']['dedup_entries']}", ""]
  o += ["enum cn_msg_type {"]
  o += [f"  CN_MSG_{m['name']} = {m['id']}," for m in model["messages"]]
  o += [f"  CN_MSG_COUNT = {len(model['messages'])}", "};", ""]
  o += ["enum cn_error {"]
  o += [f"  CN_ERR_{e} = {i}," for i, e in enumerate(model["errors"])]
  o += [f"  CN_ERR_COUNT = {len(model['errors'])}", "};", ""]
  o += [
    "enum cn_kind { CN_KIND_UINT = 0, CN_KIND_ENUM, CN_KIND_BOOL, CN_KIND_BITSET, CN_KIND_DIGEST };",
    "",
  ]
  o += [f"#define CN_CHANNELS {len(model['channels'])}", f"#define CN_DIRS {len(DIRS)}"]
  o += [f"#define CN_LINKS ({len(model['channels'])} * {len(DIRS)})", ""]
  for e, members in model["enums"].items():
    o += [f"enum cn_enum_{e} {{"]
    o += [f"  CN_{_ident(e)}_{_ident(m)} = {i}," for i, m in enumerate(members)]
    o += ["};"]
  o += [""]
  o += [
    "typedef struct {",
    "  const char* name;",
    "  uint8_t kind;",
    "  uint8_t offset;",
    "  uint8_t words;",
    "  uint64_t limit; /* enum: member count (+1 if optional); bool: 1; bitset: legal bits; uint: 0 */",
    "} cn_field_t;",
    "typedef struct {",
    "  uint8_t kind; /* 0: words a and b equal; 1: word a nonzero iff word b equals value */",
    "  uint8_t a;",
    "  uint8_t b;",
    "  uint8_t value;",
    "} cn_cond_t;",
    "typedef struct {",
    "  const char* name;",
    "  uint8_t channel;",
    "  uint8_t dir;",
    "  uint8_t host_only;",
    "  uint8_t payload_words;",
    "  uint8_t nfields;",
    "  uint8_t nconds;",
    "  int8_t txn, ctx_slot, ctx_gen, stage, ack_for;",
    "  uint8_t establishes;",
    "  const cn_field_t* fields;",
    "  const cn_cond_t* conds;",
    "} cn_msg_t;",
    "",
  ]
  for m in model["messages"]:
    o += ["typedef struct {"]
    for f in m["fields"]:
      o += [f"  uint64_t {f['name']}[{f['words']}];" if f["words"] > 1 else f"  uint64_t {f['name']};"]
    if not m["fields"]:
      o += ["  uint64_t unused_[0];"]
    o += [f"}} cn_{m['name'].lower()}_t;"]
    o += [
      f'_Static_assert(sizeof(cn_{m["name"].lower()}_t) == {m["payload_words"] * 8}, "{m["name"]} size");'
    ]
    for f in m["fields"]:
      o += [
        f"_Static_assert(offsetof(cn_{m['name'].lower()}_t, {f['name']}) == {f['offset'] * 8}, "
        f'"{m["name"]}.{f["name"]} offset");'
      ]
  o += [""]
  for m in model["messages"]:
    o += [f"static const cn_field_t cn_fields_{m['name'].lower()}[] = {{"]
    for f in m["fields"]:
      kind = f"CN_KIND_{f['kind'].upper()}"
      o += [f'  {{"{f["name"]}", {kind}, {f["offset"]}, {f["words"]}, {f["limit"]}ull}},']
    if not m["fields"]:
      o += ['  {"", CN_KIND_UINT, 0, 0, 0ull},']
    o += ["};"]
    o += [f"static const cn_cond_t cn_conds_{m['name'].lower()}[] = {{"]
    for c in m["conditions"]:
      o += [f"  {{{0 if c['kind'] == 'equal' else 1}, {c['a']}, {c['b']}, {c['value']}}},"]
    if not m["conditions"]:
      o += ["  {0, 0, 0, 0},"]
    o += ["};"]
  o += ["", "static const cn_msg_t cn_msgs[CN_MSG_COUNT] = {"]
  for m in model["messages"]:
    n = m["name"].lower()
    o += [
      f'  {{"{m["name"]}", {m["channel_index"]}, {m["dir_index"]}, {int(m["host_only"])}, '
      f"{m['payload_words']}, {len(m['fields'])}, {len(m['conditions'])}, {m['txn']}, "
      f"{m['ctx_slot']}, {m['ctx_gen']}, {m['stage']}, "
      f"{m['ack_for']}, {int(m['establishes'])}, cn_fields_{n}, cn_conds_{n}}},"
    ]
  o += ["};", ""]
  o += ["static const char* const cn_error_names[CN_ERR_COUNT] = {"]
  o += [f'  "{e}",' for e in model["errors"]]
  o += ["};", ""]
  for c in model["csr"]:
    o += [f"#define CN_CSR_{_ident(c['name'])} 0x{c['address']:03x}"]
  o += ["", "#endif /* CN_MESSAGES_H_ */", ""]
  return "\n".join(o)


def render_sv(model: dict[str, Any]) -> str:
  msgs = model["messages"]
  o: list[str] = [f"// {_HEADER_NOTE.replace(chr(10), chr(10) + '// ')}", ""]
  o += [f"localparam int CN_HEADER_WORDS = {model['header']['header_words']};"]
  o += [f"localparam int CN_CONTEXT_SLOTS = {model['consumer_model']['context_slots']};"]
  o += [f"localparam int CN_DEDUP_ENTRIES = {model['consumer_model']['dedup_entries']};"]
  o += [
    f"localparam int CN_MSG_COUNT = {len(msgs)};",
    f"localparam int CN_ERR_COUNT = {len(model['errors'])};",
  ]
  o += [f"localparam int CN_LINKS = {len(model['channels']) * len(DIRS)};"]
  o += [f"localparam int CN_DIRS = {len(DIRS)};"]
  o += [f"localparam int CN_MSG_{m['name']} = {m['id']};" for m in msgs]
  o += [f"localparam int CN_ERR_{e} = {i};" for i, e in enumerate(model["errors"])]
  for m in msgs:
    o += [f"localparam int CN_WORDS_{m['name']} = {m['payload_words']};"]
    o += [f"localparam int CN_OFF_{m['name']}_{_ident(f['name'])} = {f['offset']};" for f in m["fields"]]
  for c in model["csr"]:
    o += [f"localparam int CN_CSR_{_ident(c['name'])} = 12'h{c['address']:03x};"]
  o += [""]

  def table(fname: str, ret: str, rows: dict[int, Any], default: Any) -> None:
    o.append(f"function automatic {ret} {fname}(input int m);")
    o.append("  case (m)")
    for k, v in rows.items():
      o.append(f"    {k}: {fname} = {v};")
    o.append(f"    default: {fname} = {default};")
    o.extend(["  endcase", "endfunction"])

  table("cn_msg_words", "int", {m["id"]: m["payload_words"] for m in msgs}, -1)
  table("cn_msg_chan", "int", {m["id"]: m["channel_index"] for m in msgs}, -1)
  table("cn_msg_dir", "int", {m["id"]: m["dir_index"] for m in msgs}, -1)
  table("cn_msg_nfields", "int", {m["id"]: len(m["fields"]) for m in msgs}, 0)
  table("cn_msg_nconds", "int", {m["id"]: len(m["conditions"]) for m in msgs}, 0)
  for role in ("txn", "ctx_slot", "ctx_gen", "stage", "ack_for"):
    table(f"cn_msg_{role}", "int", {m["id"]: m[role] for m in msgs}, -1)
  table("cn_msg_establishes", "int", {m["id"]: int(m["establishes"]) for m in msgs}, 0)

  def indexed(fname: str, ret: str, rows: dict[tuple[int, int], Any], default: Any) -> None:
    o.append(f"function automatic {ret} {fname}(input int m, input int i);")
    o.append("  case ({m[15:0], i[15:0]})")
    for (m, i), v in rows.items():
      o.append(f"    32'h{m:04x}{i:04x}: {fname} = {v};")
    o.append(f"    default: {fname} = {default};")
    o.extend(["  endcase", "endfunction"])

  indexed(
    "cn_field_off", "int", {(m["id"], i): f["offset"] for m in msgs for i, f in enumerate(m["fields"])}, -1
  )
  indexed(
    "cn_field_kind",
    "int",
    {(m["id"], i): KINDS.index(f["kind"]) for m in msgs for i, f in enumerate(m["fields"])},
    0,
  )
  indexed(
    "cn_field_limit",
    "longint unsigned",
    {(m["id"], i): f"64'd{f['limit']}" for m in msgs for i, f in enumerate(m["fields"])},
    "64'd0",
  )
  conds = {(m["id"], i): c for m in msgs for i, c in enumerate(m["conditions"])}
  indexed("cn_cond_kind", "int", {k: (0 if c["kind"] == "equal" else 1) for k, c in conds.items()}, 0)
  indexed("cn_cond_a", "int", {k: c["a"] for k, c in conds.items()}, 0)
  indexed("cn_cond_b", "int", {k: c["b"] for k, c in conds.items()}, 0)
  indexed("cn_cond_value", "int", {k: c["value"] for k, c in conds.items()}, 0)
  o.append("function automatic string cn_err_name(input int e);")
  o.append("  case (e)")
  o += [f'    {i}: cn_err_name = "{e}";' for i, e in enumerate(model["errors"])]
  o += ['    default: cn_err_name = "?";', "  endcase", "endfunction", ""]
  return "\n".join(o)


def _pf(value: Any) -> str:
  return pprint.pformat(value, width=104, compact=True, sort_dicts=False)


def render_wire_py(model: dict[str, Any]) -> str:
  messages = [
    {
      "name": m["name"],
      "channel": m["channel_index"],
      "dir": m["dir_index"],
      "host_only": m["host_only"],
      "words": m["payload_words"],
      "fields": [
        (f["name"], f["kind"], f["offset"], f["words"], f["limit"], f["enum"], f["optional"])
        for f in m["fields"]
      ],
      "conds": [(0 if c["kind"] == "equal" else 1, c["a"], c["b"], c["value"]) for c in m["conditions"]],
      "txn": m["txn"],
      "ctx_slot": m["ctx_slot"],
      "ctx_gen": m["ctx_gen"],
      "stage": m["stage"],
      "ack_for": m["ack_for"],
      "establishes": m["establishes"],
    }
    for m in model["messages"]
  ]
  parts = [
    '"""GENERATED by hardware/coral_nexus/protocol/gen_protocol.py from wire_schema_v1.yaml; do not edit.',
    "",
    "DRAFT co-simulation transport layout (C017): 64-bit little-endian words, no silicon opcode or width.",
    "pack/unpack and the receive model (Rx) are the Python side of the same layout the generated C header",
    'and SystemVerilog offsets describe."""',
    "",
    "from __future__ import annotations",
    "",
    "from dataclasses import dataclass, field",
    "from typing import Any",
    "",
    f"HEADER_WORDS = {model['header']['header_words']}",
    f"CONTEXT_SLOTS = {model['consumer_model']['context_slots']}",
    f"DEDUP_ENTRIES = {model['consumer_model']['dedup_entries']}",
    f"CHANNELS = {_pf(model['channels'])}",
    f"DIRS = {_pf(list(DIRS))}",
    f"ERRORS = {_pf(model['errors'])}",
    f"ENUMS: dict[str, list[str]] = {_pf(model['enums'])}",
    f"BITSETS: dict[str, list[str]] = {_pf(model['bitsets'])}",
    "# (name, channel index, dir index, host_only, payload words, fields, conds, txn, ctx_slot, ctx_gen,",
    "#  stage,",
    "#  ack_for, establishes); field = (name, kind, offset, words, limit, enum/bitset name, optional)",
    f"MESSAGES: list[dict[str, Any]] = {_pf(messages)}",
    f"CSR: dict[str, int] = {_pf({c['name']: c['address'] for c in model['csr']})}",
    "MSG_INDEX = {m['name']: i for i, m in enumerate(MESSAGES)}",
    "MASK64 = (1 << 64) - 1",
    "",
    "",
    "class WireError(Exception):",
    '  """A rejected message or a bad pack request; `code` is a name from ERRORS or a pack-side code."""',
    "",
    '  def __init__(self, code: str, detail: str = "") -> None:',
    '    super().__init__(f"{code}: {detail}" if detail else code)',
    "    self.code = code",
    "",
    "",
    "def _enum_index(enum: str, value: Any, name: str, optional: bool) -> int:",
    "  members = ENUMS.get(enum) or BITSETS[enum]",
    '  if optional and (value is None or value == "NONE"):',
    "    return 0",
    "  if isinstance(value, str):",
    "    if value not in members:",
    '      raise WireError("ENUM_RANGE", f"{name}={value!r}")',
    "    return members.index(value) + (1 if optional else 0)",
    "  return int(value)",
    "",
    "",
    "def pack(message: str, values: dict[str, Any], message_seq: int = 0) -> bytes:",
    '  """Encode one message. Unknown/missing fields and values that do not fit a word are rejected."""',
    "  if message not in MSG_INDEX:",
    '    raise WireError("UNKNOWN_MSG", message)',
    "  spec = MESSAGES[MSG_INDEX[message]]",
    '  names = [f[0] for f in spec["fields"]]',
    "  unknown = sorted(set(values) - set(names))",
    "  if unknown:",
    '    raise WireError("FIELD_UNKNOWN", f"{message}: {unknown}")',
    '  words = [0] * spec["words"]',
    '  for name, kind, offset, count, _limit, enum, optional in spec["fields"]:',
    "    if name not in values:",
    '      raise WireError("FIELD_MISSING", f"{message}.{name}")',
    "    value = values[name]",
    '    if kind == "digest":',
    '      raw = value.to_bytes(count * 8, "little") if isinstance(value, int) else bytes(value)',
    "      if len(raw) != count * 8:",
    '        raise WireError("OVERFLOW", f"{message}.{name}")',
    "      for i in range(count):",
    '        words[offset + i] = int.from_bytes(raw[i * 8 : i * 8 + 8], "little")',
    "      continue",
    '    number = _enum_index(enum, value, name, optional) if kind == "enum" else int(value)',
    "    if number < 0 or number > MASK64:",
    '      raise WireError("OVERFLOW", f"{message}.{name}={value!r} does not fit one 64-bit word")',
    "    words[offset] = number",
    '  header = [MSG_INDEX[message], spec["words"], message_seq, 0]',
    "  if not 0 <= message_seq <= MASK64:",
    '    raise WireError("OVERFLOW", f"message_seq {message_seq}")',
    '  return b"".join(w.to_bytes(8, "little") for w in header + words)',
    "",
    "",
    "def _words(blob: bytes) -> list[int]:",
    '  return [int.from_bytes(blob[i : i + 8], "little") for i in range(0, len(blob) - len(blob) % 8, 8)]',
    "",
    "",
    "@dataclass",
    "class Decoded:",
    "  message: str",
    "  message_seq: int",
    "  words: list[int]",
    "  values: dict[str, Any] = field(default_factory=dict)",
    "",
    "",
    "def structural(blob: bytes, port: tuple[int, int]) -> tuple[str, Decoded | None]:",
    '  """Apply every stateless rejection; returns (error name, decoded message or None)."""',
    "  if len(blob) < HEADER_WORDS * 8:",
    '    return "TRUNCATED", None',
    "  head = _words(blob[: HEADER_WORDS * 8])",
    "  if head[0] >= len(MESSAGES):",
    '    return "UNKNOWN_MSG", None',
    "  if head[3] != 0:",
    '    return "RESERVED", None',
    "  spec = MESSAGES[head[0]]",
    '  if head[1] != spec["words"]:',
    '    return "LENGTH_MISMATCH", None',
    '  total = (HEADER_WORDS + spec["words"]) * 8',
    "  if len(blob) < total:",
    '    return "TRUNCATED", None',
    "  if len(blob) > total:",
    '    return "TRAILING", None',
    '  if (spec["channel"], spec["dir"]) != port:',
    '    return "WRONG_PORT", None',
    "  words = _words(blob[HEADER_WORDS * 8 : total])",
    '  for _name, kind, offset, _count, limit, _enum, _optional in spec["fields"]:',
    "    value = words[offset]",
    '    if kind == "enum" and value >= limit:',
    '      return "ENUM_RANGE", None',
    '    if kind == "bool" and value > 1:',
    '      return "BOOL_RANGE", None',
    '    if kind == "bitset" and value & ~limit & MASK64:',
    '      return "BITSET_RANGE", None',
    '  for kind, a, b, value in spec["conds"]:',
    "    if kind == 0 and words[a] != words[b]:",
    '      return "CONDITION", None',
    "    if kind == 1 and (words[a] != 0) != (words[b] == value):",
    '      return "CONDITION", None',
    '  return "OK", Decoded(spec["name"], head[2], words)',
    "",
    "",
    "def unpack(blob: bytes, port: tuple[int, int] | None = None) -> Decoded:",
    '  """Stateless decode. `port` defaults to the message\'s own (channel, dir)."""',
    "  if port is None and len(blob) >= 8:",
    '    index = int.from_bytes(blob[:8], "little")',
    "    if index < len(MESSAGES):",
    '      port = (MESSAGES[index]["channel"], MESSAGES[index]["dir"])',
    "  code, decoded = structural(blob, port or (-1, -1))",
    "  if decoded is None:",
    "    raise WireError(code)",
    "  spec = MESSAGES[MSG_INDEX[decoded.message]]",
    '  for name, kind, offset, count, _limit, enum, optional in spec["fields"]:',
    '    if kind == "digest":',
    "      raw = decoded.words[offset : offset + count]",
    '      decoded.values[name] = b"".join(w.to_bytes(8, "little") for w in raw)',
    '    elif kind == "enum" and enum:',
    "      members = ENUMS[enum]",
    "      index = decoded.words[offset] - (1 if optional else 0)",
    "      decoded.values[name] = members[index] if 0 <= index < len(members) else None",
    "    else:",
    "      decoded.values[name] = decoded.words[offset]",
    "  return decoded",
    "",
    "",
    "class Rx:",
    '  """Receive model of ONE Tile link: sequence, dedup, ACK reference and stale-generation rules."""',
    "",
    "  def __init__(self) -> None:",
    "    self.next_seq = [0] * (len(CHANNELS) * len(DIRS))",
    "    self.dedup: list[tuple[int, int, int]] = []",
    "    self.requests: set[tuple[int, int]] = set()",
    "    self.live = [0] * CONTEXT_SLOTS",
    "    self.effects = 0",
    "",
    "  def deliver(self, port: tuple[int, int], blob: bytes) -> tuple[str, list[int] | None]:",
    '    """Returns (error name, payload words when accepted). State changes only as the rules say."""',
    "    code, decoded = structural(blob, port)",
    "    if decoded is None:",
    "      return code, None",
    "    spec = MESSAGES[MSG_INDEX[decoded.message]]",
    "    words = decoded.words",
    '    slot = words[spec["ctx_slot"]] if spec["ctx_slot"] >= 0 else -1',
    "    if slot >= CONTEXT_SLOTS:",
    '      return "CAPACITY", None',
    "    link = port[0] * len(DIRS) + port[1]",
    "    if decoded.message_seq < self.next_seq[link]:",
    '      return "TRANSPORT_REPLAY", None',
    "    if decoded.message_seq > self.next_seq[link]:",
    '      return "INCOMPLETE", None',
    "    self.next_seq[link] += 1",
    '    gen = words[spec["ctx_gen"]] if spec["ctx_gen"] >= 0 else 0',
    "    if slot >= 0 and gen < self.live[slot]:",
    '      return "STALE_GENERATION", None',
    '    txn = words[spec["txn"]] if spec["txn"] >= 0 else None',
    '    if spec["name"] == "ACK" and txn is not None:',
    '      if (txn, words[spec["ack_for"]]) not in self.requests:',
    '        return "ACK_MISMATCH", None',
    "    key = None",
    "    if txn is not None:",
    '      key = (txn, MSG_INDEX[spec["name"]], words[spec["stage"]] if spec["stage"] >= 0 else 0)',
    "      if key in self.dedup:",
    '        return "DUPLICATE_STAGE", None',
    "      if len(self.dedup) >= DEDUP_ENTRIES:",
    '        return "CAPACITY", None',
    "      self.dedup.append(key)",
    '      self.requests.add((txn, MSG_INDEX[spec["name"]]))',
    '    if slot >= 0 and spec["establishes"]:',
    "      self.live[slot] = gen",
    "    self.effects += 1",
    '    return "OK", words',
    "",
    "  def line(self, code: str, words: list[int] | None) -> str:",
    '    """One canonical output line; the C and SystemVerilog consumers print the identical text."""',
    '    seq = ",".join(str(n) for n in self.next_seq)',
    "    live = sum(self.live) & MASK64",
    '    fields = ",".join(format(w, "x") for w in words) if words is not None else "-"',
    '    return f"{code} eff={self.effects} dd={len(self.dedup)} seq={seq} live={live} f={fields}"',
    "",
  ]
  return "\n".join(parts)


def render_all(model: dict[str, Any]) -> dict[Path, str]:
  return {
    GENERATED_DIR / "cn_layout.json": render_layout(model),
    GENERATED_DIR / "cn_messages.h": render_c(model),
    GENERATED_DIR / "cn_offsets.svh": render_sv(model),
    WIRE_PY: render_wire_py(model),
  }


def _atomic_write(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
  try:
    with os.fdopen(fd, "w") as handle:
      handle.write(text)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
  except BaseException:
    Path(tmp).unlink(missing_ok=True)
    raise


def generate(directory: Path = PROTOCOL_DIR, out: dict[Path, Path] | None = None) -> dict[Path, str]:
  """Build + validate + render everything first, then write. Failure writes nothing.

  `out` maps a default output path to a replacement (tests redirect into tmp_path).
  """
  model = build_model(directory)
  rendered = render_all(model)
  if WIRE_PY.parent.is_dir() is False and (out is None or WIRE_PY not in out):
    raise GenError([f"{WIRE_PY.parent} does not exist"])
  for path, text in rendered.items():
    _atomic_write((out or {}).get(path, path), text)
  return rendered


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dir", default=str(PROTOCOL_DIR))
  args = parser.parse_args()
  try:
    rendered = generate(Path(args.dir))
  except GenError as exc:
    print("gen_protocol: FAILED, nothing written", file=sys.stderr)
    print("\n".join(f"  {e}" for e in exc.errors), file=sys.stderr)
    raise SystemExit(1) from exc
  for path in rendered:
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
  main()
