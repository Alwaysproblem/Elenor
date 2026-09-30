#!/usr/bin/env python3
"""Token-preserving MLIR line wrapper: reflows >100-column lines.

stdlib-only (importable with no project dependencies) so the pre-commit
hook can run it with the ambient ``python3``. The transformer workload
generators also import ``format_text`` from here to keep their emitted
fixtures wrapped.

Guarantees:
- Never mutates token text on code lines; only inserts newlines + indent
  at unit boundaries. Unit = attr group (`name = value`), call group
  (`name(...)`), dict region (`name { ... }`), type ascription (`: (...)`),
  op head (`%r = op.name @sym(args)` / `op.name @sym(args)`), or single atom.
- `// case: {JSON}` comments re-serialize with indent (checker asserts
  json.loads equality); unparseable `case:` lines untouched.
- Plain `//` comments word-wrap; every continuation keeps a `//` marker.
- Lines <= WIDTH stay untouched.

CLI: ``python3 scripts/format_mlir.py [--check] [PATH ...]`` -- PATH may be
a file or a directory (searched recursively for ``*.mlir``); with no PATH
the ``examples/`` tree is formatted. ``--check`` exits 1 listing files that
would change, without writing (used by the pre-commit ``format-mlir`` hook).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

WIDTH = 100
CONT = "  "
WORDCHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
SSACHARS = WORDCHARS | {"-"}
# (kind, text, gap_before): kind in {str, sig, ssa, sym, word, num, punct}
Token = tuple[str, str, bool]
OPENERS = {"(", "[", "{"}
CLOSER = {"(": ")", "[": "]", "{": "}"}

CASE_RE = re.compile(r"^(\s*)//\s*case:\s*(\{.*\})\s*$")
SHORT_ARRAY_RE = re.compile(r"\[\s+([^\[\]{}]*?)\s+\]")


def tokenize(line: str) -> list[Token]:
  toks = []  # (kind, text, gap_before)
  i, n = 0, len(line)
  gap = False
  while i < n:
    c = line[i]
    if c in " \t":
      gap = True
      i += 1
      continue
    if c == '"':
      j = i + 1
      while j < n:
        if line[j] == "\\":
          j += 2
          continue
        if line[j] == '"':
          j += 1
          break
        j += 1
      text, kind = line[i:j], "str"
    elif c in "!#":
      j = i + 1
      while j < n and line[j] in WORDCHARS:
        j += 1
      if j < n and line[j] == "<":
        depth, k = 0, j
        while k < n:
          ch = line[k]
          if ch == '"':
            k += 1
            while k < n and line[k] != '"':
              k += 1
          elif ch == "<":
            depth += 1
          elif ch == ">":
            depth -= 1
            if depth == 0:
              k += 1
              break
          k += 1
        j = k
      text, kind = line[i:j], "sig"
    elif c in "%@^":
      j = i + 1
      while j < n and line[j] in SSACHARS:
        j += 1
      text, kind = line[i:j], ("ssa" if c == "%" else "sym")
    elif c.isalpha() or c == "_":
      j = i
      while j < n and line[j] in WORDCHARS:
        j += 1
      text, kind = line[i:j], "word"
    elif c.isdigit():
      j = i
      while j < n and (line[j].isalnum() or line[j] in "xX"):
        j += 1
      text, kind = line[i:j], "num"
    else:
      text, kind = c, "punct"
    toks.append((kind, text, gap))
    gap = False
    i += len(text)
  return toks


def read_group(toks: list[Token], i: int) -> tuple[list[Token], int]:
  """Consume a balanced bracket run starting at opener toks[i]."""
  depth = 0
  out = []
  while i < len(toks):
    kind, text, _gap = toks[i]
    out.append(toks[i])
    i += 1
    if kind == "punct" and text in OPENERS:
      depth += 1
    elif kind == "punct" and text in ")]}":
      depth -= 1
      if depth == 0:
        break
  return out, i


def render_tokens(group: list[Token]) -> str:
  parts: list[str] = []
  for _kind, text, gap in group:
    if not parts:
      parts.append(text)
      continue
    prev = parts[-1]
    if gap and text not in ",)]}" and prev not in "([{":
      parts.append(" ")
    parts.append(text)
  return "".join(parts)


def head_extra(head: list[Token]) -> dict[str, Any]:
  """If head ends with a balanced (...) group, expose it for splitting."""
  if len(head) >= 2 and head[-1][0] == "punct" and head[-1][1] == ")":
    depth = 0
    for idx in range(len(head) - 1, -1, -1):
      t = head[idx]
      if t[0] == "punct" and t[1] in ")]}":
        depth += 1
      elif t[0] == "punct" and t[1] in OPENERS:
        depth -= 1
        if depth == 0:
          if head[idx][1] == "(":
            return {"group": head[idx:]}
          return {}
  return {}


def build_units(toks: list[Token]) -> list[dict[str, Any]]:
  units: list[dict[str, Any]] = []
  i, n = 0, len(toks)
  while i < n:
    kind, text, _gap = toks[i]
    nxt = toks[i + 1] if i + 1 < n else None
    # result head: ssa (, ssa)* '=' opword [@sym|"str" [(args)]]
    if kind == "ssa":
      j = i
      while j + 2 < n and toks[j + 1][0] == "punct" and toks[j + 1][1] == "," and toks[j + 2][0] == "ssa":
        j += 2
      if j + 1 < n and toks[j + 1][0] == "punct" and toks[j + 1][1] == "=":
        k = j + 2
        head = list(toks[i:k])
        if k < n and toks[k][0] == "word":
          head.append(toks[k])
          k += 1
          if k < n and toks[k][0] in ("sym", "str"):
            head.append(toks[k])
            k += 1
            if k < n and toks[k][0] == "punct" and toks[k][1] == "(":
              grp, k = read_group(toks, k)
              head.extend(grp)
        units.append({"kind": "head", "toks": head, **head_extra(head)})
        i = k
        continue
    if kind == "word":
      # dict region: `name { ... }`
      if nxt and nxt[0] == "punct" and nxt[1] == "{":
        grp, k = read_group(toks, i + 1)
        units.append({"kind": "dict", "name": text, "toks": [toks[i], *grp], "group": grp})
        i = k
        continue
      if nxt and nxt[0] == "punct" and nxt[1] == "=":
        val = toks[i + 2] if i + 2 < n else None
        if val and val[0] == "punct" and val[1] in OPENERS:
          grp, k = read_group(toks, i + 2)
          units.append({"kind": "attr", "name": text, "toks": [toks[i], toks[i + 1], *grp], "group": grp})
          i = k
        elif val:
          units.append({"kind": "attr", "name": text, "toks": toks[i : i + 3]})
          i += 3
        else:
          units.append({"kind": "atom", "toks": [toks[i], toks[i + 1]]})
          i += 2
        continue
      if nxt and nxt[0] == "punct" and nxt[1] == "(":
        grp, k = read_group(toks, i + 1)
        units.append({"kind": "call", "name": text, "toks": [toks[i], *grp], "group": grp})
        i = k
        continue
      if nxt and nxt[0] in ("sym", "str"):
        head = [toks[i], nxt]
        k = i + 2
        if k < n and toks[k][0] == "punct" and toks[k][1] == "(":
          grp, k = read_group(toks, k)
          head.extend(grp)
        units.append({"kind": "head", "toks": head, **head_extra(head)})
        i = k
        continue
      if text in ("into", "from") and nxt and nxt[0] in ("ssa", "sym"):
        units.append({"kind": "atom", "toks": [toks[i], nxt]})
        i += 2
        continue
      units.append({"kind": "atom", "toks": [toks[i]]})
      i += 1
      continue
    if kind == "punct" and text == ":":
      k = i + 1
      if k < n and toks[k][0] == "punct" and toks[k][1] in OPENERS:
        grp, k = read_group(toks, k)
        units.append({"kind": "type", "toks": [toks[i], *grp], "group": grp})
      elif k < n:
        units.append({"kind": "type", "toks": toks[i : k + 1]})
        k += 1
      i = k
      continue
    units.append({"kind": "atom", "toks": [toks[i]]})
    i += 1
  return units


def split_chunks(tokens: list[Token]) -> tuple[list[list[Token]], list[Token]]:
  """Split token list into comma-terminated chunks at nesting depth 0."""
  chunks = []
  cur = []
  depth = 0
  for t in tokens:
    kind, text, _gap = t
    if kind == "punct" and text in OPENERS:
      depth += 1
    elif kind == "punct" and text in ")]}":
      depth -= 1
    cur.append(t)
    if kind == "punct" and text == "," and depth == 0:
      chunks.append(cur)
      cur = []
  return chunks, cur


class Wrapper:
  def __init__(self, indent: str) -> None:
    self.indent = indent
    self.cont = indent + CONT
    self.lines: list[str] = []
    self.cur: str | None = None
    self.cur_indent = indent

  def limit(self) -> int:
    return WIDTH - len(self.cur_indent)

  def flush(self) -> None:
    if self.cur is not None:
      self.lines.append(self.cur_indent + self.cur.rstrip())
    self.cur = None

  def append_unit(self, unit: dict[str, Any]) -> None:
    text = render_tokens(unit["toks"])
    sep = "" if text[:1] in ",)]}" else " "
    splittable = bool(unit.get("group")) and unit["kind"] in ("call", "type", "attr", "head")
    alone_over = len(text) + len(self.cont) > WIDTH
    if self.cur is None:
      if (
        unit["kind"] == "atom" and text == "{" and self.lines and not self.lines[-1].rstrip().endswith("{")
      ):
        self.lines[-1] = self.lines[-1].rstrip() + " {"
        return
      if unit["kind"] == "head" and alone_over and unit["toks"][0][0] == "ssa":
        self.split_head_results(unit)
        return
      if alone_over and splittable:
        self.split_unit(unit, None)
        return
      self.cur = text
      return
    cand = self.cur + sep + text
    glued = (unit["kind"] == "atom" and unit["toks"][0][1] == "{") or self.cur.strip() == "}"
    if len(cand) <= self.limit():
      self.cur = cand
      return
    if glued:
      if splittable:
        # keep `}` glued to the unit head, but split its group
        self.split_unit(unit, self.cur)
        return
      self.cur = cand
      return
    # does not fit after current segment
    if splittable and len(text) + len(self.cont) > WIDTH:
      self.split_unit(unit, self.cur)
      return
    if unit["kind"] == "head" and unit["toks"][0][0] == "ssa" and len(text) + len(self.cont) > WIDTH:
      self.split_head_results(unit)
      return
    self.flush()
    self.cur_indent = self.cont
    self.cur = text

  def split_head_results(self, unit: dict[str, Any]) -> None:
    """pow-style: put `%a, %b =` on its own line, op starts next line."""
    toks = unit["toks"]
    eq = next(i for i, t in enumerate(toks) if t[0] == "punct" and t[1] == "=")
    results = render_tokens(toks[: eq + 1]).rstrip()
    op = render_tokens(toks[eq + 1 :])
    if self.cur is not None:
      self.flush()
    self.lines.append(self.cur_indent + results)
    self.cur_indent = self.cont
    rest = op
    if len(rest) + len(self.cur_indent) > WIDTH and unit.get("group"):
      sub = {"kind": "head", "toks": toks[eq + 1 :], "group": unit["group"]}
      self.append_unit(sub)
      return
    self.cur = rest

  def split_unit(self, unit: dict[str, Any], prefix: str | None) -> None:
    """Split unit's bracket group across lines; prefix is the text that
    preceded this unit on the current segment (never contains it)."""
    group = unit["group"]
    opener = group[0][1]
    closer = CLOSER[opener]
    name = render_tokens(unit["toks"][: -len(group)]).rstrip()
    sep = " " if (name.endswith("=") or name.endswith(":")) else ""
    self.cur = None
    head = (prefix + " " if prefix else "") + name + sep + opener
    if prefix and len(self.cur_indent + head) > WIDTH:
      self.lines.append(self.cur_indent + prefix.rstrip())
      self.cur_indent = self.cont
      head = name + sep + opener
    chunks, tail = split_chunks(group[1:-1])
    allc = [*chunks, tail] if tail else chunks
    if not allc:
      self.lines.append(self.cur_indent + head + closer)
      self.cur_indent = self.cont
      return
    self.lines.append(self.cur_indent + head)
    ci = self.cur_indent + CONT
    line = ""
    for ch in allc:
      t = render_tokens(ch).rstrip()
      if not line:
        line = t
      elif len(ci + line + " " + t) <= WIDTH - 2:
        line = line + " " + t
      else:
        self.lines.append(ci + line)
        line = t
    if not line.endswith(closer):
      line = line + closer
    self.lines.append(ci + line)
    self.cur_indent = self.cont

  def dict_unit(self, unit: dict[str, Any]) -> None:
    if self.cur is not None:
      self.flush()
    self.cur_indent = self.cont
    self.lines.append(self.cur_indent + unit["name"] + " {")
    inner = self.cont + CONT
    chunks, tail = split_chunks(unit["group"][1:-1])
    entries = [render_tokens(c).rstrip().rstrip(",") for c in chunks]
    if tail:
      tail_text = render_tokens(tail).strip()
      if tail_text:
        entries.append(tail_text)
    if not chunks and not tail:
      entries = [render_tokens(unit["group"][1:-1]).strip()]
    for t in entries:
      if not t:
        continue
      if len(inner + t) <= WIDTH:
        self.lines.append(inner + t)
      else:
        sub = Wrapper(inner)
        for u in build_units(tokenize(t)):
          sub.append_unit(u)
        sub.flush()
        self.lines.extend(sub.lines)
    self.cur_indent = self.cont
    self.cur = "}"

  def finish(self) -> list[str]:
    self.flush()
    return self.lines


SIG_ATTR_RE = re.compile(
  r"^(?P<ind>[ \t]*)(?P<pre>.+?) = (?P<sig>[!#][A-Za-z_][\w.]*)<(?P<inner>.*)>(?P<tail>\s*\{?)\s*$"
)


def split_sig_commas(inner: str) -> tuple[list[str], str]:
  """Split sig interior at bracket-depth-0 commas."""
  chunks = []
  depth = 0
  start = 0
  i, n = 0, len(inner)
  while i < n:
    c = inner[i]
    if c == '"':
      i += 1
      while i < n and inner[i] != '"':
        i += 2 if inner[i] == "\\" else 1
    elif c in "([{":
      depth += 1
    elif c in ")]}":
      depth -= 1
    elif c == "," and depth == 0:
      chunks.append(inner[start : i + 1])
      start = i + 1
    i += 1
  tail = inner[start:]
  return chunks, tail


DICT_ATTR_RE = re.compile(r"^(?P<ind>[ \t]*)(?P<pre>.+?) = \{(?P<inner>.*)\}(?P<tail>\s*(>\s*\{?)?,?)\s*$")
DICT_TAIL_RE = re.compile(r"^(?P<comma>,?)\s*(?P<angle>>)?\s*(?P<brace>\{?)$")


def split_dict_line(line: str) -> list[str] | None:
  """Break an oversized `key = {a = x, b = y}` dict chunk at its commas."""
  if len(line) <= WIDTH:
    return None
  m = DICT_ATTR_RE.match(line)
  if not m:
    return None
  inner = m.group("inner")
  if "{" in inner or "}" in inner:
    return None
  chunks, tail = split_sig_commas(inner)
  if not chunks:
    return None
  allc = [*chunks, tail] if tail.strip() else chunks
  ind = m.group("ind")
  ci = ind + CONT
  head = f"{ind}{m.group('pre')} = {{"
  out = [head]
  cur = ""
  for ch in allc:
    t = ch.strip()
    if not cur:
      cur = t
    elif len(ci + cur + " " + t) <= WIDTH:
      cur = cur + " " + t
    else:
      out.append(ci + cur)
      cur = t
  tm = DICT_TAIL_RE.match(m.group("tail").strip() or "")
  closer = "}" + ((tm.group("comma") or "") if tm else "")
  if tm and tm.group("angle"):
    closer += ">" + (" {" if tm.group("brace") else "")
  if cur:
    out.append(ci + cur + closer)
  return out or None


def split_sig_line(line: str) -> list[str] | None:
  """Break an oversized `... = #attr<...>` / `!type<...>` line at its
  top-level commas. Returns None when not applicable/unbreakable."""
  if len(line) <= WIDTH or "<" not in line:
    return None
  m = SIG_ATTR_RE.match(line)
  if not m:
    return None
  inner = m.group("inner")
  if "<" in inner or ">" in inner:
    return None  # nested angle literals: leave untouched
  chunks, tail = split_sig_commas(inner)
  if not chunks:
    return None
  allc = [*chunks, tail] if tail.strip() else chunks
  ind = m.group("ind")
  ci = ind + CONT
  head = f"{ind}{m.group('pre')} = {m.group('sig')}<"
  out = []
  cur = head
  base = ind
  for ch in allc:
    t = ch.strip()
    joiner = "" if cur is head else " "
    if len(base + cur + joiner + t) <= WIDTH:
      cur = cur + joiner + t
    else:
      out.append(base + cur)
      cur, base = t, ci
  tail_txt = m.group("tail").strip()
  closer = ">" + ((" " + tail_txt) if tail_txt else "")
  if cur:
    if not cur.endswith(">"):
      cur = cur + closer
    out.append(base + cur)
  return out or None


def wrap_code_line(line: str) -> list[str]:
  match = re.match(r"[ \t]*", line)
  indent = match.group(0) if match else ""
  body = line[len(indent) :]
  if not body.strip() or body.rstrip().endswith("="):
    return [line]
  units = build_units(tokenize(body))
  w = Wrapper(indent)
  for u in units:
    if u["kind"] == "dict":
      w.dict_unit(u)
    else:
      w.append_unit(u)
  return w.finish() or [line]


def compact_json_dump(obj: Any) -> str:
  dump = json.dumps(obj, ensure_ascii=False, indent=2)

  def _compact(m):
    inner = " ".join(m.group(1).split())
    return "[" + inner + "]" if len(inner) <= 60 else m.group(0)

  prev = None
  while prev != dump:
    prev = dump
    dump = SHORT_ARRAY_RE.sub(_compact, dump)
  return dump


def wrap_comment(line: str) -> list[str]:
  if len(line) <= WIDTH:
    return [line]
  m = CASE_RE.match(line)
  if m:
    try:
      obj = json.loads(m.group(2))
    except Exception:
      return [line]
    dump = compact_json_dump(obj)
    out = []
    for idx, dl in enumerate(dump.splitlines()):
      marker = m.group(1) + "// " + ("case: " if idx == 0 else "    ")
      out.append(marker + dl)
    return out
  lead = line[: line.index("//") + 2]
  text = line[len(lead) :].strip()
  match = re.match(r"[ \t]*", lead)
  cont_lead = (match.group(0) if match else "") + "//"
  words = text.split()
  out = []
  cur = lead
  first = True
  for wd in words:
    cand = cur + ("" if first else " ") + wd
    if len(cand) <= WIDTH or first:
      cur = cand
    else:
      out.append(cur)
      cur = cont_lead + " " + wd
    first = False
  out.append(cur)
  return out


def format_lines(lines: list[str]) -> tuple[list[str], dict[str, int]]:
  out: list[str] = []
  stats = {"code": 0, "comment": 0, "skipped": 0}
  for ln in lines:
    if len(ln) <= WIDTH:
      out.append(ln)
      continue
    if ln.lstrip().startswith("//"):
      res = wrap_comment(ln)
      if res == [ln]:
        stats["skipped"] += 1
      else:
        stats["comment"] += 1
      out.extend(res)
      continue
    try:
      res = wrap_code_line(ln)
    except Exception:
      stats["skipped"] += 1
      out.append(ln)
      continue
    res2 = []
    for r in res:
      stage = split_sig_line(r) or [r]
      for s in stage:
        dic = split_dict_line(s)
        res2.extend(dic if dic else [s])
    if len(res2) == 1:
      stats["skipped"] += 1
    else:
      stats["code"] += 1
    out.extend(res2)
  return out, stats


def format_text(text: str) -> str:
  """Return ``text`` with every >100-column line wrapped in place.

  Idempotent: formatting already-formatted text is a no-op. Lines the
  wrapper cannot split (e.g. one giant string literal) pass through.
  """
  wrapped, _stats = format_lines(text.split("\n"))
  return "\n".join(wrapped)


def collect_mlir_files(paths: list[str]) -> list[Path]:
  """Expand files/directories into a sorted, de-duplicated *.mlir list."""
  found: list[Path] = []
  for item in paths:
    path = Path(item)
    if path.is_dir():
      found.extend(path.rglob("*.mlir"))
    else:
      found.append(path)
  return sorted(set(found))


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("paths", nargs="*", help="MLIR files or directories to format (default: examples/)")
  parser.add_argument(
    "--check", action="store_true", help="exit 1 listing files that would change, without writing"
  )
  args = parser.parse_args(argv)
  targets = collect_mlir_files(args.paths or ["examples"])
  changed = 0
  for path in targets:
    src = path.read_text(encoding="utf-8")
    new = format_text(src)
    if new == src:
      continue
    changed += 1
    if args.check:
      print(f"would reformat {path}")
    else:
      path.write_text(new, encoding="utf-8")
      print(f"formatted {path}")
  if args.check:
    return 1 if changed else 0
  print(f"{changed} file(s) reformatted")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
