"""Shared W6 scenario IO: fail-closed loader, vector serializer, seal receipts.

The C033 and C034 unit runners consume their nine scenario YAMLs identically:
validate exactly one row per case id with strict unsigned-word mappings, then
materialize sorted a..i vector rows and hash the C015 contract fragments
against the real C020 seal. This module is that narrow shared functionality;
it owns no geometry, filenames or verdict rules.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

WORD_MASK = (1 << 64) - 1


def word(value: Any, where: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= WORD_MASK:
    raise AssertionError(f"{where}: expected one unsigned 64-bit word, got {value!r}")
  return value


def load_scenarios(
  directory: Path,
  pattern: str,
  case_ids: Sequence[str],
  seed_keys: tuple[str, ...],
  expect_keys: tuple[str, ...],
  *,
  seed_name: str = "seed",
) -> list[dict[str, Any]]:
  """Validate exactly one row per case id (sorted a..i); both mappings strict words."""
  rows: list[dict[str, Any]] = []
  for path in sorted(directory.glob(pattern)):
    doc = yaml.safe_load(path.read_text())
    where = f"{path}:"
    if not isinstance(doc, dict) or set(doc) != {"case", seed_name, "expect"}:
      raise AssertionError(f"{where} expected exactly case/{seed_name}/expect, got {doc!r}")
    if doc["case"] not in case_ids:
      raise AssertionError(f"{where} case {doc['case']!r} is outside {case_ids[0]}..{case_ids[-1]}")
    seed_doc, expect_doc = doc[seed_name], doc["expect"]
    if not isinstance(seed_doc, dict) or set(seed_doc) != set(seed_keys):
      raise AssertionError(f"{where} {seed_name} must have exactly {seed_keys}")
    if not isinstance(expect_doc, dict) or set(expect_doc) != set(expect_keys):
      raise AssertionError(f"{where} expect must have exactly {expect_keys}")
    rows.append({
      "case": doc["case"],
      "file": path,
      seed_name: {key: word(seed_doc[key], f"{where} {seed_name}.{key}") for key in seed_keys},
      "expect": {key: word(expect_doc[key], f"{where} expect.{key}") for key in expect_keys},
    })
  ids = [row["case"] for row in rows]
  if sorted(ids) != list(case_ids):
    raise AssertionError(f"{directory}: {pattern} must yield each of {list(case_ids)} once, got {ids}")
  return sorted(rows, key=lambda row: row["case"])


def vectors(
  rows: Sequence[dict[str, Any]],
  seed_keys: tuple[str, ...],
  expect_keys: tuple[str, ...],
  *,
  seed_name: str = "seed",
) -> str:
  """One lowercase-hex word row per case: seed words then expected words."""
  lines = []
  for row in rows:
    words = [row[seed_name][key] for key in seed_keys]
    words += [row["expect"][key] for key in expect_keys]
    lines.append(" ".join(f"{word_value:x}" for word_value in words))
  return "".join(f"{line}\n" for line in lines)


def fragment_receipts(fragments: Sequence[Path], seal: Path, repo: Path) -> dict[str, str]:
  """Hash each fragment and cross-check it against the real C020 seal inputs."""
  inputs = json.loads(seal.read_text())["inputs"]
  receipts = {}
  for path in fragments:
    rel = str(path.relative_to(repo))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert inputs.get(rel) == digest, f"{rel}: {digest} drifted from the C020 seal"
    receipts[rel] = digest
  return receipts
