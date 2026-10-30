"""Deterministic golden documents; source identity excludes generated outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import GENERATOR_VERSION, PROFILE
from .model_cases import generate_conv_cases, generate_epilogue_cases, generate_matmul_cases
from .scalar_cases import SPECIAL_WORDS, bf16_boundary_cases, directed_cases, evaluate, result_dict, vector


def generator_hash() -> str:
  package = Path(__file__).resolve().parent
  reference = package.parent
  digest = hashlib.sha256()
  for path in sorted([*package.glob("*.py"), reference / "cli"]):
    digest.update(path.relative_to(reference).as_posix().encode() + b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
  return digest.hexdigest()


def document(kind: str, cases: list[dict], seed: int, *, tile_k: int | None = None) -> dict:
  return {
    "schema": 1,
    "profile": PROFILE,
    "generator_version": GENERATOR_VERSION,
    "generator_hash": generator_hash(),
    "kind": kind,
    "seed": seed,
    "parameters": {} if tile_k is None else {"tile_k": tile_k},
    "cases": cases,
  }


def primitive_golden_cases() -> list[dict]:
  candidates = [*directed_cases(), *bf16_boundary_cases()]
  for op in ("bf16", "i32"):
    for index, word in enumerate(SPECIAL_WORDS):
      candidates.append(vector(f"G-{op}-{index}", op, word, category="special-conversion"))
  seen = set()
  result = []
  for item in candidates:
    key = tuple(item[name] for name in ("op", "a", "b", "c"))
    if key not in seen:
      seen.add(key)
      result.append({**item, "expected": result_dict(item["op"], evaluate(item))})
  return result


def generate_documents(
  seed: int, tile_k: int = 8, kinds: tuple[str, ...] = ("primitives", "matmul", "epilogue", "conv")
) -> dict[str, dict]:
  if type(seed) is not int or type(tile_k) is not int or tile_k < 1:
    raise ValueError("seed must be an integer and tile_k must be a positive integer")
  factories = {
    "primitives": lambda: document("primitives", primitive_golden_cases(), seed),
    "matmul": lambda: document("matmul", generate_matmul_cases(seed, tile_k), seed, tile_k=tile_k),
    "epilogue": lambda: document("epilogue", generate_epilogue_cases(seed), seed),
    "conv": lambda: document("conv", generate_conv_cases(seed, tile_k), seed, tile_k=tile_k),
  }
  if any(kind not in factories for kind in kinds):
    raise ValueError(f"unknown golden document kinds: {kinds}")
  return {f"{kind}.json": factories[kind]() for kind in kinds}


def encoded(document_value: dict) -> bytes:
  return (json.dumps(document_value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_documents(destination: Path, documents: dict[str, dict]) -> dict[str, str]:
  destination.mkdir(parents=True, exist_ok=True)
  hashes = {}
  for filename, content in documents.items():
    data = encoded(content)
    (destination / filename).write_bytes(data)
    hashes[filename] = hashlib.sha256(data).hexdigest()
  return hashes
