"""Deeply immutable, deterministic values shared by compiler and executor."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


# ``Generic[...]`` is required while the supported runtime remains Python 3.11.
class FrozenMap(Mapping[K, V], Generic[K, V]):  # noqa: UP046
  """An owned mapping; input containers cannot mutate a published value."""

  __slots__ = ("__data",)
  __data: Mapping[K, V]

  def __init__(self, values: Mapping[K, V] | None = None):
    if values is None:
      values = {}
    if not isinstance(values, Mapping):
      raise TypeError("FrozenMap requires a mapping")
    owned = {freeze(key): freeze(value) for key, value in values.items()}
    object.__setattr__(self, "_FrozenMap__data", MappingProxyType(owned))

  def __getitem__(self, key: K) -> V:
    return self.__data[key]

  def __iter__(self) -> Iterator[K]:
    return iter(self.__data)

  def __len__(self) -> int:
    return len(self.__data)

  def __setattr__(self, name, value):
    raise AttributeError("immutable mapping")

  def __deepcopy__(self, memo):
    return self

  def __repr__(self) -> str:
    return f"FrozenMap({self.__data!r})"


def freeze(value):
  """Return an owned, transitively immutable representation of ``value``."""
  if isinstance(value, FrozenMap):
    return value
  if isinstance(value, Mapping):
    return FrozenMap(value)
  if isinstance(value, tuple):
    items = tuple(freeze(item) for item in value)
    return value if all(old is new for old, new in zip(value, items)) else items
  if isinstance(value, list):
    return tuple(freeze(item) for item in value)
  if isinstance(value, (set, frozenset)):
    return frozenset(freeze(item) for item in value)
  if is_dataclass(value) and not isinstance(value, type):
    params = getattr(type(value), "__dataclass_params__", None)
    if not params or not params.frozen:
      raise TypeError(f"mutable dataclass cannot be frozen: {type(value).__name__}")
    updates = {item.name: freeze(getattr(value, item.name)) for item in fields(value)}
    if any(updates[item.name] is not getattr(value, item.name) for item in fields(value)):
      return replace(value, **updates)
    return value
  if value is None or isinstance(value, Enum) or type(value) in (str, int, bool, float):
    return value
  raise TypeError(f"unsupported mutable value: {type(value).__name__}")


class FrozenRecord:
  """Dataclass post-init normalization for immutable execution containers."""

  __slots__ = ()

  def __post_init__(self):
    for item in fields(self):
      object.__setattr__(self, item.name, freeze(getattr(self, item.name)))


def canonical_value(value):
  """JSON-compatible structural values, independent of mapping insertion order."""
  if isinstance(value, Enum):
    return value.value
  if is_dataclass(value) and not isinstance(value, type):
    return {f.name: canonical_value(getattr(value, f.name)) for f in fields(value)}
  if isinstance(value, Mapping):
    return {str(k): canonical_value(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)):
    return [canonical_value(v) for v in value]
  if value is None or isinstance(value, (str, bool, int)):
    return value
  if isinstance(value, float) and math.isfinite(value):
    return value
  raise ValueError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value) -> str:
  return json.dumps(canonical_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
  return hashlib.sha256(canonical_json(value).encode()).hexdigest()
