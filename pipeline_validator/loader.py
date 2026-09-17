"""Read-only loading of verified compiled executables."""

from __future__ import annotations

from collections.abc import Mapping

from .compiled_program import CompiledProgram, LoadedProgram
from .config import HardwareConfig, SimConfig
from .execution_ir import GlobalBinding
from .execution_verifier import target_fingerprint, verify_actual_bindings, verify_compiled_program
from .immutable import FrozenMap


def load_program(
  program: CompiledProgram,
  hw: HardwareConfig,
  sim: SimConfig,
  *,
  actual_bindings: Mapping[str, GlobalBinding] | None = None,
) -> LoadedProgram:
  """Validate and bind an immutable artifact without compiling or rewriting it."""
  verify_compiled_program(program, hw, sim)
  bindings = FrozenMap(actual_bindings or {})
  verify_actual_bindings(program, bindings, hw)
  return LoadedProgram(compiled=program, actual_bindings=bindings, target_hash=target_fingerprint(hw, sim))
