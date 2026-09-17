"""Explicit source-to-executable compilation.

Runtime and Loader modules must not import this package.  ``compile_program``
is the sole public compiler API; diagnostic helpers remain in ``compiler.api``.
"""

from __future__ import annotations

from .api import compile_program

__all__ = ("compile_program",)
