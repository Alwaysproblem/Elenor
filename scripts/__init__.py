"""Repo-local utility scripts (formatters, exporters).

Currently exposes ``format_mlir`` -- the token-preserving 100-column MLIR
wrapper used by the pre-commit ``format-mlir`` hook and by the transformer
workload generators' ``write_workload``.
"""

from __future__ import annotations
