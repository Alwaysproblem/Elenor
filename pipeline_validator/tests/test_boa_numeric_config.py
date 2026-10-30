"""P06 rejects unsupported numerical semantics before any artifact is built."""

from __future__ import annotations

import pytest
import yaml

from pipeline_validator.config import HardwareConfig


@pytest.mark.parametrize(
  ("key", "value"),
  [
    ("numeric_profile", "binary64"),
    ("numeric_profile", None),
    ("numeric_profile", False),
    ("numeric_tile_k", 0),
    ("numeric_tile_k", -1),
    ("numeric_tile_k", True),
    ("numeric_tile_k", 8.0),
    ("numeric_tile_k", "8"),
    ("numeric_tile_k", 1 << 64),
  ],
)
def test_invalid_numeric_configuration_is_not_loaded(tmp_path, key, value):
  path = tmp_path / "hardware.yaml"
  path.write_text(yaml.safe_dump({"engines": {"boa": {key: value}}}), encoding="utf-8")
  with pytest.raises(ValueError, match=key):
    HardwareConfig.from_yaml(path)
