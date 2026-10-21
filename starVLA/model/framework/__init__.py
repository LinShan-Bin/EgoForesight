# Copyright 2026 Chubin Zhang. All rights reserved.
# SPDX-License-Identifier: MIT

"""The public release intentionally registers only Qwen35MoT3."""
from typing import Any

from starVLA.model.tools import FRAMEWORK_REGISTRY


def build_framework(cfg: Any) -> Any:
    """Build EgoForesight while preserving the original ``cfg`` keyword API."""
    from starVLA.model.framework.Qwen35MoT3 import Qwen35MoT3

    if cfg.framework.name != "Qwen35MoT3":
        raise ValueError(f"Unsupported framework {cfg.framework.name!r}; expected Qwen35MoT3")
    return Qwen35MoT3(cfg)
