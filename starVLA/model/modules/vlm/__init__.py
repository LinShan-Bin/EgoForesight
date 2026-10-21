# Copyright 2026 Chubin Zhang. All rights reserved.
# SPDX-License-Identifier: MIT


def get_vlm_model(config):
    """Build only the Qwen3.5 visual-language backbone."""
    from .QWen3_5 import _QWen3_5_VL_Interface

    return _QWen3_5_VL_Interface(config)
