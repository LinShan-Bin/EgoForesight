# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# EgoForesight modifications Copyright 2026 Chubin Zhang.

from enum import Enum
from numbers import Integral


class EmbodimentTag(Enum):
    GR1 = "gr1"
    ROBOCASA_GR1_EEF = "robocasa_gr1_eef"
    ROBOTWIN_AUG_ALIGNED = "robotwin_aug_aligned"
    ROBOCASA_GR1_AUG_ALIGNED = "robocasa_gr1_aug_aligned"
    NEW_EMBODIMENT = "new_embodiment"
    ROBOTWIN = "robotwin"


EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.ROBOCASA_GR1_EEF.value: 24,
    EmbodimentTag.ROBOCASA_GR1_AUG_ALIGNED.value: 24,
    EmbodimentTag.ROBOTWIN_AUG_ALIGNED.value: 27,
    EmbodimentTag.ROBOTWIN.value: 27,
}


def embodiment_tag_to_index(tag: int | str | EmbodimentTag) -> int:
    """Resolve the action-projector index without relying on enum order."""

    if isinstance(tag, Integral):
        return int(tag)
    if isinstance(tag, EmbodimentTag):
        value = tag.value
    else:
        raw = str(tag)
        if raw in EMBODIMENT_TAG_MAPPING:
            value = raw
        else:
            try:
                value = EmbodimentTag[raw.upper()].value
            except KeyError as exc:
                raise KeyError(f"Unknown embodiment tag {tag!r}") from exc
    try:
        return int(EMBODIMENT_TAG_MAPPING[value])
    except KeyError as exc:
        raise KeyError(f"Embodiment tag {value!r} has no projector mapping") from exc


ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "custom_robot_config": EmbodimentTag.NEW_EMBODIMENT,
    "fourier_gr1_arms_waist": EmbodimentTag.GR1,
    "fourier_gr1_arms_waist_absolute": EmbodimentTag.GR1,
    "robocasa_gr1_eef": EmbodimentTag.ROBOCASA_GR1_EEF,
    "robocasa_gr1_aug_aligned": EmbodimentTag.ROBOCASA_GR1_AUG_ALIGNED,
    "my_robotwin": EmbodimentTag.ROBOTWIN,
    "robotwin_aug_aligned": EmbodimentTag.ROBOTWIN_AUG_ALIGNED,
}
