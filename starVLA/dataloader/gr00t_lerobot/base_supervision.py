# Copyright 2026 Chubin Zhang. All rights reserved.
# SPDX-License-Identifier: MIT

"""Public datasets have no private episode-specific base-label overrides."""
from pathlib import Path

BaseSupervisionPolicy = bool | frozenset[int] | None
BASE_SUPERVISION_POLICY_BY_DATASET: dict[str, BaseSupervisionPolicy] = {}


def get_base_supervision_policy(data_name: str) -> BaseSupervisionPolicy:
    """Return the base-label policy for a dataset path or basename."""
    return BASE_SUPERVISION_POLICY_BY_DATASET.get(Path(data_name).name)


def is_base_supervision_enabled(policy: BaseSupervisionPolicy, task_index: int | None) -> bool:
    """Resolve a full-dataset or task-filtered policy for one sample."""
    if policy is True:
        return True
    if not policy or task_index is None:
        return False
    return int(task_index) in policy
