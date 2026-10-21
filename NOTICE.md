<!-- Copyright 2026 Chubin Zhang. SPDX-License-Identifier: MIT -->
# Copyright, licenses and provenance

**EgoForesight additions and modifications: Copyright (c) 2026 Chubin Zhang.** The root [LICENSE](LICENSE) applies to those additions. It is **not a blanket relicensing of inherited code, third-party assets, datasets or model weights**. Existing file-level notices and the applicable upstream terms remain in force.

## Retained code

| Origin | Retained use / scope | Attribution and license text |
|---|---|---|
| [StarVLA](https://github.com/starVLA/starVLA) | Framework/training scaffolding, model interfaces, loaders, serving and evaluation adaptations | StarVLA Team / starVLA community; named contributions by Jinhui YE and Fangjing Wang are retained. [Inherited license](licenses/StarVLA-LICENSE.txt). |
| [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `starVLA/dataloader/gr00t_lerobot/` portions with NVIDIA notices; GR1 simulator evaluation and wrappers | Copyright 2025 NVIDIA Corporation & Affiliates. [Apache-2.0](licenses/Apache-2.0.txt). |
| [Hugging Face Transformers / Qwen](https://github.com/huggingface/transformers) | Qwen3.5 vision-attention portion of `starVLA/model/modules/vlm/qwen35_compute.py` | Copyright 2025 The Qwen Team and The HuggingFace Inc. team. [Transformers license](licenses/Transformers-LICENSE.txt). The surrounding integration keeps its separate notices. |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | Rotation transformer adaptation in `gr00t_lerobot/transform/state_action.py`; the original source revision remains in its docstring | Copyright 2023 Columbia Artificial Intelligence and Robotics Lab. [MIT](licenses/DiffusionPolicy-LICENSE.txt). |
| [OpenVLA / Prismatic](https://github.com/openvla/openvla) | `starVLA/training/trainer_utils/overwatch.py`, with the retained StarVLA logging modifications | Copyright 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti. [MIT](licenses/OpenVLA-LICENSE.txt). |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | NumPy MsgPack codec in `deployment/model_server/tools/msgpack_numpy.py`; original codec logic retained | Physical Intelligence / OpenPI contributors. [Apache-2.0](licenses/Apache-2.0.txt). |
| [msgpack-numpy](https://github.com/lebedov/msgpack-numpy) | NumPy serialization strategy acknowledged by the OpenPI codec; object/pickle fallback is not retained | Copyright 2013–2022 Lev E. Givon. [BSD-3-Clause](licenses/msgpack-numpy-LICENSE.txt). |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | Patched `third_party/robotwin_client/robotwin_replace_file/{_base_task,eval_policy}.py` | Copyright 2025 Tianxing Chen (陈天行). [MIT](licenses/RoboTwin-LICENSE.txt). |

The inherited StarVLA license includes additional **rebasing and commit-attribution wording**. It is copied verbatim, rather than replaced with a standard MIT template. Preserve that notice and the original author notices when redistributing the corresponding code. This directory contains no nested Git repository and does not claim that upstream authors endorse EgoForesight.


## External dependencies and assets

Qwen3.5, DINOv3, DeepSpeed/Muon, FlashAttention, FLA, Quack/CUTLASS, PyTorch, LeRobot-related dependencies, robosuite/RoboCasa and other installed packages retain their own licenses. External source trees and model weights are **not bundled**. Dependency pins describe the supported dependency versions, not permission to redistribute those assets under the root MIT license.

The official RoboTwin and NVIDIA GR00T Teleop-Sim datasets, simulator meshes/textures, R1Pro recordings, calibration files and robot URDFs have separate provenance and permissions. No such dataset, pretrained checkpoint or private asset is included. The release's conversion code does not change the source data's terms.

## Paper illustrations

The README includes web renditions of figures and task images from the accompanying **EgoForesight: Effectively Scaling Policy Learning with Noisy Egocentric Actions** paper. Original artwork and source-image attribution remain with their respective sources; the code license does not replace any applicable source-image terms. Figure export, task-panel layout, and repository documentation: **Chubin Zhang**. These illustrations are not a release of the underlying recordings or datasets.
