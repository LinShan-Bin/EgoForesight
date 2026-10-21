<!-- Copyright 2026 Chubin Zhang. SPDX-License-Identifier: MIT -->
# Model and method

## What the extraction preserves

`starVLA/model/framework/Qwen35MoT3.py` is the external model API. `_mot_base.py` contains the shared flow/data plumbing needed by MoT3; it is not an additional public model variant. `modeling_mot3.py` and `layers.py` implement the two-stream transformer. Unrelated action experts, VLMs, pretrained action codecs and pretraining dataset dispatch are not part of this tree.

The three original aligned FF3 stage launchers share this model. Stage-specific initialization, data mixtures, schedules and private pretraining orchestration are not published. `configs/model/qwen35_mot3_ff3.yaml` records the retained architecture; post-training YAMLs are self-contained and use the teacher immediately. Original pretraining's delayed representation-loss start can still be represented by `representation_start_step` / `representation_warmup_steps`; this is not a released pretraining recipe.

## Structure

1. Qwen3.5-4B encodes the current image canvas and task. Features from layers 4, 8, …, 32 condition successive pairs of MoT blocks.
2. Frozen DINOv3 ViT-B/16 encodes visual flow targets; spatial pooling is 2×2. Future frames are training targets, never policy inputs at deployment.
3. A vision stream and an action stream share joint attention and maintain separate projections/feed-forward paths. Canonical state and metadata condition the model.
4. Rectified flow predicts action and future-vision velocities; Euler integration starts at noise and integrates toward clean outputs. The convention is **clean = t=0**, **noise = t=1**, target velocity `noise - clean`.
5. Foresight Forcing samples modality-specific noise levels. With the retained schedule, ordinary action time is no cleaner than vision time. Timesteps have a 0.005 floor, except the explicit clean-vision edge case. An EMA MoT teacher provides a less-noisy action representation target; its action input is not silently changed to clean ground-truth actions.

| Setting | Released post-training recipes |
|---|---|
| MoT hidden size / depth / heads | 1536 / 16 / 24 |
| Vision / action FFN width | 6144 / 2048 |
| Qwen context / DINO feature width | 2560 / 768 |
| Timestep and AdaLN rank | 128 |
| Action and state width | 130 canonical slots |
| State tokenization | one token; MLP hidden size 512 |
| Foresight alpha bounds | 0.5–3.0 |
| Teacher action-time ratio | uniform [0, 1] in the pre-floor domain |
| Teacher EMA decay | 0.999 |
| Student / teacher feature layers | 5 / 11 (configuration indices) |
| Representation projector / loss weight | 3072 / 0.05 |
| Clean-vision teacher-edge probability | 0.02 |
| Visual flow loss weight | 0.1 |
| Default inference flow steps | 4 |
| Action horizon | RoboTwin 48; RoboCasa and R1Pro 16 |

The `flow.beta_*` settings describe the generic flow sampler; the enabled Foresight Forcing path has its own coupled schedule. Do not infer that both modalities independently use the generic beta sampler.

## Representation and compatibility

- EEF actions are camera-spatial relative poses; joint targets remain absolute. Validity/schema masks distinguish unsupported slots from actual zero-valued actions.
- Training and deployment share the same slot ordering, quaternion/rotvec conventions, camera frame, normalization statistics and image canvas. Do not independently renormalize outputs in a client.
- The image canvas is head 256×320 above two 128×160 wrists, giving 384×320; the single-camera RoboCasa path uses its configured view role.
- `Qwen35MoT3`, `mot_model.*`, and the existing `starVLA` namespace are checkpoint interfaces and are not renamed.
- The legacy trainer callback `update_self_flow_teacher` is retained because it also invokes the **Foresight Forcing EMA update**. It does not expose a separate Self-Flow training method in this release.
- R1Pro retains one historical opaque `dataset_family` identity string because metadata is hashed into learned conditioning. It is not a path, a loader, or a bundled private dataset.

Teacher export replaces only matching `mot_model.*` tensors. Qwen/backbone weights remain those of the paired online checkpoint. Do not combine an EMA sidecar with a checkpoint from a different step.
