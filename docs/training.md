<!-- Copyright 2026 Chubin Zhang. SPDX-License-Identifier: MIT -->
# Post-training, distribution and resume

Run from the repository root with `PYTHONPATH="$PWD"`. Configuration uses OmegaConf; dotted CLI overrides follow the native trainer (`--trainer.max_train_steps 3`, not `key=value`).

## Choosing a recipe

| YAML in `configs/posttrain/` | Data |
|---|---|
| `robotwin_released.yaml` | published RoboTwin checkpoint: full data, absolute EEF |
| `robocasa_released.yaml` | published RoboCasa GR1 checkpoint: full data |
| `robotwin_c2r.yaml` | clean-50 with mirrored variants; evaluate clean/randomized |
| `robotwin_full.yaml` | clean-50 + randomized-500 with mirrored variants |
| `robotwin_c2r_no_mirror.yaml` | clean-50, no mirror-language assets required |
| `robotwin_full_no_mirror.yaml` | clean-50 + randomized-500, no mirrors |
| `robocasa.yaml` | aligned GR1 tabletop 24-task corpus |
| `r1pro.yaml` | calibrated, emitted-IK R1Pro cache with full-update sampling |

Set `QWEN_MODEL`, `DINO_REPO`, `DINO_WEIGHTS`, `PRETRAINED_CHECKPOINT`; simulation recipes also need `DATA_ROOT`, R1Pro needs `R1PRO_CACHE` and `R1PRO_DATASET_ID`. Paths in the **saved** config must remain resolvable when serving the checkpoint on another machine. Only trusted local checkpoint/model files should be loaded.

## Released checkpoint training

`robocasa_released.yaml` and `robotwin_released.yaml` preserve the saved settings
of the linked Hugging Face checkpoints. Asset/data paths are configurable, and
`PRETRAINED_CHECKPOINT` points to the downloaded **EgoForesight-pretrain** local
weight file. No private pretraining recipes are included.

| Setting | RoboCasa GR1 | RoboTwin |
|---|---|---|
| Data mixture | `robocasa_1000_aug_aligned` | `robotwin_full_aug_aligned` |
| EEF representation | Relative | Absolute |
| Action chunk | 16 | 48 |
| Per-device batch | 16 | 16 |
| Global batch (saved configuration) | 512 | 512 |
| Original GPU geometry | 4 nodes × 8 GPUs | 2 nodes × 8 GPUs |
| Gradient accumulation | 1 | 2 |
| Global gradient clipping | Disabled | Disabled |
| Auxiliary AdamW clipping | 1.0 | 1.0 |
| Torch compile | Disabled | Enabled (`mot_blocks`) |

Use the dedicated DeepSpeed configuration below: the generic configuration has
global clipping enabled and does not match these runs. For example, on one
8-GPU machine (accumulation is recalculated to preserve global batch):

```bash
ACCELERATE_CONFIG=configs/distributed/zero2_released.yaml \
NUM_GPUS=8 BATCH_SIZE=16 GLOBAL_BATCH_SIZE=512 \
  bash scripts/posttrain.sh configs/posttrain/robocasa_released.yaml

ACCELERATE_CONFIG=configs/distributed/zero2_released.yaml \
NUM_GPUS=8 BATCH_SIZE=16 GLOBAL_BATCH_SIZE=512 \
  bash scripts/posttrain.sh configs/posttrain/robotwin_released.yaml
```

For the original multi-node geometry, set `NUM_NODES=4` (RoboCasa) or `2`
(RoboTwin), `NUM_GPUS=8`, the node-specific `NODE_RANK`, and `MASTER_ADDR`.
The archived RoboTwin run initially used global batch 256, then restarted with
512; the released recipe reflects the saved configuration, not a reconstruction
of that earlier batch transition. Generic relative-EEF and no-mirror RoboTwin
recipes remain available but are not the published checkpoint's recipe.

## Batch accounting

`scripts/posttrain.sh` computes:

```text
world_size = NUM_GPUS × NUM_NODES
accumulation = GLOBAL_BATCH_SIZE / (world_size × BATCH_SIZE)
```

The division must be exact. `NUM_GPUS` is GPUs **per node**; Accelerate's `--num_processes` receives the total. For multiple nodes, invoke the same recipe on every node, with a unique `NODE_RANK`, common `MASTER_ADDR` / `MASTER_PORT`, and identical data/model paths. Supply site-specific network/RDMA settings externally. No interface names or cluster launchers are hard-coded.

Mixed precision is owned by `configs/distributed/zero2.json`. Do not add a contradictory `mixed_precision` field to the Accelerate YAML. The setup uses ZeRO-2 and bf16. `torch_compile_modules: mot_blocks` compiles the block forward path without renaming checkpoint parameters; disable compilation for a fast initial diagnostic or unsupported GPU/runtime.

## Retained corrections

- Rank-aware configuration/statistics writes, training metrics and checkpoint ownership; collective optimizer/ZeRO save and restore run on all required ranks.
- Resume validates saved normalization rather than overwriting it with a different dataset/statistics computation. Canonical masks and normalization are persisted with the run.
- Global clipping-input norms are reduced and logged consistently, including invalid-norm checks and the actual clipping coefficient.
- Logging reports optimizer updates and accumulation correctly; LR summaries avoid dumping every shape-specific Muon group. JSONL works without network telemetry.
- Model and EMA weights are saved separately. EMA is restored only after the student has its prepared device/dtype.
- Step-indexed optimizer metadata prevents restoring the wrong optimizer state for a pinned model checkpoint. Scheduler state and data position are restored consistently with the saved step.
- The R1Pro sampler truncates an epoch to complete global optimizer updates, rather than ending with a short accumulation update.

Implementation: `training/train_starvla.py` and `training/trainer_utils/{trainer_tools,optimizers,muon_runtime,muon_fused,muon_ns,config_tracker,overwatch}.py` beneath `starVLA/`.

## Muon

The default optimizer is DeepSpeed Muon with auxiliary AdamW groups. Suitable 2-D hidden weights use Muon; embeddings, biases/norms and output heads use the configured AdamW treatment. Parameter roles and fused projection layouts are explicit, rather than blindly treating every matrix the same.

The release retains momentum 0.95, Gram Newton–Schulz, shape-aware `match_rms_adamw` LR scaling, fused QKV/projection handling, and separate auxiliary-AdamW clipping (default norm 1). `ns_kernel: hybrid` selects the optimized Quack/CUTLASS Gram path where appropriate and DeepSpeed otherwise. The import compatibility shim is scoped to the relevant Quack import.

For an unsupported fast-kernel environment, `--trainer.optimizer.ns_kernel deepspeed` selects the native implementation; do not claim this is the exact speed/numerical path of the hybrid kernel path. Similarly, `--framework.qwenvl.attn_implementation sdpa` is an explicit diagnostic alternative, not the published fast-kernel configuration.

## Checkpoints

A run contains, among other files:

```text
config_full.yaml
dataset_statistics.json
dataset_manifest.json                  # for R1Pro
checkpoints/steps_<N>_pytorch_model.pt  # lightweight model state dict
checkpoints/steps_<N>_ema_teachers.pt   # actual training EMA
checkpoints/latest_optimizer_state/   # step-tagged native resume state / ZeRO shards
final_model/pytorch_model.pt          # when final save is enabled
```

Keep all shards/metadata for a resume point. The lightweight model file alone is sufficient for a **warm start or inference**, not an exact optimizer continuation. Despite the legacy `latest_optimizer_state` directory name, state is step-tagged; monitor disk usage. Checkpoint writes must finish before stopping/copying a run.

- **Warm start:** `trainer.is_resume=false` and `trainer.pretrained_checkpoint=/path/model.pt`; optimizer/schedule start anew.
- **Resume same run:** same `run_root_dir`, `run_id`, recipe and data; set `trainer.is_resume=true`.
- **Pinned resume:** also set `trainer.resume_checkpoint` to `steps_<N>_pytorch_model.pt` **inside that run's checkpoint directory**. Matching optimizer metadata/state is required; arbitrary external files are rejected.
- **EMA inference:** server option `--ema_teacher foresight_forcing` loads the paired sidecar. Alternatively `python -m scripts.export_mot_ema_policy /path/steps_<N>_ema_teachers.pt` exports a separately named policy; it does not overwrite online weights.

The inherited trainer does not preserve every RNG state for an uninterrupted bitwise replay. Do not equate successful resume with guaranteed bitwise equivalence to a never-interrupted run, especially with asynchronous data loading/accelerated kernels. The short-run audit controls RNG separately to compare the extracted and original code paths.
