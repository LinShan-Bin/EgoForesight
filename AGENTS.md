<!-- Copyright 2026 Chubin Zhang. SPDX-License-Identifier: MIT -->
# EgoForesight agent and contributor guide

This guide applies to this directory and its descendants. Use it as the project root, including when it is nested in another checkout. The public-facing overview belongs in [README.md](README.md); engineering details live in [docs/](docs/).

## Scope and compatibility

- Keep the release standalone: imports, launchers, and configuration must not depend on a parent repository.
- Maintain the released MoT3 / Foresight Forcing model, post-training, official RoboTwin / RoboCasa data preparation, evaluation dashboard, and R1Pro data/deployment paths.
- Pretraining launchers and data pipelines are intentionally excluded. Do not reintroduce private recordings, internal configurations, cluster-specific settings, pretrained weights, calibration/URDF assets, unrelated frameworks, or experimental model branches.
- Retain the `starVLA` Python namespace, `Qwen35MoT3` registry name, and checkpoint parameter names such as `mot_model.*`. They are compatibility interfaces, not naming cleanups.
- Preserve canonical action/state slot ordering, masks, coordinate frames, image layout, and normalization between training and deployment. Future frames are training targets, not deployment inputs.
- RoboCasa here means the **24-task GR1 tabletop suite**, not RoboCasa365. RoboTwin support targets **RoboTwin 2.0**.

## Repository map

| Path | Responsibility |
|---|---|
| `configs/model/` | Retained architecture specification; not a pretraining recipe |
| `configs/posttrain/` | RoboCasa, RoboTwin clean-to-randomized/full, and R1Pro recipes |
| `configs/distributed/` | Accelerate and DeepSpeed ZeRO-2 configuration |
| `starVLA/model/` | MoT3, Qwen3.5, DINOv3 targets, canonical action decoding |
| `starVLA/dataloader/` | Simulation and R1Pro loaders and transforms |
| `starVLA/training/` | Native trainer; distributed, Muon, logging, and checkpoint utilities |
| `scripts/` | Data conversion, post-training launcher, EMA export, and dashboard |
| `examples/Robocasa_tabletop/` | GR1 simulation adapters and rollout runner |
| `third_party/robotwin_client/` | RoboTwin adapter and simulator patches |
| `deployment/model_server/` | Benchmark WebSocket server and separate R1Pro protocol server |
| `assets/` | Paper-derived README illustrations and their provenance |

## Environment and configuration

Follow [environment setup](docs/environment.md) and the README installation commands. The training path uses Linux, Python 3.10, CUDA, and the pinned dependencies; data conversion and simulators have separate environments. Do not assume a local CPU-only import proves GPU runtime compatibility.

Run from the release root with `PYTHONPATH="$PWD"`. Configure local paths through [.env.example](.env.example):

- All post-training recipes: `QWEN_MODEL`, `DINO_REPO`, `DINO_WEIGHTS`, `PRETRAINED_CHECKPOINT`.
- Simulation: `DATA_ROOT`, pointing at prepared aligned data with the required statistics.
- R1Pro: `R1PRO_CACHE` and `R1PRO_DATASET_ID`.
- Evaluation: `POLICY_PYTHON`, `ROBOTWIN_ROOT`, `ROBOTWIN_PYTHON`, and/or `ROBOCASA_PYTHON` as appropriate.

Only load trusted checkpoint/model files. Paths saved in run configurations must still resolve when moving a run to another machine. The README links the base and benchmark checkpoints and their corresponding training configurations.

Configuration uses OmegaConf and dotted CLI overrides: `--trainer.max_train_steps 3`, **not** `trainer.max_train_steps=3`. Use `DRY_RUN=1` with `scripts/posttrain.sh` to inspect the launch command without starting a job.

## Development conventions

- Python 3.10+, type hints on public interfaces, and concise docstrings. Use Black's preview mode and 121-character lines; follow the Ruff configuration in `pyproject.toml`. Format/check changed files rather than sweeping unrelated or vendored code.
- Keep imports ordered standard library → third-party → local. Use PascalCase classes, snake_case functions/configuration keys, and uppercase constants.
- Frameworks expose the model API; loaders return model-agnostic observations, language, actions, state, and required masks/metadata. Do not move model preprocessing into data conversion.
- Use Accelerate logging for training and descriptive errors for invalid configurations or missing assets. Preserve rank-aware writes; do not add unconditional per-rank logging.
- Use explicit tensor dtypes/devices, appropriate autocast, and `torch.inference_mode()` for inference. Do not silently switch kernels, attention implementations, or coordinate conventions while claiming unchanged results.

## Development checks

Prefer a few real training updates and same-checkpoint inference over a large collection of synthetic smoke tests. For documentation-only edits, check links, rendered figures, and packaging metadata; no GPU run is needed.

After preparing the environment, checkpoint, and data, a small native run is:

```bash
NUM_GPUS=2 BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 \
  bash scripts/posttrain.sh configs/posttrain/robocasa.yaml \
  --run_id robocasa_short --trainer.max_train_steps 3 \
  --trainer.num_warmup_steps 1 --trainer.save_interval 2 \
  --trainer.logging_frequency 1 --trainer.use_torch_compile false
```

Use a complete dataset or explicitly change the mixture. A one-episode conversion is useful for checking a converter, but it does not satisfy a multi-task training recipe. Write diagnostic data/statistics to separate locations so they do not replace a complete corpus's outputs.

For model, loader, or trainer changes, compare against a known-good implementation with the **same checkpoint, real samples, configuration, dtype, and RNG state**:

1. Compare losses, gradient/clipping metrics, and selected updated weight/EMA tensors over a few native optimizer updates.
2. Compare predicted actions and future-vision features on the same inference samples; include denormalized outputs when changing action handling.
3. Exercise checkpoint save/resume when changing trainer state. Check raw-to-aligned outputs when changing a converter.
4. If results differ, investigate the affected path before expanding the test surface. Record the runtime, tolerances, and actual observations rather than claiming broad equivalence.


## Distributed training, Muon, and checkpoints

See [training](docs/training.md) for the complete behavior. In particular:

- `NUM_GPUS` is per node. `world_size = NUM_GPUS × NUM_NODES`; `GLOBAL_BATCH_SIZE` must be divisible by `world_size × BATCH_SIZE`. The launcher computes accumulation and passes the total process count to Accelerate.
- Every node uses the same data/model paths and schedule, with its own `NODE_RANK`. Supply network/RDMA settings externally; do not hard-code a cluster interface.
- DeepSpeed's configuration owns mixed precision. Keep collective optimizer/ZeRO save/restore calls on all required ranks while preserving main-rank ownership of ordinary files.
- Preserve Muon parameter-role grouping, fused-projection handling, shape-aware LR scaling, and auxiliary AdamW clipping. A fallback such as `--trainer.optimizer.ns_kernel deepspeed` is an explicit runtime change, not the same hybrid kernel path.
- Keep saved normalization, canonical masks, and dataset identity consistent. Resume must not overwrite the original run's statistics with a new dataset's estimates.

A run includes:

```text
config.yaml / config_full.yaml
dataset_statistics.json
checkpoints/steps_<N>_pytorch_model.pt
checkpoints/steps_<N>_ema_teachers.pt
checkpoints/latest_optimizer_state/   # native resume metadata and ZeRO state
```

Unlike a weights-only export, the trainer **does save native optimizer/resume state**. Keep all matching shards and metadata.

- **Warm start:** `trainer.is_resume=false` with `trainer.pretrained_checkpoint`; optimizer and schedule start anew.
- **Resume:** rerun the short command above with `--trainer.is_resume true`, keeping the same run ID/root, data, batch geometry, and total schedule. For an interruption check, stop only after the step-2 model and all sidecars have finished writing.
- **Pinned resume:** add `--trainer.resume_checkpoint /absolute/path/to/run/checkpoints/steps_<N>_pytorch_model.pt`; it must belong to that run and have matching native state.
- **EMA inference:** use `--ema_teacher foresight_forcing` with the paired sidecar, or export it with `python -m scripts.export_mot_ema_policy /path/to/steps_<N>_ema_teachers.pt`. Do not overwrite online weights or mix steps.

The lightweight model file supports warm starts and inference with the required configuration/statistics, but not optimizer continuation by itself. Successful resume is not a guarantee of uninterrupted bitwise replay; not every RNG state is persisted.

## Evaluation and robot deployment

Use [evaluation](docs/evaluation.md) for simulator installation, dashboard launch, and the policy server. The dashboard's `--directory` is the **parent of experiment directories**, not a checkpoint file. Start with a small task selection and inspect logs/videos before launching a full benchmark. Keep the administration UI on localhost or behind an SSH tunnel.

The benchmark WebSocket transport and the R1Pro protocol are different. Preserve each adapter's action frame, normalization, state representation, and control mode. Model-side tests are not authorization to move hardware; follow the [R1Pro deployment checks](docs/r1pro.md) before any supervised real-robot run.

## Documentation and attribution

- Keep internal release-verification scripts, environment snapshots, validation reports and diagnostic logs out of the distributed tree.

- Keep public method explanations and paper results in the README; keep operational details in `docs/`. Distinguish paper benchmark numbers from local release checks, and do not invent checkpoint links, authors, or publication metadata.
- Preserve upstream copyright/license notices. Credit new EgoForesight code and documentation to **Chubin Zhang** with the existing copyright/SPDX style; see [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
- Include new distributable documents/assets in `MANIFEST.in` as appropriate. Preserve paper-image attribution in `NOTICE.md`; do not copy private LaTeX notes, unrelated drafts, or raw datasets into the release.
