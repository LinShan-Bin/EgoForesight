<!-- Copyright 2026 Chubin Zhang. SPDX-License-Identifier: MIT -->
# Simulation data: official raw sources to training

Run commands from the EgoForesight root. Use the **data environment** for conversion and the **training environment** for normalization statistics / training; see [environment](environment.md). Outputs must be separate from the raw data. Do not publish data, assets or instructions without checking their own upstream terms.

## RoboTwin 2.0

Start with the official [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) HDF5 demonstrations and their instruction JSON files. Arrange the extracted download as:

```text
RAW/
  <task>/
    aloha-agilex_clean_50/
      data/episode0.hdf5
      instructions/episode0.json
      ...
    aloha-agilex_randomized_500/
      data/episode0.hdf5
      instructions/episode0.json
      ...
```

The converter reads the actual HDF5 camera / end-pose channels and official instructions. It removes near-static frames using the retained thresholds, then pairs each retained observation with the **next retained** pose/gripper target. The source-episode and frame-index mappings are saved. Joint augmentation uses these mappings to recover the corresponding raw joint rows; it does not assume that output row numbers equal raw frame numbers. Head-camera transforms and quaternion conventions are handled in `scripts/robotwin_aligned/common.py`.

### Build without mirrors

```bash
# In the data environment; add randomized_500 for full training.
python -m scripts.robotwin_aligned.build_all \
  --raw-root /absolute/path/to/RAW \
  --output-root /absolute/path/to/robotwin-aligned \
  --settings clean_50 --workers 4 --skip-stats
```

Use `--tasks adjust_bottle ...` to select tasks explicitly. An individual source conversion can be inspected with:

```bash
python -m scripts.robotwin_aligned.build_lerobot \
  --raw-root /absolute/path/to/RAW --output-root /absolute/path/to/robotwin-aligned \
  --task adjust_bottle --setting clean_50 --max-episodes 1
```

Use a **different output directory** for this one-episode diagnostic; a partial task is not the complete training corpus. The orchestration script resumes complete task repositories and refuses to silently delete an incomplete or differently sized output.

The full conversion produces:

```text
robotwin-aligned/
  aloha_agilex-clean_50_single_task/<task>/            # aligned EEF data
  aloha_agilex-clean_50_single_task-joint-aug/<task>/  # EEF + joints, used for training
  aloha_agilex-randomized_500_single_task/<task>/
  aloha_agilex-randomized_500_single_task-joint-aug/<task>/
```

Generate the horizon-specific relative statistics in the **training environment**:

```bash
python -m scripts.robotwin_aligned.precompute_joint_stats \
  --root /absolute/path/to/robotwin-aligned/aloha_agilex-clean_50_single_task-joint-aug \
  --horizon 48 --workers 4
# Repeat for randomized_500 when preparing the full recipe.
```

Statistics have a signature covering horizon, delta indices, frame convention and camera compensation. Keep the generated `meta/relative_action_stats*.json` with each task. Do not reuse statistics from another horizon or fit a diagnostic subset under a complete corpus's identity.

### Optional mirrors

No mirroring or language relabeling is implicit. To reproduce a mirrored mixture, run `build_all` with **one** of:

- `--mirror-prompt-dir /absolute/path/to/prompts`, containing `<setting>/<task>.json`. Each JSON maps **every dense output episode index** (string keys `"0"`, `"1"`, …) to its explicitly reviewed reflected instruction.
- `--swap-direction-words`, an explicit opt-in to geometric reflection and left/right word exchange only. This does not correct asymmetric objects or other semantic differences; inspect the resulting data before using it.

The result adds `<task>-mirrored` repositories. There is no VLM-based relabeler, private prompt cache or hidden episode exclusion list. Without mirrors, choose a `_no_mirror` recipe.

### Training roots matter

| Recipe | `DATA_ROOT` |
|---|---|
| `configs/posttrain/robotwin_c2r_no_mirror.yaml` | `.../robotwin-aligned/aloha_agilex-clean_50_single_task-joint-aug` |
| `configs/posttrain/robotwin_c2r.yaml` | Same root, **with** mirrored tasks |
| `configs/posttrain/robotwin_full_no_mirror.yaml` | `.../robotwin-aligned` (both settings below it) |
| `configs/posttrain/robotwin_full.yaml` | Same parent root, both settings **and** mirrors |

```bash
export DATA_ROOT=/absolute/path/to/robotwin-aligned/aloha_agilex-clean_50_single_task-joint-aug
bash scripts/posttrain.sh configs/posttrain/robotwin_c2r_no_mirror.yaml
```

The shipped mixtures cover 50 tasks; reduce the named mixture explicitly when training on fewer tasks. See `starVLA/dataloader/gr00t_lerobot/mixtures.py`.

## RoboCasa GR1 tabletop: 24 tasks

Use the official [NVIDIA PhysicalAI-Robotics-GR00T-Teleop-Sim dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim): **both** its unaugmented `LeRobot/gr1_unified.<Task>` data and matching `HDF5/<Task>.hdf5`. This is the GR1 tabletop benchmark, not RoboCasa365. Obtain the corresponding [RoboCasa GR1 tabletop simulator](https://github.com/robocasa/robocasa-gr1-tabletop-tasks), its specified robosuite fork, and their assets.

```text
OFFICIAL/
  LeRobot/gr1_unified.<Task>/
    meta/{info.json,modality.json,episodes.jsonl,tasks.jsonl}
    data/chunk-*/episode_*.parquet
    videos/chunk-*/*/episode_*.mp4
  HDF5/<Task>.hdf5
```

The converter verifies the official **44-D joint schema at 20 Hz** and matches each LeRobot episode to its HDF5 trajectory ID. It restores recorded MuJoCo `qpos/qvel`, runs `mj_forward` to recover body/camera geometry, and never steps the simulation, replays controllers or removes scene objects. Joint commands remain unchanged. EEF action labels are the next observed grip-site poses, with the terminal pose repeated. The retained geometry/rotation conversion then yields the aligned representation consumed by the loader.

```bash
python -m scripts.robocasa_from_official \
  --official-lerobot /absolute/path/to/OFFICIAL/LeRobot/gr1_unified.PnPBottleToCabinetClose \
  --hdf5 /absolute/path/to/OFFICIAL/HDF5/PnPBottleToCabinetClose.hdf5 \
  --output /absolute/path/to/robocasa-aligned/gr1_unified.PnPBottleToCabinetClose \
  --assets-root /absolute/path/to/robosuite/robosuite/models/assets \
  --assets-root /absolute/path/to/robocasa-gr1-tabletop-tasks/robocasa/models/assets \
  --mesh-root /absolute/path/to/robosuite/robosuite/models/assets/grippers/meshes/fourier_hands \
  --path-map /absolute/path/to/asset_paths.json
```

`--path-map` is needed when both asset roots contain the same suffix (for example `textures/ceramic.png`). Inspect the recorded episode XML's paths and supply an explicit old-prefix → local-prefix JSON; do not resolve collisions by deleting objects or picking a basename at random:

```json
{
  "/original/location/robosuite/models/assets/": "/absolute/path/to/robosuite/robosuite/models/assets/",
  "/original/location/robocasa/models/assets/": "/absolute/path/to/robocasa-gr1-tabletop-tasks/robocasa/models/assets/"
}
```

Replace the example keys with the prefixes actually stored in the HDF5 XML. The most-specific matching explicit mapping takes precedence over suffix searches, and a missing mapped asset is an error. Without ambiguity, `--path-map` can be omitted. Interrupted work is left in a `.building` sibling for inspection; use a fresh output location rather than overwriting a completed dataset.

By default, the converter **copies the official video unchanged**, typically 256×256. To render the recorded states with the supplied assets, add `--render-video --height 480 --width 640` (requires a working renderer). The default video is **not pixel-equivalent to a separately rendered high-resolution derivative**. Neither camera resizing nor the short loader comparison is a claim of image equivalence to that derivative.

Repeat for the selected tasks, then in the training environment:

```bash
python -m scripts.precompute_robocasa_aug_stats \
  --root /absolute/path/to/robocasa-aligned --horizon 16 --workers 4
export DATA_ROOT=/absolute/path/to/robocasa-aligned
bash scripts/posttrain.sh configs/posttrain/robocasa.yaml
```

The recipe expects all 24 tasks. Both benchmarks use `lerobot_datasets` and the retained GR00T-based transforms, returning model-agnostic images, text, state/actions, schema masks and metadata. Model-specific Qwen/DINO processing stays in the model, not in the raw converter.
