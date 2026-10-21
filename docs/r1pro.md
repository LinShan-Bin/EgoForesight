<!-- Copyright 2026 Chubin Zhang. SPDX-License-Identifier: MIT -->
# R1Pro: recordings, loading and model-side deployment

No real recordings, calibration files, URDF, robot credentials or robot-control SDK are distributed. Supply your own assets and recording permissions. Historical `pick_fruit_*` module/manifest names remain for checkpoint compatibility; the preparation CLI takes an explicit dataset ID and contains no private task path or implicit split.

## Native recording contract

The supported input is a **LeRobot v3 recording at 10 Hz**, not an invented NPZ interchange format. Each subset has `meta/info.json`, episode metadata, `data/chunk-*/*.parquet` and three videos (head, left wrist, right wrist). Packed fields are:

| Column | Shape and meaning |
|---|---|
| `action` | 58 values: left/right EEF xyz+xyzw (14), then two 22-joint hands |
| `observation.state` | 58 values: left/right **measured** arm joints (14), then two measured hands |
| `action.arm_{left,right}.joint.emitted_ik` | 7 absolute commanded arm joints, radians, recorded at the verified emitted-IK boundary |
| `observation.arm_{left,right}.joint` | 7 measured arm joints, radians |
| `{action,observation}.arm_{left,right}.eef.{torso_link4,head_rgb_optical}` | EEF poses in explicitly named frames |
| `{action,observation}.hand_{left,right}.joint` | 22 hand joints in the recording's declared order |
| `provenance.source` | Original recording/episode identity, used to keep fragments in the same split |

For emitted-command recordings, `training_features.json` must describe channel units, joint ordering and calibration provenance. The supported boundary is `verified_IK_output_at_jointTracker_input`; arm names are `arm_joint1` … `arm_joint7`. The converter checks the head extrinsic hash/transform against the supplied calibration and checks packed-versus-native channels. Do not mark measured joints as emitted commands to bypass these checks.

The exact hand mappings are in `starVLA/dataloader/gr00t_lerobot/hand_joint_layout.py` and `starVLA/dataloader/r1pro_transforms.py`. Recording, canonical and wire hand orders are **not interchangeable**; conversion is explicit.

### Choose splits and prepare caches

Create a recordings JSON whose episode lists cover each subset exactly once:

```json
{
  "dataset_id": "r1pro_my_task",
  "arm_joint_source": "emitted_ik",
  "subsets": [
    {
      "name": "session_a",
      "path": "/absolute/path/to/native_lerobot_v3",
      "train": [0, 1, 2],
      "validation": [3]
    }
  ]
}
```

The example assumes exactly four episodes; replace the lists with your actual split. Keep all fragments of a source episode on the same side. Both train and validation must be nonempty across the dataset. Normalization is fitted on training rows only.

```bash
python -m scripts.prepare_r1pro \
  --recordings /absolute/path/to/recordings.json \
  --raw-cache /absolute/path/to/new/raw_cache \
  --output /absolute/path/to/new/camera_spatial_cache \
  --calibration /absolute/path/to/head_calibration.json \
  --urdf /absolute/path/to/r1pro_kinematics.urdf --horizon 16
```

Use the training environment (the preparation code imports the native loaders). The calibration must identify `torso_link4` → `head_rgb_optical`, with inverse `T_C_B` / `T_B_C`; URDF FK tips are `left_gripper_link` and `right_gripper_link`. Both output paths must be new. The final cache links image arrays in the raw cache, so retain both when moving/deploying the dataset. Manifests record calibration/URDF hashes, splits and normalization semantics.

The cache has 130 canonical slots, with 70 supervised action slots and 58 current-state slots:

- Arms: `[0:7]` / `[29:36]`; relative EEF xyz+rotvec: `[7:13]` / `[36:42]`.
- Hands: `[80:102]` / `[105:127]`.
- State contains **measured arm/hand joints only**. FK is label/decoder geometry, never an extra model input.
- Images form a 384×320 canvas: head 256×320 over two 128×160 wrists. Future images are training targets.

Emitted arm labels retain the native semantic description: **recorded emitted IK joint commands; approximate-rpc alignment; no causal EEF pairing**. Hand targets remain the recorded controller targets at the source row, without a time shift. Do not advertise stronger synchronization guarantees than the recording provides.

An explicit `"arm_joint_source": "measured"` supports older recordings, but its arm labels are **measured joints at the same future source rows; not commanded joint targets**. Use `pick_fruit_dataset` for that path; the published full-update `r1pro_merged_dataset` recipe intentionally requires emitted-command labels. The legacy name of that loader does not imply that it loads another internal dataset.

### Train

```bash
export R1PRO_CACHE=/absolute/path/to/new/camera_spatial_cache
export R1PRO_DATASET_ID=r1pro_my_task
# Also set QWEN_MODEL, DINO_REPO, DINO_WEIGHTS and PRETRAINED_CHECKPOINT.
NUM_GPUS=2 BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 \
  bash scripts/posttrain.sh configs/posttrain/r1pro.yaml
```

The dataset ID must match the manifest and saved normalization key. The sampler uses complete global optimizer updates, avoiding a short accumulation at epoch boundaries. Keep `dataset_manifest.json`, `dataset_statistics.json`, the resolved configuration and the matching checkpoint/EMA sidecar together for deployment.

## Model-side server (no physical actuation)

First validate the saved contracts/assets without loading the model:

```bash
python -m deployment.model_server.server_r1pro \
  --run-dir /absolute/path/to/run \
  --checkpoint /absolute/path/to/run/checkpoints/steps_10000_pytorch_model.pt \
  --calibration /absolute/path/to/head_calibration.json \
  --urdf /absolute/path/to/r1pro_kinematics.urdf \
  --control-mode joint --validate-only
```

Then serve from the policy environment:

```bash
python -m deployment.model_server.server_r1pro \
  --run-dir /absolute/path/to/run \
  --checkpoint /absolute/path/to/run/checkpoints/steps_10000_pytorch_model.pt \
  --calibration /absolute/path/to/head_calibration.json \
  --urdf /absolute/path/to/r1pro_kinematics.urdf \
  --control-mode joint --host 127.0.0.1 --port 10093 --safety-mode
```

This defaults to the paired `foresight_forcing` EMA; use `--ema-teacher online` explicitly if that is the intended checkpoint. The adapter checks saved model settings, masks, horizon, image layout and calibration/URDF hashes. It denormalizes once, then emits either absolute joint commands or, with `--control-mode eef`, absolute `torso_link4` xyz+xyzw poses plus hand targets.

### Transport contract

The R1 server uses **client-first `policy.inference` version 1.0**, not the benchmark's `WebsocketClientPolicy`. Do not send the generic benchmark image/task dict to it.

1. The client negotiates version, limits and extensions with `hello`; the server replies with the action/observation contract.
2. The client sends the advertised channels with sample IDs, measurement/arrival/selection chronology and one current sample per channel. Images are RGB `uint8 [720,1280,3]`; arm/hand joint channels are float32 radians. Observations older than the advertised 200 ms are rejected.
3. Inference uses the observation-aligned 10 Hz horizon. Scheduling, reset, request IDs, bounded sessions and tensor validation are implemented in `deployment/model_server/tools/policy_inference.py`.
4. `--async` selects the distinct `policy.inference.async` protocol. Its contract and sender/receiver lifecycle are in `tools/async_policy_inference.py`; it is not a switch a synchronous client can ignore.

Use the negotiated contract and these protocol implementations as the integration reference; there is deliberately no fabricated robot-SDK example. The server predicts targets only. Your robot integration must provide clock alignment, stale-command rejection, joint/workspace/velocity limits, an emergency stop and an operator-controlled enable. `--safety-mode` adds an initial-action operator preview per session/reset; it is **not** a substitute for hardware safety. Use loopback/SSH forwarding or an appropriately secured robot network.
