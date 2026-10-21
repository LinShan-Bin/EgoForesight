<!-- Copyright 2026 Chubin Zhang. SPDX-License-Identifier: MIT -->
# Environment

## Policy/training runtime

Use Linux, Python **3.10**, and PyTorch **2.8.0 + CUDA 12.8**. Dependency versions are pinned in `requirements/train.txt`.

```bash
conda env create -f environment.yml
conda activate egofore
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# Install torch first: source-built extensions may inspect the installed torch/CUDA.
python -m pip install -r requirements/train.txt
python -m pip install --no-deps -e .
export PYTHONPATH="$PWD"
```

CUDA driver/toolkit, GPU architecture and extension wheels must be compatible. FlashAttention-4 / FLA / Quack are fast paths, not optional imports silently bypassed by the default recipe. If a wheel is unavailable, build that exact upstream version against the installed PyTorch/CUDA following its upstream instructions.

Do not install this alongside another editable package exposing `starVLA`. Run from this root, set `PYTHONPATH` explicitly, and verify `python -c 'import starVLA; print(starVLA.__file__)'` points here. No parent checkout is required. Installing with `--no-deps -e .` registers this source tree; requirements files own dependency installation.

## Backbones and weights

- **Qwen3.5-4B**: obtain the official model/tokenizer/processor files; set `QWEN_MODEL` to that local directory. Requires the Qwen3.5 implementation in the pinned Transformers runtime.
- **DINOv3**: obtain the official `facebookresearch/dinov3` source and ViT-B/16 checkpoint; set `DINO_REPO` and `DINO_WEIGHTS`. The release does not download or relicense these assets.
- **EgoForesight**: supply a compatible MoT3 checkpoint through `PRETRAINED_CHECKPOINT`. Post-training from that checkpoint is supported; private pretraining is outside this release.
- Model asset paths are environment variables in the templates. Do not publish a run's resolved private absolute paths or credentials with a checkpoint.

A copyable variable template is `.env.example`; edit it before use. It contains no credentials or real assets.

### Download DINOv3

`DINO_REPO` is the **local source checkout**, containing `hubconf.py`; it is not
a weight file or a Hugging Face model directory. From the EgoForesight root:

```bash
mkdir -p external
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
export DINO_REPO="$PWD/external/dinov3"
```

The Git repository does **not** include pretrained weights. Request access via
Meta's [official DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
After approval, use the supplied **ViT-B/16 distilled, LVD-1689M** download URL.
EgoForesight expects the native PyTorch `.pth` weights, not a Transformers export:

```bash
# Replace this placeholder with the authorized URL from Meta; do not commit the URL.
DINO_DOWNLOAD_URL='<YOUR_AUTHORIZED_VITB16_DOWNLOAD_URL>'
mkdir -p external/weights
wget "$DINO_DOWNLOAD_URL" \
  -O external/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
export DINO_WEIGHTS="$PWD/external/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
unset DINO_DOWNLOAD_URL
```

Keep both variables set when training or serving. The model uses the local
DINOv3 source checkout to construct the encoder and loads `DINO_WEIGHTS`.
Follow the upstream license and access terms; do not redistribute authorized
download URLs. See the [official DINOv3 repository](https://github.com/facebookresearch/dinov3)
for upstream setup and model documentation.


## Data conversion

`requirements/data.txt` covers official raw conversion: NumPy, SciPy, Arrow, PyAV, OpenCV, HDF5 and MuJoCo. Relative-action statistics use the training dataloader and should be generated in the training environment.

```bash
conda create -n egofore-data python=3.10 pip -y
conda activate egofore-data
python -m pip install -r requirements/data.txt
export PYTHONPATH=/absolute/path/to/EgoForesight
```

RoboCasa needs the matching **official simulator mesh/texture assets**, even when not rendering images. The converter resolves full asset suffixes/path maps; it never deletes scene objects to make an XML file load. Rendering needs working EGL/OpenGL and H.264 encoding; headless servers commonly set `MUJOCO_GL=egl`. Do not overwrite system GPU-driver files.

## Separate simulator environments

Keep policy/training and simulator Python environments separate. The dashboard accepts an explicit Python executable for each.

### RoboTwin

Use the official [RoboTwin repository and installation guide](https://github.com/RoboTwin-Platform/RoboTwin), including its simulator/robot assets. `requirements/robotwin-client.txt` adds only transport/adapter dependencies; it does not install the simulator. Run `scripts.posttrain_dashboard.setup_robotwin` against a **dedicated** simulator checkout to install the included evaluation patches and relocate copied embodiment asset paths. Original files are backed up to `.posttrain_originals`; do not apply to an active evaluation checkout.

### RoboCasa GR1 tabletop

Use [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks), not the unrelated 365-task setup. Install its specified robosuite fork and assets, then `requirements/robocasa-client.txt` in that simulator environment.
