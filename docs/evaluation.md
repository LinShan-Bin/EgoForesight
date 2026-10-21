<!-- Copyright 2026 Chubin Zhang. SPDX-License-Identifier: MIT -->
# Simulation evaluation and dashboard

The policy runs in the training environment; each simulator runs in its own upstream environment. The dashboard launches the included clients and policy server, tracks jobs, and displays logs, success rates and videos. It supports **RoboTwin 2.0 (50 tasks)** and **RoboCasa GR1 tabletop (24 tasks)**, not RoboCasa365.

## Prepare the simulator clients

Follow [environment setup](environment.md) and the upstream simulator/asset instructions. For RoboTwin, use a dedicated, inactive checkout before applying the included patches:

```bash
python -m scripts.posttrain_dashboard.setup_robotwin \
  --robotwin-path /absolute/path/to/RoboTwin
```

This backs up replaced files to `.posttrain_originals`, installs the retained evaluation patches, and relocates known embodiment asset paths. It is not an installer for SAPIEN or robot assets. The RoboTwin adapter lives in `third_party/robotwin_client/posttrain_interface.py`; RoboCasa adapters live in `examples/Robocasa_tabletop/eval_files/`.

## Start the dashboard

```bash
export POLICY_PYTHON=/absolute/path/to/egofore/bin/python
export ROBOTWIN_ROOT=/absolute/path/to/RoboTwin
export ROBOTWIN_PYTHON=/absolute/path/to/robotwin/bin/python
export ROBOCASA_PYTHON=/absolute/path/to/robocasa/bin/python

"$POLICY_PYTHON" -m scripts.posttrain_dashboard.serve \
  --directory /absolute/path/to/results --port 8766
```

`--directory` is the **parent of experiment directories**, not the checkpoint file or an individual `checkpoints/` directory:

```text
results/
  experiment_a/
    config.yaml
    config_full.yaml
    dataset_statistics.json
    checkpoints/steps_<N>_pytorch_model.pt
    checkpoints/steps_<N>_ema_teachers.pt
    simulation_eval/...
  experiment_b/...
```

The dashboard only lists model files with the required saved configuration and statistics. It also recognizes compatible `final_model/` and `derived_checkpoints/` exports. The lightweight checkpoint and its metadata must belong to the same run. If the recorded backbone paths are no longer valid on the evaluation host, update the transferred run's paths to the same local assets without changing model/data semantics.

The default bind address is **127.0.0.1**. For a remote host:

```bash
ssh -N -L 8766:127.0.0.1:8766 your-evaluation-host
```

Open `http://127.0.0.1:8766` locally. This is a local administration UI, not an Internet-facing authenticated service. Submit/stop operations use a same-origin token. Do not expose the port publicly.

## Submit an evaluation

1. Select the benchmark, checkpoint and weight source (**online** or its paired **foresight_forcing EMA**). The UI rejects an unavailable EMA sidecar.
2. Select tasks, seed, episode count and control mode. RoboTwin supports clean, randomized or both settings; RoboCasa uses the 24-task GR1 environment list.
3. Set policy GPU IDs, simulator GPU IDs, clients per GPU, flow steps and execution chunk length. GPU IDs refer to the evaluation host, not a launcher's already-remapped local index. Allocate only idle GPUs.
4. Submit, then inspect the first task's server/client logs and video. Run a small task selection before an entire benchmark; do not interpret an incomplete or errored run as a complete success-rate evaluation.

The configuration, process records and logs are persisted with each evaluation job. The supervisor bounds concurrent clients, records failures, manages process groups, and stops owned child processes. Dashboard stop controls evaluation jobs, not a separately launched training job. Removing a result uses the dashboard's result archive/trash workflow; it does not delete model checkpoint files.

The adapters match the training image layout, prompt style, normalization and action frame. RoboTwin keeps the same joint state input whether decoding joints or EEF poses; camera-spatial EEF outputs are converted back to the simulator frame. RoboCasa's absolute-joint and relative-EEF clients are separate. Do not feed one control mode's compact output directly to the other controller.

## Native simulator failures

Start RoboCasa with **one client per simulator GPU** and increase concurrency only after a small real rollout completes with readable videos. Available VRAM alone does not establish that concurrent EGL rendering is stable. The dashboard keeps the requested concurrency; it does not silently change inference parameters or reschedule failed episodes.

- `TASK_QUEUED` means a task has not started. `TASK_START` records its actual simulator GPU, policy port, EGL device and launch time. `job.json` also records each child process's GPU UUID and PCI bus ID.
- `TASK_FAILED exit=-6 signal=SIGABRT` is a native abort, not an ordinary Python exception. Child processes enable Python's fault handler, so inspect the current-thread stack in `task_*.log`. Existing logs without the signal name are also decoded by the dashboard.
- If the stack ends in MuJoCo rendering (`mjr_readPixels`, `MjrContext.free`, or `robosuite/utils/binding_utils.py`), correlate its time and PCI bus ID with `dmesg -T` / NVIDIA Xid messages. Retry a small task at one client per GPU or use a known-working rendering GPU. This is a workaround to validate, not proof of a driver or hardware root cause. Do not reset a GPU or change its driver while other jobs use it.
- Policy readiness completes the WebSocket handshake and receives metadata without inference. It no longer deliberately opens and drops an empty TCP connection, which produced misleading `opening handshake failed` tracebacks.

Preserve failed logs and do not include partial diagnostics in full-benchmark success rates. Model/preprocessing warnings and simulator observation-space warnings require separate investigation; they are not, by themselves, evidence of the native abort's cause.

## Policy server without the dashboard

```bash
python -m deployment.model_server.server_policy \
  --ckpt_path /absolute/path/to/run/checkpoints/steps_10000_pytorch_model.pt \
  --ema_teacher online --host 127.0.0.1 --port 10093 \
  --use_bf16 --seed 42
```

For EMA, set `--ema_teacher foresight_forcing` and supply the matching sidecar. `--help` describes flow-step, CFG, visual schedule and batching overrides; record any change when reporting results. The benchmark clients use the included MsgPack/WebSocket policy transport. The **R1Pro protocol is different**; see [R1Pro](r1pro.md).

## Dashboard defaults

The interface opens in English with the RoboCasa preset selected. Presets are
shared between the browser and submission API in
`scripts/posttrain_dashboard/defaults.json`.

| Parameter | RoboCasa default |
|---|---|
| Episodes per task / seed | 50 / 0 |
| Control / execution length | Joint / 16 actions |
| Flow steps / CFG / visual gamma | 20 / 1.0 / 1.5 |
| Temporal ensemble / half-life | Exponential / 4 steps |
| Maximum episode length | 720 |
| Server GPU / client GPU | 0 / 1 |
| Clients per GPU / base port | 8 / 6100 |
| Weights | Foresight EMA if a paired sidecar exists; otherwise checkpoint weights |

Set GPU IDs and client concurrency for your hardware; reduce concurrency if
rendering is unstable. Select `online` for the published, already-merged EMA
checkpoints. Execution length is limited by the selected checkpoint's action
chunk. Temporal averaging only combines overlapping predictions: with a
16-action chunk and 16 executed actions, there is no overlap to average.
Changing a runtime preset does not change a checkpoint's trained chunk length.
