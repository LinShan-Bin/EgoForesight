# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Validate, persist and supervise posttrain evaluation submissions.
"""Local eval-host job supervisor with independent server/client GPU assignment."""

import argparse
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

BENCHMARK_DEFAULTS = json.loads(Path(__file__).with_name("defaults.json").read_text())

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = dict(
    robotwin_path=os.environ.get("ROBOTWIN_ROOT", str(REPO / "external/RoboTwin")),
    server_python=os.environ.get("POLICY_PYTHON", sys.executable),
    client_python=os.environ.get("ROBOTWIN_PYTHON", sys.executable),
    robocasa_python=os.environ.get("ROBOCASA_PYTHON", sys.executable),
)


def atomic_json(path: Path, value: dict) -> None:
    """Publish a complete state snapshot."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def identity(pid: int) -> str | None:
    """Linux process creation ticks prevent signalling a reused PID."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def alive(record: dict) -> bool:
    """Check process identity before inspecting or stopping it."""
    return record.get("start") is not None and identity(record["pid"]) == record["start"]


def process_exit_description(code: int) -> str:
    """Distinguish native signals from ordinary Python/process failures."""
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = str(-code)
        return f"exit={code} signal={name}"
    return f"exit={code}"


def policy_ready(port: int) -> bool:
    """Complete the native handshake, without sending an inference request."""
    import msgpack
    from websockets.exceptions import ConnectionClosed, InvalidHandshake
    from websockets.sync.client import connect

    try:
        with connect(
            f"ws://127.0.0.1:{port}",
            proxy=None,
            compression=None,
            open_timeout=1,
            close_timeout=1,
            ping_interval=None,
        ) as connection:
            metadata = connection.recv(timeout=2)
            return isinstance(metadata, bytes) and isinstance(msgpack.unpackb(metadata, raw=False), dict)
    except (OSError, TimeoutError, ConnectionClosed, InvalidHandshake, ValueError):
        return False


def configured_ema_teachers(config_text: str) -> list[str]:
    """Return the EMA teachers that the Foresight Forcing config actually trains."""
    config = yaml.safe_load(config_text) or {}
    flow = config.get("framework", {}).get("flow", {}) if isinstance(config, dict) else {}
    if not isinstance(flow, dict):
        return []
    teachers = []
    foresight = flow.get("foresight_forcing", {})
    if (
        isinstance(foresight, dict)
        and foresight.get("enabled", False) is True
        and foresight.get("teacher_enabled", True) is True
    ):
        teachers.append("foresight_forcing")
    return teachers


def checkpoint_step(checkpoint: str) -> int | None:
    """Read a step from the checkpoint basename, including derived EMA exports."""
    match = re.search(r"(?:^|_)steps_(\d+)(?:_|\.)", Path(checkpoint).name)
    return int(match[1]) if match else None


def scan_checkpoints(root: Path) -> list[dict]:
    """Return model files with the config/statistics required by the policy loader."""
    root = root.resolve()
    result = []
    for group in ("checkpoints", "final_model", "derived_checkpoints"):
        for path in root.glob(f"*/{group}/*.pt"):
            if not path.name.endswith(("pytorch_model.pt", "_ema_policy.pt")):
                continue
            if not path.resolve().is_relative_to(root):
                continue
            run = path.parents[1]
            if all((run / name).is_file() for name in ("config.yaml", "dataset_statistics.json")):
                config_text = (run / "config.yaml").read_text()
                mix = re.search(r"^\s*data_mix:\s*(\S+)", config_text, re.MULTILINE)
                benchmark = (
                    "robotwin"
                    if mix and "robotwin" in mix[1]
                    else "robocasa" if mix and "robocasa" in mix[1] else "other"
                )
                config = yaml.safe_load(config_text) or {}
                action_model = config.get("framework", {}).get("action_model", {})
                action_space = action_model.get("action_space", {})
                eef_representation = action_space.get("representation") if benchmark == "robotwin" else None
                chunk_size = action_model.get("chunk_size")
                stem = path.name.removesuffix("_pytorch_model.pt")
                sidecar = path.with_name(
                    "ema_teachers.pt" if path.name == "pytorch_model.pt" else stem + "_ema_teachers.pt"
                )
                ema_teachers = (
                    configured_ema_teachers(config_text)
                    if sidecar.is_file() and sidecar.resolve().is_relative_to(root)
                    else []
                )
                result.append(
                    dict(
                        benchmark=benchmark,
                        chunk_size=chunk_size,
                        eef_representation=eef_representation,
                        checkpoint_step=checkpoint_step(path.name),
                        ema_available=bool(ema_teachers),
                        ema_teachers=ema_teachers,
                        ema_export="ema" in path.stem and not ema_teachers,
                        path=str(path.relative_to(root)),
                        run=run.name,
                        bytes=path.stat().st_size,
                        modified=path.stat().st_mtime,
                    )
                )
    return sorted(result, key=lambda item: item["modified"], reverse=True)


def task_names(benchmark: str = "robotwin") -> list[str]:
    """Read the public benchmark task lists without importing either simulator."""
    tasks = json.loads((Path(__file__).with_name("benchmark_tasks.json")).read_text())
    if benchmark not in tasks:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return tasks[benchmark]


def validate(root: Path, request: dict, runtime: dict) -> dict:
    """Reject unknown checkpoints, malformed allocations and unavailable runtimes."""
    benchmark = request.get("benchmark", "robocasa")
    request = {**BENCHMARK_DEFAULTS.get(benchmark, {}), **request}
    available = {item["path"]: item for item in scan_checkpoints(root)}
    if request.get("checkpoint") not in available:
        raise ValueError("Choose an available checkpoint with config.yaml and dataset_statistics.json")
    result = dict(runtime)
    result["benchmark"] = request.get("benchmark", "robocasa")
    if result["benchmark"] not in {"robotwin", "robocasa"}:
        raise ValueError("Unknown benchmark")
    result["ema_teacher"] = request.get("ema_teacher", "online")
    if result["ema_teacher"] == "auto":
        teachers = available[request["checkpoint"]]["ema_teachers"]
        result["ema_teacher"] = "foresight_forcing" if "foresight_forcing" in teachers else "online"
    result["weight_source"] = (
        "paired_ema"
        if result["ema_teacher"] != "online"
        else "exported_ema" if available[request["checkpoint"]]["ema_export"] else "online"
    )
    if result["ema_teacher"] not in {"online", "foresight_forcing"}:
        raise ValueError("Unknown EMA teacher")
    if (
        result["ema_teacher"] != "online"
        and result["ema_teacher"] not in available[request["checkpoint"]]["ema_teachers"]
    ):
        raise ValueError(f"Selected checkpoint has no {result['ema_teacher']} EMA teacher")
    result["checkpoint"] = str((root / request["checkpoint"]).resolve())
    result["checkpoint_step"] = available[request["checkpoint"]]["checkpoint_step"]
    value = request.get("cfg_scale", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("cfg_scale must be a finite non-negative number (1 disables CFG)")
    result["cfg_scale"] = float(value)
    gamma = request.get("gamma", 1.0)
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive number")
    result["gamma"] = float(gamma)
    result["temporal_ensemble"] = request.get("temporal_ensemble", "disabled")
    if result["temporal_ensemble"] not in {"disabled", "latest", "uniform", "exponential"}:
        raise ValueError("Unknown temporal ensemble mode")
    half_life = request.get("temporal_half_life_steps", 4.0)
    if (
        isinstance(half_life, bool)
        or not isinstance(half_life, (int, float))
        or not math.isfinite(half_life)
        or half_life <= 0
    ):
        raise ValueError("temporal_half_life_steps must be a finite positive number")
    result["temporal_half_life_steps"] = float(half_life)
    for field, default, lower, upper in [
        ("episodes", 100, 5, 10000),
        ("seed", 0, 0, 1000000),
        ("execution_steps", 24, 1, 512),
        ("flow_steps", 10, 1, 1000),
        ("clients_per_gpu", 1, 1, 16),
        ("base_port", 6100, 1024, 65000),
        ("max_episode_steps", 720, 1, 10000),
    ]:
        value = request.get(field, default)
        if isinstance(value, bool) or not str(value).isdigit() or not lower <= int(value) <= upper:
            raise ValueError(f"{field} must be an integer in [{lower}, {upper}]")
        result[field] = int(value)
    for field, default in [("server_gpus", [0]), ("client_gpus", [1])]:
        values = request.get(field, default)
        if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 or v > 63 for v in values):
            raise ValueError(f"{field} must be a non-empty GPU index list")
        if len(set(values)) != len(values):
            raise ValueError(f"{field} contains duplicate GPUs")
        result[field] = values
    result["control_mode"] = request.get("control_mode", "joint")
    result["setting"] = request.get("setting", "demo_randomized")
    if result["control_mode"] not in {"joint", "eef"} or result["setting"] not in {
        "demo_clean",
        "demo_randomized",
        "both",
    }:
        raise ValueError("Invalid control mode or RoboTwin setting")
    result["tasks"] = request.get("tasks") or task_names(result["benchmark"])
    if (
        not isinstance(result["tasks"], list)
        or not result["tasks"]
        or any(task not in task_names(result["benchmark"]) for task in result["tasks"])
        or len(set(result["tasks"])) != len(result["tasks"])
    ):
        raise ValueError("Choose unique benchmark task names")
    required_runtime = (
        ("server_python", "robotwin_path", "client_python")
        if result["benchmark"] == "robotwin"
        else ("server_python", "robocasa_python")
    )
    for field in required_runtime:
        if not Path(result[field]).exists():
            raise ValueError(f"Missing runtime: {field}={result[field]}")
    # Validate the resolved training contract in the policy runtime, without loading weights.
    check = """from omegaconf import OmegaConf
from types import SimpleNamespace
from starVLA.model.action_space import resolve_action_space_spec
from pathlib import Path
import sys,json
c=OmegaConf.load(sys.argv[1]); a=c.framework.action_model
stats=json.loads((Path(sys.argv[1]).parent/"dataset_statistics.json").read_text())
s=resolve_action_space_spec(SimpleNamespace(config=c,norm_stats=stats))
assert s, "Checkpoint has no supported action-space metadata"
assert int(sys.argv[2]) <= a.chunk_size, "Execution steps exceed training chunk"
if sys.argv[3] == "robotwin":
    assert s["name"] == "robotwin_aug_aligned", "Choose a RoboTwin augmented checkpoint"
    assert s["representation"] in {"relative", "absolute"} and s["joint_representation"] == "absolute"
    assert s["relative_eef_frame"] == "camera_spatial" and s["static_camera_frame"]
    assert s["frame_convention"] == "robotwin_head_opencv_v1"
else:
    expected = {"robocasa_gr1_joint", "robocasa_gr1_aug_aligned"} if sys.argv[4] == "joint" else {"robocasa_gr1_eef"}
    assert s["name"] in expected, "RoboCasa checkpoint does not support selected control mode"
if float(sys.argv[7]) != 1.0:
    cfg = OmegaConf.select(c, "framework.flow.classifier_free_guidance", default={})
    assert cfg.get("enabled", False) and float(cfg.get("context_dropout_prob", 0)) > 0, "CFG requires context-dropout training"
ema = None
if sys.argv[5] != "online":
    from deployment.model_server.tools.ema_policy import read_teacher
    _, ema = read_teacher(Path(sys.argv[6]), sys.argv[5])
print(json.dumps({"action_space":s,"chunk_size":int(a.chunk_size),"ema":ema}))
"""
    checked = subprocess.run(
        [
            result["server_python"],
            "-c",
            check,
            str(Path(result["checkpoint"]).parents[1] / "config.yaml"),
            str(result["execution_steps"]),
            result["benchmark"],
            result["control_mode"],
            result["ema_teacher"],
            result["checkpoint"],
            str(result["cfg_scale"]),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=30,
    )
    if checked.returncode:
        raise ValueError(checked.stderr[-2000:])
    result["contract"] = json.loads(checked.stdout)
    if result["temporal_ensemble"] != "disabled" and result["control_mode"] == "eef":
        if result["benchmark"] != "robotwin" or result["contract"]["action_space"]["representation"] != "absolute":
            raise ValueError("EEF temporal fusion requires a RoboTwin absolute-EEF checkpoint")
    return result


def task_plan(config: dict) -> list[tuple[str, str]]:
    """Expand each task into independent clean/randomized trials on shared servers."""
    settings = ["demo_clean", "demo_randomized"] if config["setting"] == "both" else [config["setting"]]
    if config["benchmark"] != "robotwin":
        settings = [config["setting"]]
    return [(task, setting) for task in config["tasks"] for setting in settings]


def submit(root: Path, request: dict, runtime: dict) -> dict:
    """Create a durable run and launch its detached supervisor."""
    config = validate(root, request, runtime)
    parent = Path(config["checkpoint"]).parents[1] / "simulation_eval"
    run = parent / (config["benchmark"] + "_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    config["run"] = str(run)
    atomic_json(run / "job.json", dict(config=config, status="queued", processes=[]))
    fields = dict(
        BENCHMARK=config["benchmark"],
        CHECKPOINT=config["checkpoint"],
        CHECKPOINT_STEP=config["checkpoint_step"] if config["checkpoint_step"] is not None else "",
        CFG_SCALE=config["cfg_scale"],
        GAMMA=config["gamma"],
        EMA_TEACHER=config["ema_teacher"],
        WEIGHT_SOURCE=config["weight_source"],
        EMA_DECAY=(config["contract"].get("ema") or {}).get("decay", ""),
        EMA_CHECKPOINT=(config["contract"].get("ema") or {}).get("checkpoint", ""),
        CONTROL_MODE=config["control_mode"],
        EEF_REPRESENTATION=(
            config["contract"]["action_space"].get("representation", "") if config["benchmark"] == "robotwin" else ""
        ),
        TASK_CONFIG=config["setting"],
        SEED=config["seed"],
        N_EPISODES_PER_TASK=config["episodes"],
        N_ACTION_STEPS=config["execution_steps"],
        TEMPORAL_ENSEMBLE=config["temporal_ensemble"],
        TEMPORAL_HALF_LIFE_STEPS=config["temporal_half_life_steps"],
        NUM_FLOW_STEPS=config["flow_steps"],
        SERVER_GPUS=config["server_gpus"],
        CLIENT_GPUS=config["client_gpus"],
        VIDEOS_PER_TASK=5 if config["benchmark"] == "robotwin" else config["episodes"],
        MAX_EPISODE_STEPS=config["max_episode_steps"],
        STOP_ON_SUCCESS="true",
    )
    (run / "run.env").write_text("".join(f"{k}={v}\n" for k, v in fields.items()))
    for index, (task, setting) in enumerate(task_plan(config)):
        (run / f"task_{index:03d}.log").write_text(f"TASK_QUEUED index={index} env={task} setting={setting}\n")
    with (run / "supervisor.log").open("ab") as log:
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--run", str(run)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return dict(id=str(run.relative_to(root)))


def stop(run: Path) -> None:
    """Request cancellation, including after a dashboard restart."""
    job = json.loads((run / "job.json").read_text())
    if job.get("status") in {"done", "stopped", "error"}:
        return
    (run / "STOP").touch()
    if job.get("status") == "queued":
        return
    if not alive(job.get("supervisor", {})):
        for record in job.get("processes", []):
            if alive(record):
                os.killpg(record["pid"], signal.SIGTERM)
        job["status"] = "stopped"
        atomic_json(run / "job.json", job)


def supervise(run: Path) -> None:
    """Own child process groups, propagate failures, and reap all workers on exit."""
    job = json.loads((run / "job.json").read_text())
    c = job["config"]
    job.update(status="starting", supervisor=dict(pid=os.getpid(), start=identity(os.getpid())))
    children = []
    logs = []
    locks = []
    cancelled = False

    def persist() -> None:
        atomic_json(run / "job.json", job)

    def spawn(cmd: list[str], gpu: int, log: Path, cwd: Path, extra_env: dict | None = None) -> subprocess.Popen:
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=devices[gpu][0],
            ROBOTWIN_RENDER_DEVICE="cuda:0",
            ROBOTWIN_EXPECTED_PCI=devices[gpu][1],
            PYTHONUNBUFFERED="1",
            PYTHONFAULTHANDLER="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONHASHSEED=str(c["seed"]),
            CUROBO_TORCH_COMPILE="1",
            PYTHONPATH=f"{REPO}:{c['robotwin_path']}:{REPO / 'third_party/robotwin_client'}",
            VK_ICD_FILENAMES=str(REPO / "scripts/posttrain_dashboard/nvidia_icd.json"),
            TOKENIZERS_PARALLELISM="false",
            NO_ALBUMENTATIONS_UPDATE="1",
        )
        env.update(extra_env or {})
        stream = log.open("ab")
        logs.append(stream)
        process = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        children.append(process)
        job["processes"].append(
            dict(
                pid=process.pid,
                start=identity(process.pid),
                command=cmd,
                gpu=gpu,
                gpu_uuid=devices[gpu][0],
                pci_bus_id=devices[gpu][1],
                egl_device=env.get("MUJOCO_EGL_DEVICE_ID"),
                log=log.name,
            )
        )
        persist()
        return process

    def check_stop() -> None:
        if (run / "STOP").exists():
            raise InterruptedError("Evaluation cancelled")

    def handle_signal(signum: int, frame: object) -> None:
        (run / "STOP").touch()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        import fcntl

        check_stop()
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"], text=True
        )
        devices = {
            int(parts[0]): (parts[1].strip(), parts[2].strip())
            for line in inventory.splitlines()
            if (parts := line.split(","))
        }

        (REPO / "results").mkdir(exist_ok=True)
        # GPU sharing is intentional; only listening ports must be exclusive.
        for name in [f'port_{c["base_port"] + i}' for i in range(len(c["server_gpus"]))]:
            lock = (REPO / "results" / f".posttrain_{name}.lock").open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        egl_map = {}
        if c["benchmark"] == "robocasa":
            detected = subprocess.check_output(
                [
                    c["robocasa_python"],
                    str(REPO / "deployment/model_server/tools/egl_device_map.py"),
                    *map(str, c["client_gpus"]),
                ],
                text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            egl_map = dict(zip(c["client_gpus"], map(int, detected.split()), strict=True))
            job["egl_mapping"] = egl_map
            persist()
        servers = []
        ports = [c["base_port"] + i for i in range(len(c["server_gpus"]))]
        for gpu, port in zip(c["server_gpus"], ports, strict=True):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
            servers.append(
                spawn(
                    [
                        c["server_python"],
                        str(REPO / "deployment/model_server/server_policy.py"),
                        "--ema_teacher",
                        c.get("ema_teacher", "online"),
                        "--ckpt_path",
                        c["checkpoint"],
                        "--port",
                        str(port),
                        "--seed",
                        str(c["seed"]),
                        "--use_bf16",
                        "--idle_timeout",
                        "-1",
                        "--max_batch_size",
                        "1",
                        "--num_inference_timesteps",
                        str(c["flow_steps"]),
                        "--cfg_scale",
                        str(c.get("cfg_scale", 1.0)),
                        "--inference_visual_timestep_alpha",
                        str(c.get("gamma", 1.0)),
                    ],
                    gpu,
                    run / f"server_gpu{gpu}.log",
                    REPO,
                )
            )
        deadline = time.monotonic() + 1200
        for port in ports:
            while True:
                check_stop()
                if any(p.poll() is not None for p in servers):
                    raise RuntimeError("Policy server exited; inspect server_gpu*.log")
                if policy_ready(port):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Policy server startup exceeded 20 minutes")
                time.sleep(1)
        job["status"] = "running"
        persist()
        queue = list(enumerate(task_plan(c)))
        slots = [
            (gpu, ports[i % len(ports)]) for i, gpu in enumerate(c["client_gpus"]) for _ in range(c["clients_per_gpu"])
        ]
        running = {}
        failed_tasks = []
        while queue or running:
            check_stop()
            if any(p.poll() is not None for p in servers):
                raise RuntimeError("Policy server died during evaluation")
            for slot_id, (gpu, port) in enumerate(slots):
                if slot_id in running:
                    process, index = running[slot_id]
                    code = process.poll()
                    if code is None:
                        continue
                    description = process_exit_description(code)
                    with (run / f"task_{index:03d}.log").open("a") as log:
                        log.write("\nTASK_DONE\n" if code == 0 else f"\nTASK_FAILED {description}\n")
                    del running[slot_id]
                    if code:
                        failed_tasks.append(dict(index=index, exit_code=code, gpu=gpu, port=port, error=description))
                        job["failed_tasks"] = failed_tasks
                        job["error"] = (
                            f"{len(failed_tasks)} task(s) failed; latest task {index} on GPU {gpu}: {description}; "
                            "remaining tasks continue"
                        )
                        persist()
                if queue:
                    index, (task, setting) = queue.pop(0)
                    output = run / "videos" / f"task_{index:03d}"
                    with (run / f"task_{index:03d}.log").open("a") as log:
                        log.write(
                            f"TASK_START index={index} sim_gpu={gpu} port={port} env={task} setting={setting} "
                            f"egl_device={egl_map.get(gpu, 'none')} time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
                        )
                    args = dict(
                        port=port,
                        policy_ckpt_path=c["checkpoint"],
                        task_name=task,
                        task_config=setting,
                        ckpt_setting=run.name,
                        seed=c["seed"],
                        start_seed=100000 * (1 + c["seed"]),
                        test_num=c["episodes"],
                        worker_id=0,
                        num_workers=1,
                        policy_name="posttrain_interface",
                        control_mode=c["control_mode"],
                        action_execution_steps=c["execution_steps"],
                        temporal_ensemble=c.get("temporal_ensemble", "disabled"),
                        temporal_half_life_steps=c.get("temporal_half_life_steps", 4.0),
                        eval_video_count=5,
                        output_dir=output,
                    )
                    cmd = [
                        c["client_python"],
                        str(Path(c["robotwin_path"]) / "script/eval_policy.py"),
                        "--config",
                        str(REPO / "third_party/robotwin_client/deploy_policy.yml"),
                        "--overrides",
                    ]
                    for key, value in args.items():
                        cmd.extend([f"--{key}", str(value)])
                    prompt_env = dict(
                        POSTTRAIN_PROMPT_LOG=str(run / f"task_{index:03d}.prompts.jsonl"),
                        POSTTRAIN_TASK=task,
                        POSTTRAIN_SETTING=setting if c["benchmark"] == "robotwin" else "",
                    )
                    if c["benchmark"] == "robotwin":
                        process = spawn(cmd, gpu, run / f"task_{index:03d}.log", Path(c["robotwin_path"]), prompt_env)
                    else:
                        command = [
                            c["robocasa_python"],
                            str(REPO / "examples/Robocasa_tabletop/eval_files/simulation_env.py"),
                        ]
                        options = {
                            "env-name": task,
                            "host": "127.0.0.1",
                            "port": port,
                            "n-episodes": c["episodes"],
                            "n-envs": 1,
                            "max-episode-steps": c["max_episode_steps"],
                            "n-action-steps": c["execution_steps"],
                            "temporal-ensemble": c.get("temporal_ensemble", "disabled"),
                            "temporal-half-life-steps": c.get("temporal_half_life_steps", 4.0),
                            "action-format": "absolute_joint" if c["control_mode"] == "joint" else "relative_eef",
                            "video-out-path": output,
                            "pretrained-path": c["checkpoint"],
                            "seed": c["seed"],
                        }
                        for key, value in options.items():
                            command.extend(["--args." + key, str(value)])
                        command.append("--args.stop-on-success")
                        process = spawn(
                            command,
                            gpu,
                            run / f"task_{index:03d}.log",
                            REPO,
                            {
                                **prompt_env,
                                "CUDA_VISIBLE_DEVICES": "",
                                "MUJOCO_EGL_DEVICE_ID": str(egl_map[gpu]),
                                "MUJOCO_GL": "egl",
                                "PYOPENGL_PLATFORM": "egl",
                            },
                        )
                    running[slot_id] = process, index
            time.sleep(0.5)
        job["status"] = "error" if failed_tasks else "done"
        if failed_tasks:
            job["error"] = "Failed tasks: " + ", ".join(str(task["index"]) for task in failed_tasks)
    except InterruptedError:
        cancelled = True
        job["status"] = "stopped"
    except Exception as error:
        job.update(status="error", error=str(error))
    finally:
        # Group leaders can exit while their ffmpeg/planner descendants survive.
        for record in job["processes"]:
            if alive(record) or identity(record["pid"]) is None:
                try:
                    os.killpg(record["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 8
        while any(p.poll() is None for p in children) and time.monotonic() < deadline:
            time.sleep(0.1)
        for record in job["processes"]:
            if alive(record) or identity(record["pid"]) is None:
                try:
                    os.killpg(record["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in children:
            process.wait()
        for stream in logs + locks:
            stream.close()
        if cancelled or (run / "STOP").exists():
            job["status"] = "stopped"
        persist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    supervise(parser.parse_args().run)
