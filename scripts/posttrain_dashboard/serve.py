# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Submit and stop posttrain evaluations while serving metrics and seekable videos.
"""Usage: python serve.py --directory /absolute/checkpoint/simulation_eval."""

import argparse
import datetime
import json
import re
import secrets
import subprocess
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

if __package__:
    from . import jobs
else:
    import jobs

PROGRESS = re.compile(r"EVAL_PROGRESS env=(\S+) completed=(\d+)/(\d+) successes=(\d+) success_rate=([\d.]+)")


def experiment_paths(root: Path) -> list[Path]:
    """Discover runs in a simulation_eval directory or across checkpoint roots."""
    candidates = list(root.iterdir()) + list(root.glob("*/simulation_eval/*"))
    return sorted(
        {
            p.resolve()
            for p in candidates
            if not p.is_symlink() and p.is_dir() and (p / "run.env").is_file() and p.resolve().is_relative_to(root)
        }
    )


@lru_cache(maxsize=4096)
def log_summary(path: Path, modified_ns: int, size: int) -> dict:
    """Cache only parsed metrics, never large unchanged historical log contents."""
    content = path.read_text(errors="replace")
    matches = list(PROGRESS.finditer(content))
    match = matches[-1] if matches else None
    start = re.search(r"TASK_START index=(\d+) sim_gpu=(\d+) port=\d+ env=(\S+)", content)
    queued = re.search(r"TASK_QUEUED index=\d+ env=(\S+)", content)
    failure = re.search(r"TASK_FAILED exit=(-?\d+)", content)
    fatal = re.search(r"Fatal Python error: ([^\n]+)", content)
    error = (
        jobs.process_exit_description(int(failure[1]))
        if failure
        else fatal[0] if fatal else "Python traceback" if "Traceback" in content else None
    )
    return dict(
        history={int(m[2]): int(m[4]) for m in matches},
        setting=(m[1] if (m := re.search(r"TASK_(?:START|QUEUED)[^\n]* setting=(\S+)", content)) else None),
        name=match[1] if match else start[3] if start else queued[1] if queued else path.stem,
        error=error,
        completed=int(match[2]) if match else 0,
        total=int(match[3]) if match else None,
        successes=int(match[4]) if match else 0,
        status=(
            "error"
            if error or "TASK_FAILED" in content
            else "done" if "TASK_DONE" in content else "queued" if queued and not start else "running"
        ),
    )


def read_run(path: Path, active_commands: list[str]) -> dict:
    """Read cumulative progress; incomplete log lines are naturally ignored."""
    config = dict(line.split("=", 1) for line in (path / "run.env").read_text().splitlines() if "=" in line)
    if not config.get("CHECKPOINT_STEP"):
        step = jobs.checkpoint_step(config.get("CHECKPOINT", ""))
        config["CHECKPOINT_STEP"] = str(step) if step is not None else ""
    config.setdefault("CFG_SCALE", "1.0")
    tasks = []
    histories = []
    for log in sorted(path.glob("task_*.log")):
        stat = log.stat()
        summary = log_summary(log, stat.st_mtime_ns, stat.st_size)
        histories.append(summary["history"])
        task_dir = path / "videos" / log.stem
        tasks.append(
            dict(
                id=log.stem,
                setting=summary["setting"] or config.get("TASK_CONFIG", ""),
                name=summary["name"].split("/")[-1].removesuffix("_GR1ArmsAndWaistFourierHands_Env"),
                completed=summary["completed"],
                total=summary["total"] or int(config.get("N_EPISODES_PER_TASK", 50)),
                successes=summary["successes"],
                videos=len(list(task_dir.glob("*_success[01].mp4"))),
                status=summary["status"],
                error=summary["error"],
            )
        )
    completed = sum(t["completed"] for t in tasks)
    successes = sum(t["successes"] for t in tasks)

    # A point at k uses exactly the first k episodes of EVERY task, preventing
    # faster tasks from being over-represented in the comparison curve.
    def metrics(indices: list[int]) -> dict:
        group = [histories[i] for i in indices]
        common = min((max(h, default=0) for h in group), default=0)
        return dict(
            completed=sum(tasks[i]["completed"] for i in indices),
            successes=sum(tasks[i]["successes"] for i in indices),
            total=sum(tasks[i]["total"] for i in indices),
            curve=[
                dict(episode=k, rate=sum(h[k] for h in group) / (k * len(group)))
                for k in range(1, common + 1)
                if all(k in h for h in group)
            ],
        )

    setting_results = {
        setting: metrics([i for i, task in enumerate(tasks) if task["setting"] == setting])
        for setting in dict.fromkeys(task["setting"] for task in tasks)
    }
    curve = metrics(list(range(len(tasks))))["curve"] if len(setting_results) <= 1 else []
    active = any(str(path) + "/" in command for command in active_commands)
    finished = tasks and all(t["completed"] >= t["total"] for t in tasks)
    status = (
        "invalid"
        if (path / "INVALIDATED").exists()
        else (
            "done"
            if finished
            else ("running" if active else ("error" if any(t["status"] == "error" for t in tasks) else "stopped"))
        )
    )
    job_file = path / "job.json"
    managed = job_file.exists()
    error = None
    if managed:
        job = json.loads(job_file.read_text())
        state = job.get("status", "error")
        live = jobs.alive(job.get("supervisor", {}))
        status = "running" if state in {"queued", "starting", "running"} and live else state
        if status in {"queued", "starting", "running"} and not live:
            status = "running" if state == "queued" and time.time() - job_file.stat().st_mtime < 30 else "stopped"
        if (path / "STOP").exists() and live:
            status = "running"
        error = job.get("error")
    if (path / "INVALIDATED").exists():
        status = "invalid"
        error = (path / "INVALIDATED").read_text().strip() or error
    metadata_file = path / "dashboard.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
    label = path.name
    return dict(
        id=path.name,
        label=label,
        managed=managed,
        pinned=metadata.get("pinned", False),
        tags=metadata.get("tags", []),
        logs=[p.name for p in sorted(path.glob("*.log"))],
        error=error,
        config=config,
        tasks=tasks,
        completed=completed,
        successes=successes,
        total=sum(t["total"] for t in tasks),
        curve=curve,
        setting_results=setting_results,
        status=status,
        directory=path.name,
    )


def update_metadata(root: Path, request: dict) -> dict:
    """Persist independent dashboard annotations without touching supervisor state."""
    run = (root / request["run"]).resolve()
    if run not in experiment_paths(root):
        raise ValueError("Unknown evaluation run")
    path = run / "dashboard.json"
    metadata = json.loads(path.read_text()) if path.is_file() else {}
    if "pinned" in request:
        if type(request["pinned"]) is not bool:
            raise ValueError("pinned must be a boolean")
        metadata["pinned"] = request["pinned"]
    if "tags" in request:
        tags = request["tags"]
        if (
            not isinstance(tags, list)
            or len(tags) > 12
            or any(
                not isinstance(tag, str) or not 1 <= len(tag.strip()) <= 32 or any(ord(c) < 32 for c in tag)
                for tag in tags
            )
        ):
            raise ValueError("Use at most 12 tags, each 1-32 characters")
        metadata["tags"] = list(dict.fromkeys(tag.strip() for tag in tags))
    jobs.atomic_json(path, metadata)
    return metadata


def gpu_snapshot() -> dict:
    """Read physical GPU utilization without confusing CUDA and EGL indices."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
        devices = []
        for line in output.splitlines():
            index, name, utilization, used, total, temperature = [v.strip() for v in line.split(",")]

            def number(value: str) -> float | None:
                return float(value) if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) else None

            devices.append(
                dict(
                    index=int(index),
                    name=name,
                    utilization=number(utilization),
                    memory_used=number(used),
                    memory_total=number(total),
                    temperature=number(temperature),
                )
            )
        return dict(devices=devices, error=None)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return dict(devices=[], error=str(error))


def trash_run(root: Path, run_id: str) -> dict:
    """Move inactive evaluation output into a recoverable, unlisted directory."""
    run = (root / run_id).resolve()
    if run not in experiment_paths(root):
        raise ValueError("Unknown evaluation run")
    commands = []
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            commands.append(proc.read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except OSError:
            continue
    result = read_run(run, commands)
    job = json.loads((run / "job.json").read_text()) if result["managed"] else {}
    if (
        result["status"] in {"running", "queued", "starting"}
        or any(str(run) + "/" in command for command in commands)
        or jobs.alive(job.get("supervisor", {}))
        or any(jobs.alive(record) for record in job.get("processes", []))
    ):
        raise ValueError("请先停止评测, 等待进程退出后再删除")
    trash = root / ".posttrain-trash"
    if trash.is_symlink():
        raise ValueError("Invalid trash directory")
    trash.mkdir(exist_ok=True)
    destination = trash / (run.name + "_" + secrets.token_hex(4))
    run.rename(destination)
    return dict(deleted=True, trash=str(destination.relative_to(root)))


class Handler(BaseHTTPRequestHandler):
    """Expose only configured run summaries and their finalized videos."""

    def log_message(self, message_format: str, *args: object) -> None:
        """Keep polling requests out of the service log."""

    def do_POST(self) -> None:
        """Accept same-origin token-protected submit/stop operations."""
        if self.headers.get("X-Posttrain-Token") != self.server.token:
            self.send_error(403, "Refresh the dashboard before submitting")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            with self.server.mutation_lock:
                if self.path == "/api/submit":
                    result = jobs.submit(self.server.directory, request, self.server.runtime)
                elif self.path == "/api/metadata":
                    result = update_metadata(self.server.directory, request)
                elif self.path == "/api/delete":
                    result = trash_run(self.server.directory, request["run"])
                elif self.path == "/api/stop":
                    run = (self.server.directory / request["run"]).resolve()
                    if run not in experiment_paths(self.server.directory) or not (run / "job.json").is_file():
                        raise ValueError("Only supervisor-owned evaluation jobs can be stopped")
                    jobs.stop(run)
                    result = {"stopping": True}
                else:
                    self.send_error(404)
                    return
            self._json(result)
        except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
            body = json.dumps({"error": str(error)}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_GET(self) -> None:
        """Handle the page, JSON endpoints, and single-range MP4 requests."""
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, KeyError, IndexError, FileNotFoundError):
            self.send_error(404)

    def _json(self, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get(self) -> None:
        url = urlparse(self.path)
        if url.path == "/":
            body = Path(__file__).with_name("index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if url.path == "/api/gpus":
            self._json(gpu_snapshot())
            return
        if url.path == "/api/options":
            self._json(
                dict(
                    token=self.server.token,
                    checkpoints=jobs.scan_checkpoints(self.server.directory),
                    tasks=jobs.task_names(),
                    robocasa_tasks=jobs.task_names("robocasa"),
                    runtime=self.server.runtime,
                    defaults=jobs.BENCHMARK_DEFAULTS,
                )
            )
            return
        if url.path == "/api/state":
            commands = []
            for proc in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    command = proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
                    if "simulation_env.py" in command or "server_policy.py" in command:
                        commands.append(command)
                except OSError:
                    continue
            runs = []
            for path in experiment_paths(self.server.directory):
                try:
                    run = read_run(path, commands)
                    run["id"] = str(path.relative_to(self.server.directory))
                    if path.parent != self.server.directory:
                        run["label"] = path.parent.parent.name + " / " + path.name
                    runs.append(run)
                except (OSError, ValueError) as error:
                    runs.append(
                        dict(
                            id=str(path.relative_to(self.server.directory)),
                            label=path.name,
                            status="error",
                            tasks=[],
                            curve=[],
                            completed=0,
                            successes=0,
                            total=0,
                            config={},
                            error=str(error),
                        )
                    )
            priority = {"running": 0, "done": 1, "stopped": 2, "error": 3, "invalid": 4}
            runs.sort(
                key=lambda r: (
                    r["status"] != "running",
                    not r.get("pinned", False),
                    priority[r["status"]],
                    -r["successes"] / max(1, r["completed"]),
                    r["id"],
                )
            )
            self._json(
                dict(
                    updated_at=datetime.datetime.now().astimezone().isoformat(),
                    directory=str(self.server.directory),
                    runs=runs,
                )
            )
            return
        if url.path not in {"/api/videos", "/video", "/api/log", "/api/prompts"}:
            self.send_error(404)
            return
        query = parse_qs(url.query)
        run_id = query["run"][0]
        path = (self.server.directory / run_id).resolve()
        if path not in experiment_paths(self.server.directory):
            raise ValueError("Unknown run")
        if url.path == "/api/log":
            name = query["file"][0]
            log = (path / name).resolve()
            if log.parent != path or log.suffix != ".log":
                raise ValueError("Invalid log")
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                self._json({"text": stream.read().decode(errors="replace")})
            return
        task = query["task"][0]
        if task not in {log.stem for log in path.glob("task_*.log")}:
            raise ValueError("Unknown task")
        if url.path == "/api/prompts":
            prompt_file = (path / (task + ".prompts.jsonl")).resolve()
            if prompt_file.parent != path:
                raise ValueError("Invalid prompt path")
            records = []
            truncated = False
            if prompt_file.is_file():
                with prompt_file.open("rb") as stream:
                    offset = max(0, prompt_file.stat().st_size - 262144)
                    stream.seek(offset)
                    if offset:
                        stream.readline()
                        truncated = True
                    for line in stream:
                        try:
                            records.append(json.loads(line))
                        except (ValueError, UnicodeDecodeError):
                            continue
            self._json(dict(records=records[-200:], truncated=truncated or len(records) > 200))
            return
        directory = (path / "videos" / task).resolve()
        if not directory.is_relative_to(path):
            raise ValueError("Invalid video directory")
        if url.path == "/api/videos":
            videos = []
            ordered_videos = sorted(directory.glob("*_success[01].mp4"), key=lambda p: p.stat().st_mtime)
            for fallback_episode, video in reversed(list(enumerate(ordered_videos))):
                stat = video.stat()
                if not video.resolve().is_relative_to(directory) or not stat.st_size:
                    continue
                episode_match = re.search(r"episode_?(\d+)", video.stem)
                videos.append(
                    dict(
                        name=video.name,
                        episode=int(episode_match[1]) if episode_match else fallback_episode,
                        success=video.stem.endswith("success1"),
                        modified=stat.st_mtime,
                        bytes=stat.st_size,
                        url=f"/video?run={quote(run_id)}&task={quote(task)}&file={quote(video.name)}",
                    )
                )
            self._json(videos)
            return
        name = query["file"][0]
        video = (directory / name).resolve()
        if not video.is_relative_to(directory) or not re.fullmatch(r"[\w-]+_success[01]\.mp4", name):
            raise ValueError("Invalid video")
        size = video.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), end) if match[2] else end
            else:
                start = max(0, size - int(match[2]))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with video.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    """Start a loopback-only dashboard, suitable for SSH forwarding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=jobs.REPO / "results",
        help="Scan direct child experiment directories",
    )
    parser.add_argument("--port", type=int, default=8766)
    for name, default in jobs.DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), default=default)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.directory = directory
    server.runtime = {key: getattr(args, key) for key in jobs.DEFAULTS}
    server.token = secrets.token_urlsafe(32)
    server.mutation_lock = threading.Lock()
    print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
