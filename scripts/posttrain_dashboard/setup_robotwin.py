# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Relocate copied RoboTwin assets and install tracked evaluation patches.
"""Run once after syncing the reference RoboTwin checkout/environment to eval."""

import argparse
import re
import shutil
from pathlib import Path


def configure(root: Path) -> list[str]:
    """Back up and patch only a dedicated user simulator checkout; reject active evaluation clients."""
    root = root.resolve(strict=True)
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if str(root / "script/eval_policy.py") in command:
                raise RuntimeError("Stop RoboTwin clients before modifying the simulator")
        except OSError:
            continue
    backup = root / ".posttrain_originals"
    changes = []

    def save(path: Path, content: str) -> None:
        old = path.read_text() if path.exists() else None
        if old == content:
            return
        original = backup / path.relative_to(root)
        if path.exists() and not original.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, original)
        path.write_text(content)
        changes.append(str(path.relative_to(root)))

    # Reference configurations embed another user's absolute asset root.
    # Relocate only known embodiment files which actually exist in this copy.
    pattern = re.compile(r"/[^\s\'\"]+/assets/embodiments/[^\s\'\"]+")
    for path in (root / "assets/embodiments").rglob("*.yml"):
        content = path.read_text()
        for old in set(pattern.findall(content)):
            suffix = old.split("/assets/embodiments/", 1)[1]
            target = root / "assets/embodiments" / suffix
            if not target.exists():
                raise FileNotFoundError(f"Missing relocated asset: {target} (from {path})")
            content = content.replace(old, str(target))
        save(path, content)
    repo = Path(__file__).resolve().parents[2]
    for name, destination in [
        ("_base_task.py", "envs/_base_task.py"),
        ("eval_policy.py", "script/eval_policy.py"),
    ]:
        source = repo / "third_party/robotwin_client/robotwin_replace_file" / name
        save(root / destination, source.read_text())
    return changes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-path", type=Path, required=True)
    print("\n".join(configure(parser.parse_args().robotwin_path)))
