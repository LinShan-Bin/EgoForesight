# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Record exact outgoing language inputs with episode and seed context.
"""Optional per-client JSONL log, one entry per episode/prompt change."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class PromptLog:
    """Record language only, without serializing image/state arrays."""

    def __init__(self) -> None:
        self.path = Path(os.environ["POSTTRAIN_PROMPT_LOG"]) if os.environ.get("POSTTRAIN_PROMPT_LOG") else None
        self.episode = None
        self.seed = None
        self.request_index = 0
        self.previous = {}

    def set_episode(self, episode: int | None, seed: int | None) -> None:
        """Begin a new episode, including when its prompt repeats the previous episode."""
        self.episode, self.seed = episode, seed
        self.request_index = 0
        self.previous = {}

    def record(self, query: dict) -> None:
        """Log the first actual request and any subsequent prompt changes."""
        if self.path is None:
            return
        payload = query.get("payload", query)
        for index, example in enumerate(payload.get("examples", [])):
            prompt = example.get("lang")
            if prompt is None or self.previous.get(index) == prompt:
                continue
            record = dict(
                timestamp=datetime.now(timezone.utc).isoformat(),
                task=os.environ.get("POSTTRAIN_TASK"),
                setting=os.environ.get("POSTTRAIN_SETTING"),
                episode=self.episode,
                seed=self.seed,
                request_index=self.request_index,
                sample_index=index,
                prompt=prompt,
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.previous[index] = prompt
        self.request_index += 1
