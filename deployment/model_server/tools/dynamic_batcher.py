# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Dynamically batch compatible policy-server inference requests.
"""Generic dynamic batching for policy requests that carry an ``examples`` list."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class _PendingRequest:
    """One request waiting for a dynamically formed inference batch."""

    payload: dict[str, Any]
    future: asyncio.Future
    enqueued_at: float
    sequence: int


class DynamicBatcher:
    """Merge compatible requests, call a policy once, and split batched outputs.

    Requests are compatible when they both contain a non-empty ``examples``
    list and every other payload argument is equal. Outputs may be arbitrarily
    nested dictionaries; NumPy arrays, lists, and tuples whose leading length
    matches the merged sample count are sliced back into per-request batches.
    """

    def __init__(
        self,
        policy: Any,
        max_batch_size: int,
        batch_wait_ms: float,
        log_interval: int = 100,
        trace_path: Optional[str] = None,
    ) -> None:
        if max_batch_size <= 1:
            raise ValueError(f"DynamicBatcher requires max_batch_size > 1, got {max_batch_size}")
        if batch_wait_ms < 0:
            raise ValueError(f"batch_wait_ms must be non-negative, got {batch_wait_ms}")
        if log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {log_interval}")

        self._policy = policy
        self._max_batch_size = max_batch_size
        self._batch_wait_s = batch_wait_ms / 1000.0
        self._log_interval = log_interval
        self._trace_path = trace_path
        self._inference_lock = threading.Lock()
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None
        self._batch_count = 0
        self._request_count = 0
        self._sample_count = 0
        self._next_sequence = 0
        self._trace_file = None

    async def start(self) -> None:
        """Start the worker in the current event loop."""
        if self._worker is not None:
            raise RuntimeError("Dynamic batch worker is already running")
        if self._trace_path:
            trace_path = Path(self._trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_file = trace_path.open("w", encoding="utf-8", buffering=1)
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._batch_worker())
        logging.info(
            "Dynamic batching enabled: max_batch_size=%d, batch_wait_ms=%.1f",
            self._max_batch_size,
            self._batch_wait_s * 1000.0,
        )
        self._write_trace(
            {
                "event": "config",
                "max_batch_size": self._max_batch_size,
                "batch_wait_ms": self._batch_wait_s * 1000.0,
            }
        )

    async def close(self) -> None:
        """Stop the worker."""
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        self._worker = None
        self._queue = None
        if self._trace_file is not None:
            self._trace_file.close()
        self._trace_file = None

    async def infer(self, payload: dict[str, Any]) -> Any:
        """Submit a policy payload and return its per-request output."""
        if self._queue is None or self._worker is None:
            raise RuntimeError("Dynamic batch worker is not running")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        enqueued_at = time.monotonic()
        sequence = self._next_sequence
        self._next_sequence += 1
        self._write_trace(
            {
                "event": "arrival",
                "sequence": sequence,
                "time_s": enqueued_at,
                "samples": self._payload_sample_count(payload),
            }
        )
        await self._queue.put(
            _PendingRequest(
                payload=payload,
                future=future,
                enqueued_at=enqueued_at,
                sequence=sequence,
            )
        )
        return await future

    @property
    def stats(self) -> dict[str, float]:
        """Return aggregate batching statistics for diagnostics and tests."""
        divisor = max(1, self._batch_count)
        return {
            "batches": self._batch_count,
            "requests": self._request_count,
            "samples": self._sample_count,
            "average_requests": self._request_count / divisor,
            "average_samples": self._sample_count / divisor,
        }

    async def _batch_worker(self) -> None:
        if self._queue is None:
            raise RuntimeError("Dynamic batch queue is not initialized")

        deferred = None
        while True:
            first = deferred if deferred is not None else await self._queue.get()
            deferred = None
            pending = [first]
            sample_count = self._payload_sample_count(first.payload)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._batch_wait_s

            if self._batch_wait_s == 0:
                # Let requests made ready in the same event-loop turn enqueue.
                await asyncio.sleep(0)

            while sample_count < self._max_batch_size:
                try:
                    if self._batch_wait_s == 0:
                        candidate = self._queue.get_nowait()
                    else:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            break
                        candidate = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except (asyncio.QueueEmpty, asyncio.TimeoutError):
                    break

                candidate_samples = self._payload_sample_count(candidate.payload)
                if sample_count + candidate_samples > self._max_batch_size:
                    deferred = candidate
                    break
                pending.append(candidate)
                sample_count += candidate_samples

            outputs, executed_batches = await asyncio.to_thread(self._execute_pending, pending)
            for request, output in zip(pending, outputs):
                if not request.future.done():
                    if isinstance(output, Exception):
                        request.future.set_exception(output)
                    else:
                        request.future.set_result(output)
                self._queue.task_done()

            for requests, samples, started_at, finished_at, sequences in executed_batches:
                self._record_batch(
                    requests,
                    samples,
                    finished_at - started_at,
                    finished_at - pending[0].enqueued_at,
                    started_at,
                    finished_at,
                    sequences,
                )

    def _execute_pending(
        self, pending: list[_PendingRequest]
    ) -> tuple[list[Any], list[tuple[int, int, float, float, list[int]]]]:
        groups: list[list[tuple[int, _PendingRequest]]] = []
        for index, request in enumerate(pending):
            for group in groups:
                if self._payloads_are_compatible(group[0][1].payload, request.payload):
                    group.append((index, request))
                    break
            else:
                groups.append([(index, request)])

        outputs: list[Any] = [RuntimeError("Missing dynamic batch output")] * len(pending)
        executed_batches = []
        for group in groups:
            requests = [request for _, request in group]
            started_at = time.monotonic()
            try:
                group_outputs = self._execute_group(requests)
            except Exception as error:
                logging.exception("Policy dynamic batch inference failed")
                group_outputs = [error] * len(requests)
            finished_at = time.monotonic()
            samples = sum(self._payload_sample_count(request.payload) for request in requests)
            executed_batches.append(
                (len(requests), samples, started_at, finished_at, [request.sequence for request in requests])
            )
            for (index, _), output in zip(group, group_outputs):
                outputs[index] = output
        return outputs, executed_batches

    def _execute_group(self, pending: list[_PendingRequest]) -> list[Any]:
        if len(pending) == 1:
            payload = pending[0].payload
            sample_counts = [self._payload_sample_count(payload)]
        else:
            payload, sample_counts = self._merge_payloads(pending)

        with self._inference_lock:
            output = self._policy.predict_action(**payload)
        return self._split_output(output, sample_counts)

    @staticmethod
    def _payload_sample_count(payload: dict[str, Any]) -> int:
        examples = payload.get("examples")
        return len(examples) if isinstance(examples, list) and examples else 1

    @classmethod
    def _payloads_are_compatible(cls, first: dict[str, Any], second: dict[str, Any]) -> bool:
        if not cls._has_batchable_examples(first) or not cls._has_batchable_examples(second):
            return False
        if first.keys() != second.keys():
            return False
        if cls._example_schema(first) != cls._example_schema(second):
            return False
        return all(cls._values_are_equal(first[key], second[key]) for key in first if key != "examples")

    @staticmethod
    def _has_batchable_examples(payload: dict[str, Any]) -> bool:
        examples = payload.get("examples")
        return isinstance(examples, list) and bool(examples)

    @staticmethod
    def _example_schema(payload: dict[str, Any]) -> tuple[Any, ...]:
        """Describe fields that must remain structurally uniform in one model batch."""
        schemas = []
        for example in payload["examples"]:
            if not isinstance(example, dict):
                schemas.append((type(example).__name__,))
                continue
            schema = [tuple(sorted(example))]
            images = example.get("image")
            schema.append(("image_views", len(images) if isinstance(images, (list, tuple)) else 1))
            for key in ("state", "robotwin_ee_pose", "action_mask"):
                if key in example:
                    schema.append((key, np.shape(example[key])))
            schemas.append(tuple(schema))
        if all(schema == schemas[0] for schema in schemas[1:]):
            return schemas[0]
        return ("mixed", *schemas)

    @classmethod
    def _values_are_equal(cls, first: Any, second: Any) -> bool:
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            try:
                return bool(np.array_equal(first, second))
            except (TypeError, ValueError):
                return False
        if isinstance(first, dict) and isinstance(second, dict):
            return first.keys() == second.keys() and all(cls._values_are_equal(first[key], second[key]) for key in first)
        if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
            return len(first) == len(second) and all(
                cls._values_are_equal(first_value, second_value) for first_value, second_value in zip(first, second)
            )
        try:
            return bool(first == second)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _merge_payloads(pending: list[_PendingRequest]) -> tuple[dict[str, Any], list[int]]:
        merged = dict(pending[0].payload)
        merged_examples = []
        sample_counts = []
        for request in pending:
            examples = request.payload["examples"]
            sample_counts.append(len(examples))
            merged_examples.extend(examples)
        merged["examples"] = merged_examples
        return merged, sample_counts

    @classmethod
    def _split_output(cls, output: Any, sample_counts: list[int]) -> list[Any]:
        total_samples = sum(sample_counts)
        outputs = []
        start = 0
        for sample_count in sample_counts:
            end = start + sample_count
            outputs.append(cls._slice_batched_value(output, start, end, total_samples))
            start = end
        return outputs

    @classmethod
    def _slice_batched_value(cls, value: Any, start: int, end: int, total_samples: int) -> Any:
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] == total_samples:
                return value[start:end]
            return value
        if isinstance(value, dict):
            return {key: cls._slice_batched_value(item, start, end, total_samples) for key, item in value.items()}
        if isinstance(value, list) and len(value) == total_samples:
            return value[start:end]
        if isinstance(value, tuple) and len(value) == total_samples:
            return value[start:end]
        return value

    def _record_batch(
        self,
        requests: int,
        samples: int,
        inference_s: float,
        total_s: float,
        started_at: float,
        finished_at: float,
        sequences: list[int],
    ) -> None:
        self._batch_count += 1
        self._request_count += requests
        self._sample_count += samples
        self._write_trace(
            {
                "event": "dispatch",
                "batch": self._batch_count,
                "sequences": sequences,
                "requests": requests,
                "samples": samples,
                "started_s": started_at,
                "finished_s": finished_at,
                "inference_s": inference_s,
                "total_s": total_s,
            }
        )
        if self._batch_count == 1 or self._batch_count % self._log_interval == 0:
            stats = self.stats
            logging.info(
                "Dynamic batch stats: batches=%d, last_requests=%d, last_samples=%d, "
                "last_inference_s=%.3f, last_total_s=%.3f, avg_requests=%.2f, avg_samples=%.2f",
                self._batch_count,
                requests,
                samples,
                inference_s,
                total_s,
                stats["average_requests"],
                stats["average_samples"],
            )

    def _write_trace(self, event: dict[str, Any]) -> None:
        if self._trace_file is not None:
            self._trace_file.write(json.dumps(event, sort_keys=True) + "\n")
