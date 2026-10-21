# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Implement bounded client-first policy.inference 1.0 transport for absolute R1 channels.

"""Portable MessagePack transport with cancellation and reset publication barriers.

This server offers absolute, one-sample observations and absolute actions only.
Unadvertised delta/history/extension/timeline modes are rejected, never guessed.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import msgpack
import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

LOGGER = logging.getLogger(__name__)
MAX_BYTES = 67_108_864
MAX_INT = 2**63 - 1
IDENTIFIER = re.compile(r"[A-Za-z0-9_.:/-]{1,128}\Z")
DTYPES = {
    name: np.dtype(code)
    for name, code in {"uint8": "u1", "int32": "<i4", "int64": "<i8", "float32": "<f4", "float64": "<f8"}.items()
}
COMMON = {"type", "session_id", "contract_id", "request_id", "context_epoch"}


class ProtocolError(ValueError):
    """A stable, sanitized public protocol error."""

    def __init__(self, message: str, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


def check(condition: bool, message: str, code: str = "invalid_request") -> None:
    """Reject malformed input without relying on optimizable assertions."""
    if not condition:
        raise ProtocolError(message, code)


def fields(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    """Require a map with exactly the declared core keys and optional extensions."""
    check(isinstance(value, dict), "Expected a map")
    allowed = required | (optional or set()) | {"extensions"}
    check(required <= value.keys() and value.keys() <= allowed, "Missing or unknown core fields")
    if "extensions" in value:
        check(isinstance(value["extensions"], dict), "extensions must be a map")


def identifier(value: Any) -> str:
    """Validate an opaque protocol ID without interpreting it as a path."""
    check(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None, "Invalid identifier")
    return value


def integer(value: Any, minimum: int = 0, maximum: int = MAX_INT) -> int:
    """Validate protocol integers, explicitly excluding Python booleans."""
    check(type(value) is int and minimum <= value <= maximum, "Invalid integer range or type")
    return value


def decode_tensor(value: Any, limit: int = MAX_BYTES) -> np.ndarray:
    """Validate lengths before creating a little-endian, read-only array view."""
    check(isinstance(value, dict) and set(value) == {"tensor"}, "Invalid tensor wrapper")
    tensor = value["tensor"]
    check(isinstance(tensor, dict) and set(tensor) == {"dtype", "shape", "data"}, "Invalid tensor fields")
    dtype = tensor["dtype"]
    check(isinstance(dtype, str) and dtype in DTYPES, "Unsupported tensor dtype")
    shape = tensor["shape"]
    check(isinstance(shape, list) and 1 <= len(shape) <= 8, "Invalid tensor rank")
    size = DTYPES[dtype].itemsize
    for dim in shape:
        size *= integer(dim, 1)
        check(size <= limit, "Tensor exceeds byte limit")
    check(isinstance(tensor["data"], bytes) and len(tensor["data"]) == size, "Tensor byte length mismatch")
    array = np.frombuffer(tensor["data"], dtype=DTYPES[dtype]).reshape(shape)
    check(np.isfinite(array).all(), "Nonfinite physical tensor")
    return array


def encode_tensor(value: np.ndarray) -> dict:
    """Encode a finite supported numeric array without MessagePack extensions."""
    array = np.asarray(value)
    check(array.dtype.name in DTYPES, "Unsupported output dtype")
    check(1 <= array.ndim <= 8 and all(d > 0 for d in array.shape), "Invalid output shape")
    check(np.isfinite(array).all(), "Nonfinite output")
    array = np.ascontiguousarray(array, dtype=DTYPES[array.dtype.name])
    return {"tensor": {"dtype": array.dtype.name, "shape": list(array.shape), "data": array.tobytes()}}


def _pairs(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        check(isinstance(key, str), "Map keys must be UTF-8 strings")
        check(key not in result, "Duplicate map key")
        result[key] = value
    return result


def _extension(code: int, data: bytes) -> None:
    raise ProtocolError("MessagePack extension types are forbidden")


def decode_message(raw: Any, limit: int = MAX_BYTES) -> dict:
    """Defensively parse one complete binary message under aggregate resource caps."""
    check(isinstance(raw, bytes), "Only binary WebSocket messages are supported")
    check(len(raw) <= limit, "Message exceeds negotiated byte limit")
    try:
        message = msgpack.unpackb(
            raw,
            raw=False,
            strict_map_key=False,
            object_pairs_hook=_pairs,
            ext_hook=_extension,
            max_str_len=limit,
            max_bin_len=limit,
            max_array_len=100_000,
            max_map_len=10_000,
            max_ext_len=0,
        )
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise ProtocolError("Invalid MessagePack payload") from exc
    nodes = entries = payload = 0

    def visit(value: Any, depth: int = 0) -> None:
        nonlocal nodes, entries, payload
        nodes += 1
        check(depth <= 32 and nodes <= 100_000, "Message nesting/node limit exceeded")
        if isinstance(value, dict):
            entries += len(value)
            check(entries <= 10_000, "Message map-entry limit exceeded")
            if "tensor" in value:
                payload += decode_tensor(value, limit).nbytes
                check(payload <= limit, "Aggregate tensor byte limit exceeded")
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)
        elif isinstance(value, float):
            check(math.isfinite(value), "Nonfinite scalar metadata")
        else:
            check(value is None or isinstance(value, (str, bytes, int, bool)), "Unsupported MessagePack value")

    visit(message)
    check(isinstance(message, dict), "Top-level message must be a map")
    return message


def validate_observation(observation: dict, contract: dict) -> dict[str, np.ndarray]:
    """Admit exactly one current measured sample per advertised absolute channel."""
    fields(observation, {"observation_id", "clock_id", "selected_ns", "task", "channels"}, {"anchors"})
    identifier(observation["observation_id"])
    identifier(observation["clock_id"])
    selected = integer(observation["selected_ns"], 1)
    check(isinstance(observation["task"], str) and len(observation["task"]) <= 65_536, "Invalid task text")
    check(observation.get("anchors", {}) == {}, "Absolute observations cannot carry delta anchors")
    check(isinstance(observation["channels"], dict), "channels must be a map")
    check(set(observation["channels"]) == set(contract["observations"]), "Missing or unknown observation channels")
    result, sample_ids = {}, set()
    for channel, descriptor in contract["observations"].items():
        samples = observation["channels"][channel]
        check(isinstance(samples, list) and len(samples) == 1, "This model requires exactly one current sample")
        sample = samples[0]
        fields(sample, {"sample_id", "measurement_ns", "arrival_ns", "sequence", "source_clock", "source_time", "value"})
        sample_id = identifier(sample["sample_id"])
        check(sample_id not in sample_ids, "Duplicate sample ID within observation")
        sample_ids.add(sample_id)
        measured, arrived = integer(sample["measurement_ns"], 1), integer(sample["arrival_ns"], 1)
        integer(sample["sequence"])
        identifier(sample["source_clock"])
        source_time = sample["source_time"]
        check(type(source_time) in (int, float, str), "Invalid source_time")
        check(measured <= arrived <= selected, "Invalid measurement/arrival/selection chronology")
        check(selected - measured <= descriptor["history"]["max_age_ns"], "Stale observation sample")
        value = decode_tensor(sample["value"])
        check(
            list(value.shape) == descriptor["shape"] and value.dtype.name == descriptor["dtype"],
            "Channel tensor mismatch",
        )
        result[channel] = value.copy()
        result[channel].flags.writeable = False
    return result


def validate_schedule(request: dict, contract: dict) -> dict:
    """Require the exact observation-aligned model horizon with explicit retiming."""
    fields(request, {"mode", "origin_ns", "offsets_ns", "anchors"})
    check(
        request["mode"] == "relative_sequence" and request["origin_ns"] is None,
        "Only relative_sequence with null origin is offered",
        "unsupported_schedule",
    )
    offsets = request["offsets_ns"]
    check(isinstance(offsets, list), "offsets_ns must be a list", "unsupported_schedule")
    for value in offsets:
        integer(value)
    check(
        offsets == contract["timing"]["offsets_ns"],
        "Offsets must equal the advertised model horizon",
        "unsupported_schedule",
    )
    check(request["anchors"] == {}, "Absolute actions cannot carry anchors")
    return {key: copy.deepcopy(request[key]) for key in ("mode", "origin_ns", "offsets_ns")}


@dataclass
class Inference:
    """Retain terminal status for cancellation races for the complete session lifetime."""

    request: dict
    observation_id: str
    status: str = "pending"
    task: asyncio.Task | None = None
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)


class Session:
    """One serialized receiver and publisher; model work continues in a separate thread."""

    def __init__(self, server: PolicyInferenceServer, socket: Any) -> None:
        self.server, self.socket = server, socket
        self.session_id = uuid.uuid4().hex
        self.contract_id = uuid.uuid4().hex
        self.epoch = 0
        self.clock_id: str | None = None
        self.ids: set[str] = set()
        self.observation_ids: set[str] = set()
        self.inferences: dict[str, Inference] = {}
        self.publication = asyncio.Lock()
        self.limit = server.max_message_bytes
        self.open = False
        self.operator_confirmed = False

    async def send(self, message: dict) -> None:
        """Send one complete binary response within the negotiated byte ceiling."""
        payload = msgpack.packb(message, use_bin_type=True)
        check(len(payload) <= self.limit, "Response exceeds negotiated byte limit")
        await self.socket.send(payload)

    async def handshake(self) -> None:
        """Accept the client-first hello only after service initialization/warmup."""
        raw = await asyncio.wait_for(self.socket.recv(), self.server.handshake_timeout)
        hello = decode_message(raw, self.limit)
        fields(hello, {"type", "protocol", "versions", "request_id", "limits", "supported_extensions"})
        check(hello["type"] == "hello" and hello["protocol"] == "policy.inference", "Unsupported protocol")
        versions = hello["versions"]
        check(isinstance(versions, list) and bool(versions), "Missing protocol versions")
        for pair in versions:
            check(isinstance(pair, list) and len(pair) == 2, "Invalid protocol version")
            for value in pair:
                integer(value)
        check([1, 0] in versions, "Version 1.0 must be offered")
        request_id = identifier(hello["request_id"])
        fields(hello["limits"], {"max_message_bytes", "max_in_flight"})
        client_bytes = integer(hello["limits"]["max_message_bytes"], 1024, MAX_BYTES)
        integer(hello["limits"]["max_in_flight"], 1, 64)
        self.limit = min(self.limit, client_bytes)
        extensions = hello["supported_extensions"]
        check(isinstance(extensions, list), "supported_extensions must be a list")
        for extension in extensions:
            identifier(extension)
        check(len(set(extensions)) == len(extensions), "Duplicate supported extension")
        check(set(self.server.contract["required_extensions"]) <= set(extensions), "Missing required extension")
        await self.send(
            {
                "type": "welcome",
                "protocol": "policy.inference",
                "version": [1, 0],
                "request_id": request_id,
                "session_id": self.session_id,
                "contract_id": self.contract_id,
                "limits": {"max_message_bytes": self.limit, "max_in_flight": 1},
                "contract": self.server.contract,
                "memory": "stateless",
                "ready": True,
                "service_kind": self.server.service_kind,
                "provenance": self.server.provenance,
            }
        )
        self.ids.add(request_id)
        self.open = True

    def common(self, request: dict) -> dict:
        """Copy correlation without leaking request extensions into response semantics."""
        return {key: request[key] for key in COMMON - {"type"}}

    async def terminal(self, entry: Inference, response: dict) -> bool:
        """Arbitrate completion and cancellation under the same publication lock."""
        async with self.publication:
            if self.open and entry.status == "pending":
                await self.send({**self.common(entry.request), "observation_id": entry.observation_id, **response})
                entry.status = "completed"
                return True
        return False

    async def error(self, entry: Inference, code: str, message: str) -> None:
        """Publish a correlated inference error, unless cancellation already won."""
        await self.terminal(entry, {"type": "infer_error", "error": {"code": code, "message": message[:4096]}})

    async def infer(self, request: dict, received_ns: int) -> None:
        """Validate first, then atomically reserve the sole model worker without a queue."""
        observation = request.get("observation")
        check(isinstance(observation, dict), "Missing observation correlation")
        observation_id = identifier(observation.get("observation_id"))
        # Retain correlation/terminal status, not megabytes of immutable RGB for the session lifetime.
        entry = Inference({key: request[key] for key in COMMON}, observation_id)
        self.inferences[request["request_id"]] = entry
        try:
            fields(request, COMMON | {"observation", "action_request", "budget_ns"})
            check(request["context_epoch"] == self.epoch, "Wrong context epoch", "context_invalid")
            budget = integer(request["budget_ns"], 1)
            values = validate_observation(observation, self.server.contract)
            schedule = validate_schedule(request["action_request"], self.server.contract)
            check(observation_id not in self.observation_ids, "Observation ID reused")
            check(self.clock_id in (None, observation["clock_id"]), "Client clock changed within session")
            self.clock_id = observation["clock_id"]
            self.observation_ids.add(observation_id)
            check(time.monotonic_ns() - received_ns < budget, "Server budget exhausted", "deadline_exceeded")
            check(self.server.worker is None, "Model worker is occupied; no inference queue", "busy")
        except ProtocolError as exc:
            await self.error(entry, exc.code, str(exc))
            return
        entry.task = asyncio.create_task(
            self.compute(entry, values, observation["task"], schedule, received_ns + budget)
        )
        self.server.worker = entry.task

    async def compute(self, entry: Inference, values: dict, task: str, schedule: dict, deadline: int) -> None:
        """Bound publication by the budget while retaining capacity until GPU work actually stops."""
        started = time.monotonic_ns()
        future = asyncio.get_running_loop().run_in_executor(
            self.server.executor, self.server.policy.predict, values, task
        )
        try:
            done, _ = await asyncio.wait({future}, timeout=max(0, (deadline - time.monotonic_ns()) / 1e9))
            if not done:
                await self.error(entry, "deadline_exceeded", "Server inference budget exhausted")
                # Cancellation is publication suppression, not GPU preemption. Reset waits here too.
                await asyncio.shield(future)
                return
            output = future.result()
            check(
                isinstance(output, dict) and set(output) == set(self.server.contract["actions"]), "Wrong output channels"
            )
            actions = {}
            for key, desc in self.server.contract["actions"].items():
                array = np.asarray(output[key])
                expected = (len(schedule["offsets_ns"]), *desc["shape"])
                check(array.shape == expected and array.dtype.name == desc["dtype"], "Wrong output shape/dtype")
                if desc["quantity"] == "pose":
                    check(
                        np.all(np.abs(np.linalg.norm(array[:, 3:], axis=-1) - 1) <= 0.001), "Invalid output quaternion"
                    )
                actions[key] = encode_tensor(array)
            if self.server.publication_guard is not None:
                label = (
                    f"session={self.session_id} epoch={entry.request['context_epoch']} "
                    f"request={entry.request['request_id']} observation={entry.observation_id}"
                )
                accepted = await self.server.publication_guard.review(
                    values, output, label, not self.operator_confirmed, entry.cancelled, deadline
                )
                if not accepted:
                    return
            if time.monotonic_ns() >= deadline:
                await self.error(entry, "deadline_exceeded", "Server inference budget exhausted")
                return
            published = await self.terminal(
                entry,
                {
                    "type": "infer_result",
                    "prediction_id": uuid.uuid4().hex,
                    "schedule": schedule,
                    "actions": actions,
                    "server_timing": {"queue_ns": 0, "compute_ns": time.monotonic_ns() - started},
                },
            )
            if published:
                self.operator_confirmed = True
        except ConnectionClosed:
            self.open = False
        except ProtocolError as exc:
            LOGGER.warning("Prediction rejected (request %s): %s", entry.request["request_id"], exc)
            code = "deadline_exceeded" if exc.code == "deadline_exceeded" else "inference_failed"
            try:
                await self.error(entry, code, str(exc))
            except ConnectionClosed:
                self.open = False
        except Exception:
            LOGGER.exception("Inference failed (request %s)", entry.request["request_id"])
            try:
                await self.error(entry, "inference_failed", "Model execution or output validation failed")
            except ConnectionClosed:
                self.open = False
        finally:
            # The worker slot is process-wide, including work from disconnected clients.
            self.server.worker = None

    async def dispatch(self, request: dict, received_ns: int) -> None:
        """Route requests while an admitted inference runs independently."""
        check(COMMON <= request.keys(), "Missing request correlation")
        check(
            request["session_id"] == self.session_id and request["contract_id"] == self.contract_id,
            "Wrong session or contract",
        )
        request_id = identifier(request["request_id"])
        integer(request["context_epoch"])
        check(request_id not in self.ids, "Request ID reused")
        check(len(self.ids) < self.server.max_requests, "Session request limit reached; reconnect")
        self.ids.add(request_id)
        kind = request["type"]
        if kind == "infer":
            await self.infer(request, received_ns)
        elif kind == "cancel":
            fields(request, COMMON | {"target_request_id"})
            check(request["context_epoch"] == self.epoch, "Wrong cancel epoch")
            target = identifier(request["target_request_id"])
            check(target in self.inferences, "Unknown cancel target")
            entry = self.inferences[target]
            check(entry.request["context_epoch"] == self.epoch, "Cancel target belongs to another epoch")
            async with self.publication:
                check(entry.status != "cancelled", "Inference was already cancelled")
                status = "already_completed" if entry.status == "completed" else "cancelled"
                await self.send(
                    {**self.common(request), "type": "cancel_ack", "target_request_id": target, "status": status}
                )
                if status == "cancelled":
                    entry.status = "cancelled"
                    entry.cancelled.set()
        elif kind == "reset_context":
            fields(request, COMMON)
            check(request["context_epoch"] == self.epoch + 1, "Reset epoch must increment by one")
            check(all(entry.status != "pending" for entry in self.inferences.values()), "Cancel/drain before reset")
            work = [entry.task for entry in self.inferences.values() if entry.task is not None and not entry.task.done()]
            if work:
                # wait_for must not cancel the underlying GPU worker or release its slot early.
                await asyncio.wait_for(asyncio.shield(asyncio.gather(*work)), self.server.reset_timeout)
            self.epoch += 1
            self.operator_confirmed = False
            await self.send({**self.common(request), "type": "context_ready"})
        else:
            raise ProtocolError("Unknown message type")

    async def run(self) -> None:
        """End malformed/expired sessions; revoke publication on every disconnect path."""
        try:
            await self.handshake()
            expires = time.monotonic() + self.server.session_lifetime
            while True:
                raw = await asyncio.wait_for(self.socket.recv(), max(0, expires - time.monotonic()))
                received = time.monotonic_ns()
                await self.dispatch(decode_message(raw, self.limit), received)
        except ConnectionClosed:
            pass
        except (ProtocolError, asyncio.TimeoutError) as exc:
            if self.open:
                async with self.publication:
                    self.open = False
                    try:
                        await self.send(
                            {
                                "type": "session_error",
                                "session_id": self.session_id,
                                "contract_id": self.contract_id,
                                "error": {"code": "invalid_request", "message": str(exc)[:4096] or "Session timed out"},
                            }
                        )
                    except (ConnectionClosed, ProtocolError):
                        pass
            await self.socket.close(code=1008, reason="Protocol/session validation failed")
        finally:
            self.open = False
            for entry in self.inferences.values():
                entry.cancelled.set()


class PolicyInferenceServer:
    """Bounded stateless service, with one global GPU worker and no inference queue."""

    def __init__(
        self,
        policy: Any,
        provenance: dict,
        service_kind: str = "production",
        max_message_bytes: int = MAX_BYTES,
        max_requests: int = 10_000,
        session_lifetime: float = 3600,
        max_sessions: int = 8,
        handshake_timeout: float = 10,
        reset_timeout: float = 60,
        publication_guard: Any = None,
    ) -> None:
        self.policy = policy
        self.contract = copy.deepcopy(policy.contract)
        self.provenance = copy.deepcopy(provenance)
        self.service_kind = service_kind
        self.publication_guard = publication_guard
        integer(max_message_bytes, 1024, MAX_BYTES)
        integer(max_requests, 2)
        integer(max_sessions, 1)
        check(service_kind in {"production", "test"}, "Invalid service kind")
        check(
            all(math.isfinite(t) and t > 0 for t in (session_lifetime, handshake_timeout, reset_timeout)),
            "Invalid timeout",
        )
        check(self.contract["required_extensions"] == [], "This adapter does not implement semantic extensions")
        for desc in self.contract["observations"].values():
            check(
                desc["representation"] == "absolute"
                and desc["history"]["min_samples"] == desc["history"]["max_samples"] == 1,
                "This server implements single current absolute observations",
            )
        check(
            all(desc["representation"] == "absolute" for desc in self.contract["actions"].values()),
            "Absolute outputs required",
        )
        self.max_message_bytes, self.max_requests = max_message_bytes, max_requests
        self.session_lifetime, self.max_sessions = session_lifetime, max_sessions
        self.handshake_timeout, self.reset_timeout = handshake_timeout, reset_timeout
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r1pro-inference")
        self.worker: asyncio.Task | None = None
        self.sessions = 0

    async def handler(self, socket: Any) -> None:
        """Serve a connection or close it before welcome when capacity is exhausted."""
        if self.sessions >= self.max_sessions:
            await socket.close(code=1013, reason="Session capacity exhausted")
            return
        self.sessions += 1
        try:
            await Session(self, socket).run()
        finally:
            self.sessions -= 1

    async def run(self, host: str, port: int) -> None:
        """Listen with compression disabled and bounded WebSocket receive buffers."""
        try:
            async with serve(
                self.handler, host, port, compression=None, max_size=self.max_message_bytes, max_queue=1, close_timeout=5
            ) as server:
                LOGGER.info("policy.inference 1.0 ready on ws://%s:%s", host, port)
                await server.serve_forever()
        finally:
            if self.worker is not None:
                await asyncio.shield(self.worker)
            self.executor.shutdown(wait=True)
