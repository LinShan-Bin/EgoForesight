# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoForesight modifications by Chubin Zhang, 2026.
# Modification: Serve bounded async observations with raw RGB or size-validated JPEG images.

"""Async R1 transport, separate from the portable policy.inference wire format.

One endpoint belongs to one robot: the protocol has no client/session pairing ID.
State rows are arm_left, arm_right, hand_left, hand_right (7+7+22+22).
Action rows use the same order, substituting absolute XYZW poses in EEF mode.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
from PIL import Image
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from deployment.model_server.tools import msgpack_numpy
from deployment.model_server.tools.policy_inference import MAX_BYTES, ProtocolError, check, integer

LOGGER = logging.getLogger(__name__)
PROTOCOL = "policy.inference.async"
VERSION = [1, 0]
ACTION_DIM = 58
STATE_SLICES = {
    "arm_left": slice(0, 7),
    "arm_right": slice(7, 14),
    "hand_left": slice(14, 36),
    "hand_right": slice(36, 58),
}
IMAGE_KEYS = {"head", "wrist_left", "wrist_right"}
OBSERVATION_FIELDS = {"type", "generation", "obs_timestamp_ns", "prompt", "state", "images"}
JPEG_FIELDS = {"image_encoding", "image_color", "image_sizes"}


def exact_fields(value: Any, required: set) -> None:
    """This protocol defines exact maps, not the sync protocol's extensions field."""
    check(isinstance(value, dict) and value.keys() == required, "Missing or unknown core fields")


def decode_message(raw: Any, limit: int = MAX_BYTES) -> dict:
    """Decode the repository's numpy map format without pickle or unchecked allocations."""
    check(isinstance(raw, bytes) and len(raw) <= limit, "Expected a bounded binary MessagePack message")

    def pairs(items: list[tuple]) -> Any:
        result = {}
        for key, value in items:
            check(isinstance(key, (str, bytes)), "Invalid map key")
            check(key not in result, "Duplicate map key")
            result[key] = value
        if b"__ndarray__" in result:
            exact_fields(result, {b"__ndarray__", b"dtype", b"shape", b"data"})
            check(result[b"__ndarray__"] is True, "Invalid numpy marker")
            dtype = result[b"dtype"]
            check(isinstance(dtype, str) and dtype in {"<f4", ">f4", "|u1"}, "Unsupported numpy dtype")
            shape = result[b"shape"]
            check(isinstance(shape, list) and 1 <= len(shape) <= 3, "Invalid numpy rank")
            size = np.dtype(dtype).itemsize
            for dim in shape:
                size *= integer(dim, 1)
                check(size <= limit, "Tensor exceeds byte limit")
            data = result[b"data"]
            check(isinstance(data, bytes) and len(data) == size, "Tensor byte length mismatch")
            array = np.frombuffer(data, dtype=dtype).reshape(shape)
            check(np.isfinite(array).all(), "Nonfinite tensor")
            return array
        check(all(isinstance(key, str) for key in result), "Core map keys must be strings")
        return result

    def extension(code: int, data: bytes) -> None:
        raise ProtocolError("Unsupported MessagePack extension tag")

    try:
        message = msgpack.unpackb(
            raw,
            raw=False,
            strict_map_key=False,
            object_pairs_hook=pairs,
            ext_hook=extension,
            max_str_len=65_536,
            max_bin_len=limit,
            max_array_len=100_000,
            max_map_len=1000,
            max_ext_len=0,
        )
    except (ValueError, TypeError, OverflowError, msgpack.UnpackException) as exc:
        raise ProtocolError("Invalid MessagePack/numpy payload") from exc
    nodes = 0

    def visit(value: Any, depth: int = 0) -> None:
        nonlocal nodes
        nodes += 1
        check(nodes <= 100_000 and depth <= 32, "Message nesting/node limit exceeded")
        if isinstance(value, dict):
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)
        elif isinstance(value, float):
            check(math.isfinite(value), "Nonfinite metadata")
        else:
            check(value is None or isinstance(value, (str, bytes, int, bool, np.ndarray)), "Unsupported value")

    visit(message)
    check(isinstance(message, dict), "Top-level message must be a map")
    return message


def decode_images(
    message: dict,
    contract: dict,
    expected_shapes: dict[str, tuple[int, ...]] | None,
    limit: int = MAX_BYTES,
) -> dict[str, np.ndarray]:
    """Decode owned RGB frames, bounding declared allocations before JPEG decompression.

    Raw frames retain the deployment contract. JPEG frames declare independent
    per-camera sizes; the session locks those sizes after its first valid input.
    Neither path resizes, pads, rotates or substitutes a previous observation.
    """
    compressed = "image_encoding" in message
    exact_fields(message, OBSERVATION_FIELDS | JPEG_FIELDS if compressed else OBSERVATION_FIELDS)
    exact_fields(message["images"], IMAGE_KEYS)
    if compressed:
        check(
            isinstance(message["image_encoding"], str) and message["image_encoding"] == "jpeg", "Expected jpeg encoding"
        )
        check(isinstance(message["image_color"], str) and message["image_color"] == "RGB", "Expected RGB image color")
        exact_fields(message["image_sizes"], IMAGE_KEYS)

    shapes = {}
    decoded_bytes = ACTION_DIM * np.dtype(np.float32).itemsize
    for key, image in message["images"].items():
        if compressed:
            check(isinstance(image, bytes) and 0 < len(image) <= limit, f"Expected bounded JPEG bytes for {key}")
            size = message["image_sizes"][key]
            check(isinstance(size, list) and len(size) == 2, f"Expected image_sizes.{key} [height, width]")
            shape = (integer(size[0], 1), integer(size[1], 1), 3)
        else:
            check(
                isinstance(image, np.ndarray)
                and image.dtype == np.uint8
                and list(image.shape) == contract["observations"][key]["shape"],
                f"Image shape/dtype differs from deployment configuration for {key}",
            )
            shape = image.shape
        decoded_bytes += shape[0] * shape[1] * 3
        check(decoded_bytes <= limit, "Decoded observation exceeds byte limit")
        shapes[key] = shape
    check(expected_shapes is None or shapes == expected_shapes, "Image dimensions must remain fixed for the session")

    images = {}
    for key, image in message["images"].items():
        if compressed:
            try:
                with Image.open(io.BytesIO(image), formats=["JPEG"]) as decoded:
                    height, width, _ = shapes[key]
                    # Inspect the header before load() can allocate the full pixel buffer.
                    check(decoded.size == (width, height), f"JPEG dimensions differ from image_sizes.{key}")
                    check(decoded.mode == "RGB", f"Expected three-channel RGB JPEG for {key}")
                    decoded.load()
                    images[key] = np.array(decoded, dtype=np.uint8, copy=True)
            except (
                OSError,
                ValueError,
                SyntaxError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ) as exc:
                raise ProtocolError(f"Invalid JPEG image for {key}") from exc
        else:
            images[key] = image.copy()
    return images


def validate_contract(contract: dict) -> tuple[str, int]:
    """Fail closed on history/delta/extension models that cannot fit this fixed R1 profile."""
    observations, actions = contract["observations"], contract["actions"]
    check(contract["required_extensions"] == [], "Async profile cannot negotiate semantic extensions")
    check(set(observations) == set(STATE_SLICES) | IMAGE_KEYS, "Async requires four joint channels and three cameras")
    for key, desc in observations.items():
        check(desc["representation"] == "absolute", "Async requires absolute observations")
        check(desc["history"]["min_samples"] == desc["history"]["max_samples"] == 1, "Async requires current state only")
        if key in STATE_SLICES:
            part = STATE_SLICES[key]
            check(desc["shape"] == [part.stop - part.start] and desc["dtype"] == "float32", "Invalid state profile")
            check(desc["quantity"] == "joint_position", "Async state must contain measured joints")
        else:
            check(
                desc["dtype"] == "uint8" and desc["quantity"] == "rgb" and desc["color"] == "RGB", "Invalid RGB profile"
            )
            check(len(desc["shape"]) == 3 and desc["shape"][-1] == 3, "Invalid image shape")
    mode = "joint" if "arm_left" in actions else "eef"
    order = (
        f"{'arm' if mode == 'joint' else 'eef'}_left",
        f"{'arm' if mode == 'joint' else 'eef'}_right",
        "hand_left",
        "hand_right",
    )
    check(set(actions) == set(order), "Invalid async action channels")
    for key, width in zip(order, (7, 7, 22, 22), strict=True):
        desc = actions[key]
        check(
            desc["representation"] == "absolute" and desc["dtype"] == "float32" and desc["shape"] == [width],
            "Invalid action profile",
        )
        if key.startswith("eef_"):
            check(
                desc["quantity"] == "pose" and desc["parent_frame"] == "torso_link4" and desc["convention"] == "xyzw",
                "Async EEF profile must use absolute torso XYZW poses",
            )
        else:
            check(desc["quantity"] == "joint_position", "Async joint targets must be absolute radians")
    horizon = integer(len(contract["timing"]["offsets_ns"]), 1, 100)
    return mode, horizon


@dataclass
class Observation:
    """One immutable input snapshot and its revocable generation publication token."""

    generation: int
    timestamp: int
    channels: dict[str, np.ndarray]
    prompt: str
    cancelled: asyncio.Event


class AsyncSession:
    """One sender/receiver pair, one pending observation and one pending action chunk."""

    def __init__(self, server: AsyncPolicyInferenceServer, horizon: int) -> None:
        self.server, self.horizon = server, horizon
        self.sockets: dict[str, Any] = {}
        self.receiver_ready = False
        self.receiver_handshake = asyncio.Event()
        self.generation: int | None = None
        self.last_reset: int | None = None
        self.last_timestamp = -1
        self.image_shapes: dict[str, tuple[int, ...]] | None = None
        self.requests = 0
        self.open = True
        self.operator_confirmed = False
        self.cancelled = asyncio.Event()
        self.pending: Observation | None = None
        self.action: tuple[Observation, dict, Any] | None = None
        self.observed, self.produced = asyncio.Event(), asyncio.Event()
        self.expires = time.monotonic() + server.session_lifetime
        self.sending: asyncio.Task | None = None
        server.track(self.compute())
        server.track(self.publish())

    def invalidate(self) -> None:
        """Revoke queued and in-flight work synchronously before yielding to socket IO."""
        self.cancelled.set()
        self.cancelled = asyncio.Event()
        self.pending = self.action = None
        self.operator_confirmed = False

    async def send(self, socket: Any, message: dict) -> None:
        """Bound backpressure by closing a stalled socket, not cancelling a live send."""
        payload = msgpack_numpy.packb(message, use_bin_type=True)
        check(len(payload) <= self.server.max_message_bytes, "Output exceeds byte limit")
        task = asyncio.create_task(socket.send(payload))
        try:
            done, _ = await asyncio.wait({task}, timeout=self.server.send_timeout)
            if not done:
                await socket.close(code=1013, reason="Async send timed out")
            await task
        finally:
            if not task.done():
                await socket.close()
                await asyncio.gather(task, return_exceptions=True)

    async def finish(self, error: ProtocolError | None = None) -> None:
        """Close both roles; in-flight GPU work drains without publishing or slot reuse."""
        if not self.open:
            return
        self.open = False
        self.invalidate()
        self.observed.set()
        self.produced.set()
        self.receiver_handshake.set()

        async def close(socket: Any) -> None:
            try:
                if error is not None:
                    await self.send(
                        socket, {"type": "error", "code": error.code, "message": str(error)[:4096], "fatal": True}
                    )
            except (ConnectionClosed, ProtocolError):
                pass
            finally:
                await socket.close(code=1008 if error else 1000)

        await asyncio.gather(*(close(socket) for socket in list(self.sockets.values())))
        if self.server.session is self:
            self.server.session = None

    async def reset_model(self) -> None:
        """Drain prior model work before clearing its context on the same worker."""
        # All invocations, including those from retired sessions, share this lock/executor.
        async with self.server.model_lock:
            await asyncio.get_running_loop().run_in_executor(self.server.executor, self.server.policy.reset)

    async def reset(self, generation: int) -> None:
        """Establish a fresh generation exactly once, with no old publication after ACK."""
        if self.last_reset == generation and self.generation == generation:
            return
        check(
            self.generation is None or generation > self.generation, "Reset must advance generation", "wrong_generation"
        )
        self.invalidate()
        self.generation, self.last_timestamp = generation, -1
        await self.revoke_send()
        await asyncio.wait_for(self.reset_model(), self.server.reset_timeout)
        self.last_reset = generation

    async def revoke_send(self) -> None:
        """Close a stalled receiver so a queued send cannot cross a control barrier."""
        if self.sending is not None and not self.sending.done():
            receiver = self.sockets.get("receiver")
            if receiver is not None:
                await receiver.close(code=1012, reason="Publication revoked; reconnect receiver")

    def observe(self, message: dict) -> None:
        """Replace the sole pending observation after validating every current channel."""
        exact_fields(message, OBSERVATION_FIELDS | JPEG_FIELDS if "image_encoding" in message else OBSERVATION_FIELDS)
        generation = integer(message["generation"])
        timestamp = integer(message["obs_timestamp_ns"])
        check(
            self.generation in (None, generation),
            "Observation requires a reset for the new generation",
            "wrong_generation",
        )
        check(timestamp > self.last_timestamp, "Observation timestamp must strictly increase")
        check(isinstance(message["prompt"], str) and len(message["prompt"]) <= 65_536, "Invalid prompt")
        state = message["state"]
        check(
            isinstance(state, np.ndarray) and state.shape == (ACTION_DIM,) and state.dtype.name == "float32",
            "Expected float32 state[58]",
        )
        check(np.isfinite(state).all(), "Nonfinite state")
        channels = {key: state[part].copy() for key, part in STATE_SLICES.items()}
        images = decode_images(message, self.server.policy.contract, self.image_shapes, self.server.max_message_bytes)
        channels.update(images)
        for value in channels.values():
            value.flags.writeable = False
        self.image_shapes = {key: image.shape for key, image in images.items()}
        self.generation, self.last_timestamp = generation, timestamp
        self.pending = Observation(generation, timestamp, channels, message["prompt"], self.cancelled)
        self.observed.set()

    async def dispatch(self, socket: Any, message: dict) -> None:
        """Route sender-only observations and generation control messages."""
        self.requests += 1
        check(self.requests <= self.server.max_requests, "Session request limit reached; reconnect")
        kind = message.get("type")
        if kind == "observation":
            self.observe(message)
        elif kind in ("reset", "stop"):
            exact_fields(message, {"type", "generation"})
            generation = integer(message["generation"])
            if kind == "reset":
                await self.reset(generation)
                await self.send(socket, {"type": "reset_ack", "generation": generation})
            else:
                check(self.generation in (None, generation), "Wrong stop generation", "wrong_generation")
                self.invalidate()
                await self.revoke_send()
                await self.send(socket, {"type": "stop_ack", "generation": generation})
                await self.finish()
        else:
            raise ProtocolError("Unknown sender message type")

    def valid(self, observation: Observation) -> bool:
        """A result may outlive its computation, but never its sender/generation token."""
        return self.open and not observation.cancelled.is_set() and "sender" in self.sockets

    def validate_actions(self, output: Any) -> dict[str, np.ndarray]:
        """Validate the complete physical model chunk before truncating the horizon."""
        check(isinstance(output, dict) and set(output) == set(self.server.action_order), "Wrong output channels")
        result = {}
        for key in self.server.action_order:
            value = output[key]
            desc = self.server.policy.contract["actions"][key]
            check(
                isinstance(value, np.ndarray)
                and value.shape == (self.server.model_horizon, *desc["shape"])
                and value.dtype.name == "float32",
                "Invalid output shape/dtype",
            )
            check(np.isfinite(value).all(), "Nonfinite output")
            if desc["quantity"] == "pose":
                check(np.all(np.abs(np.linalg.norm(value[:, 3:], axis=-1) - 1) <= 0.001), "Invalid output quaternion")
            result[key] = value.copy()
        return result

    async def compute(self) -> None:
        """Run one model call at a time without blocking either network direction."""
        try:
            while self.open:
                await self.observed.wait()
                self.observed.clear()
                observation, self.pending = self.pending, None
                if observation is None or not self.valid(observation):
                    continue
                receiver = self.sockets.get("receiver")
                try:
                    async with self.server.model_lock:
                        if not self.valid(observation):
                            continue
                        output = await asyncio.get_running_loop().run_in_executor(
                            self.server.executor, self.server.policy.predict, observation.channels, observation.prompt
                        )
                    if not self.valid(observation):
                        continue
                    actions = self.validate_actions(output)
                    if receiver is None or self.sockets.get("receiver") is not receiver:
                        continue
                    guard = self.server.publication_guard
                    if guard is not None:
                        accepted = await guard.review(
                            observation.channels,
                            actions,
                            f"async generation={observation.generation} observation={observation.timestamp}",
                            not self.operator_confirmed,
                            observation.cancelled,
                            time.monotonic_ns() + int(max(0, self.expires - time.monotonic()) * 1e9),
                        )
                        if not accepted:
                            continue
                    if not self.valid(observation) or receiver is None or self.sockets.get("receiver") is not receiver:
                        continue
                    rows = np.concatenate([actions[key][: self.horizon] for key in self.server.action_order], axis=1)
                    check(rows.shape == (self.horizon, ACTION_DIM) and np.isfinite(rows).all(), "Invalid action rows")
                    message = {
                        "type": "action",
                        "generation": observation.generation,
                        "obs_timestamp_ns": observation.timestamp,
                        "actions": rows,
                    }
                    self.operator_confirmed = True
                    self.action = (observation, message, receiver)
                    self.produced.set()
                except Exception:
                    if self.valid(observation):
                        LOGGER.exception("Async inference/output validation failed")
                        await self.finish(
                            ProtocolError("Model execution or action validation failed", "inference_failed")
                        )
        finally:
            self.pending = None

    async def publish(self) -> None:
        """Publish only live tokens, keeping socket backpressure off the input loop."""
        while self.open:
            await self.produced.wait()
            self.produced.clear()
            action, self.action = self.action, None
            if action is None:
                continue
            observation, message, receiver = action
            await self.receiver_handshake.wait()
            if not self.valid(observation) or not self.receiver_ready or self.sockets.get("receiver") is not receiver:
                continue
            try:
                self.sending = asyncio.create_task(self.send(receiver, message))
                await self.sending
                if self.valid(observation):
                    self.operator_confirmed = True
            except ConnectionClosed:
                # Sender ingestion and inference continue; reconnect never replays old chunks.
                pass
            except ProtocolError as exc:
                await self.finish(exc)
            finally:
                self.sending = None


class AsyncPolicyInferenceServer:
    """Dedicated async endpoint with a process-wide, non-preemptible single model worker."""

    def __init__(
        self,
        policy: Any,
        *,
        max_horizon: int = 100,
        max_message_bytes: int = MAX_BYTES,
        max_requests: int = 10_000,
        session_lifetime: float = 3600,
        handshake_timeout: float = 10,
        reset_timeout: float = 60,
        send_timeout: float = 5,
        publication_guard: Any = None,
    ) -> None:
        mode, self.model_horizon = validate_contract(policy.contract)
        check(callable(getattr(policy, "reset", None)), "Async policy must implement reset()")
        self.max_horizon = min(integer(max_horizon, 1, 100), self.model_horizon)
        self.max_message_bytes = integer(max_message_bytes, 1024, MAX_BYTES)
        self.max_requests = integer(max_requests, 2)
        check(
            all(
                math.isfinite(value) and value > 0
                for value in (session_lifetime, handshake_timeout, reset_timeout, send_timeout)
            ),
            "Invalid timeout",
        )
        self.policy, self.publication_guard = policy, publication_guard
        self.session_lifetime, self.handshake_timeout = session_lifetime, handshake_timeout
        self.reset_timeout, self.send_timeout = reset_timeout, send_timeout
        arm = "arm" if mode == "joint" else "eef"
        self.action_order = (f"{arm}_left", f"{arm}_right", "hand_left", "hand_right")
        self.server_id = uuid.uuid4().hex
        self.session: AsyncSession | None = None
        self.tasks: set[asyncio.Task] = set()
        self.model_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r1pro-async")

    def track(self, coroutine: Any) -> None:
        """Retain background tasks until they drain, including disconnected sessions."""
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def handler(self, socket: Any) -> None:
        """Validate client-first hello, attach a vacant role, then receive independently."""
        session = None
        role = None
        try:
            hello = decode_message(await asyncio.wait_for(socket.recv(), self.handshake_timeout), self.max_message_bytes)
            exact_fields(hello, {"type", "protocol", "version", "role", "action_dim", "max_horizon"})
            check(hello["type"] == "hello" and hello["protocol"] == PROTOCOL, "Unsupported protocol")
            check(
                isinstance(hello["version"], list)
                and all(type(v) is int for v in hello["version"])
                and hello["version"] == VERSION,
                "Unsupported version",
            )
            role = hello["role"]
            check(role in ("sender", "receiver"), "Unsupported role")
            check(integer(hello["action_dim"]) == ACTION_DIM, "Unsupported action dimension")
            horizon = min(integer(hello["max_horizon"], 1, 100), self.max_horizon)
            current = self.session
            check(
                current is None or (current.open and role not in current.sockets),
                "Endpoint already has this role",
                "busy",
            )
            check(
                current is None or horizon >= current.horizon,
                "Peer max_horizon is below the paired connection's horizon",
            )
            if current is None:
                current = self.session = AsyncSession(self, horizon)
            session = current
            session.sockets[role] = socket
            if role == "receiver":
                session.receiver_handshake = asyncio.Event()
            if role == "sender":
                session.invalidate()
                session.generation = session.last_reset = None
                session.last_timestamp = -1
                await asyncio.wait_for(session.reset_model(), self.reset_timeout)
            await session.send(
                socket,
                {
                    "type": "welcome",
                    "protocol": PROTOCOL,
                    "version": VERSION,
                    "role": role,
                    "action_dim": ACTION_DIM,
                    "max_horizon": session.horizon,
                    "server_id": self.server_id,
                },
            )
            if role == "receiver":
                session.receiver_ready = True
                session.receiver_handshake.set()
            while session.open:
                raw = await asyncio.wait_for(socket.recv(), max(0, session.expires - time.monotonic()))
                check(role == "sender", "Receiver connection must not send messages after hello")
                await session.dispatch(socket, decode_message(raw, self.max_message_bytes))
        except ConnectionClosed as exc:
            codes = {frame.code for frame in (exc.rcvd, exc.sent) if frame is not None}
            if session is not None and codes & {1002, 1003, 1007, 1009}:
                await session.finish(ProtocolError("Invalid or oversized WebSocket frame"))
        except (ProtocolError, asyncio.TimeoutError) as exc:
            error = exc if isinstance(exc, ProtocolError) else ProtocolError("Async session timed out", "timeout")
            if session is not None:
                await session.finish(error)
            else:
                # A rejected third connection cannot tear down the active robot's pair.
                try:
                    await asyncio.wait_for(
                        socket.send(
                            msgpack_numpy.packb(
                                {"type": "error", "code": error.code, "message": str(error), "fatal": True},
                                use_bin_type=True,
                            )
                        ),
                        self.send_timeout,
                    )
                except (ConnectionClosed, asyncio.TimeoutError):
                    pass
                await socket.close(code=1008)
        except Exception:
            LOGGER.exception("Async connection/context failed")
            if session is not None:
                await session.finish(ProtocolError("Server context failure", "inference_failed"))
            else:
                await socket.close(code=1011)
        finally:
            if session is not None and session.sockets.get(role) is socket:
                del session.sockets[role]
                if role == "sender":
                    session.invalidate()
                    await session.revoke_send()
                else:
                    session.receiver_ready = False
                    session.receiver_handshake.set()
                    session.action = None
                if not session.sockets:
                    await session.finish()

    async def shutdown(self) -> None:
        """Close the pair and drain all retired workers before shutting down the executor."""
        if self.session is not None:
            await self.session.finish()
        if self.tasks:
            await asyncio.gather(*self.tasks)
        self.executor.shutdown(wait=True)

    async def run(self, host: str, port: int) -> None:
        """Listen on the dedicated async endpoint with bounded receive buffers."""
        try:
            async with serve(
                self.handler, host, port, compression=None, max_size=self.max_message_bytes, max_queue=1, close_timeout=5
            ) as server:
                LOGGER.info(
                    "%s 1.0 ready on ws://%s:%s (one robot pair, max_horizon=%s)", PROTOCOL, host, port, self.max_horizon
                )
                await server.serve_forever()
        finally:
            await self.shutdown()
