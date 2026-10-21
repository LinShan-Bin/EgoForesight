# EgoForesight modifications Copyright 2026 Chubin Zhang.
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modification: Add optional dynamic batching for concurrent inference requests.

import asyncio
import logging
import threading
import time
import traceback
from typing import Any, Dict, Optional

import websockets
import websockets.asyncio.server
import websockets.frames

from . import msgpack_numpy
from .dynamic_batcher import DynamicBatcher


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # seconds; -1 means never shutdown
        metadata: Optional[dict] = None,
        ping_interval: Optional[float] = 30.0,
        ping_timeout: Optional[float] = 300.0,
        max_batch_size: int = 1,
        batch_wait_ms: float = 0.0,
        batch_log_interval: int = 100,
        batch_trace_path: Optional[str] = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if batch_wait_ms < 0:
            raise ValueError(f"batch_wait_ms must be non-negative, got {batch_wait_ms}")
        if batch_log_interval <= 0:
            raise ValueError(f"batch_log_interval must be positive, got {batch_log_interval}")

        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()

        # WebSocket keepalive tuning (server side!)
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

        # Thread lock for concurrent inference protection
        self._inference_lock = threading.Lock()
        self._dynamic_batcher = (
            DynamicBatcher(
                policy=policy,
                max_batch_size=max_batch_size,
                batch_wait_ms=batch_wait_ms,
                log_interval=batch_log_interval,
                trace_path=batch_trace_path,
            )
            if max_batch_size > 1
            else None
        )

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        if self._dynamic_batcher is not None:
            await self._dynamic_batcher.start()
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
                close_timeout=10,
            ) as server:
                if self._idle_timeout > 0:
                    await self._idle_watchdog(server)
                else:
                    await server.serve_forever()
        finally:
            if self._dynamic_batcher is not None:
                await self._dynamic_batcher.close()

    async def _idle_watchdog(self, server: websockets.asyncio.server.Server) -> None:
        """Monitor server-level idle time; if no requests for idle_timeout seconds, shut down server."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send metadata as first binary frame
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                raw = await websocket.recv()
                msg = msgpack_numpy.unpackb(raw)
                self._last_active = time.time()

                # Do not block the event loop with inference.
                if self._dynamic_batcher is None:
                    ret = await asyncio.to_thread(self._route_message, msg)
                else:
                    ret = await self._route_message_async(msg)

                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # Send traceback as text, then close with 1011
                tb = traceback.format_exc()
                try:
                    await websocket.send(tb)
                except Exception:
                    pass
                try:
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except Exception:
                    pass
                raise

    @staticmethod
    def _normalize_message(msg: Any) -> tuple[Any, str, Any]:
        """Return request id, message type, and payload for either protocol style."""
        if isinstance(msg, dict):
            return msg.get("request_id", "default"), msg.get("type", "infer"), msg.get("payload", msg)
        return "default", "infer", msg

    @staticmethod
    def _error_response(request_id: Any, message: str, **details: Any) -> Dict[str, Any]:
        error = {"message": message}
        error.update(details)
        return {
            "status": "error",
            "ok": False,
            "type": "inference_result",
            "request_id": request_id,
            "error": error,
        }

    async def _route_message_async(self, msg: Any) -> Dict[str, Any]:
        """Submit inference to the batcher while routing control messages immediately."""
        request_id, message_type, payload = self._normalize_message(msg)
        if message_type not in ("infer", "predict_action"):
            return self._route_message(msg)
        if not isinstance(payload, dict):
            return self._error_response(request_id, "Payload must be a dict", payload_type=str(type(payload)))
        if self._dynamic_batcher is None:
            return await asyncio.to_thread(self._route_message, msg)

        try:
            output = await self._dynamic_batcher.infer(payload)
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": request_id,
                "data": output,
            }
        except Exception as error:
            logging.exception("Policy inference error (request_id=%s)", request_id)
            return self._error_response(request_id, str(error))

    def _route_message(self, msg: Any) -> Dict[str, Any]:
        """
        Fault-tolerant routing:
        Accepts either:
          1) protocol envelope:
             {"type": "ping|infer|reset|...", "request_id": "...", "payload": {...}}
          2) legacy style:
             { ...payload... }  (treated as infer)
        """
        req_id, mtype, payload = self._normalize_message(msg)

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer / predict_action
        if mtype in ("infer", "predict_action"):
            if not isinstance(payload, dict):
                return self._error_response(req_id, "Payload must be a dict", payload_type=str(type(payload)))

            try:
                # IMPORTANT: only pass payload into policy, not control fields
                # Use lock to ensure thread-safe concurrent inference
                with self._inference_lock:
                    output_dict = self._policy.predict_action(**payload)
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "inference_result",
                    "request_id": req_id,
                    "data": output_dict,
                }
            except Exception as error:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                return self._error_response(req_id, str(error))

        # unknown request type
        return {
            "status": "error",
            "ok": False,
            "type": "unknown",
            "request_id": req_id,
            "error": {"message": f"Unsupported message type '{mtype}'"},
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise NotImplementedError("This module is not intended to be run directly.")
