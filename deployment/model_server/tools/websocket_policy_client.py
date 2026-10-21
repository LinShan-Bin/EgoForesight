# EgoForesight modifications Copyright 2026 Chubin Zhang.
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import websockets
import websockets.sync.client
from typing_extensions import override

from . import msgpack_numpy
from .prompt_log import PromptLog


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = 10093,
        api_key: Optional[str] = None,
        connect_timeout: float = 300.0,
        open_timeout: float = 150.0,
        ping_interval: Optional[float] = 30.0,
        ping_timeout: Optional[float] = 300.0,
        reconnect_max_attempts: int = 5,
    ) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        self._packer = msgpack_numpy.Packer()
        self.prompt_log = PromptLog()
        self._api_key = api_key

        self._connect_timeout = connect_timeout
        self._open_timeout = open_timeout
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._reconnect_max_attempts = reconnect_max_attempts

        self._ws, self._server_metadata = self._wait_for_server(timeout=self._connect_timeout)

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def close(self) -> None:
        try:
            if getattr(self, "_ws", None) is not None:
                self._ws.close()
        except Exception:
            pass

    def _drop_proxy_env(self) -> None:
        # Some environments set proxies that break ws://127.0.0.1
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

    def _connect_once(self) -> Tuple[websockets.sync.client.ClientConnection, Dict[str, Any]]:
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        conn = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=self._open_timeout,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            close_timeout=10,
        )
        # First frame: metadata
        metadata = msgpack_numpy.unpackb(conn.recv())
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Server metadata must be dict, got: {type(metadata)}")
        return conn, metadata

    def _wait_for_server(self, timeout: float = 300.0) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()

        self._drop_proxy_env()

        backoff = 0.5  # seconds
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")

            try:
                conn, metadata = self._connect_once()
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for server {self._uri} ...")
            except Exception as e:
                logging.warning(f"Connect failed: {repr(e)}")

            time.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)

    def _reconnect(self) -> None:
        """Reconnect with limited attempts. Keeps it synchronous and simple."""
        self.close()
        last_err = None
        for attempt in range(1, self._reconnect_max_attempts + 1):
            try:
                self._ws, self._server_metadata = self._wait_for_server(timeout=60.0)
                return
            except Exception as e:
                last_err = e
                logging.warning(f"Reconnect attempt {attempt}/{self._reconnect_max_attempts} failed: {repr(e)}")
        raise RuntimeError(f"Failed to reconnect after {self._reconnect_max_attempts} attempts") from last_err

    @override
    def predict_action(self, query_info: Dict) -> Dict:
        """
        Sends a request to server. Supports both:
        - legacy payload dict (query_info)
        - or query_info already containing {type, request_id, payload}
        """
        self.prompt_log.record(query_info)
        data = self._packer.pack(query_info)

        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosedError as e:
            logging.warning(f"WebSocket closed, reconnecting once... ({e})")
            self._reconnect()
            self._ws.send(data)
            response = self._ws.recv()

        if isinstance(response, str):
            # server sent traceback text frame
            raise RuntimeError(f"Error in inference server:\n{response}")

        ret = msgpack_numpy.unpackb(response)
        if not isinstance(ret, dict):
            raise RuntimeError(f"Invalid response type: {type(ret)}")

        # Optional: normalize error replies
        if ret.get("ok") is False and ret.get("status") == "error":
            err = ret.get("error", {})
            raise RuntimeError(f"Server error: {err}")

        return ret
