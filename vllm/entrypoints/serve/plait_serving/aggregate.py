#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Huanbing
"""Serve one OpenAI endpoint across versions of a committed Plait group."""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .group import (
    DIRECTORY_POLL_SECONDS,
    DIRECTORY_READ_TTL,
    HOP_HEADERS,
    HTTP_ERRORS,
    INFERENCE_PATHS,
    MAX_JSON_BYTES,
    PRIVATE_HEADERS,
    Control,
    ControlError,
    Endpoint,
    admission_rejected,
    endpoint_ready,
)
from .replay import (
    RECOVERY_HEADER,
    RECOVERY_MODE,
    RESUME_KV_MODE,
    CompletionReplay,
    ReplayMismatch,
    UpstreamInterrupted,
    sse,
)


class ActiveGroupDirectory:
    observation = "observe_group"
    recovery_modes = (RECOVERY_MODE, RESUME_KV_MODE)

    def __init__(self, model: str, group: bytes, control: Control) -> None:
        self.model = model
        self.group = list(group)
        self.control = control
        self.lock = threading.Lock()
        self.epoch = 0
        self.current: Endpoint | None = None
        self.closed = False
        self.until = 0.0

    def refresh(self) -> None:
        sent = time.monotonic()
        result = self.control.request(
            {
                "request": self.observation,
                "group": self.group,
                "id": self.control.next_operation_id(),
            }
        )["result"]
        if result.get("Err") == {"NoActiveGroup": self.group}:
            with self.lock:
                self.current = None
                self.closed = True
            return

        view = result["Ok"]
        version = view["version"]
        epoch = version["epoch"]
        if version["group"] != self.group or type(epoch) is not int or epoch < 1:
            raise ValueError("control returned a different group")
        endpoint, closed = self.resolve(view, epoch)

        with self.lock:
            if epoch < self.epoch or (epoch == self.epoch and self.closed):
                return
            self.epoch = epoch
            self.current = endpoint
            self.closed = closed
            self.until = sent + DIRECTORY_READ_TTL

    def endpoint_at(self, url: str, epoch: int) -> Endpoint:
        return Endpoint.parse(
            {
                "url": url,
                "version": f"{bytes(self.group).hex()}@{epoch}",
            },
            "aggregated",
        )

    def resolve(self, view: dict, epoch: int) -> tuple[Endpoint | None, bool]:
        advertised = view["endpoint"]
        if (
            view["role"] == "Aggregated"
            and view["phase"] == "Active"
            and advertised
            and advertised["api_model"] == self.model
        ):
            return self.endpoint_at("http://" + advertised["http"], epoch), False
        return None, True

    def endpoint(self) -> Endpoint | None:
        with self.lock:
            return self.current if time.monotonic() < self.until else None

    def watch(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                with contextlib.suppress(
                    ControlError, ValueError, KeyError, TypeError, AttributeError
                ):
                    self.refresh()
                stop.wait(DIRECTORY_POLL_SECONDS)
        finally:
            self.control.close()


class NativeGroupDirectory(ActiveGroupDirectory):
    observation = "observe"
    recovery_modes = (RECOVERY_MODE,)

    def __init__(
        self, model: str, group: bytes, control: Control, routes: Path
    ) -> None:
        super().__init__(model, group, control)
        self.routes = routes

    def resolve(self, view: dict, epoch: int) -> tuple[Endpoint | None, bool]:
        if view["phase"] != "Active":
            return None, True
        try:
            routes = json.loads(self.routes.read_text(encoding="utf-8"))
            url = routes[str(epoch)]
            if isinstance(url, str):
                return self.endpoint_at(url, epoch), False
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None, False


def server(
    bind: str,
    port: int,
    directory: ActiveGroupDirectory,
    timeout: float,
) -> ThreadingHTTPServer:
    class Entry(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self.write_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": directory.model, "object": "model"}],
                    },
                )
            elif self.path == "/health":
                endpoint = directory.endpoint()
                if endpoint is None or not endpoint_ready(endpoint, timeout):
                    self.fail(503, "active execution group is not ready")
                    return
                self.write_json(200, {}, endpoint.headers())
            else:
                self.fail(404, "unknown endpoint")

        def do_POST(self) -> None:
            if self.path not in INFERENCE_PATHS:
                self.fail(404, "unknown endpoint")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if (
                    self.headers.get("Transfer-Encoding")
                    or not 0 < size <= MAX_JSON_BYTES
                ):
                    raise ValueError("expected a bounded Content-Length")
                self.connection.settimeout(timeout)
                raw = self.rfile.read(size)
                body = json.loads(raw)
                if (
                    len(raw) != size
                    or not isinstance(body, dict)
                    or body.get("model") != directory.model
                ):
                    raise ValueError(
                        "expected a complete request for the configured model"
                    )
                recovery = self.headers.get(RECOVERY_HEADER)
                replay = None
                if recovery is not None:
                    if (
                        recovery not in directory.recovery_modes
                        or self.path != "/v1/completions"
                    ):
                        raise ValueError("unsupported stream recovery mode")
                    replay = CompletionReplay(body, recovery)
                    if replay.seed is not None:
                        raise ValueError(
                            "seeded KV resume requires the unified serving ingress"
                        )
            except (OSError, ValueError) as error:
                self.fail(400, str(error))
                return

            deadline = time.monotonic() + timeout
            request_id = uuid.uuid4().hex
            while (budget := deadline - time.monotonic()) > 0:
                endpoint = directory.endpoint()
                if endpoint is not None and (
                    replay is None or endpoint.epoch > replay.failed_epoch
                ):
                    payload = raw if replay is None else replay.attempt_payload()
                    if self.try_endpoint(endpoint, payload, request_id, budget, replay):
                        return
                time.sleep(
                    min(DIRECTORY_POLL_SECONDS, max(0.0, deadline - time.monotonic()))
                )
            self.fail_forward(
                503, "no active execution group accepted the request", replay
            )

        def try_endpoint(
            self,
            endpoint: Endpoint,
            raw: bytes,
            request_id: str,
            budget: float,
            replay: CompletionReplay | None,
        ) -> bool:
            version_headers = endpoint.headers()
            headers = {
                **version_headers,
                "Content-Type": "application/json",
                "X-Request-Id": request_id,
            }
            with contextlib.closing(endpoint.connect(min(2.0, budget))) as connection:
                try:
                    connection.connect()
                except HTTP_ERRORS:
                    # No request bytes have been sent.
                    return False

                try:
                    connection.sock.settimeout(budget)
                    connection.request("POST", self.path, body=raw, headers=headers)
                    response = connection.getresponse()
                except HTTP_ERRORS:
                    if replay is not None and replay.started:
                        replay.failed_epoch = endpoint.epoch
                        return False
                    self.fail(
                        502, "execution group disconnected after forwarding began"
                    )
                    return True

                with response:
                    if admission_rejected(response):
                        return False
                    if any(
                        response.getheader(key) != value
                        for key, value in version_headers.items()
                    ):
                        self.fail_forward(
                            502,
                            "upstream response does not match the admitted group",
                            replay,
                        )
                        return True
                    if replay is not None:
                        if response.status == 200 and response.getheader(
                            "Content-Type", ""
                        ).startswith("text/event-stream"):
                            return self.forward_replay(
                                response, version_headers, endpoint, replay
                            )
                        if replay.started:
                            self.fail_forward(
                                502,
                                "replacement did not return a completion stream",
                                replay,
                            )
                            return True
                    with contextlib.suppress(*HTTP_ERRORS):
                        self.send_response(response.status, response.reason)
                        for key, value in response.getheaders():
                            if (
                                key.lower() not in HOP_HEADERS
                                and key.lower() not in PRIVATE_HEADERS
                            ):
                                self.send_header(key, value)
                        for key, value in version_headers.items():
                            self.send_header(key, value)
                        self.end_headers()
                        while chunk := response.read1(65536):
                            self.wfile.write(chunk)
            return True

        def forward_replay(self, response, version_headers, endpoint, replay) -> bool:
            try:
                if not replay.started:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header(RECOVERY_HEADER, replay.mode)
                    for key, value in version_headers.items():
                        self.send_header(key, value)
                    self.end_headers()
                    replay.started = True
                replay.drain_attempt(response, self.wfile.write)
            except UpstreamInterrupted:
                replay.failed_epoch = endpoint.epoch
                return False
            except ReplayMismatch as error:
                self.fail_forward(502, str(error), replay)
            except HTTP_ERRORS:
                # The client's connection has closed.
                pass
            return True

        def write_json(
            self, status: int, body: dict, headers: dict | None = None
        ) -> None:
            data = json.dumps(body).encode()
            with contextlib.suppress(*HTTP_ERRORS):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

        def fail(self, status: int, message: str) -> None:
            self.write_json(
                status, {"error": {"message": message, "type": "plait_error"}}
            )

        def fail_forward(
            self, status: int, message: str, replay: CompletionReplay | None
        ) -> None:
            if replay is not None and replay.started:
                with contextlib.suppress(*HTTP_ERRORS):
                    self.wfile.write(
                        sse(
                            {
                                "error": {
                                    "message": message,
                                    "type": "plait_recovery_error",
                                }
                            }
                        )
                    )
            else:
                self.fail(status, message)

    return ThreadingHTTPServer((bind, port), Entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="stable API model alias")
    parser.add_argument(
        "--group-id", required=True, help="hex ID of the versioned aggregate group"
    )
    parser.add_argument("--control-bin", default="plait-node")
    parser.add_argument("--control-peer", action="append", default=[])
    parser.add_argument("--control-seed", action="append", default=[])
    parser.add_argument("--control-listen", default="127.0.0.1:0")
    parser.add_argument(
        "--native-routes",
        type=Path,
        help="JSON map from native group epoch to HTTP origin",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        group = bytes.fromhex(args.group_id)
        if len(group) != 16 or len(args.group_id) != 32:
            raise ValueError("expected 32 hexadecimal digits for --group-id")
    except ValueError as error:
        parser.error(str(error))
    if (
        not args.model
        or not 0 < args.port < 65536
        or not 0 < args.timeout < float("inf")
    ):
        parser.error("model, port and timeout must be valid")
    if bool(args.control_peer) == bool(args.control_seed):
        parser.error("provide either --control-peer or --control-seed")

    control = Control(
        args.control_bin,
        args.control_peer,
        args.control_listen,
        seeds=args.control_seed,
        command="group" if args.native_routes else "serving",
    )
    if args.native_routes:
        directory = NativeGroupDirectory(args.model, group, control, args.native_routes)
    else:
        directory = ActiveGroupDirectory(args.model, group, control)
    stop = threading.Event()
    watcher = threading.Thread(target=directory.watch, args=(stop,))

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, interrupt)
    watcher.start()
    try:
        with server(args.host, args.port, directory, args.timeout) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        watcher.join()
        signal.signal(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    main()
