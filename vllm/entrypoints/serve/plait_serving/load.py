# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Huanbing
"""Model-level demand counters for the P/D planner."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import socket
import socketserver
import struct
import threading
import time
import uuid

MAGIC = b"PLTPDL01"


class Load:
    def __init__(self, model: str) -> None:
        self.model = model
        self.session = uuid.uuid4().hex
        self.started = time.monotonic_ns()
        self.lock = threading.Lock()
        self.arrivals = 0
        self.completed = 0
        self.failed = 0
        self.in_flight = 0
        self.checkpoint_arrivals = 0
        self.checkpoint_in_flight = 0

    def begin(self, checkpoint: bool = False) -> None:
        with self.lock:
            self.arrivals += 1
            self.in_flight += 1
            if checkpoint:
                self.checkpoint_arrivals += 1
                self.checkpoint_in_flight += 1

    def finish(self, completed: bool, checkpoint: bool = False) -> None:
        with self.lock:
            self.in_flight -= 1
            if checkpoint:
                self.checkpoint_in_flight -= 1
            if completed:
                self.completed += 1
            else:
                self.failed += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "model": self.model,
                "session": self.session,
                "uptime_ms": (time.monotonic_ns() - self.started) // 1_000_000,
                "arrivals": self.arrivals,
                "completed": self.completed,
                "failed": self.failed,
                "in_flight": self.in_flight,
                "checkpoints": {
                    "arrivals": self.checkpoint_arrivals,
                    "in_flight": self.checkpoint_in_flight,
                },
            }


def parse_address(text: str) -> tuple[str, int]:
    host, separator, port = text.rpartition(":")
    if (
        not separator
        or not port.isascii()
        or not port.isdigit()
        or not 0 < int(port) < 65536
    ):
        raise ValueError("expected an IP address and TCP port")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    address = ipaddress.ip_address(host)
    if address.is_unspecified or address.is_multicast:
        raise ValueError("load listener requires a concrete unicast address")
    return str(address), int(port)


@contextlib.contextmanager
def listen(address: tuple[str, int], load: Load):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            self.connection.settimeout(1.0)
            with contextlib.suppress(OSError):
                if self.rfile.read(len(MAGIC)) == MAGIC:
                    body = json.dumps(load.snapshot(), separators=(",", ":")).encode()
                    self.wfile.write(struct.pack("!I", len(body)) + body)

    class Server(socketserver.TCPServer):
        allow_reuse_address = True
        address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET

    with Server(address, Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, name="plait-load-listener"
        )
        thread.start()
        try:
            yield
        finally:
            server.shutdown()
            thread.join()
