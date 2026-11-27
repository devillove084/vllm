# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind an OpenAI API process to its committed Plait group version."""

import asyncio
import json
import os
import re
from pathlib import Path

from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.logger import init_logger

VERSION_HEADER = "X-Plait-Group-Version"
ROLE_HEADER = "X-Plait-Group-Role"
DRAIN_PATH = "/_plait/drain"
logger = init_logger(__name__)
_VERSION = VERSION_HEADER.lower().encode()
_ROLE = ROLE_HEADER.lower().encode()
_ADMISSION = b"x-plait-admission"


class NativeGroupHeaders:
    def __init__(self, app: ASGIApp):
        self.app = app
        descriptor = os.environ.get("VLLM_PLAIT_TP_GROUP_CONFIG")
        if os.environ.get("VLLM_USE_PLAIT") != "1" or descriptor is None:
            raise ValueError("group middleware requires a committed Plait launch")
        deployment = json.loads(Path(descriptor).read_text(encoding="utf-8"))
        groups = deployment.get("tp_groups", []) + deployment.get("pp_groups", [])
        if len(groups) != 1 or not re.fullmatch(
            r"[a-f0-9]{32}@[1-9][0-9]*", groups[0]["version"]
        ):
            raise ValueError(
                "group middleware requires one valid execution group version"
            )
        self.version = groups[0]["version"]
        self.response_headers = [
            (_VERSION, self.version.encode("ascii")),
            (_ROLE, b"aggregated"),
        ]
        self.drain_marker = Path(descriptor).with_suffix(".drain")
        self.draining = False
        self.inflight = 0
        self.pause: asyncio.Task[None] | None = None

    async def drain_status(self, scope: Scope) -> int:
        if not self.draining or self.inflight:
            return 202
        if self.pause is None:
            engine = scope["app"].state.engine_client
            self.pause = asyncio.create_task(
                engine.pause_generation(mode="wait", clear_cache=False)
            )
        if not self.pause.done():
            return 202
        try:
            self.pause.result()
        except Exception:
            logger.exception("Plait group %s could not pause generation", self.version)
            self.pause = None
            return 202
        return 204

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        versions = [value for key, value in scope["headers"] if key.lower() == _VERSION]
        response_headers = {VERSION_HEADER: self.version, ROLE_HEADER: "aggregated"}
        if versions and versions != [self.response_headers[0][1]]:
            response = JSONResponse(
                {"error": {"message": "execution group version does not match"}},
                status_code=409,
                headers=response_headers | {"X-Plait-Admission": "rejected"},
            )
            await response(scope, receive, send)
            return
        # Checking the marker and increasing inflight on the API event loop
        # makes closing admission atomic with dispatch.
        self.draining = self.draining or self.drain_marker.exists()
        if scope["path"] == DRAIN_PATH:
            if scope["method"] != "GET" or not versions:
                status = 404
            else:
                status = await self.drain_status(scope)
            await Response(status_code=status, headers=response_headers)(
                scope, receive, send
            )
            return
        counted = scope["path"] != "/health"
        if self.draining and counted:
            response = JSONResponse(
                {"error": {"message": "execution group is draining"}},
                status_code=503,
                headers=response_headers | {"X-Plait-Admission": "rejected"},
            )
            await response(scope, receive, send)
            return

        async def send_versioned(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() not in (_VERSION, _ROLE, _ADMISSION)
                ]
                message = {**message, "headers": headers + self.response_headers}
            await send(message)

        self.inflight += counted
        try:
            await self.app(scope, receive, send_versioned)
        finally:
            self.inflight -= counted
