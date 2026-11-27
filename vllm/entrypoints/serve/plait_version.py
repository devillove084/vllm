# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind an OpenAI API process to its committed Plait group version."""

import json
import os
import re
from pathlib import Path

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

VERSION_HEADER = "X-Plait-Group-Version"
ROLE_HEADER = "X-Plait-Group-Role"
_VERSION = VERSION_HEADER.lower().encode()
_ROLE = ROLE_HEADER.lower().encode()
_ADMISSION = b"x-plait-admission"


class NativeGroupHeaders:
    def __init__(self, app: ASGIApp):
        self.app = app
        descriptor = os.environ.get("VLLM_PLAIT_TP_GROUP_CONFIG")
        if os.environ.get("VLLM_USE_PLAIT") != "1" or descriptor is None:
            raise ValueError("group middleware requires a committed Plait launch")
        groups = json.loads(Path(descriptor).read_text(encoding="utf-8"))["tp_groups"]
        if len(groups) != 1 or not re.fullmatch(
            r"[a-f0-9]{32}@[1-9][0-9]*", groups[0]["version"]
        ):
            raise ValueError("group middleware requires one valid TP group version")
        self.version = groups[0]["version"]
        self.response_headers = [
            (_VERSION, self.version.encode("ascii")),
            (_ROLE, b"aggregated"),
        ]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        versions = [value for key, value in scope["headers"] if key.lower() == _VERSION]
        if versions and versions != [self.response_headers[0][1]]:
            response = JSONResponse(
                {"error": {"message": "execution group version does not match"}},
                status_code=409,
                headers={
                    VERSION_HEADER: self.version,
                    ROLE_HEADER: "aggregated",
                    "X-Plait-Admission": "rejected",
                },
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

        await self.app(scope, receive, send_versioned)
