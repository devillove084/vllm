#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Huanbing
"""Route OpenAI completions through versioned vLLM groups."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import ipaddress
import itertools
import json
import re
import signal
import threading
import time
import uuid
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .bindings import Bindings
from .calibration import Calibrator, Target
from .group import (
    DIRECTORY_POLL_SECONDS,
    DIRECTORY_READ_TTL,
    HANDOFF_HEADER,
    HTTP_ERRORS,
    INFERENCE_PATHS,
    MAX_JSON_BYTES,
    RELEASE_KV_PATH,
    Control,
    ControlError,
    Endpoint,
    admission_rejected,
    endpoint_ready,
    probe_endpoint,
)
from .launch import SEEDED_KV_RESUME
from .load import Load
from .load import listen as load_listener
from .load import parse_address as parse_load_address
from .mode import ModeController, ModeView
from .replay import (
    RECOVERY_HEADER,
    RECOVERY_MODE,
    RESUME_KV_MODE,
    CompletionReplay,
    PlannedHandoff,
    ReplayMismatch,
    UpstreamInterrupted,
    sse,
)


class UpstreamError(RuntimeError):
    pass


class PrefillError(UpstreamError):
    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self.body = body


@dataclass(frozen=True)
class Transfer:
    request_id: str
    token: str
    params: dict


@dataclass(frozen=True)
class Route:
    prefill: Endpoint
    decode: Endpoint
    kv_host: str
    kv_port: int
    prefill_lease: bool = True

    @classmethod
    def parse(cls, config: dict) -> Route:
        prefill = Endpoint.parse(config["prefill"], "prefill")
        decode = Endpoint.parse(config["decode"], "decode")
        host = ipaddress.IPv4Address(config["prefill"]["kv_host"])
        port = config["prefill"]["kv_port"]
        if (
            type(port) is not int
            or not 0 < port < 65536
            or prefill.version == decode.version
        ):
            raise ValueError("invalid KV source or execution group pair")
        return cls(prefill, decode, str(host), port)


Destination = Endpoint | Route


def generator(route: Destination) -> Endpoint:
    return route.decode if isinstance(route, Route) else route


class Directory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def routes(self) -> tuple[str, list[Destination]]:
        return parse_directory(json.loads(self.path.read_text()))


def parse_directory(document: dict) -> tuple[str, list[Destination]]:
    model = document["model"]
    if not isinstance(model, str) or not model:
        raise ValueError("route directory requires a model")
    return model, [
        *(Route.parse(item) for item in document["paths"]),
        *(
            Endpoint.parse(item, "aggregated")
            for item in document.get("aggregated", [])
        ),
    ]


class ControlDirectory:
    def __init__(self, model: str, control: Control) -> None:
        self.model = model
        self.control = control
        self.lock = threading.Lock()
        self.revision = 0
        self.paths: list[Destination] = []
        self.until = 0.0

    def refresh(self) -> None:
        sent = time.monotonic()
        result = self.control.execute(
            {"ObserveRoutes": {"model": self.model}}, retry=False
        )
        if "Ok" not in result:
            return
        observed = result["Ok"]["RoutesObserved"]["revision"]
        minimum = max(self.revision, observed)
        document = self.control.request(
            {
                "request": "routes",
                "model": self.model,
                "minimum_revision": minimum,
            }
        )["directory"]
        if document is None:
            return
        model, routes = parse_directory(document)
        revision = document["revision"]
        if model != self.model or type(revision) is not int or revision < 1:
            raise ValueError("unexpected control directory")
        with self.lock:
            if revision < minimum:
                return
            if revision == self.revision:
                observed = set(routes)
                # A group version that has stopped accepting cannot be active again.
                routes = [route for route in self.paths if route in observed]
            self.revision = revision
            self.paths = routes
            self.until = sent + DIRECTORY_READ_TTL

    def routes(self) -> tuple[str, list[Destination]]:
        with self.lock:
            if time.monotonic() >= self.until:
                raise ValueError("routing directory is unavailable or expired")
            return self.model, self.paths

    def watch(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                with contextlib.suppress(
                    ControlError,
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError,
                ):
                    self.refresh()
                stop.wait(DIRECTORY_POLL_SECONDS)
        finally:
            self.control.close()


class NativeDirectory(ControlDirectory):
    """Route only to exact versions present in one ordered group observation."""

    def __init__(self, model: str, control: Control, path: Path) -> None:
        super().__init__(model, control)
        self.path = path
        self.mode_view: ModeView | None = None
        self.configured: list[Destination] = []
        self.discovery: dict = {}
        self.resolved: list[Destination] = []

    def refresh(self) -> None:
        document = json.loads(self.path.read_text(encoding="utf-8"))
        model, paths = parse_directory(document)
        revision = document["revision"]
        if model != self.model or type(revision) is not int or revision < 1:
            raise ValueError("unexpected native route directory")
        if revision < self.revision:
            return
        discovery = document.get("discover", {})
        if revision == self.revision and (
            paths != self.configured or discovery != self.discovery
        ):
            raise ValueError("native route changes require a new revision")
        dynamic = Bindings.read(self.model, self.path.parent, discovery)
        static_groups = {
            endpoint.version.split("@", 1)[0]
            for path in paths
            for endpoint in (
                (path.prefill, path.decode) if isinstance(path, Route) else (path,)
            )
        }
        if static_groups & dynamic.deployments.keys():
            raise ValueError("use one native binding source per group")
        groups = sorted(static_groups | dynamic.deployments.keys())
        sent = time.monotonic()
        active = frozenset(self.observe(groups)) if groups else frozenset()
        _, discovered = parse_directory(dynamic.resolve(self.control, active))
        configured = list(dict.fromkeys(paths + discovered))
        bindings = tuple(
            (path.prefill, path.decode) if isinstance(path, Route) else (path,)
            for path in configured
        )
        with self.lock:
            self.revision = revision
            self.configured = paths
            self.discovery = discovery
            self.resolved = [
                replace(path, prefill_lease=False) if isinstance(path, Route) else path
                for path in configured
            ]
            self.paths = [
                path
                for path, endpoints in zip(self.resolved, bindings, strict=True)
                if all(endpoint.version in active for endpoint in endpoints)
            ]
            self.mode_view = ModeView(revision, bindings, tuple(groups), active)
            self.until = sent + DIRECTORY_READ_TTL

    def mode_snapshot(self) -> ModeView | None:
        with self.lock:
            return self.mode_view if time.monotonic() < self.until else None

    def calibration_routes(self, target: Target) -> tuple[str, list[Destination]]:
        with self.lock:
            view = self.mode_view
            if (
                view is None
                or time.monotonic() >= self.until
                or not target.matches(view.revision, view.active, view.routes)
            ):
                raise ValueError("candidate calibration route expired")
            return self.model, [
                path
                for path, endpoints in zip(self.resolved, view.routes, strict=True)
                if endpoints in target.routes
            ]

    def observe(self, groups: list[str]) -> set[str]:
        result = self.control.request(
            {
                "request": "apply",
                "operation": {
                    "id": self.control.next_operation_id(),
                    "action": {
                        "ObserveActiveSet": {
                            "groups": [list(bytes.fromhex(group)) for group in groups],
                        }
                    },
                },
            }
        )["result"]
        if "Ok" not in result:
            raise ControlError("Plait rejected the active group observation")
        epochs = result["Ok"]["ActiveSet"]
        if [bytes(group).hex() for group, _ in epochs] != groups or any(
            epoch is not None and (type(epoch) is not int or epoch < 1)
            for _, epoch in epochs
        ):
            raise ControlError("Plait returned a different active group set")
        return {
            f"{group}@{epoch}"
            for group, (_, epoch) in zip(groups, epochs)
            if epoch is not None
        }


def json_object(raw: bytes) -> dict:
    item = json.loads(raw)
    if not isinstance(item, dict):
        raise ValueError("expected a JSON object")
    return item


def read_json(response: http.client.HTTPResponse) -> dict:
    body = response.read(MAX_JSON_BYTES + 1)
    if len(body) > MAX_JSON_BYTES:
        raise UpstreamError("engine response exceeds the JSON size limit")
    try:
        return json_object(body)
    except ValueError as error:
        raise UpstreamError("engine returned an invalid JSON response") from error


def prefill_payload(request: dict, plait_ticket: str | None = None) -> dict:
    payload = {**request, "stream": False}
    payload.pop("stream_options", None)
    payload["max_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    if "min_tokens" in payload:
        payload["min_tokens"] = 0
    payload["kv_transfer_params"] = (
        {"plait_handoff": {"ticket": plait_ticket}}
        if plait_ticket is not None
        else {"do_remote_decode": True, "do_remote_prefill": False}
    )
    return payload


def transfer_descriptor(
    response: dict, route: Route, plait_ticket: str | None = None
) -> dict:
    params = response.get("kv_transfer_params")
    if plait_ticket is not None:
        receipt = params.get("plait_handoff") if isinstance(params, dict) else None
        if (
            not isinstance(receipt, dict)
            or receipt.get("ticket") != plait_ticket
            or receipt.get("store") != f"{route.kv_host}:{route.kv_port}"
            or not isinstance(receipt.get("prompt_sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", receipt["prompt_sha256"]) is None
            or type(receipt.get("cached_tokens")) is not int
            or receipt["cached_tokens"] < 0
        ):
            raise UpstreamError(
                "prefill did not publish the requested Plait KV handoff"
            )
        return {"plait_handoff": receipt}
    if (
        not isinstance(params, dict)
        or params.get("do_remote_prefill") is not True
        or not params.get("remote_request_id")
        or not params.get("remote_engine_id")
        or not params.get("remote_block_ids")
        or params.get("remote_host") != route.kv_host
        or params.get("remote_port") != route.kv_port
    ):
        raise UpstreamError("prefill did not export KV for the configured source")
    return params


def release_transfer(route: Route, token: str) -> None:
    if not route.prefill_lease:
        return
    endpoint = route.prefill
    headers = {**endpoint.headers(), HANDOFF_HEADER: token}
    with (
        contextlib.suppress(*HTTP_ERRORS),
        contextlib.closing(endpoint.connect(1.0)) as connection,
    ):
        connection.request("POST", RELEASE_KV_PATH, body=b"", headers=headers)
        with connection.getresponse() as response:
            response.read()


def request_engine(
    connection: http.client.HTTPConnection,
    endpoint: Endpoint,
    path: str,
    body: dict,
    request_id: str,
    handoff: str | None = None,
) -> http.client.HTTPResponse:
    headers = {
        **endpoint.headers(),
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
    }
    if handoff is not None:
        headers[HANDOFF_HEADER] = handoff
    connection.request("POST", path, body=json.dumps(body).encode(), headers=headers)
    return connection.getresponse()


def request_prefill(
    route: Route, path: str, body: dict, deadline: float, use_plait: bool
) -> Transfer | None:
    request_id, token = uuid.uuid4().hex, uuid.uuid4().hex
    ticket = token if use_plait else None
    budget = max(0.001, deadline - time.monotonic())
    with (
        contextlib.closing(route.prefill.connect(budget)) as connection,
        request_engine(
            connection,
            route.prefill,
            path,
            prefill_payload(body, ticket),
            request_id,
            token,
        ) as response,
    ):
        if admission_rejected(response):
            return None
        if not route.prefill.matches(response):
            raise UpstreamError("response does not match the admitted prefill group")
        if response.status != 200:
            raise PrefillError(response.status, read_json(response))
        try:
            params = transfer_descriptor(read_json(response), route, ticket)
        except UpstreamError:
            release_transfer(route, token)
            raise
    return Transfer(request_id, token, params)


@dataclass(frozen=True)
class CalibrationDirectory:
    source: NativeDirectory
    target: Target

    def routes(self) -> tuple[str, list[Destination]]:
        return self.source.calibration_routes(self.target)


@contextlib.contextmanager
def calibration_ingress(directory: NativeDirectory, target: Target, timeout: float):
    candidate = CalibrationDirectory(directory, target)
    candidate.routes()
    with server("127.0.0.1", 0, candidate, timeout, "plait") as httpd:
        thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="plait-calibration-ingress",
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_port}"
        finally:
            httpd.shutdown()
            thread.join()


def server(
    bind: str,
    port: int,
    directory: Directory | ControlDirectory | CalibrationDirectory,
    timeout: float,
    kv_transport: str = "nixl",
    load: Load | None = None,
) -> ThreadingHTTPServer:
    next_path = itertools.count()

    class Entry(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            config = self.read_directory()
            if config is None:
                return
            model, routes = config
            if self.path == "/v1/models":
                self.write_json(
                    200, {"object": "list", "data": [{"id": model, "object": "model"}]}
                )
            elif self.path == "/health" and routes:
                self.write_json(200, {})
            elif self.path == "/health":
                self.fail(503, "no serving routes are configured")
            else:
                self.fail(404, "unknown endpoint")

        def do_POST(self) -> None:
            if self.path not in INFERENCE_PATHS:
                self.fail(404, "unknown endpoint")
                return
            if load is None:
                config = self.read_directory()
                if config is None:
                    return
                model, routes = config
            else:
                model = load.model
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if (
                    self.headers.get("Transfer-Encoding")
                    or not 0 < size <= MAX_JSON_BYTES
                ):
                    raise ValueError("expected a bounded Content-Length")
                self.connection.settimeout(timeout)
                body = json_object(self.rfile.read(size))
                prompt = body.get("prompt")
                if body.get("model") != model or body.get("n", 1) != 1:
                    raise ValueError("expected the configured model and one completion")
                if isinstance(prompt, list) and any(
                    type(token) is not int for token in prompt
                ):
                    raise ValueError("batched prompts need separate P/D requests")
                recovery = self.headers.get(RECOVERY_HEADER)
                replay = None
                if recovery is not None:
                    if (
                        recovery not in (RECOVERY_MODE, RESUME_KV_MODE)
                        or kv_transport != "plait"
                        or self.path != "/v1/completions"
                    ):
                        raise ValueError(
                            "stream recovery requires Plait KV and a text completion"
                        )
                    replay = CompletionReplay(body, recovery)
                body.pop("kv_transfer_params", None)
            except (OSError, ValueError) as error:
                self.fail(400, str(error))
                return

            if load is None:
                self.route_request(routes, body, replay)
                return
            checkpoint = replay is not None and replay.mode == RESUME_KV_MODE
            load.begin(checkpoint)
            completed = False
            try:
                config = self.read_directory()
                if config is not None:
                    _, routes = config
                    completed = self.route_request(routes, body, replay)
            finally:
                load.finish(completed, checkpoint)

        def route_request(
            self,
            routes: list[Destination],
            body: dict,
            replay: CompletionReplay | None = None,
        ) -> bool:
            deadline = time.monotonic() + timeout
            if routes:
                start = next(next_path) % len(routes)
                routes = routes[start:] + routes[:start]
            if replay is not None:
                return self.route_replay(routes, replay, deadline)
            for route in routes:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    break
                endpoint = generator(route)
                if not endpoint_ready(endpoint, budget):
                    continue
                try:
                    if isinstance(route, Route):
                        completed = self.try_route(route, body, deadline)
                    else:
                        completed = self.try_aggregated(endpoint, body, deadline)
                    if completed is not None:
                        return completed
                except (OSError, http.client.HTTPException, UpstreamError) as error:
                    self.fail(502, str(error))
                    return False
            self.fail(503, "no serving route accepted the request")
            return False

        def try_aggregated(
            self, endpoint: Endpoint, body: dict, deadline: float
        ) -> bool | None:
            budget = deadline - time.monotonic()
            if budget <= 0:
                return None
            with (
                contextlib.closing(endpoint.connect(budget)) as connection,
                request_engine(
                    connection, endpoint, self.path, body, uuid.uuid4().hex
                ) as response,
            ):
                if admission_rejected(response):
                    return None
                if not endpoint.matches(response):
                    raise UpstreamError(
                        "response does not match the admitted aggregate group"
                    )
                return self.write_decode(response)

        def try_route(self, route: Route, body: dict, deadline: float) -> bool | None:
            try:
                transfer = request_prefill(
                    route, self.path, body, deadline, kv_transport == "plait"
                )
            except PrefillError as error:
                self.write_json(error.status, error.body)
                return False
            if transfer is None:
                return None

            budget = deadline - time.monotonic()
            if budget <= 0:
                release_transfer(route, transfer.token)
                raise UpstreamError("request deadline elapsed before decode")
            with (
                contextlib.closing(route.decode.connect(budget)) as connection,
                request_engine(
                    connection,
                    route.decode,
                    self.path,
                    {**body, "kv_transfer_params": transfer.params},
                    transfer.request_id,
                ) as response,
            ):
                if admission_rejected(response):
                    release_transfer(route, transfer.token)
                    return None
                if not route.decode.matches(response):
                    raise UpstreamError(
                        "response does not match the admitted decode group"
                    )
                completed = self.write_decode(response)
                if completed:
                    release_transfer(route, transfer.token)
            return completed

        def route_replay(
            self, routes: list[Destination], replay: CompletionReplay, deadline: float
        ) -> bool:
            failed_generators = set()
            while time.monotonic() < deadline:
                tried_prefills = set()
                handed_off = False
                for route in routes:
                    budget = deadline - time.monotonic()
                    if budget <= 0:
                        break
                    endpoint = generator(route)
                    if endpoint.version in failed_generators:
                        continue
                    capability = self.probe_generation(
                        endpoint, min(1.0, budget), replay
                    )
                    if capability is None:
                        continue
                    transfer = None
                    if isinstance(route, Route):
                        if route.prefill.version in tried_prefills:
                            continue
                        try:
                            transfer = request_prefill(
                                route, self.path, replay.request, deadline, True
                            )
                        except PrefillError as error:
                            if not replay.started:
                                self.write_json(error.status, error.body)
                                return False
                        except (*HTTP_ERRORS, UpstreamError):
                            pass
                        if transfer is None:
                            tried_prefills.add(route.prefill.version)
                            continue
                    try:
                        completed = self.replay_generation(
                            endpoint,
                            transfer,
                            deadline,
                            replay,
                            failed_generators,
                            bool(capability),
                        )
                        if completed is not None:
                            return completed
                    except PlannedHandoff:
                        failed_generators.add(endpoint.version)
                        handed_off = True
                        break
                    except (*HTTP_ERRORS, UpstreamError, UpstreamInterrupted):
                        failed_generators.add(endpoint.version)
                    except ReplayMismatch as error:
                        self.fail_replay(502, str(error), replay)
                        return False
                    finally:
                        # The store keeps prompt pages past this handoff lease.
                        if isinstance(route, Route) and transfer is not None:
                            release_transfer(route, transfer.token)
                if not handed_off:
                    time.sleep(
                        min(
                            DIRECTORY_POLL_SECONDS,
                            max(0.0, deadline - time.monotonic()),
                        )
                    )
                routes = self.current_routes(replay.model)
            self.fail_replay(
                503,
                "no serving replacement accepted the completion before its deadline",
                replay,
            )
            return False

        def current_routes(self, model: str) -> list[Destination]:
            with contextlib.suppress(
                OSError, ValueError, KeyError, TypeError, AttributeError
            ):
                current_model, routes = directory.routes()
                if current_model == model:
                    return routes
            return []

        def probe_generation(
            self, endpoint: Endpoint, budget: float, replay: CompletionReplay
        ) -> str | None:
            capability = probe_endpoint(endpoint, budget)
            if replay.seed is not None and capability != SEEDED_KV_RESUME:
                return None
            return capability

        def replay_generation(
            self,
            endpoint: Endpoint,
            transfer: Transfer | None,
            deadline: float,
            replay: CompletionReplay,
            failed_generators: set[str],
            can_resume: bool,
        ) -> bool | None:
            budget = deadline - time.monotonic()
            if budget <= 0:
                return None
            next_check = 0.0

            def should_handoff() -> bool:
                nonlocal next_check
                now = time.monotonic()
                if now < next_check or now >= deadline:
                    return False
                next_check = now + DIRECTORY_POLL_SECONDS
                routes = self.current_routes(replay.model)
                if any(
                    generator(route).version == endpoint.version for route in routes
                ):
                    return False
                for route in routes:
                    budget = min(0.2, deadline - time.monotonic())
                    if budget <= 0:
                        break
                    candidate = generator(route)
                    if candidate.version in failed_generators:
                        continue
                    # A replacement must be accepting before the source is closed.
                    if self.probe_generation(
                        candidate, budget, replay
                    ) is not None and (
                        not isinstance(route, Route)
                        or endpoint_ready(route.prefill, budget)
                    ):
                        return True
                return False

            with contextlib.closing(endpoint.connect(budget)) as connection:
                payload = replay.generation_request(can_resume)
                kv = payload.get("kv_transfer_params", {})
                if replay.mode == RESUME_KV_MODE and (
                    can_resume or transfer is not None
                ):
                    kv["plait_checkpoint"] = True
                if transfer is not None:
                    kv.update(transfer.params)
                if kv:
                    payload["kv_transfer_params"] = kv
                request_id = (
                    uuid.uuid4().hex if transfer is None else transfer.request_id
                )
                with request_engine(
                    connection, endpoint, self.path, payload, request_id
                ) as response:
                    if admission_rejected(response):
                        return None
                    headers = endpoint.headers()
                    if not endpoint.matches(response):
                        raise ReplayMismatch(
                            "response does not match the admitted serving group"
                        )
                    if response.status != 200 or not response.getheader(
                        "Content-Type", ""
                    ).startswith("text/event-stream"):
                        if replay.started:
                            raise UpstreamInterrupted
                        if response.status != 200:
                            return self.write_decode(response)
                        raise ReplayMismatch(
                            "engine did not return a completion stream"
                        )
                    try:
                        if not replay.started:
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.send_header(RECOVERY_HEADER, replay.mode)
                            for key, value in headers.items():
                                self.send_header(key, value)
                            self.end_headers()
                            replay.started = True
                        replay.drain_attempt(
                            response,
                            self.wfile.write,
                            should_handoff if replay.mode == RESUME_KV_MODE else None,
                            allow_resume=can_resume,
                        )
                    except HTTP_ERRORS:
                        return False
                    return True

        def fail_replay(
            self, status: int, message: str, replay: CompletionReplay
        ) -> None:
            if replay.started:
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

        def write_decode(self, response: http.client.HTTPResponse) -> bool:
            content_type = response.getheader("Content-Type", "")
            if not content_type.startswith("text/event-stream"):
                content = read_json(response)
                content.pop("kv_transfer_params", None)
                self.write_json(response.status, content)
                return response.status == 200 and "error" not in content

            done = False
            failed = False
            try:
                self.send_response(response.status)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                while line := response.readline(MAX_JSON_BYTES):
                    self.wfile.write(line)
                    if line.startswith(b"data:"):
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            done = True
                        else:
                            try:
                                failed = "error" in json_object(payload) or failed
                            except ValueError:
                                failed = True
            except HTTP_ERRORS:
                return False
            return response.status == 200 and done and not failed

        def read_directory(self) -> tuple[str, list[Destination]] | None:
            try:
                return directory.routes()
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                self.fail(503, "serving route configuration is unavailable")
                return None

        def write_json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            with contextlib.suppress(*HTTP_ERRORS):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def fail(self, status: int, message: str) -> None:
            kind = "invalid_request_error" if status < 500 else "server_error"
            self.write_json(status, {"error": {"message": message, "type": kind}})

    return ThreadingHTTPServer((bind, port), Entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--routes", type=Path, help="JSON file with versioned P/D paths"
    )
    source.add_argument(
        "--native-routes",
        type=Path,
        help="engine route directory using native group epochs",
    )
    source.add_argument(
        "--model", help="model alias published in the replicated directory"
    )
    parser.add_argument("--control-bin", default="plait-node")
    parser.add_argument("--control-peer", action="append", default=[])
    parser.add_argument("--control-seed", action="append", default=[])
    parser.add_argument("--control-listen", default="127.0.0.1:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--kv-transport", choices=("nixl", "plait"))
    parser.add_argument("--load-listen", help="private IP:port for P/D demand counters")
    parser.add_argument(
        "--mode-policy",
        type=Path,
        help="prepare and switch native layouts using serving capacity profiles",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not 0 < args.port < 65536 or not 0 < args.timeout < float("inf"):
        parser.error("port and timeout must be valid")
    if args.native_routes is not None:
        if bool(args.control_peer) == bool(args.control_seed):
            parser.error(
                "--native-routes requires either --control-peer or --control-seed"
            )
        if args.kv_transport not in (None, "plait"):
            parser.error("--native-routes uses the shared PlaitKvConnector page store")
        args.kv_transport = "plait"
        try:
            args.model, _ = parse_directory(
                json.loads(args.native_routes.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    elif args.control_seed:
        parser.error("--control-seed requires --native-routes")
    elif args.routes is None and not args.control_peer:
        parser.error("--model requires at least one --control-peer")
    if args.mode_policy is not None and args.native_routes is None:
        parser.error("--mode-policy requires --native-routes")
    if args.kv_transport is None:
        args.kv_transport = "nixl"
    load = (
        Load(args.model)
        if args.model is not None
        and (args.load_listen is not None or args.mode_policy is not None)
        else None
    )
    load_context = contextlib.nullcontext()
    if args.load_listen is not None:
        if load is None:
            parser.error("--load-listen requires a managed model")
        try:
            address = parse_load_address(args.load_listen)
        except ValueError as error:
            parser.error(str(error))
        load_context = load_listener(address, load)

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        with (
            load_context,
            route_source(args, load) as directory,
            server(
                args.host,
                args.port,
                directory,
                args.timeout,
                args.kv_transport,
                load,
            ) as httpd,
        ):
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, old_handler)


@contextlib.contextmanager
def route_source(args: argparse.Namespace, load: Load | None = None):
    if args.routes is not None:
        yield Directory(args.routes)
        return
    native = args.native_routes is not None

    def control(listen: str = args.control_listen) -> Control:
        return Control(
            args.control_bin,
            args.control_peer,
            listen,
            seeds=args.control_seed,
            command="group" if native else "serving",
        )

    directory = (
        NativeDirectory(args.model, control(), args.native_routes)
        if native
        else ControlDirectory(args.model, control())
    )
    stop = threading.Event()
    watcher = threading.Thread(
        target=directory.watch, args=(stop,), name="plait-route-directory"
    )
    threads = [watcher]
    if (
        isinstance(directory, NativeDirectory)
        and args.mode_policy is not None
        and load is not None
    ):
        host, _, _port = args.control_listen.rpartition(":")
        controller = ModeController(
            args.mode_policy,
            load,
            control(f"{host}:0"),
            args.model,
            Calibrator(
                lambda target: calibration_ingress(directory, target, args.timeout)
            ),
        )
        threads.append(
            threading.Thread(
                target=controller.watch,
                args=(stop, directory.mode_snapshot),
                name="plait-serving-mode",
            )
        )
    for thread in threads:
        thread.start()
    try:
        yield directory
    finally:
        stop.set()
        for thread in threads:
            thread.join()


if __name__ == "__main__":
    main()
