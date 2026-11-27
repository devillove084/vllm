# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Huanbing
"""Run a committed native vLLM group behind an admission proxy."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import re
import select
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import launch as serving_launch

ADMISSION_SECONDS = 4.0
POLL_SECONDS = 0.5
VERSION_HEADER = "X-Plait-Group-Version"
ROLE_HEADER = "X-Plait-Group-Role"
HANDOFF_HEADER = "X-Plait-KV-Handoff"
ADMISSION_HEADER = "X-Plait-Admission"
KV_RESUME_HEADER = "X-Plait-KV-Resume"
CHECKPOINT_ADMISSION_HEADER = "X-Plait-KV-Checkpoint-Admission"
CHECKPOINT_PEERS_HEADER = "X-Plait-KV-Checkpoint-Peers"
RELEASE_KV_PATH = "/plait/release-kv"
INFERENCE_PATHS = frozenset({"/v1/completions", "/v1/chat/completions"})
MAX_JSON_BYTES = 16 * 1024 * 1024
HTTP_ERRORS = (OSError, http.client.HTTPException)
DIRECTORY_POLL_SECONDS = 0.5
DIRECTORY_READ_TTL = 8.0


class ControlError(RuntimeError):
    pass


class Rejected(ControlError):
    pass


class Control:
    def __init__(
        self,
        binary: str,
        peers: list[str],
        listen: str,
        *,
        seeds: tuple[str, ...] | list[str] = (),
        command: str = "serving",
    ) -> None:
        self.argv = [binary, command, "--listen", listen]
        for peer in peers:
            self.argv.extend(("--peer", peer))
        for seed in seeds:
            self.argv.extend(("--seed", seed))
        self.timeout = 12.0 if seeds else 3.0 + 2.0 * len(peers)
        self.process: subprocess.Popen | None = None
        self.holder = list(uuid.uuid4().bytes)
        self.counter = 0

    def execute(self, action: dict, *, retry: bool = True) -> dict:
        operation = {"id": self.next_operation_id(), "action": action}
        request = {"request": "execute", "operation": operation}
        try:
            return self.request(request)["result"]
        except ControlError:
            if not retry:
                raise
            return self.request(request)["result"]

    def next_operation_id(self) -> dict:
        self.counter += 1
        return {"origin": self.holder, "counter": self.counter}

    def request(self, request: dict) -> dict:
        try:
            if self.process is None:
                self.process = subprocess.Popen(
                    self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
                )
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            readable, _, _ = select.select([self.process.stdout], [], [], self.timeout)
            if not readable:
                raise ControlError("serving control timed out")
            line = self.process.stdout.readline()
            if not line:
                raise ControlError("serving control exited")
            reply = json.loads(line)
        except (OSError, ValueError, ControlError) as error:
            self.close()
            raise ControlError(str(error)) from error
        if "error" in reply:
            raise ControlError(reply["error"])
        return reply

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()


class Gate:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.until = 0.0
        self.running: set[socket.socket] = set()
        self.revoked = False
        self.transfers: dict[str, float] = {}

    def allow_until(self, deadline: float) -> None:
        with self.lock:
            if not self.revoked:
                self.until = deadline

    def close(self, *, revoke: bool = False) -> None:
        with self.lock:
            self.until = 0.0
            self.revoked = self.revoked or revoke
            connections = tuple(self.running) if revoke else ()
            if revoke:
                self.transfers.clear()
        for connection in connections:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)

    def admit(self, connection: socket.socket) -> bool:
        with self.lock:
            if self.revoked or time.monotonic() >= self.until:
                return False
            self.running.add(connection)
            return True

    def release(self, connection: socket.socket) -> None:
        with self.lock:
            self.running.remove(connection)

    def drained(self) -> bool:
        with self.lock:
            return not self.running and not self.transfers

    def hold_transfer(self, token: str, lease: float) -> None:
        with self.lock:
            if not self.revoked:
                self.transfers[token] = time.monotonic() + lease

    def release_transfer(self, token: str) -> None:
        with self.lock:
            self.transfers.pop(token, None)

    def expire_transfers(self) -> None:
        with self.lock:
            now = time.monotonic()
            self.transfers = {
                token: end for token, end in self.transfers.items() if end > now
            }

    def may_stream(self) -> bool:
        with self.lock:
            return not self.revoked


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    version: str
    role: str

    @classmethod
    def parse(cls, config: dict, role: str) -> Endpoint:
        url = urlsplit(config["url"])
        if (
            url.scheme != "http"
            or not url.hostname
            or not url.port
            or url.path not in ("", "/")
            or url.query
            or url.fragment
            or url.username is not None
        ):
            raise ValueError("group URL must be an HTTP origin with an explicit port")
        version = config["version"]
        if not re.fullmatch(r"[a-f0-9]{32}@[1-9][0-9]*", version):
            raise ValueError("invalid group version")
        return cls(url.hostname, url.port, version, role)

    def connect(self, timeout: float) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def headers(self) -> dict[str, str]:
        return {VERSION_HEADER: self.version, ROLE_HEADER: self.role}

    @property
    def epoch(self) -> int:
        return int(self.version.rsplit("@", 1)[1])


def admission_rejected(response: http.client.HTTPResponse) -> bool:
    return (
        response.status in (409, 503)
        and response.getheader(ADMISSION_HEADER) == "rejected"
    )


def probe_endpoint(endpoint: Endpoint, timeout: float) -> str | None:
    """Return the ready engine's KV resume capability, or None while unavailable."""
    try:
        with contextlib.closing(endpoint.connect(min(2.0, timeout))) as connection:
            connection.request("GET", "/health", headers=endpoint.headers())
            with connection.getresponse() as response:
                if (
                    response.status == 200
                    and response.getheader(VERSION_HEADER) == endpoint.version
                    and response.getheader(ROLE_HEADER) == endpoint.role
                ):
                    capability = response.getheader(KV_RESUME_HEADER)
                    if capability in (
                        serving_launch.GREEDY_KV_RESUME,
                        serving_launch.SEEDED_KV_RESUME,
                    ):
                        return capability
                    return ""
    except HTTP_ERRORS:
        pass
    return None


def endpoint_ready(endpoint: Endpoint, timeout: float) -> bool:
    return probe_endpoint(endpoint, timeout) is not None


def verify_backend_capabilities(port: int, expected: dict[str, str]) -> None:
    try:
        with contextlib.closing(
            http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        ) as connection:
            connection.request("GET", "/health")
            with connection.getresponse() as response:
                if response.status == 200 and all(
                    response.getheader(name) == value
                    for name, value in expected.items()
                ):
                    return
    except HTTP_ERRORS as error:
        raise ControlError(
            "cannot verify the local engine's KV capabilities"
        ) from error
    raise ControlError(f"the local engine did not advertise {', '.join(expected)}")


HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
PRIVATE_HEADERS = frozenset(
    header.lower()
    for header in (
        VERSION_HEADER,
        ROLE_HEADER,
        HANDOFF_HEADER,
        ADMISSION_HEADER,
        KV_RESUME_HEADER,
        CHECKPOINT_ADMISSION_HEADER,
        CHECKPOINT_PEERS_HEADER,
    )
)


def handoff_token(value: str | None) -> str | None:
    if value is not None and re.fullmatch(r"[a-f0-9]{32}", value):
        return value
    return None


def group_proxy(
    port: int,
    backend_port: int,
    gate: Gate,
    version: str,
    *,
    bind: str = "127.0.0.1",
    role: str = "aggregated",
    kv_lease: float = 0,
    kv_resume: str = "",
) -> ThreadingHTTPServer:
    class Proxy(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.relay()

        def do_POST(self) -> None:
            self.relay()

        def relay(self) -> None:
            release = (
                self.path == RELEASE_KV_PATH
                and role == "prefill"
                and self.command == "POST"
            )
            native = (
                self.path.startswith("/v1/")
                if role == "aggregated"
                else (
                    (self.command == "POST" and self.path in INFERENCE_PATHS)
                    or self.path == "/v1/models"
                )
            )
            if not release and self.path != "/health" and not native:
                self.send_error(404)
                return
            requested = self.headers.get(VERSION_HEADER)
            required = release or (
                role != "aggregated" and self.path in INFERENCE_PATHS
            )
            if requested != version and (requested is not None or required):
                self.reject(409, "stale or missing execution group version")
                return
            requested_role = self.headers.get(ROLE_HEADER)
            if requested_role is not None and requested_role != role:
                self.reject(409, "execution group role changed")
                return
            token = handoff_token(self.headers.get(HANDOFF_HEADER))
            if release:
                if token is None:
                    self.send_error(400, "invalid KV handoff token")
                    return
                gate.release_transfer(token)
                self.send_response(204)
                self.end_headers()
                return
            if role == "prefill" and self.path in INFERENCE_PATHS and token is None:
                self.send_error(400, "missing KV handoff token")
                return
            if not gate.admit(self.connection):
                self.reject(503, "execution group is not admitting requests")
                return
            try:
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header(VERSION_HEADER, version)
                    self.send_header(ROLE_HEADER, role)
                    if kv_resume:
                        self.send_header(KV_RESUME_HEADER, kv_resume)
                    self.end_headers()
                    return
                self.forward(
                    token
                    if role == "prefill" and self.path in INFERENCE_PATHS
                    else None
                )
            finally:
                gate.release(self.connection)

        def reject(self, status: int, reason: str) -> None:
            self.send_response(status, reason)
            self.send_header(ADMISSION_HEADER, "rejected")
            self.end_headers()

        def forward(self, transfer: str | None) -> None:
            upstream = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=60)
            sent_headers = False
            try:
                if self.headers.get("Transfer-Encoding"):
                    self.send_error(411)
                    return
                self.connection.settimeout(60)
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0:
                    self.send_error(400)
                    return
                body = self.rfile.read(length) if length else None
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in HOP_HEADERS and k.lower() not in PRIVATE_HEADERS
                }
                upstream.request(self.command, self.path, body=body, headers=headers)
                response = upstream.getresponse()
                if not gate.may_stream():
                    self.send_error(503, "execution group was revoked")
                    return
                if transfer is not None and response.status == 200:
                    # The native P response exports blocks that D has yet to read.
                    gate.hold_transfer(transfer, kv_lease)
                self.send_response(response.status, response.reason)
                self.send_header(VERSION_HEADER, version)
                self.send_header(ROLE_HEADER, role)
                for key, value in response.getheaders():
                    if (
                        key.lower() not in HOP_HEADERS
                        and key.lower() not in PRIVATE_HEADERS
                    ):
                        self.send_header(key, value)
                self.end_headers()
                sent_headers = True
                # HTTP/1.0 uses connection close to delimit the dechunked response.
                while gate.may_stream():
                    chunk = response.read1(65536)
                    if not chunk or not gate.may_stream():
                        break
                    self.wfile.write(chunk)
            except (OSError, ValueError, http.client.HTTPException):
                if not sent_headers:
                    with contextlib.suppress(OSError):
                        self.send_error(502)
            finally:
                upstream.close()

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    return ThreadingHTTPServer((bind, port), Proxy)


def expect_phase(result: dict) -> str:
    if "Err" in result:
        raise Rejected(f"serving operation rejected: {result['Err']}")
    return result["Ok"]["Group"]["phase"]


def stop_process_group(process: subprocess.Popen, grace: float) -> bool:
    def send(sig: int) -> bool:
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            if sig == 0:
                return True
            raise

    stopped_at = time.monotonic() + grace
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not send(sig):
            process.poll()
            return True
        while time.monotonic() < stopped_at:
            process.poll()
            if not send(0):
                return True
            time.sleep(0.05)
        stopped_at = time.monotonic() + 1.0
    return False


def inspect_group(control: Control, version: dict) -> dict:
    return control.request({"request": "inspect", "version": version})["group"]


def check_backend_port(port: int) -> None:
    with socket.socket() as listener:
        # Match the HTTP backend's bind behavior after an earlier group has drained.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)


def api_host_stopped(view: dict, api_host: str) -> bool:
    return sum(
        bytes(device["node"]).hex() == api_host for device in view["stopped"]
    ) == sum(
        bytes(grant["device"]["node"]).hex() == api_host for grant in view["grants"]
    )


def record_host_stop(
    control: Control,
    identity: dict,
    workers: list[dict],
    drain_timeout: float,
) -> None:
    version = identity["version"]
    deadline = time.monotonic() + drain_timeout + 50
    while True:
        result = control.execute({"HostLaunchStopped": identity})
        if result.get("Err") != "WorkersStillRunning":
            break
        if time.monotonic() >= deadline:
            raise ControlError(
                "timed out waiting for the API host to release its grants"
            )
        time.sleep(POLL_SECONDS)
    try:
        expect_phase(result)
    except Rejected:
        if inspect_group(control, version)["phase"] == "Retired":
            return
        raise
    result = control.execute({"Retire": {"version": version, "workers": workers}})
    if result.get("Err") in ("WorkersStillRunning", {"InvalidPhase": "Retired"}):
        return
    expect_phase(result)


def run(
    args: argparse.Namespace,
    command: list[str],
    launch_digest: list[int],
    env: dict[str, str],
    probe: Callable[[int], bool],
    should_stop: Callable[[], bool],
    *,
    endpoint: dict | None = None,
    host_launch_spec: dict | None = None,
    holder: list[int] | None = None,
) -> int:
    version = {"group": list(bytes.fromhex(args.group_id)), "epoch": args.group_epoch}
    version_tag = f"{args.group_id.lower()}@{args.group_epoch}"
    host = list(bytes.fromhex(args.host_id))
    host_id = bytes(host).hex()
    hosted = host_launch_spec is not None
    api_host = host_launch_spec["api_host"] if hosted else host_id
    follower = host_id != api_host
    config = None
    listen = args.control_listen
    if hosted:
        config = host_launch_spec["hosts"].get(host_id)
        if config is None:
            raise ControlError("this host is absent from --host-launch-spec")
        address = config["address"]
        if listen is None:
            listen = f"[{address}]:0" if ":" in address else f"{address}:0"
    control = Control(
        args.control_bin,
        args.control_peer,
        listen or "127.0.0.1:0",
        seeds=getattr(args, "control_seed", ()),
    )
    if holder is not None:
        control.holder = holder
    identity = {"version": version, "holder": control.holder}
    if hosted:
        identity["host"] = host
    gate = Gate()
    process = None
    claimed = False
    server = None
    proxy_thread = None
    phase = None
    drain_until = None
    ready_pipe = None
    release_driver = getattr(args, "release_driver", None)
    release_request = None
    release_env = None

    def change(name: str, **fields: object) -> str:
        return expect_phase(
            control.execute(
                {name: {**identity, **fields}},
                retry=name not in ("ObserveLaunch", "ObserveHostLaunch"),
            )
        )

    try:
        view = inspect_group(control, version)
        if not view or view["phase"] != "Preparing":
            raise ControlError("expected a prepared execution group")
        pp = view["pipeline_parallel"]
        if (
            view["version"] != version
            or view["role"] != args.role.capitalize()
            or view["model"] != args.model
            or view["tensor_parallel"] != args.tp
            or (pp != 1 and not hosted)
            or view["launch_digest"] != launch_digest
        ):
            raise ControlError(
                "the native launch configuration differs from the committed plan"
            )
        grants = view["grants"]
        workers = [grant["device"] for grant in grants]
        if len(grants) != args.tp * pp:
            raise ControlError("the number of grants differs from TP times PP")
        device_env = "CUDA_VISIBLE_DEVICES"
        child_env = env.copy()
        if hosted:
            try:
                node_rank, layout, grants = serving_launch.placement(
                    host_launch_spec, grants, host_id
                )
            except ValueError as error:
                raise ControlError(str(error)) from error
            if pp > 1:
                command = [*command, "--pipeline-parallel-size", str(pp)]
            command = serving_launch.host_command(
                command, host_launch_spec, host_id, node_rank, layout
            )
            child_env.update(config.get("env", {}))
            for mask in serving_launch.DEVICE_MASKS:
                child_env.pop(mask, None)
            device_env = config["device_env"]
            child_env["VLLM_USE_PLAIT"] = "1"
            child_env["VLLM_HOST_IP"] = config["address"]
            if release_driver is not None:
                for transport in (
                    "PLAIT_NCCL_SHM",
                    "PLAIT_NCCL_IPC",
                    "PLAIT_NCCL_KERNEL",
                ):
                    child_env[transport] = "0"
                release_request = {
                    "monitor": release_driver,
                    "host": host,
                    "version": version,
                    "grants": grants,
                }
                release_env = {
                    key: value
                    for key, value in child_env.items()
                    if key not in serving_launch.DEVICE_MASKS
                }
                if release_driver["driver"] == "amd_kfd":
                    capture = subprocess.run(
                        [args.control_bin, "capture-host-release"],
                        input=json.dumps(release_request) + "\n",
                        env=release_env,
                        text=True,
                        stdout=subprocess.PIPE,
                        timeout=25,
                        check=True,
                    )
                    release_request["identities"] = json.loads(capture.stdout)
        elif any(grant["device"]["node"] != host for grant in grants):
            raise ControlError("all grants must belong to the local host")
        elif len({grant["device"]["ordinal"] for grant in grants}) != args.tp:
            raise ControlError("duplicate device ordinal in group grants")

        expected_capabilities = {}
        if not follower:
            options = serving_launch.plait_kv_consumer_options(command)
            kv_resume = serving_launch.kv_resume_capability(options, child_env)
            if kv_resume == serving_launch.SEEDED_KV_RESUME:
                expected_capabilities[KV_RESUME_HEADER] = kv_resume
            if options and options.get("checkpoint_admission") is True:
                expected_capabilities[CHECKPOINT_ADMISSION_HEADER] = "1"
            if options and options.get("checkpoint_peer_budget") is True:
                expected_capabilities[CHECKPOINT_PEERS_HEADER] = "1"
            server = group_proxy(
                args.port_base,
                args.backend_port,
                gate,
                version_tag,
                bind=args.proxy_host,
                role=args.role,
                kv_lease=args.kv_lease_seconds,
                kv_resume=kv_resume,
            )
            check_backend_port(args.backend_port)
        phase = change(
            "ClaimHostLaunch" if hosted else "ClaimLaunch",
            host=host,
            launch_digest=launch_digest,
        )
        claimed = True
        if should_stop():
            return 0
        gpus = ",".join(str(grant["device"]["ordinal"]) for grant in grants)
        child_env[device_env] = gpus
        child_env["PLAIT_TP_GROUP_VERSION"] = version_tag
        child_env.pop("VLLM_PLAIT_READY_FD", None)
        if follower:
            ready_pipe, ready_writer = os.pipe()
            try:
                child_env["VLLM_PLAIT_READY_FD"] = str(ready_writer)
                process = subprocess.Popen(
                    command,
                    env=child_env,
                    start_new_session=True,
                    pass_fds=(ready_writer,),
                )
            finally:
                os.close(ready_writer)
        else:
            process = subprocess.Popen(command, env=child_env, start_new_session=True)
            proxy_thread = threading.Thread(target=server.serve_forever, daemon=True)
            proxy_thread.start()
        deadline = time.monotonic() + args.startup_timeout
        announced = False
        while not should_stop() and process.poll() is None:
            gate.expire_transfers()
            if drain_until is not None and (
                gate.drained() or time.monotonic() >= drain_until
            ):
                return 0
            if not announced:
                if follower:
                    ready = bool(select.select([ready_pipe], [], [], 0)[0])
                    if ready:
                        if os.read(ready_pipe, 1) != b"1":
                            raise ControlError(
                                "vLLM follower closed its readiness pipe"
                            )
                        os.close(ready_pipe)
                        ready_pipe = None
                else:
                    ready = probe(args.backend_port)
                if ready and process.poll() is None:
                    if expected_capabilities:
                        verify_backend_capabilities(
                            args.backend_port, expected_capabilities
                        )
                    if hosted:
                        phase = change(
                            "HostLaunchReady", endpoint=None if follower else endpoint
                        )
                    else:
                        phase = (
                            change("LaunchReady")
                            if endpoint is None
                            else change("LaunchReadyAt", endpoint=endpoint)
                        )
                    announced = True

            sent = time.monotonic()
            try:
                phase = change("ObserveHostLaunch" if hosted else "ObserveLaunch")
            except Rejected:
                raise
            except ControlError:
                # Existing streams can finish while the admission window expires.
                if not announced and time.monotonic() >= deadline:
                    raise
                time.sleep(POLL_SECONDS)
                continue

            if phase == "Active":
                if (
                    not follower
                    and announced
                    and process.poll() is None
                    and probe(args.backend_port)
                ):
                    gate.allow_until(sent + ADMISSION_SECONDS)
                else:
                    gate.close()
            elif phase in ("Revoked", "Retired"):
                return 1
            elif phase == "Draining":
                gate.close()
                if follower:
                    try:
                        view = inspect_group(control, version)
                    except ControlError:
                        time.sleep(POLL_SECONDS)
                        continue
                    if api_host_stopped(view, api_host):
                        return 0
                elif drain_until is None:
                    drain_until = time.monotonic() + args.drain_timeout

            if not announced and time.monotonic() >= deadline:
                raise ControlError("vLLM startup timed out")
            time.sleep(POLL_SECONDS)
        return 0 if should_stop() else 1
    finally:
        try:
            gate.close(revoke=True)
            if claimed and phase not in ("Draining", "Revoked", "Retired"):
                with contextlib.suppress(ControlError):
                    control.execute({"Revoke": version}, retry=False)
            if ready_pipe is not None:
                os.close(ready_pipe)
            if server is not None:
                if proxy_thread is not None:
                    server.shutdown()
                    proxy_thread.join()
                server.server_close()
            exited = process is None or stop_process_group(process, 20.0)
            deadline = time.monotonic() + 1.0
            while not gate.drained() and time.monotonic() < deadline:
                time.sleep(0.05)
            if claimed and exited and gate.drained():
                if process is None:
                    released = True
                elif release_request is not None:
                    try:
                        proof = subprocess.run(
                            [args.control_bin, "verify-host-release"],
                            input=json.dumps(release_request) + "\n",
                            env=release_env,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                        released = proof.returncode == 0
                    except (OSError, subprocess.TimeoutExpired):
                        released = False
                elif args.verify_release:
                    # The verifier attests device and transport cleanup.
                    proof = subprocess.run(
                        [args.verify_release],
                        input=json.dumps({"version": version, "grants": grants}) + "\n",
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    released = proof.returncode == 0
                else:
                    released = False
                if released:
                    if hosted:
                        record_host_stop(control, identity, workers, args.drain_timeout)
                    else:
                        change("LaunchStopped")
                        expect_phase(
                            control.execute(
                                {"Retire": {"version": version, "workers": workers}}
                            )
                        )
                elif hosted and not follower and phase == "Draining":
                    expect_phase(control.execute({"Revoke": version}))
        finally:
            control.close()
