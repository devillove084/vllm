# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch vLLM from Plait grants and publish healthy, versioned origins."""

import argparse
import ctypes
import fcntl
import ipaddress
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack, closing, suppress
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from socket import AF_INET
from typing import Any

import psutil

from vllm.entrypoints.serve.plait_version import DRAIN_PATH, ROLE_HEADER, VERSION_HEADER


class ControlUnavailable(RuntimeError):
    pass


class Control:
    def __init__(self, binary: str, listen: str, seeds: list[str]):
        self.argv = [binary, "group", "--listen", listen]
        for seed in seeds:
            self.argv.extend(("--seed", seed))

    def request(self, request: str, **fields: Any) -> dict:
        try:
            result = subprocess.run(
                self.argv,
                input=json.dumps({"request": request, **fields}) + "\n",
                capture_output=True,
                encoding="utf-8",
                timeout=15,
                check=True,
            )
            reply = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise ControlUnavailable("Plait control request failed") from error
        if "error" in reply:
            raise ControlUnavailable(reply["error"])
        return reply

    def observe(self, version: dict) -> dict | None:
        result = self.request("observe", id=operation_id(), **version)["result"]
        if "Ok" in result:
            return result["Ok"]
        if isinstance(result["Err"], dict) and "UnknownVersion" in result["Err"]:
            return None
        raise ControlUnavailable(f"Plait observation rejected: {result['Err']}")


def operation_id() -> dict:
    return {"origin": list(uuid.uuid4().bytes), "counter": 1}


def parse_id(text: str) -> bytes:
    if not re.fullmatch("[0-9a-f]{32}", text):
        raise ValueError("expected a lowercase 32-digit hexadecimal Plait ID")
    return bytes.fromhex(text)


@dataclass
class Deployment:
    group_id: str
    seeds: list[str]
    model_alias: str
    hosts: dict[str, str]
    api_port_base: int
    rendezvous_port_base: int
    epoch_ports: int
    compute_calibration: dict | None = None
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 128
    pipeline_layers: dict[str, list[int]] | None = None

    def __post_init__(self) -> None:
        parse_id(self.group_id)
        if not self.model_alias or not self.seeds or not self.hosts:
            raise ValueError("model alias, seeds and host addresses are required")
        if not 0 < self.max_num_seqs <= self.max_num_batched_tokens:
            raise ValueError("batch limits must allow at least one token per sequence")
        for width, stages in (self.pipeline_layers or {}).items():
            if (
                width != str(len(stages))
                or len(stages) < 2
                or any(type(layers) is not int or layers < 1 for layers in stages)
            ):
                raise ValueError(
                    "pipeline_layers needs one positive layer count per stage"
                )
        for host, address in self.hosts.items():
            parse_id(host)
            ip = ipaddress.ip_address(address)
            if ip.version != 4 or ip.is_unspecified or ip.is_multicast:
                raise ValueError("deployment requires reachable IPv4 host addresses")
        bases = sorted((self.api_port_base, self.rendezvous_port_base))
        if (
            self.epoch_ports < 2
            or bases[0] < 1
            or bases[0] + self.epoch_ports > bases[1]
            or bases[1] + self.epoch_ports > 65536
        ):
            raise ValueError("configure disjoint API and rendezvous port ranges")

    def place(self, view: dict) -> "Placement":
        version = view["version"]
        if bytes(version["group"]).hex() != self.group_id or version["epoch"] < 1:
            raise ValueError("group view does not match this deployment")
        devices: dict[str, list[int]] = {}
        for grant in view["grants"]:
            device = grant["device"]
            host = bytes(device["node"]).hex()
            devices.setdefault(host, []).append(device["ordinal"])
        if not devices or devices.keys() - self.hosts.keys():
            raise ValueError("placement needs a host outside the deployment")
        ordered = {host: sorted(ordinals) for host, ordinals in sorted(devices.items())}
        width = sum(len(ordinals) for ordinals in ordered.values())
        layers = (self.pipeline_layers or {}).get(str(width), ())
        offset = (version["epoch"] - 1) % self.epoch_ports
        return Placement(
            version,
            f"{self.group_id}@{version['epoch']}",
            ordered,
            self.hosts[next(iter(ordered))],
            self.api_port_base + offset,
            self.rendezvous_port_base + offset,
            tuple(layers),
        )


@dataclass
class Placement:
    version: dict
    version_text: str
    devices: dict[str, list[int]]
    api_address: str
    api_port: int
    rendezvous_port: int
    pipeline_layers: tuple[int, ...] = ()

    @property
    def origin(self) -> str:
        return f"http://{self.api_address}:{self.api_port}"

    def descriptor(
        self, host: str, seeds: list[str], calibration: dict | None = None
    ) -> dict:
        devices = [
            [node, ordinal]
            for node, ordinals in self.devices.items()
            for ordinal in ordinals
        ]
        group = {
            "version": self.version_text,
            "ranks": list(range(len(devices))),
            "devices": devices,
        }
        if calibration is not None:
            group["compute_calibration"] = calibration
        axis = "pp" if self.pipeline_layers else "tp"
        return {"host_id": host, "seeds": seeds, f"{axis}_groups": [group]}


@dataclass
class Engine:
    argv: list[str]
    model: str
    device_env: str
    args: list[str]
    env: dict[str, str]

    def __post_init__(self) -> None:
        if not self.argv or not self.model or not self.device_env:
            raise ValueError("engine needs an executable, model and device mask")
        reserved = {
            "--host",
            "--port",
            "--headless",
            "--api-server-count",
            "--nnodes",
            "--node-rank",
            "--master-addr",
            "--master-port",
            "--plait-tp-layout",
            "--served-model-name",
            "--distributed-executor-backend",
            "--model",
            "--tensor-parallel-size",
            "--pipeline-parallel-size",
            "--data-parallel-size",
            "--prefill-context-parallel-size",
            "--decode-context-parallel-size",
            "--max-num-batched-tokens",
            "--max-num-seqs",
            "--external-launcher",
            "--config",
            "-tp",
            "-pp",
            "-dp",
            "-dcp",
            "-pcp",
            "-n",
            "-r",
            "--",
        }
        # Include abbreviations because argparse accepts unique flag prefixes.
        if any(
            flag.startswith(arg.split("=", 1)[0].replace("_", "-"))
            for arg in self.args
            if arg.startswith("-")
            for flag in reserved
        ):
            raise ValueError("engine arguments contain a deployment-owned flag")

    def start(
        self, deployment: Deployment, placement: Placement, host: str, state: Path
    ) -> subprocess.Popen[bytes]:
        descriptor = state / f"{placement.version_text}.json"
        nodes = list(placement.devices)
        counts = [len(placement.devices[node]) for node in nodes]
        node_rank = nodes.index(host)
        tp = 1 if placement.pipeline_layers else sum(counts)
        pp = len(placement.pipeline_layers) or 1
        command = [
            *self.argv,
            "serve",
            self.model,
            *self.args,
            "--tensor-parallel-size",
            str(tp),
            "--pipeline-parallel-size",
            str(pp),
            "--served-model-name",
            deployment.model_alias,
            "--distributed-executor-backend",
            "mp",
            "--max-num-batched-tokens",
            str(deployment.max_num_batched_tokens),
            "--max-num-seqs",
            str(deployment.max_num_seqs),
        ]
        if len(nodes) > 1:
            command.extend(
                (
                    "--nnodes",
                    str(len(nodes)),
                    "--node-rank",
                    str(node_rank),
                    "--master-addr",
                    placement.api_address,
                    "--master-port",
                    str(placement.rendezvous_port),
                    "--plait-tp-layout",
                    *map(str, counts),
                )
            )
        if node_rank:
            command.append("--headless")
        else:
            command.extend(
                (
                    "--host",
                    placement.api_address,
                    "--port",
                    str(placement.api_port),
                    "--api-server-count",
                    "1",
                    "--middleware",
                    "vllm.entrypoints.serve.plait_version.NativeGroupHeaders",
                )
            )
        env = (
            os.environ
            | self.env
            | {
                self.device_env: ",".join(map(str, placement.devices[host])),
                "VLLM_USE_PLAIT": "1",
                "VLLM_HOST_IP": deployment.hosts[host],
                "VLLM_PLAIT_TP_GROUP_CONFIG": str(descriptor.resolve()),
            }
        )
        if placement.pipeline_layers:
            env["VLLM_PP_LAYER_PARTITION"] = ",".join(
                map(str, placement.pipeline_layers)
            )
        if len(nodes) > 1 and "GLOO_SOCKET_IFNAME" not in env:
            # Gloo otherwise picks a NIC using the hostname, which can resolve
            # to a management address on multi-homed workers.
            for interface, addresses in psutil.net_if_addrs().items():
                if any(
                    addr.family == AF_INET and addr.address == env["VLLM_HOST_IP"]
                    for addr in addresses
                ):
                    env["GLOO_SOCKET_IFNAME"] = interface
                    break
        # A surviving engine owns its launch lock across supervisor restarts.
        with descriptor.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptor.write_text(
                json.dumps(
                    placement.descriptor(
                        host, deployment.seeds, deployment.compute_calibration
                    )
                ),
                encoding="utf-8",
            )
            return subprocess.Popen(
                command, env=env, start_new_session=True, pass_fds=(lock.fileno(),)
            )


def versioned_status(placement: Placement, path: str, status: int) -> bool:
    headers = {VERSION_HEADER: placement.version_text, ROLE_HEADER: "aggregated"}
    try:
        with closing(
            HTTPConnection(placement.api_address, placement.api_port, 2)
        ) as conn:
            conn.request("GET", path, headers=headers)
            with conn.getresponse() as response:
                return response.status == status and all(
                    response.getheader(key) == value for key, value in headers.items()
                )
    except (OSError, HTTPException):
        return False


def healthy(placement: Placement) -> bool:
    return versioned_status(placement, "/health", 200)


def publish(path: Path, placement: Placement) -> None:
    try:
        routes = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        routes = {}
    epoch = str(placement.version["epoch"])
    if epoch in routes:
        if routes[epoch] != placement.origin:
            raise ValueError(f"epoch {epoch} already has a different HTTP origin")
        return
    routes[epoch] = placement.origin
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        pending = Path(file.name)
        try:
            json.dump(routes, file)
            file.flush()
            os.fsync(file.fileno())
            file.close()
            pending.replace(path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            pending.unlink(missing_ok=True)


class RoutePublisher:
    def __init__(self, path: Path):
        self.path = path
        self.pending: dict | None = None
        self.bound_epoch: int | None = None

    def reconcile(self, control: Control, placement: Placement, phase: str) -> None:
        if phase not in ("Ready", "Active") or not healthy(placement):
            return
        publish(self.path, placement)
        if phase == "Active":
            self.pending = None
            self.bound_epoch = placement.version["epoch"]
            return
        if (
            self.pending is None
            or self.pending["action"]["Activate"] != placement.version
        ):
            self.pending = {
                "id": operation_id(),
                "action": {"Activate": placement.version},
            }
        result = control.request("apply", operation=self.pending)["result"]
        self.pending = None
        if "Ok" not in result:
            raise ControlUnavailable(f"Plait activation rejected: {result['Err']}")
        self.bound_epoch = placement.version["epoch"]
        print(f"Activated Plait group {placement.version_text}", flush=True)


@dataclass
class WorkerProcess:
    process: subprocess.Popen[bytes]
    placement: Placement
    grants: list[dict]
    drain_marker: Path | None
    closing: bool = False
    pending: dict | None = None
    release_index: int = 0
    group_exited: bool = False

    def apply(self, control: Control, action: dict) -> None:
        if self.pending is None or self.pending["action"] != action:
            self.pending = {"id": operation_id(), "action": action}
        result = control.request("apply", operation=self.pending)["result"]
        self.pending = None
        if "Ok" not in result:
            raise ControlUnavailable(f"Plait group operation rejected: {result['Err']}")

    def gone(self) -> bool:
        if self.group_exited:
            return True
        if self.process.poll() is None:
            return False
        # Reap adopted descendants in this process group after its leader
        # exits. A leader exit alone does not prove the device owner left.
        with suppress(ChildProcessError):
            while os.waitpid(-self.process.pid, os.WNOHANG)[0]:
                pass
        try:
            os.killpg(self.process.pid, 0)
        except ProcessLookupError:
            self.group_exited = True
            return True
        return False

    def reconcile(self, control: Control) -> bool:
        version = self.placement.version
        if not self.closing:
            view = control.request("view", version=version)["group"]
            phase = view["phase"] if view else None
            if phase in ("Draining", "Revoked", "Retired"):
                view = control.observe(version)
                phase = view["phase"] if view else None
            if phase not in ("Revoked", "Retired") and self.process.poll() is not None:
                self.apply(
                    control,
                    {"ReportFailure": {"grant": self.grants[0], "fault": "Device"}},
                )
            elif phase == "Draining" and self.drain_marker is not None:
                self.drain_marker.touch(exist_ok=True)
                if versioned_status(self.placement, DRAIN_PATH, 204):
                    self.apply(control, {"Revoke": version})
            elif phase in ("Revoked", "Retired"):
                if not self.gone():
                    self.signal(signal.SIGTERM)
                self.closing = True
        if not self.closing or not self.gone():
            return False
        while self.release_index < len(self.grants):
            self.apply(control, {"Release": self.grants[self.release_index]})
            self.release_index += 1
        print(
            f"Released local Plait grants for {self.placement.version_text}", flush=True
        )
        return True

    def signal(self, sig: signal.Signals) -> None:
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, sig)


def reap_orphaned_workers() -> None:
    if sys.platform == "linux":
        # Keep orphaned vLLM workers attached to this launcher, including
        # inside containers whose PID 1 does not reap children.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))


def supervise(
    deployment: Deployment,
    control: Control,
    host: str | None,
    engine: Engine | None,
    state: Path | None,
    router: RoutePublisher | None,
) -> None:
    group = list(parse_id(deployment.group_id))
    children: dict[int, WorkerProcess] = {}
    last_error = None
    if engine is not None:
        reap_orphaned_workers()
    try:
        local_errors: dict[int, str] = {}
        while True:
            for epoch, child in list(children.items()):
                try:
                    if child.reconcile(control):
                        del children[epoch]
                    local_errors.pop(epoch, None)
                except (ControlUnavailable, OSError, ValueError) as error:
                    if local_errors.get(epoch) != str(error):
                        local_errors[epoch] = str(error)
                        print(
                            f"Plait deployment for epoch {epoch}: {error}", flush=True
                        )
            try:
                latest = control.request("placement", group=group)["placement"][
                    "latest"
                ]
                if latest is not None:
                    version, phase = latest
                    epoch = version["epoch"]
                    launch = (
                        engine is not None
                        and phase == "Preparing"
                        and epoch not in children
                    )
                    route = (
                        router is not None
                        and phase in ("Ready", "Active")
                        and router.bound_epoch != epoch
                    )
                    if launch or route:
                        view = control.observe(version)
                        if view is not None and view["phase"] == phase:
                            placement = deployment.place(view)
                            if (
                                launch
                                and engine is not None
                                and state is not None
                                and host is not None
                                and host in placement.devices
                            ):
                                grants = [
                                    grant
                                    for grant in view["grants"]
                                    if bytes(grant["device"]["node"]).hex() == host
                                ]
                                drain_marker = None
                                if host == next(iter(placement.devices)):
                                    drain_marker = (
                                        state / f"{placement.version_text}.drain"
                                    )
                                process = engine.start(
                                    deployment, placement, host, state
                                )
                                children[epoch] = WorkerProcess(
                                    process,
                                    placement,
                                    grants,
                                    drain_marker,
                                )
                            if route and router is not None:
                                router.reconcile(control, placement, phase)
                last_error = None
            except (ControlUnavailable, OSError, ValueError) as error:
                message = str(error)
                if message != last_error:
                    print(f"Plait deployment: {message}", flush=True)
                    last_error = message
            time.sleep(0.5)
    finally:
        for child in children.values():
            if not child.gone():
                child.signal(signal.SIGTERM)
        for child in children.values():
            try:
                child.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.signal(signal.SIGKILL)
                child.process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deployment", type=Path)
    parser.add_argument("--control-bin", default="plait-node")
    parser.add_argument("--control-listen", required=True)
    parser.add_argument("--host-id")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--routes", type=Path)
    args = parser.parse_args()
    if args.engine is None and args.routes is None:
        parser.error("provide --engine or --routes")
    if args.engine is not None and (args.host_id is None or args.state_dir is None):
        parser.error("--engine requires --host-id and --state-dir")
    deployment = Deployment(**json.loads(args.deployment.read_text(encoding="utf-8")))
    engine = None
    if args.engine is not None:
        if args.host_id not in deployment.hosts:
            parser.error("--host-id must be present in the deployment")
        engine = Engine(**json.loads(args.engine.read_text(encoding="utf-8")))
        args.state_dir.mkdir(parents=True, exist_ok=True)
    control = Control(args.control_bin, args.control_listen, deployment.seeds)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        with ExitStack() as stack:
            router = None
            if args.routes is not None:
                args.routes.parent.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context(args.routes.with_suffix(".lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                router = RoutePublisher(args.routes)
            supervise(deployment, control, args.host_id, engine, args.state_dir, router)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
