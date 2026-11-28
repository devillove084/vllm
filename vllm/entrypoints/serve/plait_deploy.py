# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch vLLM from Plait grants and publish healthy, versioned origins."""

import argparse
import ctypes
import errno
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
from dataclasses import dataclass, replace
from hashlib import sha256
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from socket import AF_INET
from typing import IO, Any

import psutil

from vllm.entrypoints.serve.plait_version import (
    DRAIN_PATH,
    GROUP_ROLES,
    ROLE_HEADER,
    VERSION_HEADER,
)


class ControlUnavailable(RuntimeError):
    pass


ENGINE_STOP_GRACE_SECONDS = 10


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

    def observe(self, version: dict, *, with_recipe: bool = False) -> dict | None:
        fields = {"with_recipe": True} if with_recipe else {}
        reply = self.request("observe", id=operation_id(), **version, **fields)
        result = reply["result"]
        if "Ok" in result:
            view = result["Ok"]
            if with_recipe and "recipe" in reply:
                binding = reply["recipe"]
                if binding["version"] != view["version"]:
                    raise ControlUnavailable("recipe and group view versions differ")
                view["recipe"] = binding["recipe"]
            return view
        if isinstance(result["Err"], dict) and "UnknownVersion" in result["Err"]:
            return None
        raise ControlUnavailable(f"Plait observation rejected: {result['Err']}")


def operation_id() -> dict:
    return {"origin": list(uuid.uuid4().bytes), "counter": 1}


def parse_id(text: str) -> bytes:
    if not re.fullmatch("[0-9a-f]{32}", text):
        raise ValueError("expected a lowercase 32-digit hexadecimal Plait ID")
    return bytes.fromhex(text)


def recipe_binding_id(base_id: str, width: int, egress_bytes: int | None) -> str:
    if egress_bytes is None:
        return base_id
    digest = sha256(b"plait-peer-all-reduce-recipe-v1")
    digest.update(parse_id(base_id))
    digest.update(width.to_bytes(8, "little"))
    digest.update(egress_bytes.to_bytes(8, "little"))
    return digest.hexdigest()[:32]


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
    layout_recipes: dict[str, dict] | None = None

    @classmethod
    def read(cls, path: Path) -> "Deployment":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def reload_hosts(self, path: Path) -> "Deployment":
        """Read host and recipe additions while retaining existing bindings."""
        updated = self.read(path)
        old_recipes = self.layout_recipes or {}
        new_recipes = updated.layout_recipes or {}
        if (
            replace(updated, hosts=self.hosts, layout_recipes=self.layout_recipes)
            != self
            or not self.hosts.items() <= updated.hosts.items()
            or any(new_recipes.get(key) != value for key, value in old_recipes.items())
        ):
            raise ValueError("live deployment changes may only add hosts or recipes")
        return updated

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
        if self.layout_recipes is not None and not isinstance(
            self.layout_recipes, dict
        ):
            raise ValueError("layout_recipes must map IDs to recipe objects")
        for recipe_id, recipe in (self.layout_recipes or {}).items():
            parse_id(recipe_id)
            if not isinstance(recipe, dict):
                raise ValueError("layout_recipes must map IDs to recipe objects")
            width = recipe.get("width")
            layers = recipe.get("pipeline_layers", [])
            egress = recipe.get("replicated_egress_bytes")
            if (
                set(recipe) - {"width", "pipeline_layers", "replicated_egress_bytes"}
                or type(width) is not int
                or not 0 < width < 2**32
                or not isinstance(layers, list)
                or (
                    egress is not None
                    and (layers or type(egress) is not int or not 0 <= egress < 2**64)
                )
                or (
                    layers
                    and (
                        len(layers) != width
                        or width < 2
                        or any(type(layer) is not int or layer < 1 for layer in layers)
                    )
                )
            ):
                raise ValueError(
                    "layout recipe needs a valid width, layers and TP AllReduce budget"
                )
        if len(self.recipe_bindings()) != len(self.layout_recipes or {}):
            raise ValueError("duplicate committed Plait recipe binding")
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

    def recipe_bindings(self) -> dict[str, dict]:
        return {
            recipe_binding_id(
                key, recipe["width"], recipe.get("replicated_egress_bytes")
            ): recipe
            for key, recipe in (self.layout_recipes or {}).items()
        }

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
        egress = None
        if "recipe" not in view and self.layout_recipes:
            raise ValueError(
                "layout recipes require a recipe-aware Plait control plane"
            )
        if view.get("recipe") is not None:
            recipe_id = bytes(view["recipe"]).hex()
            recipe = self.recipe_bindings().get(recipe_id)
            if recipe is None or recipe["width"] != width:
                raise ValueError(
                    f"missing launch definition for committed recipe {recipe_id}"
                )
            layers = recipe.get("pipeline_layers", ())
            egress = recipe.get("replicated_egress_bytes")
        offset = (version["epoch"] - 1) % self.epoch_ports
        return Placement(
            version,
            f"{self.group_id}@{version['epoch']}",
            ordered,
            self.hosts[next(iter(ordered))],
            self.api_port_base + offset,
            self.rendezvous_port_base + offset,
            tuple(layers),
            egress,
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
    replicated_egress_bytes: int | None = None

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
        if self.replicated_egress_bytes is not None:
            group["replicated_egress_bytes"] = self.replicated_egress_bytes
        axis = "pp" if self.pipeline_layers else "tp"
        return {"host_id": host, "seeds": seeds, f"{axis}_groups": [group]}


@dataclass
class EngineGrantReturn:
    path: Path
    record: dict
    lock: IO[str]

    @classmethod
    def persist(
        cls,
        path: Path,
        host: str,
        version: dict,
        grants: list[dict],
        lock: IO[str],
        error: OSError | None = None,
    ) -> "EngineGrantReturn":
        if not grants:
            raise ValueError("engine has no local Plait grants")
        record = {
            "host": host,
            "version": version,
            "grants": grants,
            "origin": uuid.uuid4().hex,
        }
        if error is not None:
            record["error"] = str(error)
        write_json(path, record)
        return cls(path, record, lock)

    @classmethod
    def recover(
        cls, path: Path, group_id: str, host: str
    ) -> "EngineGrantReturn | None":
        prefix, evidence, _ = path.name.rsplit(".", 2)
        lock_path = path.with_name(f"{prefix}.lock")
        lock = lock_path.open("a")
        with ExitStack() as stack:
            stack.callback(lock.close)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                record = json.loads(path.read_text(encoding="utf-8"))
            except (BlockingIOError, FileNotFoundError):
                return None
            try:
                version = record["version"]
                epoch = version["epoch"]
                parse_id(record["origin"])
                if (
                    record["host"] != host
                    or version["group"] != list(parse_id(group_id))
                    or type(epoch) is not int
                    or epoch <= 0
                    or prefix != f"{group_id}@{epoch}"
                    or evidence not in ("failed", "stopped")
                    or ("error" in record) != (evidence == "failed")
                    or not isinstance(record["grants"], list)
                    or not record["grants"]
                ):
                    raise ValueError("mismatched journal identity or grants")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid engine grant return journal {path.name}"
                ) from error
            stack.pop_all()
        return cls(path, record, lock)

    def reconcile(self, control: Control) -> bool:
        version = self.record["version"]
        view = control.observe(version)
        if view is None:
            return False
        grants = self.record["grants"]
        if (
            view["version"] != version
            or local_grants(view, self.record["host"]) != grants
        ):
            raise ValueError(f"engine grants changed for {self.path.name}")
        origin = list(parse_id(self.record["origin"]))
        if view["phase"] == "Preparing" and "error" in self.record:
            apply_operation(
                control,
                {
                    "id": {"origin": origin, "counter": 1},
                    "action": {
                        "ReportFailure": {"grant": grants[0], "fault": "Device"}
                    },
                },
            )
            return False
        if view["phase"] not in ("Revoked", "Retired"):
            raise ValueError(f"engine grants cannot return during {view['phase']}")
        # Use the original IDs even when a release committed before a crash.
        for counter, grant in enumerate(grants, start=2):
            apply_operation(
                control,
                {
                    "id": {"origin": origin, "counter": counter},
                    "action": {"Release": grant},
                },
            )
        self.path.unlink(missing_ok=True)
        sync_directory(self.path.parent)
        self.lock.close()
        version_name = self.path.name.rsplit(".", 2)[0]
        print(
            f"Released local Plait grants for {version_name}",
            flush=True,
        )
        return True


def local_grants(view: dict, host: str) -> list[dict]:
    return sorted(
        (
            grant
            for grant in view["grants"]
            if bytes(grant["device"]["node"]).hex() == host
        ),
        key=lambda grant: grant["device"]["ordinal"],
    )


@dataclass
class Engine:
    argv: list[str]
    model: str
    device_env: str
    args: list[str]
    env: dict[str, str]
    api_role: str = "aggregated"

    def __post_init__(self) -> None:
        if not self.argv or not self.model or not self.device_env:
            raise ValueError("engine needs an executable, model and device mask")
        if self.api_role not in GROUP_ROLES:
            raise ValueError("expected an aggregated, prefill or decode API role")
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
        self,
        deployment: Deployment,
        placement: Placement,
        host: str,
        state: Path,
        control: Control,
    ) -> "WorkerProcess | EngineGrantReturn | None":
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
        with ExitStack() as stack:
            lock = stack.enter_context(descriptor.with_suffix(".lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            failed = descriptor.with_suffix(".failed.json")
            stopped = descriptor.with_suffix(".stopped.json")
            if failed.exists() or stopped.exists():
                return None
            view = control.observe(placement.version)
            if view is None or view["phase"] != "Preparing":
                return None
            grants = local_grants(view, host)
            if (
                view["version"] != placement.version
                or [grant["device"]["ordinal"] for grant in grants]
                != placement.devices[host]
            ):
                raise ValueError("local grants changed before engine start")
            prior_launch = descriptor.exists()
            binding = placement.descriptor(
                host, deployment.seeds, deployment.compute_calibration
            )
            if self.api_role != "aggregated":
                binding["api_role"] = self.api_role
            descriptor.write_text(json.dumps(binding), encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command, env=env, start_new_session=True, pass_fds=(lock.fileno(),)
                )
            except OSError as error:
                # An earlier launch may have left descendants. Only a first
                # attempt with a definite exec rejection proves no process ran.
                if prior_launch or error.errno not in (
                    errno.ENOENT,
                    errno.ENOTDIR,
                    errno.EACCES,
                    errno.ENOEXEC,
                ):
                    raise
                returning = EngineGrantReturn.persist(
                    failed, host, placement.version, grants, lock, error
                )
                stack.pop_all()
                return returning
            worker = WorkerProcess(
                process=process,
                placement=placement,
                grants=grants,
                drain_marker=descriptor.with_suffix(".drain")
                if node_rank == 0
                else None,
                api_role=self.api_role,
                host=host,
                stop_record=stopped,
                lock=lock,
            )
            stack.pop_all()
            return worker


def versioned_status(
    placement: Placement, path: str, status: int, *, role: str = "aggregated"
) -> bool:
    headers = {VERSION_HEADER: placement.version_text, ROLE_HEADER: role}
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


def sync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path: Path, document: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        pending = Path(file.name)
        try:
            json.dump(document, file)
            file.flush()
            os.fsync(file.fileno())
            file.close()
            pending.replace(path)
            sync_directory(path.parent)
        finally:
            pending.unlink(missing_ok=True)


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
    write_json(path, routes)


def apply_operation(control: Control, operation: dict) -> None:
    result = control.request("apply", operation=operation)["result"]
    if "Ok" not in result:
        raise ControlUnavailable(f"Plait group operation rejected: {result['Err']}")


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
    host: str
    stop_record: Path
    lock: IO[str]
    api_role: str = "aggregated"
    closing: bool = False
    pending: dict | None = None
    kill_after: float | None = None
    group_exited: bool = False
    returning: EngineGrantReturn | None = None

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
                if versioned_status(
                    self.placement, DRAIN_PATH, 204, role=self.api_role
                ):
                    self.apply(control, {"Revoke": version})
            elif phase in ("Revoked", "Retired"):
                if not self.gone():
                    self.signal(signal.SIGTERM)
                    self.kill_after = time.monotonic() + ENGINE_STOP_GRACE_SECONDS
                self.closing = True
        if not self.closing:
            return False
        if not self.gone():
            if self.kill_after is not None and time.monotonic() >= self.kill_after:
                self.signal(signal.SIGKILL)
                self.kill_after = None
            return False
        if self.returning is None:
            self.returning = EngineGrantReturn.persist(
                self.stop_record, self.host, version, self.grants, self.lock
            )
        return self.returning.reconcile(control)

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
    deployment_file: Path | None = None,
) -> None:
    group = list(parse_id(deployment.group_id))
    children: dict[int, WorkerProcess] = {}
    grant_returns: dict[Path, EngineGrantReturn] = {}
    last_error = None
    if engine is not None:
        reap_orphaned_workers()
    try:
        local_errors: dict[int, str] = {}
        return_errors: dict[Path, str] = {}
        while True:
            if engine is not None and state is not None and host is not None:
                journals = set(state.glob(f"{deployment.group_id}@*.failed.json"))
                journals.update(state.glob(f"{deployment.group_id}@*.stopped.json"))
                for journal in sorted(journals.union(grant_returns)):
                    try:
                        returning = grant_returns.get(journal)
                        if returning is None:
                            returning = EngineGrantReturn.recover(
                                journal, deployment.group_id, host
                            )
                            if returning is None:
                                continue
                            grant_returns[journal] = returning
                        if returning.reconcile(control):
                            del grant_returns[journal]
                        return_errors.pop(journal, None)
                    except (ControlUnavailable, OSError, ValueError) as error:
                        if return_errors.get(journal) != str(error):
                            return_errors[journal] = str(error)
                            print(
                                f"Plait deployment for {journal.name}: {error}",
                                flush=True,
                            )
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
                        if deployment_file is not None:
                            deployment = deployment.reload_hosts(deployment_file)
                        view = control.observe(version, with_recipe=True)
                        if view is not None and view["phase"] == phase:
                            placement = deployment.place(view)
                            if (
                                launch
                                and engine is not None
                                and state is not None
                                and host is not None
                                and host in placement.devices
                            ):
                                process = engine.start(
                                    deployment, placement, host, state, control
                                )
                                if isinstance(process, EngineGrantReturn):
                                    grant_returns[process.path] = process
                                    print(
                                        f"Plait engine for epoch {epoch} "
                                        "failed to start: "
                                        f"{process.record['error']}",
                                        flush=True,
                                    )
                                elif process is not None:
                                    children[epoch] = process
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
        for returning in grant_returns.values():
            returning.lock.close()
        for child in children.values():
            if not child.gone():
                child.signal(signal.SIGTERM)
        for child in children.values():
            try:
                child.process.wait(timeout=ENGINE_STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                child.signal(signal.SIGKILL)
                child.process.wait()
            finally:
                child.lock.close()


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
    deployment = Deployment.read(args.deployment)
    engine = None
    if args.engine is not None:
        if args.host_id not in deployment.hosts:
            parser.error("--host-id must be present in the deployment")
        engine = Engine(**json.loads(args.engine.read_text(encoding="utf-8")))
        if engine.api_role != "aggregated" and args.routes is not None:
            parser.error(
                "--routes publishes one aggregate group. Activate P/D groups as a set."
            )
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
            supervise(
                deployment,
                control,
                args.host_id,
                engine,
                args.state_dir,
                router,
                args.deployment,
            )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
