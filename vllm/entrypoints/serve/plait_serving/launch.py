# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Huanbing
"""Compile a hosted vLLM launch from the committed group's device grants."""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path

DEVICE_MASKS = {
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
}
PLACEMENT_FLAGS = {
    "--nnodes",
    "--node-rank",
    "-n",
    "-r",
    "--master-addr",
    "--master-port",
    "--plait-tp-layout",
    "--headless",
    "--api-server-count",
    "--distributed-executor-backend",
    "--tensor-parallel-size",
    "-tp",
    "--pipeline-parallel-size",
    "-pp",
    "--data-parallel-size",
    "-dp",
    "--prefill-context-parallel-size",
    "-pcp",
    "--host",
    "--port",
    "--served-model-name",
    "--model",
    "--no-enforce-eager",
    "--",
}


def load_spec(path: Path) -> dict:
    """Validate the launch document before hashing or starting a process."""
    return validate_spec(json.loads(path.read_text()))


def resident_spec(recipe: dict, engine: dict, *, pd: bool = False) -> dict:
    """Project the committed topology onto the installed local engine profile."""
    api_host = bytes(recipe["api_host"]).hex()
    program = {
        key: engine[key] for key in ("argv", "model_path", "device_env", "args", "env")
    }
    if pd and any(
        arg.split("=", 1)[0].replace("_", "-")
        in {"--kv-transfer-config", "--block-size"}
        for arg in program["args"]
    ):
        raise ValueError("the hosted plan supplies the KV connector and block size")
    hosts = {
        bytes(entry["host"]).hex(): {**program, "address": entry["address"]}
        for entry in recipe["hosts"]
    }
    return validate_spec(
        {
            "api_host": api_host,
            "master_addr": hosts[api_host]["address"],
            "master_port": recipe["master_port"],
            "hosts": hosts,
        },
        allow_single=True,
    )


def validate_spec(spec: dict, *, allow_single: bool = False) -> dict:
    if not isinstance(spec, dict) or set(spec) != {
        "api_host",
        "master_addr",
        "master_port",
        "hosts",
    }:
        raise ValueError("expected api_host, master_addr, master_port and hosts")
    unicast_address(spec["master_addr"])
    if type(spec["master_port"]) is not int or not 0 < spec["master_port"] < 65536:
        raise ValueError("master_port must be a valid TCP port")
    hosts = spec["hosts"]
    if (
        not isinstance(hosts, dict)
        or len(hosts) < (1 if allow_single else 2)
        or not isinstance(spec["api_host"], str)
        or spec["api_host"] not in hosts
    ):
        raise ValueError("invalid launch hosts or missing api_host")
    for host, config in hosts.items():
        if not re.fullmatch("[0-9a-f]{32}", host):
            raise ValueError("host IDs must be lowercase hexadecimal identifiers")
        if (
            not isinstance(config, dict)
            or set(config)
            - {"address", "argv", "model_path", "args", "env", "device_env"}
            or not string_list(config.get("argv"))
            or not config["argv"]
            or not string_list([config.get("model_path")])
            or not string_list(config.get("args", []))
            or not isinstance(config.get("device_env"), str)
            or config["device_env"] not in DEVICE_MASKS
        ):
            raise ValueError(f"invalid vLLM launch configuration for host {host}")
        unicast_address(config.get("address"))
        if any(
            arg.split("=", 1)[0] in PLACEMENT_FLAGS for arg in config.get("args", [])
        ):
            raise ValueError(f"host {host} supplies a flag reserved by the launcher")
        env = config.get("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(value, str)
            or not key
            or "=" in key
            or "\0" in key
            or "\0" in value
            for key, value in env.items()
        ):
            raise ValueError(f"invalid environment for host {host}")
    return spec


def unicast_address(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("host addresses must be IP addresses")
    address = ipaddress.ip_address(value)
    if address.is_unspecified or address.is_multicast:
        raise ValueError("host addresses must be reachable unicast addresses")


def string_list(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and item and "\0" not in item for item in value
    )


def placement(
    spec: dict, grants: list[dict], host: str
) -> tuple[int, list[int], list[dict]]:
    """Assign the API host slot zero and order the other hosts by identity."""
    by_host: dict[str, dict[int, dict]] = {}
    for grant in grants:
        device = grant["device"]
        devices = by_host.setdefault(bytes(device["node"]).hex(), {})
        if device["ordinal"] in devices:
            raise ValueError("duplicate device in the committed group")
        devices[device["ordinal"]] = grant
    if by_host.keys() != spec["hosts"].keys() or host not in by_host:
        raise ValueError("launch hosts differ from the committed device grants")
    api_host = spec["api_host"]
    ordered_hosts = [api_host, *sorted(by_host.keys() - {api_host})]
    local = by_host[host]
    return (
        ordered_hosts.index(host),
        [len(by_host[node]) for node in ordered_hosts],
        [local[device] for device in sorted(local)],
    )


def host_command(
    command: list[str],
    spec: dict,
    host: str,
    node_rank: int,
    layout: list[int],
) -> list[str]:
    config = spec["hosts"][host]
    command = [
        *config["argv"],
        "serve",
        config["model_path"],
        *command[3:],
        *config.get("args", []),
    ]
    if layout == [1]:
        return command
    command.extend(("--distributed-executor-backend", "mp"))
    if len(layout) == 1:
        return command
    return [
        *command,
        "--nnodes",
        str(len(layout)),
        "--node-rank",
        str(node_rank),
        "--master-addr",
        spec["master_addr"],
        "--master-port",
        str(spec["master_port"]),
        "--plait-tp-layout",
        *map(str, layout),
        *(["--headless"] if node_rank else []),
    ]


GREEDY_KV_RESUME = "1"
SEEDED_KV_RESUME = "seeded-v2"


def plait_kv_consumer_options(command: list[str]) -> dict | None:
    config = None
    for index, argument in enumerate(command):
        flag, separator, value = argument.partition("=")
        if flag.replace("_", "-") != "--kv-transfer-config":
            continue
        if separator:
            config = value
        elif index + 1 < len(command):
            config = command[index + 1]
    if config is None:
        return None
    try:
        item = json.loads(config)
    except ValueError:
        return None
    if (
        not isinstance(item, dict)
        or item.get("kv_connector") != "PlaitKvConnector"
        or item.get("kv_role") not in ("kv_consumer", "kv_both")
    ):
        return None
    extra = item.get("kv_connector_extra_config")
    return extra if isinstance(extra, dict) else {}


def kv_resume_capability(options: dict | None, env: dict[str, str]) -> str:
    if options is None or env.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        return ""
    if options.get("seeded_resume") is True:
        return SEEDED_KV_RESUME
    return GREEDY_KV_RESUME
