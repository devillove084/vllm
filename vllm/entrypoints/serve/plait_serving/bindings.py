# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive versioned HTTP origins from launch configurations and group grants."""

import ipaddress
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from vllm.entrypoints.serve.plait_deploy import Deployment

from .group import Control, ControlError
from .mode import active_epochs


@dataclass
class Bindings:
    model: str
    deployments: dict[str, Deployment]
    aggregate: tuple[str, ...]
    paths: tuple[dict, ...]

    @classmethod
    def read(cls, model: str, directory: Path, document: dict) -> "Bindings":
        sources = document.get("deployments", {})
        deploy = {
            group: Deployment.read(directory / source)
            for group, source in sources.items()
        }
        if any(
            group != entry.group_id or entry.model_alias != model
            for group, entry in deploy.items()
        ):
            raise ValueError("deployment group or model differs from native bindings")
        aggregate = tuple(document.get("aggregated", []))
        paths = tuple(document.get("paths", []))
        roles = {group: "aggregated" for group in aggregate}
        for path in paths:
            ipaddress.IPv4Address(path["kv_host"])
            port = path["kv_port"]
            if type(port) is not int or not 0 < port < 65536:
                raise ValueError("invalid KV port in native bindings")
            for group, role in (
                (path["prefill"], "prefill"),
                (path["decode"], "decode"),
            ):
                if roles.setdefault(group, role) != role:
                    raise ValueError(
                        "a discovered group can have only one serving role"
                    )
        if roles.keys() != deploy.keys() or len(set(aggregate)) != len(aggregate):
            raise ValueError(
                "native bindings require one deployment per referenced group"
            )
        return cls(model, deploy, aggregate, paths)

    def resolve(self, control: Control, active: frozenset[str]) -> dict:
        epochs = active_epochs(active)
        resolved = {
            group: self.origins(control, deployment, epochs.get(group))
            for group, deployment in self.deployments.items()
        }
        return {
            "model": self.model,
            "aggregated": [
                origin for group in self.aggregate for origin in resolved[group]
            ],
            "paths": [
                {
                    "prefill": {
                        **p,
                        "kv_host": path["kv_host"],
                        "kv_port": path["kv_port"],
                    },
                    "decode": d,
                }
                for path in self.paths
                for p, d in product(resolved[path["prefill"]], resolved[path["decode"]])
            ],
        }

    @staticmethod
    def origins(
        control: Control, deployment: Deployment, active: int | None
    ) -> list[dict]:
        group = list(bytes.fromhex(deployment.group_id))
        latest = control.request({"request": "placement", "group": group})["placement"][
            "latest"
        ]
        epochs = set()
        if active is not None:
            epochs.add(active)
        if latest is not None and latest[1] == "Ready":
            epochs.add(latest[0]["epoch"])
        origins = []
        for epoch in sorted(epochs):
            version = {"group": group, "epoch": epoch}
            view = control.request({"request": "view", "version": version})["group"]
            if view is None or (epoch != active and view["phase"] != "Ready"):
                continue
            if view["version"] != version:
                raise ControlError("native binding view returned a different version")
            if deployment.layout_recipes:
                recipe = control.request({"request": "recipe", "version": version})[
                    "recipe"
                ]
                if recipe is None:
                    continue
                if recipe["version"] != version:
                    raise ControlError(
                        "native binding recipe returned a different version"
                    )
                view = {**view, "recipe": recipe["recipe"]}
            placement = deployment.place(view)
            origins.append({"version": placement.version_text, "url": placement.origin})
        return origins
