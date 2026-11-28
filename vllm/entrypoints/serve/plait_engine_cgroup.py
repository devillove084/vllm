# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Create and verify the kernel process boundary of a Linux engine."""

import os
from contextlib import contextmanager, suppress
from pathlib import Path
from uuid import UUID


def boot_id() -> str:
    return UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()).hex


def cgroup_path(root: str, origin: str) -> Path:
    if not Path(root).is_absolute() or UUID(origin).hex != origin:
        raise ValueError("invalid engine cgroup location")
    return Path(root) / f"plait-{origin}"


def check_cgroup2(fd: int) -> None:
    fields = Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
    mount = next(
        (line.split()[1] for line in fields if line.startswith("mnt_id:")), None
    )
    with Path("/proc/self/mountinfo").open() as mounts:
        for entry in mounts:
            info, kind = entry.split(" - ", 1)
            if info.split()[0] == mount and kind.split()[0] == "cgroup2":
                return
    raise ValueError("engine containment requires a cgroup v2 filesystem")


@contextmanager
def open_cgroup(path: Path):
    fd = os.open(path, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        check_cgroup2(fd)
        yield fd
    finally:
        os.close(fd)


def write_control(fd: int, name: str, value: str) -> None:
    with os.fdopen(os.open(name, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=fd), "w") as file:
        file.write(value)


def populated(fd: int) -> bool:
    events = os.open("cgroup.events", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    with os.fdopen(events, encoding="ascii") as file:
        values = [line.split()[1:] for line in file if line.startswith("populated ")]
    if values not in ([["0"]], [["1"]]):
        raise ValueError("invalid cgroup population report")
    return values == [["1"]]


def create_cgroup(root: str, origin: str) -> dict:
    parent = Path(root).resolve(strict=True)
    with open_cgroup(parent):
        path = cgroup_path(str(parent), origin)
        path.mkdir()
        try:
            with open_cgroup(path) as fd:
                # Opening the control verifies that this leaf supports atomic kill.
                os.close(os.open("cgroup.kill", os.O_WRONLY, dir_fd=fd))
                identity = os.fstat(fd)
                return {
                    "root": str(parent),
                    "boot": boot_id(),
                    "device": identity.st_dev,
                    "inode": identity.st_ino,
                }
        except (OSError, ValueError):
            with suppress(OSError):
                path.rmdir()
            raise


@contextmanager
def verified_cgroup(record: dict):
    try:
        binding = record["cgroup"]
        path = cgroup_path(binding["root"], record["origin"])
        expected = (binding["device"], binding["inode"])
        expected_boot = binding["boot"]
        if UUID(expected_boot).hex != expected_boot:
            raise ValueError("invalid engine boot identity")
        if any(type(value) is not int or value < 0 for value in expected):
            raise ValueError("invalid engine cgroup identity")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid engine cgroup record") from error
    if expected_boot != boot_id():
        raise ValueError("engine cgroup belongs to a different Linux boot")
    with open_cgroup(path) as fd:
        identity = os.fstat(fd)
        if (identity.st_dev, identity.st_ino) != expected:
            raise ValueError("engine cgroup identity changed")
        yield fd


def known_populated(record: dict) -> bool:
    # A terminal journal can outlive a cgroup already removed after receipts.
    with suppress(FileNotFoundError), verified_cgroup(record) as fd:
        return populated(fd)
    return False


def remove_cgroup(record: dict) -> None:
    with suppress(FileNotFoundError), verified_cgroup(record) as fd:
        if populated(fd):
            raise OSError("engine cgroup is still populated")
        cgroup_path(record["cgroup"]["root"], record["origin"]).rmdir()
