# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Own a Linux engine session independently of the deployment supervisor."""

import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

if __package__:
    from .plait_engine_cgroup import create_cgroup, verified_cgroup, write_control
else:
    from plait_engine_cgroup import create_cgroup, verified_cgroup, write_control

ENGINE_STOP_GRACE_SECONDS = 10
EXEC_REJECTIONS = (errno.ENOENT, errno.ENOTDIR, errno.EACCES, errno.ENOEXEC)


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


def reap_children(engine: subprocess.Popen) -> bool:
    """Return true only when the kernel reports that every child has exited."""
    try:
        while True:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                return False
            if pid == engine.pid:
                engine.returncode = os.waitstatus_to_exitcode(status)
    except ChildProcessError:
        return True


def stop_engine(engine: subprocess.Popen, children: Path) -> None:
    # Keep the engine leader waitable until TERM so its group ID is still held.
    with suppress(ProcessLookupError):
        os.killpg(engine.pid, signal.SIGTERM)
    deadline = time.monotonic() + ENGINE_STOP_GRACE_SECONDS
    while not reap_children(engine):
        if time.monotonic() >= deadline:
            # Only this process reaps these PIDs. Exiting after the read keeps
            # a child waitable, so its PID cannot be assigned to another owner.
            for child in children.read_text(encoding="utf-8").split():
                with suppress(ProcessLookupError):
                    os.kill(int(child), signal.SIGKILL)
        time.sleep(0.1)


def run(descriptor: Path, record: dict, binding: dict, command: list[str]) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    children.read_text(encoding="utf-8")

    owner = descriptor.with_suffix(".owner.json")
    receipt = {name: record[name] for name in ("host", "version", "grants", "origin")}
    cgroup_root = record.pop("cgroup_root", None)
    try:
        if cgroup_root is not None:
            record["cgroup"] = create_cgroup(cgroup_root, record["origin"])
            write_json(owner, record)
            with verified_cgroup(record) as fd:
                write_control(fd, "cgroup.procs", "0")
        else:
            write_json(owner, record)
    except (OSError, ValueError) as error:
        # Enrollment and journaling both precede the only engine spawn.
        receipt["error"] = str(error)
        write_json(descriptor.with_suffix(".failed.json"), receipt)
        return
    write_json(descriptor, binding)
    try:
        engine = subprocess.Popen(command, start_new_session=True)
    except OSError as error:
        if error.errno not in EXEC_REJECTIONS:
            raise
        receipt["error"] = str(error)
        write_json(descriptor.with_suffix(".failed.json"), receipt)
        return

    stop = descriptor.with_suffix(".stop.json")
    while True:
        try:
            stopping = json.loads(stop.read_text(encoding="utf-8")) == {
                "origin": record["origin"]
            }
        except (FileNotFoundError, json.JSONDecodeError):
            stopping = False
        # WNOWAIT leaves the leader's PID reserved until stop_engine signals it.
        ended = os.waitid(os.P_PID, engine.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        if stopping or ended is not None:
            break
        time.sleep(0.1)

    if not stopping:
        record = {**record, "failed": True}
        write_json(owner, record)
        receipt["unexpected"] = True
    stop_engine(engine, children)
    write_json(descriptor.with_suffix(".stopped.json"), receipt)


if __name__ == "__main__":
    # The descriptor stays open throughout child execution and exit recording.
    path, inherited_lock, metadata, binding, *argv = sys.argv[1:]
    with os.fdopen(int(inherited_lock), "a"):
        run(Path(path), json.loads(metadata), json.loads(binding), argv)
