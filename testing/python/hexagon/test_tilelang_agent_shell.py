"""Run the real transport wrapper with local HLOS stand-ins, never ADB or SSH."""

import contextlib
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import time

import pytest

from tilelang.hexagon.runtime.device import AdbConnection

pytestmark = pytest.mark.skipif(not Path("/proc/self").exists(), reason="the lifecycle checks inspect Linux /proc child ownership")


@pytest.fixture(params=["sh", "bash", "mksh"])
def shell(request):
    executable = shutil.which(request.param)
    if executable is None:
        pytest.skip(f"{request.param} is not installed")
    return executable


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _start(tmp_path, shell, mode):
    directory = tmp_path / "private agent files"
    directory.mkdir()
    agent = directory / "agent"
    agent.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "print(os.getpid(), flush=True)\n"
        "mode = sys.argv[1]\n"
        "if mode.startswith('exit'):\n"
        "    deadline = time.monotonic() + 5\n"
        "    while not Path(sys.argv[0]).with_name('release').exists():\n"
        "        if time.monotonic() >= deadline: sys.exit(99)\n"
        "        time.sleep(0.001)\n"
        "    sys.exit(int(mode[4:]))\n"
        "time.sleep(60)\n"
    )
    conn = AdbConnection(workdir=str(directory))

    def local_shell(command):
        # Isolate process-group signals from the test runner. The command is
        # exactly what production start_host_binary generates.
        return subprocess.Popen(
            [shell, "-c", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    conn.start_shell = local_shell
    remote = str(agent if mode != "missing" else directory / "missing")
    proc = conn.start_host_binary(remote, [mode])
    return proc, directory


def _read_pid(proc):
    assert select.select([proc.stdout], [], [], 5)[0], "agent did not start"
    agent = int(proc.stdout.readline().strip())
    children = Path(f"/proc/{proc.pid}/task/{proc.pid}/children")
    deadline = time.monotonic() + 1
    while True:
        pids = set(map(int, children.read_text().split()))
        if agent in pids and len(pids) == 2:
            return agent, (pids - {agent}).pop()
        assert time.monotonic() < deadline, ("missing agent/watchdog children", pids)
        time.sleep(0.001)


def _finish(proc):
    with contextlib.suppress(ProcessLookupError):
        # Also clean a surviving descendant if a failed test lost its wrapper.
        os.killpg(proc.pid, signal.SIGKILL)
    if proc.poll() is None:
        proc.wait(timeout=5)
    if proc.stdin is not None and not proc.stdin.closed:
        proc.stdin.close()
    proc.stdout.close()


@pytest.mark.parametrize("status", [0, 7])
def test_agent_status_survives_watchdog_cleanup(tmp_path, shell, status):
    proc, directory = _start(tmp_path, shell, f"exit{status}")
    try:
        pid, watch = _read_pid(proc)
        (directory / "release").touch()
        # Keep stdin open: this is normal child exit, not EOF cancellation.
        assert proc.wait(timeout=5) == status
        assert _gone(pid)
        assert _gone(watch), "read watchdog survived or was not reaped"
        assert not directory.exists()
        # All descendants must have closed the output pipe, not just the agent.
        assert select.select([proc.stdout], [], [], 1)[0]
        assert proc.stdout.read() == b""
    finally:
        _finish(proc)


@pytest.mark.parametrize("cancel", ["eof", "term"])
def test_transport_cancellation_stops_agent_and_watchdog(tmp_path, shell, cancel):
    proc, directory = _start(tmp_path, shell, "sleep")
    try:
        pid, watch = _read_pid(proc)
        if cancel == "eof":
            proc.stdin.close()
        else:
            proc.terminate()
        assert proc.wait(timeout=5) == (128 + signal.SIGTERM if cancel == "eof" else 1)
        deadline = time.monotonic() + 2
        while not _gone(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _gone(pid), "agent survived transport cancellation"
        assert _gone(watch), "read watchdog survived or was not reaped"
        assert not directory.exists()
        assert select.select([proc.stdout], [], [], 1)[0]
        # Shells may print a diagnostic for the signal; EOF proves no watcher
        # still holds the inherited output descriptor.
        proc.stdout.read()
    finally:
        _finish(proc)


def test_startup_error_remains_nonzero_and_removes_private_directory(tmp_path, shell):
    proc, directory = _start(tmp_path, shell, "missing")
    try:
        assert proc.wait(timeout=5) != 0
        assert not directory.exists()
        assert select.select([proc.stdout], [], [], 1)[0]
        assert b"missing" in proc.stdout.read()
    finally:
        _finish(proc)
