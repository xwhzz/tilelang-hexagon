"""Docker SDK preparation and launcher configuration, without Docker or a board."""

import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker/hexagon"


def test_image_copies_all_native_cython_inputs_before_building():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    inputs = set(re.findall(r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/([^\"\s]+\.pyx)", cmake))
    before_build = (DOCKER / "Dockerfile").read_text().split("cmake -S", 1)[0]
    sources = []
    for line in before_build.splitlines():
        if line.startswith("COPY "):
            sources.extend(Path(source) for source in shlex.split(line)[1:-1])
    assert inputs
    for source in inputs:
        assert any(Path(source).is_relative_to(copied) for copied in sources), f"Missing native build input: {source}"


@pytest.mark.parametrize("preinstalled", [False, True])
def test_prepare_sdk_installs_qaic_for_unprivileged_runtime(tmp_path, preinstalled):
    spec = importlib.util.spec_from_file_location("prepare_sdk", DOCKER / "prepare-sdk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qaic_dir = tmp_path / "ipc/fastrpc/qaic"
    location = qaic_dir / ("bin" if preinstalled else "Ubuntu")
    location.mkdir(parents=True)
    (location / "qaic").write_text("#!/bin/sh\nprintf 'qaic ready\\n'\n")
    (location / "qaic").chmod(0o750)
    if not preinstalled:
        (qaic_dir / "Makefile").write_text("all:\n\tmkdir -p bin\n\tcp -p Ubuntu/qaic bin/qaic\n")
    helpers = tmp_path / "utils/examples"
    helpers.mkdir(parents=True)
    for suffix in ("c", "h"):
        (helpers / f"dsp_capabilities_utils.{suffix}").touch()
    setup = tmp_path / "setup_sdk_env.source"
    setup.write_text("make -C $HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/ clean\nmake -C $HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/\n")
    module.prepare_sdk(tmp_path)
    qaic = qaic_dir / "bin/qaic"
    assert qaic.stat().st_mode & 0o777 == 0o755
    assert subprocess.check_output([str(qaic)], text=True) == "qaic ready\n"
    assert "make -C" not in setup.read_text()
    module.prepare_sdk(tmp_path)  # Repeated preparation keeps the installed tool.
    (helpers / "dsp_capabilities_utils.c").unlink()
    with pytest.raises(RuntimeError, match="supply the full SDK"):
        module.prepare_sdk(tmp_path)


@pytest.fixture
def launcher_environment(tmp_path):
    # Execute the real launcher and capture Docker argv instead of starting it.
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "with open(os.environ['DOCKER_CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(1 if sys.argv[1:3] == ['container', 'inspect'] else 0)\n"
    )
    fake_docker.chmod(0o755)
    fake_id = tmp_path / "id"
    fake_id.write_text('#!/bin/sh\nprintf "%s\\n" "$LAUNCHER_UID"\n')
    fake_id.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    environ = {k: v for k, v in os.environ.items() if not k.startswith("TILELANG_HEXAGON_") and k not in {"IMAGE", "SSH_AUTH_SOCK"}}
    environ.update(PATH=str(tmp_path) + os.pathsep + environ["PATH"], DOCKER_CALLS=str(calls), LAUNCHER_UID=str(os.getuid()))
    return environ, calls


@pytest.mark.parametrize(
    "hlos,remote,expected",
    [
        ("linux", None, "/tmp/tilelang_sample_hexagon"),
        ("linux", "/custom/path with spaces", "/custom/path with spaces"),
        ("android", None, "/data/local/tmp/tilelang_sample_hexagon"),
    ],
)
def test_launcher_preserves_remote_override_and_image_name(tmp_path, launcher_environment, hlos, remote, expected):
    environ, calls = launcher_environment
    environ["TILELANG_HEXAGON_HLOS"] = hlos
    if remote:
        environ["TILELANG_HEXAGON_REMOTE_DIR"] = remote
    subprocess.run(["bash", str(DOCKER / "enter-docker.sh"), "sample"], cwd=tmp_path, env=environ, check=True)
    run = next(call for call in map(json.loads, calls.read_text().splitlines()) if call[0] == "run")
    assert f"TILELANG_HEXAGON_REMOTE_DIR={expected}" in run
    assert "tilelang-hexagon:sa8650-v73" in run
    assert "TMPDIR=/tmp" in run
    assert "TILELANG_HEXAGON_DOMAIN=3" in run


@pytest.mark.parametrize("hlos,uid", [("linux", 0), ("linux", 1001), ("android", 1001)])
def test_launcher_mounts_ssh_credentials_only_for_linux(tmp_path, launcher_environment, hlos, uid):
    environ, calls = launcher_environment
    ssh_dir = tmp_path / "ssh files"
    ssh_dir.mkdir()
    known_hosts = ssh_dir / "known_hosts"
    known_hosts.touch()
    agent_socket = tmp_path / "agent.sock"
    environ.update(
        TILELANG_HEXAGON_HLOS=hlos,
        TILELANG_HEXAGON_SSH_HOST="root@board",
        TILELANG_HEXAGON_SSH_DIR=str(ssh_dir),
        TILELANG_HEXAGON_SSH_KNOWN_HOSTS=str(known_hosts),
        TILELANG_HEXAGON_DOMAIN="4",
        TILELANG_HEXAGON_DSP_LIBRARY_PATH="/vendor/dsp/cdsp1;/usr/share/fastrpc",
        SSH_AUTH_SOCK=str(agent_socket),
        LAUNCHER_UID=str(uid),
    )
    with socket.socket(socket.AF_UNIX) as agent:
        agent.bind(str(agent_socket))
        subprocess.run(["bash", str(DOCKER / "enter-docker.sh"), "sample"], cwd=tmp_path, env=environ, check=True)
    commands = list(map(json.loads, calls.read_text().splitlines()))
    run = next(command for command in commands if command[0] == "run")
    assert "TILELANG_HEXAGON_DOMAIN=4" in run
    assert "TILELANG_HEXAGON_DSP_LIBRARY_PATH" in run
    home = "/root" if uid == 0 else f"/tmp/tilelang-home-{uid}"
    assert (f"{ssh_dir}:{home}/.ssh:ro" in run) == (hlos == "linux")
    assert (f"{known_hosts}:/run/tilelang-known-hosts:ro" in run) == (hlos == "linux")
    assert ("TILELANG_HEXAGON_SSH_KNOWN_HOSTS=/run/tilelang-known-hosts" in run) == (hlos == "linux")
    assert (f"{agent_socket}:/run/tilelang-ssh-agent" in run) == (hlos == "linux")
    assert ("SSH_AUTH_SOCK=/run/tilelang-ssh-agent" in run) == (hlos == "linux")
    assert commands[-1] == ["exec", "-it", "sample_hexagon", "/usr/local/bin/tilelang-entrypoint", "bash"]


def test_launcher_allows_ssh_setup_after_entering(tmp_path, launcher_environment):
    environ, calls = launcher_environment
    environ.update(
        TILELANG_HEXAGON_HLOS="linux",
        TILELANG_HEXAGON_SSH_HOST="root@board",
        TILELANG_HEXAGON_SSH_DIR=str(tmp_path / "missing"),
    )
    subprocess.run(["bash", str(DOCKER / "enter-docker.sh"), "sample"], cwd=tmp_path, env=environ, check=True)
    commands = list(map(json.loads, calls.read_text().splitlines()))
    assert commands[-1] == ["exec", "-it", "sample_hexagon", "/usr/local/bin/tilelang-entrypoint", "bash"]


def test_launcher_rejects_missing_explicit_known_hosts(tmp_path, launcher_environment):
    environ, calls = launcher_environment
    environ.update(
        TILELANG_HEXAGON_HLOS="linux",
        TILELANG_HEXAGON_SSH_HOST="root@board",
        TILELANG_HEXAGON_SSH_KNOWN_HOSTS=str(tmp_path / "missing"),
    )
    result = subprocess.run(["bash", str(DOCKER / "enter-docker.sh"), "sample"], cwd=tmp_path, env=environ, text=True, capture_output=True)
    assert result.returncode == 1
    assert "SSH known_hosts file missing" in result.stderr
    assert all(command[0] != "run" for command in map(json.loads, calls.read_text().splitlines()))
