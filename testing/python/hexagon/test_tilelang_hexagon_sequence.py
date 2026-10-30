"""Resident sequence framing and cache-visible dependencies on a native agent."""

import os
from pathlib import Path
import re
import select
import shutil
import socket
import struct
import subprocess
from types import SimpleNamespace

import pytest

from tilelang.engine.param import KernelParam
from tilelang.hexagon.runtime import fastrpc, sequence, session
from tilelang.hexagon.runtime.session import Copy, Run, SequenceError


def _params():
    return [KernelParam("int32", [32]), KernelParam("int32", [])]


@pytest.fixture(scope="module")
def native_agent(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a native C compiler is required")
    root = tmp_path_factory.mktemp("sequence-agent")
    contracts = [(f"kernel_{i}", "", fastrpc._plan(_params()[:1] if i == 3 else _params(), [])) for i in range(4)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fastrpc, "PD_SETUP", "")
        (root / "agent.c").write_text(session.gen_agent("seq_test", contracts))
    headers = {
        "AEEStdErr.h": "#define AEE_SUCCESS 0\n#define AEE_EBADPARM 14\n#define AEE_ENOMEMORY 2\n",
        "remote.h": "#pragma once\n#include <stdint.h>\ntypedef uint64_t remote_handle64;\n",
        "rpcmem.h": """#include <stdint.h>
#define RPCMEM_HEAP_ID_SYSTEM 0
#define RPCMEM_DEFAULT_FLAGS 0
#define RPCMEM_FLAG_UNCACHED 1
#define FASTRPC_MAP_FD 0
void* rpcmem_alloc(int, int, int);
void rpcmem_free(void*);
int rpcmem_to_fd(void*);
int fastrpc_mmap(int, int, void*, int, uint64_t, int);
int fastrpc_munmap(int, int, void*, uint64_t);
""",
        "dsp_capabilities_utils.h": """#define CDSP_DOMAIN_ID 3
#define MAX_DOMAIN_URI_SIZE 32
typedef struct { const char* uri; } domain;
static domain* get_domain(int id) { static domain d = {"domain"}; (void)id; return &d; }
""",
        "seq_test.h": """#include <stdint.h>
#include "remote.h"
#define seq_test_URI "sequence"
int seq_test_open(const char*, remote_handle64*);
int seq_test_close(remote_handle64);
int seq_test_release(remote_handle64, int32_t);
int seq_test_sync(remote_handle64, int32_t, uint32_t, uint32_t, int32_t);
int seq_test_run(remote_handle64, int32_t, const int32_t*, int, const uint8_t*, int, uint8_t*, int);
""",
    }
    for name, text in headers.items():
        (root / name).write_text(text)
    harness = Path(__file__).with_name("session_sequence_host.c")
    executable = root / "agent"
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if os.environ.get("TL_SEQUENCE_SANITIZERS") == "1" else []
    result = subprocess.run(
        [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O1", *flags, "-I", str(root), str(harness), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.fixture
def agent(native_agent):
    proc = subprocess.Popen([str(native_agent), "0"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # The generated main flushes this line only after its real TCP listener is ready.
    if not select.select([proc.stdout], [], [], 5)[0]:
        proc.kill()
        _, error = proc.communicate()
        pytest.fail(f"native agent startup timed out: {error!r}")
    ready = proc.stdout.readline()
    if not re.fullmatch(rb"AGENT_READY port=\d+\n", ready):
        proc.kill()
        _, error = proc.communicate()
        pytest.fail(f"invalid native agent ready line: {ready!r}, {error!r}")
    connection = socket.create_connection(("127.0.0.1", int(ready.split(b"=")[1])), timeout=5)
    connection.settimeout(5)
    instance = session.Session.__new__(session.Session)
    instance._sock, instance._proc = connection, proc
    instance._closed, instance._holds_device = False, False
    instance._allocations, instance._buffer_generations = {}, {}
    instance._conn = SimpleNamespace(cleanup=lambda: None)
    instance.kernels = [session.SessionKernel(instance, i, f"kernel_{i}", _params()[:1] if i == 3 else _params(), []) for i in range(4)]
    try:
        yield instance
    finally:
        instance.close()
        error = proc.stderr.read().decode()
        proc.stderr.close()
        assert proc.returncode == 0, error
        assert "ERROR:" not in error and "runtime error:" not in error, error


def _ints(buffer):
    return list(struct.unpack(f"<{buffer.nbytes // 4}i", buffer.read()))


def _stats(agent):
    buffer = agent.buffer(128)
    agent.kernels[3](buffer)
    out = _ints(buffer)[:3]
    buffer.free()
    return out


def test_sequence_resident_feedback_cache_lines_and_timings(agent):
    tokens, history = agent.buffer(128), agent.buffer(128)
    tokens.write(struct.pack("<i", 10))
    operations = []
    for i in range(5):
        operations.extend(
            [
                Run(agent.kernels[0], (tokens, i + 1)),
                Copy(tokens, history, 4, src_offset=4, dst_offset=i * 4),
                Copy(tokens, tokens, 4, src_offset=4),
            ]
        )
    result = agent.run_sequence(operations)
    assert result.completed_operations == 15
    assert [run.operation_index for run in result.runs] == [0, 3, 6, 9, 12]
    assert all(run.kernel_index == 0 and run.arm_call_ns >= 0 for run in result.runs)
    assert result.arm_call_ns_sum == sum(run.arm_call_ns for run in result.runs)
    assert result.board_wall_ns >= result.arm_call_ns_sum
    assert _ints(history)[:5] == [11, 13, 16, 20, 25]
    assert _ints(tokens)[0] == 25 and _ints(tokens)[31] == 5
    assert _stats(agent) == [5, 5, 5]


def test_copy_memmove_overlap_and_unaligned_bytes(agent):
    buffer = agent.buffer(256)
    before = bytes(range(256))
    buffer.write(before)
    expected = bytearray(before)
    expected[5:205] = expected[1:201]
    agent.run_sequence([Copy(buffer, buffer, 200, src_offset=1, dst_offset=5)])
    assert buffer.read() == expected
    assert agent.run_sequence([Copy(buffer, buffer, 0, src_offset=256, dst_offset=256)]).completed_operations == 1


@pytest.mark.parametrize("raw", [0x1003, -2147482611])
def test_kernel_failure_preserves_raw_status_and_stops_before_copy(agent, raw):
    source, target = agent.buffer(128), agent.buffer(128)
    with pytest.raises(SequenceError) as failure:
        agent.run_sequence([Run(agent.kernels[0], (source, 7)), Run(agent.kernels[1], (source, raw)), Copy(source, target, 128)])
    error = failure.value
    assert (error.failed_index, error.failure_stage, error.raw_status) == (1, "run", raw & 0xFFFFFFFF)
    assert error.result.completed_operations == 1 and len(error.result.runs) == 2
    assert _ints(source)[1:3] == [7, 1] and _ints(target) == [0] * 32
    assert _stats(agent) == [2, 2, 2]


@pytest.mark.parametrize("sync", [1, 2])
def test_cache_sync_failure_stops_copy_and_next_run(agent, sync):
    source, target = agent.buffer(128), agent.buffer(128)
    source.write(struct.pack("<i", 19))
    with pytest.raises(SequenceError) as failure:
        agent.run_sequence([Run(agent.kernels[2], (source, sync)), Copy(source, target, 4), Run(agent.kernels[0], (source, 1))])
    error = failure.value
    assert (error.failed_index, error.failure_stage, error.raw_status) == (1, "copy", 0x80000073)
    assert error.result.completed_operations == 1 and len(error.result.runs) == 1
    assert _ints(target)[0] == 0 and _stats(agent) == [1, 1, 1]


def test_freed_views_cannot_alias_reallocated_buffer(agent):
    base = agent.buffer(256)
    view = base.at(128)
    old_generation = base._allocation.generation
    base.free()
    replacement = agent.buffer(256)
    assert replacement.id == base.id and replacement._allocation.generation == old_generation + 1
    for stale in (base, view):
        for operation in (
            lambda stale=stale: stale.read(),
            lambda stale=stale: stale.write(b"x"),
            stale.free,
            lambda stale=stale: agent.kernels[0](stale, 1),
            lambda stale=stale: agent.run_sequence([Copy(stale, replacement, 1)]),
        ):
            with pytest.raises(ValueError, match="no longer live"):
                operation()
    assert replacement.read() == bytes(256)


def test_host_prevalidation_rejects_bad_descriptors_without_execution(agent):
    buffer = agent.buffer(256)
    other = session.Session.__new__(session.Session)
    other._closed = True
    foreign = session.DeviceBuffer(other, buffer.id, 128, _allocation=buffer._allocation)
    cases = [
        [],
        [object()],
        [Run(agent.kernels[0], (buffer,))],
        [Run(agent.kernels[0], (buffer, 1 << 31))],
        [Run(agent.kernels[0], (buffer, 1.5))],
        [Run(agent.kernels[0], (None, 1))],
        [Copy(buffer, buffer, -1)],
        [Copy(buffer, buffer, 1, src_offset=1 << 64)],
        [Copy(buffer, foreign, 1)],
        [Copy(buffer, buffer, 257)],
        [Copy(buffer, buffer, 0)] * (sequence.MAX_OPERATIONS + 1),
    ]
    for ops in cases:
        with pytest.raises(ValueError):
            agent.run_sequence(ops)
    with pytest.raises(ValueError, match="128-byte"):
        buffer.at(4)
    assert _stats(agent) == [0, 0, 0]


def test_agent_prevalidation_checks_later_operations_before_any_execution(agent):
    buffer = agent.buffer(128)
    operations, wire = sequence.encode(agent, [Run(agent.kernels[0], (buffer, 1)), Copy(buffer, buffer, 4)])
    # Damage the second Copy's source offset after Python validation.
    bad = bytearray(wire)
    copy = 9 + 12 + 2 * 24
    struct.pack_into("<Q", bad, copy + 4 + 8 + 16, 1 << 63)
    agent._sock.sendall(bad)
    with pytest.raises(SequenceError) as failure:
        sequence.receive(agent, operations)
    assert failure.value.failure_stage == "validation" and failure.value.failed_index == 1
    assert failure.value.result.completed_operations == 0 and not failure.value.result.runs
    assert _ints(buffer) == [0] * 32 and _stats(agent) == [0, 0, 0]


def test_agent_rejects_stale_generation_even_if_buffer_id_is_live(agent):
    buffer = agent.buffer(128)
    operations, wire = sequence.encode(agent, [Run(agent.kernels[0], (buffer, 1))])
    buffer.free()
    replacement = agent.buffer(128)
    assert replacement.id == buffer.id
    agent._sock.sendall(wire)
    with pytest.raises(SequenceError) as failure:
        sequence.receive(agent, operations)
    assert failure.value.failure_stage == "validation" and failure.value.result.completed_operations == 0
    assert _ints(replacement) == [0] * 32


def test_default_run_and_timing_protocol_remain_usable(agent):
    buffer = agent.buffer(128)
    agent.kernels[0](buffer, 9)
    assert _ints(buffer)[1] == 9
    assert agent.last_run_seconds() >= 0
    agent.run_sequence([Run(agent.kernels[0], (buffer, 11))])
    agent.kernels[0](buffer, 13)
    assert _ints(buffer)[1] == 13 and _stats(agent) == [3, 3, 3]


@pytest.mark.parametrize("damage", ["kernel", "arity", "kind", "scalar_offset", "alignment", "size", "truncated", "trailing", "count"])
def test_malformed_wire_contracts_execute_nothing_and_recover(agent, damage):
    buffer = agent.buffer(128)
    operations, wire = sequence.encode(agent, [Run(agent.kernels[0], (buffer, 1))])
    bad = bytearray(wire)
    if damage == "kernel":
        struct.pack_into("<I", bad, 13, 4)
    elif damage == "arity":
        struct.pack_into("<I", bad, 17, 0xFFFFFFFF)
    elif damage == "kind":
        struct.pack_into("<I", bad, 21, 1)  # Inline inputs are not a sequence argument.
    elif damage == "scalar_offset":
        struct.pack_into("<Q", bad, 53, 128)
    elif damage == "alignment":
        struct.pack_into("<Q", bad, 29, 4)
    elif damage == "size":
        struct.pack_into("<Q", bad, 29, 128)  # Aligned, but the kernel's 128 bytes do not fit.
    elif damage == "truncated":
        bad = bad[:-1]
        struct.pack_into("<I", bad, 1, len(bad) - 9)
    elif damage == "trailing":
        bad += b"x"
        struct.pack_into("<I", bad, 1, len(bad) - 9)
    else:
        struct.pack_into("<I", bad, 5, sequence.MAX_OPERATIONS + 1)
    agent._sock.sendall(bad)
    with pytest.raises(SequenceError) as failure:
        sequence.receive(agent, operations)
    assert failure.value.failure_stage == "validation"
    assert failure.value.result.completed_operations == 0
    assert _stats(agent) == [0, 0, 0]
    assert agent.run_sequence(operations).completed_operations == 1


def test_maximum_operation_count_and_long_recurrence(agent):
    buffer = agent.buffer(128)
    result = agent.run_sequence([Copy(buffer, buffer, 0)] * sequence.MAX_OPERATIONS)
    assert result.completed_operations == sequence.MAX_OPERATIONS and not result.runs
    ops = [
        op
        for _ in range(519)
        for op in (Run(agent.kernels[0], (buffer, 1)), Copy(buffer, buffer, 4, src_offset=4), Copy(buffer, buffer, 0))
    ]
    result = agent.run_sequence(ops)
    assert result.completed_operations == 1557 and len(result.runs) == 519
    assert _ints(buffer)[0] == 519 and _stats(agent) == [519, 519, 519]


def test_view_free_and_session_close_invalidate_all_views(agent):
    buffer = agent.buffer(256)
    view = buffer.at(128)
    view.free()
    with pytest.raises(ValueError, match="no longer live"):
        buffer.read()
    live = agent.buffer(128)
    agent.close()
    with pytest.raises(ValueError, match="no longer live"):
        live.read()
    with pytest.raises(ValueError, match="closed"):
        agent.buffer(128)


def test_lost_allocation_reply_closes_session_and_invalidates_views(agent, monkeypatch):
    live = agent.buffer(128)

    def lost(_):
        raise ConnectionError("allocation reply lost")

    monkeypatch.setattr(agent, "_recv", lost)
    with pytest.raises(ConnectionError, match="reply lost"):
        agent.buffer(128)
    assert agent._closed and not live._allocation.alive
    assert agent._buffer_generations[live.id] == 1


@pytest.mark.parametrize(
    "response",
    [
        struct.pack("<IIIIIq", 0, 0, 2, 0x8000040D, 0, 10),  # Failed Run timing missing.
        struct.pack("<IIIIIq", 0, 0, 3, 1, 0, 10),  # Run reported as Copy failure.
        struct.pack("<IIIIIq", 1, 0, 2, 1, 0, 10),  # Completed and failed index disagree.
        struct.pack("<IIIIIq", 0, 0xFFFFFFFF, 0, 0, 0, 10),  # Incomplete success.
        struct.pack("<IIIIIq", 0, 0xFFFFFFFF, 1, 0, 0, 0),  # Failure without status.
    ],
)
def test_malformed_response_closes_session_without_replaying(agent, monkeypatch, response):
    buffer = agent.buffer(128)
    monkeypatch.setattr(agent, "_recv", lambda n: response)
    with pytest.raises(RuntimeError, match="Hexagon sequence"):
        agent.run_sequence([Run(agent.kernels[0], (buffer, 1))])
    assert agent._closed and not buffer._allocation.alive


def test_payload_limit_and_forged_kernel_identity(agent, monkeypatch):
    buffer = agent.buffer(128)
    forged = session.SessionKernel(agent, 123456, "forged", _params(), [])
    with pytest.raises(ValueError, match="kernel must belong"):
        agent.run_sequence([Run(forged, (buffer, 1))])
    monkeypatch.setattr(sequence, "MAX_BYTES", 59)
    with pytest.raises(ValueError, match="payload may not exceed"):
        agent.run_sequence([Run(agent.kernels[0], (buffer, 1))])
    assert _stats(agent) == [0, 0, 0]


def test_invalid_allocation_size_does_not_close_session(agent):
    buffer = agent.buffer(128)
    for size in (1.5, "128", -1, 0, 1 << 31):
        with pytest.raises(ValueError):
            agent.buffer(size)
    assert buffer.read() == bytes(128) and not agent._closed


@pytest.mark.parametrize("size", [0, 1 << 31, 1 << 32, (1 << 64) - 1])
def test_agent_rejects_oversized_alloc_before_narrowing_and_preserves_generations(agent, size):
    agent._sock.sendall(bytes([session._OP_ALLOC]) + struct.pack("<Q", size))
    assert struct.unpack("<i", agent._recv(4)) == (-1,)
    buffer = agent.buffer(128)
    assert buffer.id == 0 and buffer._allocation.generation == 1
    assert agent.run_sequence([Run(agent.kernels[0], (buffer, 7))]).completed_operations == 1
    assert _ints(buffer)[1] == 7


def test_failed_free_closes_session_and_invalidates_views(agent, monkeypatch):
    buffer = agent.buffer(128)
    sock = agent._sock

    class BrokenFree:
        def sendall(self, data):
            if data[0] == session._OP_FREE:
                raise ConnectionError("FREE send failed")
            return sock.sendall(data)

        def close(self):
            sock.close()

    monkeypatch.setattr(agent, "_sock", BrokenFree())
    with pytest.raises(ConnectionError, match="FREE send failed"):
        buffer.free()
    assert agent._closed and not buffer._allocation.alive


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_resident_sequence_on_device():
    source = """#include <stdint.h>
int32_t feedback(int32_t* values, int32_t scalar) {
  values[1] = values[0] + scalar;
  values[31] += 1;
  return 0;
}
"""
    failure = """#include <stdint.h>
int32_t fail_after_write(int32_t* values, int32_t status) {
  values[2] += 1;
  return status;
}
"""
    with session.Session.of_lowered([("feedback", source, _params(), []), ("fail_after_write", failure, _params(), [])]) as device:
        tokens, history = device.buffer(128), device.buffer(128)
        tokens.write(struct.pack("<i", 10))
        operations = []
        for i in range(8):
            operations.extend(
                [
                    Run(device.kernels[0], (tokens, i + 1)),
                    Copy(tokens, history, 4, src_offset=4, dst_offset=i * 4),
                    Copy(tokens, tokens, 4, src_offset=4),
                ]
            )
        result = device.run_sequence(operations)
        assert result.completed_operations == 24 and len(result.runs) == 8
        assert result.board_wall_ns >= result.arm_call_ns_sum > 0
        assert _ints(history)[:8] == [11, 13, 16, 20, 25, 31, 38, 46]
        assert _ints(tokens)[0] == 46 and _ints(tokens)[31] == 8
        before = history.read()
        with pytest.raises(SequenceError) as error:
            # The DSP dispatcher adds KERNEL_STATUS_BASE to the kernel's TL_ERR_*.
            device.run_sequence([Run(device.kernels[1], (tokens, 3)), Copy(tokens, history, 128)])
        assert (error.value.failed_index, error.value.raw_status, error.value.result.completed_operations) == (0, 0x1003, 0)
        assert len(error.value.result.runs) == 1 and history.read() == before
        assert _ints(tokens)[2] == 1
        # Reuse the same synchronized connection after a kernel failure.
        assert device.run_sequence([Copy(tokens, history, 4)]).completed_operations == 1
        assert _ints(history)[0] == 46
