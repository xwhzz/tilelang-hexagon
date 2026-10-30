"""Explicit resident Run/Copy commands, encoded once for the HLOS agent."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import operator
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import DeviceBuffer, SessionKernel

MAX_OPERATIONS = 4096
MAX_BYTES = 1 << 20
OP_SEQUENCE = 9


@dataclass(frozen=True)
class Run:
    """One existing kernel invocation, with all arguments resident or int32."""

    kernel: SessionKernel
    args: tuple

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))


@dataclass(frozen=True)
class Copy:
    """An overlapping-safe byte copy; offsets are relative to each buffer view."""

    src: DeviceBuffer
    dst: DeviceBuffer
    nbytes: int
    src_offset: int = 0
    dst_offset: int = 0


@dataclass(frozen=True)
class SequenceTiming:
    operation_index: int
    kernel_index: int
    arm_call_ns: int


@dataclass(frozen=True)
class SequenceResult:
    completed_operations: int
    runs: tuple[SequenceTiming, ...]
    board_wall_ns: int

    @property
    def arm_call_ns_sum(self) -> int:
        return sum(run.arm_call_ns for run in self.runs)


class SequenceError(RuntimeError):
    """Stopped sequence; even the failed operation may have partial side effects.

    Effects are neither rolled back nor retried.
    """

    def __init__(self, failed_index, raw_status, failure_stage, result):
        from .fastrpc import KERNEL_STATUS, KERNEL_STATUS_BASE

        self.failed_index = failed_index
        self.raw_status = raw_status
        self.failure_stage = failure_stage
        self.result = result
        reason = KERNEL_STATUS.get(raw_status - KERNEL_STATUS_BASE, "FastRPC or agent error")
        super().__init__(f"Hexagon sequence {failure_stage} failed at operation {failed_index}: {reason} (0x{raw_status:08x})")


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None


def encode(session, operations):
    from .session import DeviceBuffer, SessionKernel

    operations = tuple(islice(operations, MAX_OPERATIONS + 1))
    if not 1 <= len(operations) <= MAX_OPERATIONS:
        raise ValueError(f"a sequence needs 1 to {MAX_OPERATIONS} operations")
    payload = bytearray()

    def buffer(value):
        if not isinstance(value, DeviceBuffer) or value.session is not session:
            raise ValueError("a sequence buffer must belong to this session")
        return value._live_allocation()

    for op in operations:
        if isinstance(op, Run):
            kernel = op.kernel
            if (
                not isinstance(kernel, SessionKernel)
                or kernel.session is not session
                or not 0 <= kernel.index < len(session.kernels)
                or session.kernels[kernel.index] is not kernel
            ):
                raise ValueError("a sequence kernel must belong to this session")
            if len(op.args) != len(kernel.plans):
                raise ValueError(f"{kernel.name} needs all {len(kernel.plans)} arguments")
            payload += struct.pack("<III", 0, kernel.index, len(op.args))
            for plan, value in zip(kernel.plans, op.args):
                if plan.is_scalar:
                    scalar = _integer(value, "a Run scalar")
                    if plan.ctype != "int32_t" or not -(1 << 31) <= scalar < 1 << 31:
                        raise ValueError("Run supports int32 scalar parameters only")
                    payload += struct.pack("<IIQQ", 3, scalar & 0xFFFFFFFF, 0, 0)
                else:
                    allocation = buffer(value)
                    if value.offset % 128 or value.nbytes < plan.nbytes:
                        raise ValueError(f"{kernel.name} requires a 128-byte aligned buffer of at least {plan.nbytes} bytes")
                    payload += struct.pack("<IIQQ", 0, value.id, value.offset, allocation.generation)
        elif isinstance(op, Copy):
            src, dst = buffer(op.src), buffer(op.dst)
            size = _integer(op.nbytes, "copy size")
            source = _integer(op.src_offset, "copy source offset")
            target = _integer(op.dst_offset, "copy destination offset")
            if size < 0 or source < 0 or target < 0 or source > op.src.nbytes or target > op.dst.nbytes:
                raise ValueError("copy range is outside its buffer view")
            if size > op.src.nbytes - source or size > op.dst.nbytes - target:
                raise ValueError("copy range is outside its buffer view")
            payload += struct.pack(
                "<IIIQQQQQ", 1, op.src.id, op.dst.id, src.generation, dst.generation, op.src.offset + source, op.dst.offset + target, size
            )
        else:
            raise ValueError("a sequence accepts only Run and Copy descriptors")
        if len(payload) > MAX_BYTES:
            raise ValueError(f"a sequence payload may not exceed {MAX_BYTES} bytes")
    return operations, bytes([OP_SEQUENCE]) + struct.pack("<II", len(payload), len(operations)) + payload


def receive(session, operations):
    completed, failed, stage, status, count, wall = struct.unpack("<IIIIIq", session._recv(28))
    if completed > len(operations) or count > len(operations) or stage > 3 or wall < 0:
        raise RuntimeError("invalid Hexagon sequence response")
    runs = []
    for _ in range(count):
        index, kernel, ns = struct.unpack("<IIq", session._recv(16))
        if index >= len(operations) or not isinstance(operations[index], Run) or operations[index].kernel.index != kernel or ns < 0:
            raise RuntimeError("invalid Hexagon sequence timing")
        if runs and index <= runs[-1].operation_index:
            raise RuntimeError("unordered Hexagon sequence timings")
        runs.append(SequenceTiming(index, kernel, ns))
    result = SequenceResult(completed, tuple(runs), wall)
    if stage:
        if not status or (failed != 0xFFFFFFFF and failed >= len(operations)) or (stage == 1 and (completed or runs)):
            raise RuntimeError("invalid Hexagon sequence failure")
        if stage in (2, 3):
            if failed != completed or failed >= len(operations) or not isinstance(operations[failed], Run if stage == 2 else Copy):
                raise RuntimeError("invalid Hexagon sequence failure position")
            executed = operations[: completed + (stage == 2)]
            if [run.operation_index for run in runs] != [i for i, op in enumerate(executed) if isinstance(op, Run)]:
                raise RuntimeError("incomplete Hexagon sequence failure timings")
        raise SequenceError(None if failed == 0xFFFFFFFF else failed, status, ("", "validation", "run", "copy")[stage], result)
    if status or failed != 0xFFFFFFFF or completed != len(operations) or len(runs) != sum(isinstance(op, Run) for op in operations):
        raise RuntimeError("incomplete Hexagon sequence response")
    return result


def agent_source(iface, kernels):
    """Kernel contracts used by the agent to validate an entire sequence."""
    offsets, params, cursor = [], [], 0
    for _, _, plans in kernels:
        offsets.append(f"{{{cursor}, {len(plans)}}}")
        params.extend(f"{{{int(plan.is_scalar)}, {plan.nbytes}u}}" for plan in plans)
        cursor += len(plans)
    metadata = (
        f"#define TL_SEQ_MAX_OPS {MAX_OPERATIONS}u\n#define TL_SEQ_MAX_BYTES {MAX_BYTES}u\n"
        f"#define TL_SEQ_KERNELS {len(kernels)}u\n"
        "static const struct { uint32_t first, count; } tl_seq_kernels[] = {" + ",".join(offsets or ["{0,0}"]) + "};\n"
        "static const struct { uint32_t scalar; uint64_t bytes; } tl_seq_params[] = {" + ",".join(params or ["{0,0}"]) + "};\n"
    )
    return metadata + _AGENT_SOURCE.replace("TL_IFACE", iface)


_AGENT_SOURCE = r"""
struct tl_seq_op {
  uint32_t kind, kernel, nargs, src, dst;
  int32_t* args;
  uint64_t source, target, size;
};
struct tl_seq_time { uint32_t index, kernel; int64_t ns; };

static int64_t tl_seq_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}
static int tl_seq_u32(const uint8_t** p, const uint8_t* end, uint32_t* value) {
  if ((size_t)(end - *p) < 4) return -1;
  memcpy(value, *p, 4); *p += 4; return 0;
}
static int tl_seq_u64(const uint8_t** p, const uint8_t* end, uint64_t* value) {
  if ((size_t)(end - *p) < 8) return -1;
  memcpy(value, *p, 8); *p += 8; return 0;
}
static int tl_seq_range(uint32_t id, uint64_t generation, uint64_t offset, uint64_t size) {
  return id < MAX_BUFFERS && g_buffers[id].ptr && g_buffers[id].generation == generation &&
         offset <= g_buffers[id].size && size <= g_buffers[id].size - offset;
}
static int tl_seq_send(int cli, uint32_t completed, uint32_t failed, uint32_t stage,
                       uint32_t status, struct tl_seq_time* runs, uint32_t count, int64_t wall) {
  // One bounded write avoids a delayed ACK between the header and timings.
  size_t size = 28 + (size_t)count * 16;
  uint8_t* reply = (uint8_t*)malloc(size);
  if (!reply) return 1;
  uint32_t head[5] = {completed, failed, stage, status, count};
  memcpy(reply, head, sizeof(head)); memcpy(reply + 20, &wall, 8);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t entry[2] = {runs[i].index, runs[i].kernel};
    memcpy(reply + 28 + (size_t)i * 16, entry, 8);
    memcpy(reply + 36 + (size_t)i * 16, &runs[i].ns, 8);
  }
  int result = write_full(cli, reply, size);
  free(reply);
  return result;
}
static int tl_serve_sequence(int cli) {
  uint32_t head[2];
  if (read_full(cli, head, sizeof(head))) return 1;
  uint32_t bytes = head[0], count = head[1];
  if (bytes > TL_SEQ_MAX_BYTES) {
    tl_seq_send(cli, 0, UINT32_MAX, 1, AEE_EBADPARM, 0, 0, 0);
    return 1; // Do not try to drain an unbounded malformed frame.
  }
  uint8_t* payload = (uint8_t*)malloc(bytes ? bytes : 1);
  if (!payload) return 1;
  if (read_full(cli, payload, bytes)) { free(payload); return 1; }
  if (!count || count > TL_SEQ_MAX_OPS) {
    free(payload);
    return tl_seq_send(cli, 0, UINT32_MAX, 1, AEE_EBADPARM, 0, 0, 0);
  }
  struct tl_seq_op* ops = (struct tl_seq_op*)calloc(count, sizeof(*ops));
  struct tl_seq_time* times = (struct tl_seq_time*)calloc(count, sizeof(*times));
  uint32_t completed = 0, failed = UINT32_MAX, stage = 0, status = 0, nruns = 0;
  int64_t wall = 0;
  if (!ops || !times) { stage = 1; status = AEE_ENOMEMORY; goto done; }
  const uint8_t* p = payload;
  const uint8_t* end = payload + bytes;
  for (uint32_t i = 0; i < count; ++i) {
    struct tl_seq_op* op = &ops[i];
    failed = i; stage = 1; status = AEE_EBADPARM;
    if (tl_seq_u32(&p, end, &op->kind)) goto done;
    if (op->kind == 0) {
      if (tl_seq_u32(&p, end, &op->kernel) || tl_seq_u32(&p, end, &op->nargs) || op->kernel >= TL_SEQ_KERNELS ||
          op->nargs != tl_seq_kernels[op->kernel].count) goto done;
      op->args = (int32_t*)calloc(op->nargs ? 3 * op->nargs : 1, sizeof(int32_t));
      if (!op->args) { status = AEE_ENOMEMORY; goto done; }
      for (uint32_t j = 0; j < op->nargs; ++j) {
        uint32_t kind, value;
        uint64_t offset, generation;
        if (tl_seq_u32(&p, end, &kind) || tl_seq_u32(&p, end, &value) ||
            tl_seq_u64(&p, end, &offset) || tl_seq_u64(&p, end, &generation)) goto done;
        uint32_t param = tl_seq_kernels[op->kernel].first + j;
        op->args[3 * j] = (int32_t)kind;
        if (tl_seq_params[param].scalar) {
          if (kind != 3 || offset || generation) goto done;
          op->args[3 * j + 1] = (int32_t)value;
        } else {
          if (kind != 0 || offset % 128 || !tl_seq_range(value, generation, offset, tl_seq_params[param].bytes)) goto done;
          op->args[3 * j + 1] = g_buffers[value].fd;
          op->args[3 * j + 2] = (int32_t)offset;
        }
      }
    } else if (op->kind == 1) {
      uint64_t sg, dg;
      if (tl_seq_u32(&p, end, &op->src) || tl_seq_u32(&p, end, &op->dst) || tl_seq_u64(&p, end, &sg) ||
          tl_seq_u64(&p, end, &dg) || tl_seq_u64(&p, end, &op->source) || tl_seq_u64(&p, end, &op->target) ||
          tl_seq_u64(&p, end, &op->size) || !tl_seq_range(op->src, sg, op->source, op->size) ||
          !tl_seq_range(op->dst, dg, op->target, op->size)) goto done;
    } else goto done;
  }
  if (p != end) { failed = UINT32_MAX; goto done; }
  // Inline arguments are unsupported. Preserve the existing non-null RPC
  // allocation convention even though these sequences have zero inline bytes.
  g_inputs = grow(g_inputs, &g_inputs_size, 1);
  g_outputs = grow(g_outputs, &g_outputs_size, 1);
  if (!g_inputs || !g_outputs) { failed = UINT32_MAX; status = AEE_ENOMEMORY; goto done; }
  stage = 0; status = 0; failed = UINT32_MAX;
  int64_t start = tl_seq_now();
  for (uint32_t i = 0; i < count; ++i) {
    struct tl_seq_op* op = &ops[i];
    int e = AEE_SUCCESS;
    if (op->kind == 0) {
      int64_t begin = tl_seq_now();
      e = TL_IFACE_run(g_handle, (int32_t)op->kernel, op->args, 3 * (int)op->nargs, g_inputs, 0, g_outputs, 0);
      g_last_run_ns = tl_seq_now() - begin;
      times[nruns++] = (struct tl_seq_time){i, op->kernel, g_last_run_ns};
    } else if (op->size) {
      e = TL_IFACE_sync(g_handle, g_buffers[op->src].fd, (uint32_t)op->source, (uint32_t)op->size, 1);
      if (e == AEE_SUCCESS)
        e = TL_IFACE_sync(g_handle, g_buffers[op->dst].fd, (uint32_t)op->target, (uint32_t)op->size, 0);
      if (e == AEE_SUCCESS)
        memmove(g_buffers[op->dst].ptr + op->target, g_buffers[op->src].ptr + op->source, (size_t)op->size);
    }
    if (e != AEE_SUCCESS) { failed = i; status = (uint32_t)e; stage = op->kind == 0 ? 2 : 3; break; }
    ++completed;
  }
  wall = tl_seq_now() - start;
done:
  if (ops) for (uint32_t i = 0; i < count; ++i) free(ops[i].args);
  free(ops); free(payload);
  int result = tl_seq_send(cli, completed, failed, stage, status, times, nruns, wall);
  free(times);
  return result;
}
"""
