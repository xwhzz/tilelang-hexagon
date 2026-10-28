"""The weight streams of the megakernels, and the decode's w4a16 GEMV blocks.

A decode step reads every weight once, in an order fixed ahead: phase by
phase, block by block. Each worker program streams its share of the blocks
into a ring of VTCM slots with ``T.dma_copy``, ``depth`` blocks ahead of the
one it multiplies, across phase boundaries: the first blocks of a phase
arrive while the workers still finish the last one and meet at
``T.sync_grid()``.

A block is ``block`` rows of one slice of a projection's columns, contiguous
in the bundle with its scales (``bundle.py``), so one DMA each brings it. The
schedule is planned in Python while the kernel is traced: a ``Stream`` knows
every phase of the step, so the blocks a loop submits at its end are the next
phase's. Blocks of one shape share a ring. A step that a loop repeats, a token
of a generation, is a cycle: its last blocks submit the first ones of the next
round. The prefill streams each layer's phases in two streams that share their
rings, for the layer its loop is at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.panel import make_hvx_panel_layout


@dataclass
class Projection:
    """A stacked projection of the bundle: codes [L, S, N, cols] in the panel
    layout, S slices of its columns, and their fp16 scales
    [L, S, N / block, cols / group, block]."""

    w: object
    s: object
    group: int
    block: int
    bits: int = 4

    @property
    def cols(self) -> int:
        return int(self.w.shape[3])

    @property
    def slices(self) -> int:
        return int(self.w.shape[1])


@dataclass
class Phase:
    """One worker's share of a GEMV of the step: `blocks` blocks of layer
    `layer` (a Python or a TIR index), block b being rows [row(b), row(b) +
    proj.block) of slice slice(b). `row` and `slice` take a Python or a TIR
    index."""

    proj: Projection
    layer: int
    blocks: int
    row: Callable
    slice: Callable = lambda b: 0

    @classmethod
    def rows(cls, proj: Projection, layer: int, row0, blocks: int) -> Phase:
        """Consecutive blocks of rows from row0, all slices of each in turn."""
        S = proj.slices
        return cls(proj, layer, blocks * S, lambda b: row0 + b // S * proj.block, lambda b: b % S)


@T.macro
def submit(w, s, ring_w, ring_s, layer, sl, row, k, rows: int):
    """The codes and the scales of rows [row, row + rows) of slice sl: two DMA
    groups."""
    T.dma_copy(w[layer, sl, row : row + rows, :], ring_w[k, :, :])
    T.dma_copy(s[layer, sl, row // rows, :, :], ring_s[k, :, :])


@T.macro
def steady(ph: Phase, ring_w, ring_s, base, n: int, depth: int, consume):
    """Blocks [0, n) of phase ph, the first in ring position `base`: each waits
    for itself, is consumed, and submits the block `depth` later of the same
    phase."""
    p = ph.proj
    for b in T.serial(n):
        T.dma_wait(2 * (depth - 1))
        consume(b, ring_w, ring_s, (base + b) % depth)
        submit(p.w, p.s, ring_w, ring_s, ph.layer, ph.slice(b + depth), ph.row(b + depth), (base + b + depth) % depth, p.block)


@T.macro
def last(b: int, ring_w, ring_s, k, pending: int, consume):
    T.dma_wait(pending)
    consume(b, ring_w, ring_s, k)


class Stream:
    """A worker's DMA schedule over the phases of a step. With `cycle`, the
    index of the loop that repeats the step (a TIR variable), the last blocks
    of a round submit the first ones of the next. Streams that run one after
    another, each drained before the next begins, can share their rings:
    pass the same dict as `rings`."""

    def __init__(self, phases: list, depth: int, cycle=None, rings: dict | None = None):
        self.phases = phases
        self.depth = depth
        self.cycle = cycle
        # One ring per block shape: `depth` slots of codes and of scales.
        self.rings = {} if rings is None else rings
        for ph in phases:
            key = self._key(ph)
            if key not in self.rings:
                p = ph.proj
                w = T.alloc_shared((depth, p.block, p.cols), f"int{p.bits}")
                s = T.alloc_shared((depth, p.cols // p.group, p.block), "float16")
                self.rings[key] = (w, s)
        # The ring position of each phase's first block in a round, and the
        # blocks of a round per ring.
        self.base, self.count = [], {}
        for ph in phases:
            key = self._key(ph)
            self.base.append(self.count.get(key, 0))
            self.count[key] = self.count.get(key, 0) + ph.blocks
        self.items = [(i, b) for i, ph in enumerate(phases) for b in range(ph.blocks)]

    def layouts(self) -> dict:
        """The panel layouts of the projections and of their rings, for
        T.annotate_layout: the codes keep the layout they stream in."""
        out = {}
        for ph in self.phases:
            out[ph.proj.w] = make_hvx_panel_layout(ph.proj.w)
        for w, _ in self.rings.values():
            out[w] = make_hvx_panel_layout(w)
        return out

    @staticmethod
    def _key(ph: Phase):
        p = ph.proj
        return (p.bits, p.block, p.cols, p.group)

    def _base(self, i: int, round_=None):
        """The ring position of phase i's first block, in round `round_` (the
        current one by default) of a cycle."""
        if self.cycle is None:
            return self.base[i]
        r = self.cycle if round_ is None else round_
        return r * self.count[self._key(self.phases[i])] + self.base[i]

    def _submit(self, i: int, b: int, round_=None):
        ph = self.phases[i]
        p = ph.proj
        ring_w, ring_s = self.rings[self._key(ph)]
        submit(p.w, p.s, ring_w, ring_s, ph.layer, ph.slice(b), ph.row(b), (self._base(i, round_) + b) % self.depth, p.block)

    def prologue(self):
        """Submit the first `depth` blocks of the step (of round 0 of a cycle)."""
        for i, b in self.items[: self.depth]:
            self._submit(i, b, 0)

    def begin(self, cycle):
        """Start the rounds of a cycle: `cycle` is the loop's index."""
        self.cycle = cycle

    def run(self, i: int, consume):
        """Phase i: `consume(b, ring_w, ring_s, k)` each block b, in slot k of
        the rings, once it arrives, keeping `depth` blocks in flight."""
        ph = self.phases[i]
        d = self.depth
        first = sum(q.blocks for q in self.phases[:i])
        ring_w, ring_s = self.rings[self._key(ph)]
        n = max(0, ph.blocks - d)
        if n > 0:
            steady(ph, ring_w, ring_s, self._base(i), n, d, consume)
        # The last blocks submit the next phases' first ones, of the next round
        # past the end of a cycle's.
        for b in range(n, ph.blocks):
            item = first + b
            ahead = item + d
            in_flight = d if self.cycle is not None else min(d, len(self.items) - item)
            last(b, ring_w, ring_s, (self._base(i) + b) % d, 2 * (in_flight - 1), consume)
            if ahead < len(self.items):
                self._submit(*self.items[ahead])
            elif self.cycle is not None:
                self._submit(*self.items[ahead - len(self.items)], self.cycle + 1)


@T.macro
def quantize(src, xq, sa, amax, K: int):
    """xq = round(src / sa[0]) as int16, with sa[0] = max|src| / 32767 (amax[0]
    holds max|src|)."""
    T.reduce_absmax(src, amax, dim=1)
    for i, k in T.Parallel(1, K):
        v = src[i, k] * (T.float32(32767) / T.max(amax[0], T.float32(1e-30)))
        xq[i, k] = T.Cast("int16", v + T.if_then_else(v >= T.float32(0), T.float32(0.5), T.float32(-0.5)))
    sa[0] = T.max(amax[0], T.float32(1e-30)) / T.float32(32767)


@T.macro
def gemv_block(xq, sa, ring_w, ring_s, k, acc, c0, cols: int, G: int, clear):
    """acc[0, j] (+)= sa[0] sum_g ring_s[k, g, j] sum_{c in g} ring_w[k, j, c] xq[0, c0 + c]:
    the block in slot k, a slice of `cols` columns from c0, with its fp16
    scales per G columns; `clear` starts acc over."""
    T.gemm_blockscaled(
        xq[:, c0 : c0 + cols],
        ring_w[k, :, :],
        acc,
        sa,
        ring_s[k, :, :],
        transpose_B=True,
        clear_accum=clear,
        k_start=0,
        sf_a_granularity_k=cols,
        sf_b_granularity_k=G,
    )
