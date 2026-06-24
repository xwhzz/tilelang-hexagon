"""Compare the device output (project/C.bin) against the golden reference."""
import os

import numpy as np

M = N = 256
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "project")

C = np.fromfile(os.path.join(P, "C.bin"), dtype=np.float16).reshape(M, N).astype(np.float32)
ref = np.load(os.path.join(P, "ref.npy"))
err = float(np.abs(C - ref).max())
ok = err < 0.1  # fp16 rounding
print(f"offline matmul {M}x{N}x{M} on HMX vs golden: max abs err = {err:.4g}  "
      f"({'PASS' if ok else 'FAIL'})")
raise SystemExit(0 if ok else 1)
