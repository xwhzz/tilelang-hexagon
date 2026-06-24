"""Write the golden inputs + fp32 reference into ./project/ — numpy only, no tilelang.

Fixed seed, so A.bin/B.bin/ref.npy are reproducible.  Run by reproduce.sh before
the device run; verify.py compares the device C.bin against ref.npy.
"""
import os

import numpy as np

M = N = K = 256
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project")
os.makedirs(P, exist_ok=True)

rng = np.random.default_rng(0)
A = (rng.standard_normal((M, K), dtype=np.float32) * 0.25).astype(np.float16)
B = (rng.standard_normal((K, N), dtype=np.float32) * 0.25).astype(np.float16)
A.tofile(os.path.join(P, "A.bin"))
B.tofile(os.path.join(P, "B.bin"))
np.save(os.path.join(P, "ref.npy"), A.astype(np.float32) @ B.astype(np.float32))
print(f"golden: A.bin B.bin ({M}x{K}, {K}x{N} fp16) + ref.npy ({M}x{N} fp32) in {P}")
