"""
compare.py
==========
Field-by-field check of a C++/CUDA run against the NumPy reference.

    python compare.py ot_state.npz cuda/ot_state_gpu.npy

Both runs start from the same initial condition and take the same time steps,
so the only difference between them should be floating-point rounding: the
order of a few operations, the last bit of exp() on a GPU, fused
multiply-adds. That is ~1e-16 per operation, and it grows slowly over a
smooth run. Anything much bigger than round-off is a bug.

For each field the error is normalised by the size of that field,
max|a - b| / max|b|, so a field that is small everywhere (the entropy, which
starts at zero) is judged on its own scale.

On the 256^2 run to t = 0.5, the C++ CPU build lands at 8e-14, and the same
build with fused multiply-adds everywhere (as a GPU does) at 1e-13. The
default tolerance of 1e-10 leaves about three orders of magnitude over that.
A real bug is far outside it: a kernel-ordering race caught during
development showed up at 1e-1.

Exit code is 0 on a pass and 1 on a fail, so it can sit in a script.
"""

import argparse
import sys

import numpy as np

NAMES = ("lnrho", "ux", "uy", "aa", "ss")


def load(path):
    d = np.load(path)
    if isinstance(d, np.lib.npyio.NpzFile):
        return d["f"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="ot_state.npz from run_ot.py")
    ap.add_argument("candidate", help=".npy from mhd2d_cpu or mhd2d_gpu")
    ap.add_argument("--tol", type=float, default=1e-10)
    args = ap.parse_args()

    ref = load(args.reference)
    new = load(args.candidate)

    if ref.shape != new.shape:
        print(f"shape mismatch: reference {ref.shape}, candidate {new.shape}")
        print("was the candidate run on the same grid (--nx)?")
        return 1

    if not np.all(np.isfinite(new)):
        print("candidate contains NaN or Inf")
        return 1

    print(f"grid {ref.shape[2]} x {ref.shape[1]}, tolerance {args.tol:g}")
    print(f"{'field':>6}  {'max|ref|':>11}  {'max|diff|':>11}  {'normalised':>11}")

    worst = 0.0
    for v, name in enumerate(NAMES):
        scale = np.abs(ref[v]).max()
        diff = np.abs(new[v] - ref[v]).max()
        rel = diff / scale if scale > 0 else diff
        worst = max(worst, rel)
        flag = "" if rel <= args.tol else "   <-- FAIL"
        print(f"{name:>6}  {scale:11.4e}  {diff:11.4e}  {rel:11.4e}{flag}")

    ok = worst <= args.tol
    print()
    print(f"worst normalised difference {worst:.3e}:  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
