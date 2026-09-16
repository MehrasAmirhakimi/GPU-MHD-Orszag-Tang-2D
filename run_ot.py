"""
run_ot.py
=========
Orszag-Tang vortex: the standard first benchmark for a compressible MHD code.

Smooth initial conditions steepen into a network of interacting shocks and
current sheets, so the test exercises the whole solver at once. The pressure
field at t = 0.5 is the usual thing to compare against published results.

    python run_ot.py --nx 256 --tmax 0.5

Writes ot_pressure.png and ot_diagnostics.csv into the working directory, and
saves the final state as ot_state.npz so the CUDA port can be diffed against
it field by field.
"""

import argparse
import time

import numpy as np

import mhd2d as m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--tmax", type=float, default=0.5)
    ap.add_argument("--nu", type=float, default=2.0e-3)
    ap.add_argument("--eta", type=float, default=2.0e-3)
    ap.add_argument("--chi", type=float, default=2.0e-3)
    ap.add_argument("--every", type=int, default=20, help="diagnostics interval")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    p = m.Params(nx=args.nx, ny=args.nx, nu=args.nu, eta=args.eta, chi=args.chi)
    f = m.init_orszag_tang(p)

    d0 = m.diagnostics(f, p)
    print(f"grid {p.nx} x {p.ny}   nu = {p.nu:g}  eta = {p.eta:g}  chi = {p.chi:g}")
    print(f"t = 0.000  Etot = {d0['etot']:.8f}  divB = {d0['divB_max']:.2e}")

    rows = [("t", "dt", "ekin", "emag", "ethm", "etot", "rho_min", "divB_max")]
    t, step = 0.0, 0
    wall = time.time()

    while t < args.tmax - 1e-12:
        dt = min(m.timestep(f, p), args.tmax - t)
        f = m.rk3_step(f, dt, p)
        t += dt
        step += 1

        if step % args.every == 0 or t >= args.tmax - 1e-12:
            d = m.diagnostics(f, p)
            rows.append(
                (
                    f"{t:.6f}",
                    f"{dt:.6e}",
                    f"{d['ekin']:.8f}",
                    f"{d['emag']:.8f}",
                    f"{d['ethm']:.8f}",
                    f"{d['etot']:.8f}",
                    f"{d['rho_min']:.6f}",
                    f"{d['divB_max']:.3e}",
                )
            )
            print(
                f"t = {t:.3f}  dt = {dt:.2e}  Etot = {d['etot']:.8f}  "
                f"rho_min = {d['rho_min']:.4f}  divB = {d['divB_max']:.2e}"
            )

    wall = time.time() - wall
    d1 = m.diagnostics(f, p)

    print()
    print(f"{step} steps in {wall:.1f} s  ({1e6*wall/(step*p.nx*p.ny):.2f} us per point-step)")
    print(f"energy drift  {abs(d1['etot']-d0['etot'])/d0['etot']:.3e}")
    print(f"mass drift    {abs(d1['rho_mean']-d0['rho_mean'])/d0['rho_mean']:.3e}")
    print(f"max |div B|   {d1['divB_max']:.3e}")
    print(f"pressure      min {d1['p_min']:.4f}   max {d1['p_max']:.4f}")

    with open("ot_diagnostics.csv", "w") as fh:
        for r in rows:
            fh.write(",".join(r) + "\n")
    np.savez_compressed("ot_state.npz", f=f, nx=p.nx, ny=p.ny, t=t)

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pres = m.pressure_field(f, p)
        aa = f[m.IAA]

        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        im = ax[0].imshow(
            pres, origin="lower", extent=[0, p.Lx, 0, p.Ly], cmap="inferno"
        )
        ax[0].set_title(f"pressure, t = {t:.2f}")
        fig.colorbar(im, ax=ax[0], fraction=0.046)

        ax[1].contour(
            np.linspace(0, p.Lx, p.nx),
            np.linspace(0, p.Ly, p.ny),
            aa,
            levels=30,
            colors="k",
            linewidths=0.6,
        )
        ax[1].set_aspect("equal")
        ax[1].set_xlim(0, p.Lx)
        ax[1].set_ylim(0, p.Ly)
        ax[1].set_title("field lines (contours of $A_z$)")

        for a in ax:
            a.set_xlabel("x")
            a.set_ylabel("y")
        fig.suptitle(f"Orszag-Tang vortex, {p.nx}$^2$, 6th order FD + RK3")
        fig.tight_layout()
        fig.savefig("ot_pressure.png", dpi=140)
        print("wrote ot_pressure.png")


if __name__ == "__main__":
    main()
