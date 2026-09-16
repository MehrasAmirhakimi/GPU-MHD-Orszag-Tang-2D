"""
test_convergence.py
===================
Two checks that the scheme is doing what it claims.

1. Operator order. The first derivative and the Laplacian are applied to a
   function whose derivatives are known exactly. The error should fall by a
   factor of 64 every time the resolution doubles.

2. Solver self-convergence. A small-amplitude sound wave is run at several
   resolutions and compared against a high-resolution reference on the same
   grid points, which the vertex-centred grid makes exact. Comparing against
   the *linear* analytic solution would not work: the code is nonlinear, and
   at these amplitudes the nonlinear correction is larger than the truncation
   error being measured.

   The time step is held fixed across resolutions, so that the RK3 error is
   the same constant everywhere and the spatial order is what gets measured.
   That error is still a floor: once the spatial error drops below it, the
   measured order falls off. Where that happens is printed, not hidden.

    python test_convergence.py
"""

import numpy as np

import mhd2d as m


def order(e_coarse, e_fine):
    return np.log2(e_coarse / e_fine)


def operator_test():
    print("1. operator convergence")
    print("   " + "-" * 52)
    print(f"   {'N':>5}  {'err d/dx':>12} {'order':>7}  {'err lap':>12} {'order':>7}")

    prev1 = prev2 = None
    for N in (32, 64, 128, 256):
        p = m.Params(nx=N, ny=N)
        x, y = p.grid()
        f = np.sin(x) * np.cos(2.0 * y)

        e1 = np.abs(m.ddx(f, p) - np.cos(x) * np.cos(2.0 * y)).max()
        e2 = np.abs(m.laplacian(f, p) + 5.0 * f).max()

        o1 = f"{order(prev1, e1):7.2f}" if prev1 else "      -"
        o2 = f"{order(prev2, e2):7.2f}" if prev2 else "      -"
        print(f"   {N:5d}  {e1:12.4e} {o1}  {e2:12.4e} {o2}")
        prev1, prev2 = e1, e2
    print()


def run_wave(N, tmax, dt, amplitude=1.0e-4, ny=16):
    p = m.Params(nx=N, ny=ny, nu=0.0, eta=0.0, chi=0.0, c_shock=0.0)
    f = m.init_sound_wave(p, amplitude=amplitude)
    t = 0.0
    while t < tmax - 1e-12:
        step = min(dt, tmax - t)
        f = m.rk3_step(f, step, p)
        t += step
    return f[m.ILNRHO][0, :]  # the solution is uniform in y


def self_convergence_test():
    tmax, dt, amp = 0.2, 2.5e-4, 1.0e-4
    Nref = 512
    resolutions = (32, 64, 128)

    print("2. solver self-convergence, sound wave")
    print(f"   reference {Nref}, t = {tmax}, fixed dt = {dt:g}, amplitude = {amp:g}")
    print("   " + "-" * 40)
    print(f"   {'N':>5}  {'L2 error':>12} {'order':>7}")

    ref = run_wave(Nref, tmax, dt, amp)

    prev = None
    for N in resolutions:
        sol = run_wave(N, tmax, dt, amp)
        stride = Nref // N
        err = np.sqrt(np.mean((sol - ref[::stride]) ** 2))
        o = f"{order(prev, err):7.2f}" if prev else "      -"
        print(f"   {N:5d}  {err:12.4e} {o}")
        prev = err
    print()


def conservation_test():
    """Mass, energy and div(B) over an Orszag-Tang run."""
    print("3. conservation over 0.5 time units of Orszag-Tang, 128^2")
    print("   " + "-" * 40)

    p = m.Params(nx=128, ny=128, nu=3.0e-3, eta=3.0e-3, chi=3.0e-3)
    f = m.init_orszag_tang(p)
    d0 = m.diagnostics(f, p)

    t = 0.0
    while t < 0.5 - 1e-12:
        dt = min(m.timestep(f, p), 0.5 - t)
        f = m.rk3_step(f, dt, p)
        t += dt
    d1 = m.diagnostics(f, p)

    print(f"   mass drift     {abs(d1['rho_mean']-d0['rho_mean'])/d0['rho_mean']:.3e}")
    print(f"   energy drift   {abs(d1['etot']-d0['etot'])/d0['etot']:.3e}")
    print(f"   max |div B|    {d1['divB_max']:.3e}")
    print(
        f"   kinetic + magnetic lost {d0['ekin']+d0['emag']-d1['ekin']-d1['emag']:.6f}, "
        f"thermal gained {d1['ethm']-d0['ethm']:.6f}"
    )
    print()


if __name__ == "__main__":
    operator_test()
    self_convergence_test()
    conservation_test()
