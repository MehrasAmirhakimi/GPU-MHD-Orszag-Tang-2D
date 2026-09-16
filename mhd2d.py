"""
mhd2d.py
========
2D compressible resistive MHD in primitive variables. NumPy reference
implementation.

This is the CPU reference for a GPU port. Everything here is written to be
easy to check, not fast: every operation is a whole-array expression, and the
right-hand side is one flat function with no hidden state.

Variable set (the same one the Pencil Code and Astaroth evolve):

    lnrho   logarithmic density
    ux, uy  velocity
    aa      z-component of the magnetic vector potential
    ss      specific entropy

The magnetic field comes from the potential, B = curl(aa * e_z):

    Bx =  d(aa)/dy
    By = -d(aa)/dx

so div(B) = 0 holds identically, to machine precision, with no divergence
cleaning of any kind. The current density is J = -lap(aa), in the z direction.

Equation of state (ideal gas, entropy formulation):

    cs2 = cs0^2 * exp(gamma*ss/cp + (gamma - 1)*(lnrho - lnrho0))
    p   = rho * cs2 / gamma
    T   = cs2 / (cp*(gamma - 1))

Numerics:

    - 6th-order centred finite differences, periodic in x and y
    - 3rd-order 2N-storage Runge-Kutta (Williamson), the Pencil Code scheme
    - constant shear viscosity nu, magnetic diffusivity eta, entropy
      diffusion chi
    - Nordlund-type shock viscosity, needed for the compressive shocks that
      form in the Orszag-Tang vortex

Units: mu0 = 1.
"""

import numpy as np

# --------------------------------------------------------------------------
# field layout
# --------------------------------------------------------------------------

ILNRHO, IUX, IUY, IAA, ISS = 0, 1, 2, 3, 4
NVAR = 5
VARNAMES = ("lnrho", "ux", "uy", "aa", "ss")


class Params:
    """Physical and numerical parameters for a run."""

    def __init__(
        self,
        nx=256,
        ny=256,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        gamma=5.0 / 3.0,
        cp=1.0,
        cs0=1.0,
        lnrho0=0.0,
        nu=2.0e-3,
        eta=2.0e-3,
        chi=2.0e-3,
        c_shock=1.0,
        cdt=0.4,
        cdtv=0.25,
    ):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.gamma = gamma
        self.cp = cp
        self.cv = cp / gamma
        self.cs0 = cs0
        self.lnrho0 = lnrho0
        self.nu = nu
        self.eta = eta
        self.chi = chi
        self.c_shock = c_shock
        self.cdt = cdt
        self.cdtv = cdtv

    def grid(self):
        """
        Vertex-centred coordinate arrays, shaped (ny, nx).

        Points sit at x_i = i*dx, so a grid of N points is an exact subset of a
        grid of 2N points. The self-convergence test relies on that: it can
        compare resolutions by plain subsampling, with no interpolation error
        of its own getting mixed into the measurement.
        """
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        return np.meshgrid(x, y)


# --------------------------------------------------------------------------
# 6th-order centred derivatives, periodic
#
#   f'_i   = (1/(60 h))  [ -f_{i-3} +  9 f_{i-2} -  45 f_{i-1}
#                          + 45 f_{i+1} -  9 f_{i+2} +     f_{i+3} ]
#
#   f''_i  = (1/(180 h^2))[ 2 f_{i-3} - 27 f_{i-2} + 270 f_{i-1} - 490 f_i
#                          + 270 f_{i+1} - 27 f_{i+2} +   2 f_{i+3} ]
#
# Arrays are indexed [y, x], so axis 0 is y and axis 1 is x.
# --------------------------------------------------------------------------


def _d1(f, h, axis):
    r = np.roll
    return (
        -r(f, 3, axis=axis)
        + 9.0 * r(f, 2, axis=axis)
        - 45.0 * r(f, 1, axis=axis)
        + 45.0 * r(f, -1, axis=axis)
        - 9.0 * r(f, -2, axis=axis)
        + r(f, -3, axis=axis)
    ) / (60.0 * h)


def _d2(f, h, axis):
    r = np.roll
    return (
        2.0 * r(f, 3, axis=axis)
        - 27.0 * r(f, 2, axis=axis)
        + 270.0 * r(f, 1, axis=axis)
        - 490.0 * f
        + 270.0 * r(f, -1, axis=axis)
        - 27.0 * r(f, -2, axis=axis)
        + 2.0 * r(f, -3, axis=axis)
    ) / (180.0 * h * h)


def ddx(f, p):
    return _d1(f, p.dx, 1)


def ddy(f, p):
    return _d1(f, p.dy, 0)


def laplacian(f, p):
    return _d2(f, p.dx, 1) + _d2(f, p.dy, 0)


# --------------------------------------------------------------------------
# shock viscosity (Nordlund type)
#
# zeta = c_shock * dx^2 * < max(-div u, 0) >, where the maximum is taken over
# the nearest neighbours and the result is then smoothed once in each
# direction. This switches on only where the flow is compressing.
# --------------------------------------------------------------------------


def shock_viscosity(divu, p):
    q = np.maximum(-divu, 0.0)
    for ax in (0, 1):
        q = np.maximum(q, np.maximum(np.roll(q, 1, axis=ax), np.roll(q, -1, axis=ax)))
    for ax in (0, 1):
        q = 0.25 * np.roll(q, 1, axis=ax) + 0.5 * q + 0.25 * np.roll(q, -1, axis=ax)
    return p.c_shock * p.dx * p.dy * q


# --------------------------------------------------------------------------
# right-hand side
# --------------------------------------------------------------------------


def rhs(f, p):
    """d(f)/dt for the whole state vector. Returns a new (NVAR, ny, nx) array."""
    lnrho, ux, uy, aa, ss = f[ILNRHO], f[IUX], f[IUY], f[IAA], f[ISS]

    # --- derivatives we need more than once -------------------------------
    dlnrho_dx, dlnrho_dy = ddx(lnrho, p), ddy(lnrho, p)
    dux_dx, dux_dy = ddx(ux, p), ddy(ux, p)
    duy_dx, duy_dy = ddx(uy, p), ddy(uy, p)
    divu = dux_dx + duy_dy

    # --- thermodynamics ---------------------------------------------------
    cs2 = p.cs0**2 * np.exp(
        p.gamma * ss / p.cp + (p.gamma - 1.0) * (lnrho - p.lnrho0)
    )
    TT = cs2 / (p.cp * (p.gamma - 1.0))
    rho = np.exp(lnrho)

    # --- magnetic field, current, Lorentz force ---------------------------
    Bx = ddy(aa, p)
    By = -ddx(aa, p)
    lap_aa = laplacian(aa, p)
    Jz = -lap_aa
    # (J x B) with J = (0,0,Jz) is (-Jz*By, Jz*Bx, 0)
    lorentz_x = -Jz * By / rho
    lorentz_y = Jz * Bx / rho

    # --- shock viscosity --------------------------------------------------
    zeta = shock_viscosity(divu, p)
    dzeta_dx, dzeta_dy = ddx(zeta, p), ddy(zeta, p)
    ddivu_dx, ddivu_dy = ddx(divu, p), ddy(divu, p)

    # --- rate-of-strain tensor (3D traceless form, u_z = 0, d/dz = 0) -----
    Sxx = dux_dx - divu / 3.0
    Syy = duy_dy - divu / 3.0
    Szz = -divu / 3.0
    Sxy = 0.5 * (dux_dy + duy_dx)
    S2 = Sxx**2 + Syy**2 + Szz**2 + 2.0 * Sxy**2

    # --- continuity -------------------------------------------------------
    # d(lnrho)/dt = -u.grad(lnrho) - div(u), plus shock mass diffusion so the
    # density does not ring at the shock fronts.
    dlnrho = -(ux * dlnrho_dx + uy * dlnrho_dy) - divu
    dlnrho += zeta * (
        laplacian(lnrho, p) + dlnrho_dx**2 + dlnrho_dy**2
    ) + (dzeta_dx * dlnrho_dx + dzeta_dy * dlnrho_dy)

    # --- momentum ---------------------------------------------------------
    # du/dt = -u.grad(u) - cs2*grad(lnrho + ss/cp) + JxB/rho + viscous
    dux = -(ux * dux_dx + uy * dux_dy)
    duy = -(ux * duy_dx + uy * duy_dy)

    dux += -cs2 * (dlnrho_dx + ddx(ss, p) / p.cp) + lorentz_x
    duy += -cs2 * (dlnrho_dy + ddy(ss, p) / p.cp) + lorentz_y

    # constant shear viscosity: nu*(lap(u) + grad(div u)/3 + 2 S.grad(lnrho))
    dux += p.nu * (
        laplacian(ux, p)
        + ddivu_dx / 3.0
        + 2.0 * (Sxx * dlnrho_dx + Sxy * dlnrho_dy)
    )
    duy += p.nu * (
        laplacian(uy, p)
        + ddivu_dy / 3.0
        + 2.0 * (Sxy * dlnrho_dx + Syy * dlnrho_dy)
    )

    # shock viscosity: zeta*(grad(div u) + div(u)*grad(lnrho)) + div(u)*grad(zeta)
    dux += zeta * (ddivu_dx + divu * dlnrho_dx) + divu * dzeta_dx
    duy += zeta * (ddivu_dy + divu * dlnrho_dy) + divu * dzeta_dy

    # --- induction (resistive gauge) --------------------------------------
    # d(aa)/dt = (u x B)_z + eta*lap(aa)
    daa = (ux * By - uy * Bx) + p.eta * lap_aa

    # --- entropy ----------------------------------------------------------
    # rho*T*Ds/Dt = 2*rho*nu*S^2 + eta*J^2 + zeta*(div u)^2
    dss = -(ux * ddx(ss, p) + uy * ddy(ss, p))
    dss += (2.0 * p.nu * S2 + p.eta * Jz**2 / rho + zeta * divu**2) / TT
    # entropy diffusion, a numerical smoothing term rather than a physical
    # conductivity; keeps the entropy field clean at this resolution
    dss += p.chi * laplacian(ss, p)

    df = np.empty_like(f)
    df[ILNRHO], df[IUX], df[IUY], df[IAA], df[ISS] = dlnrho, dux, duy, daa, dss
    return df


# --------------------------------------------------------------------------
# time step
# --------------------------------------------------------------------------


def timestep(f, p):
    """CFL-limited step: advective plus fast magnetosonic, and diffusive."""
    lnrho, ux, uy, aa, ss = f[ILNRHO], f[IUX], f[IUY], f[IAA], f[ISS]
    rho = np.exp(lnrho)
    cs2 = p.cs0**2 * np.exp(
        p.gamma * ss / p.cp + (p.gamma - 1.0) * (lnrho - p.lnrho0)
    )
    Bx, By = ddy(aa, p), -ddx(aa, p)
    va2 = (Bx**2 + By**2) / rho
    speed = np.sqrt(ux**2 + uy**2) + np.sqrt(cs2 + va2)

    h = min(p.dx, p.dy)
    dt_adv = p.cdt * h / max(speed.max(), 1e-12)

    divu = ddx(ux, p) + ddy(uy, p)
    diff = max(p.nu, p.eta, p.chi, shock_viscosity(divu, p).max())
    dt_diff = p.cdtv * h * h / max(diff, 1e-12)

    return min(dt_adv, dt_diff)


# --------------------------------------------------------------------------
# 2N-storage Runge-Kutta 3 (Williamson)
# --------------------------------------------------------------------------

RK_ALPHA = (0.0, -5.0 / 9.0, -153.0 / 128.0)
RK_BETA = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)


def rk3_step(f, dt, p):
    """One full RK3 step, in place. Two arrays of storage, as on the GPU."""
    df = np.zeros_like(f)
    for alpha, beta in zip(RK_ALPHA, RK_BETA):
        df *= alpha
        df += dt * rhs(f, p)
        f += beta * df
    return f


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def diagnostics(f, p):
    """Volume-averaged quantities, for checking a run behaves."""
    lnrho, ux, uy, aa, ss = f[ILNRHO], f[IUX], f[IUY], f[IAA], f[ISS]
    rho = np.exp(lnrho)
    cs2 = p.cs0**2 * np.exp(
        p.gamma * ss / p.cp + (p.gamma - 1.0) * (lnrho - p.lnrho0)
    )
    pressure = rho * cs2 / p.gamma
    Bx, By = ddy(aa, p), -ddx(aa, p)

    ekin = 0.5 * rho * (ux**2 + uy**2)
    emag = 0.5 * (Bx**2 + By**2)
    ethm = pressure / (p.gamma - 1.0)

    # div(B) recomputed the long way round, as an independent check that the
    # potential formulation really is doing what it claims
    divB = ddx(Bx, p) + ddy(By, p)

    return {
        "rho_mean": float(rho.mean()),
        "rho_min": float(rho.min()),
        "ekin": float(ekin.mean()),
        "emag": float(emag.mean()),
        "ethm": float(ethm.mean()),
        "etot": float((ekin + emag + ethm).mean()),
        "umax": float(np.sqrt(ux**2 + uy**2).max()),
        "divB_max": float(np.abs(divB).max()),
        "p_min": float(pressure.min()),
        "p_max": float(pressure.max()),
    }


def pressure_field(f, p):
    lnrho, ss = f[ILNRHO], f[ISS]
    rho = np.exp(lnrho)
    cs2 = p.cs0**2 * np.exp(
        p.gamma * ss / p.cp + (p.gamma - 1.0) * (lnrho - p.lnrho0)
    )
    return rho * cs2 / p.gamma


# --------------------------------------------------------------------------
# initial conditions
# --------------------------------------------------------------------------


def init_orszag_tang(p):
    """
    Orszag-Tang vortex, the usual astrophysical normalisation:

        rho = gamma^2,  p = gamma,  so cs = 1
        u = (-sin y,  sin x)
        B = (-sin y,  sin 2x)

    The vector potential that gives that field is aa = cos(y) + cos(2x)/2.
    Density and pressure are uniform at t = 0, so the entropy is uniform too
    and we can simply set it to zero.
    """
    x, y = p.grid()
    f = np.zeros((NVAR, p.ny, p.nx))
    f[ILNRHO] = np.log(p.gamma**2)
    f[IUX] = -np.sin(y)
    f[IUY] = np.sin(x)
    f[IAA] = np.cos(y) + 0.5 * np.cos(2.0 * x)
    f[ISS] = 0.0

    # cs0 and lnrho0 are the reference state of the equation of state; setting
    # them to the initial state makes ss = 0 the correct initial entropy
    p.cs0 = 1.0
    p.lnrho0 = np.log(p.gamma**2)
    return f


def init_sound_wave(p, amplitude=1.0e-4, kx=1):
    """
    Sound wave travelling in +x, used for the convergence test.

    Background is rho = 1, cs = 1, u = 0, no magnetic field. For a rightward
    adiabatic wave the density and velocity perturbations are in phase, with
    u' = cs * rho'/rho. The entropy perturbation is zero, which with this
    equation of state is exactly the adiabatic relation p ~ rho^gamma.

    Note on what is *not* here: a shear Alfven wave would be the more natural
    MHD test, but it needs a uniform background field, and a uniform field has
    no periodic vector potential. Carrying the background separately from the
    evolved potential would fix that; it is not done yet.
    """
    x, _ = p.grid()
    k = 2.0 * np.pi * kx / p.Lx

    f = np.zeros((NVAR, p.ny, p.nx))
    f[ILNRHO] = amplitude * np.cos(k * x)
    f[IUX] = amplitude * np.cos(k * x)  # cs = 1
    f[IUY] = 0.0
    f[IAA] = 0.0
    f[ISS] = 0.0

    p.cs0 = 1.0
    p.lnrho0 = 0.0
    return f
