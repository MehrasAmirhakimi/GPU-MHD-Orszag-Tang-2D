// mhd_core.h
// ==========
// The physics, written once. Every function here is a pure function of the
// grid point (i, j) and read-only arrays, so the same code runs inside a CPU
// loop and inside a CUDA kernel. The two backends differ only in how they
// loop, where memory lives, and how they reduce.
//
// This is a line-by-line transcription of mhd2d.py. Where floating-point
// evaluation order matters, the C++ keeps the order NumPy uses, so the two
// codes agree to round-off and a disagreement means a bug, not a rounding
// difference. Comments marked "order" point at the places where that took
// some care.
//
// Layout: one contiguous array of NVAR fields, each ny rows of nx points,
// row-major, x fastest:   f[v*nx*ny + j*nx + i]
// Axis convention matches NumPy's [y, x] indexing.

#pragma once

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP_CPU_RT__)
#define HD __host__ __device__ inline
#else
#define HD inline
#endif

typedef double real;

enum { ILNRHO = 0, IUX = 1, IUY = 2, IAA = 3, ISS = 4, NVAR = 5 };

// 2N-storage RK3 (Williamson)
static const real RK_ALPHA[3] = {0.0, -5.0 / 9.0, -153.0 / 128.0};
static const real RK_BETA[3] = {1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0};

struct Params {
    int nx, ny;
    real Lx, Ly, dx, dy;
    real gamma, cp, cs0, lnrho0;
    real nu, eta, chi, c_shock, cdt, cdtv;

    // derived, set by finalize(); computed on the host once, the way Python
    // computes them once as scalars before touching any array
    real cs0sq;       // cs0**2
    real cp_gm1;      // cp*(gamma - 1)
    real shock_pref;  // c_shock*dx*dy
};

inline Params default_params(int nx, int ny) {
    Params p;
    p.nx = nx;
    p.ny = ny;
    p.Lx = 2.0 * M_PI;
    p.Ly = 2.0 * M_PI;
    p.gamma = 5.0 / 3.0;
    p.cp = 1.0;
    p.cs0 = 1.0;
    p.lnrho0 = 0.0;
    p.nu = 2.0e-3;
    p.eta = 2.0e-3;
    p.chi = 2.0e-3;
    p.c_shock = 1.0;
    p.cdt = 0.4;
    p.cdtv = 0.25;
    return p;
}

inline void finalize(Params& p) {
    p.dx = p.Lx / p.nx;
    p.dy = p.Ly / p.ny;
    p.cs0sq = p.cs0 * p.cs0;
    p.cp_gm1 = p.cp * (p.gamma - 1.0);
    p.shock_pref = p.c_shock * p.dx * p.dy;  // order: (c*dx)*dy, as Python
}

// --------------------------------------------------------------------------
// indexing
// --------------------------------------------------------------------------

// periodic wrap for offsets of at most one period, which is all a
// 7-point stencil ever needs
HD int wrap(int k, int n) { return k < 0 ? k + n : (k >= n ? k - n : k); }

HD real maxr(real a, real b) { return a > b ? a : b; }

// --------------------------------------------------------------------------
// 6th-order centred differences
// order: left to right, exactly as the NumPy expression is written
// --------------------------------------------------------------------------

HD real d1(real m3, real m2, real m1, real p1, real p2, real p3, real h) {
    return (((((-m3 + 9.0 * m2) - 45.0 * m1) + 45.0 * p1) - 9.0 * p2) + p3) /
           (60.0 * h);
}

HD real d2(real m3, real m2, real m1, real c, real p1, real p2, real p3, real h) {
    return ((((((2.0 * m3 - 27.0 * m2) + 270.0 * m1) - 490.0 * c) + 270.0 * p1) -
             27.0 * p2) +
            2.0 * p3) /
           (180.0 * h * h);
}

// a points at the first element of one field

HD real ddx(const real* a, int i, int j, const Params& p) {
    const real* r = a + j * p.nx;
    const int n = p.nx;
    return d1(r[wrap(i - 3, n)], r[wrap(i - 2, n)], r[wrap(i - 1, n)],
              r[wrap(i + 1, n)], r[wrap(i + 2, n)], r[wrap(i + 3, n)], p.dx);
}

HD real ddy(const real* a, int i, int j, const Params& p) {
    const int n = p.ny, s = p.nx;
    return d1(a[wrap(j - 3, n) * s + i], a[wrap(j - 2, n) * s + i],
              a[wrap(j - 1, n) * s + i], a[wrap(j + 1, n) * s + i],
              a[wrap(j + 2, n) * s + i], a[wrap(j + 3, n) * s + i], p.dy);
}

HD real d2x(const real* a, int i, int j, const Params& p) {
    const real* r = a + j * p.nx;
    const int n = p.nx;
    return d2(r[wrap(i - 3, n)], r[wrap(i - 2, n)], r[wrap(i - 1, n)], r[i],
              r[wrap(i + 1, n)], r[wrap(i + 2, n)], r[wrap(i + 3, n)], p.dx);
}

HD real d2y(const real* a, int i, int j, const Params& p) {
    const int n = p.ny, s = p.nx;
    return d2(a[wrap(j - 3, n) * s + i], a[wrap(j - 2, n) * s + i],
              a[wrap(j - 1, n) * s + i], a[j * s + i], a[wrap(j + 1, n) * s + i],
              a[wrap(j + 2, n) * s + i], a[wrap(j + 3, n) * s + i], p.dy);
}

HD real lap(const real* a, int i, int j, const Params& p) {
    return d2x(a, i, j, p) + d2y(a, i, j, p);
}

// --------------------------------------------------------------------------
// equation of state
// --------------------------------------------------------------------------

HD real sound_speed2(real lnrho, real ss, const Params& p) {
    return p.cs0sq *
           exp(p.gamma * ss / p.cp + (p.gamma - 1.0) * (lnrho - p.lnrho0));
}

// --------------------------------------------------------------------------
// auxiliary fields: div(u) and the shock viscosity
//
// zeta needs div(u) at neighbours, then a 3x3 max, then two smoothing
// passes, so it cannot be computed pointwise inside the RHS. It is built in
// four sweeps, each one a pure stencil of the previous sweep's output.
// --------------------------------------------------------------------------

HD real divu_point(const real* f, int i, int j, const Params& p) {
    const int n = p.nx * p.ny;
    return ddx(f + IUX * n, i, j, p) + ddy(f + IUY * n, i, j, p);
}

// max over the 3x3 neighbourhood of max(-div u, 0). Taking the max in one go
// is exactly the same as NumPy's two one-axis passes, since max is exact.
HD real shock_max_point(const real* divu, int i, int j, const Params& p) {
    real q = 0.0;
    for (int dj = -1; dj <= 1; ++dj) {
        const int jj = wrap(j + dj, p.ny) * p.nx;
        for (int di = -1; di <= 1; ++di) q = maxr(q, -divu[jj + wrap(i + di, p.nx)]);
    }
    return q;
}

HD real smooth_y_point(const real* q, int i, int j, const Params& p) {
    const int s = p.nx;
    return 0.25 * q[wrap(j - 1, p.ny) * s + i] + 0.5 * q[j * s + i] +
           0.25 * q[wrap(j + 1, p.ny) * s + i];
}

HD real zeta_point(const real* qy, int i, int j, const Params& p) {
    const real* r = qy + j * p.nx;
    return p.shock_pref * (0.25 * r[wrap(i - 1, p.nx)] + 0.5 * r[i] +
                           0.25 * r[wrap(i + 1, p.nx)]);
}

// --------------------------------------------------------------------------
// right-hand side at one point
// --------------------------------------------------------------------------

HD void rhs_point(const real* f, const real* divu, const real* zeta, int i,
                  int j, const Params& p, real out[NVAR]) {
    const int n = p.nx * p.ny;
    const int k = j * p.nx + i;
    const real* LNRHO = f + ILNRHO * n;
    const real* UX = f + IUX * n;
    const real* UY = f + IUY * n;
    const real* AA = f + IAA * n;
    const real* SS = f + ISS * n;

    const real lnrho = LNRHO[k], ux = UX[k], uy = UY[k], ss = SS[k];

    const real dlnrho_dx = ddx(LNRHO, i, j, p), dlnrho_dy = ddy(LNRHO, i, j, p);
    const real dux_dx = ddx(UX, i, j, p), dux_dy = ddy(UX, i, j, p);
    const real duy_dx = ddx(UY, i, j, p), duy_dy = ddy(UY, i, j, p);
    const real dv = dux_dx + duy_dy;  // same expression as divu_point

    // thermodynamics
    const real cs2 = sound_speed2(lnrho, ss, p);
    const real TT = cs2 / p.cp_gm1;
    const real rho = exp(lnrho);

    // field, current, Lorentz force
    const real Bx = ddy(AA, i, j, p);
    const real By = -ddx(AA, i, j, p);
    const real lap_aa = lap(AA, i, j, p);
    const real Jz = -lap_aa;
    const real lorentz_x = -Jz * By / rho;
    const real lorentz_y = Jz * Bx / rho;

    // shock viscosity
    const real z = zeta[k];
    const real dzeta_dx = ddx(zeta, i, j, p), dzeta_dy = ddy(zeta, i, j, p);
    const real ddivu_dx = ddx(divu, i, j, p), ddivu_dy = ddy(divu, i, j, p);

    // rate of strain
    const real Sxx = dux_dx - dv / 3.0;
    const real Syy = duy_dy - dv / 3.0;
    const real Szz = -dv / 3.0;
    const real Sxy = 0.5 * (dux_dy + duy_dx);
    const real S2 = Sxx * Sxx + Syy * Syy + Szz * Szz + 2.0 * (Sxy * Sxy);

    // continuity
    real dlnrho = -(ux * dlnrho_dx + uy * dlnrho_dy) - dv;
    dlnrho += z * (lap(LNRHO, i, j, p) + dlnrho_dx * dlnrho_dx +
                   dlnrho_dy * dlnrho_dy) +
              (dzeta_dx * dlnrho_dx + dzeta_dy * dlnrho_dy);

    // momentum
    const real dss_dx = ddx(SS, i, j, p), dss_dy = ddy(SS, i, j, p);
    real dux = -(ux * dux_dx + uy * dux_dy);
    real duy = -(ux * duy_dx + uy * duy_dy);

    dux += -cs2 * (dlnrho_dx + dss_dx / p.cp) + lorentz_x;
    duy += -cs2 * (dlnrho_dy + dss_dy / p.cp) + lorentz_y;

    dux += p.nu * (lap(UX, i, j, p) + ddivu_dx / 3.0 +
                   2.0 * (Sxx * dlnrho_dx + Sxy * dlnrho_dy));
    duy += p.nu * (lap(UY, i, j, p) + ddivu_dy / 3.0 +
                   2.0 * (Sxy * dlnrho_dx + Syy * dlnrho_dy));

    dux += z * (ddivu_dx + dv * dlnrho_dx) + dv * dzeta_dx;
    duy += z * (ddivu_dy + dv * dlnrho_dy) + dv * dzeta_dy;

    // induction, resistive gauge
    const real daa = (ux * By - uy * Bx) + p.eta * lap_aa;

    // entropy
    // order: eta*(Jz^2)/rho and zeta*(divu^2), squares first, as NumPy's **2
    real dss = -(ux * dss_dx + uy * dss_dy);
    dss += (2.0 * p.nu * S2 + p.eta * (Jz * Jz) / rho + z * (dv * dv)) / TT;
    dss += p.chi * lap(SS, i, j, p);

    out[ILNRHO] = dlnrho;
    out[IUX] = dux;
    out[IUY] = duy;
    out[IAA] = daa;
    out[ISS] = dss;
}

// --------------------------------------------------------------------------
// time step
// --------------------------------------------------------------------------

// advective plus fast magnetosonic speed at one point
HD real speed_point(const real* f, int i, int j, const Params& p) {
    const int n = p.nx * p.ny;
    const int k = j * p.nx + i;
    const real lnrho = f[ILNRHO * n + k];
    const real ux = f[IUX * n + k], uy = f[IUY * n + k];
    const real ss = f[ISS * n + k];
    const real* AA = f + IAA * n;

    const real rho = exp(lnrho);
    const real cs2 = sound_speed2(lnrho, ss, p);
    const real Bx = ddy(AA, i, j, p);
    const real By = -ddx(AA, i, j, p);
    const real va2 = (Bx * Bx + By * By) / rho;
    return sqrt(ux * ux + uy * uy) + sqrt(cs2 + va2);
}

// host side: turn the two global maxima into a step size
inline real dt_from_maxima(real speed_max, real zeta_max, const Params& p) {
    const real h = p.dx < p.dy ? p.dx : p.dy;
    const real dt_adv = p.cdt * h / maxr(speed_max, 1e-12);
    real diff = p.nu;
    diff = maxr(diff, p.eta);
    diff = maxr(diff, p.chi);
    diff = maxr(diff, zeta_max);
    const real dt_diff = p.cdtv * h * h / maxr(diff, 1e-12);
    return dt_adv < dt_diff ? dt_adv : dt_diff;
}
