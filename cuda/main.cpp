// main.cpp
// ========
// Orszag-Tang driver, the C++ counterpart of run_ot.py. Linked against one
// backend at build time: `make cpu` gives mhd2d_cpu, `make gpu` gives
// mhd2d_gpu. Same command line, same printout, same output files.
//
//   ./mhd2d_gpu --nx 256 --tmax 0.5          # validation run
//   ./mhd2d_gpu --nx 1024 --steps 100 --every 0   # timing run
//
// Writes ot_state_<tag>.npy (shape (5, ny, nx), float64) and
// ot_diagnostics_<tag>.csv. The .npy is what compare.py checks against the
// NumPy reference.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "mhd_core.h"
#include "npy.h"

// --------------------------------------------------------------------------
// initial condition, as init_orszag_tang in mhd2d.py
// --------------------------------------------------------------------------

static std::vector<real> init_orszag_tang(Params& p) {
    p.cs0 = 1.0;
    p.lnrho0 = std::log(p.gamma * p.gamma);
    finalize(p);

    const int n = p.nx * p.ny;
    std::vector<real> f(NVAR * n);
    for (int j = 0; j < p.ny; ++j)
        for (int i = 0; i < p.nx; ++i) {
            const int k = j * p.nx + i;
            const real x = i * p.dx, y = j * p.dy;  // vertex-centred grid
            f[ILNRHO * n + k] = p.lnrho0;
            f[IUX * n + k] = -std::sin(y);
            f[IUY * n + k] = std::sin(x);
            f[IAA * n + k] = std::cos(y) + 0.5 * std::cos(2.0 * x);
            f[ISS * n + k] = 0.0;
        }
    return f;
}

// --------------------------------------------------------------------------
// diagnostics, as diagnostics() in mhd2d.py. Host side, at output cadence.
// --------------------------------------------------------------------------

struct Diag {
    double rho_mean, rho_min, ekin, emag, ethm, etot, umax, divB_max, p_min, p_max;
};

static Diag diagnostics(const std::vector<real>& f, const Params& p) {
    const int n = p.nx * p.ny;
    const real* AA = f.data() + IAA * n;

    // B first as whole arrays, so div(B) can be taken of them the long way
    std::vector<real> Bx(n), By(n);
    for (int j = 0; j < p.ny; ++j)
        for (int i = 0; i < p.nx; ++i) {
            Bx[j * p.nx + i] = ddy(AA, i, j, p);
            By[j * p.nx + i] = -ddx(AA, i, j, p);
        }

    long double s_rho = 0, s_ekin = 0, s_emag = 0, s_ethm = 0, s_etot = 0;
    Diag d;
    d.rho_min = d.p_min = INFINITY;
    d.umax = d.divB_max = d.p_max = 0.0;

    for (int j = 0; j < p.ny; ++j)
        for (int i = 0; i < p.nx; ++i) {
            const int k = j * p.nx + i;
            const real lnrho = f[ILNRHO * n + k], ux = f[IUX * n + k],
                       uy = f[IUY * n + k], ss = f[ISS * n + k];
            const real rho = std::exp(lnrho);
            const real pr = rho * sound_speed2(lnrho, ss, p) / p.gamma;
            const real ek = 0.5 * rho * (ux * ux + uy * uy);
            const real em = 0.5 * (Bx[k] * Bx[k] + By[k] * By[k]);
            const real et = pr / (p.gamma - 1.0);
            const real divB = ddx(Bx.data(), i, j, p) + ddy(By.data(), i, j, p);

            s_rho += rho;
            s_ekin += ek;
            s_emag += em;
            s_ethm += et;
            s_etot += (ek + em) + et;
            d.rho_min = std::fmin(d.rho_min, rho);
            d.umax = std::fmax(d.umax, std::sqrt(ux * ux + uy * uy));
            d.divB_max = std::fmax(d.divB_max, std::fabs(divB));
            d.p_min = std::fmin(d.p_min, pr);
            d.p_max = std::fmax(d.p_max, pr);
        }
    d.rho_mean = (double)(s_rho / n);
    d.ekin = (double)(s_ekin / n);
    d.emag = (double)(s_emag / n);
    d.ethm = (double)(s_ethm / n);
    d.etot = (double)(s_etot / n);
    return d;
}

static bool all_finite(const std::vector<real>& f) {
    for (real v : f)
        if (!std::isfinite(v)) return false;
    return true;
}

// --------------------------------------------------------------------------
// command line
// --------------------------------------------------------------------------

struct Args {
    int nx = 256, ny = 0;
    double tmax = 0.5;
    long steps = 0;  // > 0: run exactly this many steps, ignore tmax
    double nu = 2.0e-3, eta = 2.0e-3, chi = 2.0e-3;
    int every = 20;  // 0: diagnostics only at the start and the end
    std::string out, csv;
    bool quiet = false;
};

static void usage() {
    std::printf(
        "options:\n"
        "  --nx N          grid points in x (default 256)\n"
        "  --ny N          grid points in y (default: same as nx)\n"
        "  --tmax T        end time (default 0.5)\n"
        "  --steps N       run exactly N steps instead, for timing\n"
        "  --nu, --eta, --chi   diffusivities (default 2e-3 each)\n"
        "  --every N       diagnostics every N steps, 0 = start and end only\n"
        "  --out FILE      final state, .npy (default ot_state_<backend>.npy)\n"
        "  --csv FILE      diagnostics (default ot_diagnostics_<backend>.csv)\n"
        "  --quiet         no per-interval printout\n");
}

static Args parse(int argc, char** argv) {
    Args a;
    for (int k = 1; k < argc; ++k) {
        const std::string s = argv[k];
        auto next = [&]() -> const char* {
            if (k + 1 >= argc) {
                std::fprintf(stderr, "missing value after %s\n", s.c_str());
                std::exit(2);
            }
            return argv[++k];
        };
        if (s == "--nx") a.nx = std::atoi(next());
        else if (s == "--ny") a.ny = std::atoi(next());
        else if (s == "--tmax") a.tmax = std::atof(next());
        else if (s == "--steps") a.steps = std::atol(next());
        else if (s == "--nu") a.nu = std::atof(next());
        else if (s == "--eta") a.eta = std::atof(next());
        else if (s == "--chi") a.chi = std::atof(next());
        else if (s == "--every") a.every = std::atoi(next());
        else if (s == "--out") a.out = next();
        else if (s == "--csv") a.csv = next();
        else if (s == "--quiet") a.quiet = true;
        else if (s == "-h" || s == "--help") { usage(); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown option %s\n", s.c_str());
            usage();
            std::exit(2);
        }
    }
    if (a.ny == 0) a.ny = a.nx;
    if (a.nx < 7 || a.ny < 7) {
        std::fprintf(stderr, "need at least 7 points per direction for the stencil\n");
        std::exit(2);
    }
    return a;
}

// --------------------------------------------------------------------------

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);

    Params p = default_params(a.nx, a.ny);
    p.nu = a.nu;
    p.eta = a.eta;
    p.chi = a.chi;
    std::vector<real> f = init_orszag_tang(p);

    std::unique_ptr<Backend> be(make_backend(p));
    const std::string out = a.out.empty() ? "ot_state_" + be->tag() + ".npy" : a.out;
    const std::string csv = a.csv.empty() ? "ot_diagnostics_" + be->tag() + ".csv" : a.csv;

    std::printf("grid %d x %d   nu = %g  eta = %g  chi = %g\n", p.nx, p.ny, p.nu, p.eta, p.chi);
    std::printf("backend %s\n", be->name().c_str());

    const Diag d0 = diagnostics(f, p);
    std::printf("t = 0.000  Etot = %.8f  divB = %.2e\n", d0.etot, d0.divB_max);

    be->upload(f);
    be->timestep();  // warm-up: first launch pays for module loading
    be->sync();

    FILE* fcsv = std::fopen(csv.c_str(), "w");
    if (fcsv) std::fprintf(fcsv, "t,dt,ekin,emag,ethm,etot,rho_min,divB_max\n");

    using clk = std::chrono::steady_clock;
    double step_seconds = 0.0;
    double t = 0.0;
    long step = 0;
    const bool fixed = a.steps > 0;

    for (;;) {
        if (fixed ? step >= a.steps : t >= a.tmax - 1e-12) break;

        const auto c0 = clk::now();
        real dt = be->timestep();
        if (!fixed && dt > a.tmax - t) dt = a.tmax - t;
        be->rk3_step(dt);
        be->sync();
        step_seconds += std::chrono::duration<double>(clk::now() - c0).count();

        t += dt;
        ++step;

        const bool last = fixed ? step >= a.steps : t >= a.tmax - 1e-12;
        if ((a.every > 0 && step % a.every == 0) || last) {
            be->download(f);
            if (!all_finite(f)) {
                std::fprintf(stderr, "non-finite values at step %ld, t = %.6f\n", step, t);
                return 1;
            }
            const Diag d = diagnostics(f, p);
            if (fcsv)
                std::fprintf(fcsv, "%.6f,%.6e,%.8f,%.8f,%.8f,%.8f,%.6f,%.3e\n", t, dt, d.ekin,
                             d.emag, d.ethm, d.etot, d.rho_min, d.divB_max);
            if (!a.quiet || last)
                std::printf("t = %.3f  dt = %.2e  Etot = %.8f  rho_min = %.4f  divB = %.2e\n", t,
                            dt, d.etot, d.rho_min, d.divB_max);
        }
    }
    if (fcsv) std::fclose(fcsv);

    const Diag d1 = diagnostics(f, p);
    const double pts = double(p.nx) * p.ny * step;
    std::printf("\n%ld steps in %.3f s  (%.4g us per point-step, %.1f Mpoint-steps/s)\n", step,
                step_seconds, 1e6 * step_seconds / pts, pts / step_seconds / 1e6);
    std::printf("energy drift  %.3e\n", std::fabs(d1.etot - d0.etot) / d0.etot);
    std::printf("mass drift    %.3e\n", std::fabs(d1.rho_mean - d0.rho_mean) / d0.rho_mean);
    std::printf("max |div B|   %.3e\n", d1.divB_max);
    std::printf("pressure      min %.4f   max %.4f\n", d1.p_min, d1.p_max);

    if (!write_npy(out, f, {NVAR, p.ny, p.nx})) {
        std::fprintf(stderr, "could not write %s\n", out.c_str());
        return 1;
    }
    std::printf("wrote %s, %s\n", out.c_str(), csv.c_str());
    return 0;
}
