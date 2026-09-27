// backend_cpu.cpp
// ===============
// Plain loops over the shared point functions. This is the C++ reference: it
// shares every line of physics with the CUDA backend, so if this one matches
// NumPy and the CUDA one does not, the fault is in the GPU plumbing.

#include "backend.h"

#include <vector>

class CpuBackend : public Backend {
    Params p;
    int n;
    std::vector<real> f, df, divu, q, qy, zeta;

    template <class F>
    void for_each_point(F fn) {
#pragma omp parallel for schedule(static)
        for (int j = 0; j < p.ny; ++j)
            for (int i = 0; i < p.nx; ++i) fn(i, j, j * p.nx + i);
    }

    // div(u) and the shock viscosity, four sweeps
    void aux() {
        const real* F = f.data();
        for_each_point([&](int i, int j, int k) { divu[k] = divu_point(F, i, j, p); });
        for_each_point([&](int i, int j, int k) { q[k] = shock_max_point(divu.data(), i, j, p); });
        for_each_point([&](int i, int j, int k) { qy[k] = smooth_y_point(q.data(), i, j, p); });
        for_each_point([&](int i, int j, int k) { zeta[k] = zeta_point(qy.data(), i, j, p); });
    }

public:
    explicit CpuBackend(const Params& p_)
        : p(p_), n(p_.nx * p_.ny),
          f(NVAR * n), df(NVAR * n, 0.0), divu(n), q(n), qy(n), zeta(n) {}

    std::string name() const override {
#ifdef _OPENMP
        return "cpu (OpenMP)";
#else
        return "cpu (serial)";
#endif
    }
    std::string tag() const override { return "cpu"; }

    void upload(const std::vector<real>& h) override { f = h; }
    void download(std::vector<real>& h) override { h = f; }

    real timestep() override {
        aux();
        real smax = 0.0, zmax = 0.0;
        for (int j = 0; j < p.ny; ++j)
            for (int i = 0; i < p.nx; ++i) {
                smax = maxr(smax, speed_point(f.data(), i, j, p));
                zmax = maxr(zmax, zeta[j * p.nx + i]);
            }
        return dt_from_maxima(smax, zmax, p);
    }

    void rk3_step(real dt) override {
        for (int s = 0; s < 3; ++s) {
            const real alpha = RK_ALPHA[s], beta = RK_BETA[s];
            aux();
            const real* F = f.data();
            for_each_point([&](int i, int j, int k) {
                real r[NVAR];
                rhs_point(F, divu.data(), zeta.data(), i, j, p, r);
                for (int v = 0; v < NVAR; ++v)
                    df[v * n + k] = df[v * n + k] * alpha + dt * r[v];
            });
            // update only after the whole RHS is in, as NumPy does
            for (int k = 0; k < NVAR * n; ++k) f[k] = f[k] + beta * df[k];
        }
    }
};

Backend* make_backend(const Params& p) { return new CpuBackend(p); }
