// backend_cuda.cu
// ===============
// Stage 2 of the port: correct first, fast later.
//
// Global memory only, one thread per grid point, one kernel per sweep. Every
// kernel body is a call into mhd_core.h, the same functions the CPU backend
// loops over, so the physics is not duplicated anywhere. What lives here is
// only what is specific to the GPU: memory, launch shapes, and the reduction
// for the time step.
//
// Per RK stage:
//   k_divu -> k_shock_max -> k_smooth_y -> k_zeta    (the shock viscosity)
//   k_rhs                                             (df = alpha*df + dt*rhs)
//   k_update                                          (f += beta*df)
// Per step, before the stages:
//   aux sweeps + k_maxima, then the block maxima go to the host (2 doubles
//   per block) to set dt.
//
// Written to stay hipify-clean: no warp intrinsics, no cooperative groups,
// static shared memory only, and the runtime API calls all have HIP twins.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "backend.h"
#include "mhd_core.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d\n  %s\n  %s\n", __FILE__, \
                         __LINE__, #call, cudaGetErrorString(err_));            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// With -DDEBUG_SYNC every launch is followed by a device-wide sync, so a fault
// is reported at the kernel that caused it rather than at the next blocking
// call. Slow; for debugging only. (It is also what the CPU emulation test in
// the README needs, see there.)
#ifdef DEBUG_SYNC
#define CUDA_CHECK_LAUNCH()                  \
    do {                                     \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
#else
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
#endif

// 2D kernels: 32 x 8 = 256 threads, x fastest so a warp reads a contiguous
// piece of a row
static const int BX = 32;
static const int BY = 8;
// 1D kernels and the reduction
static const int NT = 256;

// --------------------------------------------------------------------------
// kernels
// --------------------------------------------------------------------------

__global__ void k_divu(const real* f, real* divu, Params p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= p.nx || j >= p.ny) return;
    divu[j * p.nx + i] = divu_point(f, i, j, p);
}

__global__ void k_shock_max(const real* divu, real* q, Params p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= p.nx || j >= p.ny) return;
    q[j * p.nx + i] = shock_max_point(divu, i, j, p);
}

__global__ void k_smooth_y(const real* q, real* qy, Params p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= p.nx || j >= p.ny) return;
    qy[j * p.nx + i] = smooth_y_point(q, i, j, p);
}

__global__ void k_zeta(const real* qy, real* zeta, Params p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= p.nx || j >= p.ny) return;
    zeta[j * p.nx + i] = zeta_point(qy, i, j, p);
}

__global__ void k_rhs(const real* f, const real* divu, const real* zeta, real* df,
                      real alpha, real dt, Params p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= p.nx || j >= p.ny) return;

    real r[NVAR];
    rhs_point(f, divu, zeta, i, j, p, r);

    const int n = p.nx * p.ny;
    const int k = j * p.nx + i;
    for (int v = 0; v < NVAR; ++v) df[v * n + k] = df[v * n + k] * alpha + dt * r[v];
}

// f += beta*df over all NVAR fields at once, as a flat 1D array
__global__ void k_update(real* f, const real* df, real beta, int ntot) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < ntot) f[k] = f[k] + beta * df[k];
}

// Per-block maxima of the signal speed and of zeta. Block b writes
// partial[2b] and partial[2b + 1]; the host takes the max over blocks.
// Both quantities are non-negative, so 0 is a safe value for the padding
// threads of the last block.
__global__ void k_maxima(const real* f, const real* zeta, real* partial, Params p) {
    __shared__ real s_speed[NT];
    __shared__ real s_zeta[NT];

    const int t = threadIdx.x;
    const int k = blockIdx.x * blockDim.x + t;
    const int n = p.nx * p.ny;

    real sp = 0.0, zz = 0.0;
    if (k < n) {
        sp = speed_point(f, k % p.nx, k / p.nx, p);
        zz = zeta[k];
    }
    s_speed[t] = sp;
    s_zeta[t] = zz;
    __syncthreads();

    for (int s = NT / 2; s > 0; s >>= 1) {
        if (t < s) {
            s_speed[t] = maxr(s_speed[t], s_speed[t + s]);
            s_zeta[t] = maxr(s_zeta[t], s_zeta[t + s]);
        }
        __syncthreads();
    }
    if (t == 0) {
        partial[2 * blockIdx.x] = s_speed[0];
        partial[2 * blockIdx.x + 1] = s_zeta[0];
    }
}

// --------------------------------------------------------------------------
// backend
// --------------------------------------------------------------------------

class CudaBackend : public Backend {
    Params p;
    int n;
    dim3 block2, grid2;
    int grid1_all, grid1_pts;
    std::string device;

    real *d_f = nullptr, *d_df = nullptr;
    real *d_divu = nullptr, *d_q = nullptr, *d_qy = nullptr, *d_zeta = nullptr;
    real* d_partial = nullptr;
    std::vector<real> h_partial;

    void aux() {
        k_divu<<<grid2, block2>>>(d_f, d_divu, p);
        CUDA_CHECK_LAUNCH();
        k_shock_max<<<grid2, block2>>>(d_divu, d_q, p);
        CUDA_CHECK_LAUNCH();
        k_smooth_y<<<grid2, block2>>>(d_q, d_qy, p);
        CUDA_CHECK_LAUNCH();
        k_zeta<<<grid2, block2>>>(d_qy, d_zeta, p);
        CUDA_CHECK_LAUNCH();
    }

public:
    explicit CudaBackend(const Params& p_) : p(p_), n(p_.nx * p_.ny) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        device = std::string("cuda, ") + prop.name + ", sm_" + std::to_string(prop.major) +
                 std::to_string(prop.minor);

        block2 = dim3(BX, BY);
        grid2 = dim3((p.nx + BX - 1) / BX, (p.ny + BY - 1) / BY);
        grid1_all = (NVAR * n + NT - 1) / NT;
        grid1_pts = (n + NT - 1) / NT;

        const size_t fb = sizeof(real) * NVAR * n, sb = sizeof(real) * n;
        CUDA_CHECK(cudaMalloc(&d_f, fb));
        CUDA_CHECK(cudaMalloc(&d_df, fb));
        CUDA_CHECK(cudaMalloc(&d_divu, sb));
        CUDA_CHECK(cudaMalloc(&d_q, sb));
        CUDA_CHECK(cudaMalloc(&d_qy, sb));
        CUDA_CHECK(cudaMalloc(&d_zeta, sb));
        CUDA_CHECK(cudaMalloc(&d_partial, sizeof(real) * 2 * grid1_pts));
        // df must start at zero: stage 1 computes 0*df + dt*rhs, and 0*NaN
        // from uninitialised memory would be NaN
        CUDA_CHECK(cudaMemset(d_df, 0, fb));
        h_partial.resize(2 * grid1_pts);
    }

    ~CudaBackend() override {
        cudaFree(d_f);
        cudaFree(d_df);
        cudaFree(d_divu);
        cudaFree(d_q);
        cudaFree(d_qy);
        cudaFree(d_zeta);
        cudaFree(d_partial);
    }

    std::string name() const override { return device; }
    std::string tag() const override { return "gpu"; }

    void upload(const std::vector<real>& h) override {
        CUDA_CHECK(cudaMemcpy(d_f, h.data(), sizeof(real) * NVAR * n, cudaMemcpyHostToDevice));
    }

    void download(std::vector<real>& h) override {
        h.resize(NVAR * n);
        CUDA_CHECK(cudaMemcpy(h.data(), d_f, sizeof(real) * NVAR * n, cudaMemcpyDeviceToHost));
    }

    real timestep() override {
        aux();
        k_maxima<<<grid1_pts, NT>>>(d_f, d_zeta, d_partial, p);
        CUDA_CHECK_LAUNCH();
        // blocking copy, so it also surfaces any error from the kernels above
        CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, sizeof(real) * 2 * grid1_pts,
                              cudaMemcpyDeviceToHost));
        real smax = 0.0, zmax = 0.0;
        for (int b = 0; b < grid1_pts; ++b) {
            smax = maxr(smax, h_partial[2 * b]);
            zmax = maxr(zmax, h_partial[2 * b + 1]);
        }
        return dt_from_maxima(smax, zmax, p);
    }

    void rk3_step(real dt) override {
        for (int s = 0; s < 3; ++s) {
            aux();
            k_rhs<<<grid2, block2>>>(d_f, d_divu, d_zeta, d_df, RK_ALPHA[s], dt, p);
            CUDA_CHECK_LAUNCH();
            k_update<<<grid1_all, NT>>>(d_f, d_df, RK_BETA[s], NVAR * n);
            CUDA_CHECK_LAUNCH();
        }
    }

    void sync() override { CUDA_CHECK(cudaDeviceSynchronize()); }
};

Backend* make_backend(const Params& p) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr, "no CUDA device found\n");
        std::exit(1);
    }
    return new CudaBackend(p);
}
