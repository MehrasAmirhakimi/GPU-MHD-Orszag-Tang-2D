# GPU-MHD-Orszag-Tang-2D

A 2D compressible resistive MHD solver, written first in NumPy as a reference
and then ported to C++ and CUDA.

The point of the reference is not speed. It is to have a correct, readable
implementation whose output the GPU version can be diffed against field by
field, so that a disagreement in the port is unambiguously a bug in the port.

**Status:** stage 2 of 4 done. The CUDA port reproduces the NumPy reference
to round-off on a Tesla T4 (worst field difference 7.8e-14) and runs about
100 times faster than NumPy on the same machine. Next is optimisation. The
notebook repeats the whole check and the timings in one click, and the copy
in the repository is saved with the outputs of that run:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MehrasAmirhakimi/GPU-MHD-Orszag-Tang-2D/blob/main/notebooks/colab_gpu_port.ipynb)

## Equations

Primitive variables, the set the Pencil Code and Astaroth evolve:

| symbol | meaning |
| --- | --- |
| `lnrho` | logarithmic density |
| `ux, uy` | velocity |
| `aa` | z-component of the magnetic vector potential |
| `ss` | specific entropy |

d(lnrho)/dt = -u.grad(lnrho) - div(u)

du/dt = -u.grad(u) - cs2 grad(lnrho + ss/cp) + (J x B)/rho + F_visc

d(aa)/dt = (u x B)_z + eta lap(aa)

rho T ds/dt = 2 rho nu S:S + eta J^2 + zeta (div u)^2

with mu0 = 1 and the ideal-gas entropy equation of state

cs2 = cs0^2 exp(gamma ss/cp + (gamma - 1)(lnrho - lnrho0))

**Why the vector potential.** The field is `B = curl(aa e_z)`, so `div(B) = 0`
is satisfied by construction rather than enforced afterwards. There is no
divergence cleaning anywhere in the code. The measured `max|div B|` below is
round-off and stays there.

## Numerics

- 6th-order centred finite differences, periodic in both directions
- 3rd-order 2N-storage Runge-Kutta (Williamson), two arrays of storage
- constant shear viscosity, magnetic diffusivity and entropy diffusion
- Nordlund-type shock viscosity, switched on only where the flow compresses

Centred differences with explicit diffusion, rather than a Riemann solver.
That is the choice the Pencil Code family makes, and it is the one that ports
cleanly to a GPU: every operation is a local stencil with no branching.

## Validation

All numbers below are produced by `python test_convergence.py` and
`python run_ot.py`.

**Operator order.** Error against analytic derivatives of `sin(x)cos(2y)`:

| N | error d/dx | order | error lap | order |
| --- | --- | --- | --- | --- |
| 32 | 4.06e-07 | - | 2.57e-05 | - |
| 64 | 6.38e-09 | 5.99 | 4.08e-07 | 5.97 |
| 128 | 9.99e-11 | 6.00 | 6.41e-09 | 5.99 |
| 256 | 1.57e-12 | 5.99 | 1.01e-10 | 5.98 |

**Solver order.** Sound wave, self-convergence against a 512-point reference,
time step held fixed across resolutions so the spatial error is what is being
measured:

| N | L2 error | order |
| --- | --- | --- |
| 32 | 5.75e-12 | - |
| 64 | 9.03e-14 | 5.99 |
| 128 | 1.41e-15 | 6.00 |

The comparison is against a numerical reference, not the linear analytic
solution: the code is nonlinear, and at these amplitudes the nonlinear
correction is larger than the truncation error being measured.

**Orszag-Tang vortex, 256^2, to t = 0.5:**

```
energy drift   6.80e-06
mass drift     3.80e-09
max |div B|    3.72e-13
```

Run on to t = 2.0, through shock formation, the energy drift grows to 1.2e-02
as the shock viscosity does its work, while `max|div B|` stays at 3.8e-13.
Kinetic and magnetic energy lost matches thermal energy gained to about 0.2
per cent, so the dissipated energy is going where it should.

![Orszag-Tang](ot_pressure.png)

## The CUDA port

```
cuda/mhd_core.h        the physics, written once: stencils, equation of state,
                       right-hand side at one grid point
cuda/backend_cuda.cu   GPU memory, kernels, and the reduction for the time step
cuda/backend_cpu.cpp   the same point functions in plain loops (OpenMP)
cuda/main.cpp          Orszag-Tang driver; one command line for both backends
compare.py             field-by-field check against ot_state.npz
notebooks/             build, validate and time on a Colab GPU
```

Every kernel body is a call into `mhd_core.h`, the same functions the CPU
backend loops over, so the physics exists in one place. What `backend_cuda.cu`
adds is only what is specific to the GPU: device memory, launch shapes, and a
block reduction for the time step.

This is the deliberately naive version: global memory only, one thread per
grid point, one kernel per sweep. Per RK stage, four sweeps build the shock
viscosity (div u, 3x3 max, two smoothing passes), one computes the right-hand
side, one applies the update. Before each step, a reduction kernel writes
per-block maxima of the signal speed and of the shock viscosity, and the host
turns them into dt. Everything is float64.

The C++ keeps NumPy's order of floating-point operations wherever it matters,
so the two codes differ only by rounding, and `compare.py` can use a tight
tolerance.

**What has been checked.** Orszag-Tang, 256^2, to t = 0.5, worst field
difference against the NumPy state, normalised by the size of each field:

| run | worst difference |
| --- | --- |
| C++ CPU backend | 8.0e-14 |
| C++ CPU backend, fused multiply-add everywhere, as on a GPU | 9.6e-14 |
| CUDA backend, converted with `hipify-perl` and executed on the CPU by [HIP-CPU](https://github.com/ROCm/HIP-CPU) | 8.0e-14 |
| **CUDA backend on a Tesla T4** (Colab, CUDA 12.8, nvcc with fused multiply-add) | **7.8e-14** |

On the T4 the run takes the same 139 steps as NumPy, ends at the same final
dt, and reports the same energy drift (6.796e-06) and mass drift (3.797e-09).

The HIP-CPU row runs the actual kernels, launch configurations and reduction
from `backend_cuda.cu`, not a rewrite of them. `hipify-perl` converted the
file with no manual edits, which is also the evidence that it is
hipify-clean and would build for AMD GPUs.

That run needed `DEBUG_SYNC=1`, a synchronisation after every launch.
Without it, the emulator disagreed at 1e-1, although each kernel on its own
matched the CPU loops bit for bit. The cause is in HIP-CPU, not in the port:
its default stream can start a queued kernel before the previous one has
finished, and a real CUDA default stream never does. The T4 run above used
the normal build, without `DEBUG_SYNC`.

The device code compiles for `sm_75` (T4), `sm_80` (A100) and `sm_89` (L4).
With clang's CUDA front end and NVIDIA's `ptxas`, the right-hand-side kernel
uses 208 registers per thread on `sm_75` with no spills. That allows one
256-thread block per multiprocessor on a T4, which is one of the first things
the optimisation stage has to deal with. nvcc may allocate differently.

## Running it

Python reference:

```
pip install numpy matplotlib

python test_convergence.py          # the three checks above
python run_ot.py --nx 256 --tmax 0.5
```

`run_ot.py` writes `ot_pressure.png`, `ot_diagnostics.csv` and `ot_state.npz`
into the current directory. The `ot_state.npz` in the repository root is the
regression target for the port, so run `run_ot.py` from another directory
if you do not want to overwrite it.

C++ and CUDA:

```
cd cuda
make cpu && ./mhd2d_cpu --nx 256 --tmax 0.5
make gpu && ./mhd2d_gpu --nx 256 --tmax 0.5    # needs nvcc and an NVIDIA GPU
cd ..
python compare.py ot_state.npz cuda/ot_state_gpu.npy
```

`--steps N` runs a fixed number of steps instead of stopping at `--tmax`,
which is what the timing runs use. `make gpu ARCH=sm_75` names the card when
the build machine has none; `DEBUG_SYNC=1` reports a fault at the kernel that
caused it.

Or skip all of that and open the notebook in Colab.

## Performance

One Colab machine, one session, float64, from the notebook: a Tesla T4 and
two CPU cores.

| implementation | grid | us per point-step | Mpoint-steps/s | vs NumPy |
| --- | --- | --- | --- | --- |
| NumPy, one core | 256^2 | 2.19 | 0.5 | 1x |
| C++ CPU backend, OpenMP | 256^2 | 1.05 | 1.0 | 2x |
| C++ CPU backend, OpenMP | 512^2 | 1.35 | 0.7 | 2x |
| CUDA, Tesla T4 | 256^2 | 0.0335 | 29.9 | 65x |
| CUDA, Tesla T4 | 512^2 | 0.0206 | 48.4 | 106x |
| CUDA, Tesla T4 | 1024^2 | 0.0206 | 48.5 | 106x |
| CUDA, Tesla T4 | 2048^2 | 0.0207 | 48.3 | 106x |

The last column compares each run with the NumPy run at 256^2.

From 512^2 up, the GPU throughput is flat at about 48 million point-steps per
second, so something on the card is saturated. At 256^2 it is lower, most
likely because 65,536 points are too few to keep the card busy while each step
also waits for the time-step maxima to come back to the host. A T4 runs
float64 at 1/32 of its float32 rate, so whether the plateau is set by
arithmetic or by memory bandwidth is the first thing stage 3 has to measure,
before changing anything.

These numbers only compare within the session. On the 2-core development
machine the same C++ run takes 0.40 us per point-step and NumPy takes 1.46,
and even that machine gave NumPy 1.04 on another day. Shared cloud machines
drift by tens of per cent from one day to the next.

## Where this is going

1. ~~NumPy reference, validated~~
2. ~~CUDA port, global memory only, one kernel per sweep~~. Matches NumPy
   to 7.8e-14 on a Tesla T4.
3. Profile the T4 run to find whether float64 arithmetic or memory bandwidth
   sets the plateau. Then shared-memory tiling for the stencils, fused
   sweeps, lower register pressure in the right-hand side, and host
   transfers cut to the output cadence. Report achieved bandwidth and
   arithmetic rate against the card's peaks.
4. Timings before and after stage 3 on more than one card.

## What this does not do

Honest list, so nobody has to read the source to find out:

- 2D only, and `u_z = B_z = 0`. No out-of-plane components.
- Periodic boundaries only.
- Constant diffusivities plus shock viscosity. No hyperdiffusion yet, which
  is what a production run at high Reynolds number would want.
- Entropy diffusion is a plain Laplacian smoothing term, not a physical
  thermal conductivity.
- No Alfven wave convergence test, because a uniform background field has no
  periodic vector potential. Carrying the background separately from the
  evolved potential would fix this and is the obvious next test to add.
- No self-gravity, no rotation, no radiative transfer.
- The port is float64 throughout. Consumer and inference GPUs run float64 at
  a small fraction of their float32 rate (1/32 on a T4), so a float32 build is
  an obvious later option. It would need its own tolerance in `compare.py`.
- Setting dt copies the per-block maxima to the host every step, which ties
  host and device together once per step.
