// backend.h
// =========
// What the driver needs from a backend. The state lives wherever the backend
// keeps it (host vector or device memory); the driver only moves it across at
// start-up, at the diagnostics cadence, and at the end.

#pragma once

#include <string>
#include <vector>

#include "mhd_core.h"

class Backend {
public:
    virtual ~Backend() {}
    virtual std::string name() const = 0;  // for the log, e.g. device name
    virtual std::string tag() const = 0;   // short, for file names: cpu, gpu

    virtual void upload(const std::vector<real>& f) = 0;
    virtual void download(std::vector<real>& f) = 0;

    // CFL time step for the current state
    virtual real timestep() = 0;

    // one full RK3 step, in place
    virtual void rk3_step(real dt) = 0;

    // wait until all queued work is done (a no-op on the CPU)
    virtual void sync() {}
};

// each backend source file defines this
Backend* make_backend(const Params& p);
