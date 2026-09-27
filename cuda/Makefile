# make cpu     C++ reference backend, any machine with g++
# make gpu     CUDA backend, needs nvcc
#
# ARCH picks the GPU architecture. "native" asks nvcc to detect the card in
# the machine; on a build host without a GPU, name it instead:
#   make gpu ARCH=sm_75     # T4 (Colab free tier)
#   make gpu ARCH=sm_80     # A100
#   make gpu ARCH=sm_89     # L4
#
# FMAD=0 turns off fused multiply-add contraction on the GPU. The results then
# follow the same rounding as the CPU build more closely; useful when chasing
# a difference, not needed otherwise.
#
# DEBUG_SYNC=1 synchronises after every kernel launch, so an error is reported
# at the kernel that caused it. Slow.

CXX      ?= g++
NVCC     ?= nvcc
ARCH     ?= native
FMAD     ?= 1
DEBUG_SYNC ?= 0

CXXFLAGS  = -O2 -std=c++17 -Wall -Wextra
NVFLAGS   = -O2 -std=c++17 -arch=$(ARCH) --fmad=$(if $(filter 0,$(FMAD)),false,true)
ifeq ($(DEBUG_SYNC),1)
NVFLAGS  += -DDEBUG_SYNC
endif

HEADERS   = mhd_core.h backend.h npy.h

.PHONY: all cpu gpu clean
all: cpu

cpu: mhd2d_cpu
gpu: mhd2d_gpu

mhd2d_cpu: main.cpp backend_cpu.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -fopenmp main.cpp backend_cpu.cpp -o $@

mhd2d_gpu: main.cpp backend_cuda.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) main.cpp backend_cuda.cu -o $@

clean:
	rm -f mhd2d_cpu mhd2d_gpu *.o ot_state_*.npy ot_diagnostics_*.csv
