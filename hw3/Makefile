CXX = g++
CXXFLAGS = -O3 -fopenmp -msse2 -msse4.1
LDLIBS = -pthread
CFLAGS = -O3 -fopenmp -msse2 -msse4.1
CFLAGSGPU = -lm -O3
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a
LDFLAGS = -lm
LDAMDFLAGS = -lgomp
XCFLAGS := -fopenmp
EXES = hw3-1 hw3-2 hw3-3 hw3-2-amd hw3-3-amd

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

hw3-1: hw3-1.cc
	mpicxx $(CXXFLAGS) -o $@ $^ $(LDLIBS)

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-amd: hw3-2.hip
	hipcc $(HIPCCFLAGS) $(LDAMDFLAGS) -o $@ $<

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(CFLAGSGPU) $(LDFLAGS) -Xcompiler="$(XCFLAGS)" -o $@ $?

hw3-3-amd: hw3-3.hip
	hipcc $(HIPCCFLAGS) $(LDAMDFLAGS) -o $@ $<
