CXX = g++
CFLAGS = -lm -O3 -mavx512f -march=native -pipe
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a

LDFLAGS = -lm
EXES = orange_single_gpu compare_output

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

orange_single_gpu: orange_single_gpu.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?


orange_single_gpu-amd: orange_single_gpu.hip
	hipcc $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

compare_output: compare_output.o
	$(CXX) $(CFLAGS) -o $@ $^
