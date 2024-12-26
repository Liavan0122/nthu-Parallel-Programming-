NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS = -lm
EXES = hw3-2-B16 hw3-2-B32 hw3-2-B64

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

hw3-2-B16: hw3-2-B16.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-B32: hw3-2-B32.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-B64: hw3-2-B64.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?


