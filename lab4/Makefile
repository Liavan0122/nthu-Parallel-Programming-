NVCC = nvcc 
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61

LDFLAGS = -lpng -lz

TARGETS = sobel

sobel: sobel_opt.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
