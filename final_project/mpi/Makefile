CC = mpicc
CXX = mpicxx
CFLAGS = -lm -O3 -fopenmp -mavx512f -march=native -pipe
TARGETS = orange_hybrid2 compare_output

all: $(TARGETS)

orange_hybrid2: orange_hybrid2.o
	$(CXX) $(CFLAGS) -o $@ $^

compare_output: compare_output.o
	$(CXX) $(CFLAGS) -o $@ $^


clean:
	rm -f *.o $(TARGETS)
