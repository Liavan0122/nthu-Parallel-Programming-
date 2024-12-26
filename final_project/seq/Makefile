CC = gcc
CXX = g++
CFLAGS = -lm -O3 -mavx512f -march=native -pipe
TARGETS = orange_seq compare_output

all: $(TARGETS)

orange_seq: orange_seq.o
	$(CXX) $(CFLAGS) -o $@ $^

compare_output: compare_output.o
	$(CXX) $(CFLAGS) -o $@ $^


clean:
	rm -f *.o $(TARGETS)
