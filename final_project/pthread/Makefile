CC = gcc
CXX = g++
CFLAGS = -lm -O3 -mavx512f -march=native -std=c++11
orange_pthread: CFLAGS += -pthread
orange_hybrid: CC = mpicc
orange_hybrid: CXX = mpicxx
orange_hybrid: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = orange_seq orange_pthread orange_hybrid rotting_oranges

.PHONY: all
all: $(TARGETS)

orange_seq: orange_seq.o
	$(CXX) -o $@ $^ $(LDLIBS)

orange_pthread: orange_pthread.o
	$(CXX) -o $@ $^ $(LDLIBS)

orange_hybrid: orange_hybrid.o
	$(CXX) -o $@ $^ $(LDLIBS)

rotting_oranges: rotting_oranges.o
	$(CXX) -o $@ $^ $(LDLIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
