#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#define min(a,b) (((a)<(b))?(a):(b))

unsigned long long r;
unsigned long long k;
unsigned long long rr;

int main(int argc, char* argv[]) {
    assert(argc == 3);  
    r = atoll(argv[1]); 
    k = atoll(argv[2]); 
    rr = r * r;         
    unsigned long long ans = 0;
	unsigned long long pixels = 0;


	int size, rank;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	unsigned long long unit = r / size;
	unsigned long long remain = r % size;
	unsigned long long offset = unit * rank + min(remain, rank);
	unsigned long long length = unit + (rank < remain);

	int size_omp = omp_get_max_threads();
	unsigned long long unit_omp = length/size_omp, remain_omp = length%size_omp; 

    #pragma omp parallel for num_threads(size_omp) schedule(static, 1) reduction(+:pixels)    
	for(unsigned long long i = 0; i < size_omp; i++) {
		unsigned long long offset_omp = offset + unit_omp * i + min(remain_omp, i);
		unsigned long long length_omp = unit_omp + (i < remain_omp);

		unsigned long long pixel_current = ceil(sqrtl(rr - offset * offset));

		for(unsigned long long j = offset_omp; j < offset_omp+length_omp; j++){
			unsigned long long prev = rr - j * j;
			while(prev <= (pixel_current - 1) * (pixel_current - 1)) pixel_current -= 4;
			while(prev > pixel_current * pixel_current) pixel_current++;
			pixels += pixel_current;
		}
		pixels %= k;
	}


	MPI_Reduce(&pixels, &ans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Finalize();

    if(rank == 0) printf("%llu\n", (4 * ans) % k);

    return 0;
}
