#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define ull unsigned long long
#define min(a,b) ((a<b) ? (a): (b))

int main(int argc, char** argv)
{
    int rank, size;
    ull r = atoll(argv[1]);  // ASCII to Long Long
    ull k = atoll(argv[2]);
    ull pixels = 0;

    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ull unit = r / size, remain = r % size;
    ull start = rank * unit + min(rank, remain);  // (a < b) is true, 分別多餘的工作量
    ull length = unit + (rank < remain);
    ull end = start+ length;

    for(ull i = start; i < end; i++){
        ull current_pixel = ceil(sqrtl(r*r - i*i));
        pixels += current_pixel;
    }

    pixels %= k;
    ull total_pixels = 0;

    MPI_Reduce(&pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Finalize();

    if(rank == 0) printf("%llu\n", (4 * total_pixels) % k);

    return 0;
}
