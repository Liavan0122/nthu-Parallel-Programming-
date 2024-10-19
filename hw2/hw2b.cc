#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <immintrin.h> // for AVX512
#include <math.h>

// Global variables
long long current_row = 0;
long long height, width;
int iters;
double left, right, lower, upper;
int *image;
omp_lock_t lock;

// Function to write the result as a PNG image
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);



    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    long long total = (long long)width * height;

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        printf("%d cpus available\n", ncpus);
    }

    // /* Calculate the range for each process */
    // MPI_Barrier(MPI_COMM_WORLD);
    // double calc_range_start = MPI_Wtime();

    long long unit = height / size;
    long long remain = height % size;
    long long start_row = unit * rank + (rank < remain ? rank : remain);
    long long num_rows = unit + (rank < remain ? 1 : 0);

    // MPI_Barrier(MPI_COMM_WORLD);
    // double calc_range_end = MPI_Wtime();
    // if (rank == 0) {
    //     printf("Time taken to calculate the range for each process: %f seconds\n", calc_range_end - calc_range_start);
    // }

    /* allocate memory for image */
    image = (int*)malloc(num_rows * width * sizeof(int));
    assert(image);

    // /* Start timing */
    // MPI_Barrier(MPI_COMM_WORLD);
    // double calc_start = MPI_Wtime();

    /* mandelbrot set calculation */
    #pragma omp parallel num_threads(ncpus)
    {
        long long row;
        #pragma omp for schedule(dynamic) reduction(+:current_row)
        for (long long row = start_row; row < start_row + num_rows; ++row) {

            if (row < start_row || row >= start_row + num_rows) {
                continue;
            }

            double y0 = row * ((upper - lower) / height) + lower;
            for (int i = 0; i < width; i += 8) {
                // Load initial x0 values
                __m512d x0 = _mm512_set_pd(
                    (i + 7) * ((right - left) / width) + left,
                    (i + 6) * ((right - left) / width) + left,
                    (i + 5) * ((right - left) / width) + left,
                    (i + 4) * ((right - left) / width) + left,
                    (i + 3) * ((right - left) / width) + left,
                    (i + 2) * ((right - left) / width) + left,
                    (i + 1) * ((right - left) / width) + left,
                    i * ((right - left) / width) + left
                );

                __m512d y0_vec = _mm512_set1_pd(y0);
                __m512d x = _mm512_set1_pd(0.0);
                __m512d y = _mm512_set1_pd(0.0);
                __m512d length_squared = _mm512_set1_pd(0.0);
                __m512d two = _mm512_set1_pd(2.0);
                __m512d four = _mm512_set1_pd(4.0);

                __mmask8 active_mask = 0xFF;
                __m512i repeats = _mm512_set1_epi32(0);

                // Mandelbrot iteration
                for (int k = 0; k < iters; ++k) {
                    __m512d x2 = _mm512_mul_pd(x, x);
                    __m512d y2 = _mm512_mul_pd(y, y);
                    length_squared = _mm512_add_pd(x2, y2);

                    // Check if length_squared >= 4
                    __mmask8 mask = _mm512_cmp_pd_mask(length_squared, four, _CMP_LT_OQ);
                    active_mask &= mask;

                    if (active_mask == 0) {
                        break; // Exit if all pixels have escaped
                    }

                    // Mandelbrot iteration formula
                    __m512d xy = _mm512_mul_pd(x, y);
                    __m512d temp = _mm512_sub_pd(x2, y2);
                    x = _mm512_add_pd(temp, x0);
                    y = _mm512_add_pd(_mm512_mul_pd(xy, two), y0_vec);

                    // Update repeats using mask
                    repeats = _mm512_mask_add_epi32(repeats, mask, repeats, _mm512_set1_epi32(1));
                }

                // Store results back to the buffer
                int buffer_offset = (row - start_row) * width + i;
                int repeat_vals[8];
                _mm512_storeu_si512((__m512i*)repeat_vals, repeats);
                #pragma omp critical
                {
                    memcpy(&image[buffer_offset], repeat_vals, sizeof(int) * (width - i < 8 ? width - i : 8));
                }
            }
        }
    }

    // /* End timing Mandelbrot set calculation */
    // MPI_Barrier(MPI_COMM_WORLD);
    // double calc_end = MPI_Wtime();
    // if (rank == 0) {
    //     printf("Time taken for Mandelbrot set calculation: %f seconds\n", calc_end - calc_start);
    // }

    // /* Start timing for gather results */
    // MPI_Barrier(MPI_COMM_WORLD);
    // double gather_start = MPI_Wtime();

    /* gather results to rank 0 */
    if (rank == 0) {
        int* full_image = (int*)malloc(total * sizeof(int));
        assert(full_image);
        memcpy(full_image + start_row * width, image, num_rows * width * sizeof(int));

        for (int i = 1; i < size; i++) {
            long long recv_start_row = unit * i + (i < remain ? i : remain);
            long long recv_num_rows = unit + (i < remain ? 1 : 0);
            MPI_Recv(full_image + recv_start_row * width, recv_num_rows * width, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        write_png(filename, iters, width, height, full_image);
        free(full_image);
    } else {
        MPI_Send(image, num_rows * width, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    ///* End timing for gather results */
    // MPI_Barrier(MPI_COMM_WORLD);
    // double gather_end = MPI_Wtime();
    // if (rank == 0) {
    //     printf("Time taken to gather results to rank 0: %f seconds\n", gather_end - gather_start);
    // }

    MPI_Finalize();
    free(image);
    return 0;
}
