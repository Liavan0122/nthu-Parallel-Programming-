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
#include <pthread.h>
#include <immintrin.h> // for AVX512
#include <math.h>
#include <sys/time.h>  // for time test

// Global variables and mutex
long long current_row = 0; // Global row index for dynamic allocation
pthread_mutex_t mutex;

// Thread data structure
typedef struct {
    int iters;
    int width;
    int height;
    double left;
    double right;
    double lower;
    double upper;
    int* buffer;
} thread_data_t;

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

void* mandelbrot_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int iters = data->iters;
    int width = data->width;
    int height = data->height;
    double left = data->left;
    double right = data->right;
    double lower = data->lower;
    double upper = data->upper;
    int* buffer = data->buffer;
    
    while (true) {
        // Dynamic row allocation
        int row;

        // Lock to get the current row
        pthread_mutex_lock(&mutex);
        if (current_row >= height) {
            pthread_mutex_unlock(&mutex);
            break; // Exit if all rows are processed
        }
        row = current_row;
        current_row++;
        pthread_mutex_unlock(&mutex);

        // Compute Mandelbrot set for the allocated row
        double y0 = row * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; i += 8) {
            // Load initial x0 values
            __m512d x0 = _mm512_set_pd(
                (i+7) * ((right - left) / width) + left,
                (i+6) * ((right - left) / width) + left,
                (i+5) * ((right - left) / width) + left,
                (i+4) * ((right - left) / width) + left,
                (i+3) * ((right - left) / width) + left,
                (i+2) * ((right - left) / width) + left,
                (i+1) * ((right - left) / width) + left,
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
            int buffer_offset = row * width + i;
            int repeat_vals[8];
            _mm512_storeu_si512((__m512i*)repeat_vals, repeats);
            for (int k = 0; k < 8 && (i + k) < width; ++k) {
                buffer[buffer_offset + k] = repeat_vals[k];
            }
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);

    /* create threads */
    int num_threads = 48; // Use 96 threads
    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];

    pthread_mutex_init(&mutex, 0);

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].iters = iters;
        thread_data[i].width = width;
        thread_data[i].height = height;
        thread_data[i].left = left;
        thread_data[i].right = right;
        thread_data[i].lower = lower;
        thread_data[i].upper = upper;
        thread_data[i].buffer = image;
        pthread_create(&threads[i], NULL, mandelbrot_thread, (void*)&thread_data[i]);
    }

    /* join threads */
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
    // End timing
    gettimeofday(&end, NULL);

    // Calculate elapsed time
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Execution time: %f seconds\n", elapsed_time);
    
    pthread_mutex_destroy(&mutex);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    return 0;
}
