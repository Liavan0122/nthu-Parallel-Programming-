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

pthread_mutex_t mutex;

typedef struct {
    int thread_id;
    int iters;
    int width;
    int height;
    double left;
    double right;
    double lower;
    double upper;
    int* buffer;
    int start_row;
    int end_row;
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
    int start_row = data->start_row;
    int end_row = data->end_row;

    pthread_mutex_lock(&mutex);
    printf("Thread %d: processing rows from %d to %d\n", data->thread_id, data->start_row, data->end_row);
    pthread_mutex_unlock(&mutex);
    
    for (int j = start_row; j < end_row; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; i += 8) {
            // Load initial x0 for 8 pixels using AVX512
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

            int repeats[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int k = 0; k < iters; ++k) {
                __m512d x2 = _mm512_mul_pd(x, x);
                __m512d y2 = _mm512_mul_pd(y, y);
                length_squared = _mm512_add_pd(x2, y2);
                
                // Check if any length_squared >= 4
                __mmask8 mask = _mm512_cmp_pd_mask(length_squared, four, _CMP_LT_OQ);

                if (mask == 0) {
                    break;
                }

                // Mandelbrot iteration: temp = x^2 - y^2 + x0
                __m512d xy = _mm512_mul_pd(x, y);
                __m512d temp = _mm512_sub_pd(x2, y2);
                x = _mm512_add_pd(temp, x0);
                y = _mm512_add_pd(_mm512_mul_pd(xy, two), y0_vec);

                // Update the repeat count for each pixel
                for (int lane = 0; lane < 8; ++lane) {
                    if (repeats[lane] < iters) {
                        if (((double*)&length_squared)[lane] < 4.0) {
                            repeats[lane]++;
                        }
                    }
                }
            }

            // Store the results for these 8 pixels
            buffer[j * width + i] = repeats[0];
            buffer[j * width + i + 1] = repeats[1];
            buffer[j * width + i + 2] = repeats[2];
            buffer[j * width + i + 3] = repeats[3];
            buffer[j * width + i + 4] = repeats[4];
            buffer[j * width + i + 5] = repeats[5];
            buffer[j * width + i + 6] = repeats[6];
            buffer[j * width + i + 7] = repeats[7];
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

    /* create threads */
    pthread_t threads[ncpus];
    thread_data_t thread_data[ncpus];
    long rows_per_thread = height / ncpus;

    pthread_mutex_init(&mutex, 0);

    for (int i = 0; i < ncpus; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].iters = iters;
        thread_data[i].width = width;
        thread_data[i].height = height;
        thread_data[i].left = left;
        thread_data[i].right = right;
        thread_data[i].lower = lower;
        thread_data[i].upper = upper;
        thread_data[i].buffer = image;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == ncpus - 1) ? height : (i + 1) * rows_per_thread;
        pthread_create(&threads[i], NULL, mandelbrot_thread, (void*)&thread_data[i]);
    }

    /* join threads */
    for (int i = 0; i < ncpus; ++i) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    return 0;
}