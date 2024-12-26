#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda_runtime.h>
#include <chrono>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8

using namespace std;
using namespace chrono;

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
    return (val >= lower && val < upper);
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ unsigned char shared_mem[32][32][3];  // 增加 shared memory 大小
    int shared_x = threadIdx.x + xBound;
    int shared_y = threadIdx.y + yBound;
    
    // 載入周圍像素區域到 shared memory
    for (int v = -yBound; v <= yBound; ++v) {
        for (int u = -xBound; u <= xBound; ++u) {
            int global_x = x + u;
            int global_y = y + v;
            if (bound_check(global_x, 0, width) && bound_check(global_y, 0, height)) {
                shared_mem[shared_y + v][shared_x + u][0] = s[channels * (width * global_y + global_x) + 0];
                shared_mem[shared_y + v][shared_x + u][1] = s[channels * (width * global_y + global_x) + 1];
                shared_mem[shared_y + v][shared_x + u][2] = s[channels * (width * global_y + global_x) + 2];
            } else {
                shared_mem[shared_y + v][shared_x + u][0] = 0;
                shared_mem[shared_y + v][shared_x + u][1] = 0;
                shared_mem[shared_y + v][shared_x + u][2] = 0;
            }
        }
    }
    __syncthreads();

    // 累積平方和計算
    float totalR = 0.0f, totalG = 0.0f, totalB = 0.0f;
    for (int i = 0; i < Z; ++i) {
        float valR = 0.0f, valG = 0.0f, valB = 0.0f;
        for (int v = -yBound; v <= yBound; ++v) {
            for (int u = -xBound; u <= xBound; ++u) {
                valR += shared_mem[shared_y + v][shared_x + u][2] * mask[i][v + yBound][u + xBound];
                valG += shared_mem[shared_y + v][shared_x + u][1] * mask[i][v + yBound][u + xBound];
                valB += shared_mem[shared_y + v][shared_x + u][0] * mask[i][v + yBound][u + xBound];
            }
        }
        totalR += valR * valR;
        totalG += valG * valG;
        totalB += valB * valB;
    }

    totalR = sqrt(totalR) / SCALE;
    totalG = sqrt(totalG) / SCALE;
    totalB = sqrt(totalB) / SCALE;

    if (x < width && y < height) {
        t[channels * (width * y + x) + 2] = (totalR > 255.0f) ? 255 : totalR;
        t[channels * (width * y + x) + 1] = (totalG > 255.0f) ? 255 : totalG;
        t[channels * (width * y + x) + 0] = (totalB > 255.0f) ? 255 : totalB;
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    // Read the image to src, and get height, width, channels
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // create cuda events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record start time
    cudaEventRecord(start);

    // Define block and grid sizes for optimal coalesced access
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Launch cuda kernel
    sobel<<<grid_size, block_size>>>(dsrc, ddst, height, width, channels);

    // record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);

    // Free memory
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
