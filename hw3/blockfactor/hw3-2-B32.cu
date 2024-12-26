// blocking factor = 32 or 16
// blocks = 40000 / 32 or 16 = 625 *625
// threads = 32*32 or 16*16

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

#define INF ((1 << 30) - 1)
// #define V 40010  // 2 ≤ V ≤ 40000 (Single-GPU)
#define B 32  // 32 or 16

void input(char *infile);
void output(char *outFileName);
void block_FW();
__global__ void phase1(int r, int *d_Dist, int n);
__global__ void phase2(int r, int *d_Dist, int n);
__global__ void phase3(int r, int *d_Dist, int n);
double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}
int *h_Dist;
int *d_Dist;
int n, m, n_original;

int ceil(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char *argv[]) {
    double start, end, io_time = 0.0, output_time = 0.0, compute_time = 0.0, input_time = 0.0,start_time = 0.0f, end_time = 0.0f, total_time = 0.0f;
    start_time = getTimeStamp();
    start = getTimeStamp();
    input(argv[1]);
    end = getTimeStamp();
    input_time += end - start;

    start = getTimeStamp();
    block_FW();
    end = getTimeStamp();
    compute_time += end - start;

    start = getTimeStamp();
    output(argv[2]);
    end = getTimeStamp();
    output_time += end - start;
    io_time = input_time+output_time;
    end_time = getTimeStamp();
    total_time = end_time-start_time;
    // 打印結果
    printf("I/O Input Time: %.2f seconds\n", input_time);
    printf("I/O output Time: %.2f seconds\n", output_time);
    printf("I/O total Time: %.2f seconds\n", io_time);
    printf("Compute Time: %.2f seconds\n", compute_time);
    printf("Total Time: %.2f seconds\n", total_time);       
    return 0;  

}
void block_FW() {
    int round = ceil(n / B);  // Perform ⌈V/B⌉ rounds
    int shared_memory_size = B * B * sizeof(int);  // 每個 Block 的共享記憶體大小
    dim3 block(B, B);  // 每個 Block 使用 B x B Threads

    // Allocate memory for d_Dist and copy data from host to device
    cudaMalloc(&d_Dist, n * n * sizeof(int));
    cudaMemcpy(d_Dist, h_Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

    for (int r = 0; r < round; r++) {
        phase1<<<1, block, shared_memory_size>>>(r, d_Dist, n);
        phase2<<<dim3(2, round-1), block, 2 * shared_memory_size>>>(r, d_Dist, n);
        phase3<<<dim3(round-1, round-1), block, 2 * shared_memory_size>>>(r, d_Dist, n);
    }

    // Copy data from device to host
    cudaMemcpy(h_Dist, d_Dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);
}

__global__ void phase1(int r, int *d_Dist, int n) {
    extern __shared__ int shared_memory[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pivot Block 起始位置
    int b_x = r * B;
    int b_y = r * B;

    // Load data to shared memory
    shared_memory[ty * B + tx] = d_Dist[(b_y + ty) * n + (b_x + tx)];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        __syncthreads();
        shared_memory[ty * B + tx] = min(shared_memory[ty * B + tx],
                                         shared_memory[ty * B + k] + shared_memory[k * B + tx]);
    }

    // Write back to global memory
    d_Dist[(b_y + ty) * n + (b_x + tx)] = shared_memory[ty * B + tx];
}

__global__ void phase2(int r, int *d_Dist, int n) {
    extern __shared__ int shared_memory[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;  // 行（row major）或列（column major）
    int by = blockIdx.y;  // 0: row major; 1: column major
    int b_offset = r * B;

    int b_i = (bx * r    +    (!bx) * (by + (by >= r))) * B;
    int b_j = (bx * (by + (by >= r))    +    (!bx)*r) * B;

    int val0 = d_Dist[(b_i+ty)*n+(b_j+tx)];
    shared_memory[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_offset + tx)];                                 
    shared_memory[B*B + ty*B + tx] =  d_Dist[(b_offset + ty) * n + (b_j + tx)];                               
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        val0 = min(val0,shared_memory[ty*B + k]+ shared_memory[B*B + k*B + tx]);
    }

    // Write back to global memory
    d_Dist[(b_i + ty) * n + (b_j + tx)] = val0;
}

__global__ void phase3(int r, int *d_Dist, int n) {
    extern __shared__ int shared_memory[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block 起始位置
    int b_i = (blockIdx.x + (blockIdx.x >= r)) * B;
    int b_j = (blockIdx.y + (blockIdx.y >= r)) * B;
    int b_offset = r * B;

    int val0 = d_Dist[(b_i+ty)*n+(b_j+tx)];
    shared_memory[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_offset + tx)];                                
    shared_memory[B*B + ty*B + tx] =  d_Dist[(b_offset + ty) * n + (b_j + tx)];  
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        val0 = min(val0, shared_memory[ty*B + k]+ shared_memory[B*B + k*B + tx]);
    }

    // Write back to global memory
    d_Dist[(b_i + ty) * n + (b_j + tx)] = val0;
}


void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    char *buffer = (char *)malloc(file_size);
    fread(buffer, 1, file_size, file);
    fclose(file);

    int *ptr = (int *)buffer;
    n = *ptr++;
    m = *ptr++;

    n_original = n;
    n += B - ((n % B + B - 1) % B + 1);

    // Allocate pinned memory for h_Dist
    cudaMallocHost(&h_Dist, n * n * sizeof(int));

    // Initialize h_Dist in parallel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i < n_original && j < n_original) {
                h_Dist[i * n + j] = (i == j) ? 0 : INF;
            } else {
                h_Dist[i * n + j] = INF;
            }
        }
    }

    // Parse edges directly from buffer
    int *edges = ptr;
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        int u = edges[i * 3];
        int v = edges[i * 3 + 1];
        int w = edges[i * 3 + 2];
        h_Dist[u * n + v] = w;
    }

    free(buffer);
}

void output(char *outFileName) {
    FILE *file = fopen(outFileName, "wb");

    int *output_buffer = (int *)malloc(n_original * n_original * sizeof(int));

    // 填充 output_buffer
    #pragma omp parallel for
    for (int i = 0; i < n_original; i++) {
        for (int j = 0; j < n_original; j++) {
            int dist_value = h_Dist[i * n + j];
            if (dist_value >= INF) {
                dist_value = 1073741823;
            }
            output_buffer[i * n_original + j] = dist_value;
        }
    }

    fwrite(output_buffer, sizeof(int), n_original * n_original, file);

    fclose(file);
    free(output_buffer);
    cudaFreeHost(h_Dist);
}

