// blocking factor = 64
// blocks = 40000 / 64 = 625 *625
// threads = 32*32

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define INF ((1 << 30) - 1)
// #define V 40010  // 2 ≤ V ≤ 40000 (Single-GPU)
#define B 64  

void input(char *infile);
void output(char *outFileName);
void block_FW();
__global__ void phase1(int r, int *d_Dist, int n);
__global__ void phase2(int r, int *d_Dist, int n);
__global__ void phase3(int r, int *d_Dist, int n);

int *h_Dist;
int *d_Dist;
int n, m, n_original;

int ceil(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char *argv[]) {
    input(argv[1]);

    block_FW();

    output(argv[2]);

    return 0;
}
void block_FW(){
    int round = ceil(n/B);      // perform ⌈𝑉/𝐵⌉ rounds
    int shared_memory_size = B * B * sizeof(int);
    dim3 block(32, 32);   

    // Allocate memory for d_Dist and copy data from host to device
    cudaMalloc(&d_Dist, n * n * sizeof(int));
    cudaMemcpy(d_Dist, h_Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

    for (int r = 0; r < round; r++) {
        phase1<<<1, block, shared_memory_size>>>(r, d_Dist, n);
        phase2<<<dim3(2, round - 1), block, 2 * shared_memory_size>>>(r, d_Dist, n);
        phase3<<<dim3(round - 1, round - 1), block, 2 * shared_memory_size>>>(r, d_Dist, n);
    }

    // Copy data from device to host
    cudaMemcpy(h_Dist, d_Dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);

}
__global__ void phase1(int r, int *d_Dist, int n) {
    extern __shared__ int shared_memory[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_x = r * B;    // 對d_dist來說 Block x y 的起始座標 
    int b_y = r * B;

    // Copy data from global memory to shared memory
    shared_memory[ty * B + tx] = d_Dist[(b_y + ty) * n + (b_x + tx)];                           // upper left
    shared_memory[ty * B + (B/2 + tx)] = d_Dist[(b_y + ty) * n + (b_x + B/2 + tx)];             // upper right
    shared_memory[(B/2 + ty) * B + tx] = d_Dist[(b_y + ty + B/2)*n + b_x + tx];                 // Bottom left
    shared_memory[(B/2 + ty) * B + (B/2 + tx)] = d_Dist[(b_y + ty + B/2)*n + (b_x + B/2 + tx)]; // Bottom right
    __syncthreads();


    // shared_memory[i][j]=min(shared_memory[i][j],shared_memory[i][k]+shared_memory[k][j])
    #pragma unroll
    for(int k = 0; k < B; ++k){
        shared_memory[ty * B + tx] = min(shared_memory[ty * B + tx],
                                         shared_memory[ty * B + k] + shared_memory[k * B + tx]);
        shared_memory[ty * B + (B/2 + tx)] = min(shared_memory[ty * B + (B/2 + tx)],
                                                 shared_memory[ty * B + k] + shared_memory[k*B + (tx + B / 2)]);
        shared_memory[(B/2 + ty) * B + tx] = min(shared_memory[(B/2 + ty) * B + tx], 
                                                 shared_memory[(ty + B / 2) * B + k] + shared_memory[k * B + tx]);
        shared_memory[(B/2 + ty) * B + (B/2 + tx)] = min(shared_memory[(B/2 + ty) * B + (B/2 + tx)],
                                                         shared_memory[(ty + B / 2) * B + k] + shared_memory[k*B + (tx + B / 2)]);                                
    }

    // Copy data from shared memory to global memory
    d_Dist[(b_y + ty) * n + (b_x + tx)] = shared_memory[ty * B + tx];
    d_Dist[(b_y + ty) * n + (b_x + B/2 + tx)] = shared_memory[ty * B + (B/2 + tx)];
    d_Dist[(b_y + ty + B/2)*n + b_x + tx] = shared_memory[(B/2 + ty) * B + tx];
    d_Dist[(b_y + ty + B/2)*n + (b_x + B/2 + tx)] = shared_memory[(B/2 + ty) * B + (B/2 + tx)];
}
__global__ void phase2(int r, int *d_Dist, int n){


    extern __shared__ int shared_memory[];
    int tx = threadIdx.x;
    int ty = threadIdx.y; 
    int bx = blockIdx.x;  // range [0,1]
    int by = blockIdx.y;
    int b_offset = r * B;

    // (!bx) => row or column major, bx = 0 do column major
    int b_i = (bx * r + (!bx) * (by + (by >= r)) ) * B;
    int b_j = (bx * (by + (by >= r)) + (!bx)*r) * B;

    // Copy data from pivot, registers faster
    int val0 =  d_Dist[(b_i + ty) * n + (b_j + tx)];                
    int val1 =  d_Dist[(b_i + ty) * n + (b_j + tx + B/2)];          
    int val2 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx)];          
    int val3 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)];    

    // Copy data from global memory to shared memory for row major
    shared_memory[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_offset + tx)];                                 
    shared_memory[ty*B + (tx + B/2)] =  d_Dist[(b_i + ty) * n + (b_offset + tx + B/2)];                   
    shared_memory[(ty + B/2)*B + tx] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx)];                   
    shared_memory[(ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx + B/2)];   

    // Copy data from global memory to shared memory for column major
    shared_memory[B*B + ty*B + tx] =  d_Dist[(b_offset + ty) * n + (b_j + tx)];                               
    shared_memory[B*B + ty*B + (tx + B/2)] =  d_Dist[(b_offset + ty) * n + (b_j + tx + B/2)];                 
    shared_memory[B*B + (ty + B/2)*B + tx] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx)];                 
    shared_memory[B*B + (ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx + B/2)];   
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        val0 =  min(val0, shared_memory[ty*B + k]+ shared_memory[B*B + k*B + tx]);                 //Block upper left
        val1 =  min(val1, shared_memory[ty*B + k]+ shared_memory[B*B + k*B + (tx + B/2)]);          //Block upper right
        val2 =  min(val2, shared_memory[(ty + B/2)*B + k]+ shared_memory[B*B + k*B + tx]);          //Block upper left
        val3 =  min(val3, shared_memory[(ty + B/2)*B + k]+ shared_memory[B*B + k*B + (tx + B/2)]);   //Block upper left
    }    

    // Copy data from shared memory to global memory
    d_Dist[(b_i + ty) * n + (b_j + tx)] = val0;              
    d_Dist[(b_i + ty) * n + (b_j + tx + B/2)] = val1;        
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx)] = val2 ;       
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)] = val3;  

}
__global__ void phase3(int r, int *d_Dist, int n){
    extern __shared__ int shared_memory[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_i = (blockIdx.x + (blockIdx.x >= r)) * B;
    int b_j = (blockIdx.y + (blockIdx.y >= r)) * B;
    int b_offset = r * B;    

    // Data from current Block, not pivot
    int val0 =  d_Dist[(b_i + ty) * n + (b_j + tx)];                
    int val1 =  d_Dist[(b_i + ty) * n + (b_j + tx + B/2)];          
    int val2 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx)];          
    int val3 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)]; 

    // Copy data from global memory to shared memory for row major from phase 2
    shared_memory[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_offset + tx)];                                
    shared_memory[ty*B + (tx + B/2)] =  d_Dist[(b_i + ty) * n + (b_offset + tx + B/2)];                  
    shared_memory[(ty + B/2)*B + tx] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx)];                  
    shared_memory[(ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx + B/2)];     
    
    // Copy data from global memory to shared memory for column major from phase 2
    shared_memory[B*B + ty*B + tx] =  d_Dist[(b_offset + ty) * n + (b_j + tx)];                      
    shared_memory[B*B + ty*B + (tx + B/2)] =  d_Dist[(b_offset + ty) * n + (b_j + tx + B/2)];       
    shared_memory[B*B + (ty + B/2)*B + tx] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx)];          
    shared_memory[B*B + (ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx + B/2)];    
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        val0 =  min(val0, shared_memory[ty*B + k]+ shared_memory[B*B + k*B + tx]);                
        val1 =  min(val1, shared_memory[ty*B + k]+ shared_memory[B*B + k*B + (tx + B/2)]);         
        val2 =  min(val2, shared_memory[(ty + B/2)*B + k]+ shared_memory[B*B + k*B + tx]);         
        val3 =  min(val3, shared_memory[(ty + B/2)*B + k]+ shared_memory[B*B + k*B + (tx + B/2)]);   
    }

    d_Dist[(b_i + ty) * n + (b_j + tx)] = val0;             
    d_Dist[(b_i + ty) * n + (b_j + tx + B/2)] = val1;       
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx)] = val2 ;       
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)] = val3;
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

    // output_buffer
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

