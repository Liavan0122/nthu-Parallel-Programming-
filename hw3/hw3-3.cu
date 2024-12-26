#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

#define INF ((1 << 30) - 1)
// #define V 50010  // 2 ≤ V ≤ 50000 (Multi-GPU)
#define B 64     

void input(char *infile);
void output(char *outFileName);
__global__ void phase1(int r, int *d_Dist, int n);
__global__ void phase2(int r, int *d_Dist, int n);
__global__ void phase3(int r, int *d_Dist, int n, int start);

int *h_Dist;
int *d_Dist[2];
int n, m, n_original;
int ceil(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char *argv[]) {
    input(argv[1]);

    int round = ceil(n, B);
    int shared_memory_size = B * B * sizeof(int);
    dim3 block(B / 2, B / 2);

    #pragma omp parallel num_threads(2)
    {
        int id = omp_get_thread_num(); // Device ID
        cudaSetDevice(id);

        // Create a CUDA stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Allocate memory for d_Dist
        cudaMalloc(&d_Dist[id], n * n * sizeof(int));


        // Determine workload for each GPU
        int start = (round / 2) * id;
        int size_multi_gpu = (round / 2) + (round % 2) * id;

        // Copy data from host to device asynchronously
        cudaMemcpyAsync(d_Dist[id] + (start * B * n),
                        h_Dist + (start * B * n),
                        sizeof(int) * size_multi_gpu * B * n,
                        cudaMemcpyHostToDevice,
                        stream);


        // Synchronize to ensure data is copied before starting computation
        cudaStreamSynchronize(stream);
        #pragma omp barrier

        for (int r = 0; r < round; r++) {
            // Data transfer between GPUs using cudaMemcpyPeer
            if (id == 0 && r >= start + size_multi_gpu) {
                // GPU 0 needs data from GPU 1
                cudaMemcpyPeerAsync(
                    d_Dist[0] + (r * B * n),  // Target on GPU 0
                    0,                        // GPU 0
                    d_Dist[1] + (r * B * n),  // Source from GPU 1
                    1,                        // GPU 1
                    B * n * sizeof(int),      // Size of the data
                    stream                    // Stream
                );
            } else if (id == 1 && r < start) {
                // GPU 1 needs data from GPU 0
                cudaMemcpyPeerAsync(
                    d_Dist[1] + (r * B * n),  // Target on GPU 1
                    1,                        // GPU 1
                    d_Dist[0] + (r * B * n),  // Source from GPU 0
                    0,                        // GPU 0
                    B * n * sizeof(int),      // Size of the data
                    stream                    // Stream
                );
            }

            // Synchronize stream before launching kernels
            cudaStreamSynchronize(stream);
            #pragma omp barrier

            // Launch kernels in the stream
            phase1<<<1, block, shared_memory_size, stream>>>(r, d_Dist[id], n);
            phase2<<<dim3(2, round - 1), block, 2 * shared_memory_size, stream>>>(r, d_Dist[id], n);
            phase3<<<dim3(size_multi_gpu, round - 1), block, 2 * shared_memory_size, stream>>>(r, d_Dist[id], n, start);

            // Synchronize stream after kernel execution
            cudaStreamSynchronize(stream);
            #pragma omp barrier
        }

        // Copy data from device to host asynchronously
        cudaMemcpyAsync(h_Dist + (start * B * n),
                        d_Dist[id] + (start * B * n),
                        size_multi_gpu * n * B * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);

        // Synchronize to ensure data is copied before exiting
        cudaStreamSynchronize(stream);

        // Clean up
        cudaFree(d_Dist[id]);
        cudaStreamDestroy(stream);
    }

    output(argv[2]);
    return 0;
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
void output(char *outFileName){
    FILE *file = fopen(outFileName, "w");
    for(int i = 0; i < n_original; i++){
        fwrite(&h_Dist[i*n], sizeof(int), n_original, file);
    }
    fclose(file);
    cudaFreeHost(h_Dist);
}



__global__ void phase1(int r, int *d_Dist, int n) {
    extern __shared__ int shared_Dist[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int b_i = r * B;
    int b_j = r * B;

    // Copy data from global memory to shared memory
    shared_Dist[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_i + tx)];                                 
    shared_Dist[ty*B + (tx + B/2)] =  d_Dist[(b_i + ty) * n + (b_j + (tx + B / 2))];                   
    shared_Dist[(ty + B/2)*B + tx] =  d_Dist[(b_i + ty + B/2) * n + (b_i + tx)];                   
    shared_Dist[(ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_i + ty + B/2) * n + (b_j + (tx + B / 2))];     

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        __syncthreads();
        shared_Dist[ty * B + tx] = min(shared_Dist[ty * B + tx],   
                                       shared_Dist[ty * B + k] + shared_Dist[k * B + tx]);
        shared_Dist[ty * B + (tx + B / 2)] = min(shared_Dist[ty * B + (tx + B / 2)],
                                                 shared_Dist[ty * B + k] + shared_Dist[k * B + (tx + B / 2)]);
        shared_Dist[(ty + B / 2) * B + tx] = min(shared_Dist[(ty + B / 2) * B + tx],
                                                 shared_Dist[(ty + B / 2) * B + k] + shared_Dist[k * B + tx]);
        shared_Dist[(ty + B / 2) * B + (tx + B / 2)] = min(shared_Dist[(ty + B / 2) * B + (tx + B / 2)],
                                                           shared_Dist[(ty + B / 2) * B + k] + shared_Dist[k * B + (tx + B / 2)]);
    }

    // Copy data from shared memory to global memory
    d_Dist[(b_i + ty) * n + (b_j + tx)] = shared_Dist[ty * B + tx];
    d_Dist[(b_i + ty) * n + (b_j + tx + B/2)] = shared_Dist[ty * B + (tx + B / 2)];
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx)] = shared_Dist[(ty + B / 2) * B + tx];
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)] = shared_Dist[(ty + B / 2) * B + (tx + B / 2)];
}

__global__ void phase2(int r, int *d_Dist, int n) {
    extern __shared__ int shared_Dist[];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_offset = r * B;

    int b_i = (bx * r    +    (!bx) * (by + (by >= r))) * B;
    int b_j = (bx * (by + (by >= r))    +    (!bx)*r) * B;

    int val0 =  d_Dist[(b_i + ty) * n + (b_j + tx)];                
    int val1 =  d_Dist[(b_i + ty) * n + (b_j + tx + B/2)];          
    int val2 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx)];          
    int val3 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)];    

    shared_Dist[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_offset + tx)];                                 
    shared_Dist[ty*B + (tx + B/2)] =  d_Dist[(b_i + ty) * n + (b_offset + tx + B/2)];                   
    shared_Dist[(ty + B/2)*B + tx] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx)];                   
    shared_Dist[(ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx + B/2)];     

    shared_Dist[B*B + ty*B + tx] =  d_Dist[(b_offset + ty) * n + (b_j + tx)];                               
    shared_Dist[B*B + ty*B + (tx + B/2)] =  d_Dist[(b_offset + ty) * n + (b_j + tx + B/2)];                 
    shared_Dist[B*B + (ty + B/2)*B + tx] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx)];                 
    shared_Dist[B*B + (ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx + B/2)];   

    __syncthreads();
    #pragma unroll
    for (int k = 0; k < B; ++k) {
        val0 =  min(val0, shared_Dist[ty*B + k]+ shared_Dist[B*B + k*64 + tx]);                 
        val1 =  min(val1, shared_Dist[ty*B + k]+ shared_Dist[B*B + k*64 + (tx + 32)]);          
        val2 =  min(val2, shared_Dist[(ty + 32)*B + k]+ shared_Dist[B*B + k*64 + tx]);          
        val3 =  min(val3, shared_Dist[(ty + 32)*B + k]+ shared_Dist[B*B + k*64 + (tx + 32)]);   
    }

    // Copy data from shared memory to global memory
    d_Dist[(b_i + ty) * n + (b_j + tx)] = val0;              
    d_Dist[(b_i + ty) * n + (b_j + tx + B/2)] = val1;        
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx)] = val2 ;       
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)] = val3;  

}

__global__ void phase3(int r, int *d_Dist, int n, int start) {
    extern __shared__ int shared_Dist[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_i = (start + blockIdx.x) * B;
    int b_j = (blockIdx.y + (blockIdx.y >= r)) * B;
    int b_offset = r * B;

    int val0 =  d_Dist[(b_i + ty) * n + (b_j + tx)];                
    int val1 =  d_Dist[(b_i + ty) * n + (b_j + tx + B/2)];          
    int val2 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx)];          
    int val3 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)];    

    shared_Dist[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_offset + tx)];                                 
    shared_Dist[ty*B + (tx + B/2)] =  d_Dist[(b_i + ty) * n + (b_offset + tx + B/2)];                   
    shared_Dist[(ty + B/2)*B + tx] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx)];                   
    shared_Dist[(ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx + B/2)];     

    shared_Dist[B*B + ty*B + tx] =  d_Dist[(b_offset + ty) * n + (b_j + tx)];                        
    shared_Dist[B*B + ty*B + (tx + B/2)] =  d_Dist[(b_offset + ty) * n + (b_j + tx + B/2)];          
    shared_Dist[B*B + (ty + B/2)*B + tx] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx)];          
    shared_Dist[B*B + (ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx + B/2)];    
    __syncthreads();

 
    #pragma unroll
    for (int k = 0; k < B; ++k) {
        val0 =  min(val0, shared_Dist[ty*B + k]+ shared_Dist[B*B + k*64 + tx]);                 
        val1 =  min(val1, shared_Dist[ty*B + k]+ shared_Dist[B*B + k*64 + (tx + 32)]);         
        val2 =  min(val2, shared_Dist[(ty + 32)*B + k]+ shared_Dist[B*B + k*64 + tx]);          
        val3 =  min(val3, shared_Dist[(ty + 32)*B + k]+ shared_Dist[B*B + k*64 + (tx + 32)]);   
    }

    // Copy data from shared memory to global memory
    d_Dist[(b_i + ty) * n + (b_j + tx)] = val0;
    d_Dist[(b_i + ty) * n + (b_j + tx + B/2)] = val1;
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx)] = val2;
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)] = val3;

}
