#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define INF ((1 << 30) - 1)
#define V 40010  // 2 ≤ V ≤ 40000 (Single-GPU)
#define B 64    // 將 B 定義為完，設定為 64 或 32

void input(char *infile);
void output(char *outFileName);
void block_FW();
__global__ void phase1(int r, int *d_Dist, int n);
__global__ void phase2(int r, int *d_Dist, int n);
__global__ void phase3(int r, int *d_Dist, int n);

int *h_Dist;
int *d_Dist;
int n, m, n_original;

int main(int argc, char *argv[]) {
    input(argv[1]);

    // Allocate memory for d_Dist and copy data from host to device
    cudaMalloc(&d_Dist, n * n * sizeof(int));
    cudaMemcpy(d_Dist, h_Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

    block_FW();

    output(argv[2]);

    cudaFree(d_Dist);
    return 0;
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    n_original = n;

    // 調整 n，使其對齊到 B 的倍數
    n += B - ((n % B + B - 1) % B + 1);

    // Allocate pinned memory for h_Dist
    cudaMallocHost(&h_Dist, n * n * sizeof(int));

    // Initialize h_Dist with INF and 0 for diagonal elements
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i < n_original && j < n_original) {
                h_Dist[i * n + j] = (i == j) ? 0 : INF;
            } else {
                h_Dist[i * n + j] = INF;
            }
        }
    }

    // Batch read edges
    int *edges = (int *)malloc(m * 3 * sizeof(int));
    fread(edges, sizeof(int), m * 3, file);
    for (int i = 0; i < m; ++i) {
        int u = edges[i * 3];
        int v = edges[i * 3 + 1];
        int w = edges[i * 3 + 2];
        h_Dist[u * n + v] = w;
    }
    free(edges);

    // // Read edges
    // int pair[3];
    // for (int i = 0; i < m; ++i) {
    //     fread(pair, sizeof(int), 3, file);
    //     int idx = pair[0]*n pair[1];
    //     h_Dist[idx] = pair[2];
    // }

    // Print h_Dist for debugging
    // printf("h_Dist after input:\n");
    // for (int i = 0; i < n_original; i++) {
    //     for (int j = 0; j < n_original; j++) {
    //         if (h_Dist[i * n + j] == INF) {
    //             printf("INF ");
    //         } else {
    //             printf("%d ", h_Dist[i * n + j]);
    //         }
    //     }
    //     printf("\n");
    // }
    fclose(file);
}
void output(char *outFileName) {
    FILE *file = fopen(outFileName, "wb");

    // 先將 h_Dist 的內容寫入到一個暫存緩衝區中，再一次性將該緩衝區的所有內容寫入文件，這樣可以大幅減少 I/O 的次數，提高輸出速度。
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

    // 差別很大，對大資料來說差了快一半的時間
    // Write results to file
    // for (int i = 0; i < n_original; i++) {
    //     for (int j = 0; j < n_original; j++) {
    //         int dist_value = h_Dist[i * n + j];


    //         // 檢查是否無法到達，若無法到達則輸出為 1073741823
    //         if (dist_value >= INF) {
    //             dist_value = 1073741823;
    //         }
    //         // printf("%d\t", dist_value);

    //         fwrite(&dist_value, sizeof(int), 1, file);
    //     }
    //     // printf("\n");
    // }
    // 批次寫入
    fwrite(output_buffer, sizeof(int), n_original * n_original, file);

    fclose(file);
    free(output_buffer);
    cudaFreeHost(h_Dist);
}


void block_FW() {
    int round = ceil(n, B);
    int shared_memory_size = B * B * sizeof(int);

    dim3 block(B / 2, B / 2);  // 由於我們在 kernel 中使用了 (tx + B/2)，所以 block 大小為 (B/2, B/2)

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
    extern __shared__ int shared_Dist[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // b_i 和 b_j 的目的是計算當前 Block 的偏移位置，以便確定每個 Block 在矩陣中的起始點。
    int b_i = r * B;
    int b_j = r * B;


    // 4 個 Threads Block 並不會同時載入全部共享記憶體，而是分批次進行。
    // ┌─────────────┌──────────────┌ tx coordinate
    // │             │              │
    // │ Block 1     │ Block 2      │
    // │ (Upper Left)│ (Upper Right)│
    // │             │              │
    // ├────────────────────────────├
    // │             │              │
    // │ Block 3     │ Block 4      │
    // │ (Lower Left)│ (Lower Right)│
    // │             │              │
    // └─────────────└──────────────└
    // ty coordinate

    // └─────────────└──────────────└ : B =64
    // └───────────── B/2 = 32

    // Copy data from global memory to shared memory
    shared_Dist[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_i + tx)];                                     //Block upper left
    shared_Dist[ty*B + (tx + B/2)] =  d_Dist[(b_i + ty) * n + (b_j + (tx + B / 2))];                   //Block upper right
    shared_Dist[(ty + B/2)*B + tx] =  d_Dist[(b_i + ty + B/2) * n + (b_i + tx)];                       //Block upper left
    shared_Dist[(ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_i + ty + B/2) * n + (b_j + (tx + B / 2))];     //Block upper left
    __syncthreads();

    // Unrolled for loop
    // 在每次 k 的更新之前，都會使用 __syncthreads() 來確保所有線程完成當前步驟的計算，然後再進行下一步。
    // 這是因為 Floyd-Warshall 算法的更新需要用到中間節點 k 的距離，而這些距離可能在同一個 for 迴圈內被更新。
    // 如果將 __syncthreads() 放在結尾，可能會導致某些執行緒在未更新完畢時就開始下一個 k 值的計算，從而讀取到未完成更新的值，導致結果錯誤。
    #pragma unroll
    for (int k = 0; k < B; ++k) {
        __syncthreads();
        shared_Dist[ty * B + tx] = min(shared_Dist[ty * B + tx],   // 假設0->1的w，去判斷 0->k->1會不會更短
                                       shared_Dist[ty * B + k] + shared_Dist[k * B + tx]);
        shared_Dist[ty * B + (tx + B / 2)] = min(shared_Dist[ty * B + (tx + B / 2)],
                                                 shared_Dist[ty * B + k] + shared_Dist[k * B + (tx + B / 2)]);
        shared_Dist[(ty + B / 2) * B + tx] = min(shared_Dist[(ty + B / 2) * B + tx],
                                                 shared_Dist[(ty + B / 2) * B + k] + shared_Dist[k * B + tx]);
        shared_Dist[(ty + B / 2) * B + (tx + B / 2)] = min(shared_Dist[(ty + B / 2) * B + (tx + B / 2)],
                                                           shared_Dist[(ty + B / 2) * B + k] + shared_Dist[k * B + (tx + B / 2)]);
    }

    // // Debugging
    // #pragma unroll
    // for (int k = 0; k < B; ++k) {
    //     __syncthreads();

    //     // 檢查是否為 INF，如果不是則打印
    //     if (shared_Dist[ty * B + tx] < INF &&
    //         shared_Dist[ty * B + k] < INF &&
    //         shared_Dist[k * B + tx] < INF) {

    //         printf("Iteration k=%d, thread (%d, %d): Before update - shared_Dist[%d] = %d, shared_Dist[ty * B + k] = %d, shared_Dist[k * B + tx] = %d\n",
    //             k, ty, tx, ty * B + tx, shared_Dist[ty * B + tx], shared_Dist[ty * B + k], shared_Dist[k * B + tx]);
    //     }

    //     // 更新 shared memory 中的最小距離
    //     if (shared_Dist[ty * B + k] < INF && shared_Dist[k * B + tx] < INF) {
    //         shared_Dist[ty * B + tx] = min(shared_Dist[ty * B + tx],
    //                                     shared_Dist[ty * B + k] + shared_Dist[k * B + tx]);
    //     }

    //     // 更新後再檢查並打印非 INF 的值
    //     if (shared_Dist[ty * B + tx] < INF) {
    //         printf("Iteration k=%d, thread (%d, %d): After update - shared_Dist[%d] = %d\n",
    //             k, ty, tx, ty * B + tx, shared_Dist[ty * B + tx]);
    //     }
    // }

    // Copy data from shared memory to global memory
    d_Dist[(b_i + ty) * n + (b_j + tx)] = shared_Dist[ty * B + tx];
    d_Dist[(b_i + ty) * n + (b_j + tx + B/2)] = shared_Dist[ty * B + (tx + B / 2)];
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx)] = shared_Dist[(ty + B / 2) * B + tx];
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)] = shared_Dist[(ty + B / 2) * B + (tx + B / 2)];


    // // Debugging: 打印每個回寫的結果
    // if (ty == 1 && tx == 0) { // 或選擇特定範圍內的執行緒
    //     printf("\nThread (%d, %d) writing back - d_Dist[%d][%d] = %d\n\n",
    //         ty, tx, b_i + ty, b_j + tx, d_Dist[(b_i + ty) * n + (b_j + tx)]);
    // }
    // //選擇一個執行緒來打印完整的 d_Dist 矩陣
    // if (ty == 0 && tx == 0) {
    //     printf("d_Dist matrix after updating phase1:\n");
    //     for (int i = 0; i < n; i++) {
    //         for (int j = 0; j < n; j++) {
    //             if (d_Dist[i * n + j] < INF) {
    //                 printf("%d ", d_Dist[i * n + j]);
    //             } else {
    //                 printf("INF ");
    //             }
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }


}

__global__ void phase2(int r, int *d_Dist, int n) {
    extern __shared__ int shared_Dist[];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_offset = r * B;


    /*
    ┌──────────────┬──────────────┬──────────────┬──────────────┐
    │  Block (0,0) │  Block (1,0) │  Block (2,0) │  Block (3,0) │
    ├──────────────┼──────────────┼──────────────┼──────────────┤   Block(blockIdx.x, blockIdx.y)
    │  Block (0,1) │  Block (1,1) │  Block (2,1) │  Block (3,1) │
    ├──────────────┼──────────────┼──────────────┼──────────────┤
    │  Block (0,2) │  Block (1,2) │  Block (2,2) │  Block (3,2) │
    ├──────────────┼──────────────┼──────────────┼──────────────┤
    │  Block (0,3) │  Block (1,3) │  Block (2,3) │  Block (3,3) │  64x64
    └──────────────┴──────────────┴──────────────┴──────────────┘
    */

    // b_i 和 b_j 的目的是計算當前 Block 的偏移位置，以便確定每個 Block 在矩陣中的起始點。
    int b_i = (bx * r    +    (!bx) * (by + (by >= r))) * B;
    int b_j = (bx * (by + (by >= r))    +    (!bx)*r) * B;

    // set current distance values in vals
    int val0 =  d_Dist[(b_i + ty) * n + (b_j + tx)];                //Block upper left
    int val1 =  d_Dist[(b_i + ty) * n + (b_j + tx + B/2)];          //Block upper right
    int val2 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx)];          //Block upper left
    int val3 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)];    //Block upper left

    // Copy data from global memory to shared memory
    shared_Dist[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_offset + tx)];                                 //Block upper left
    shared_Dist[ty*B + (tx + B/2)] =  d_Dist[(b_i + ty) * n + (b_offset + tx + B/2)];                   //Block upper right
    shared_Dist[(ty + B/2)*B + tx] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx)];                   //Block upper left
    shared_Dist[(ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx + B/2)];     //Block upper left
    // // Debugging: 打印每個回寫的結果
    // if (ty == 0 && tx == 1) { // 或選擇特定範圍內的執行緒
    //     printf("\nThread (%d, %d) writing back:\n", ty, tx);
    //     printf("val0 = d_Dist[%d] = %d\n", (b_i + ty) * n + (b_j + tx), val0);
    //     printf("val1 = d_Dist[%d] = %d\n", (b_i + ty) * n + (b_j + tx + B / 2), val1);
    //     printf("val2 = d_Dist[%d] = %d\n", (b_i + ty + B / 2) * n + (b_j + tx), val2);
    //     printf("val3 = d_Dist[%d] = %d\n", (b_i + ty + B / 2) * n + (b_j + tx + B / 2), val3);
    //     printf("b_i = %d, b_j = %d\n", b_i, b_j);
    // }

    // if (ty == 1 && tx == 0) { // 或選擇特定範圍內的執行緒
    //     printf("\nThread (%d, %d) writing back:\n", ty, tx);
    //     printf("val0 = d_Dist[%d] = %d\n", (b_i + ty) * n + (b_j + tx), val0);
    //     printf("val1 = d_Dist[%d] = %d\n", (b_i + ty) * n + (b_j + tx + B / 2), val1);
    //     printf("val2 = d_Dist[%d] = %d\n", (b_i + ty + B / 2) * n + (b_j + tx), val2);
    //     printf("val3 = d_Dist[%d] = %d\n", (b_i + ty + B / 2) * n + (b_j + tx + B / 2), val3);
    //     printf("b_i = %d, b_j = %d\n", b_i, b_j);
    // }
    // 將 pivot block（也就是 (r, r) 區塊）當成中介點，用來更新相同行或列的其他區塊的最短路徑。
    shared_Dist[B*B + ty*B + tx] =  d_Dist[(b_offset + ty) * n + (b_j + tx)];                               //Block upper left
    shared_Dist[B*B + ty*B + (tx + B/2)] =  d_Dist[(b_offset + ty) * n + (b_j + tx + B/2)];                 //Block upper right
    shared_Dist[B*B + (ty + B/2)*B + tx] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx)];                 //Block upper left
    shared_Dist[B*B + (ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx + B/2)];   //Block upper left

    __syncthreads();
    // Unrolled for loop
    #pragma unroll
    for (int k = 0; k < B; ++k) {
        val0 =  min(val0, shared_Dist[ty*B + k]+ shared_Dist[B*B + k*B + tx]);                 //Block upper left
        val1 =  min(val1, shared_Dist[ty*B + k]+ shared_Dist[B*B + k*B + (tx + B/2)]);          //Block upper right
        val2 =  min(val2, shared_Dist[(ty + B/2)*B + k]+ shared_Dist[B*B + k*B + tx]);          //Block upper left
        val3 =  min(val3, shared_Dist[(ty + B/2)*B + k]+ shared_Dist[B*B + k*B + (tx + B/2)]);   //Block upper left
    }

    // Copy data from shared memory to global memory
    d_Dist[(b_i + ty) * n + (b_j + tx)] = val0;              //Block upper left
    d_Dist[(b_i + ty) * n + (b_j + tx + B/2)] = val1;        //Block upper right
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx)] = val2 ;       //Block upper left
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)] = val3;  //Block upper left

    //     // Debugging: 打印每個回寫的結果
    // if (ty == 1 && tx == 0) { // 或選擇特定範圍內的執行緒
    //     printf("\nThread (%d, %d) writing back - d_Dist[%d][%d] = %d\n\n",
    //         ty, tx, b_i + ty, b_j + tx, d_Dist[(b_i + ty) * n + (b_j + tx)]);
    // }
}

__global__ void phase3(int r, int *d_Dist, int n) {
    extern __shared__ int shared_Dist[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_i = (blockIdx.x + (blockIdx.x >= r)) * B;
    int b_j = (blockIdx.y + (blockIdx.y >= r)) * B;
    int b_offset = r * B;

    // set current distance values in vals
    int val0 =  d_Dist[(b_i + ty) * n + (b_j + tx)];                //Block upper left
    int val1 =  d_Dist[(b_i + ty) * n + (b_j + tx + B/2)];          //Block upper right
    int val2 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx)];          //Block upper left
    int val3 =  d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)];    //Block upper left

    // Copy data from global memory to shared memory
    shared_Dist[ty*B + tx] =  d_Dist[(b_i + ty) * n + (b_offset + tx)];                                 //Block upper left
    shared_Dist[ty*B + (tx + B/2)] =  d_Dist[(b_i + ty) * n + (b_offset + tx + B/2)];                   //Block upper right
    shared_Dist[(ty + B/2)*B + tx] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx)];                   //Block upper left
    shared_Dist[(ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_i + ty + B/2) * n + (b_offset + tx + B/2)];     //Block upper left

    // s[4096...8191]（後半部分）與 phase2 的結果依賴性有關，因為 phase3 需要使用來自 phase2 的結果來進行進一步的計算。
    // 也是為何要 3 * shared_memory_size
    shared_Dist[B*B + ty*B + tx] =  d_Dist[(b_offset + ty) * n + (b_j + tx)];                        //Block upper left
    shared_Dist[B*B + ty*B + (tx + B/2)] =  d_Dist[(b_offset + ty) * n + (b_j + tx + B/2)];          //Block upper right
    shared_Dist[B*B + (ty + B/2)*B + tx] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx)];          //Block upper left
    shared_Dist[B*B + (ty + B/2)*B + (tx + B/2)] =  d_Dist[(b_offset + ty + B/2) * n + (b_j + tx + B/2)];    //Block upper left
    __syncthreads();

    // Unrolled for loop
    #pragma unroll
    for (int k = 0; k < B; ++k) {
        val0 =  min(val0, shared_Dist[ty*B + k]+ shared_Dist[B*B + k*B + tx]);                 //Block upper left
        val1 =  min(val1, shared_Dist[ty*B + k]+ shared_Dist[B*B + k*B + (tx + B/2)]);          //Block upper right
        val2 =  min(val2, shared_Dist[(ty + B/2)*B + k]+ shared_Dist[B*B + k*B + tx]);          //Block upper left
        val3 =  min(val3, shared_Dist[(ty + B/2)*B + k]+ shared_Dist[B*B + k*B + (tx + B/2)]);   //Block upper left
    }

    // Copy data from shared memory to global memory
    d_Dist[(b_i + ty) * n + (b_j + tx)] = val0;
    d_Dist[(b_i + ty) * n + (b_j + tx + B/2)] = val1;
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx)] = val2;
    d_Dist[(b_i + ty + B/2) * n + (b_j + tx + B/2)] = val3;

        // Debugging: 打印每個回寫的結果
    // if (ty == 1 && tx == 0) { // 或選擇特定範圍內的執行緒
    //     printf("\nThread (%d, %d) writing back - d_Dist[%d][%d] = %d\n\n",
    //         ty, tx, b_i + ty, b_j + tx, d_Dist[(b_i + ty) * n + (b_j + tx)]);
    // }

}
