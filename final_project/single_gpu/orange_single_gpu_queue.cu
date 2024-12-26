#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>

#define ALIGN_FACTOR 64  // 調整矩陣大小的對齊因子（填充到64的倍數）

// 全域變數
int *h_grid;   // 存放對齊後的整數型矩陣
int *d_grid;
int row, column, result;
int max_row_col;

// 函數宣告
int** read_input(const char* filename, int* row, int* column);
void write_output(const char* filename);
double getTimeStamp();

__global__ void process_rotten_queue(int* d_grid, int* d_current_queue, int current_size,
                                     int* d_next_queue, int* d_next_size,
                                     int* d_fresh_count, int max_row_col);
__device__ void gpu_delay(int iterations);

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(0);

    double start_time = 0.0f, end_time = 0.0f, total_time = 0.0f;
    start_time = getTimeStamp();

    // 使用新的讀取函數
    int** grid = read_input(argv[1], &row, &column);

    // 對齊 max_row_col
    max_row_col = (row > column) ? row : column;
    max_row_col = (max_row_col + ALIGN_FACTOR - 1) / ALIGN_FACTOR * ALIGN_FACTOR;

    // 分配頁鎖定記憶體 (若過大造成問題，可改為 malloc)
    cudaError_t err = cudaMallocHost((void**)&h_grid, max_row_col * max_row_col * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host pinned memory: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 初始化 h_grid 為 0
    for (int i = 0; i < max_row_col * max_row_col; i++) {
        h_grid[i] = 0;
    }

    // 將原本的 grid 資料複製至 h_grid
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            h_grid[i * max_row_col + j] = grid[i][j];
        }
    }

    // 釋放原本 2D grid 記憶體
    for (int i = 0; i < row; i++) {
        free(grid[i]);
    }
    free(grid);

    // GPU 記憶體分配與拷貝
    cudaMalloc((void**)&d_grid, max_row_col * max_row_col * sizeof(int));
    cudaMemcpy(d_grid, h_grid, max_row_col * max_row_col * sizeof(int), cudaMemcpyHostToDevice);

    // BFS-like運算
    {
        int *d_current_queue, *d_next_queue;
        int *d_fresh_count, *d_next_size;

        int max_elements = max_row_col * max_row_col;
        cudaMalloc((void**)&d_current_queue, max_elements * sizeof(int));
        cudaMalloc((void**)&d_next_queue, max_elements * sizeof(int));
        cudaMalloc((void**)&d_fresh_count, sizeof(int));
        cudaMalloc((void**)&d_next_size, sizeof(int));

        int h_fresh_count = 0;
        int h_init_rotten_count = 0;
        int *h_init_rotten_queue = (int*)malloc(max_elements * sizeof(int));

        // 計算 fresh_count 並初始化 queue
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                int val = h_grid[i * max_row_col + j];
                if (val == 1) {
                    h_fresh_count++;
                } else if (val == 2) {
                    // 初始腐爛橘子加入 current_queue
                    h_init_rotten_queue[h_init_rotten_count++] = i * max_row_col + j;
                }
            }
        }

        cudaMemcpy(d_current_queue, h_init_rotten_queue, h_init_rotten_count * sizeof(int), cudaMemcpyHostToDevice);
        free(h_init_rotten_queue);

        int current_queue_size = h_init_rotten_count;
        int time = 0;
        int prev_fresh = h_fresh_count;

        while (h_fresh_count > 0) {

            int h_next_size = 0;
            cudaMemcpy(d_fresh_count, &h_fresh_count, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_next_size, &h_next_size, sizeof(int), cudaMemcpyHostToDevice);

            // 決定 kernel 大小
            int threadsPerBlock = 256;
            int blocks = (current_queue_size + threadsPerBlock - 1) / threadsPerBlock;

            // 執行 kernel
            process_rotten_queue<<<blocks, threadsPerBlock>>>(
                d_grid, d_current_queue, current_queue_size,
                d_next_queue, d_next_size, d_fresh_count, max_row_col
            );
            cudaDeviceSynchronize();

            // 取得新的 fresh_count 與 next_size
            cudaMemcpy(&h_fresh_count, d_fresh_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&current_queue_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);

            if (h_fresh_count == prev_fresh) {
                // 沒有任何進展
                result = -1;
                break;
            }

            prev_fresh = h_fresh_count;
            time++;

            // swap current_queue 和 next_queue
            {
                int* temp = d_current_queue;
                d_current_queue = d_next_queue;
                d_next_queue = temp;
            }
        }

        if (h_fresh_count == 0) {
            result = time;
        }

        // 將結果拷回 host
        cudaMemcpy(h_grid, d_grid, max_row_col * max_row_col * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_current_queue);
        cudaFree(d_next_queue);
        cudaFree(d_fresh_count);
        cudaFree(d_next_size);
    }

    // 輸出結果
    write_output(argv[2]);

    end_time = getTimeStamp();
    total_time = end_time - start_time;
    printf("Total Time: %.6f seconds.\t", total_time);

    cudaFreeHost(h_grid);
    cudaFree(d_grid);
    return 0;
}

// Kernel：處理上一輪腐爛的橘子，腐爛附近的新鮮橘子
__global__ void process_rotten_queue(int* d_grid, int* d_current_queue, int current_size,
                                     int* d_next_queue, int* d_next_size,
                                     int* d_fresh_count, int max_row_col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current_size) return;

    int pos = d_current_queue[idx];
    int x = pos % max_row_col;
    int y = pos / max_row_col;

    // 現在pos位置的橘子應該已經是腐爛(2)或剛腐爛(3->2)
    // 嘗試腐爛其附近的新鮮橘子(1->3)
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};

    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx < 0 || nx >= max_row_col || ny < 0 || ny >= max_row_col) continue;

        int neighbor_idx = ny * max_row_col + nx;

        // 使用atomicCAS確保只有一個thread能成功改變狀態
        int old_val = atomicCAS(&d_grid[neighbor_idx], 1, 3);
        if (old_val == 1) {
            atomicSub(d_fresh_count, 1);
            // 將這顆橘子加入下一輪的queue（下輪會將3轉成2並繼續擴散）
            int insert_pos = atomicAdd(d_next_size, 1);
            d_next_queue[insert_pos] = neighbor_idx;
            gpu_delay(10000000);
        }
    }

    // 備註：此處 3 -> 2 的轉換可延後到下一輪 iteration 開始前統一處理，
    // 或在本 kernel 中再跑一個 loop 處理所有 3。但為簡化流程，
    // 我們預設在下一輪開始前，可以再用一個 kernel 將所有3轉為2，或於下輪
    // 腐爛時檢查到3時直接當成已腐爛。
    // 為簡化，這裡假設 "3" 即表示下輪會使用該點當腐爛來源。
    // 實務上可再加一個 small kernel 在每輪結束將所有3改為2。
}
__device__ void gpu_delay(int iterations) {
    for (int i = 0; i < iterations; i++) {
        // 空迴圈，模擬延遲
        asm volatile("");
    }
}
// File handling functions
int** read_input(const char* filename, int* row, int* column) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Unable to open input file");
        exit(EXIT_FAILURE);
    }

    if (fscanf(file, "%d %d", row, column) != 2) {
        fprintf(stderr, "Failed to read row and column\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    int** grid = (int**)malloc((*row) * sizeof(int*));
    for (int i = 0; i < *row; i++) {
        grid[i] = (int*)malloc((*column) * sizeof(int));
    }

    char ch;
    int i = 0, j = 0;
    while ((ch = fgetc(file)) != EOF && i < *row) {
        if (ch >= '0' && ch <= '9') {
            grid[i][j] = ch - '0';
            j++;
            if (j == *column) {
                j = 0;
                i++;
            }
        }
    }

    fclose(file);
    return grid;
}

void write_output(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%d\n", result);
    fclose(file);
}

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}
