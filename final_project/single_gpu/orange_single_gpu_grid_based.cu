#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>

#define BLOCK_SIZE 32
#define ALIGN_FACTOR 64  // 調整矩陣大小的對齊因子（填充到64的倍數）

// 全域變數
int *h_grid;  // 存放對齊後的整數型矩陣
int *d_grid;
int row, column, result;
int max_row_col;

// 函數宣告
int** read_input(const char* filename, int* row, int* column);
void write_output(const char* filename);
void orangesRottingCuda();
__global__ void process_core(int* d_grid, int* d_fresh_count, int max_row_col);
__global__ void process_halo(int* d_grid, int* d_fresh_count, int max_row_col);
double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

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
    max_row_col = row > column ? row : column;
    max_row_col = (max_row_col + ALIGN_FACTOR - 1) / ALIGN_FACTOR * ALIGN_FACTOR;

    // 分配頁鎖定記憶體 (若過大造成問題，可改為 malloc)
    cudaError_t err = cudaMallocHost((void**)&h_grid, max_row_col * max_row_col * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host pinned memory: %s\n", cudaGetErrorString(err));
        // 如果 pinned memory 配置失敗，可以嘗試一般 malloc
        // h_grid = (int*)malloc(max_row_col * max_row_col * sizeof(int));
        // if (h_grid == NULL) {
        //     fprintf(stderr, "Failed to allocate host memory\n");
        //     exit(EXIT_FAILURE);
        // }
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

    // 呼叫運算函數
    orangesRottingCuda();

    // 輸出結果
    write_output(argv[2]);

    end_time = getTimeStamp();
    total_time = end_time - start_time;
    printf("Total Time: %.2f seconds.\t", total_time);

    cudaFreeHost(h_grid);
    return 0;
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
    // 持續讀檔直到填滿 row*column 的資料
    // 假設檔案中只要有 '0'-'9' 數字，就代表一個元素
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

void orangesRottingCuda() {

    int time = 0; // 次數
    // 計算新鮮橘子數量
    int* d_fresh_count;
    int h_fresh_count = 0;
    cudaMalloc((void**)&d_fresh_count, sizeof(int));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            if (h_grid[i * max_row_col + j] == 1) {
                h_fresh_count++;
            }
        }
    }

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((max_row_col + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (max_row_col + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int shared_memory_size = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(int);  // Core kernel halo
    int halo_shared_mem = 4 * BLOCK_SIZE * sizeof(int); // halo kernel

    int prev_fresh_count = h_fresh_count;
    while (h_fresh_count > 0) {
        cudaMemcpy(d_fresh_count, &h_fresh_count, sizeof(int), cudaMemcpyHostToDevice);

        // 呼叫 process_core kernel
        process_core<<<gridDim, blockDim, shared_memory_size>>>(d_grid, d_fresh_count, max_row_col);
        cudaDeviceSynchronize();

        // 呼叫 process_halo kernel
        process_halo<<<gridDim, blockDim, halo_shared_mem>>>(d_grid, d_fresh_count, max_row_col);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_fresh_count, d_fresh_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_fresh_count == prev_fresh_count) {
            result = -1;
            break;
        }
        prev_fresh_count = h_fresh_count;
        time++;

        if (h_fresh_count == 0) {
            result = time;  // 記錄最終需要的時間
        }
    }

    cudaMemcpy(h_grid, d_grid, max_row_col * max_row_col * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_grid);
    cudaFree(d_fresh_count);
}


__global__ void process_core(int* d_grid, int* d_fresh_count, int max_row_col) {
    extern __shared__ int shared_mem[]; // 動態分配的共享記憶體
    int halo_size = BLOCK_SIZE + 2;     // 包含 Halo Cells 的共享記憶體大小

    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;

    int shared_idx = local_y * halo_size + local_x;

    // 載入到共享記憶體
    if (global_x < max_row_col && global_y < max_row_col) {
        shared_mem[shared_idx] = d_grid[global_y * max_row_col + global_x];
    } else {
        shared_mem[shared_idx] = 0;
    }

    __syncthreads();

    if (shared_mem[shared_idx] == 2) { // 腐爛橘子擴散
        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};
        for (int i = 0; i < 4; i++) {
            int nx = local_x + dx[i];
            int ny = local_y + dy[i];
            int n_idx = ny * halo_size + nx;
            if (nx >= 1 && nx <= BLOCK_SIZE && ny >= 1 && ny <= BLOCK_SIZE) {
                if (shared_mem[n_idx] == 1) {
                    shared_mem[n_idx] = 3; // 標記為即將腐爛
                }
            }
        }
    }

    __syncthreads();

    // 將3更新為2並減少 fresh_count
    if (shared_mem[shared_idx] == 3) {
        shared_mem[shared_idx] = 2;
        atomicSub(d_fresh_count, 1);
    }

    if (global_x < max_row_col && global_y < max_row_col) {
        d_grid[global_y * max_row_col + global_x] = shared_mem[shared_idx];
    }
}

__global__ void process_halo(int* d_grid, int* d_fresh_count, int max_row_col) {
    extern __shared__ int shared_mem[];
    int* top_halo = &shared_mem[0];                // 上邊界
    int* bottom_halo = &shared_mem[BLOCK_SIZE];    // 下邊界
    int* left_halo = &shared_mem[2 * BLOCK_SIZE];  // 左邊界
    int* right_halo = &shared_mem[3 * BLOCK_SIZE]; // 右邊界

    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // 載入 halo 資料
    if (threadIdx.y == 0 && global_y > 0) {
        top_halo[threadIdx.x] = d_grid[(global_y - 1) * max_row_col + global_x];
    }
    if (threadIdx.y == blockDim.y - 1 && global_y < max_row_col - 1) {
        bottom_halo[threadIdx.x] = d_grid[(global_y + 1) * max_row_col + global_x];
    }
    if (threadIdx.x == 0 && global_x > 0) {
        left_halo[threadIdx.y] = d_grid[global_y * max_row_col + (global_x - 1)];
    }
    if (threadIdx.x == blockDim.x - 1 && global_x < max_row_col - 1) {
        right_halo[threadIdx.y] = d_grid[global_y * max_row_col + (global_x + 1)];
    }

    __syncthreads();

    // 處理 halo 擴散
    // 左邊界
    if (threadIdx.x == 0 && global_x > 0) {
        if (left_halo[threadIdx.y] == 2 && d_grid[global_y * max_row_col + global_x] == 1) {
            d_grid[global_y * max_row_col + global_x] = 3;
        }
    }
    // 右邊界
    if (threadIdx.x == blockDim.x - 1 && global_x < max_row_col - 1) {
        if (right_halo[threadIdx.y] == 2 && d_grid[global_y * max_row_col + global_x] == 1) {
            d_grid[global_y * max_row_col + global_x] = 3;
        }
    }
    // 上邊界
    if (threadIdx.y == 0 && global_y > 0) {
        if (top_halo[threadIdx.x] == 2 && d_grid[global_y * max_row_col + global_x] == 1) {
            d_grid[global_y * max_row_col + global_x] = 3;
        }
    }
    // 下邊界
    if (threadIdx.y == blockDim.y - 1 && global_y < max_row_col - 1) {
        if (bottom_halo[threadIdx.x] == 2 && d_grid[global_y * max_row_col + global_x] == 1) {
            d_grid[global_y * max_row_col + global_x] = 3;
        }
    }

    __syncthreads();
}
