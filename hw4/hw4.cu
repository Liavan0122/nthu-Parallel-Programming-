#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define NEG_INFINITY __int_as_float(0xff800000)

#define BLOCK_ROWS 32
#define BLOCK_COLS 32
#define TILE_SIZE_K 64 // 維度分塊大小
#define TILE_ROWS 4
#define TILE_COLS 4

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

__global__ void flash_attention_kernel(float *output, float* key, float* query, float* value, int seq_length, int dim, int batch_size)
{
    // block維度: blockIdx.x -> batch，blockIdx.y -> seq方向上的tile
    // gridDim.x = batch_size, gridDim.y = ceil(seq_length / BLOCK_ROWS)
    // blockDim: 將 (BLOCK_ROWS x BLOCK_COLS) / (TILE_ROWS x TILE_COLS) threads 做為計算單元
    // BLOCK_ROWS=32, BLOCK_COLS=32, TILE_ROWS=2, TILE_COLS=2
    // threads_per_block = (32*32)/(2*2)=256

    int batch_id = blockIdx.x;
    int tile_row_id = blockIdx.y; // tile_row_id對應seq_length方向的分塊
    int batch_offset = batch_id * seq_length * dim;

    // 計算該block對應的序列起始位置 (row方向)
    int row_start = tile_row_id * BLOCK_ROWS;
    int row_end = min(row_start + BLOCK_ROWS, seq_length);
    int actual_rows = row_end - row_start;

    float scale = 1.0f / sqrtf((float)dim);

    // 驗證參數
    //printf("blockDim=(%d,%d), block_sz=(%d,%d)\n",blockDim.x,blockDim.y,block_sz.x,block_sz.y);

    const uint total_threads_per_tile = BLOCK_ROWS * BLOCK_COLS;
    const uint threads_per_block = total_threads_per_tile / (TILE_ROWS * TILE_COLS);
    // 1D index
    const int thread_flat_id = threadIdx.y * blockDim.x + threadIdx.x;
    // 2D index
    const int thread_col_index = thread_flat_id % (BLOCK_COLS / TILE_COLS);
    const int thread_row_index = thread_flat_id / (BLOCK_COLS / TILE_COLS);

    // Shared memory 分配
    __shared__ float query_tile[BLOCK_ROWS][TILE_SIZE_K];
    __shared__ float key_tile[BLOCK_COLS][TILE_SIZE_K];
    __shared__ float value_tile[BLOCK_COLS][TILE_SIZE_K];
    __shared__ float score_tile[BLOCK_ROWS][BLOCK_COLS+1]; // +1避免bank conflict

    // 用於softmax的暫存
    float local_row_sum[TILE_ROWS] = {0.0f};
    float local_row_max[TILE_ROWS];
    float prev_row_max[TILE_ROWS];
    for (int i = 0; i < TILE_ROWS; i++) {
        local_row_max[i] = -INFINITY;
    }

    float output_accum[TILE_ROWS * TILE_SIZE_K] = {0.0f};

    int num_tiles_col = (seq_length + BLOCK_COLS - 1) / BLOCK_COLS;
    int num_dim_tiles = (dim + TILE_SIZE_K - 1) / TILE_SIZE_K;

    // 迴圈：對維度dim分塊處理
    // 每次處理TILE_SIZE_K個dimension (或最後一塊不滿64)
    for (int d_tile = 0; d_tile < num_dim_tiles; d_tile++){
        int d_start = d_tile * TILE_SIZE_K;
        int d_end = min(d_start + TILE_SIZE_K, dim);
        int d_len = d_end - d_start;

        // 載入本維度區塊的Q tile
        // Q大小: seq_length x dim
        // 該block負責的row範圍: [row_start, row_start+BLOCK_ROWS)
        // 利用 i 在 [0, BLOCK_ROWS*d_len) 範圍內平行載入
        for (int i = thread_flat_id; i < actual_rows * d_len; i += threads_per_block) {
            int r = i / d_len; // row in [0, actual_rows)
            int c = i % d_len;
            query_tile[r][c] = query[batch_offset + (row_start + r)*dim + (d_start + c)];
        }

        // 超出actual_rows的部份填0
        for (int i = thread_flat_id + actual_rows * d_len; i < BLOCK_ROWS * d_len; i += threads_per_block) {
            int r = i / d_len;
            int c = i % d_len;
            if (r < BLOCK_ROWS) query_tile[r][c] = 0.0f;
        }

        // 若d_len < TILE_SIZE_K，將多出部份補0
        for (int i = thread_flat_id; i < BLOCK_ROWS*(TILE_SIZE_K - d_len); i+=threads_per_block){
            int base_col = d_len + i % (TILE_SIZE_K - d_len);
            int r = i / (TILE_SIZE_K - d_len);
            if (r < BLOCK_ROWS && base_col < TILE_SIZE_K)
                query_tile[r][base_col] = 0.0f;
        }

        __syncthreads();

        // 對seq_length方向分塊計算QK^T並累積softmax
        for (int col_block_idx = 0; col_block_idx < num_tiles_col; col_block_idx++) {
            int col_start = col_block_idx * BLOCK_COLS;
            int col_end = min(col_start + BLOCK_COLS, seq_length);
            int actual_cols = col_end - col_start;

            // 載入K tile
            for (int i = thread_flat_id; i < actual_cols * d_len; i += threads_per_block) {
                int r = i / d_len; // row in [0, actual_cols)
                int c = i % d_len;
                key_tile[r][c] = key[batch_offset + (col_start + r)*dim + (d_start + c)];
            }
            // 超出actual_cols範圍的row填0
            for (int i = thread_flat_id + actual_cols*d_len; i < BLOCK_COLS*d_len; i+=threads_per_block) {
                int r = i / d_len;
                int c = i % d_len;
                if (r < BLOCK_COLS) key_tile[r][c] = 0.0f;
            }
            // 將多出部份補0 (若d_len < TILE_SIZE_K)
            for (int i = thread_flat_id; i < BLOCK_COLS*(TILE_SIZE_K - d_len); i+=threads_per_block){
                int base_col = d_len + i % (TILE_SIZE_K - d_len);
                int r = i / (TILE_SIZE_K - d_len);
                if (r < BLOCK_COLS && base_col < TILE_SIZE_K)
                    key_tile[r][base_col] = 0.0f;
            }

            __syncthreads();

            // 計算 QK^T 的部分和
            float thread_results[TILE_ROWS * TILE_COLS] = {0.0f};

            for (int dim_idx = 0; dim_idx < TILE_SIZE_K; dim_idx++) {
                // 從shared memory讀取Q,K到暫存reg
                float reg_query[TILE_ROWS];
                float reg_key[TILE_COLS];

                for (uint row_idx = 0; row_idx < TILE_ROWS; ++row_idx) {
                    int rr = thread_row_index * TILE_ROWS + row_idx;
                    if (rr < BLOCK_ROWS && dim_idx < d_len)
                        reg_query[row_idx] = query_tile[rr][dim_idx];
                    else
                        reg_query[row_idx] = 0.0f;
                }

                for (uint col_idx = 0; col_idx < TILE_COLS; ++col_idx) {
                    int cc = thread_col_index * TILE_COLS + col_idx;
                    if (cc < BLOCK_COLS && dim_idx < d_len)
                        reg_key[col_idx] = key_tile[cc][dim_idx];
                    else
                        reg_key[col_idx] = 0.0f;
                }

                for (uint rr = 0; rr < TILE_ROWS; ++rr) {
                    for (uint cc = 0; cc < TILE_COLS; ++cc) {
                        thread_results[rr*TILE_COLS+cc] += reg_query[rr]*reg_key[cc];
                    }
                }
            }

            __syncthreads();

            // 將結果寫入score_tile並乘以scale
            for (uint rr = 0; rr < TILE_ROWS; ++rr) {
                for (uint cc = 0; cc < TILE_COLS; ++cc) {
                    int row_idx = thread_row_index*TILE_ROWS + rr;
                    int col_idx = thread_col_index*TILE_COLS + cc;
                    float val = thread_results[rr*TILE_COLS+cc]*scale;
                    if (row_idx < actual_rows && col_idx < actual_cols)
                        score_tile[row_idx][col_idx] = val;
                    else
                        score_tile[row_idx][col_idx] = -INFINITY;
                }
            }

            __syncthreads();

            // 更新row max
            for (int rr = 0; rr < TILE_ROWS; rr++){
                int global_row = thread_row_index*TILE_ROWS+rr;
                if (global_row < actual_rows) {
                    prev_row_max[rr] = local_row_max[rr];
                    float max_val = local_row_max[rr];
                    for (int cc = 0; cc < actual_cols; cc++)
                        max_val = fmaxf(max_val, score_tile[global_row][cc]);
                    local_row_max[rr] = max_val;
                }
            }
            __syncthreads();

            // 重新正規化先前結果
            for (int rr = 0; rr < TILE_ROWS; rr++){
                int global_row = thread_row_index*TILE_ROWS+rr;
                if (global_row < actual_rows) {
                    float ratio = expf(prev_row_max[rr]-local_row_max[rr]);
                    local_row_sum[rr]*=ratio;
                    for (int c = 0; c < TILE_SIZE_K; c++){
                        output_accum[rr*TILE_SIZE_K+c] *= ratio;
                    }
                }
            }
            __syncthreads();

            // 載入 Value tile
            for (int i = thread_flat_id; i < actual_cols*d_len; i+=threads_per_block){
                int r = i/d_len;
                int c = i%d_len;
                value_tile[r][c] = value[batch_offset+(col_start+r)*dim+(d_start+c)];
            }
            for (int i = thread_flat_id+actual_cols*d_len; i< BLOCK_COLS*d_len; i+=threads_per_block){
                int r = i/d_len;
                int c = i%d_len;
                if (r< BLOCK_COLS) value_tile[r][c]=0.0f;
            }
            // 填補多餘部份
            for (int i = thread_flat_id; i < BLOCK_COLS*(TILE_SIZE_K - d_len); i+=threads_per_block){
                int base_col = d_len + i%(TILE_SIZE_K - d_len);
                int r = i/(TILE_SIZE_K - d_len);
                if (r < BLOCK_COLS && base_col < TILE_SIZE_K) value_tile[r][base_col]=0.0f;
            }

            __syncthreads();

            // 計算 exp(s_i - m_i)*V 並累積
            for (int rr = 0; rr < TILE_ROWS; rr++){
                int global_row = thread_row_index*TILE_ROWS+rr;
                if (global_row < actual_rows) {
                    for (int cc = 0; cc < actual_cols; cc++) {
                        float exp_val = expf(score_tile[global_row][cc]-local_row_max[rr]);
                        local_row_sum[rr]+=exp_val;
                        // 累積到output_accum
                        for (int c = 0; c < TILE_SIZE_K; c++){
                            output_accum[rr*TILE_SIZE_K+c] += exp_val * value_tile[cc][c];
                        }
                    }
                }
            }

            __syncthreads();

        } // end of col_block_idx loop

    } // end of d_tile loop

    // 寫回global memory (normalize by local_row_sum)
    for (int rr = 0; rr < TILE_ROWS; rr++){
        int global_row = thread_row_index*TILE_ROWS+rr;
        if (global_row < actual_rows) {
            for (int c = 0; c < TILE_SIZE_K; c++){
                int global_col = c; // 因為output_accum大小同dimension tile size，但dim可能不只是64
                if (global_col < dim) { // 此程式假設dim ≤64 或呼叫者確保適配
                    output[batch_offset+(row_start+global_row)*dim+global_col] = output_accum[rr*TILE_SIZE_K+c]/local_row_sum[rr];
                }
            }
        }
    }
}


void execute_flash_attention(float* output_device, float* key_device, float* query_device, float* value_device, int batch_size, int sequence_length, int dimension) {
    dim3 blockDim(BLOCK_ROWS / TILE_COLS, BLOCK_COLS / TILE_ROWS);
    dim3 gridDim(batch_size, (sequence_length + BLOCK_ROWS - 1) / BLOCK_ROWS);
    flash_attention_kernel<<<gridDim, blockDim>>>(output_device, key_device, query_device, value_device , sequence_length, dimension, batch_size);
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    FILE *input_file = fopen(argv[1], "rb");
    if(!input_file){
        fprintf(stderr, "Error: cannot open input file.\n");
        return 1;
    }

    int batch_size, sequence_length, dimension;
    fread(&batch_size, sizeof(int), 1, input_file);
    fread(&sequence_length, sizeof(int), 1, input_file);
    fread(&dimension, sizeof(int), 1, input_file);

    float *query_host = (float*) malloc(batch_size * sequence_length * dimension * sizeof(float));
    float *key_host = (float*) malloc(batch_size * sequence_length * dimension * sizeof(float));
    float *value_host = (float*) malloc(batch_size * sequence_length * dimension * sizeof(float));

    for (int i = 0; i < batch_size; i++) {
        fread(query_host + (i * sequence_length * dimension), sizeof(float), sequence_length * dimension, input_file);
        fread(key_host + (i * sequence_length * dimension), sizeof(float), sequence_length * dimension, input_file);
        fread(value_host + (i * sequence_length * dimension), sizeof(float), sequence_length * dimension, input_file);
    }
    fclose(input_file);

    float *output_device, *key_device, *query_device, *value_device;
    cudaMalloc((void**)&output_device, batch_size * sequence_length * dimension * sizeof(float));
    cudaMalloc((void**)&key_device, batch_size * sequence_length * dimension * sizeof(float));
    cudaMalloc((void**)&query_device, batch_size * sequence_length * dimension * sizeof(float));
    cudaMalloc((void**)&value_device, batch_size * sequence_length * dimension * sizeof(float));

    cudaMemcpy(query_device, query_host, batch_size * sequence_length * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(key_device, key_host, batch_size * sequence_length * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(value_device, value_host, batch_size * sequence_length * dimension * sizeof(float), cudaMemcpyHostToDevice);

    float *output_host = (float*) malloc(batch_size * sequence_length * dimension * sizeof(float));

    // double start, end;
    // start = getTimeStamp();

    execute_flash_attention(output_device, key_device, query_device, value_device, batch_size, sequence_length, dimension);

    cudaDeviceSynchronize();
    // end = getTimeStamp();

    // printf("(B, N, d): (%d, %d, %d)\n", batch_size, sequence_length, dimension);
    // printf("Time: %f\n", end - start);

    cudaMemcpy(output_host, output_device, batch_size * sequence_length * dimension * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i = 0; i< 64 && i<batch_size*sequence_length*dimension ; i++){
    //     printf("CUDA Final O[%d]: %f\n", i, output_host[i]);
    // }

    FILE *output_file = fopen(argv[2], "wb");
    if(!output_file){
        fprintf(stderr, "Error: cannot open output file.\n");
        return 1;
    }
    fwrite(output_host, sizeof(float), batch_size * sequence_length * dimension, output_file);
    fclose(output_file);

    free(query_host);
    free(key_host);
    free(value_host);
    free(output_host);
    cudaFree(output_device);
    cudaFree(key_device);
    cudaFree(query_device);
    cudaFree(value_device);

    return 0;
}
