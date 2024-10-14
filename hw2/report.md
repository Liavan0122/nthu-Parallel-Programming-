
## Title, name, student ID
楊峻銘 112062576


## Implementation
### Pthread
* Load Balance
  由於qct-cpu，96 per node and 2 threaads each core，所以我直接把總工作量分成96個分支，並在執行程式前採用動態分配每個threads的工作量，
  將正方形總面積由上而下的按行劃分，讓每個threads運行不同行的元素。這是一個無限迴圈，每個線程都會嘗試獲取一行來計算。如果所有行都被分配，線程就會退出。
  ```
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
  ```

  使用 AVX512 指令一次處理 8 個元素，因此每次 for 迴圈遞增 8。使用 _mm512_set_pd 指令將 8 個初始的 x0 值由座標左至右載入到 SIMD 暫存器 x0 中。
  y0為計算該行在複數平面上的值。
  ```
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
  ```

* SIMD+Algo
  
`Intel R AVX-512 family` 指令集可以加速計算，演算法概念部分和sequencial雷同，新增了`repeats`變量紀錄每個元素迭代次數，將 SIMD 暫存器中的 repeats 結果存儲回緩衝區 buffer 中， buffer為整個面積元素對應位置。
```
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
```

### Hybrid

* Load Balance
  工作量平均分給每個節點，在每個rank底下，分配給多個threads，並且 OpenMP 並行部分的工作分配方式使用 `schedule(dynamic)`，而且原來的 pthread 使用了互斥鎖來控制 current_row 的更新，但在 OpenMP 中使用更為高效的 schedule 子句來自動分配行給線程。
  
  ```
    long long unit = height / size;
    long long remain = height % size;
    long long start_row = unit * rank + (rank < remain ? rank : remain);
    long long num_rows = unit + (rank < remain ? 1 : 0);
    image = (int*)malloc(num_rows * width * sizeof(int));

  ```
  ```
    #pragma omp for schedule(dynamic) reduction(+:current_row)
  ```
  
* SIMD+Algo
  
  整體概念與pthread相同，這裡改嘗試用reduction最終合併。
  
* Gather
  收集所有 MPI 節點計算的結果，最終由 Rank 0 節點來匯總和生成最終的 PNG 圖像。
  ```
    if (rank == 0) {
        int* full_image = (int*)malloc(total * sizeof(int));
        assert(full_image);
        memcpy(full_image + start_row * width, image, num_rows * width * sizeof(int));

        for (int i = 1; i < size; i++) {
            long long recv_start_row = unit * i + (i < remain ? i : remain);
            long long recv_num_rows = unit + (i < remain ? 1 : 0);
            MPI_Recv(full_image + recv_start_row * width, recv_num_rows * width, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        write_png(filename, iters, width, height, full_image);
        free(full_image);
    } else {
        MPI_Send(image, num_rows * width, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  ```


## Experiment & Analysis
### Methodology
#### System Spec
課程所提供的cluster，就以最後一個testcase 40.txt 作為測資，n: 536869888。
#### Performance Metrics

### Plots: Speedup Factor & Profile
#### Experimental Method:
#### Performance Measurement:
#### Analysis of Results:
#### Optimization Strategies:


## Experiences / Conclusion
