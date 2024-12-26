#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <smmintrin.h>  // SSE4.1 intrinsics
#define CHUNK_SIZE 10

const int INF = ((1 << 30) - 1);
const int V = 6010;   // spec 0 <= V <= 6000
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m, NUM_THREADS;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    input(argv[1]);
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NUM_THREADS = CPU_COUNT(&cpu_set);
    int B = 64;   // B = 512 takes 21.07 second
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("n: %d, m: %d\n", n, m);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }
int min(int a, int b) { return a<b?a:b; }

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/

        cal(B, r, r, 0, r, 1);
        cal(B, r, r, r + 1, round - r - 1, 1);
        cal(B, r, 0, r, 1, r);
        cal(B, r, r + 1, r, 1, round - r - 1);


        /* Phase 3*/

        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);

    }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {

    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    int k_start = Round * B;
    int k_end = ((Round + 1) * B < n) ? (Round + 1) * B : n;

    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        int block_internal_start_x = b_i * B;
        int block_internal_end_x = min((b_i + 1) * B , n);

        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = min((b_j + 1) * B , n);

            for (int k = k_start; k < k_end; ++k) {
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    int Dist_ik = Dist[i][k];
                    __m128i v_Dist_ik = _mm_set1_epi32(Dist_ik);
                    int j;

                    // 每次處理 4 個元素
                    for (j = block_internal_start_y; j + 3 < block_internal_end_y; j += 4) {
                        // 從 Dist[i][j] 和 Dist[k][j] 加載元素
                        __m128i v_Dist_ij = _mm_loadu_si128((__m128i*)&Dist[i][j]);
                        __m128i v_Dist_kj = _mm_loadu_si128((__m128i*)&Dist[k][j]);

                        // 計算 Dist_ik + Dist_kj
                        __m128i v_sum = _mm_add_epi32(v_Dist_ik, v_Dist_kj);

                        // 計算 v_Dist_ij 和 v_sum 的最小值
                        __m128i v_new_Dist_ij = _mm_min_epi32(v_Dist_ij, v_sum);

                        // 將結果存回 Dist[i][j]
                        _mm_storeu_si128((__m128i*)&Dist[i][j], v_new_Dist_ij);
                    }
                    // 處理剩餘的元素
                    for (; j < block_internal_end_y; ++j) {
                        Dist[i][j] = min(Dist[i][j], Dist[i][k]+Dist[k][j]);
                    }
                }
            }
        }
    }
}
