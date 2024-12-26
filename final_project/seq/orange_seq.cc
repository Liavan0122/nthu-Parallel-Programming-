#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

// #define MAX_QUEUE_SIZE 1000000000
#define MAX_QUEUE_SIZE 14000000000

typedef struct {
    int x;
    int y;
} Point;

typedef struct {
    Point data[MAX_QUEUE_SIZE];
    int front;
    int rear;
} Queue;

void initQueue(Queue* q) {
    q->front = 0;
    q->rear = 0;
}

int isEmpty(Queue* q) {
    return q->front == q->rear;
}

void enqueue(Queue* q, Point p) {
    q->data[q->rear++] = p;
}

Point dequeue(Queue* q) {
    return q->data[q->front++];
}

int orangesRotting(int** grid, int row, int column) {
    int ans = 0;
    int fresh_count = 0;
    Queue q;
    initQueue(&q);
    int dirs[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

    for(int i = 0; i < row; i++) {
        for(int j = 0; j < column; j++) {
            if(grid[i][j] == 1) fresh_count++;
            else if(grid[i][j] == 2) {
                Point p = {i, j};
                enqueue(&q, p);
            }
        }
    }

    // BFS
    while(!isEmpty(&q) && fresh_count > 0) {
        int size = q.rear - q.front;
        for(int i = 0; i < size; i++) {
            Point current = dequeue(&q);
            for(int k = 0; k < 4; k++) { // 4 directions
                int x = current.x + dirs[k][0];
                int y = current.y + dirs[k][1];
                if(x < 0 || x >= row || y < 0 || y >= column || grid[x][y] != 1) continue;
                grid[x][y] = 2;
                // sleep(1);
                Point new_point = {x, y};
                enqueue(&q, new_point);
                --fresh_count;
            }
        }
        ans++;
    }

    return (fresh_count > 0) ? -1 : ans;
}

void write_output(const char* filename, int result) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%d\n", result);
    fclose(file);
}

int** read_input(const char* filename, int* row, int* column) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open input file");
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d %d", row, column);

    int** grid = (int**)malloc((*row) * sizeof(int*));
    for(int i = 0; i < *row; i++) {
        grid[i] = (int*)malloc((*column) * sizeof(int));
    }

    // Read and parse grid values from file
    char ch;
    int i = 0, j = 0;
    while ((ch = fgetc(file)) != EOF) {
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

int main(int argc, char* argv[]) {
    // 開始計時
    clock_t start, end;
    start = clock();

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int row, column;
    int** grid = read_input(argv[1], &row, &column);

    int result = orangesRotting(grid, row, column);
    write_output(argv[2], result);

    for(int i = 0; i < row; i++) {
        free(grid[i]);
    }
    free(grid);

    // 結束計時
    end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Total time used: %f seconds.\t", cpu_time_used);

    return 0;
}
