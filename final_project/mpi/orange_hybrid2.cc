#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <unistd.h>
#define MAX_QUEUE_SIZE 14000000
enum {
    data_tag,
    terminate_tag,
    result_tag,
    pendding_tag
};
typedef struct {
    int x;
    int y;
    int step;
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

MPI_Datatype createMPITypeForPoint() {
    MPI_Datatype MPI_POINT; // 新的 MPI 数据类型
    int block_lengths[3] = {1, 1, 1}; // 每个字段有一个元素
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT}; // 两个字段的类型都为 MPI_INT

    // 计算每个字段的偏移量
    displacements[0] = offsetof(Point, x);
    displacements[1] = offsetof(Point, y);
    displacements[2] = offsetof(Point, step);

    // 创建自定义数据类型
    MPI_Type_create_struct(3, block_lengths, displacements, types, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT); // 提交类型

    return MPI_POINT;
}

int orangesRotting(int** grid, int row, int column, int rank, int num_procs) {
    MPI_Datatype MPI_POINT = createMPITypeForPoint();
    if (rank == 0){
        int ans = 0;
        int fresh_count = 0;
        Queue q;
        initQueue(&q);
        int dirs[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        for(int i = 0; i < row; i++) {
            for(int j = 0; j < column; j++) {
                if(grid[i][j] == 1){
                        fresh_count++;
                }else if(grid[i][j] == 2) {
                    Point p = {i, j, 0};
                    enqueue(&q, p);
                }
            }
        }
        // BFS
        MPI_Status status;
        int finish = 0;
        int size = q.rear - q.front;
        int worker_rank;

        //先使得所有的worker都有第一個task
        for (int i = 1; i < num_procs; i++) {
            if (i<=size){
                Point current = dequeue(&q);
                sleep(0.01);
                int task_grid[4];
                task_grid[0] = (current.x + 1 < row) ? grid[current.x + 1][current.y] : -1;
                task_grid[1] = (current.x - 1 >= 0) ? grid[current.x - 1][current.y] : -1;
                task_grid[2] = (current.y + 1 < column) ? grid[current.x][current.y + 1] : -1;
                task_grid[3] = (current.y - 1 >= 0) ? grid[current.x][current.y - 1] : -1;
                MPI_Send(&current, 1, MPI_POINT, i, data_tag, MPI_COMM_WORLD);
                MPI_Send(task_grid, 4, MPI_INT, i, data_tag, MPI_COMM_WORLD);
                //printf("Rank 0: send task to %d\n", i);
            } else {
                Point current = {-1, -1, -1};
                int task_grid[4] = {-1, -1, -1, -1};
                MPI_Send(&current, 1, MPI_POINT, i, pendding_tag, MPI_COMM_WORLD);
                MPI_Send(task_grid, 4, MPI_INT, i, pendding_tag, MPI_COMM_WORLD);
            }
        }
        int work_recv = 0;
        while(work_recv < size){
        //等待接收worker完成first round
            Point new_points[5];
            MPI_Recv(new_points, 5, MPI_POINT, MPI_ANY_SOURCE, result_tag, MPI_COMM_WORLD, &status);
            worker_rank = status.MPI_SOURCE;
            if (!isEmpty(&q)) {
                Point current = dequeue(&q);
                sleep(0.01);
                int task_grid[4];
                task_grid[0] = (current.x + 1 < row) ? grid[current.x + 1][current.y] : -1;
                task_grid[1] = (current.x - 1 >= 0) ? grid[current.x - 1][current.y] : -1;
                task_grid[2] = (current.y + 1 < column) ? grid[current.x][current.y + 1] : -1;
                task_grid[3] = (current.y - 1 >= 0) ? grid[current.x][current.y - 1] : -1;
                MPI_Send(&current, 1, MPI_POINT, worker_rank, data_tag, MPI_COMM_WORLD);
                MPI_Send(task_grid, 4, MPI_INT, worker_rank, data_tag, MPI_COMM_WORLD);
            } else {
                Point current = {-1, -1, -1};
                int task_grid[4] = {-1, -1, -1, -1};
                MPI_Send(&current, 1, MPI_POINT, worker_rank, pendding_tag, MPI_COMM_WORLD);
                MPI_Send(task_grid, 4, MPI_INT, worker_rank, pendding_tag, MPI_COMM_WORLD);
            }

            //處理接收到的data
            for (int k = 0; k < 4; k++) {
                if (new_points[k].x == -1) continue;
                if (grid[new_points[k].x][new_points[k].y] == 1) {
                    grid[new_points[k].x][new_points[k].y] = 2;
                    enqueue(&q, new_points[k]);
                    fresh_count--;
                    size++;

                }
                ans = std::max (ans, new_points[k].step);
            }
           // printf("fresh_count = %d\n", fresh_count);
            //printf("ans = %d\n", ans);
            work_recv++;
        }

        for (int i = 1; i < num_procs; i++) {
            Point current = {-1, -1, -1};
            MPI_Send(&current, 1, MPI_POINT, i, terminate_tag, MPI_COMM_WORLD);
        }
        //等待所有worker完成
        //更新grid


        return (fresh_count > 0) ? -1 : ans;
    } else {
        while (true) {
            MPI_Status status;
            int task_grid[4];
            Point current;
            //printf("Rank %d: waiting for task\n", rank);
            MPI_Recv(&current, 1, MPI_POINT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == terminate_tag) break;
            //printf("Rank %d: received task\n", rank);

            MPI_Recv(task_grid, 4, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            Point new_points[5] = {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, current};
            if (status.MPI_TAG == data_tag) {
                for(int k = 0; k < 4; k++) {
                    if(task_grid[k] == 1) {
                        int x = current.x + ((k == 0) ? 1 : (k == 1) ? -1 : 0);
                        int y = current.y + ((k == 2) ? 1 : (k == 3) ? -1 : 0);
                        new_points[k] = (Point){x, y, current.step+1};
                    }
                }
            }
            MPI_Send(new_points, 5, MPI_POINT, 0, result_tag, MPI_COMM_WORLD);
        }
    }
    //printf("Rank %d: finish\n", rank);
    return 0;

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
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int row, column;
    int** grid = read_input(argv[1], &row, &column);


    // Compute result
    int result = orangesRotting(grid, row, column, rank, num_procs);

    // Write output to file
    if(rank == 0)write_output(argv[2], result);


    // Free allocated memory

    for(int i = 0; i < row; i++) {
        free(grid[i]);
    }
    free(grid);


    MPI_Finalize();
    return 0;
}
