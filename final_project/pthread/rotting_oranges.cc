#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h> // for usleep
#include <assert.h>  // for assert
#include <cstdio>    // for C standard I/O functions like fopen, fclose, perror

#define INITIAL_QUEUE_SIZE 10000000

typedef struct {
    int x;
    int y;
} Point;

typedef struct {
    Point* data;
    int front;
    int rear;
    int capacity;
    pthread_mutex_t lock;
} Queue;

void initQueue(Queue* q, int capacity) {
    q->front = 0;
    q->rear = 0;
    q->capacity = capacity;
    q->data = (Point*)malloc(sizeof(Point) * q->capacity);
    if (q->data == NULL) {
        perror("Failed to allocate memory for queue");
        exit(EXIT_FAILURE);
    }
    pthread_mutex_init(&q->lock, NULL);
}

int isEmpty(Queue* q) {
    return (q->front == q->rear);
}

void enqueue(Queue* q, Point p) {
    // Resize the queue if it's full
    if (q->rear >= q->capacity) {
        q->capacity *= 2;
        q->data = (Point*)realloc(q->data, sizeof(Point) * q->capacity);
        if (q->data == NULL) {
            perror("Failed to reallocate memory for queue");
            exit(EXIT_FAILURE);
        }
    }
    q->data[q->rear++] = p;
}

Point dequeue(Queue* q) {
    return q->data[q->front++];
}

// Shared variables
int** grid;
int row, column;
int fresh_count;
int dirs[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
Queue current_queue;
Queue next_queue;
pthread_mutex_t fresh_count_lock;
int max_time = 0;
pthread_mutex_t time_lock;

void* process_grid(void* arg) {
    while (1) {
        Point current;

        // Lock current queue
        pthread_mutex_lock(&current_queue.lock);
        if (isEmpty(&current_queue)) {
            pthread_mutex_unlock(&current_queue.lock);
            break;
        }
        current = dequeue(&current_queue);
        pthread_mutex_unlock(&current_queue.lock);

        // Process four directions
        for (int k = 0; k < 4; k++) {
            int x = current.x + dirs[k][0];
            int y = current.y + dirs[k][1];
            if (x < 0 || x >= row || y < 0 || y >= column) {
                continue;
            }

            pthread_mutex_lock(&fresh_count_lock);
            if (grid[x][y] == 1) {
                grid[x][y] = 2;
                fresh_count--;
                pthread_mutex_unlock(&fresh_count_lock);

                Point new_point = {x, y};
                pthread_mutex_lock(&next_queue.lock);
                enqueue(&next_queue, new_point);
                pthread_mutex_unlock(&next_queue.lock);
            } else {
                pthread_mutex_unlock(&fresh_count_lock);
            }
        }
    }

    return NULL;
}

int orangesRotting(int** input_grid, int input_row, int input_column, int num_threads) {
    grid = input_grid;
    row = input_row;
    column = input_column;

    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    int* thread_ids = (int*)malloc(num_threads * sizeof(int));
    fresh_count = 0;
    max_time = 0;

    initQueue(&current_queue, row * column);
    initQueue(&next_queue, row * column);
    pthread_mutex_init(&fresh_count_lock, NULL);
    pthread_mutex_init(&time_lock, NULL);

    // Initialize grid and queues
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            if (grid[i][j] == 1) {
                fresh_count++;
            } else if (grid[i][j] == 2) {
                Point p = {i, j};
                enqueue(&current_queue, p);
            }
        }
    }

    while (1) {
        if (isEmpty(&current_queue)) {
            break;
        }

        // Create threads to process the grid
        for (int i = 0; i < num_threads; i++) {
            thread_ids[i] = i;
            pthread_create(&threads[i], NULL, process_grid, &thread_ids[i]);
        }

        // Join threads
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }


        // Increment time if any processing occurred
        if (!isEmpty(&next_queue)) {
            pthread_mutex_lock(&time_lock);
            max_time++;
            pthread_mutex_unlock(&time_lock);
        }

        // Swap queues
        pthread_mutex_lock(&current_queue.lock);
        pthread_mutex_lock(&next_queue.lock);
        Queue temp = current_queue;
        current_queue = next_queue;
        next_queue = temp;
        next_queue.front = 0;
        next_queue.rear = 0;
        pthread_mutex_unlock(&next_queue.lock);
        pthread_mutex_unlock(&current_queue.lock);
    }

    int result = (fresh_count > 0) ? -1 : max_time;

    // Cleanup
    pthread_mutex_destroy(&current_queue.lock);
    pthread_mutex_destroy(&next_queue.lock);
    pthread_mutex_destroy(&fresh_count_lock);
    pthread_mutex_destroy(&time_lock);

    free(current_queue.data);
    free(next_queue.data);
    free(threads);
    free(thread_ids);

    return result;
}

// File handling functions
int** read_input(const char* filename, int* row, int* column) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Unable to open input file");
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d %d", row, column);

    int** grid = (int**)malloc((*row) * sizeof(int*));
    for (int i = 0; i < *row; i++) {
        grid[i] = (int*)malloc((*column) * sizeof(int));
    }

    // Read grid data
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

void write_output(const char* filename, int result) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Unable to open output file");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%d\n", result);
    fclose(file);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input file> <output file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int row, column;
    int** grid = read_input(argv[1], &row, &column);
    int num_threads = 1;
    int result = orangesRotting(grid, row, column, num_threads);

    write_output(argv[2], result);

    for (int i = 0; i < row; i++) {
        free(grid[i]);
    }
    free(grid);

    return 0;
}
