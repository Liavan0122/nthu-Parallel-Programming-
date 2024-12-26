#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#define min(a,b) (((a)<(b))?(a):(b))

unsigned long long r;
unsigned long long k;
unsigned long long rr;
unsigned long long *partial_sum;

int ncpus;

void* calculate_pixel(void* threadid){
    int tid = *(int*)threadid;

    unsigned long long unit = r / ncpus, remain = r % ncpus;
    unsigned long long offset = unit*tid + min(remain, tid);
    unsigned long long length = unit + (tid < remain);

    unsigned long long pixel = ceil(sqrtl(rr - offset * offset));
    partial_sum[tid] = 0;

    for(unsigned long long i = offset; i < offset+length; i++){
        unsigned long long prev = rr-i*i;
        while(prev <= (pixel - 1)*(pixel - 1) )  pixel -= 4;
        while(prev > pixel * pixel) pixel++;
        partial_sum[tid] += pixel;
    }

    // partial_sum[tid] %= k;
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    assert(argc == 3);
    r = atoll(argv[1]);
	k = atoll(argv[2]);
	rr = r * r;
    partial_sum = new unsigned long long[ncpus]();

    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);

    pthread_t threads[ncpus];
    int ID[ncpus];

    for(int i = 0; i< ncpus; i++){
        ID[i] = i;
        pthread_create(&threads[i], NULL, calculate_pixel, (void*)&ID[i]);
    }

    unsigned long long ans = 0;
    for (int i = 0; i < ncpus; i++) {
        pthread_join(threads[i], NULL);
        ans = (ans+partial_sum[i])%k;
    }

    printf("%llu\n", (4 * ans) % k);

    return 0;
}