#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#define min(a,b) (((a)<(b))?(a):(b))

unsigned long long r;
unsigned long long k;
unsigned long long rr;

int main(int argc, char* argv[]) {
    assert(argc == 3);
    r = atoll(argv[1]);
    k = atoll(argv[2]);
    rr = r * r;
    unsigned long long ans = 0;
	int nthreads;
    // 設定使用的執行緒數量，這裡使用所有可用的執行緒
    int ncpus = omp_get_max_threads();

    #pragma omp parallel num_threads(ncpus) reduction(+:ans)
    {
		//確保只有一個線程來執行一些初始化操作
		#pragma omp single
		{
        	nthreads = omp_get_num_threads();
		}
        int tid = omp_get_thread_num();

        unsigned long long unit = r / nthreads;
        unsigned long long remain = r % nthreads;
        unsigned long long offset = unit * tid + min(remain, tid);
        unsigned long long length = unit + (tid < remain);

		unsigned long long pixel = ceil(sqrtl(rr - offset * offset));
		
		for(unsigned long long i = offset; i < offset+length; i++){
			unsigned long long prev = rr-i*i;
			while(prev <= (pixel - 1)*(pixel - 1) )  pixel -= 4;
			while(prev > pixel * pixel) pixel++;
			ans += pixel;
		}

		ans %= k;
    }

    printf("%llu\n", (4 * ans) % k);

    return 0;
}


