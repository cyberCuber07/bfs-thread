
#include <stdio.h>


__global__
void call_2(int idx_2, int idx_1) {
    int tid = threadIdx.x;
    printf( "       %d\n", (idx_2 * 2 + idx_1) * 2 + tid);
}


__global__
void call_1(int idx_1) {
    int tid = threadIdx.x;
    printf( "   %d\n", idx_1 * 2 + tid);
    call_2 <<< 1, 4 >>> (tid, idx_1);
    cudaDeviceSynchronize();
}


__global__
void run(){
    int tid = threadIdx.x;
    printf( "%d\n", tid );
    call_1 <<< 1, 3 >>> (tid);
    cudaDeviceSynchronize();
}


int main()
{
    run <<< 1, 2 >>> ();
    cudaDeviceSynchronize();
}
