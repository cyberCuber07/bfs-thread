
#include <stdio.h>


__global__
void call_2(int idx_2, int idx_1) {
    int tid = threadIdx.x;
    printf( "       %d\n", (tid * 2 + idx_1) * 2 + idx_2);
}


__global__
void call_1(int idx_1) {
    int tid = threadIdx.x;
    printf( "   %d\n", tid * 2 + idx_1);
    call_2 <<< 1, 2 >>> (tid, idx_1);
    cudaDeviceSynchronize();
}


__global__
void run(){
    int tid = threadIdx.x;
    printf( "%d\n", tid );
    call_1 <<< 1, 2 >>> (tid);
    cudaDeviceSynchronize();
}


int main()
{
    run <<< 1, 2 >>> ();
    cudaDeviceSynchronize();
}
