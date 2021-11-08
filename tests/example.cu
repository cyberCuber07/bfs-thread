
#include <iostream>

#define SIZE 1024


template <typename It>
void print(It start, It end) {
    while ( start != end ) {
        std::cout << *start++ << " ";
    }
    std::cout << "\n";
}


__global__
void mod (int * log)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    log[tid] = tid;
}


int main()
{
    const long long N = 1 << 27;
    int * h_log = new int[N],
        * d_log;
    std::size_t size = N * sizeof(int);
    cudaMalloc( &d_log, size );

    int blockSize = SIZE,
        gridSize = N / SIZE + 1;
    mod <<< gridSize, blockSize >>> (d_log);
    cudaDeviceSynchronize();

    cudaMemcpy( h_log, d_log, size, cudaMemcpyDeviceToHost );

    std::cout << h_log[N - 1] << "\n";
}
