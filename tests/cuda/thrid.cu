
#include <iostream>

struct Data {
    int src, dst, w;
};

__global__
void mod ( Data * data ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid].src = tid; 
    data[tid].dst = tid; 
    data[tid].w = tid; 
}


int main()
{
    const int N = 1 << 20;
    Data * data;
    std::size_t size = N * sizeof(Data);
    cudaMalloc( &data, size );


    const int blockSize = 1024,
              gridSize = N / blockSize;
    mod <<< gridSize, blockSize >>> (data);
    cudaDeviceSynchronize();

    Data * h_data = new Data[N];

    cudaMemcpy( h_data, data, size, cudaMemcpyDeviceToHost );

    for (int i = 0; i < N; ++i) {
        std::cout << h_data[i].src << " " <<
                     h_data[i].dst << " " <<
                     h_data[i].w << "\n";
    }
}
