
#include <iostream>

struct Data {
    int src, dst, w;
};

struct Adj {
    Data * data;
};


__global__
void mod (Data* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid].src = tid;
    data[tid].dst = tid;
    data[tid].w = tid;
}


int main()
{
    const int N = 2, M = 1 << 20;
    Adj * adj = new Adj[N];
    const std::size_t size = N * sizeof(Adj),
          m_size = M * sizeof(Data);
    cudaMalloc(&adj, size);
    for (int i = 0; i < N; ++i) {
        cudaMalloc(&adj[i].data, m_size);
    }

    int blockSize = 1024,
        gridSize = M / blockSize;
    for (int i = 0; i < M; ++i) {
        mod <<< blockSize, gridSize >>> (adj[i].data);
        cudaDeviceSynchronize();
    }

    Adj * h_adj = new Adj[N];
    for (int i = 0; i < M; ++i) {
        h_adj[i].data = new Data[M];
        cudaMemcpy( h_adj[i].data, adj[i].data, size, cudaMemcpyDeviceToHost );
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
           std::cout << adj[i].data[j].src << " " << adj[i].data[j].dst << " " << adj[i].data[j].w << "\n";
        }
        std::cout << "\n";
    }
}
