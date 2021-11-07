

typedef long long _type;

struct Data {
    int a, b;
};

struct Adj {
    Data * data;
};


__global__ void write_into(_type * d_nums) {

    _type tid = blockIdx.x * blockDim.x + threadIdx.x;

    d_nums[tid] = tid;
}


int main()
{
    const long long N = 1 << 28;
    _type * h_nums = new _type[N],
          * d_nums;
    const std::size_t size = N * sizeof(_type);
    Data *data = new Data[N];
    cudaMalloc(&data, sizeof(Data) * N);
    cudaMalloc(&d_nums, size);

    // cudaMemcpy(d_nums, h_nums, size, cudaMemcpyHostToDevice);

    // ------------------------------
    // main part
    _type blockSize = 1024,
          gridSize = N / blockSize;

    write_into <<< gridSize, blockSize >>> (d_nums);

    cudaDeviceSynchronize();
    // ------------------------------

    cudaMemcpy(h_nums, d_nums, size, cudaMemcpyDeviceToHost);

    cudaFree(data);
    cudaFree(d_nums);
    delete[] h_nums;
}
