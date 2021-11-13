
#include <stdio.h>
#include <iostream>
#include <algorithm>


template <typename It>
void print (It start, It end) {
    while ( start != end ) {
        std::cout << *start++ << " ";
    }
    std::cout << "\n";
}


typedef int ll;

template <typename T>
__host__ __device__
T maxFunctor(const T a, const T b) {
    return a > b ? a : b;
}


template <typename T>
__host__ __device__
T sumFunctor(const T a, const T b) {
    return a + b;
}


namespace Util{

    template <typename It>
    __host__ __device__
    void rewrite(It s1, It e1, It s2) {
        while ( s1 != e1 ) {
            *s2++ = *s1++;
        }
    }

    template <typename It, typename T>
    __host__ __device__
    void setValue(It s, It e, T val) {
        while ( s != e ) {
            *s++ = val;
        }
    }

}


template <typename Op>
__device__
void warpReduce(volatile ll * data, int tid, Op op) {
    data[tid] = op(data[tid], data[tid + 32]);
    data[tid] = op(data[tid], data[tid + 16]);
    data[tid] = op(data[tid], data[tid + 8]);
    data[tid] = op(data[tid], data[tid + 4]);
    data[tid] = op(data[tid], data[tid + 2]);
    data[tid] = op(data[tid], data[tid + 1]);
}


template <typename Op>
__global__
void add(ll * in, ll * out, int * result, Op op) {
    extern __shared__ ll shm[];
    unsigned int tid = threadIdx.x,
                 gid = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    shm[tid] = 0;
    // shm[tid] = in[gid] + in[gid + blockDim.x];
    shm[tid] = op(shm[tid], in[gid] + in[gid + blockDim.x]);
   __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if ( tid < stride ) {
            // shm[tid] += shm[tid + stride];
            shm[tid] = op(shm[tid], shm[tid + stride]);
        }
        __syncthreads();
    }

    if ( tid < 32 ) {
        warpReduce(shm, tid, op);
    }

    if ( tid == 0 ) {
        out[blockIdx.x] = shm[0];
    }

    *result = out[0];
    printf( "%d %d | ", out[0], *result );
}


template <typename Op>
__host__ __device__
void reduce(ll * in, ll * out, int * result, int N, Op op) {
    int threadSize = N < 1024 ? N : 1024,
        gridSize = (N + threadSize) / threadSize,
        shmSize = threadSize * sizeof(ll);
    gridSize /= 2;
    // main part
    add <<< gridSize, threadSize, shmSize >>> (in, out, result, op);
    add <<< 1, threadSize, shmSize >>> (out, out, result, op);
    cudaDeviceSynchronize();
}


int main()
{
    const int N = 1 << 12;
    const std::size_t size = sizeof(ll) * N;
    ll * h_data = new ll[N];
    std::generate(h_data, h_data + N, []{static ll cnt = 0; return cnt++;});

    ll * d_in, * d_out, * result;
    cudaMalloc( &d_in, size );
    cudaMalloc( &d_out, size );
    cudaMalloc( &result, sizeof(ll) );

    Util::rewrite(h_data, h_data + size, d_in);
    Util::setValue(d_out, d_out + size, (ll)0);


    reduce(d_in, d_out, result, N, sumFunctor<ll>);
    int * h_result = new ll();

    // *h_result = (ll)0;
    // Util::rewrite(result, result + 1, h_result);

    // std::cout << *h_result << "\n";


    delete[] h_data;
    delete h_result;
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(result);
}
