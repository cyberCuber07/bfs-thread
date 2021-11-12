
#include <iostream>

typedef int ll;


__device__
void warpReduce(volatile ll * data, int tid) {
    data[tid] += data[tid + 32];
    data[tid] += data[tid + 16];
    data[tid] += data[tid + 8];
    data[tid] += data[tid + 4];
    data[tid] += data[tid + 2];
    data[tid] += data[tid + 1];
}


__global__
void add(ll * in, ll * out) {
    extern __shared__ ll shm[];
    unsigned int tid = threadIdx.x,
                 gid = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    shm[tid] = 0;
    // if ( gid < 1024 )
        shm[tid] = in[gid] + in[gid + blockDim.x];
   __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if ( tid < stride ) {
            shm[tid] += shm[tid + stride];
        }
		__syncthreads();
    }

    if ( tid < 32 ) {
        warpReduce(shm, tid);
    }

    if ( tid == 0 ) {
        out[blockIdx.x] = shm[0];
    }
}

__global__ void reduce4(ll * g_odata, ll * g_idata, unsigned int len) {
	extern __shared__ unsigned int sdata[];

	// each thread loads one element from global to shared mem
	// Do the first stage of the reduction on the global-to-shared load step
	// This reduces the previous inefficiency of having half of the threads being
	//  inactive on the first for-loop iteration below (previous first step of reduction)
	// Previously, only less than or equal to 512 out of 1024 threads in a block are active.
	// Now, all 512 threads in a block are active from the start
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = 0;

	if (i < len)
	{
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	}

	__syncthreads();

	// do reduction in shared mem
	// this loop now starts with s = 512 / 2 = 256
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

int main()
{
    const int N = (1 << 21);
    ll * h_in  = new ll[N],
        * d_in,
        * d_out;
    const std::size_t size = sizeof(ll) * N;
    cudaMalloc( &d_in, size );
    cudaMalloc( &d_out, size );
    cudaMemset( d_out, 0, size );
    for (int i = 0; i < N; ++i) h_in[i] = 1;

    cudaMemcpy( d_in, h_in, size, cudaMemcpyHostToDevice );

    int threadSize = N < 1024 ? N : 1024,
        gridSize = (N + threadSize) / threadSize,
        shmSize = threadSize * sizeof(ll);
    // reduce4<<< gridSize, threadSize, shmSize >>> (d_out, d_in, threadSize);
    // reduce4<<<        1, threadSize, shmSize >>> (d_out, d_out, threadSize);
    gridSize /= 2;
    add <<< gridSize, threadSize, shmSize >>> (d_in, d_out);
    add <<< 1, threadSize, shmSize >>> (d_out, d_out);
    // cudaDeviceSynchronize();

    cudaMemcpy( h_in, d_out, size, cudaMemcpyDeviceToHost );

    std::cout << h_in[0] << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
}
