#ifndef __KERNELS_ONE_CH
#define __KERNELS_ONE_CH

#include "../include/utils.ch"

using namespace ReduxFunctors;

__global__
void solveOneKernel (Edge * edges,
                     int * idxs,
                     bool * vis,
                     int M,
                     int N,
                     int SIZE_Y,
                     int * log)
{

    /* in this implementation looking for maximum distance
     * between any two connected points in the graph */

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx1(idxs[tid]), idx2;
    updateIndexes(&idx1, &idx2, idxs, M, tid);

    // --- REDUX --- //
    int diff = idx2 - idx1;
    // if (diff++ <= 0) return;

    int blockSize = 1024 > diff ? 1024 : diff,
        gridSize = (diff + blockSize) / blockSize;

    for (int i = idx1; i <= idx2; ++i) {
        log[i] = maxFunctor(log[i], edges[i].w);
    }
}

int solveOne (Edge * d_edges, int * d_idxs, bool * d_vis, int M, int N, int log_val, int * d_log, int * h_log) {

    const int SIZE = 1024;
    const int blockSize = SIZE,
              gridSize = (M + SIZE) / SIZE;
    solveOneKernel <<< blockSize, gridSize >>> (d_edges, d_idxs, d_vis, M, N, SIZE, d_log);
    cudaDeviceSynchronize();

    // GPU -> CPU
    cudaMemcpy( h_log, d_log, sizeof(int) * log_val, cudaMemcpyDeviceToHost );
    // Util::rewrite( d_log, d_log + log_val, h_log );
    // print( h_log, h_log + log_val );
    // --------------------------------------------------

    int max_val = 0;
    for (int i = 0; i < log_val; ++i) max_val = std::max(max_val, h_log[i]);

    return max_val;
}

#endif
