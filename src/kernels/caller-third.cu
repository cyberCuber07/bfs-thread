#ifndef __KERNELS_CALLER_THIRD_CU_
#define __KERNELS_CALLER_THIRD_CU_

#include "third.cu"

int solveThird(Edge * d_edges, int * d_idxs, bool * d_vis, int M, int N, int log_val, int * d_log, int * h_log) {

    const int SIZE = 1024;
    const int blockSize = SIZE,
              gridSize = (M + SIZE) / SIZE;
    solve_one <<< blockSize, gridSize >>> (d_edges, d_idxs, d_vis, M, N, SIZE, d_log);
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
