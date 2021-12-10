#ifndef __KERNELS_SINGLE_H_
#define __KERNELS_SINGLE_H_


#include "../include/utils.ch"


__global__
void solveThirdKernel (Edge * edges,
                       int * idxs,
                       bool * vis,
                       int M,
                       int N,
                       int SIZE_Y,
                       int * log)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x,
        idx1 = idxs[tid],
        idx2 = tid == M - 1 ? M - 1 : idxs[tid + 1];

    log[tid] = tid;

    Queue<Edge> q;
    q.push(edges[idx1]);

    while ( !q.empty() )
    {
         Edge tmp = q.top();
         q.pop();

         if ( !vis[tmp.src] )
         {
             vis[tmp.src] = true;

             // node run
             if ( log[tid] < tmp.w )
             {
                 log[tid] = tmp.w;
             }

             for (int i = idx1; i <= idx2; ++i)
             {
                 Edge e = edges[i];
                 if ( !vis[e.src] )
                 {
                     q.push(e);
                 }
             }
         }
    }
}


int solveThird(Edge * d_edges, int * d_idxs, bool * d_vis, int M, int N, int log_val, int * d_log, int * h_log) {

    const int SIZE = 1024;
    const int blockSize = SIZE,
              gridSize = (M + SIZE) / SIZE;
    solveThirdKernel <<< blockSize, gridSize >>> (d_edges, d_idxs, d_vis, M, N, SIZE, d_log);
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
