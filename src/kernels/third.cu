#ifndef __KERNELS_THIRD_H_
#define __KERNELS_THIRD_H_


#include "../include/utils.cu"


__global__
void solve_one(Edge * edges,
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


#endif
