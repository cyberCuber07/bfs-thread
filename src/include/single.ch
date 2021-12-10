#ifndef __KERNELS_SINGLE_CH_
#define __KERNELS_SINGLE_CH_


#include "../include/utils.ch"


void solveSingleH(Edge * edges,
                       int * idxs,
                       bool * vis,
                       int M,
                       int N,
                       int * log,
                       int tid)
{

    if (tid == M - 1) return;
    int idx1 = idxs[tid],
        idx2 = idxs[tid + 1];

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

             // updateIndexes(&idx1, &idx2, idxs, M, tid);
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


int solveSingle(Edge * d_edges, int * d_idxs, bool * d_vis, int M, int N, int * d_log) {

    for (int i = 0; i < M; ++i)
            solveSingleH(d_edges, d_idxs, d_vis, M, N, d_log, i);

    int max_val = 0;
    for (int i = 0; i < M; ++i) max_val = std::max(max_val, d_log[i]);

    return max_val;
}

#endif
