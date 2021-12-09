#ifndef __SOLVER_H_
#define __SOLVER_H_

#include <vector>
#include "utils.cu"
#include "../kernels/third.cu"

using namespace DataStructs;


struct BFS {

    int N, max_val, n_workers;
    std::vector<std::pair<int,int>> adj_idxs;
    vec1d<bool> vis;
    vec2d<Edge> adj;

    BFS (std::string path)
    {
        ADJ Adj ( path );
        max_val = Adj.solve();
        // adj = edge2adj<Edge>(reader.load(path), insert, N);
        // sort_adj();
        // vis.assign(N, false);
    }

    /* cuda variables */
    struct ADJ {
        /* GPU struct to:
                -- load memory on global / shared memory
                -- stores data for each node --- adjacency list form
                -- copies data to CPU
         */
        int N, M, * d_M;
        std::size_t size;
        int * d_idxs, * h_idxs;
        bool * d_vis, * h_vis;
        Edge * d_edges, * h_edges;

        void getIdxs(Edge * edges) {
            /* idxs vector stores info when each new node begins in edge list */
            int cnt = 0;
            h_idxs = new int[M];
            for (int i = 1; i < N; ++i) {
                /* compares with "cnt" value because edge numering := {0, ..., max_val - 1} */
                if ( cnt != edges[i].src ) {
                    h_idxs[cnt++] = i;
                    if ( cnt + 1 == M ) return;
                }
            }
        }

        ADJ (std::string path) {
            ReadCSV reader;
            h_edges = reader.load(path, N, M);
            size = N * sizeof(Edge);
            getIdxs(h_edges);
            print( h_idxs, h_idxs + M );
            h_vis = new bool[M];
            for (int i = 0; i < M; ++i) h_vis[i] = false;
            cudaMalloc( &d_idxs, M * sizeof(int) );
            cudaMalloc( &d_vis, M * sizeof(bool) );
            cudaMalloc( &d_edges, size );
            cudaMemcpy( d_idxs, h_idxs, M * sizeof(int), cudaMemcpyHostToDevice );
            cudaMemcpy( d_vis, h_vis, M * sizeof(bool), cudaMemcpyHostToDevice );
            cudaMemcpy( d_edges, h_edges, size, cudaMemcpyHostToDevice );
            // delete[] h_vis;
            // delete[] h_idxs;
            // delete tmp_M;
            // delete[] tmp_edges;
        }

        ~ADJ() {
            delete[] h_vis;
            delete[] h_idxs;
        }

        // ---- BFS ---- //

        int solve() {

            int log_val = M;
            int * d_log, * h_log;
            h_log = new int[log_val];
            for (int i = 0; i < log_val; ++i) h_log[i] = 0;
            cudaMalloc( &d_log, sizeof(int) * log_val );

            // main part
            int SIZE = 1024;
            int blockSize = SIZE,
                gridSize = M / SIZE + 1;
            solve_one <<< gridSize, blockSize >>> (d_edges, d_idxs, d_vis, M, N, SIZE, d_log);
            cudaDeviceSynchronize();

            // GPU -> CPU
            cudaMemcpy( h_log, d_log, sizeof(int) * log_val, cudaMemcpyDeviceToHost );
            print( h_log, h_log + M );
            cudaFree(d_edges);
            cudaFree(d_log);
            cudaFree(d_idxs);
            cudaFree(d_vis);
            // --------------------------------------------------

            int max_val(0);
            for (int i = 0; i < log_val; ++i) max_val = std::max(max_val, h_log[i]);

            return max_val;
        }
    };

};

#endif
