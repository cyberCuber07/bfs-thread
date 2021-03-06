#ifndef __SOLVER__CH_
#define __SOLVER__CH_

#include <chrono>
#include "single.ch"

struct BFS {

    int max_val {0};
    int N, M, * d_M;
    std::size_t size;
    int *d_log, * h_log;
    int * d_idxs, * h_idxs;
    bool * d_vis, * h_vis;
    Edge * d_edges, * h_edges;
    int * d_max_val, * h_max_val;

    BFS (std::string path)
    {
        ReadCSV reader;
        h_edges = reader.load(path, N, M);
    }

    void getIdxs(Edge * edges) {
        /* idxs vector stores info when each new node begins in edge list */
        int cnt = 0;
        h_idxs = new int[M];
        for (int i = 1; i < N; ++i) {
            /* compares with "cnt" value because edge numering := {0, ..., max_val - 1} */
            if ( cnt != edges[i].src ) {
                h_idxs[cnt++] = i;
                if ( cnt + 1 == M ) break;
            }
        }
        h_idxs[M - 1] = N - 1;
    }

    void cudaAllocate(int log_val) {
        cudaMalloc( &d_idxs, M * sizeof(int) );
        cudaMalloc( &d_vis, M * sizeof(bool) );
        cudaMalloc( &d_edges, size );
        cudaMemcpy( d_idxs, h_idxs, M * sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( d_vis, h_vis, M * sizeof(bool), cudaMemcpyHostToDevice );
        cudaMemcpy( d_edges, h_edges, size, cudaMemcpyHostToDevice );
        cudaMalloc( &d_log, sizeof(int) * log_val );
    }

    int allocate() {
        size = N * sizeof(Edge);
        getIdxs(h_edges);
        h_vis = new bool[M];
        for (int i = 0; i < M; ++i) h_vis[i] = false;
        const int log_val = M;
        h_log = new int[log_val];
        return log_val;
    }

    void deallocate() {
        cudaFree(d_edges);
        cudaFree(d_log);
        cudaFree(d_idxs);
        cudaFree(d_vis);
    }

    template <typename SolverT>
    void solve(SolverT solver) {
        int log_val = allocate();
        cudaAllocate(log_val);
        // --- SOLVE --- //
        auto start_time = std::chrono::steady_clock::now();
        max_val = solver(d_edges, d_idxs, d_vis, M, N, log_val, d_log, h_log);
        // --- MEASURE TIME --- //
        std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_time).count() << " microseconds\n";
        deallocate();
    }

    void solveSingleCaller() {
        allocate();
        auto start_time = std::chrono::steady_clock::now();
        max_val = solveSingle(h_edges, h_idxs, h_vis, M, N, h_log);
        std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_time).count() << " microseconds\n";
    }

    ~BFS() {
        delete[] h_vis;
        delete[] h_idxs;
        delete[] h_log;
    }

};

#endif
