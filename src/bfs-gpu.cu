
#include "utils/queue.cu"

using namespace DataStructs;

#include <string>
#include <queue>
#include <thread>
#include <algorithm>
#include <numeric>


#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdio.h>


typedef int ll;



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
void add(ll * in, ll * out, Op op) {
    extern __shared__ ll shm[];
    unsigned int tid = threadIdx.x,
                 gid = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    // shm[tid] = 0;
    shm[tid] = in[gid] + in[gid + blockDim.x];
    // shm[tid] = op(in[gid], in[gid + blockDim.x]);
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
}


template <typename Op>
__device__
void reduce(ll * in, ll * out, int N, Op op) {
    int threadSize = N < 1024 ? N : 1024,
        gridSize = (N + threadSize) / threadSize,
        shmSize = threadSize * sizeof(ll);
    gridSize /= 2;
    // main part
    add <<< gridSize, threadSize, shmSize >>> (in, out, op);
    cudaDeviceSynchronize();
    add <<< 1, threadSize, shmSize >>> (out, out, op);
    cudaDeviceSynchronize();
}


template <typename T>
using vec2d = std::vector<std::vector<T>>;
template <typename T>
using vec1d = std::vector<T>;


struct Edge {
    int src, dst, w;
};


struct ReadCSV {

    ReadCSV() {}
    
    Edge get_row(std::string line, char delim = ',') {
        std::stringstream s (line);
        std::string word;
        // NOTICE: works for edge form only!!
        Edge tmp;
        getline(s, word, delim);
        tmp.src = stoi(word);
        getline(s, word, delim);
        tmp.dst = stoi(word);
        getline(s, word, delim);
        tmp.w = stoi(word);
        return tmp;
    }
    
    vec1d<Edge> read_csv(const std::string & path) {
    
        vec1d<Edge> data;
        
        std::ifstream f (path, std::ios::in);
    
        std::string tmp;
        while ( f >> tmp ) {
            data.push_back(get_row(tmp));
        }
    
        f.close();
    
        return data;
    }
    
    int getMaxVal(const vec1d<Edge> & data) {
        int max_val(0);
        for (auto & tmp : data) {
            max_val = std::max(max_val, std::max(tmp.src, tmp.dst));
        }
        return max_val + 1; // works for 0, ..., N - 1 [N]
    }

    vec2d<Edge> edge2adj(const vec1d<Edge> & data) {

        int max_val = getMaxVal(data);
        vec2d<Edge> adj(max_val);

        for (auto & tmp : data) {
            adj[tmp.src].push_back(tmp);
        }

        return adj;
    }

    Edge * adj2edge(const vec2d<Edge> & adj, const int & N) {
        // return sorted edge list
        Edge * edges = new Edge[N];
        // convert
        int cnt = 0;
        for (auto & vec : adj) {
            for (auto & e : vec) {
                edges[cnt++] = e;
            }
        }
        return edges;
    }

    Edge * load(const std::string & path, int & N, int & M) {
        vec1d<Edge> data = read_csv(path);
        N = data.size();
        // --- SORT --- //
        // save to adj list
        vec2d<Edge> adj = edge2adj(data);
        M = adj.size();
        // move back to edge list
        return adj2edge(adj, N);
    }

};


std::ostream & operator << (std::ostream & os, Edge e) {
    os << e.src << " " << e.dst << " " << e.w << " | ";
    return os;
}


template <typename It>
void print (It start, It end) {
    while ( start != end ) {
        std::cout << *start++ << " ";
    }
    std::cout << "\n";
}


__host__ __device__
void updateIndexes(int * idx1, int * idx2, int * idxs, int M, int idx) {
    *idx1 = idxs[idx];
    *idx2 = idx == M - 1 ? M - 1 : idxs[idx + 1];
}


template <typename T>
__device__
T maxFunctor(const T a, const T b) {
    return a > b ? a : b;
}


template <typename T>
__device__
T sumFunctor(const T a, const T b) {
    return a + b;
}


namespace Util{

    template <typename It>
    __host__ __device__
    void rewrite(It s1, It e1, It s2) {
        while ( s1 != e1 ) {
            *s2= *s1;
            if ( *s1 != *s2 ) printf( "Error!!\n\n" );
            ++s2, ++s1;
        }
    }

    template <typename It, typename T>
    __device__
    void setValue(It s, It e, T val) {
        while ( s != e ) {
            *s++ = val;
        }
    }

}


// template <typename It, typename T>
// T getMaxVal(It start, It end)
__host__ __device__
int getMaxVal(int * start, int N)
{
    int tmp_max(0);
    while ( N-- )
    {
        tmp_max = max(tmp_max, start[N]);
    }
    return tmp_max;
}


__global__
void solve_one(Edge * edges,
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


struct BFS {

    int max_val;

    BFS (std::string path)
    {
        ADJ Adj ( path );
        max_val = Adj.solve();
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
        int * d_max_val, * h_max_val;

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
            // print( h_idxs, h_idxs + M );
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
            // delete h_max_val;
            // delete[] h_vis;
            // delete[] h_idxs;
        }

        // ---- BFS ---- //

        int solve() {

            const int log_val = M;
            int * d_log, * h_log;
            h_log = new int[log_val];
            cudaMalloc( &d_log, sizeof(int) * log_val );

            // main part
            const int SIZE = 1024;
            const int blockSize = SIZE,
                      gridSize = (M + SIZE) / SIZE;
            solve_one <<< blockSize, gridSize >>> (d_edges, d_idxs, d_vis, M, N, SIZE, d_log);
            cudaDeviceSynchronize();

            // GPU -> CPU
            cudaMemcpy( h_log, d_log, sizeof(int) * log_val, cudaMemcpyDeviceToHost );
            // Util::rewrite( d_log, d_log + log_val, h_log );
            // print( h_log, h_log + log_val );
            cudaFree(d_edges);
            cudaFree(d_log);
            cudaFree(d_idxs);
            cudaFree(d_vis);
            // --------------------------------------------------

            int max_val(0);
            for (int i = 0; i < log_val; ++i) max_val = std::max(max_val, h_log[i]);

            delete[] h_log;

            return max_val;
        }
    };

};


int main(int argc, char ** argv)
{
    std::string path = argv[1];
    BFS g( path );
    std::cout << g.max_val << "\n";
}
