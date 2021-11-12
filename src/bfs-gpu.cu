
#include "queue.cu"

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
void add(ll * in, ll * out, int * sum, Op op) {
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

    *sum = out[0];
    printf( "%d %d | ", out[0], *sum );
}


template <typename Op>
__device__
void reduce(ll * in, ll * out, int * sum, int N, Op op) {
    int threadSize = N < 1024 ? N : 1024,
        gridSize = (N + threadSize) / threadSize,
        shmSize = threadSize * sizeof(ll);
    gridSize /= 2;
    // main part
    add <<< gridSize, threadSize, shmSize >>> (in, out, sum, op);
    add <<< 1, threadSize, shmSize >>> (out, out, sum, op);
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
            *s2++ = *s1++;
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


__global__
void solve_one(Edge * edges,
               int * idxs,
               bool * vis,
               int * max_val,
               int M,
               int N,
               int SIZE_Y,
               int * log) {

    /* in this implementation looking for maximum distance
     * between any two connected points in the graph */

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // int tid = blockIdx.x * blockDim.x + threadIdx.x +
    //           blockIdx.y * blockDim.y + threadIdx.y;
    int idx1(idxs[tid]), idx2;

    log[tid] = tid;

    Queue<Edge> q;
    q.push(edges[idx1]);
    int ptr = 0;

    while ( !q.empty() )
    {
         Edge tmp = q.top();
         q.pop();

         if ( !vis[tmp.src] )
         {
             vis[tmp.src] = true;

             // *max_val = max(*max_val, tmp.w);
             // log[tid] = max(log[tid], tmp.w);

             updateIndexes(&idx1, &idx2, idxs, M, tmp.src);
             // create Device array for edges
             const int dim = idx2 - idx1 + 2;
             int * h_in = new int[dim], cnt = 0;
             for (int i = idx1; i <= idx2; ++i)
             {
                 Edge e = edges[i];
                 if ( !vis[e.src] )
                 {
                     q.push(e);
                     h_in[cnt++] = e.src;
                 }
             }
             // ----------------------------
             // ---- REDUCE ---- //
             // create data
             int * d_in, * d_out;
             const std::size_t size = dim * sizeof(ll);
             cudaMalloc( &d_in, size );
             cudaMalloc( &d_out, size );
             int * d;
             cudaMalloc( &d, 4);
             // rewrite values
             Util::rewrite(h_in,
                           h_in + dim,
                           d_in);
             Util::setValue(d_out,
                            d_out + dim,
                            0);
             // main part
             reduce(d_in, d_out, d, dim, maxFunctor<int>);
             // deallocate memory
             log[tid] = *d;
             cudaFree(d);
             cudaFree(d_in);
             cudaFree(d_out);
             // ----------------------------
         }
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

            int max_val = 0;
            h_max_val = &max_val;

            cudaMalloc( &d_max_val, sizeof(int) );

            int log_val = M;
            int * d_log, * h_log;
            h_log = new int[log_val];
            for (int i = 0; i < log_val; ++i) h_log[i] = 0;
            cudaMalloc( &d_log, sizeof(int) * log_val );

            // main part
            int SIZE = 1024;
            int SIZE_Y = M / SIZE + 1;
            int blockSize = SIZE,
                gridSize = M / SIZE + 1;
            solve_one <<< blockSize, gridSize >>> (d_edges, d_idxs, d_vis, d_max_val, M, N, SIZE, d_log);
            cudaDeviceSynchronize();

            // GPU -> CPU
            cudaMemcpy( h_max_val, d_max_val, sizeof(int), cudaMemcpyDeviceToHost );
            cudaMemcpy( h_log, d_log, sizeof(int) * log_val, cudaMemcpyDeviceToHost );
            print( h_log, h_log + M );
            cudaFree(d_edges);
            cudaFree(d_log);
            cudaFree(d_max_val);
            cudaFree(d_idxs);
            cudaFree(d_vis);
            // --------------------------------------------------
            max_val = *h_max_val;

            for (int i = 0; i < log_val; ++i) max_val = std::max(max_val, h_log[i]);

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
