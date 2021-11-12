
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
#include <mutex>
#include "queue.cu"

using namespace DataStructs;

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
    int idx1 = idxs[tid],
        idx2 = tid == M - 1 ? M - 1 : idxs[tid + 1];

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

             // node run
             if ( log[tid] < tmp.w )
             {
                 log[tid] = tmp.w;
             }
             if ( max_val[0] < tmp.w )
             {
                 max_val[0] = tmp.w;
             }
                                                  
             for (int i = idx1; i <= idx2; ++i)
             {
                 Edge e = edges[i];
                 if ( !vis[e.src] )
                 {
                     q.push(e);
                 }
             }
             // ----------------------------
         }
    }
}


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
            // dim3 blockSize (1, 1);
            // dim3 gridSize (SIZE, SIZE_Y);
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