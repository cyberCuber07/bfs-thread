
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


void solve_one_CPU(Edge * edges,
               int * idxs,
               bool * vis,
               int * max_val,
               int M,
               int tid) {

    /* in this implementation looking for maximum distance
     * between any two connected points in the graph */

    int idx1 = idxs[tid],
        idx2 = tid == M - 1 ? M - 1 : idxs[tid + 1];

    Edge * q = new Edge[M];
    int ptr = 0;
    q[ptr] = edges[idx1];

    while (ptr >= 0)
    {
        Edge tmp = q[ptr--];

        if ( !vis[tmp.src] )
        {
            vis[tmp.src] = true;

            // node run
            if ( max_val[0] < tmp.w )
            {
                max_val[0] = tmp.w;
            }
            
            for (int i = idx1; i <= idx2; ++i)
            {
                Edge e = edges[i];
                if ( !vis[e.src] )
                {
                    q[++ptr] = e;
                }
            }
            // ----------------------------
        }
    }

    delete[] q;
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
                }
            }
        }

        ADJ (std::string path) {
            // load data
            ReadCSV reader;
            h_edges = reader.load(path, N, M);
            size = N * sizeof(Edge);
            // create idxs vector
            getIdxs(h_edges);
            // ------------------------------
            // --M;
            // ------------------------------
            h_vis = new bool[M];
            for (int i = 0; i < M; ++i) h_vis[i] = false;
        }

        // ---- BFS ---- //

        int solve() {

            int max_val = 0;
            h_max_val = &max_val;

            // main part
            std::vector<std::thread> ths(M);
            for (int i = 0; i < M; ++i) {
                ths[i] = std::thread( solve_one_CPU, h_edges, h_idxs, h_vis, h_max_val, M, i );
            }
            for (int i = 0; i < M; ++i) {
                ths[i].join();
            }
            // --------------------------------------------------

            // max_val = *h_max_val;

            return max_val;
        }
    };

};


int main(int argc, char ** argv)
{
    BFS g( argv[1] );
    std::cout << g.max_val << "\n";
}
