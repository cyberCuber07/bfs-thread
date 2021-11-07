
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

    Edge * load(const std::string & path, int & N) {
        vec1d<Edge> data = read_csv(path);
        N = data.size();
        // --- SORT --- //
        // save to adj list
        vec2d<Edge> adj = edge2adj(data);
        // move back to edge list
        return adj2edge(adj, N);
    }

};


struct BFS {

    int N, max_val, n_workers;
    std::vector<std::pair<int,int>> adj_idxs;
    vec1d<bool> vis;
    vec2d<Edge> adj;

    BFS (std::string path)
    {
        ADJ Adj ( path );
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
        int N;
        std::size_t size;
        std::vector<int> idxs;
        Edge * adj;

        void getIdxs(Edge * edges) {
            /* idxs vector stores info when each new node begins in edge list */
            int cnt = 0;
            idxs.push_back(0);
            for (int i = 1; i < N; ++i) {
                /* compares with "cnt" value because edge numering := {0, ..., max_val - 1} */
                if ( cnt != edges[i].src ) {
                    idxs.push_back(i);
                    ++cnt;
                }
            }
        }

        ADJ (std::string path) {
            // load data
            ReadCSV reader;
            Edge * edges = reader.load(path, N);
            // create idxs vector
            getIdxs(edges);
            // allocate memory for all edges
            size = N * sizeof(Edge);
            cudaMalloc( &adj, size );
            // rewrite data
            cudaMemcpy( adj, edges, size, cudaMemcpyHostToDevice );
        }

        std::vector<std::queue<Edge>> init_qs(const int & n_nodes, const int & starting_node) {

            std::vector<std::queue<Edge>> qs (n_nodes);

            for (int i = 0; i < n_nodes; ++i) {
                qs[i].push( adj[starting_node][0] );
            }

            return qs;
        }

        std::vector<int> valid_idxs() {

            std::vector<int> idxs;

            for (int i = 0; i < N; ++i) {
                if ( adj[i].size() ) idxs.push_back(i);
            }

            return idxs;
        }

        void sort_adj() {
            adj_idxs.assign(N, {0, 0});
            std::generate(adj_idxs.begin(), adj_idxs.end(), [&](){
                    static int x = 0, y = 0;
                    return std::make_pair(x++, adj[y++].size());
                  });
            std::sort(adj_idxs.begin(), adj_idxs.end(), [&](std::pair<int,int> a, std::pair<int,int> b){
                        return a.second > b.second;
                    });
            // for (auto tmp : adj_idxs) std::cout << tmp.first << "," << tmp.second << "\n";
        }

        int solve() {

            max_val = 0;
            int K = 1;

            for (int idx = 0; idx < K; ++idx) {
                std::vector<int> idxs = valid_idxs();
                int num_ths = idxs.size();
                std::vector<std::thread> ths(num_ths);
                for (int i = 0; i < num_ths; ++i) {
                    ths[i] = std::thread( &BFS::solve_one, this, adj[idxs[i]][0] );
                }
                for (int i = 0; i < num_ths; ++i) {
                    ths[i].join();
                }
            }

            return max_val;
        }

        void node_run(std::queue<Edge> & q, Edge tmp) {

            max_val = std::max(max_val, tmp.w);
            
            for (Edge e : adj[tmp.src])
            {
                if ( !vis[e.src] ) {
                    q.push(e);
                }
            }
        }

        void solve_one(Edge val) {

            /* in this implementation looking for maximum distance
             * between any two connected points in the graph */

            std::queue<Edge> q;
            q.push(val);

            while (!q.empty())
            {
                Edge tmp = q.front();
                q.pop();

                if ( !vis[tmp.src] )
                {
                    vis[tmp.src] = true;

                    node_run(q, tmp);
                }

            }
        }
    };

};


int main(int argc, char ** argv)
{
    BFS g( argv[1] );
}
