
#ifndef __BFS_H_
#define __BFS_H_

#include <vector>
#include <string>
#include <queue>
#include <thread>
#include <algorithm>
#include <numeric>

#include "graph_converter.h"
#include "read_csv.h"


template <typename _type>
struct BFS {

    int N, max_val, n_workers;
    std::vector<std::pair<int,int>> adj_idxs;
    vec1d<bool> vis;
    vec2d<_type> adj;
 
    template <typename InsertT>
    BFS (std::string path, InsertT insert, int n_workers) : n_workers(n_workers)
    {
        ReadCSV reader;
        adj = edge2adj<_type>(reader.load(path), insert, N);
        sort_adj();
        vis.assign(N, false);
    }

    std::vector<std::queue<_type>> init_qs(const int & n_nodes, const int & starting_node) {

        std::vector<std::queue<_type>> qs (n_nodes);

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

    void node_run(std::queue<_type> & q, _type tmp) {

        max_val = std::max(max_val, tmp.w);
        
        for (_type e : adj[tmp.src])
        {
            if ( !vis[e.src] ) {
                q.push(e);
            }
        }
    }

    void solve_one(_type val) {

        /* in this implementation looking for maximum distance
         * between any two connected points in the graph */

        std::queue<_type> q;
        q.push(val);

        while (!q.empty())
        {
            _type tmp = q.front();
            q.pop();

            if ( !vis[tmp.src] )
            {
                vis[tmp.src] = true;

                node_run(q, tmp);
            }

        }
    }

};


#endif
