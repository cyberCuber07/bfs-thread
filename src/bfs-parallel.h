
#ifndef __BFS_H_
#define __BFS_H_

#include <vector>
#include <string>
#include <queue>
#include <thread>

#include "graph_converter.h"
#include "read_csv.h"


template <typename _type>
struct BFS {

    int N, max_val, n_workers;
    vec1d<bool> vis;
    vec2d<_type> adj;
 
    template <typename InsertT>
    BFS (std::string path, InsertT insert, int n_workers) : n_workers(n_workers)
    {
        ReadCSV reader;
        adj = edge2adj<_type>(reader.load(path), insert, N);
        vis.assign(N, false);
    }

    std::vector<std::queue<_type>> init_qs(const int & n_nodes, const int & starting_node) {

        std::vector<std::queue<_type>> qs (n_nodes);

        for (int i = 0; i < n_nodes; ++i) {
            qs[i].push( adj[starting_node][0] );
        }

        return qs;
    }

    std::vector<int> get_idxs (const int & idx) {

        /* method returing indexes to iterate over for each node ---
         * number of threads correspond to the number of "idxs" vector
         * NOTICE: getting only indexes for nodes that:
         *      -- hasn't been visited yet
         *      -- candidate node isn't a leaf node */

        std::vector<int> idxs;
        for (const _type & val : adj[idx]) {
            if ( !vis[val.src] && adj[val.src].size() ) idxs.push_back( val.src );
        }

        return idxs;
    }

    int solve() {

        max_val = 0;

        for (int i = 0; i < N; ++i) {

            std::vector<int> idxs = get_idxs(i);
            int n_nodes = idxs.size();
            // std::vector<std::queue<_type>> qs = init_qs(n_nodes, starting_node);

            /* NOTICE: using CPU multithreading some MAX_THREAD_NUMBER
             *          and corresponding thread runner will probably be required
             */
            // extra loop for max number of threads - THREAD_MAX
            static const int THREAD_MAX = n_workers;
            const int iter_num = n_nodes / THREAD_MAX;
            if ( n_nodes <= THREAD_MAX ) continue;
            for (int idx = 0; idx < iter_num; ++idx) {
                const int idx1 = idx * THREAD_MAX,
                          idx2 = (idx + 1) * THREAD_MAX;
                int loc_iter_num = idx2 - idx1 + 1;
                std::vector<std::thread> ths (loc_iter_num);
                for (int t = 0; t < loc_iter_num; ++t) {
                    ths[t] = std::thread( &BFS::solve_one, this, adj[idxs[t + idx1]][0] );
                }
                for (int t = 0; t < loc_iter_num; ++t) {
                    ths[t].join();
                }
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
