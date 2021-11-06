
#ifndef __BFS_H_
#define __BFS_H_

#include <vector>
#include <string>
#include <queue>

#include "graph_converter.h"
#include "read_csv.h"


template <typename _type>
struct BFS {

    int N, max_val;
    vec1d<bool> vis;
    vec2d<_type> adj;
 
    template <typename InsertT>
    BFS (std::string path, InsertT insert)
    {
        ReadCSV reader;
        adj = edge2adj<_type>(reader.load(path), insert, N);
        vis.assign(N, false);
    }


    int solve() {

        max_val = 0;

        for (int i = 0; i < N; ++i) {
            if ( adj[i].size() > 0 )
                solve_one(adj[i][0]);
        }

        return max_val;
    }

    void node_run(std::queue<_type> & q, _type tmp) {

        max_val = std::max(max_val, tmp.w);
        
        for (_type e : adj[tmp.src])
        {
            q.push(e);
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
