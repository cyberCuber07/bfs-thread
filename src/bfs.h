
#ifndef __BFS_H_
#define __BFS_H_

#include <vector>
#include <string>
#include <queue>

#include "graph_converter.h"
#include "read_csv.h"


template <typename _type>
struct BFS {

    int N;
    vec1d<bool> vis;
    vec2d<_type> adj;
 
    template <typename InsertT>
    BFS (std::string path, InsertT insert)
    {
        ReadCSV reader;
        adj = edge2adj<_type>(reader.load(path), insert, N);
        vis.assign(N, false);
    }


    template <typename GetT>
    int solve(GetT getT) {

        int max_val(0);

        for (int i = 0; i < N; ++i) {
            max_val = std::max(max_val, solve_one(getT, i));
        }

        return max_val;
    }

    template <typename GetT>
    int solve_one(GetT getT, int idx) {

        /* in this implementation looking for maximum distance
         * between any two connected points in the graph */

        std::queue<_type> q;
        q.push(adj[idx][0]);

        int max_val = 0;

        while (!q.empty())
        {
            _type tmp = q.front();
            q.pop();

            if ( !vis[tmp.src] )
            {
                vis[tmp.src] = true;

                max_val = std::max(max_val, tmp.w);

                for (_type e : adj[tmp.src])
                {
                    q.push(e);
                }

            }

        }

        return max_val;
    }

};


#endif
