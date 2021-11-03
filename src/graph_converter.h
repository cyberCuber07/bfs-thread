
#ifndef __GRAPH_CONVERTER_H
#define __GRAPH_CONVERTER_H

#include <vector>

template <typename T>
using vec1d = std::vector<T>;

template <typename T>
using vec2d = std::vector<std::vector<T>>;


struct edge {
    int src = 0,
        dst = 0,
        w = 0;
};

struct dest {
    int dst, w;
};


int numEdges(const vec1d<edge> & edges) {

    int max_val(0);
    bool any (false);

    for (edge e : edges) {

        max_val = std::max(max_val, std::max(e.src, e.dst));
        if ( e.src == 0 || e.dst == 0 ) any = true;

    }

    return any ? max_val + 1 : max_val;

}


vec2d<dest> edge2adj(const vec1d<edge> & edges) {

    int N = numEdges(edges);
    vec2d<dest> adj(N);

    for (edge e : edges) {
        adj[e.src].push_back(dest({e.dst, e.w}));
    }

    return adj;

}



#endif
