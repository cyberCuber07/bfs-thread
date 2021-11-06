
#ifndef __GRAPH_CONVERTER_H
#define __GRAPH_CONVERTER_H

#include <vector>
#include <iostream>

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


int numEdges(vec1d<edge> & edges) {

    int max_val(0);
    bool any (false);

    for (edge e : edges) {

        max_val = std::max(max_val, std::max(e.src, e.dst));
        if ( e.src == 0 || e.dst == 0 ) any = true;

    }

    if (!any) {
        std::cout << "Decreasing values!\n";
        // shift values by one (so the min := 0 and max := N - 1)
        for (auto & e : edges) {
            --e.src;
            --e.dst;
        }
        return max_val;
    }

    return max_val + 1;

}


edge insertEdge(edge e) {
    return e;
}


dest insertDest(edge e) {
    return {e.dst, e.w};
}


template <typename T, typename TInserter>
vec2d<T> edge2adj(vec1d<edge> edges, TInserter inserter, int & N) {

    N = numEdges(edges);
    vec2d<T> adj(N, std::vector<T>(0));

    for (edge e : edges) {
        adj[e.src].push_back(
                inserter(e)
                );
    }

    return adj;

}


template <typename T>
std::ostream & operator << (std::ostream & os,
                            const vec2d<T> & data)
{
    for (int i = 0; i < data.size(); ++i) {
        os << "# " << i << " #\n";
        for (T d : data[i]) {
            // os << d << "\n";
            os << d.dst << " " << d.w << " | ";
        }
        os << "\n";
    }
    return os;
}


#endif
