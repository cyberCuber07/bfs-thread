
#ifndef __BFS_H_
#define __BFS_H_

#include <vector>
#include <string>
#include <queue>

#include "graph_converter.h"
#include "read_csv.h"


struct BFS {

    vec2d<edge> data;
 
    BFS(std::string path) {

        ReadCSV reader;
        data = edge2adj<edge>(reader.load(path), insertEdge);

        std::cout << data << "\n";

    }


    void solve() {

        std::queue<edge> q;

    }

};


#endif
