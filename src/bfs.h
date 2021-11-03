
#ifndef __BFS_H_
#define __BFS_H_

#include <vector>
#include <string>

#include "graph_converter.h"
#include "read_csv.h"


struct BFS {

    vec2d<dest> data;
 
    BFS(std::string path) {

        ReadCSV reader;
        data = edge2adj(reader.load(path));

    }

};


#endif
