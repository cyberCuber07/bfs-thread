
#include "bfs.h"
#include <string>
#include <iostream>

int main(int argc, char ** argv)
{
    std::string path = argv[1];
    BFS<edge> bfs(path, insertEdge);

    std::cout << bfs.solve(insertEdge) << "\n";
}
