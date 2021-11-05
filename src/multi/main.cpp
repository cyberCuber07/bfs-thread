
#include "../bfs-parallel.h"
#include <string>
#include <iostream>

int main(int argc, char ** argv)
{
    std::string path = argv[1];
    int n_workers = 6;
    BFS<edge> bfs(path, insertEdge, n_workers);

    std::cout << bfs.solve() << "\n";
}
