
#include "../bfs-gpu.cu"
#include <string>
#include <iostream>

int main(int argc, char ** argv)
{
    std::string path = argv[1];
    int n_workers = 1000;
    BFS<edge> bfs(path, insertEdge, n_workers);

    std::cout << "For: " << n_workers << " number of threads\n" << bfs.solve() << "\n";
}
