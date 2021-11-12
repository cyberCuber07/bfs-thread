
#include "bfs-gpu.cu"


int main(int argc, char ** argv)
{
    std::string path = argv[1];
    BFS g( path );
    std::cout << g.max_val << "\n";
}
