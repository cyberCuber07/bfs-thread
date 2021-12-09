
#include <iostream>

#include "include/utils.cu"
#include "include/solver.cuh"


int main(int argc, char ** argv)
{
    std::string path = argv[1];
    BFS g( path );
    std::cout << g.max_val << "\n";
}
