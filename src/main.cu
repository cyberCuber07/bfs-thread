
#include <iostream>

#include "include/utils.cu"
#include "include/solver.cuh"
#include "kernels/one.cu"
#include "kernels/third.cu"


int main(int argc, char ** argv)
{
    std::string path = argv[1];
    BFS g( path );
    g.solve(solveOne);
    std::cout << g.max_val << "\n";
}
