
#include <iostream>

#include "include/utils.cu"
#include "include/solver.cuh"
#include "kernels/caller-one.cu"
#include "kernels/caller-third.cu"


int main(int argc, char ** argv)
{
    std::string path = argv[1];
    BFS g( path );
    g.solve(solveThird);
    std::cout << g.max_val << "\n";
}
