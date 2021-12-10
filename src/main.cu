
#include <iostream>

#include "include/utils.ch"
#include "include/solver.ch"
#include "kernels/one.ch"
#include "kernels/third.ch"


int main(int argc, char ** argv)
{
    std::string path = argv[1];
    BFS g( path );
    g.solve(solveThree);
    // g.solveSingleCaller();
    std::cout << g.max_val << "\n";
}
