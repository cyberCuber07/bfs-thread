#ifndef __UTILS_H_
#define __UTILS_H_

#include <iostream>


struct Edge {
    int src, dst, w;
};


std::ostream & operator << (std::ostream & os, Edge e) {
    os << e.src << " " << e.dst << " " << e.w << " | ";
    return os;
}


template <typename It>
void print (It start, It end) {
    while ( start != end ) {
        std::cout << *start++ << " ";
    }
    std::cout << "\n";
}

#endif
