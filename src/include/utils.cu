#ifndef __UTILS_H_
#define __UTILS_H_

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

#include "queue.cu"

using namespace DataStructs;

template <typename T>
using vec2d = std::vector<std::vector<T>>;
template <typename T>
using vec1d = std::vector<T>;


struct ReadCSV {

    ReadCSV() {}

    Edge get_row(std::string line, char delim = ',') {
        std::stringstream s (line);
        std::string word;
        // NOTICE: works for edge form only!!
        Edge tmp;
        getline(s, word, delim);
        tmp.src = stoi(word);
        getline(s, word, delim);
        tmp.dst = stoi(word);
        getline(s, word, delim);
        tmp.w = stoi(word);
        return tmp;
    }

    vec1d<Edge> read_csv(const std::string & path) {

        vec1d<Edge> data;

        std::ifstream f (path, std::ios::in);

        std::string tmp;
        while ( f >> tmp ) {
            data.push_back(get_row(tmp));
        }

        f.close();

        return data;
    }

    int getMaxVal(const vec1d<Edge> & data) {
        int max_val(0);
        for (auto & tmp : data) {
            max_val = std::max(max_val, std::max(tmp.src, tmp.dst));
        }
        return max_val + 1; // works for 0, ..., N - 1 [N]
    }

    vec2d<Edge> edge2adj(const vec1d<Edge> & data) {

        int max_val = getMaxVal(data);
        vec2d<Edge> adj(max_val);

        for (auto & tmp : data) {
            adj[tmp.src].push_back(tmp);
        }

        return adj;
    }

    Edge * adj2edge(const vec2d<Edge> & adj, const int & N) {
        // return sorted edge list
        Edge * edges = new Edge[N];
        // convert
        int cnt = 0;
        for (auto & vec : adj) {
            for (auto & e : vec) {
                edges[cnt++] = e;
            }
        }
        return edges;
    }

    Edge * load(const std::string & path, int & N, int & M) {
        vec1d<Edge> data = read_csv(path);
        N = data.size();
        // --- SORT --- //
        // save to adj list
        vec2d<Edge> adj = edge2adj(data);
        M = adj.size();
        // move back to edge list
        return adj2edge(adj, N);
    }

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
