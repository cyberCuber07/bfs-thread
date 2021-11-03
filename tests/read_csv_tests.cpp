#include "../src/read_csv.cpp"
#include <vector>
#include <iostream>

typedef std::vector<std::vector<double>> vec2d;
typedef std::vector<double> vec1d;

void print_vec(const vec2d & data) {
    for (vec1d vec : data) {
        for (double val : vec) std::cout << val << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char ** argv)
{
    ReadCSV reader;

    int h, w;
    vec2d data = reader.load(argv[1], h, w);

    print_vec(data);
}
