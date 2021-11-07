
#ifndef __READ_CSV_H
#define __READ_CSV_H


#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "graph_converter.h"

typedef std::vector<std::vector<std::string>> csv_data;


struct ReadCSV {

    ReadCSV () {}
    
    std::vector<std::string> get_row(std::string line, char delim = ',') {
        std::vector<std::string> row;
        std::stringstream s (line);
        std::string word;
        while(getline(s, word, delim)) {
            row.push_back(word);
        }
        return row;
    }
    
    
    csv_data read_csv(const std::string & path) {
    
        csv_data data;
        
        std::ifstream f (path, std::ios::in);
    
        std::string tmp;
        while ( f >> tmp ) {
            data.push_back(get_row(tmp));
        }
    
        f.close();
    
        return data;
    }
    
    
    std::vector<std::vector<double>> convert_data_type(const csv_data & data, int & ROW, int & COL) {
        ROW = data.size(), COL = data[0].size();
        std::vector<std::vector<double>> new_data (ROW, std::vector<double> (COL, 0.));
        for (int row = 0; row < ROW; ++row) {
            for (int col = 0; col < COL; ++col) {
                new_data[row][col] = stod(data[row][col]);
            }
        }
        return new_data;
    }
    
    
    std::vector<std::vector<double>> load(const std::string & path, int & ROW, int & COL) {
        csv_data data = read_csv(path);
        return convert_data_type(data, ROW, COL);
    }


    vec1d<edge> convert_data_type(const csv_data & data) {
        int ROW = data.size();
        vec1d<edge> new_data (ROW);
        for (int row = 0; row < ROW; ++row) {
            new_data[row] = {stoi(data[row][0]),
                             stoi(data[row][1]),
                             stoi(data[row][2])};
        }
        return new_data;
    }


    vec1d<edge> load(const std::string & path) {
        csv_data data = read_csv(path);
        return convert_data_type(data);
    }


    Edge * convert_data_type(const csv_data & data) {
        int ROW = data.size();
        Edge * new_data = new Edge[ROW];
        for (int row = 0; row < ROW; ++row) {
            new_data[row] = {stoi(data[row][0]),
                             stoi(data[row][1]),
                             stoi(data[row][2])};
        }
        return new_data;
    }


    Edge * load(const std::string & path) {
        csv_data data = read_csv(path);
        return convert_data_type(data);
    }

};


#endif
