single:
	g++ src/single/main.cpp src/bfs.h src/read_csv.h src/graph_converter.h -o bin/single.bin

multi:
	g++ src/multi/main.cpp src/bfs-parallel.h src/read_csv.h src/graph_converter.h -o bin/multi.bin -lpthread

test:
	g++ tests/read_csv_tests.cpp src/read_csv.cpp -o bin/read_csv_tests.test
