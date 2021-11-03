build:
	g++ src/main.cpp src/bfs.h src/read_csv.h src/graph_converter.h -o bin/test.bin

test:
	g++ tests/read_csv_tests.cpp src/read_csv.cpp -o bin/read_csv_tests.test
