single:
	g++ src/single/main.cpp src/bfs.h src/read_csv.h src/graph_converter.h -o bin/single.bin

multi:
	g++ src/multi/main.cpp src/bfs-parallel.h src/read_csv.h src/graph_converter.h -o bin/multi.bin -lpthread

cuda:
	nvcc src/bfs-gpu.cu -o bin/cuda.bin

cuda-new:
	g++ src/bfs-gpu.cpp -o bin/cuda-new.bin -lpthread

test:
	g++ tests/read_csv_tests.cpp src/read_csv.cpp -o bin/read_csv_tests.test

cuda-single:
	g++ tests/cuda/single.cpp -o bin/cuda-single.test

cuda-multi:
	nvcc tests/cuda/multi.cu -o bin/cuda-multi.test

cuda-double:
	nvcc tests/cuda/second.cu -o bin/cuda-double.test

cuda-triple:
	nvcc tests/cuda/thrid.cu -o bin/cuda-third.test

example:
	nvcc tests/example.cu -o bin/example.test
