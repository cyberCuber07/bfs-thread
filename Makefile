single:
	nvcc src/main-single.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/cuda-single.bin

multi:
	g++ src/multi/main.cpp src/bfs-parallel.h src/read_csv.h src/graph_converter.h -o bin/multi.bin -lpthread

cuda:
	nvcc src/main.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/cuda.bin


# TESTS
multi-test:
	nvcc tests/multi.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/multi.test

redux:
	nvcc tests/template-reduce.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/temp-reduce.test
