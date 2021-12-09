single:
	g++ src/single/main.cpp src/bfs.h src/read_csv.h src/graph_converter.h -o bin/single.bin

multi:
	g++ src/multi/main.cpp src/bfs-parallel.h src/read_csv.h src/graph_converter.h -o bin/multi.bin -lpthread

cuda:
	nvcc src/bfs-gpu.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/cuda.bin

cuda-third:
	nvcc src/bfs-gpu-third.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/cuda-third.bin

cuda-second:
	nvcc src/bfs-gpu-second.cu src/queue.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/cuda-second.bin

multi-test:
	nvcc tests/multi.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/multi.test

redux:
	nvcc tests/template-reduce.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/temp-reduce.test
