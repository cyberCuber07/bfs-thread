single:
	g++ src/single/main.cpp src/bfs.h src/read_csv.h src/graph_converter.h -o bin/single.bin

multi:
	g++ src/multi/main.cpp src/bfs-parallel.h src/read_csv.h src/graph_converter.h -o bin/multi.bin -lpthread

cuda:
	nvcc src/bfs-gpu.cu src/queue.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/cuda.bin

cuda-second:
	nvcc src/bfs-gpu-second.cu src/queue.cu -arch=sm_35 -rdc=true -lcudadevrt -o bin/cuda-second.bin
