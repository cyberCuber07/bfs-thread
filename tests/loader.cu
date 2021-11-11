
#include <iostream>
#include <fstream>


struct LL {
    struct LL * next;
    int val;
};


__global__
void insert(LL ** head, int val) {
    LL * new_node = new LL();
    new_node -> val = val;
    if ( *head == NULL ) {
        *head = new_node;
        return;
    }
    LL * tmp = * head;
    while( tmp -> next != NULL ) {
        tmp = tmp -> next;
    }
    tmp -> next = new_node;
}


void print(LL * head) {
    while ( head != NULL ) {
        std::cout << head -> val << " ";
        head = head -> next;
    }
    std::cout << "\n";
}


__global__
void mod(LL ** ll) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    insert <<< 1, 1 >>> (ll, tid);
}



int main()
{
    const int N = 1024, M = 2;
    int blockSize = N,
        gridSize = M / N + 1;
    LL * d_ll;

    cudaMalloc( &d_ll, sizeof(LL) * 1024 );
    mod <<< blockSize, gridSize >>> (&d_ll);
    cudaDeviceSynchronize();

    LL * h_ll = new LL[1024];
    cudaMemcpy( h_ll, d_ll, sizeof(LL) * 1024, cudaMemcpyDeviceToHost );

    for (int i = 0;  i < 1024; ++i) std::cout << h_ll[0].val << " ";
    std::cout << "\n";
}
