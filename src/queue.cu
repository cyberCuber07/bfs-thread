

namespace DataStructs{

    template <typename T>
    struct Queue {

        struct node {
            struct node * next;
            T val;
        };

        node * head, * back;

        __host__ __device__
        Queue() {
            head = nullptr;
            back = nullptr;
        }

        __host__ __device__
        ~Queue() {
            destroy();
        }

        __host__ __device__
        void destroy() {
            while (!empty()) pop();
        }

        __host__ __device__
        void push(const T val) {

            node * new_node = new node();
            new_node -> val = val;
            new_node -> next = nullptr;

            if ( back == nullptr ) {
                head = back = new_node;
                return;
            }

            back -> next = new_node;
            back = new_node;
        }

        __host__ __device__
        void pop() {

            if ( head ==  nullptr ) return;

            node * tmp = head;
            head = head -> next;

            if ( head == nullptr ) {
                back = nullptr;
            }

            delete tmp;
        }

        __host__ __device__
        T top() {
            if ( back != nullptr ) return head -> val;
            return T();
        }

        __host__ __device__
        bool empty() {
            return head == nullptr;
        }

    };

}
