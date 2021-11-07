

typedef long long _type;

int main()
{
    const _type N = 1 << 28;
    _type * nums = new _type[N];

    for (_type i = 0; i < N; ++i) {
        nums[i] = i;
    }


    delete[] nums;
}
