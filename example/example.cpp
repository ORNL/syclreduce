#include <stdio.h>

#include <syclreduce/reduce.hpp>

namespace SR = syclreduce;

int main(int argc, char *argv[]) {
	sycl::queue queue;

    struct FourInts {
        int x[4];
        FourInts() {}
        FourInts(int i) : x{1, i,i,i} {}
		const int& operator[](int i) const { return x[i]; }
		int& operator[](int i) { return x[i]; }
    };
    struct ReduceFour {
        using T = FourInts; // What we're creating.

        // Initial value for all reduction variables.
        // This is not called often by the library, but
        // is an important part of the mathematics.
        static void identity(FourInts &a) {
            a[0] = 0; a[1] = 0;
            a[2] = ~((int)1<<(sizeof(int)*8-1));
            a[3] =   (int)1<<(sizeof(int)*8-1);
		}

		// How to combine two reduction results.
        // Must be associative.
        // Does not need to be commutative.
        static void combine(FourInts &a, const FourInts &b) {
            a[0] += b[0]; // count
            a[1] += b[1]; // sum
            a[2] = b[2] < a[2] ? b[2] : a[2]; // min
            a[3] = b[3] < a[3] ? a[3] : b[3]; // max
		}
    };
    SR::Reducer<ReduceFour> result;

    // Like parallel_for, but kernel functions all return
    // a ReduceFour by value.
	queue.submit([&](sycl::handler &cgh){
		SR::parallel_reduce(cgh, sycl::nd_range({4096, 32}),
							result, [=](sycl::nd_item<1> it) {
			const size_t tid = it.get_global_id(0);

			return FourInts(tid*12345 % 4792 + 101);
		});
    });

    FourInts ans = result.get();
	printf("%d %d %d %d\n", ans.x[0], ans.x[1], ans.x[2], ans.x[3]);
}
