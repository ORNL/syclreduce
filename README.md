# SYCL Reduce Primitive

This is a tiny package implementing what is a giant
unmet need in SYCL2020 - proper reductions.

Want to sum a vector coming from every thread in a kernel
launch?  Want to accumulate a couple different kinds
of diagnostic output from a kernel?  Too bad.  SYCL doesn't
have full documentation on how span<> works, and you'll easily
get lost writing your own __undefined type__ reducer.

So, instead, try this:

    #include <syclreduce/reduce.hpp>

    ...

    namespace SR = syclreduce;

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
        void identity(FourInts &a) const {
            a[0] = 0; a[1] = 0;
            a[2] = ~((int)1<<(sizeof(int)*8-1));
            a[3] =   (int)1<<(sizeof(int)*8-1);
		}

		// How to combine two reduction results.
        // Must be associative.
        // Does not need to be commutative.
        void combine(FourInts &a, const FourInts &b) const {
            a[0] += b[0]; // count
            a[1] += b[1]; // sum
            a[2] = b[2] < a[2] ? b[2] : a[2]; // min
            a[3] = b[3] < a[3] ? a[3] : b[3]; // max
		}
    };
    SR::Reducer result{ ReduceFour() };

    // Like parallel_for, but kernel functions all return
    // a ReduceFour by value.
    SR::parallel_reduce(cgh, sycl::nd_range({4096, 32}),
				        result, [=](sycl::nd_item<1> it) {
        const size_t tid = it.get_global_id(0);

        return FourInts(tid*31337 % 4792 + 101);
    });

    FourInts ans = result.get();

The reduction is done internally with the following steps:

1. Reducing the return results from every work group in parallel,
   and storing that in a unique position in an internal buffer.

2. Copying the buffer to the host and reducing over those results.

For a work group size of `sycl::nd_range(12,4)`,
this works out like:

   out[0] = op(op(0,1), op(2,3))
   out[1] = op(op(4,5), op(6,7))
   out[2] = op(op(8,9), op(10,11))

   result = op( op(out[0], out[1]) , out[2] )

Note that you should be careful about kernel launch sizes using
this method.  In particular, don't use more than, say, 4x the
number of work groups as you have compute units.  Otherwise you
are wasting memory and storing more intermediate reduction results
than you need to.

We're assuming you want already use `sycl::nd_range` to
control launch sizes for this reason.


# Installing

This is a header-only package.  You can just copy it into an
include directory like `/usr/local/include/syclreduce/reduce.hpp`.

If you also want to test and install the cmake package description,
do the following:

    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=`which syclcc` \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          ..
    make install

# Copyright and License

Copyright 2023 UT-Battelle LLC.  See LICENSE.md for license details.
