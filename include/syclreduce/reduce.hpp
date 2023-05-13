#ifndef _SYCLREDUCE_REDUCE_HPP
#define _SYCLREDUCE_REDUCE_HPP

#include <sycl/sycl.hpp>

namespace syclreduce {

// R should follow the Reduction Interface:
//
// typename T
// static void identity(T &) {}
// static void combine(const T &, T &) {}

/** A reducer applies a Reduction to data placed
 *  into its buffer.
 */
template <typename R>
class Reducer {
    sycl::buffer<typename R::T,1> buf;
	size_t ngrp;

  public:
    using Reduction = R;
    using T = typename R::T;

	Reducer() : buf(0), ngrp(0) {}
	Reducer(size_t _ngrp) : buf(_ngrp), ngrp(_ngrp) {}

	auto accessor(sycl::handler &cgh) {
		return sycl::accessor(buf, cgh, sycl::write_only, sycl::no_init);
	}

    T get() {
	    T ret;
		if(ngrp == 0) {
			R::identity(ret);
		} else {
			sycl::host_accessor X(buf, sycl::read_only);
			ret = X[0];
			for(size_t i=1; i<ngrp; ++i) {
				R::combine(ret, X[i]);
			}
		}
		return ret;
	}
};

// Some helpful operations
template <typename TT>
struct Sum {
	using T = TT;
	static void identity(T &x) {
		x = (T)0;
	}
	static void combine(T &x, const T &y) {
		x += y;
	}
};

template <typename TT>
struct Prod {
	using T = TT;
	static void identity(T &x) {
		x = (T)1;
	}
	static void combine(T &x, const T &y) {
		x *= y;
	}
};

/** Replaces red with a new reducer.
 */
template <int Dim, typename Reducer, typename Kernel>
void parallel_reduce(
		sycl::handler &cgh, sycl::nd_range<Dim> rng,
		Reducer &red,
		const Kernel &kernel) {
	// should statically assert this:
	//using T = invoke_result_t<Kernel, nd_item<Dim> >;
	using Reduction = typename Reducer::Reduction;
	using T = typename Reducer::T;
	size_t ngrp   = rng.get_group_range().size();
	size_t nlocal = rng.get_local_range().size();
	Reducer ans( ngrp );
	auto X = ans.accessor(cgh);

	red = ans;
	sycl::local_accessor<T,1> wg_sum(nlocal, cgh);
								//sycl::read_write, sycl::no_init);

	cgh.parallel_for(rng, [=](sycl::nd_item<Dim> it) {
		T val = kernel(it);

		sycl::group<Dim> grp = it.get_group();

		// reduce over work items in this wg
		// TODO: make use of sycl::sub_group shift operators
		int li         = grp.get_local_linear_id();
		int local_size = grp.get_local_linear_range();
		wg_sum[li] = val;
		for (int offset = 1; offset < local_size; offset *= 2) {
			sycl::group_barrier(grp);
			if(li + offset < local_size)
				Reduction::combine(wg_sum[li], wg_sum[li + offset]);
		}
		if (li == 0)
			X[it.get_group(0)] = wg_sum[0];
	});
}

}
#endif
