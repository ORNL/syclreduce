#ifndef _SYCLREDUCE_REDUCE_HPP
#define _SYCLREDUCE_REDUCE_HPP

#include <type_traits>
#include <sycl/sycl.hpp>

namespace syclreduce {

// R should follow the Reduction Interface:
//
// typename T
// void identity(T &) const {}
// void combine(const T &, T &) const {}

/** A reducer applies a Reduction to data placed
 *  into its buffer.
 */
template <typename R>
class Reducer {
	friend class Reducer<R>;

    sycl::buffer<typename R::T,1> buf;
	size_t ngrp;
	R op; // const, but we need to overwrite it on operator=

  public:
    using Reduction = R;
    using T = typename R::T;

	/** Note: these constructors all place the identity element in the
	 *  first buffer entry.  This ensures that we can always fall-back
	 *  to the SYCL implementation's reduction.
	 */
	Reducer(const Reduction &op) : buf(1), ngrp(0), op(op) {
        sycl::host_accessor X(buf, sycl::write_only, sycl::no_init);
		op.identity(X[0]);
	}
	Reducer(const Reduction &op, size_t _ngrp)
			: buf(_ngrp), ngrp(_ngrp), op(op) {
		if(_ngrp < 1) {
			throw std::invalid_argument("Invalid buffer size.");
		}
		sycl::host_accessor X(buf, sycl::write_only, sycl::no_init);
		op.identity(X[0]);
	}
	Reducer(const Reducer<R> &other, size_t _ngrp)
			: buf(other.buf), ngrp(other.ngrp), op(other.op) {
		realloc(_ngrp);
	}
	void realloc(size_t _ngrp) {
		if(_ngrp < 1) {
			throw std::invalid_argument("Invalid buffer size.");
		}
		if(ngrp < _ngrp) { // actually need to realloc
			buf = sycl::buffer<typename R::T,1>(_ngrp);
			sycl::host_accessor X(buf, sycl::write_only, sycl::no_init);
			op.identity(X[0]);
		}
		ngrp = _ngrp;
	}

	auto accessor(sycl::handler &cgh) {
		return sycl::accessor(buf, cgh, sycl::write_only, sycl::no_init);
	}
	sycl::buffer<typename R::T,1> &buffer() {
		return buf;
	}

    const R oper() {
		return op;
	}

    T get() {
	    T ret;
		if(ngrp == 0) {
			op.identity(ret);
		} else {
			sycl::host_accessor X(buf, sycl::read_only);
			ret = X[0];
			for(size_t i=1; i<ngrp; ++i) {
				op.combine(ret, X[i]);
			}
		}
		return ret;
	}
};

// Some helpful operations
template <typename TT>
struct Sum {
	using T = TT;
	void identity(T &x) const {
		x = (T)0;
	}
	void combine(T &x, const T &y) const {
		x += y;
	}
};

template <typename TT>
struct Prod {
	using T = TT;
	void identity(T &x) const {
		x = (T)1;
	}
	void combine(T &x, const T &y) const {
		x *= y;
	}
};

/** Custom parallel reduce.
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
	red.realloc(ngrp);
	auto X = red.accessor(cgh);

	sycl::local_accessor<T,1> wg_sum(nlocal, cgh);
								//sycl::read_write, sycl::no_init);

	cgh.parallel_for(rng, [=,op=red.oper()](sycl::nd_item<Dim> it) {
		T val = kernel(it);

		sycl::group<Dim> grp = it.get_group();

		// Reduce over work items in this wg.
		//
		// TODO: make use of sycl::sub_group shift operators.
		//
		//   This requires specializing the algo. for different
		//   possible sub_group sizes.
		int li         = grp.get_local_linear_id();
		int local_size = grp.get_local_linear_range();
		wg_sum[li] = val;
		for (int offset = 1; offset < local_size; offset *= 2) {
			sycl::group_barrier(grp);
			if(li + offset < local_size)
				op.combine(wg_sum[li], wg_sum[li + offset]);
		}
		if (li == 0)
			X[it.get_group(0)] = wg_sum[0];
	});
}

/*
template <int Dim, typename Reducer, typename Kernel>
void parallel_reduce(
		sycl::handler &cgh, sycl::range<Dim> rng,
		Reducer &red,
		const Kernel &kernel) {
	// should statically assert this:
	//using T = invoke_result_t<Kernel, nd_item<Dim> >;
	using Reduction = typename Reducer::Reduction;
	using T = typename Reducer::T;
	size_t ngrp   = 16;
	size_t nlocal = 1;
	auto X = ans.accessor(cgh);
	sycl::nd_range<Dim> nrng( ... );

	red.realloc(1);
	sycl::local_accessor<T,1> wg_sum(nlocal, cgh);
								//sycl::read_write, sycl::no_init);

	cgh.parallel_for(nrng, [=,op=red.oper()](sycl::nd_item<Dim> it) {
		T val = kernel(it);

		sycl::group<Dim> grp = it.get_group();

		int li         = grp.get_local_linear_id();
		int local_size = grp.get_local_linear_range();
		wg_sum[li] = val;
		for (int offset = 1; offset < local_size; offset *= 2) {
			sycl::group_barrier(grp);
			if(li + offset < local_size)
				op.combine(wg_sum[li], wg_sum[li + offset]);
		}
		if (li == 0)
			X[it.get_group(0)] = wg_sum[0];
	});
}*/

///
/// Definition for compatibility with SYCL reduce spec.
template <typename R>
struct BinaryOp : public R {
	using Reduction = R;
	using T = typename R::T;
	BinaryOp(const Reduction &x) : R(x) {}
    T operator()(const T& x, const T& y) const {
		T z = x;
		this->combine(z, y);
		return z;
	}
};

// detection idiom https://en.cppreference.com/w/cpp/experimental/is_detected
namespace detail {
	template<typename = void, typename... Args>
	struct detect_reduction : std::false_type {};

	template<typename... Args>
	struct detect_reduction<std::void_t<decltype(sycl::reduction(std::declval<Args>()...))>, Args...>
		: std::true_type {};

	template<typename... Args>
	inline constexpr bool detect_reduction_v = detect_reduction<void, Args...>::value;

	/// Work-around for hipsycl's early reduction API that required
	/// accessors instead of buffers.
	template <typename T, int dim, typename BinaryOp, typename = void>
	auto reduction(sycl::buffer<T,dim> &buf, sycl::handler &cgh, BinaryOp &op) {
		if constexpr(detect_reduction_v<sycl::buffer<T,dim>&,
												sycl::handler&, BinaryOp>) {
			return sycl::reduction(buf, cgh, op);
		} else {
			sycl::accessor acc(buf, cgh, sycl::read_write);
			typename BinaryOp::T identity;

			op.identity(identity);
			return sycl::reduction(acc, identity, op);
		}
	}
}

template <int Dim, typename Reducer, typename Kernel>
void parallel_reduce(
		sycl::handler &cgh, sycl::range<Dim> rng,
		Reducer &red, const Kernel &kernel) {
	// should statically assert this:
	//using T = invoke_result_t<Kernel, nd_item<Dim> >;
	using Reduction = typename Reducer::Reduction;
	using T = typename Reducer::T;
	red.realloc(1);

	BinaryOp<Reduction> BR{red.oper()};

	cgh.parallel_for(rng, detail::reduction(red.buffer(), cgh, BR),
					 [=](sycl::item<Dim> id, auto &ret) {
		ret.combine( kernel(id) );
    });
}

/** Assuming a work group / subgroup layout dividing
 *  members via sg.get_group_linear_id() and sg.get_local_linear_id()
 *
 *  into blocks of P x Q, we have (e.g. for grp.range x sg.range = 4x16
 *  and PxQ = 2x8,
 *
 *      |               |        
 * - 0: 0 1 2 3 4 5 6 7 8 9 A B C D E F
 *   1: 0 1 2 3 4 5 6 7 8 9 A B C D E F
 * - 2: 0 1 2 3 4 5 6 7 8 9 A B C D E F
 *   3: 0 1 2 3 4 5 6 7 8 9 A B C D E F
 *
 * During reduction, we accumulate 4 results -- one for each
 * PxQ sub-block.  This is accomplished by doing a row-wise reduction
 * to sub_group id-s 0 and 8, followed by a column-wise reduction
 * among all "sg.get_group_linear_range()/P"
 *                   x "sg.get_local_linear_range()/Q" sub-group leaders.
 *  
 */

/** Reduce over all vector lanes within a sub_group.
 *  (i.e. column-wise)
 *
 *  A final result will be present on elements where:
 *    sg.get_local_linear_id() % Q == 0
 *
 *  It is assumed that sg.get_local_linear_range() % Q == 0.
 *
 */
template <int Q, typename Reduction>
void subgroup_reduce(sycl::sub_group sg,
					 const Reduction &op, typename Reduction::T &x) {
	using T = typename Reduction::T;
	const int tid = sg.get_local_linear_id();

    for(int z=1; z<Q; z*=2) {
		T y = sycl::shift_group_left(sg, x, z);
		if(tid+z < Q) {
			op.combine(x, y);
		}
    }
}

/** Shared memory reduction over all sub_group leaders.
 *  (i.e. row-wise)
 *
 *  Uses shared memory equal to rng := sg.get_group_linear_range()
 *
 *  A final result will be present on elements where:
 *    sg.get_group_linear_id() % P == 0
 *
 *  It is assumed that len(tmp) == rng.
 *
 *  To work with the kind of 2D reductions described above,
 *  call this on every sub-group thread where wg.get_local_linear_id()%Q == 0,
 *  and provide a unique shared memory address (tmp) for each
 *     q = wg.get_local_linear_id()/Q.
 *
 */
template <int P, typename Reduction>
void shmem_reduce(sycl::group<1> grp, sycl::sub_group sg, const Reduction &op,
				  typename Reduction::T &x, typename Reduction::T *tmp) {
	using T = typename Reduction::T;
	const int gid = sg.get_group_linear_id();
	const int rng = sg.get_group_linear_range(); // len(tmp)

	tmp[gid] = x;
    for(int z=1; z<P; z*=2) {
		sycl::group_barrier(grp);
		if(gid+z < P) {
			op.combine(tmp[gid], tmp[gid+z]);
		}
    }
	if(gid % P == 0)
		x = tmp[gid];
}

}
#endif
