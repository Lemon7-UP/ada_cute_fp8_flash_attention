#pragma once

#include <cute/tensor.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"

template <typename Tensor>
__device__ __forceinline__ constexpr auto retile_fragment(Tensor &&tensor) {
  using namespace cute;

  // (MMA, MMA_M, MMA_N)
  constexpr int R = decltype(tensor.layout())::rank;
  static_assert(R == 3, "we only support rank 3 fragment");

  auto thr_vmk = flatten(select<0>(tensor.layout()));
  auto tile_mk = select<1, 2>(tensor.layout());

  auto m_layout =
      coalesce(make_layout(make_shape(get<1>(thr_vmk.shape()), get<0>(tile_mk.shape())),
                           make_stride(get<1>(thr_vmk.stride()), get<0>(tile_mk.stride()))));
                  
  auto k_layout = coalesce(make_layout(
      make_shape(get<0>(thr_vmk.shape()), get<1>(tile_mk.shape())),
      make_stride(get<0>(thr_vmk.stride()),  get<1>(tile_mk.stride()))));

  return make_tensor(static_cast<Tensor &&>(tensor).data(), make_layout(m_layout, k_layout));
}

template<typename T>
struct MaxOp {
__device__ inline T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
__device__ inline float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void thread_reduce_(cute::Tensor<Engine0, Layout0> const &tensor, cute::Tensor<Engine1, Layout1> &summary, Operator &op) {
    using namespace cute;
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(cute::Tensor<Engine0, Layout0> &dst, cute::Tensor<Engine1, Layout1> &src, Operator &op) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op); 
    }
}


template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void reduce_(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}


template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_max(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_sum(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    reduce_(tensor, sum, sum_op);
}


struct ReorgCFp8toAFp8{
  int selectorEx0;
  int selectorEx1;  
  int selectorEx4;
  int selectorEx5;
  int upper_map[4] = {0,3,1,2};
  int lower_map[4] = {1,2,0,3};
  
  CUTLASS_DEVICE ReorgCFp8toAFp8() {
    int laneId = cutlass::canonical_lane_idx();
    
    if (laneId % 4 == 0 || laneId % 4 == 3) {
      selectorEx0 = 0x3210;
      selectorEx1 = 0x7654;
      selectorEx4 = 0x5410;
        selectorEx5 = 0x7632;
    } else {
      selectorEx0 = 0x7654;
      selectorEx1 = 0x3210;
      selectorEx4 = 0x1054;
        selectorEx5 = 0x3276;
    }  
  }

  template <typename Fragment>
  CUTLASS_DEVICE void operator()(Fragment &accum) {

  using namespace cute;  

  auto VT = shape<0>(accum); // number of vector elements per tile.
  auto MT = shape<1>(accum); // number of tiles along M.
  auto NT = shape<2>(accum); // number of tiles along N.

  auto data = accum.data();
  int n = 0;

#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
      for (int k = 0; k < NT / 2; ++k) {
        auto upper = *reinterpret_cast<uint32_t*>(&data[n]); 
        auto lower = *reinterpret_cast<uint32_t*>(&data[n+4]); 
        
        auto upper0 = __byte_perm(upper, lower, selectorEx0);
        auto lower0 = __byte_perm(upper, lower, selectorEx1);      
        upper0 = __shfl_sync(uint32_t(-1),upper0, upper_map[threadIdx.x%4],4);
        lower0 = __shfl_sync(uint32_t(-1),lower0, lower_map[threadIdx.x%4],4);
    
        uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
        data_32bit[0] = __byte_perm(upper0, lower0, selectorEx4);
        data_32bit[1] = __byte_perm(upper0, lower0, selectorEx5);
        n += 8;
      }
    }
  }
};
