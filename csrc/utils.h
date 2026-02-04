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

__device__ __forceinline__ float exp2f_ftz(float x) {
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float rcpf_ftz(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float quad_reduce_max_xor(float x) {
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 1), x);
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 2), x);

  return x;
}

__device__ __forceinline__ float quad_reduce_sum_xor(float x) {
  x += __shfl_xor_sync(0xFFFFFFFF, x, 1);
  x += __shfl_xor_sync(0xFFFFFFFF, x, 2);

  return x;
}

struct ReorgCFp8toAFp8{
  int selectorEx0;
  int selectorEx1;  
  int selectorEx4;
  int selectorEx5;
  
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

    // add static constexpr flag
    // use constant memory instead local memory
    static constexpr int upper_map[4] = {0,3,1,2};
    static constexpr int lower_map[4] = {1,2,0,3};
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
        lower0 = __shfl_sync(uint32_t(-1),lower0, upper_map[threadIdx.x%4] ^ 1,4);
    
        uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
        data_32bit[0] = __byte_perm(upper0, lower0, selectorEx4);
        data_32bit[1] = __byte_perm(upper0, lower0, selectorEx5);
        n += 8;
      }
    }
  }
};
