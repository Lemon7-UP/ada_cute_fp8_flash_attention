#include <cstdint>
#include <type_traits>

#include "attention_api.cuh"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <torch/python.h>
#include <vector>

#include "kernel_traits.h"
#include "flash.h"
#include "utils.h"


namespace flash {
// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
  template <int N>
  CUTE_HOST_DEVICE
  void cp_async_wait() {
  #if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
      asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
  #endif
  }
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(cute::Tensor<Engine0, Layout0> &tensor, cute::Tensor<Engine1, Layout1> const &max, const float scale) {
    using namespace cute;
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = expf(tensor(mi, ni) * scale - max_scaled);
        }
    }
}


template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, float softmax_scale) {
    using namespace cute;
    if (Is_first) {
        // no need to reduce and rescale with prev result
        reduce_max<true>(scores, scores_max);
        scale_apply_exp2(scores, scores_max, softmax_scale);
        reduce_sum(scores, scores_sum);
    } else {
        // store prev max
        Tensor scores_max_prev = make_fragment_like(scores_max); 
        cute::copy(scores_max, scores_max_prev);
       
        reduce_max<false>(scores, scores_max); 
        
        // traverse each row
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) { 
            float scores_max_cur = scores_max(mi); 
            // scale factor
            float scores_scale = expf((scores_max_prev(mi) - scores_max_cur) * softmax_scale); 
            // scale sum
            scores_sum(mi) *= scores_scale; 
            #pragma unroll
            // scale output
            for (int ni = 0; ni < size<1>(acc_o); ++ni) { acc_o(mi, ni) *= scores_scale; } 
        }

        // softmax
        scale_apply_exp2(scores, scores_max, softmax_scale);

        // sum scores
        Tensor scores_sum_cur = make_fragment_like(scores_sum); 
        reduce_sum(scores, scores_sum_cur); 
        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) { scores_sum(mi) += scores_sum_cur(mi); } 
    }
};

// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

template <typename Kernel_traits, typename Params>
__global__ void flash_attn_fp8_kernel(const Params params) {
  using namespace cute;

  // because thread block will be launched in BlockIdx.x order first
  // so utilize this to improve L2 cache hit rate
  const int m_block = blockIdx.x;

  const int base_id = blockIdx.y;
  const int tidx = threadIdx.x;

  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int kTileM = Kernel_traits::kTileM;
  constexpr int kTileN = Kernel_traits::kTileN;
  constexpr int kTileK = Kernel_traits::kTileK;
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using TiledMMA = typename Kernel_traits::TiledMma;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutK;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutV;
  using SmemLayoutO = typename Kernel_traits::SmemLayoutO;
  
  // KV tiling range
  const int seqlen_q_start = m_block * kTileM;
  const int seqlen_q_end = (m_block + 1) * kTileM;
  const int KV_Tile_Min = 0;
  const int KV_Tile_Max = cute::ceil_div(seqlen_q_end, kTileN); 

  extern __shared__ uint8_t smem_data[];
  auto *smem_q = reinterpret_cast<Element *>(smem_data);
  auto *smem_k = reinterpret_cast<Element *>(smem_q + cosize(SmemLayoutQ{}));
  auto *smem_v = reinterpret_cast<Element *>(smem_k + cosize(SmemLayoutK{}));

  const int bs_head_offset = base_id * params.head_stride;

  auto *gmem_q = reinterpret_cast<Element *>(params.q_ptr) + bs_head_offset;
  auto *gmem_k = reinterpret_cast<Element *>(params.k_ptr) + bs_head_offset;
  auto *gmem_v = reinterpret_cast<Element *>(params.v_ptr) + bs_head_offset;
  auto *gmem_o = reinterpret_cast<Element *>(params.out_ptr) + bs_head_offset;

  // gmem tensor
  Tensor Q = make_tensor(
      make_gmem_ptr(gmem_q),
      make_shape(params.q_seqlen, Int<kTileK>{}),
      make_stride(Int<kTileK>{}, Int<1>{}));
  Tensor K = make_tensor(
      make_gmem_ptr(gmem_k),
      make_shape(params.k_seqlen, Int<kTileK>{}),
      make_stride(Int<kTileK>{}, Int<1>{}));
  Tensor V = make_tensor(
      make_gmem_ptr(gmem_v),
      make_shape(Int<kTileK>{}, params.k_seqlen),
      make_stride(params.k_seqlen, Int<1>{})); //V is transposed
  Tensor O = make_tensor(
      make_gmem_ptr(gmem_o),
      make_shape(params.q_seqlen, Int<kTileK>{}),
      make_stride(Int<kTileK>{}, Int<1>{}));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<ElementAccum *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
                  
  auto gAtt_fp8 = make_tensor(make_gmem_ptr(static_cast<Element *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  auto gY =
      make_tensor(make_gmem_ptr(static_cast<Element *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{}));

  // smem tensor
  Tensor sQ = make_tensor(make_smem_ptr(smem_q), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(smem_k), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(smem_v), SmemLayoutV{});
  Tensor sO = make_tensor(make_smem_ptr(smem_q), SmemLayoutO{}); 

  // gmem tiling
  // (kTileM, kTileK, num_tile_n)
  // every thread block handles one tile of Q
  Tensor gQ = local_tile(Q, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(m_block, _));
  // first tile of K V
  Tensor gK = local_tile(K, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(0, _));
  Tensor gV = local_tile(V, make_tile(Int<kTileK>{}, Int<kTileN>{}), make_coord(_, 0)); 

  // MMA
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);

  // G2S copy
  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  // S2R copy
  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK  = thr_mma.partition_fragment_B(sK);
  Tensor tOrV  = thr_mma.partition_fragment_B(sV);

  // S2R copy Q
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::S2RCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  // S2R copy K
  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::S2RCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK); 
  
  // S2R copy V
  auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::S2RCopyAtom{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsV = smem_thr_copy_V.partition_S(sV);

  // R2S copy O
  auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);


  // load Q and first K
  cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
  // Q and K in one commit_group here
  cute::cp_async_fence();

  // allocate P and O
  auto tAtt = thr_mma.partition_fragment_C(gAtt); 
  auto tYgY = thr_mma.partition_fragment_C(gY);
  clear(tYgY);

  // used for causal mask to get row and col index
  auto gI = make_identity_tensor(gAtt.shape());
  auto tI = thr_mma.partition_C(gI);
  auto tI_mn = retile_fragment(tI);

  auto tY_mn = retile_fragment(tYgY);
  auto tAtt_mn = retile_fragment(tAtt);
  constexpr int kM = size<0>(tAtt_mn);
  constexpr int kN = size<1>(tAtt_mn);

  // allocate max and sum
  Tensor Att_max = make_tensor<ElementAccum>(Int<kM>{}); 
  Tensor Att_sum = make_tensor<ElementAccum>(Int<kM>{}); 
  clear(Att_sum);
  fill(Att_max, -std::numeric_limits<ElementAccum>::infinity());

  for (int TileId = KV_Tile_Min; TileId < KV_Tile_Max; TileId++) {
    clear(tAtt);

    // wait for K
    flash::cp_async_wait<0>();
    __syncthreads();

    // load V
    gV = local_tile(V, make_tile(Int<kTileK>{}, Int<kTileN>{}), make_coord(_,  TileId));
    tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    cute::cp_async_fence();

    // Q*K^T gemm and s2g overlap
    auto tSrQ_s2g = smem_thr_copy_Q.retile_D(tSrQ);
    auto tSrK_s2g = smem_thr_copy_K.retile_D(tSrK);
    // load the first batch
    cute::copy(smem_tiled_copy_Q, tSsQ(_, _, _0{}), tSrQ_s2g(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}), tSrK_s2g(_, _, _0{}));
    
    // pipeline
    for (int i = 0; i < size<2>(tSrQ); i++) {
      if (i < size<2>(tSrQ) - 1) {
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, i + 1), tSrQ_s2g(_, _, i + 1));
        cute::copy(smem_tiled_copy_K, tSsK(_, _, i + 1), tSrK_s2g(_, _, i + 1));
      }
      cute::gemm(tiled_mma, tSrQ(_, _, i), tSrK(_, _, i), tAtt);
    }

    // causal mask
    if (TileId * kTileN >= seqlen_q_start) {
      #pragma unroll
      for (int im = 0; im < kM; ++im) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          int irow = m_block * kTileM + get<0>(tI_mn(im, in));
          int icol = TileId * kTileN + get<1>(tI_mn(im, in));

          if ((icol > irow) || (icol >= params.k_seqlen)) {
            tAtt_mn(im, in) = -std::numeric_limits<float>::infinity();
          }
        }
      }
    }
  
    // wait for V
    flash::cp_async_wait<0>();
    __syncthreads();
    
    // load next K
    if (TileId != KV_Tile_Max - 1) {
      gK = local_tile(K, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(TileId + 1, _));
      tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
      cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
      cute::cp_async_fence();
    }
    
    // online softmax
    TileId == 0 ? softmax_rescale_o<true>(tAtt_mn, Att_max, Att_sum, tY_mn, params.softmax_scale) :
      softmax_rescale_o<false>(tAtt_mn, Att_max, Att_sum, tY_mn, params.softmax_scale);

    // fp32 -> fp8
    auto tAtt_float32x4 = recast<float4>(tAtt);
    auto tAtt_fp8 = make_fragment_like<Element>(tAtt);
    auto tAtt_fp8x4 = recast<__nv_fp8x4_e4m3>(tAtt_fp8);

#pragma unroll
    for (int i = 0; i < size(tAtt_float32x4); i++) {
      tAtt_fp8x4(i) = __nv_fp8x4_e4m3(tAtt_float32x4(i));
    }

    // fp8 MMA C layout to A layout, need to shuffle and permute
    auto reg2reg = ReorgCFp8toAFp8();
    reg2reg(tAtt_fp8);

    // A layout for P
    auto tAttA_layout = thr_mma.partition_fragment_A(gAtt_fp8).layout();
    auto tAttA_fp8 = make_tensor(tAtt_fp8.data(), tAttA_layout);

    // P*V gemm and s2g overlap
    auto tOrV_s2r = smem_thr_copy_V.retile_D(tOrV);
    cute::copy(smem_tiled_copy_V, tOsV(_, _, _0{}), tOrV_s2r(_, _, _0{}));

    for (int i = 0; i < size<2>(tOrV); i++) {
      if (i < size<2>(tOrV) - 1) {
        cute::copy(smem_tiled_copy_V, tOsV(_, _, i + 1), tOrV_s2r(_, _, i + 1));
      }
      cute::gemm(tiled_mma, tAttA_fp8(_, _, i), tOrV(_, _, i), tYgY);
    }
  } 
  
  // Epilogue
  // rescale
  #pragma unroll
  for (int m = 0; m < size<0>(tY_mn); ++m) {
    float sum = Att_sum(m);
    float inv_sum = (sum == 0.0f || sum != sum) ? 1.0f : 1.0f / sum;
    float scale = inv_sum;
    #pragma unroll
    for (int n = 0; n < size<1>(tY_mn); ++n) {
      tY_mn(m, n) *= scale;
    }
  }

  // fp32 -> fp8
  auto tY_float32x4 = recast<float4>(tYgY);
  auto tY_fp8 = make_fragment_like<Element>(tYgY);
  auto tY_fp8x4 = recast<__nv_fp8x4_e4m3>(tY_fp8);

#pragma unroll
  for (int i = 0; i < size(tY_float32x4); i++) {
    tY_fp8x4(i) = __nv_fp8x4_e4m3(tY_float32x4(i));
  }


  // output r2s
  Tensor tY_fp8_r2s = smem_thr_copy_O.retile_S(tY_fp8); 
  Tensor tOsO_r2s = smem_thr_copy_O.partition_D(sO); 
  cute::copy(smem_tiled_copy_O, tY_fp8_r2s, tOsO_r2s);

  // output s2g
  Tensor gO = local_tile(O, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(m_block, _));

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

  __syncthreads();

  cute::copy(gmem_tiled_copy_O, tOsO, tOgO);
  
}

void fill_params(Flash_fwd_params &params,
  const torch::Tensor q,
  const torch::Tensor k,
  const torch::Tensor v,
  torch::Tensor out,
  float softmax_scale) {

  memset(&params, 0, sizeof(params));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.out_ptr = out.data_ptr();

  params.bs = q.size(0);
  params.head = q.size(1);
  params.q_seqlen = q.size(2);
  params.k_seqlen = k.size(2);

  params.head_stride = q.stride(1);

  params.softmax_scale = softmax_scale;
  params.softmax_scale_log2 = softmax_scale * 1.4426950408889634f;
}

template<typename Kernel_traits>
void launch_flash_attn_fp8_kernel(Flash_fwd_params &params, cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutK;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutV;

  const int num_m_block =
      (params.q_seqlen + Kernel_traits::kTileM - 1) / Kernel_traits::kTileM;

  dim3 grid(num_m_block, params.bs * params.head, 1);
  dim3 block(Kernel_traits::kNThreads);

  //TODO: shared QK storage
  int smem_size = int(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  auto kernel = &flash_attn_fp8_kernel<Kernel_traits, Flash_fwd_params>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);  

  kernel<<<grid, block, smem_size, stream>>>(params);
}

torch::Tensor flash_attn_fp8_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, float softmax_scale) {

  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  // batch size
  int bs = q.size(0);
  // head number
  int head = q.size(1);
  // seqlen
  int seqlen = q.size(2);

  auto out = torch::empty_like(q);

  Flash_fwd_params params;
  fill_params(params, q, k, v, out, softmax_scale);

  constexpr int kTileK = 64;
  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kNWarps = 4;
  using Element = cutlass::float_e4m3_t;
  launch_flash_attn_fp8_kernel<Flash_fwd_kernel_traits<kTileK, kTileM, kTileN, kNWarps, Element>>(params, stream);

  // Wait until kernel finish.
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  return out;
}


