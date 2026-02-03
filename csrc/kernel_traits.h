#pragma once

#include "cute/algorithm/copy.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

#include "mma_sm89.hpp"
#include "mma_traits_sm89.hpp"

template<int kTileK_, int kTileM_, int kTileN_, int kNWarps_, typename elem_type>
struct Flash_fwd_kernel_traits
{
    using Element = elem_type;
    using ElementAccum = float;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    // tiling params
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;

    //MMA atom, sm89 only fp8 mma
    using MMA_Atom_Arch = cute::MMA_Atom<cute::SM89_16x8x32_F32E4M3E4M3F32_TN>;
    //G2S use async copy
    using G2SCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, elem_type>;
    //S2R use ldmatrix
    using S2RCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, elem_type>;
    //register -> smem -> gmem, to use large store instruction
    //because STSM need sm90, only use default copy
    using SmemCopyAtomO = cute::Copy_Atom<cute::DefaultCopy, elem_type>;

    using TiledMma = cute::TiledMMA<
        MMA_Atom_Arch,
        cute::Layout<cute::Shape<cute::Int<kNWarps>, cute::_1, cute::_1>>,  // warp parallel in M dim
        cute::Tile<cute::Int<16 * kNWarps>, cute::_16, cute::_32>>; // 

    using SmemLayoutAtom = decltype(
        cute::composition(cute::Swizzle<3, 4, 3>{}, cute::Layout<cute::Shape<cute::_16, cute::_64>, // (16, 64)
                           cute::Stride<cute::_64, cute::_1>>{}));

    using SmemLayoutQ = decltype(cute::tile_to_shape(
        SmemLayoutAtom{},
        cute::Shape<cute::Int<kTileM>, cute::Int<kTileK>>{}));

    using SmemLayoutK = decltype(cute::tile_to_shape(
        SmemLayoutAtom{},
        cute::Shape<cute::Int<kTileN>, cute::Int<kTileK>>{}));
    
    using SmemLayoutV = decltype(cute::tile_to_shape(
        SmemLayoutAtom{},
        cute::Shape<cute::Int<kTileK>, cute::Int<kTileN>>{}));

    using SmemLayoutO = decltype(cute::tile_to_shape(
        SmemLayoutAtom{},
        cute::Shape<cute::Int<kTileM>, cute::Int<kTileK>>{}));

    //shared memory size
    static constexpr int kSmemQCount = cute::size(SmemLayoutQ{});
    static constexpr int kSmemKVCount = cute::size(SmemLayoutK{}) + cute::size(SmemLayoutV{});
    static constexpr int kSmemQSize = kSmemQCount * sizeof(elem_type);
    static constexpr int kSmemKVSize = kSmemKVCount * sizeof(elem_type);
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

    //val layout for tiled copy
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(elem_type);
    static constexpr int kGmemThreadsPerRow = kTileK / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = cute::Layout<cute::Shape<cute::Int<kNThreads / kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>, //thr layout
                                  cute::Stride<cute::Int<kGmemThreadsPerRow>, cute::_1>>;

    using GmemTiledCopyQKV = decltype(
        cute::make_tiled_copy(G2SCopyAtom{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape<cute::_1, cute::Int<kGmemElemsPerLoad>>>{}));

    using GmemTiledCopyO = decltype(
        cute::make_tiled_copy(SmemCopyAtomO{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape<cute::_1, cute::Int<kGmemElemsPerLoad>>>{}));

};
