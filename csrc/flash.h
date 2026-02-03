#pragma once

#include <cstdint>

#ifdef _MSC_VER
#define ATTENTION_RESTRICT __restrict
#else
#define ATTENTION_RESTRICT __restrict__
#endif

struct Flash_fwd_params {

  void *ATTENTION_RESTRICT q_ptr;
  void *ATTENTION_RESTRICT k_ptr;
  void *ATTENTION_RESTRICT v_ptr;
  void *ATTENTION_RESTRICT out_ptr;

  size_t bs;
  size_t head;
  size_t q_seqlen;
  size_t k_seqlen;
  size_t head_stride;

  float softmax_scale;
  float softmax_scale_log2;    
};

