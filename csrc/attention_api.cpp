#include <torch/extension.h>
#include <torch/python.h>

#include "attention_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fp8_forward", &flash_attn_fp8_forward,
          "Flash attention v2 implement in cutlass");
}
