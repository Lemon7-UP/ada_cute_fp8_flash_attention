#pragma once

#include <cstddef>
#include <cstdint>
#include <torch/extension.h>
#include "flash.h"
#include <vector>

torch::Tensor flash_attn_fp8_forward(torch::Tensor q, torch::Tensor k,
              torch::Tensor v);
