import torch
import math
from ada_cute_fp8_flash_attention import flash_attn_fp8_forward
import time

torch.manual_seed(42)

def test():
    bs = 1
    head = 32
    seq_len = 2048
    head_dim = 64

    q = (torch.empty((bs, head, seq_len, head_dim), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((bs, head, seq_len, head_dim), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((bs, head, head_dim, seq_len), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())

    # clamp to reduce error caused by quantization
    q = q.clamp(-0.1, 0.1).to(torch.float8_e4m3fn)
    k = k.clamp(-0.1, 0.1).to(torch.float8_e4m3fn)
    v = v.clamp(-0.1, 0.1).to(torch.float8_e4m3fn)

    flash2_cutlass_ref = flash_attn_fp8_forward(q, k, v)

if __name__ == "__main__":
    test()


