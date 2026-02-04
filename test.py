import torch
import math
from ada_cute_fp8_flash_attention import flash_attn_fp8_forward
import time

torch.manual_seed(42)

def naive_self_attention(q, k, v, sm_scale):
    seq_len = q.shape[-2]
    M = torch.tril(torch.ones((seq_len, seq_len), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    return ref_out

def warmup(warmup_count, func, *args, **kwargs):
    # warmup phase
    for _ in range(warmup_count):
        func(*args, **kwargs)
    torch.cuda.synchronize()


def test():
    bs = 1
    head = 32
    seq_len = 2048
    head_dim = 64

    q = (torch.empty((bs, head, seq_len, head_dim), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((bs, head, seq_len, head_dim), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((bs, head, seq_len, head_dim), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())

    # clamp to reduce error caused by quantization
    q = q.clamp(-0.1, 0.1).to(torch.float8_e4m3fn)
    k = k.clamp(-0.1, 0.1).to(torch.float8_e4m3fn)
    v = v.clamp(-0.1, 0.1).to(torch.float8_e4m3fn)
        
    q_d16 = q.to(torch.float16)
    k_d16 = k.to(torch.float16)
    v_d16 = v.to(torch.float16)
    
    #fp16 version of naive self attention
    sm_scale = 1.0 / math.sqrt(seq_len)
    baseline = naive_self_attention(q_d16, k_d16, v_d16, sm_scale)
    
    #fp8 version
    q = q.to(torch.float8_e4m3fn)
    k = k.to(torch.float8_e4m3fn)
    v = v.to(torch.float8_e4m3fn)
    #v need to transpose first, because ldsm only support 16bit transpose
    v = v.transpose(2, 3).contiguous()

    warmup_count = 5
    flash2_time = warmup(warmup_count, flash_attn_fp8_forward, q, k, v)
    flash2_cutlass_ref = flash_attn_fp8_forward(q, k, v)

    assert torch.allclose(baseline, flash2_cutlass_ref.to(torch.float16), rtol=0, atol=1e-2)

if __name__ == "__main__":
    test()


