"""CPU stand-in for flash-attn: each sequence runs torch's scaled_dot_product_attention."""
import torch
import torch.nn.functional as F


def _attend(q, k, v):
    # q [L, H, C], k/v [M, H, C] -> [L, H, C]
    out = F.scaled_dot_product_attention(q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None], v.permute(1, 0, 2)[None])
    return out[0].permute(1, 0, 2)


def flash_attn_func(q, k, v, *args, **kwargs):
    return torch.stack([_attend(q[b], k[b], v[b]) for b in range(q.shape[0])])


def flash_attn_qkvpacked_func(qkv, *args, **kwargs):
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v)


def flash_attn_kvpacked_func(q, kv, *args, **kwargs):
    k, v = kv.unbind(dim=2)
    return flash_attn_func(q, k, v)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *args, **kwargs):
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    return torch.cat([_attend(q[cq[i]:cq[i + 1]], k[ck[i]:ck[i + 1]], v[ck[i]:ck[i + 1]]) for i in range(len(cq) - 1)])


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=None, *args, **kwargs):
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *args, **kwargs):
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k)
