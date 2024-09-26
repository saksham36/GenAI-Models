import torch
import torch.nn as nn
import einops

class GroupedQueryAttention(torch.nn.Module):
  def __init__(self, d_model, num_heads, num_kv_heads):
    super(GroupedQueryAttention, self).__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    
    assert self.d_model % self.n_heads == 0
    assert self.num_heads % self.num_kv_heads == 0

    self.d_k = self.d_model // self.num_heads
    self.num_kv_rep = self.num_heads // self.num_kv_heads

    self.wk = nn.Linear(self.d_model, self.num_kv_heads * self.d_k)
    self.wv = nn.Linear(self.d_model, self.num_kv_heads * self.d_k)
    self.wq = nn.Linear(self.d_model, self.num_heads * self.d_k)
    self.o = nn.Linear(d_model, d_model)
    
    

  def forward(self, x, rotary=True):
    B, T, _ = x.shape
    
    K = self.wk(x) # (B, T, n_k * d_k)
    Q = self.wq(x)
    V = self.wv(x)

    Q = einops.rearrange('B T (n d_k) -> B n T d_k', B=B, T=T, n=self.num_heads, d_k=self.d_k)
    K = einops.rearrange('B T (n d_k) -> B n T d_k', B=B, T=T, n=self.num_kv_heads, d_k=self.d_k)
    V = einops.rearrange('B T (n d_k) -> B n T d_k', B=B, T=T, n=self.num_kv_heads, d_k=self.d_k)

    
    if rotary:
        Q, K = self.rotary_positional_encoding(Q, K)        
    
    K = torch.repeat_interleave(K, repeats=self.num_kv_rep, dim=1)
    V = torch.repeat_interleave(V, repeats=self.num_kv_rep, dim=1)

    attention_score = torch.einsum('bntd, bnTd ->bntT', [Q,K])
    attention_score = attention_score.masked_fill(torch.triu(torch.ones(attention_score.shape[-2:]).to(attention_score.device), diagonal=1).bool(), float('-inf')) # [B, num_heads, T, T]

    attention = torch.softmax(attention_score/ (self.d_k**(1/2)), dim=-1) @ V

    attention = einops.rearrange(attention, "b num_heads T d_k -> b T (num_heads d_k)")
    attention_out = self.o(attention)

    return attention_out

