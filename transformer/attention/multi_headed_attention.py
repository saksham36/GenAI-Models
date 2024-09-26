import torch
import torch.nn as nn
import einops

class MultiHeadedAttentionBlock(torch.nn.Module):
  def __init__(self, d_model, num_heads=1):
    super(MultiHeadedAttentionBlock, self).__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = self.d_model // self.num_heads

    assert self.d_model % self.n_heads == 0

    self.kqv = nn.Linear(d_model, 3 * d_model)
    self.o = nn.Linear(d_model, d_model)
    
    
    

  def forward(self, x, rotary=True):
    B, T, d_model = x.shape

    KQV = self.kqv(x)
    KQV = einops.rearrange(KQV, 'B T (n H) -> n B T H', n=3)

    multi_headed_KQV = einops.rearrange(KQV, 'N B T (num_heads d_k) -> N B num_heads T d_k', N=3, B=B, T=T, num_heads=self.num_heads, d_k=self.d_k)

    K = multi_headed_KQV[0,:]
    Q = multi_headed_KQV[1,:]
    V = multi_headed_KQV[2,:]

    if rotary:
        Q, K = self.rotary_positional_encoding(Q, K)        
    
    attention_score = torch.einsum('bntd, bnTd ->bntT', [Q, K])
    attention_score = attention_score.masked_fill(torch.triu(torch.ones(attention_score.shape[-2:]).to(attention_score.device), diagonal=1).bool(), float('-inf')) # [B, num_heads, T, T]

    attention = torch.softmax(attention_score/ (self.d_k**(1/2)), dim=-1) @ V

    attention = einops.rearrange(attention, "b num_heads T d_k -> b T (num_heads d_k)", B=B, num_heads=self.num_heads, T=T, d_k=self.d_k)
    attention_out = self.o(attention)

    return attention_out

