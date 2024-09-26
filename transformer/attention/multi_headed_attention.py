import torch
import torch.nn as nn
import einops

class MultiHeadedAttentionBlock(torch.nn.Module):
  def __init__(self, d_model, num_heads=1):
    super(MultiHeadedAttentionBlock, self).__init__()

    self.d_model = d_model

    self.kqv = nn.Linear(d_model, 3 * d_model)
    self.o = nn.Linear(d_model, d_model)
    self.num_heads = num_heads
    

  def forward(self, x, rotary=True):
    B, T, d_model = x.shape

    KQV = self.kqv(x)
    KQV = einops.rearrange(KQV, 'B T (n H) -> n B T H', n=3)

    d_k = self.d_model // self.num_heads

    multi_headed_KQV = einops.rearrange(KQV, 'N B T (num_heads d_k) -> N B num_heads T d_k', N=3, B=B, T=T, num_heads=self.num_heads, d_k=d_k)

    if rotary:
        rotated_KQ = self.rotary_positional_encoding(multi_headed_KQV[:-1,:])
        K_encoding = rotated_KQ[0,:]
        Q_encoding = rotated_KQ[1,:]
    else:
        K_encoding = multi_headed_KQV[0,:]
        Q_encoding = multi_headed_KQV[1,:]
    
    attention_score = torch.einsum('bntd, bnTd ->bntT', [Q_encoding,K_encoding])
    V = multi_headed_KQV[2,:]
    attention_score = attention_score.masked_fill(torch.triu(torch.ones(attention_score.shape[-2:]).to(attention_score.device), diagonal=1).bool(), float('-inf')) # [B, num_heads, T, T]

    attention = torch.softmax(attention_score/ (d_k**(1/2)), dim=-1) @ V

    attention = einops.rearrange(attention, "b num_heads T d_k -> b T (num_heads d_k)")
    attention_out = self.o(attention)

    return attention_out

