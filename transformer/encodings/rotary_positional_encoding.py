import torch
import einops 

class RotaryPositionalEncoding(torch.nn.Module):
  def __init__(self, dk, max_len=5000, theta=1000.0):
    super(RotaryPositionalEncoding, self).__init__()
    assert dk % 2 == 0

    self.freqs =  self.precompute_freqs_cis(dk=dk, max_len=max_len, theta=theta)
    self.register_buffer('precomputed_freqs', self.freqs)

  def precompute_freqs_cis(self, dk, max_len, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, dk, 2)[: (dk // 2)].float() / dk))
    t = torch.arange(max_len, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs) # (max_len, dk/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis 
  
  def apply_rotation(self, x):
    B, n_heads, T, dk = x.shape
    x_ = torch.view_as_complex(einops.rearrange(x.float(), 'B n_heads T (dk_2 two) -> B n_heads T dk_2 two', B=B, T=T, n_heads=n_heads, two=2))

    x_ = torch.view_as_real(x_ * self.freqs[:T,:].unsqueeze(0).unsqueeze(0)).type_as(x) # (B nh T dk/2) * (1 1 T dk/2) = (B nh T dk/2)
    return einops.rearrange(x_, 'B n_heads T dk_2 two -> B n_heads T (dk_2 two)', B=B, T=T, n_heads=n_heads, two=2)

  def forward(self, xq, xk):
   B, n_heads, T, dk = xq.shape
   n_kv_heads = xk.shape[-2]
   return self.apply_rotation(xq), self.apply_rotation(xk)
  
