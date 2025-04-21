import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, in_feat, out_feat, inner_feat, drop_out, s_kernel=3, pad_s=1):
        super().__init__()
    
        self.key_proj = nn.Conv2d(in_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s))
        self.value_proj = nn.Conv2d(in_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s))
        self.query_proj = nn.Conv2d(in_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s))     
        self.head = out_feat // inner_feat
        if (self.head * inner_feat != out_feat):
            raise ValueError(f"out_feat must be divisible by inner_feat (got `out_feat`: {out_feat}"
                             f" and `inner_feat`: {inner_feat}).")
            
        self.inner_feat = inner_feat
        self.drop_out = drop_out
        self.scale = (inner_feat ** -0.5)
        
        
        self.out_proj = nn.Conv2d(out_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s))
        
    def forward(self, x, y=None):
        if y is None:
            y = x
        
        b, c, t, k = x.shape
        key = self.key_proj(y)
        value = self.value_proj(y)
        query = self.query_proj(x)
        
        attention = torch.einsum("bctk, bctv -> btkv", [query, key]) * self.scale
        attention = nn.functional.softmax(attention, dim=-1)
        attention = nn.functional.dropout(attention, p=self.drop_out, training=self.training)
        
        out = torch.einsum("btkv, bctv -> bctk", [attention, value])
        out = self.out_proj(out)
        
        return out 