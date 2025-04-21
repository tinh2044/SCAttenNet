import torch
import torch.nn as nn
import torch.nn.functional as F
from model.spatial_attention import SpatialAttention
from model.layers import STPositionalEncoding


class SpatialTemporalModule(nn.Module):
    def __init__(self, in_feat, out_feat, num_node, num_frame, 
                 s_kernel=3, stride=1, t_kernel=3, drop_rate=0.1, cross_attention=False):
        super().__init__()
        pad_s = (s_kernel - 1) // 2

        self.spatial_pe = STPositionalEncoding(in_feat, num_node, num_frame, 'spatial')
        self.spatial_attn = SpatialAttention(in_feat, out_feat, inner_feat=16, drop_out=drop_rate, s_kernel=s_kernel, pad_s=pad_s)        

        padd = t_kernel // 2

        self.out_spatial = nn.Conv2d(out_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s), stride=(1, 1), bias=False)

        
        self.out_temporal = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=(t_kernel, 1), padding=(padd, 0), stride=(stride, 1), bias=False),
            nn.BatchNorm2d(out_feat))

        if in_feat != out_feat or stride != 1:
            self.downs1 = self._conv_block(out_feat)
            self.downs2 = self._conv_block(out_feat)
            self.downt1 = self._conv_block(out_feat)
            self.downt2 = self._conv_block(out_feat, kernel_size=1, stride=1, padding=0)
        else:
            self.downs1 = self.downs2 = self.downt1 = self.downt2 = nn.Identity()

        self.act = nn.LeakyReLU(0.1)
        self.drop_rate = drop_rate
        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_attn = SpatialAttention(out_feat, out_feat, inner_feat=16, drop_out=drop_rate, s_kernel=s_kernel, pad_s=pad_s)
            self.out_cross = nn.Sequential(
                nn.Conv2d(out_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s), stride=(1, 1)),
                nn.BatchNorm2d(out_feat)
            )
    def _conv_block(self, channels, kernel_size=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channels))

    def forward(self, x, y=None):

        x = x + self.spatial_pe(x)

        x = self.spatial_attn(x)

        x = self.out_spatial(x) + x
                
        x = self.act(self.downs1(x)) + x
        x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.act(self.downs2(x) + x)

        
        x = self.act(self.out_temporal(x))
        x = self.act(self.downt1(x) + x)
        x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.act(self.downt2(x)) + x
        
        if y is not None and self.cross_attention:
            assert x.shape == y.shape, f"x and y must have the same shape (got {x.shape} and {y.shape})"
            x = self.cross_attn(x, y)
            x = self.act(self.out_cross(x) + x)
            x = F.dropout(x, self.drop_rate, training=self.training)

        return x 