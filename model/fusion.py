import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinatesFusion(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.0):
        super(CoordinatesFusion, self).__init__()
        mid_feat = (in_feat + out_feat) // 2
        # self.left_se = nn.Sequential(
        #     nn.Linear(in_feat, (in_feat + mid_feat) // 2),
        #     nn.GELU(),
        #     nn.Linear((in_feat + mid_feat) // 2, out_feat),
        # )
        # self.right_se = nn.Sequential(
        #     nn.Linear(in_feat, (in_feat + mid_feat) // 2),
        #     nn.GELU(),
        #     nn.Linear((in_feat + mid_feat) // 2, out_feat),
        # )
        # self.body_se = nn.Sequential(
        #     nn.Linear(in_feat, (in_feat + mid_feat) // 2),
        #     nn.GELU(),
        #     nn.Linear((in_feat + mid_feat) // 2, out_feat),
        # )
        self.left_se = nn.Linear(in_feat, out_feat)
        self.right_se = nn.Linear(in_feat, out_feat)
        self.body_se = nn.Linear(in_feat, out_feat)
        self.out_proj = nn.Linear(out_feat, out_feat)
        self.norm = nn.LayerNorm(out_feat)

        self.gelu = nn.GELU()

        self.inverted_res = InvertedResidual(out_feat, out_feat)
        self.drop_rate = drop_rate

    def forward(self, left_embed, right_embed, body_embed):
        left_out = self.left_se(left_embed)
        left_out = self.gelu(left_out)

        right_out = self.right_se(right_embed)
        right_out = self.gelu(right_out)

        body_out = self.body_se(body_embed)
        body_out = self.gelu(body_out)

        attn_weight = torch.matmul(right_out, left_out.transpose(1, 2))
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = F.dropout(attn_weight, p=self.drop_rate, training=self.training)
        fuse = torch.matmul(attn_weight, body_out)
        fuse = self.out_proj(fuse)

        fuse = self.norm(fuse)
        fuse = self.inverted_res(fuse)
        fuse = F.dropout(fuse, self.drop_rate, training=self.training)
        return fuse


class InvertedResidual(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InvertedResidual, self).__init__()
        self.linear_1 = nn.Linear(in_dim, in_dim)
        self.linear_2 = nn.Linear(in_dim, in_dim * 3)
        self.linear_3 = nn.Linear(in_dim * 3, out_dim)
        self.gelu = nn.GELU()
        self.bn1 = nn.LayerNorm(in_dim)

    def forward(self, x):
        residual = x
        out = self.linear_1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.linear_2(out)
        out = self.gelu(out)
        out = self.linear_3(out)

        return out


if __name__ == "__main__":
    m = CoordinatesFusion(512, 1024, 0.1)
    l = torch.randn(32, 45, 512)
    r = torch.randn(32, 45, 512)
    b = torch.randn(32, 45, 512)

    o = m(l, r, b)
    print(o.shape)
