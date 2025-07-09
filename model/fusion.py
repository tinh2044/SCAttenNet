import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinatesFusion(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.0):
        super(CoordinatesFusion, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_feat, in_feat // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_feat // 2, in_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.Conv_1 = Conv(in_feat, in_feat, 1, bn=True, relu=False)
        self.Conv_2 = Conv(in_feat * 3, in_feat, 1, bn=True, relu=False)
        self.norm1 = LayerNorm(in_feat * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_feat * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(in_feat * 3, eps=1e-6, data_format="channels_first")

        self.gelu = nn.GELU()

        self.inverted_res = InvertedResidual(in_feat * 3, out_feat)
        self.drop_rate = drop_rate

    def forward(self, left_embed, right_embed, body_embed):
        left_residual = left_embed
        left_max = self.maxpool(left_embed)
        left_avg = self.avgpool(left_embed)
        left_max_out = self.se(left_max)
        left_avg_out = self.se(left_avg)
        left_out = self.sigmoid(left_max_out + left_avg_out) * left_residual

        right_residual = right_embed
        right_max = self.maxpool(right_embed)
        right_avg = self.avgpool(right_embed)
        right_max_out = self.se(right_max)
        right_avg_out = self.se(right_avg)
        right_out = self.sigmoid(right_max_out + right_avg_out) * right_residual

        body_embed = self.Conv_1(body_embed)
        shortcut = body_embed
        body_embed = torch.cat([body_embed, left_embed, right_embed], 1)
        body_embed = self.norm1(body_embed)
        body_embed = self.Conv_2(body_embed)
        body_embed = self.gelu(body_embed)

        fuse = torch.cat([left_out, right_out, body_embed], 1)
        fuse = self.norm3(fuse)
        fuse = self.inverted_res(fuse)
        fuse = shortcut + F.dropout(fuse, self.drop_rate, training=self.training)
        return fuse


class LayerNorm(nn.Module):
    def __init__(self, normalized_dim, eps=1e-5, data_format="channels_first"):
        """
        Adaptive Layer Normalization that can handle both (B, C, T) and (B, T, C) formats

        Args:
            normalized_dim (int): The dimension to be normalized (C)
            eps (float): A small value added for numerical stability
            data_format (str): Either "channels_first" for (B, C, T) or "channels_last" for (B, T, C)
        """
        super().__init__()
        self.normalized_dim = normalized_dim
        self.eps = eps
        assert data_format in ["channels_first", "channels_last"]
        self.data_format = data_format

        self.weight = nn.Parameter(torch.ones(normalized_dim))
        self.bias = nn.Parameter(torch.zeros(normalized_dim))

    def forward(self, x):
        if self.data_format == "channels_first":  # (B, C, T)
            # Move channel dim to last for normalization
            x = x.transpose(1, 2)  # (B, T, C)

        # Calculate mean and var across last dimension (C)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply weight and bias
        x = x * self.weight + self.bias

        if self.data_format == "channels_first":
            # Return to original format if needed
            x = x.transpose(1, 2)  # (B, C, T)

        return x


class Conv(nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        bn=False,
        relu=True,
        bias=True,
        group=1,
    ):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv1d(
            inp_dim,
            out_dim,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(InvertedResidual, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm1d(inp_dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


if __name__ == "__main__":
    m = CoordinatesFusion(1024, 1024, 0.1)
    l = torch.randn(32, 1024, 45)
    r = torch.randn(32, 1024, 45)
    b = torch.randn(32, 1024, 45)

    o = m(l, r, b)
    print(o.shape)
