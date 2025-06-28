import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import StaticPositionalEncoding
from model.residual import ResidualNetwork


class VisualHead(torch.nn.Module):
    def __init__(self, cls_num, pe=True, residual_blocks=[], **kwargs):
        super().__init__()
        self.hidden_size = residual_blocks[-1]

        self.residual = ResidualNetwork(residual_blocks=residual_blocks)
        self.bn1 = nn.LayerNorm(residual_blocks[-1], eps=1e-6)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.1)

        if pe:
            self.pe = StaticPositionalEncoding(self.hidden_size)
        else:
            self.pe = torch.nn.Identity()

        self.gloss_output_layer = torch.nn.Linear(self.hidden_size, cls_num)

    def forward(self, x, mask, valid_len_in=None):
        if valid_len_in is None:
            valid_len_in = x.shape[1]

        x = self.pe(x)
        x, _ = self.residual(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        logits = self.gloss_output_layer(x)
        gloss_prob_log = logits.log_softmax(-1)
        gloss_prob = logits.softmax(-1)

        return {
            "gloss_feature": x,
            "gloss_feature_norm": F.normalize(x, dim=-1),
            "gloss_logits": logits,
            "gloss_prob_log": gloss_prob_log,
            "gloss_prob": gloss_prob,
            "valid_len_out": valid_len_in,
        }
