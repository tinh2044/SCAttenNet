import torch
from torch import nn
import math
import numpy as np


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__  # dict.k  ==>  dict[k]
    # __getattr__ = dict.get  # dict.k  ==>  dict.get(k)
    # __getattr__ = lambda d, k: d.get(k, '')  # dict.k  ==>  dict.get(k,default)


class LearningPositionEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, inputs_embeds):
        bsz, seq_len = inputs_embeds.shape[:2]

        positions = (
            torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
            .expand(bsz, -1)
            .to(inputs_embeds.device)
        )
        positions_embeddings = super().forward(positions + self.offset)

        return inputs_embeds + positions_embeddings


class StaticPositionalEncoding(nn.Module):
    def __init__(self, size: int = 0, max_len: int = 5000):
        super(StaticPositionalEncoding, self).__init__()
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        return emb + self.pe[:, : emb.size(1)]


class STPositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len, domain):
        super(STPositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(
            torch.arange(0, channel, 2).float() * -(math.log(10000.0) / channel)
        )  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos_emb = self.pe[:, :, : x.size(2), :]

        return pos_emb


class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Conv2d(in_feat, out_feat, kernel_size=(1, 1))
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(out_feat, in_feat, kernel_size=(1, 1))
        self.batch_norm = nn.BatchNorm2d(in_feat)

        self.dropout = dropout

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x + residual


class CoordinateMapping(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(CoordinateMapping, self).__init__()

        self.mapping_x = nn.Linear(in_feat, out_feat)
        self.mapping_y = nn.Linear(in_feat, out_feat)

    def forward(self, x_coord, y_coord):
        x_embed = self.mapping_x(x_coord)

        y_embed = self.mapping_y(y_coord)

        return x_embed, y_embed
