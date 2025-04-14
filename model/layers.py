import torch
from torch import nn
import math
import numpy as np

class LearningPositionEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, inputs_embeds):

        bsz, seq_len = inputs_embeds.shape[:2]
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device).expand(bsz, -1).to(inputs_embeds.device)
        positions_embeddings = super().forward(positions + self.offset)

        return inputs_embeds + positions_embeddings

class StaticPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):

        super(StaticPositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)
        
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, inputs_embeds):
        seq_len = inputs_embeds.size(1)
        return inputs_embeds + self.pe[:, :seq_len, :]

class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Conv2d(in_feat, out_feat, kernel_size=(1, 1))
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(out_feat, in_feat, kernel_size=(1, 1))
        self.batch_norm = nn.BatchNorm2d(in_feat)
        
        self.dropout =dropout

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x + residual

class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
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

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        pos_emb = self.pe[:, :, :x.size(2), :]
        
        return pos_emb

class CoordinateMapping(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(CoordinateMapping, self).__init__()
        
        self.mapping_x = nn.Linear(in_feat, out_feat)
        self.mapping_y = nn.Linear(in_feat, out_feat)

    def forward(self, x_coord, y_coord):
        x_embed = self.mapping_x(x_coord)       

        y_embed = self.mapping_y(y_coord)        
        
        return x_embed, y_embed
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.LeakyReLU(0.1)


    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

# Kiểm tra mô hình
if __name__ == "__main__":
    model = DepthwiseSeparableConv(in_channels=32, out_channels=64)
    x = torch.randn(1, 32, 180,  21)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
