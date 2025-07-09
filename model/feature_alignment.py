import torch
import torch.nn as nn
from model.BiLSTM import BiLSTMLayer


class FeatureAlignment(torch.nn.Module):
    def __init__(self, cls_num, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.gloss_layer = nn.Linear(input_size, cls_num)
        self.bi_lstm = BiLSTMLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )

    def forward(self, x):
        x = self.bi_lstm(x)
        logits = self.gloss_layer(x["predictions"])

        return logits
