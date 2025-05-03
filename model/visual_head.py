import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import StaticPositionalEncoding, MaskedNorm, PositionWiseFeedForward, MLPHead
from model.residual import ResidualNetwork


class VisualHead(torch.nn.Module):
    def __init__(self, 
        cls_num, input_size=512, hidden_size=1024, ff_size=2048, pe=True,
        ff_kernelsize=[3,3], residual_blocks=[]):
        super().__init__()
        self.hidden_size = hidden_size

        # if input_size is None:
        #     self.fc1 = nn.Identity()
        # else:
        #    self.fc1 = nn.Linear(input_size, self.hidden_size)
        
        self.residual = ResidualNetwork(residual_blocks=residual_blocks)
        self.bn1 = nn.LayerNorm(residual_blocks[-1], eps=1e-6)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.1)

        if pe:
            self.pe = StaticPositionalEncoding(self.hidden_size)
        else:
            self.pe = torch.nn.Identity()

        self.feedforward = PositionWiseFeedForward(input_size=self.hidden_size,
            ff_size=ff_size,
            dropout=0.1, kernel_size=ff_kernelsize, skip_connection=True)
        
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.gloss_output_layer = torch.nn.Linear(self.hidden_size, cls_num)


    def forward(self, x, mask, valid_len_in=None):
        if valid_len_in is None:
            valid_len_in = x.shape[1]
        

        #projection 1
        # x = self.fc1(x)
        x, _ = self.residual(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #pe
        x = self.pe(x)
        x = self.dropout1(x)

        #feedforward
        x = self.feedforward(x)
        x = self.layer_norm(x)

                
        #classification
        logits = self.gloss_output_layer(x) #B,T,V
        gloss_probabilities_log = logits.log_softmax(-1) 
        gloss_probabilities = logits.softmax(-1)

       
        return {
                'gloss_feature': x,
                'gloss_feature_norm': F.normalize(x, dim=-1),
                'gloss_logits':logits, 
                'gloss_probabilities_log':gloss_probabilities_log,
                'gloss_probabilities': gloss_probabilities,
                'valid_len_out':valid_len_in}