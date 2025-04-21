from torch import nn
from model.layers import CoordinateMapping
from model.encoder import Encoder
from model.decoder import Decoder
from model.residual import ResidualNetwork


class KeypointModule(nn.Module):
    def __init__(self, joint_idx, num_frame, nets, dropout=0.1, cfg=None):
        super().__init__()
        self.joint_idx = joint_idx
        self.nets = nets
        self.num_frame = num_frame
        self.coordinate_mapping = CoordinateMapping(len(joint_idx), cfg['d_model'])
        
        self.x_coord_module = Encoder(cfg)
        self.y_coord_module = Decoder(cfg)      
        
        self.residual_net = ResidualNetwork(cfg['d_model'])
    
    def forward(self, keypoints, attention_mask=None):
        x = keypoints[:, :, :, 0]
        y = keypoints[:, :, :, 1]
        
        x_embed, y_embed = self.coordinate_mapping(x, y)
        
        x_embed = self.x_coord_module(x_embed, attention_mask)
        
        y_embed = self.y_coord_module(encoder_hidden_states=x_embed, 
                                    encoder_attention_mask=attention_mask, 
                                    y_embed=y_embed, 
                                    attention_mask=attention_mask)
        
        output_embed = self.residual_net(y_embed)
        
        return output_embed 