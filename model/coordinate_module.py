import torch
import torch.nn as nn
from model.spatial_temporal_module import SpatialTemporalModule


class CoordinateModule(nn.Module):
    def __init__(self, joint_idx, num_frame, nets, dropout=0.1, cross_attention=False):
        super().__init__()
        
        self.dropout = dropout
        
        num_frame = num_frame
        nets = nets
      
        self.layers = []
        for i in range(len(nets)):
            in_feat, out_feat, s_kernel, t_kernel, stride = nets[i]
    
            self.layers.append(SpatialTemporalModule(
                in_feat=in_feat,
                out_feat=out_feat,
                num_node=len(joint_idx),
                num_frame=num_frame,
                s_kernel=s_kernel,
                stride=stride,
                t_kernel=t_kernel,
                drop_rate=self.dropout*i,
                cross_attention=cross_attention
            ))
            num_frame = num_frame // stride
            
        self.layers = nn.ModuleList(self.layers)

    def forward(self, coord_embeds, other_embeds=None):
        coord_embeds = nn.functional.dropout(coord_embeds, self.dropout, training=self.training)
        
        outputs = []
        
        for i, layer in enumerate(self.layers):
            output = layer(coord_embeds, other_embeds[i] if other_embeds is not None else None) 
            coord_embeds = output
            outputs.append(output)
            
        return coord_embeds, outputs 