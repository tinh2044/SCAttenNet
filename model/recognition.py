import torch
from torch import nn
from model.layers import CoordinateMapping, PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder
from model.visual_head import VisualHead
from model.residual import ResidualNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, in_feat, out_feat, inner_feat, drop_out, s_kernel=3, pad_s=1):
        super().__init__()
    
        self.key_proj = nn.Conv2d(in_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s))
        self.value_proj = nn.Conv2d(in_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s))
        self.query_proj = nn.Conv2d(in_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s))     
        self.head = out_feat // inner_feat
        if (self.head * inner_feat != out_feat):
            raise ValueError(f"out_feat must be divisible by inner_feat (got `out_feat`: {out_feat}"
                             f" and `inner_feat`: {inner_feat}).")
            
        self.inner_feat = inner_feat
        self.drop_out = drop_out
        self.scale = (inner_feat ** -0.5)
        
        
        self.out_proj = nn.Conv2d(out_feat, out_feat, kernel_size=(1, s_kernel), padding=(0, pad_s))
        
    def forward(self, x, y=None):
        if y is None:
            y = x
        
        b, c, t, k = x.shape
        key = self.key_proj(y)
        value = self.value_proj(y)
        query = self.query_proj(x)
        
        attention = torch.einsum("bctk, bctv -> btkv", [query, key]) * self.scale
        attention = nn.functional.softmax(attention, dim=-1)
        attention = nn.functional.dropout(attention, p=self.drop_out, training=self.training)
        
        out = torch.einsum("btkv, bctv -> bctk", [attention, value])
        out = self.out_proj(out)
        
        return out
        

class SpatialTemporalModule(nn.Module):
    def __init__(self, in_feat, out_feat, num_node, num_frame, 
                 s_kernel=3, stride=1, t_kernel=3, drop_rate=0.1, cross_attention=False):
        super().__init__()
        pad_s = (s_kernel - 1) // 2

        self.spatial_pe = PositionalEncoding(in_feat, num_node, num_frame, 'spatial')
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
            x  = nn.functional.dropout(x, self.drop_rate, training=self.training)

        return x


class CoordinateModule(nn.Module):
    def __init__(self, joint_idx, num_frame, nets, dropout=0.1, cross_attention=False):
        super().__init__()
        
        self.dropout = dropout
        
        num_frame = num_frame
        nets = nets
      
        self.layers = []
        for i in range(len(nets)):
            in_feat, out_feat,s_kernel, t_kernel, stride = nets[i]
    
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
    
class KeypointModule(nn.Module):
    def __init__(self,joint_idx, num_frame, nets, dropout=0.1, cfg=None):
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
        
        ouput_embed = self.residual_net(y_embed)
        
        return y_embed
    
        

class RecognitionNetwork(nn.Module):
    def __init__(self, cfg, gloss_tokenizer):
        super().__init__()
        self.cfg = cfg
        self.cross_distillation = cfg['cross_distillation']
        
        self.body_encoder = KeypointModule(cfg['body_idx'], num_frame=cfg['num_frame'], nets=cfg['nets'], cfg=cfg)
        self.left_encoder = KeypointModule(cfg['left_idx'], num_frame=cfg['num_frame'], nets=cfg['nets'], cfg=cfg)
        self.right_encoder = KeypointModule(cfg['right_idx'], num_frame=cfg['num_frame'], nets=cfg['nets'], cfg=cfg)
        self.face_encoder = KeypointModule(cfg['face_idx'], num_frame=cfg['num_frame'], nets=cfg['nets'], cfg=cfg)
        
        
        self.left_visual_head = VisualHead(**cfg['left_visual_head'], cls_num=len(gloss_tokenizer))
        self.right_visual_head = VisualHead(**cfg['right_visual_head'], cls_num=len(gloss_tokenizer))
        self.fuse_visual_head = VisualHead(**cfg['fuse_visual_head'], cls_num=len(gloss_tokenizer))
        
        
        
        self.loss_fn = nn.CTCLoss(
            blank=0,
            zero_infinity=True,
            reduction='sum'
        )

    def forward(self, src_input):
        keypoints = src_input['keypoints']
        mask = src_input['mask']
    
        body_embed = self.body_encoder(keypoints[:, :, self.cfg['body_idx'], :],mask )
        left_embed = self.left_encoder(keypoints[:, :, self.cfg['left_idx'], :],mask )
        right_embed = self.right_encoder(keypoints[:, :, self.cfg['right_idx'], :],mask )

        
        
        fuse_output = torch.cat([left_embed, right_embed, body_embed], dim=-1)
        left_output = torch.cat([left_embed, body_embed], dim=-1)
        right_output = torch.cat([right_embed, body_embed], dim=-1)
        
        
        
        valid_len_in = src_input['valid_len_in']
        mask_head = src_input['mask_head'][:, :fuse_output.shape[1]]
        
        left_head = self.left_visual_head(left_output, mask_head, valid_len_in)  
        right_head = self.right_visual_head(right_output, mask_head, valid_len_in)  
        fuse_head = self.fuse_visual_head(fuse_output, mask_head, valid_len_in)
        
        # head = self.visual_head(embed, mask_head, valid_len_in)
        
        head_outputs = {'ensemble_last_gloss_logits': (left_head['gloss_logits'] + 
                                                       right_head['gloss_logits'] + 
                                                       fuse_head['gloss_logits']).log(),
                            'fuse': fuse_output,
                            'fuse_gloss_logits': fuse_head['gloss_logits'],
                            'fuse_gloss_probabilities_log': fuse_head['gloss_probabilities_log'],
                            'left_gloss_logits': left_head['gloss_logits'],
                            'left_gloss_probabilities_log': left_head['gloss_probabilities_log'],
                            'right_gloss_logits': right_head['gloss_logits'],
                            'right_gloss_probabilities_log': right_head['gloss_probabilities_log'],
                            }
        # head_outputs = {
        #                     'fuse': head,
        #                     'fuse_gloss_logits': head['gloss_logits'],
        #                     'fuse_gloss_probabilities_log': head['gloss_probabilities_log'],
                           
        #                     }

        outputs = {**head_outputs,
                   'input_lengths':src_input['valid_len_in']}

        for k in ['fuse', 'left', 'right']:
            outputs[f'recognition_loss_{k}'] = self.compute_loss(
                gloss_labels=src_input['gloss_labels'],
                gloss_lengths=src_input['gloss_lengths'],
                gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                input_lengths=src_input['valid_len_in'])
        outputs['recognition_loss'] = outputs['recognition_loss_fuse'] + outputs['recognition_loss_left'] + outputs['recognition_loss_right']
        if self.cross_distillation:
            loss_func = torch.nn.KLDivLoss()
            for student in ['fuse']:
                teacher_prob = outputs['ensemble_last_gloss_probabilities_log']
                teacher_prob = teacher_prob.detach()
                student_log_prob = outputs[f'{student}_gloss_probabilities_log']
                if torch.isnan(student_log_prob).any():
                    raise ValueError("NaN in student_log_prob")
           
                if torch.isnan(teacher_prob).any():
                    raise ValueError("NaN in teacher_prob")
                outputs[f'{student}_distill_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
                
                print(f'{student}_distill_loss:', outputs[f'{student}_distill_loss'])
                outputs['recognition_loss'] += outputs[f'{student}_distill_loss']
        return outputs

        # return outputs

    def compute_loss(self, gloss_labels, gloss_lengths, gloss_probabilities_log, input_lengths):
        loss = self.loss_fn(
            log_probs = gloss_probabilities_log.permute(1,0,2),
            targets = gloss_labels,
            input_lengths = input_lengths,
            target_lengths = gloss_lengths
        )
        loss = loss/gloss_probabilities_log.shape[0]
        return loss


if __name__ == "__main__":
    pass
