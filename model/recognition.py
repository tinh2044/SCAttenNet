import torch
from torch import nn
from model.visual_head import VisualHead
from model.keypoint_module import KeypointModule



class RegionEncoder(nn.Module):
    def __init__(self, cfg, joint_idx):
        super().__init__()
        self.encoder = KeypointModule(joint_idx, num_frame=cfg['num_frame'], nets=cfg['nets'], cfg=cfg)
        
    def forward(self, keypoints, mask):
        return self.encoder(keypoints, mask)


class RecognitionHead(nn.Module):
    def __init__(self, cfg, gloss_tokenizer):
        super().__init__()
        # self.left_visual_head = VisualHead(**cfg['left_visual_head'], cls_num=len(gloss_tokenizer))
        # self.right_visual_head = VisualHead(**cfg['right_visual_head'], cls_num=len(gloss_tokenizer))
        # self.fuse_visual_head = VisualHead(**cfg['fuse_visual_head'], cls_num=len(gloss_tokenizer))
        
        self.keypoints_head = VisualHead(**cfg['visual_head'], cls_num=len(gloss_tokenizer))
        
    # def forward(self, left_output, right_output, fuse_output, mask_head, valid_len_in):
        # left_head = self.left_visual_head(left_output, mask_head, valid_len_in)  
        # right_head = self.right_visual_head(right_output, mask_head, valid_len_in)  
        # fuse_head = self.fuse_visual_head(fuse_output, mask_head, valid_len_in)
        
        # outputs = {
        #     'ensemble_last_gloss_logits': (left_head['gloss_logits'] + 
        #                                    right_head['gloss_logits'] + 
        #                                    fuse_head['gloss_logits']).log(),
        #     'fuse': fuse_output,
        #     'fuse_gloss_logits': fuse_head['gloss_logits'],
        #     'fuse_gloss_probabilities_log': fuse_head['gloss_probabilities_log'],
        #     'left_gloss_logits': left_head['gloss_logits'],
        #     'left_gloss_probabilities_log': left_head['gloss_probabilities_log'],
        #     'right_gloss_logits': right_head['gloss_logits'],
        #     'right_gloss_probabilities_log': right_head['gloss_probabilities_log'],
        # }
        
    def forward(self, keypoint_output, mask_head, valid_len_in):
        keypoint_head = self.keypoints_head(keypoint_output, mask_head, valid_len_in)
        outputs = {
            'keypoint_gloss_logits': keypoint_head['gloss_logits'],
            'keypoint_gloss_probabilities_log': keypoint_head['gloss_probabilities_log'],
            'keypoint_gloss_probabilities': keypoint_head['gloss_probabilities'],
        }
        
        return outputs


class RecognitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CTCLoss(
            blank=0,
            zero_infinity=True,
            reduction='sum'
        )
    
    def compute_loss(self, gloss_labels, gloss_lengths, gloss_probabilities_log, input_lengths):
        loss = self.loss_fn(
            log_probs = gloss_probabilities_log.permute(1,0,2),
            targets = gloss_labels,
            input_lengths = input_lengths,
            target_lengths = gloss_lengths
        )
        loss = loss/gloss_probabilities_log.shape[0]
        return loss
    
    def distillation_loss(self, student_log_prob, teacher_prob):
        loss_func = torch.nn.KLDivLoss()
        return loss_func(input=student_log_prob, target=teacher_prob)


class RecognitionNetwork(nn.Module):
    def __init__(self, cfg, gloss_tokenizer):
        super().__init__()
        self.cfg = cfg
        self.cross_distillation = cfg['cross_distillation']
        
        # Split into separate region encoders
        # self.body_encoder = RegionEncoder(cfg, cfg['body_idx'])
        # self.left_encoder = RegionEncoder(cfg, cfg['left_idx'])
        # self.right_encoder = RegionEncoder(cfg, cfg['right_idx'])
        
        self.keypoint_idx = cfg['body_idx'] + cfg['left_idx'] + cfg['right_idx']
        self.keypoint_encoder = RegionEncoder(cfg, self.keypoint_idx)
        
        # Visual heads module
        self.recognition_head = RecognitionHead(cfg, gloss_tokenizer)
        
        # Loss computation
        self.loss_module = RecognitionLoss()

    def forward(self, src_input):
        keypoints = src_input['keypoints']
        mask = src_input['mask']
        
        # # Process each region
        # body_embed = self.body_encoder(keypoints[:, :, self.cfg['body_idx'], :], mask)
        # left_embed = self.left_encoder(keypoints[:, :, self.cfg['left_idx'], :], mask)
        # right_embed = self.right_encoder(keypoints[:, :, self.cfg['right_idx'], :], mask)
        
        # # Fuse embeddings
        # fuse_output = torch.cat([left_embed, right_embed, body_embed], dim=-1)
        # left_output = torch.cat([left_embed, body_embed], dim=-1)
        # right_output = torch.cat([right_embed, body_embed], dim=-1)
        
        keypoints_embed = self.keypoint_encoder(keypoints[:, :, self.keypoint_idx, :], mask)

        valid_len_in = src_input['valid_len_in']
        mask_head = src_input['mask_head']

        
        # Get visual head outputs
        # head_outputs = self.recognition_head(left_output, right_output, fuse_output, mask_head, valid_len_in)
        head_outputs = self.recognition_head(keypoints_embed, mask_head, valid_len_in)
        
        # Prepare final outputs
        outputs = {**head_outputs, 'input_lengths': src_input['valid_len_in']}
        
        # Compute losses
        # for k in ['fuse', 'left', 'right']:
        for k in ['keypoint']:
            outputs[f'recognition_loss_{k}'] = self.loss_module.compute_loss(
                gloss_labels=src_input['gloss_labels'],
                gloss_lengths=src_input['gloss_lengths'],
                gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                input_lengths=src_input['valid_len_in'])
        
        # outputs['recognition_loss'] = outputs['recognition_loss_fuse'] + outputs['recognition_loss_left'] + outputs['recognition_loss_right']
        outputs['recognition_loss'] = outputs['recognition_loss_keypoint']
        
        # Apply distillation if enabled
        if self.cross_distillation:
            for student in ['fuse']:
                teacher_prob = outputs['ensemble_last_gloss_probabilities_log']
                teacher_prob = teacher_prob.detach()
                student_log_prob = outputs[f'{student}_gloss_probabilities_log']
                
                if torch.isnan(student_log_prob).any():
                    raise ValueError("NaN in student_log_prob")
           
                if torch.isnan(teacher_prob).any():
                    raise ValueError("NaN in teacher_prob")
                
                outputs[f'{student}_distill_loss'] = self.loss_module.distillation_loss(
                    student_log_prob=student_log_prob,
                    teacher_prob=teacher_prob
                )
                
                outputs['recognition_loss'] += outputs[f'{student}_distill_loss']
        
        return outputs


if __name__ == "__main__":
    pass

