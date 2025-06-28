import torch
from torch import nn
from model.keypoint_module import KeypointModule
from model.visual_head import VisualHead


class RecognitionHead(nn.Module):
    def __init__(self, cfg, gloss_tokenizer):
        super().__init__()
        self.left_visual_head = VisualHead(
            **cfg["left_visual_head"], cls_num=len(gloss_tokenizer)
        )
        self.right_visual_head = VisualHead(
            **cfg["right_visual_head"], cls_num=len(gloss_tokenizer)
        )
        self.fuse_visual_head = VisualHead(
            **cfg["fuse_visual_head"], cls_num=len(gloss_tokenizer)
        )
        self.body_visual_head = VisualHead(
            **cfg["body_visual_head"], cls_num=len(gloss_tokenizer)
        )

    def forward(
        self,
        left_output,
        right_output,
        fuse_output,
        body_output,
        mask_head,
        valid_len_in,
    ):
        left_head = self.left_visual_head(left_output, mask_head, valid_len_in)
        right_head = self.right_visual_head(right_output, mask_head, valid_len_in)
        fuse_head = self.fuse_visual_head(fuse_output, mask_head, valid_len_in)
        body_head = self.body_visual_head(body_output, mask_head, valid_len_in)
        outputs = {
            "ensemble_last_gloss_logits": (
                left_head["gloss_logits"]
                + right_head["gloss_logits"]
                + fuse_head["gloss_logits"]
                + body_head["gloss_logits"]
            ).log(),
            "fuse": fuse_output,
            "fuse_gloss_logits": fuse_head["gloss_logits"],
            "fuse_gloss_prob_log": fuse_head["gloss_prob_log"],
            "left_gloss_logits": left_head["gloss_logits"],
            "left_gloss_prob_log": left_head["gloss_prob_log"],
            "right_gloss_logits": right_head["gloss_logits"],
            "right_gloss_prob_log": right_head["gloss_prob_log"],
            "body_gloss_logits": body_head["gloss_logits"],
            "body_gloss_prob_log": body_head["gloss_prob_log"],
        }

        return outputs


class SelfDistillation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.self_distillation = cfg["self_distillation"]


class SCAttentNet(torch.nn.Module):
    def __init__(self, cfg, gloss_tokenizer, device="cpu"):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.text_tokenizer = None

        self.gloss_tokenizer = gloss_tokenizer

        self.self_distillation = cfg["self_distillation"]

        self.body_encoder = KeypointModule(
            cfg["body_idx"], num_frame=cfg["num_frame"], cfg=cfg
        )
        self.left_encoder = KeypointModule(
            cfg["left_idx"], num_frame=cfg["num_frame"], cfg=cfg
        )
        self.right_encoder = KeypointModule(
            cfg["right_idx"], num_frame=cfg["num_frame"], cfg=cfg
        )

        self.recognition_head = RecognitionHead(cfg, gloss_tokenizer)

        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum")

    def forward(self, src_input, **kwargs):
        if torch.cuda.is_available() and self.device == "cuda":
            src_input = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in src_input.items()
            }

        keypoints = src_input["keypoints"]
        mask = src_input["mask"]

        body_embed = self.body_encoder(keypoints[:, :, self.cfg["body_idx"], :], mask)
        left_embed = self.left_encoder(keypoints[:, :, self.cfg["left_idx"], :], mask)
        right_embed = self.right_encoder(
            keypoints[:, :, self.cfg["right_idx"], :], mask
        )

        fuse_output = torch.cat([left_embed, right_embed, body_embed], dim=-1)
        left_output = torch.cat([left_embed, body_embed], dim=-1)
        right_output = torch.cat([right_embed, body_embed], dim=-1)

        valid_len_in = src_input["valid_len_in"]
        mask_head = src_input["mask_head"]

        head_outputs = self.recognition_head(
            left_output, right_output, fuse_output, body_embed, mask_head, valid_len_in
        )

        outputs = {**head_outputs, "input_lengths": src_input["valid_len_in"]}
        outputs["total_loss"] = 0
        for k in ["fuse", "left", "right", "body"]:
            outputs[f"{k}_loss"] = self.compute_loss(
                gloss_labels=src_input["gloss_labels"],
                gloss_lengths=src_input["gloss_lengths"],
                gloss_prob_log=head_outputs[f"{k}_gloss_prob_log"],
                input_lengths=src_input["valid_len_in"],
            )

            outputs["total_loss"] += outputs[f"{k}_loss"]

        if self.self_distillation:
            for student in ["body", "left", "right", "fuse"]:
                teacher_prob = outputs["ensemble_last_gloss_prob_log"]
                teacher_prob = teacher_prob.detach()
                student_log_prob = outputs[f"{student}_gloss_prob_log"]

                if torch.isnan(student_log_prob).any():
                    raise ValueError("NaN in student_log_prob")

                if torch.isnan(teacher_prob).any():
                    raise ValueError("NaN in teacher_prob")

                outputs[f"{student}_distill_loss"] = self.distillation_loss(
                    student_log_prob=student_log_prob, teacher_prob=teacher_prob
                )

                outputs["total_loss"] += outputs[f"{student}_distill_loss"]

        return outputs

    def compute_loss(self, gloss_labels, gloss_lengths, gloss_prob_log, input_lengths):
        loss = self.loss_fn(
            log_probs=gloss_prob_log.permute(1, 0, 2),
            targets=gloss_labels,
            input_lengths=input_lengths,
            target_lengths=gloss_lengths,
        )
        loss = loss / gloss_prob_log.shape[0]
        return loss

    def distillation_loss(self, student_log_prob, teacher_prob):
        loss_func = torch.nn.KLDivLoss()
        return loss_func(input=student_log_prob, target=teacher_prob)
