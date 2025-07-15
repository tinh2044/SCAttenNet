import torch
from torch import nn
from model.keypoint_module import KeypointModule
from model.alignment_module import AlignmentModule
from model.fusion import CoordinatesFusion
from loss import SeqKD


class RecognitionHead(nn.Module):
    def __init__(self, cfg, gloss_tokenizer):
        super().__init__()

        self.left_gloss_classifier = nn.Linear(
            cfg["residual_blocks"][-1], len(gloss_tokenizer)
        )
        self.right_gloss_classifier = nn.Linear(
            cfg["residual_blocks"][-1], len(gloss_tokenizer)
        )
        self.body_gloss_classifier = nn.Linear(
            cfg["residual_blocks"][-1], len(gloss_tokenizer)
        )
        self.fuse_coord_classifier = nn.Linear(
            cfg["out_fusion_dim"], len(gloss_tokenizer)
        )

        self.fuse_alignment_head = AlignmentModule(
            **cfg["alignment_module"], cls_num=len(gloss_tokenizer)
        )

    def forward(
        self,
        left_output,
        right_output,
        fuse_output,
        body_output,
    ):
        left_logits = self.left_gloss_classifier(left_output)
        right_logits = self.right_gloss_classifier(right_output)
        fuse_logits = self.fuse_alignment_head(fuse_output.permute(1, 0, 2))
        body_logits = self.body_gloss_classifier(body_output)
        fuse_coord_logits = self.fuse_coord_classifier(fuse_output)
        outputs = {
            "alignment_gloss_logits": fuse_logits,
            "left": left_logits,
            "right": right_logits,
            "body": body_logits,
            "fuse_coord_gloss_logits": fuse_coord_logits,
        }
        return outputs


class MSCA_Net(torch.nn.Module):
    def __init__(self, cfg, gloss_tokenizer, device="cpu"):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.cfg["fuse_idx"] = cfg["left_idx"] + cfg["right_idx"] + cfg["body_idx"]

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
        self.coordinates_fusion = CoordinatesFusion(
            cfg["in_fusion_dim"], cfg["out_fusion_dim"], 0.2
        )
        self.recognition_head = RecognitionHead(cfg, gloss_tokenizer)

        self.loss_fn = nn.CTCLoss(reduction="mean", zero_infinity=False, blank=0)
        self.distillation_loss = SeqKD()

    def forward(self, src_input, **kwargs):
        if torch.cuda.is_available() and self.device == "cuda":
            src_input = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in src_input.items()
            }

        keypoints = src_input["keypoints"]
        mask = src_input["mask"]

        body_embed = self.body_encoder(
            keypoints[:, :, self.cfg["body_idx"], :], mask
        )  # (B,T/4,D)
        left_embed = self.left_encoder(
            keypoints[:, :, self.cfg["left_idx"], :], mask
        )  # (B,T/4,D)
        right_embed = self.right_encoder(
            keypoints[:, :, self.cfg["right_idx"], :],
            mask,  # (B,T/4,D)
        )
        fuse_embed = self.coordinates_fusion(
            left_embed, right_embed, body_embed
        )  # (B,T/4, D)
        head_outputs = self.recognition_head(
            left_embed, right_embed, fuse_embed, body_embed
        )

        for k, v in head_outputs.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN in {k}")
            if torch.isinf(v).any():
                raise ValueError(f"inf in {k}")
        outputs = {
            **head_outputs,
            "input_lengths": src_input["valid_len_in"],
            "total_loss": 0,
        }

        outputs["alignment_loss"] = self.compute_loss(
            labels=src_input["gloss_labels"],
            lengths=src_input["gloss_lengths"],
            logits=outputs["alignment_gloss_logits"],
            input_lengths=outputs["input_lengths"],
        )
        outputs["fuse_coord_loss"] = self.compute_loss(
            labels=src_input["gloss_labels"],
            lengths=src_input["gloss_lengths"],
            logits=outputs["fuse_coord_gloss_logits"],
            input_lengths=outputs["input_lengths"],
        )
        outputs["total_loss"] += outputs["alignment_loss"] + outputs["fuse_coord_loss"]

        if self.self_distillation:
            for student, weight in self.cfg["distillation_weight"].items():
                teacher_logits = outputs["alignment_gloss_logits"]
                teacher_logits = teacher_logits.detach()
                student_logits = outputs[f"{student}"]

                if torch.isnan(student_logits).any():
                    raise ValueError("NaN in student_logits")

                if torch.isnan(teacher_logits).any():
                    raise ValueError("NaN in teacher_logits")

                outputs[f"{student}_distill_loss"] = weight * self.distillation_loss(
                    student_logits, teacher_logits, use_blank=False
                )
                if torch.isnan(outputs[f"{student}_distill_loss"]) or torch.isinf(
                    outputs[f"{student}_distill_loss"]
                ):
                    raise ValueError(f"NaN or inf in {student}_distill_loss")

                outputs["total_loss"] += outputs[f"{student}_distill_loss"]

        return outputs

    def compute_loss(self, labels, lengths, logits, input_lengths):
        try:
            logits = logits.log_softmax(dim=-1)
            logits = logits.permute(1, 0, 2)

            loss = self.loss_fn(
                log_probs=logits,
                targets=labels,
                input_lengths=input_lengths,
                target_lengths=lengths,
            )

        except Exception as e:
            print(f"Error in CTC loss: {str(e)}")
            raise e

        return loss
