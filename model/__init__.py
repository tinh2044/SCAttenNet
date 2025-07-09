import torch
from torch import nn
from model.keypoint_module import SeparativeCoordinateAttention
from model.feature_alignment import FeatureAlignment
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

        self.fuse_alignment_head = FeatureAlignment(
            **cfg["fuse_alignment"], cls_num=len(gloss_tokenizer)
        )

    def forward(
        self,
        left_output,
        right_output,
        fuse_output,
        body_output,
    ):
        left_output = left_output.permute(2, 0, 1)
        right_output = right_output.permute(2, 0, 1)
        body_output = body_output.permute(2, 0, 1)
        fuse_output = fuse_output.permute(2, 0, 1)

        left_logits = self.left_gloss_classifier(left_output)
        right_logits = self.right_gloss_classifier(right_output)
        fuse_logits = self.fuse_alignment_head(fuse_output)
        body_logits = self.body_gloss_classifier(body_output)
        outputs = {
            "fuse_gloss_logits": fuse_logits,
            "left_gloss_logits": left_logits,
            "right_gloss_logits": right_logits,
            "body_gloss_logits": body_logits,
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

        self.body_encoder = SeparativeCoordinateAttention(
            cfg["body_idx"], num_frame=cfg["num_frame"], cfg=cfg
        )
        self.left_encoder = SeparativeCoordinateAttention(
            cfg["left_idx"], num_frame=cfg["num_frame"], cfg=cfg
        )
        self.right_encoder = SeparativeCoordinateAttention(
            cfg["right_idx"], num_frame=cfg["num_frame"], cfg=cfg
        )
        self.coordinates_fusion = CoordinatesFusion(
            cfg["residual_blocks"][-1], cfg["residual_blocks"][-1], 0.2
        )
        self.recognition_head = RecognitionHead(cfg, gloss_tokenizer)

        self.loss_fn = nn.CTCLoss(reduction="none", zero_infinity=False)
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
        )  # (B, D, T/4)
        left_embed = self.left_encoder(
            keypoints[:, :, self.cfg["left_idx"], :], mask
        )  # (B, D, T/4)
        right_embed = self.right_encoder(
            keypoints[:, :, self.cfg["right_idx"], :],
            mask,  # (B, D, T/4)
        )
        fuse_embed = self.coordinates_fusion(
            left_embed, right_embed, body_embed
        )  # (B, 3D, T/4)
        head_outputs = self.recognition_head(
            left_embed, right_embed, fuse_embed, body_embed
        )

        outputs = {**head_outputs, "input_lengths": src_input["valid_len_in"]}
        outputs["total_loss"] = 0
        for k in ["left", "right", "body", "fuse"]:
            if torch.isnan(outputs[f"{k}_gloss_logits"]).any():
                raise ValueError(f"NaN in {k}_gloss_logits")
            if torch.isinf(outputs[f"{k}_gloss_logits"]).any():
                raise ValueError(f"inf in {k}_gloss_logits")

        for k in ["left", "right", "body", "fuse"]:
            l = self.compute_loss(
                labels=src_input["gloss_labels"],
                lengths=src_input["gloss_lengths"],
                logits=outputs[f"{k}_gloss_logits"],
                input_lengths=src_input["valid_len_in"],
            )

            if torch.isnan(l) or torch.isinf(l):
                raise ValueError(f"NaN or inf in {k}_loss")

            outputs[f"{k}_loss"] = l
            outputs["total_loss"] += outputs[f"{k}_loss"]

        if self.self_distillation:
            for student, weight in self.cfg["distillation_weight"].items():
                teacher_logits = outputs["fuse_gloss_logits"]
                teacher_logits = teacher_logits.detach()
                student_logits = outputs[f"{student}_gloss_logits"]

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
        batch_size = logits.shape[1]

        if input_lengths.shape[0] != batch_size:
            raise ValueError(
                f"input_lengths size {input_lengths.shape[0]} doesn't match batch_size {batch_size}"
            )
        if lengths.shape[0] != batch_size:
            raise ValueError(
                f"target_lengths size {lengths.shape[0]} doesn't match batch_size {batch_size}"
            )

        seq_len = logits.shape[0]
        input_lengths = torch.clamp(input_lengths, max=seq_len)

        input_lengths = torch.clamp(input_lengths, min=1)
        lengths = torch.clamp(lengths, min=1)

        input_lengths = input_lengths.long()
        lengths = lengths.long()

        log_probs = torch.log_softmax(logits, dim=-1)

        try:
            loss = self.loss_fn(
                log_probs=log_probs,
                targets=labels,
                input_lengths=input_lengths,
                target_lengths=lengths,
            )

            loss = loss.mean()
        except Exception as e:
            print(f"  log_probs range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
            print(f"  labels unique values: {torch.unique(labels)}")
            print(f"  Any NaN in log_probs: {torch.isnan(log_probs).any()}")
            print(f"  Any inf in log_probs: {torch.isinf(log_probs).any()}")
            raise e

        return loss
