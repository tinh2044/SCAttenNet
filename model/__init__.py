import torch
from torch import nn
from model.keypoint_module import KeypointModule
from model.alignment_module import AlignmentModule
from model.fusion import CoordinatesFusion
from loss import SeqKD
import torch.nn.functional as F


class RecognitionHead(nn.Module):
    def __init__(self, cfg, gloss_tokenizer):
        super().__init__()

        self.left_gloss_classifier = nn.Linear(cfg["d_model"], len(gloss_tokenizer))
        self.right_gloss_classifier = nn.Linear(cfg["d_model"], len(gloss_tokenizer))
        self.body_gloss_classifier = nn.Linear(cfg["d_model"], len(gloss_tokenizer))
        self.fuse_coord_classifier = nn.Linear(
            cfg["out_fusion_dim"], len(gloss_tokenizer)
        )

        self.fuse_alignment_head = AlignmentModule(
            **cfg["alignment_module"], cls_num=len(gloss_tokenizer)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to prevent NaN/inf issues"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

        # Clamp logits to prevent extreme values
        left_logits = torch.clamp(left_logits, min=-50, max=50)
        right_logits = torch.clamp(right_logits, min=-50, max=50)
        fuse_logits = torch.clamp(fuse_logits, min=-50, max=50)
        body_logits = torch.clamp(body_logits, min=-50, max=50)
        fuse_coord_logits = torch.clamp(fuse_coord_logits, min=-50, max=50)

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
        self.left_idx = cfg["left_idx"]
        self.gloss_tokenizer = gloss_tokenizer

        self.self_distillation = cfg["self_distillation"]

        # Add gradient clipping threshold
        self.gradient_clip_val = cfg.get("gradient_clip_val", 1.0)

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

        self.loss_fn = nn.CTCLoss(
            reduction="none", zero_infinity=True, blank=0
        )  # Changed to True
        self.distillation_loss = SeqKD()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to prevent NaN/inf issues"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, src_input, **kwargs):
        if torch.cuda.is_available() and self.device == "cuda":
            src_input = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in src_input.items()
            }

        keypoints = src_input["keypoints"]
        mask = src_input["mask"]

        # Check for NaN/inf in input
        if torch.isnan(keypoints).any() or torch.isinf(keypoints).any():
            raise ValueError("NaN or inf in input keypoints")

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

        # Check embeddings for NaN/inf
        if torch.isnan(body_embed).any() or torch.isinf(body_embed).any():
            raise ValueError("NaN or inf in body_embed")
        if torch.isnan(left_embed).any() or torch.isinf(left_embed).any():
            raise ValueError("NaN or inf in left_embed")
        if torch.isnan(right_embed).any() or torch.isinf(right_embed).any():
            raise ValueError("NaN or inf in right_embed")

        fuse_embed = self.coordinates_fusion(
            left_embed, right_embed, body_embed
        )  # (B,T/4, D)

        if torch.isnan(fuse_embed).any() or torch.isinf(fuse_embed).any():
            raise ValueError("NaN or inf in fuse_embed")

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
            tgt_lengths=src_input["gloss_lengths"],
            logits=outputs["alignment_gloss_logits"],
            input_lengths=outputs["input_lengths"],
        )
        outputs["fuse_coord_loss"] = self.compute_loss(
            labels=src_input["gloss_labels"],
            tgt_lengths=src_input["gloss_lengths"],
            logits=outputs["fuse_coord_gloss_logits"],
            input_lengths=outputs["input_lengths"],
        )

        # Check for NaN/inf with more informative error messages
        if torch.isnan(outputs["alignment_loss"]) or torch.isinf(
            outputs["alignment_loss"]
        ):
            print(f"Alignment loss value: {outputs['alignment_loss']}")
            print(
                f"Alignment logits stats: min={outputs['alignment_gloss_logits'].min()}, max={outputs['alignment_gloss_logits'].max()}"
            )
            raise ValueError("NaN or inf in alignment_loss")
        if torch.isnan(outputs["fuse_coord_loss"]) or torch.isinf(
            outputs["fuse_coord_loss"]
        ):
            print(f"Fuse coord loss value: {outputs['fuse_coord_loss']}")
            print(
                f"Fuse coord logits stats: min={outputs['fuse_coord_gloss_logits'].min()}, max={outputs['fuse_coord_gloss_logits'].max()}"
            )
            print(f"Input lengths: {outputs['input_lengths']}")
            print(f"Target lengths: {src_input['gloss_lengths']}")
            raise ValueError("NaN or inf in fuse_coord_loss")

        outputs["total_loss"] += outputs["fuse_coord_loss"]

        if self.self_distillation:
            for student, weight in self.cfg["distillation_weight"].items():
                teacher_logits = outputs["fuse_coord_gloss_logits"]
                teacher_logits = teacher_logits.detach()
                student_logits = outputs[f"{student}"]

                if torch.isnan(student_logits).any():
                    raise ValueError(f"NaN in student_logits for {student}")

                if torch.isnan(teacher_logits).any():
                    raise ValueError(f"NaN in teacher_logits for {student}")

                distill_loss = weight * self.distillation_loss(
                    student_logits, teacher_logits, use_blank=False
                )

                # Clamp distillation loss to prevent extreme values
                distill_loss = torch.clamp(distill_loss, min=-100, max=100)
                outputs[f"{student}_distill_loss"] = distill_loss

                if torch.isnan(outputs[f"{student}_distill_loss"]) or torch.isinf(
                    outputs[f"{student}_distill_loss"]
                ):
                    print(
                        f"Distillation loss for {student}: {outputs[f'{student}_distill_loss']}"
                    )
                    raise ValueError(f"NaN or inf in {student}_distill_loss")

                outputs["total_loss"] += outputs[f"{student}_distill_loss"]

        return outputs

    def compute_loss(self, labels, tgt_lengths, logits, input_lengths):
        try:
            logits = logits.permute(1, 0, 2)

            # Add small epsilon to prevent log(0)
            epsilon = 1e-8
            log_probs = F.log_softmax(logits, dim=-1)

            # Ensure log_probs are stable
            log_probs = torch.clamp(log_probs, min=-100, max=0)

            # Check for invalid values before CTC loss
            if torch.isnan(log_probs).any():
                raise ValueError("NaN in log_probs")
            if torch.isinf(log_probs).any():
                raise ValueError("inf in log_probs")

            # Ensure input_lengths and tgt_lengths are valid
            input_lengths = torch.clamp(input_lengths, min=1)
            tgt_lengths = torch.clamp(tgt_lengths, min=1)

            # Ensure input_lengths >= tgt_lengths for CTC
            input_lengths = torch.maximum(input_lengths, tgt_lengths)

            loss = self.loss_fn(
                log_probs,
                labels.cpu().int(),
                input_lengths.cpu().int(),
                tgt_lengths.cpu().int(),
            )

            # Filter out infinite losses and take mean of valid losses
            valid_loss_mask = torch.isfinite(loss)
            if valid_loss_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            loss = loss[valid_loss_mask].mean()

            # Clamp final loss to prevent extreme values
            loss = torch.clamp(loss, min=0, max=100)

        except Exception as e:
            print(f"Error in CTC loss: {str(e)}")
            print(f"Logits shape: {logits.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Input lengths: {input_lengths}")
            print(f"Target lengths: {tgt_lengths}")
            raise e

        return loss
