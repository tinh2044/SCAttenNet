from torch import nn
import torch
from torch.nn import functional as F
from model.attention import CrossAttention, SelfAttention, SelfCausalAttention
from model.layers import CoordinateMapping, FeedForward, LearningPositionEmbedding

# from model.encoder import Encoder
# from model.decoder import Decoder
from model.residual import ResidualNetwork
from model.utils import create_attention_mask, create_causal_attention_mask


class KeypointModule(nn.Module):
    def __init__(self, joint_idx, num_frame, cfg=None):
        super().__init__()
        self.joint_idx = joint_idx
        self.num_frame = num_frame
        self.coordinate_mapping = CoordinateMapping(len(joint_idx), cfg["d_model"])
        self.sca = SeparativeCoordinateAttention(cfg)
        self.residual = ResidualNetwork(cfg["residual_blocks"])

    def forward(self, keypoints, attention_mask=None):
        x = keypoints[:, :, :, 0]
        y = keypoints[:, :, :, 1]

        x_embed, y_embed = self.coordinate_mapping(x, y)

        x_embed = self.sca(x_embed, y_embed, attention_mask)

        outputs, _ = self.residual(x_embed)
        return outputs


class CoordinateAttention(nn.Module):
    def __init__(self, cfg, attn_type="self_attn"):
        super(CoordinateAttention, self).__init__()
        self.attn_type = attn_type
        if attn_type == "self_attn":
            self.attn = SelfAttention(
                d_model=cfg["d_model"],
                num_heads=cfg["attention_heads"],
                dropout=cfg["attention_dropout"],
            )
            self.mlp = FeedForward(cfg["d_model"], cfg["ff_dim"], cfg["dropout"])
            self.last_layer_norm = nn.LayerNorm(cfg["d_model"])
        elif attn_type == "causal_attn":
            self.attn = SelfCausalAttention(
                d_model=cfg["d_model"],
                num_heads=cfg["attention_heads"],
                dropout=cfg["attention_dropout"],
            )
            self.mlp = nn.Identity()
            self.last_layer_norm = nn.Identity()
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        self.attn_layer_norm = nn.LayerNorm(cfg["d_model"])
        self.dropout = cfg["dropout"]
        self.activation_fn = nn.GELU()

    def forward(self, coord_embed, attention_mask=None):
        residual = coord_embed
        embed = self.attn(coord_embed, attention_mask)
        embed = F.dropout(embed, p=self.dropout, training=self.training)
        embed = residual + embed
        embed = self.attn_layer_norm(embed)

        if self.attn_type == "self_attn":
            residual = embed
            embed = self.mlp(embed)
            embed = residual + embed
            embed = self.last_layer_norm(embed)

        if embed.dtype == torch.float16 and (
            torch.isinf(embed).any() or torch.isnan(embed).any()
        ):
            clamp_value = torch.finfo(embed.dtype).max - 1000
            embed = torch.clamp(embed, min=-clamp_value, max=clamp_value)

        return embed


class CoordinatesMerge(nn.Module):
    def __init__(self, cfg):
        super(CoordinatesMerge, self).__init__()
        self.attn = CrossAttention(
            d_model=cfg["d_model"],
            num_heads=cfg["attention_heads"],
            dropout=cfg["attention_dropout"],
        )
        self.mlp = FeedForward(cfg["d_model"], cfg["ff_dim"], cfg["dropout"])
        self.attn_layer_norm = nn.LayerNorm(cfg["d_model"])
        self.last_layer_norm = nn.LayerNorm(cfg["d_model"])

        self.dropout = cfg["dropout"]

    def forward(self, y_embed, x_embed, cross_attn_mask=None):
        residual = y_embed
        embed = self.attn(y_embed, x_embed, cross_attn_mask)
        embed = F.dropout(embed, p=self.dropout, training=self.training)
        embed = residual + embed
        embed = self.attn_layer_norm(embed)

        residual = embed
        embed = self.mlp(embed)
        embed = residual + embed
        embed = self.last_layer_norm(embed)

        if embed.dtype == torch.float16 and (
            torch.isinf(embed).any() or torch.isnan(embed).any()
        ):
            clamp_value = torch.finfo(embed.dtype).max - 1000
            embed = torch.clamp(embed, min=-clamp_value, max=clamp_value)

        return embed


class SeparativeCoordinateAttention(nn.Module):
    def __init__(self, cfg=None):
        super(SeparativeCoordinateAttention, self).__init__()

        self.dropout = cfg["dropout"]

        self.self_attn_layers = nn.ModuleList(
            [
                CoordinateAttention(cfg, attn_type="self_attn")
                for _ in range(cfg["attn_layers"])
            ]
        )
        self.causal_attn_layers = nn.ModuleList(
            [
                CoordinateAttention(cfg, attn_type="causal_attn")
                for _ in range(cfg["attn_layers"])
            ]
        )

        self.coordinates_merge = nn.ModuleList(
            [CoordinatesMerge(cfg) for _ in range(cfg["attn_layers"])]
        )

        self.first_self_norm = nn.LayerNorm(cfg["d_model"])
        self.first_causal_norm = nn.LayerNorm(cfg["d_model"])

        self.self_pos_embed = LearningPositionEmbedding(
            cfg["max_position_embeddings"], cfg["d_model"]
        )
        self.causal_pos_embed = LearningPositionEmbedding(
            cfg["max_position_embeddings"], cfg["d_model"]
        )

        self.x_self = cfg.get("self_attn_x", True)

    def forward(self, x_embed, y_embed, attention_mask=None, return_attn_map=False):
        if self.x_self:
            self_embed = self.self_pos_embed(x_embed)
            causal_embed = self.causal_pos_embed(y_embed)
        else:
            self_embed = self.self_pos_embed(y_embed)
            causal_embed = self.causal_pos_embed(x_embed)

        self_embed = self.first_self_norm(self_embed)
        causal_embed = self.first_causal_norm(causal_embed)

        self_embed = F.dropout(self_embed, p=self.dropout, training=self.training)
        causal_embed = F.dropout(causal_embed, p=self.dropout, training=self.training)

        causal_shape = causal_embed.size()[:-1]
        self_attn_mask = create_attention_mask(attention_mask, self_embed.dtype)
        causal_attn_mask = create_causal_attention_mask(
            attention_mask, causal_shape, causal_embed
        )
        cross_attn_mask = create_attention_mask(
            attention_mask, causal_embed.dtype, tgt_len=causal_shape[-1]
        )

        self_attn_map = self_embed
        for self_attn_layer in self.self_attn_layers:
            self_attn_map = self_attn_layer(self_attn_map, self_attn_mask)

        causal_attn_map = causal_embed
        for causal_attn_layer, coordinates_merge in zip(
            self.causal_attn_layers, self.coordinates_merge
        ):
            causal_attn_map = causal_attn_layer(causal_attn_map, causal_attn_mask)
            causal_attn_map = coordinates_merge(
                causal_attn_map, self_attn_map, cross_attn_mask
            )

        outputs = causal_attn_map

        if return_attn_map:
            return {
                "outputs": outputs,
                "self_attn_map": self_attn_map,
                "causal_attn_map": causal_attn_map,
            }
        else:
            return outputs


# class SCA(nn.Module):
#     def __init__(self, joint_idx, num_frame, cfg=None):
#         super().__init__()
#         self.joint_idx = joint_idx
#         self.num_frame = num_frame
#         self.coordinate_mapping = CoordinateMapping(len(joint_idx), cfg["d_model"])

#         self.x_coord_module = Encoder(cfg)
#         self.y_coord_module = Decoder(cfg)

#         self.residual = ResidualNetwork(cfg["residual_blocks"])

#     def forward(self, keypoints, attention_mask=None):
#         x = keypoints[:, :, :, 0]
#         y = keypoints[:, :, :, 1]

#         x_embed, y_embed = self.coordinate_mapping(x, y)

#         x_embed = self.x_coord_module(x_embed, attention_mask)

#         y_embed = self.y_coord_module(
#             encoder_hidden_states=x_embed,
#             encoder_attention_mask=attention_mask,
#             y_embed=y_embed,
#             attention_mask=attention_mask,
#         )
#         y_embed = y_embed.permute(0, 2, 1)
#         outputs, _ = self.residual(y_embed)

#         return outputs
