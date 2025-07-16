#!/usr/bin/env python3
"""
Script to calculate FLOPs for MSCA_Net model
Usage: python calculate_flops.py --cfg_path configs/phoenix-2014.yaml
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from Tokenizer import GlossTokenizer
from model import MSCA_Net
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Calculate FLOPs for MSCA_Net model", add_help=False
    )
    parser.add_argument(
        "--cfg_path", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for FLOPs calculation"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to use for calculation (cpu/cuda)"
    )
    return parser


def main(args):
    # Load configuration
    with open(args.cfg_path, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Initialize tokenizer
    gloss_tokenizer = GlossTokenizer(config["gloss_tokenizer"])

    # Initialize model
    model = MSCA_Net(
        cfg=config["model"], gloss_tokenizer=gloss_tokenizer, device=device
    )
    model = model.to(device)
    model.eval()

    # Calculate total number of joints (maximum index + 1)
    all_joint_indices = (
        config["model"]["body_idx"]
        + config["model"]["left_idx"]
        + config["model"]["right_idx"]
    )
    max_joint_index = max(all_joint_indices)

    # Add joint_parts if they exist in data config
    if "joint_parts" in config["data"]:
        for part in config["data"]["joint_parts"]:
            if isinstance(part, list):
                max_joint_index = max(max_joint_index, max(part))

    # Prepare input shape information
    input_shape = {
        "batch_size": args.batch_size,
        "seq_len": config["data"]["max_len"],
        "num_joints": max_joint_index + 1,  # +1 because indices are 0-based
        "vocab_size": len(gloss_tokenizer),
    }

    print("=" * 60)
    print("MSCA_Net Model Analysis")
    print("=" * 60)

    print(f"Configuration: {args.cfg_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {input_shape['seq_len']}")
    print(f"Number of joints: {input_shape['num_joints']}")
    print(f"  - Body joints: {len(config['model']['body_idx'])}")
    print(f"  - Left hand joints: {len(config['model']['left_idx'])}")
    print(f"  - Right hand joints: {len(config['model']['right_idx'])}")
    print(f"Vocabulary size: {input_shape['vocab_size']}")
    print(f"Model dimension: {config['model']['d_model']}")
    print()

    # Calculate model information
    print("Calculating model information...")
    model_info = utils.get_model_info(model, input_shape, device)

    print("=" * 60)
    print("Model Information")
    print("=" * 60)

    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")
    print(f"Non-trainable parameters: {model_info['non_trainable_params']:,}")

    # Calculate parameter size in MB
    param_size_mb = model_info["total_params"] * 4 / (1024 * 1024)  # Assuming float32
    print(f"Model size (float32): {param_size_mb:.2f} MB")

    if "flops" in model_info:
        print(f"\nComputational Complexity:")
        print(f"FLOPs: {model_info['flops_str']}")
        print(f"MACs: {model_info['macs_str']}")
        print(
            f"FLOPs per sequence: {model_info['flops'] / args.batch_size / 1e9:.3f} GFLOPs"
        )

        # Memory estimation
        if device == "cuda":
            print(f"\nMemory estimation (approximate):")
            print(f"Model parameters: {param_size_mb:.2f} MB")
            # Rough estimation of activation memory
            activation_memory = (
                input_shape["batch_size"]
                * input_shape["seq_len"]
                * input_shape["num_joints"]
                * 4
                * 10
            )  # rough estimate
            print(
                f"Activation memory (rough): {activation_memory / (1024 * 1024):.2f} MB"
            )

    print("\n" + "=" * 60)
    print("Model Architecture Summary:")
    print("=" * 60)

    # Print model architecture details
    print(f"Attention layers: {config['model']['attn_layers']}")
    print(f"Attention heads: {config['model']['attention_heads']}")
    print(f"Feed-forward dimension: {config['model']['ff_dim']}")
    print(f"Residual blocks: {config['model']['residual_blocks']}")
    print(f"Fusion input dim: {config['model']['in_fusion_dim']}")
    print(f"Fusion output dim: {config['model']['out_fusion_dim']}")

    print("\n" + "=" * 60)
    print("Calculation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Calculate FLOPs for MSCA_Net", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
