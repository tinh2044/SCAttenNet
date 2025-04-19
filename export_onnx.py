import torch
import yaml
import argparse
from pathlib import Path

from model import SignLanguageModel
from Tokenizer import GlossTokenizer

def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser(description="Export SignBart model to ONNX")
    parser.add_argument('--cfg_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--onnx_path', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--device', default='cpu', help='Device to use')
    args = parser.parse_args()

    config = load_config(args.cfg_path)
    device = torch.device(args.device)
    gloss_tokenizer = GlossTokenizer(config['gloss_tokenizer'])

    model = SignLanguageModel(cfg=config, gloss_tokenizer=gloss_tokenizer, device=args.device)
    # checkpoint = torch.load(args.checkpoint, map_location=device)
    # model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    model.to(device)
    
    print("Number of parameters in the model: ")
    n = sum(p.numel() for p in model.parameters())
    print(f"{n:,} (num_params)")

    # Prepare dummy input
    batch_size = 1
    seq_len = config['data']['max_len']
    num_joints = len(config['data']['joint_parts'])
    keypoints = torch.randn(batch_size, seq_len, 600, 2, device=device)
    valid_len_in = torch.tensor([seq_len], dtype=torch.long, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    mask_head = torch.ones(batch_size, seq_len // 4, dtype=torch.long, device=device)  # adjust as needed

    gloss_labels = torch.zeros(batch_size, 10, dtype=torch.long, device=device)  # dummy
    gloss_lengths = torch.ones(batch_size, dtype=torch.long, device=device)
    gloss_input = ['DUMMY']
    text_input = ['DUMMY']
    name = ['sample']

    src_input = {
        "name": name,
        "keypoints": keypoints,
        "valid_len_in": valid_len_in,
        "mask": mask,
        "mask_head": mask_head,
        "gloss_labels": gloss_labels,
        "gloss_lengths": gloss_lengths,
        "gloss_input": gloss_input,
        "text_input": text_input
    }

    # Wrapper for ONNX export
    class SignBartONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, keypoints, valid_len_in, mask, mask_head, gloss_labels, gloss_lengths):
            src_input = {
                "name": ['sample'],
                "keypoints": keypoints,
                "valid_len_in": valid_len_in,
                "mask": mask,
                "mask_head": mask_head,
                "gloss_labels": gloss_labels,
                "gloss_lengths": gloss_lengths,
                "gloss_input": ['DUMMY'],
                "text_input": ['DUMMY']
            }
            out = self.model(src_input)
            # Return a tensor output for ONNX (e.g., gloss_logits)
            return out['fuse_gloss_logits']

    onnx_model = SignBartONNXWrapper(model)

    torch.onnx.export(
        onnx_model,
        (keypoints, valid_len_in, mask, mask_head, gloss_labels, gloss_lengths),
        args.onnx_path,
        input_names=["keypoints", "valid_len_in", "mask", "mask_head", "gloss_labels", "gloss_lengths"],
        output_names=["fuse_gloss_logits"],
        opset_version=14,  # <-- changed from 13 to 14
        dynamic_axes={k: {0: 'batch'} for k in ["keypoints", "valid_len_in", "mask", "mask_head", "gloss_labels", "gloss_lengths"]},
        do_constant_folding=True
    )
    print(f"ONNX model exported to {args.onnx_path}")

if __name__ == "__main__":
    main()
