import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, downsample=False, num_tconv=0):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.in_dim = in_dim
        self.out_dim = out_dim

        stride = 2 if downsample else 1
        self.need_projection = in_dim != out_dim or downsample

        if self.need_projection:
            self.projection = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=stride)

        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.need_projection:
            residual = self.projection(residual)

        out = out + residual
        out = self.relu(out)

        return out


class ResidualNetwork(nn.Module):
    def __init__(self, residual_blocks):
        super(ResidualNetwork, self).__init__()

        self.residual_blocks = residual_blocks
        self.blocks = nn.ModuleList()

        for i in range(len(residual_blocks)):
            in_dim = residual_blocks[i - 1] if i > 0 else residual_blocks[0]
            out_dim = residual_blocks[i]
            downsample = i in [0, 2]
            self.blocks.append(
                ResidualBlock(in_dim, out_dim, downsample=downsample, num_tconv=i + 1)
            )

    def forward(self, x):
        # Input shape: (B, D, T)
        outputs = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            outputs.append(x)

        return x, outputs


if __name__ == "__main__":
    batch_size = 32
    T = 180
    initial_dim = 256

    residual_blocks = [256, 512, 1024, 2048]

    model = ResidualNetwork(residual_blocks)
    input_tensor = torch.randn(batch_size, initial_dim, T)

    final_output, intermediate_outputs = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Final output shape: {final_output.shape}")
    print("Intermediate output shapes:")
    for i, output in enumerate(intermediate_outputs):
        print(f"  Block {i + 1}: {output.shape}")

    model.eval()

    input_names = ["input"]
    output_names = ["final_output"] + [
        f"intermediate_output_{i}" for i in range(len(residual_blocks))
    ]

    dynamic_axes = {
        "input": {0: "batch_size", 2: "sequence_length"},
        "final_output": {0: "batch_size", 2: "sequence_length"},
    }

    for i in range(len(residual_blocks)):
        dynamic_axes[f"intermediate_output_{i}"] = {
            0: "batch_size",
            2: "sequence_length",
        }
