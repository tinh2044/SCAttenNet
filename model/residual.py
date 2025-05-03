import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.need_projection = (in_dim != out_dim)
        if self.need_projection:
            self.projection = nn.Linear(in_dim, out_dim)
        
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        if self.downsample:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        residual = x
        
        if self.need_projection:
            residual = self.projection(residual)
        
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = self.relu(out)
        
        if self.downsample:
            out = out.permute(0, 2, 1)
            out = self.pool(out)
            out = out.permute(0, 2, 1)
            
        return out

class ResidualNetwork(nn.Module):
    def __init__(self, residual_blocks):
        super(ResidualNetwork, self).__init__()
        
        self.residual_blocks = residual_blocks
        self.blocks = nn.ModuleList()
        
        self.shortcuts = nn.ModuleList()
        
        for i in range(len(residual_blocks)):
            in_dim = residual_blocks[i-1] if i > 0 else residual_blocks[0]
            out_dim = residual_blocks[i]
            downsample = (i % 2 == 1)
            self.blocks.append(ResidualBlock(in_dim, out_dim, downsample=downsample))
            
            if i > 0:
                need_projection = residual_blocks[i-2] != residual_blocks[i] if i > 1 else residual_blocks[0] != residual_blocks[i]
                need_downsample = (i % 2 == 0) and ((i-1) % 2 == 1)
                
                if need_projection or need_downsample:
                    shortcut = nn.Sequential()
                    if need_projection:
                        shortcut_in_dim = residual_blocks[i-2] if i > 1 else residual_blocks[0]
                        shortcut.add_module('projection', nn.Linear(shortcut_in_dim, residual_blocks[i]))
                    
                    if need_downsample:
                        shortcut.add_module('permute1', PermuteLayer(0, 2, 1))
                        shortcut.add_module('pool', nn.MaxPool1d(kernel_size=2, stride=2))
                        shortcut.add_module('permute2', PermuteLayer(0, 2, 1))
                    
                    self.shortcuts.append(shortcut)
                else:
                    self.shortcuts.append(None)
        
    def forward(self, x):
        outputs = []
        shortcut_outputs = [x]  # Lưu trữ output cho shortcut connections
        
        for i, block in enumerate(self.blocks):
            if i == 0:
                x = block(x)
            else:
        
                shortcut_idx = i - 2 if i > 1 else 0
                shortcut_input = shortcut_outputs[shortcut_idx]
                
               
                if self.shortcuts[i-1] is not None:
                    shortcut_output = self.shortcuts[i-1](shortcut_input)
                else:
                    shortcut_output = shortcut_input
                
           
                block_output = block(x)
                
            
                if shortcut_output.shape == block_output.shape:
                    x = block_output + shortcut_output
                else:
                  
                    x = block_output
            
            outputs.append(x)
            shortcut_outputs.append(x)
        
        return x, outputs

# Lớp phụ trợ để đổi thứ tự chiều dễ dàng hơn
class PermuteLayer(nn.Module):
    def __init__(self, *dims):
        super(PermuteLayer, self).__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims)

if __name__ == "__main__":
    batch_size = 32
    T = 180
    initial_dim = 512  # Kích thước đầu vào ban đầu
    
    # Danh sách kích thước đầu ra của từng block
    residual_blocks = [512, 512, 1024, 1024]
    
    model = ResidualNetwork(residual_blocks)
    input_tensor = torch.randn(batch_size, T, initial_dim)
    
    # Lấy cả output cuối cùng và list các output trung gian
    final_output, intermediate_outputs = model(input_tensor)
    
    # Lưu mô hình
    torch.save(model.state_dict(), 'ResidualNetwork.pth')
    
    # In thông tin về kích thước
    print(f"Input shape: {input_tensor.shape}")
    print(f"Final output shape: {final_output.shape}")
    print("Intermediate output shapes:")
    for i, output in enumerate(intermediate_outputs):
        print(f"  Block {i+1}: {output.shape}")
    
    # Export sang ONNX
    model.eval()
    
    # Định nghĩa tên input và output cho mô hình ONNX
    input_names = ["input"]
    output_names = ["final_output"] + [f"intermediate_output_{i}" for i in range(len(residual_blocks))]
    
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'sequence_length'},
        'final_output': {0: 'batch_size', 1: 'sequence_length'}
    }
    
    for i in range(len(residual_blocks)):
        dynamic_axes[f"intermediate_output_{i}"] = {0: 'batch_size', 1: 'sequence_length'}
    
    torch.onnx.export(
        model,
        input_tensor,
        "ResidualNetwork.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"Model exported to ONNX format at ResidualNetwork.onnx")