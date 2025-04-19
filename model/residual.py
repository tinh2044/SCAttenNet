import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, d_model, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        
        self.linear1 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if self.downsample:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        residual = x  
        
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
    def __init__(self, d_model):
        super(ResidualNetwork, self).__init__()
        self.block1 = ResidualBlock(d_model, downsample=False)  
        self.block2 = ResidualBlock(d_model, downsample=True)   
        self.block3 = ResidualBlock(d_model, downsample=False)  
        self.block4 = ResidualBlock(d_model, downsample=True)   
        
    def forward(self, x):
        x = self.block1(x)  
        x = self.block2(x)  
        x = self.block3(x)  
        x = self.block4(x)  
        return x

if __name__ == "__main__":
    batch_size = 32
    T = 180
    d_model = 512
    
    model = ResidualNetwork(d_model)
    input_tensor = torch.randn(batch_size, T, d_model)
    output = model(input_tensor)
    torch.save(model.state_dict(), 'ResidualNetwork.pth')
    
    # Export to ONNX
    model.eval()  # Set the model to evaluation mode
    
    # Define the input names and output names for the ONNX model
    input_names = ["input"]
    output_names = ["output"]
    
    # Export the model to ONNX format
    torch.onnx.export(
        model,                     # model being run
        input_tensor,              # model input (or a tuple for multiple inputs)
        "ResidualNetwork.onnx",    # where to save the model
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=12,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=input_names,   # the model's input names
        output_names=output_names, # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},  # variable length axes
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    print(f"Model exported to ONNX format at ResidualNetwork.onnx")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}") 