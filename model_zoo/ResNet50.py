import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count
FVCORE_AVAILABLE = True

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, padding=[0, 1, 0], first=False) -> None:
        super(Bottleneck1D, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=padding[0], bias=False), # First 1x1: stride=1
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding[1], bias=False), # 3x3 conv: apply stride here
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=padding[2], bias=False), # Last 1x1: stride=1
            nn.BatchNorm1d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)

        # Shortcut part
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False), # Apply stride here for shortcut
                nn.BatchNorm1d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x) # Dimensions must match here
        out = self.relu(out)
        return out

# In a network using BN, the output of the convolutional layer is not biased
class ResNet50(nn.Module):
    def __init__(self, Bottleneck1D, num_classes=10, input_channels=2) -> None:
        super(ResNet50, self).__init__()
        self.in_channels = 64
        # The first layer is as a separate because there is no residual block
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        # conv2
        self.conv2 = self._make_layer(Bottleneck1D, 64, 3, stride=1)
        # conv3
        self.conv3 = self._make_layer(Bottleneck1D, 128, 4, stride=2)
        # conv4
        self.conv4 = self._make_layer(Bottleneck1D, 256, 6, stride=2)
        # conv5
        self.conv5 = self._make_layer(Bottleneck1D, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1) # Output size: (B, C, 1)
        self.fc = nn.Linear(512 * Bottleneck1D.expansion, num_classes) # 512 * 4 = 2048

    def _make_layer(self, Bottleneck1D, out_channels, num_blocks, stride):
        layers = []
        # Determine whether it's the first layer of each block layer that needs downsampling
        first_flag = (stride != 1) or (self.in_channels != out_channels * Bottleneck1D.expansion)
        layers.append(Bottleneck1D(self.in_channels, out_channels, stride, [0, 1, 0], first=first_flag))

        # Update in_channels after the first block
        self.in_channels = out_channels * Bottleneck1D.expansion

        # The remaining bottleneck stride is obtained from strides[1:]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck1D(self.in_channels, out_channels, 1, [0, 1, 0])) # stride=1, first=False implicitly

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3 and x.size(2) == 2:
            # Assume input is (B, L, C=2), transpose to (B, C=2, L)
            x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _calculate_flops(self, input_shape=(1, 2048, 2)):
        """尝试使用 fvcore 来计算模型的 MACs。"""
        flops_fvcore, params_fvcore = None, None

        if FVCORE_AVAILABLE:
            # Make sure the input is on the correct device, which may be important for some models
            device = next(self.parameters()).device # Obtain the device on which the model is located
            dummy_input_fvcore = torch.randn(*input_shape).to(device)
            try:
                self.eval()
                # FlopCountAnalysis will automatically process the model and input
                flops_fvcore_obj = FlopCountAnalysis(self, dummy_input_fvcore)
                flops_fvcore = flops_fvcore_obj.total() # Get the total MACs
                params_fvcore_dict = parameter_count(self) # Get param dict {'': total_params, 'module_name': sub_params, ...}
                params_fvcore = params_fvcore_dict.get('', 0) # Obtain the total number of parameters
                self.train()
                print(f"[Model Stats] Successfully calculated MACs/Params with fvcore.")
                return flops_fvcore, params_fvcore # If fvcore is successful, return its result
            except Exception as e:
                print(f"[Model Stats] Error calculating MACs with fvcore: {e}")
                import traceback
                traceback.print_exc()
                self.train()

        print("[Model Stats] Fvcore MACs calculations failed or was skipped.")
        return None, None

    def print_model_stats(self, input_shape=(1, 2048, 2)):
        """Print model statistics including MACs and number of parameters."""
        print("[Model Stats] --- Calculating Model Statistics ---")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model Stats] Total parameters: {total_params:,}")
        print(f"[Model Stats] Trainable parameters: {trainable_params:,}")

        macs, params_calc = self._calculate_flops(input_shape=input_shape)

        if macs is not None:
            # fvcore returns MACs
            flops_value = macs * 2 # Turn it into FLOPs
            print(f"[Model Stats] MACs (for input {input_shape}, via fvcore): {macs:,} MACs")
            print(f"[Model Stats] MACs: {macs / 1e9:.2f} GMACs")
            print(f"[Model Stats] Estimated FLOPs (MACs * 2): {flops_value:,} FLOPs")
            print(f"[Model Stats] Estimated FLOPs: {flops_value / 1e9:.2f} GFLOPs")

            # Print parameters
            if params_calc is not None:
               print(f"[Model Stats] Parameters (from fvcore): {params_calc:,}")
               if abs(params_calc - total_params) > 1000:
                   print(f"[Model Stats] Warning: Parameter count mismatch! Manual: {total_params:,}, fvcore: {params_calc:,}")
        else:
            print("[Model Stats] Automatic MACs calculation failed.")

        print("[Model Stats] --- End of Model Statistics ---")

def resnet50_1d(num_classes=1000, input_channels=2): # Default classes
    """Constructs a ResNet-50 model for 1D data."""
    model = ResNet50(Bottleneck1D, num_classes=num_classes, input_channels=input_channels)
    return model


# --- Test ---
if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- ResNet50 1D Model Test ---")
    print(f"[Test] Target test device: {DEVICE}")

    print(f"[Test] 1. Creating model instance...")
    res50 = resnet50_1d(num_classes=10, input_channels=2) # Create model with 10 classes
    res50.to(torch.device(DEVICE)) # Move model to device

    # Print model statistics
    res50.print_model_stats(input_shape=(1, 2048, 2)) # Use the typical input shape for stats

    print(f"\n[Test] Model successfully created and moved to: {DEVICE}")

    print(f"[Test] 2. Creating sample input data on CPU...")
    dummy_input = torch.randn(4, 2048, 2) # Batch=4, Length=2048, Channels=2
    print(f"[Test] Sample input created on: {dummy_input.device}")
    print(f"[Test] Sample input shape: {dummy_input.shape}")

    print(f"[Test] 3. Running forward pass...")
    res50.eval()
    with torch.no_grad():
        try:
            output = res50(dummy_input.to(DEVICE)) # Move input to device
            print(f"[Test] Forward pass successful!")
            print(f"[Test] Model output shape: {output.shape}") # Should be (Batch, num_classes)
            print(f"[Test] Model output device: {output.device}")
            expected_shape = (4, 10) # Batch=4, Classes=10
            assert output.shape == torch.Size(expected_shape), f"Output shape {output.shape} does not match expected {expected_shape}"
            print(f"[Test] Expected output shape: {expected_shape}")
            print(f"[Test] --- All checks passed! Model is ready for training/testing. ---")
        except Exception as e:
            print(f"[Test] Error during forward pass: {e}")
            import traceback
            traceback.print_exc()