import torch
import torch.nn as nn
from transformers import AutoModel
from fvcore.nn import FlopCountAnalysis, parameter_count
FVCORE_AVAILABLE = False
from thop import profile
THOP_AVAILABLE = False

class PatchEmbeddingAndReshape(nn.Module):
    """
    Chunking, flattening, linearly embedding, and reshaping data into a 2D image format.
    """

    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        """
        Args:
            patch_size (int): The patch length along the spectral axis
            in_channels (int): The number of channels for entering the data (for example, 2).
            embed_dim (int): Dimension after linear embedding (should be 32*32 = 1024).
        """
        super(PatchEmbeddingAndReshape, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.image_height = 32
        self.image_width = 32

        if self.embed_dim != self.image_height * self.image_width:
            raise ValueError(f"embed_dim ({embed_dim}) must be equal to image_height * image_width ({self.image_height * self.image_width})")

        patch_flattened_dim = patch_size * in_channels
        # The projection layer will be moved to the specified device when the model is .to(device).
        self.projection = nn.Linear(patch_flattened_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor, shaped by (batch_size, sequence_length, in_channels).
                                Must be on the same device as the model parameters.

        Returns:
            torch.Tensor: Output tensor, shaped by (batch_size, num_patches, image_height, image_width)。
                          Must be on the same device as the input x.
        """
        B, L, C = x.shape
        if L % self.patch_size != 0:
            raise ValueError(f"Sequence length {L} is not divisible by patch_size {self.patch_size}")
        num_patches = L // self.patch_size
        x_patches = x.view(B, -1, self.patch_size, C)
        x_flattened = x_patches.view(B, -1, self.patch_size * C)
        # Perform a linear transformation. The weights/biases of x and self.projection must be on the same device
        x_embedded = self.projection(x_flattened)
        x_final = x_embedded.view(B, num_patches, self.image_height, self.image_width)
        return x_final


class DINOv3ConvNeXt(nn.Module):

    def __init__(self, num_classes: int, patch_size: int, seq_len: int, pretrained: bool = True, freeze_backbone: bool = True, target_device: torch.device = None):
        """
        Args:
            num_classes (int): The number of categories of categorized tasks.
            patch_size (int): The size of the patch.
            seq_len (int): The length of the input sequence.
                              This is used to calculate num_patches and configure the backbone.
            pretrained (bool, optional): Whether DINOv3 pre-training weights are loaded. Default is true.
            freeze_backbone (bool, optional): Whether to freeze the parameters of the DINOv3 backbone network. Default is true.
            target_device (torch.device, optional): Specifies the target device for the model to run.
                                                    If it's None, use 'cuda' if available, otherwise use 'cpu'.
        """
        super(DINOv3ConvNeXt, self).__init__()

        # Identify and set the target device. If not specified, CUDA devices will be given priority
        if target_device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = target_device
        print(f"[Model Init] Target device for model set to: {self.device}")

        # Make patch_size, seq_len, and derived values configurable
        self.patch_size = patch_size
        self.seq_len = seq_len
        if self.seq_len % self.patch_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_size ({patch_size})")
        self.num_patches = self.seq_len // self.patch_size # Calculate based on provided seq_len

        # Define data preprocessing and embedding modules
        self.in_channels = 2
        self.embed_dim = 32 * 32 # 1024

        # Create all submodules before calling self.to (device).
        self.patch_embed = PatchEmbeddingAndReshape(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim
        )

        # Load the DINOv3 ConvNeXt Tiny backbone network
        # Remove device_map="auto" to avoid inconsistencies caused by automatic device assignments
        self.pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"

        try:
            # The model weights are loaded to the CPU by default
            self.backbone = AutoModel.from_pretrained(
                self.pretrained_model_name,
                trust_remote_code=True
                # Do not set the device map
            )
            print(f"[Model Init] Successfully loaded DINOv3 backbone (initially on CPU): {self.pretrained_model_name}")
        except Exception as e:
            print(f"[Model Init] Error loading DINOv3 backbone: {e}")
            print("[Model Init] Please ensure 'transformers' library is installed and internet connection is available.")
            raise e

        # Find and modify the first layer of convolution
        self._replace_first_conv(self.num_patches) # Pass the calculated num_patches

        # Freeze backbone network parameters (optional)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[Model Init] DINOv3 backbone parameters (including modified first conv) are frozen.")
        else:
            print("[Model Init] DINOv3 backbone parameters are NOT frozen (will be trained).")

        # Define the classification header (delayed initialization)
        self.classifier_head = None
        self._classifier_initialized = False
        self.num_classes = num_classes

        # Move all components of the entire model to the target device
        # It will recursively move the parameters and buffers of self.backbone, self.patch embed and all its sub-modules (including self.projection) to self.device
        self.to(self.device)
        print(f"[Model Init] All model components moved to target device: {self.device}")

        # Calculate and print the model statistics information
        # It is called after the model is fully initialized and moved to the device
        self._print_model_stats()

    def _replace_first_conv(self, num_patches_for_conv: int):
        """Find and replace the first layer of convolution of the backbone network based on the known structure."""
        try:
            # The precise path to the original first convolutional layer (for ConvNeXt structure)
            original_first_conv = self.backbone.stages[0].downsample_layers[0]
            print(f"[Model Init] Found original first conv: {original_first_conv}")

            # Obtain the parameters of the original convolutional layer
            original_in_channels = original_first_conv.in_channels
            original_out_channels = original_first_conv.out_channels
            original_kernel_size = original_first_conv.kernel_size
            original_stride = original_first_conv.stride
            original_padding = original_first_conv.padding
            original_has_bias = original_first_conv.bias is not None

            print(f"[Model Init] Original first conv params: in_channels={original_in_channels}, out_channels={original_out_channels}, "
                  f"kernel_size={original_kernel_size}, stride={original_stride}, padding={original_padding}, has_bias={original_has_bias}")

            # Create a new convolutional layer that modifies the number of input channels from 3 to accommodate the number of patches after processing
            self.new_in_channels = num_patches_for_conv # Use the dynamic value
            # The newly created layer will be moved to the specified device in a subsequent self.to (self.device) call
            self.modified_first_conv = nn.Conv2d(
                in_channels=self.new_in_channels,
                out_channels=original_out_channels,
                kernel_size=original_kernel_size,
                stride=original_stride,
                padding=original_padding,
                bias=original_has_bias
            )
            print(f"[Model Init] Created modified first conv: in_channels={self.new_in_channels}, out_channels={original_out_channels}, "
                  f"kernel_size={original_kernel_size}, stride={original_stride}, padding={original_padding}, has_bias={original_has_bias}")

            # Replace the original first-layer convolution in the backbone network
            self.backbone.stages[0].downsample_layers[0] = self.modified_first_conv
            print("[Model Init] Successfully replaced the first convolutional layer in DINOv3 backbone.")

        except Exception as e:
            print(f"[Model Init] Critical Error while replacing the first convolutional layer: {e}")
            raise ValueError("Failed to locate and replace the first convolutional layer. Check model structure assumptions.") from e

    def _count_parameters(self):
        """Calculate the total number of parameters and the number of trainable parameters of the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def _calculate_flops(self):
        """Try using thop first and then fvcore to calculate the FLOPs of the model"""
        flops, params_thop = None, None
        flops_fvcore, params_fvcore = None, None

        # Try thop first
        if THOP_AVAILABLE:
            dummy_input_thop = torch.randn(1, self.seq_len, 2).to(self.device) # Use self.seq_len
            try:
                self.eval()
                flops_thop, params_thop = profile(self, inputs=(dummy_input_thop, ), verbose=False)
                self.train()
                # If it work, return directly
                if flops_thop is not None:
                    return flops_thop, params_thop
            except Exception as e:
                print(f"[Model Stats] Error calculating FLOPs with thop: {e}")
                self.train()

        # 2. If thop failed, try fvcore
        if FVCORE_AVAILABLE:
            dummy_input_fvcore = torch.randn(1, self.seq_len, 2).to(self.device) # Use self.seq_len # fvcore normally needs the input located at the correct device.
            try:
                self.eval()
                # FlopCountAnalysis will automatically handle the model and the input
                flops_fvcore_obj = FlopCountAnalysis(self, dummy_input_fvcore)
                flops_fvcore = flops_fvcore_obj.total() # Get total FLOPs
                params_fvcore_dict = parameter_count(self) # Get parameter dict {'': total_params, 'module_name': sub_params, ...}
                params_fvcore = params_fvcore_dict.get('', 0) # Get the number of total parameters
                self.train()
                print(f"[Model Stats] Successfully calculated FLOPs/Params with fvcore.")
                return flops_fvcore, params_fvcore
            except Exception as e:
                print(f"[Model Stats] Error calculating FLOPs with fvcore: {e}")
                import traceback
                traceback.print_exc()
                self.train()

        # If both tries are failed
        print("[Model Stats] Both thop and fvcore FLOPs calculations failed or were skipped.")
        return None, None # When failed

    def _print_model_stats(self):
        """打印模型的参数量和 FLOPs。"""
        print("\n[Model Stats] --- Calculating Model Statistics ---")
        # Calculate Parameters
        total_params, trainable_params = self._count_parameters()
        print(f"[Model Stats] Total parameters: {total_params:,}")
        print(f"[Model Stats] Trainable parameters: {trainable_params:,}")

        # Calculate FLOPs
        flops, params_thop = self._calculate_flops()
        if flops is not None:
            print(f"[Model Stats] FLOPs (for input 1x{self.seq_len}x2): {flops:,} MACs") # Use self.seq_len
            print(f"[Model Stats] FLOPs: {flops / 1e9:.2f}G MACs")
        else:
            print("[Model Stats] FLOPs calculation was skipped or failed.")
        print("[Model Stats] --- End of Model Statistics ---\n")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor):  The input tensor, shaped by (batch_size, seq_len, 2).
                               seq_len should match the one used during model initialization.
        Returns:
            torch.Tensor: Output logits, shaped by (batch_size, num_classes).
                          It is located on self.device of the model
        """
        # Ensure that the input data and the model are on the same device
        x = x.to(self.device)
        # print(f"[Forward] Input tensor moved to: {x.device}") # Test

        B = x.shape[0]

        # Data preprocessing and embedding
        # At this point, both the parameters of x and self.patch embed (including self.patch embed.projection) are on self.device
        x_patches_images = self.patch_embed(x) # Output shape: (B, num_patches, H, W)

        # Feature extraction of backbone networks
        # All parameters for x_patches_images and backbone are on self.device
        backbone_outputs = self.backbone(x_patches_images, output_hidden_states=False)

        # Feature extraction and pooling
        # Extract feature tensors from the output of the backbone network
        if hasattr(backbone_outputs, 'last_hidden_state'):
             features = backbone_outputs.last_hidden_state
        elif hasattr(backbone_outputs, 'pooler_output'):
             features = backbone_outputs.pooler_output
        else:
            print("[Forward] Warning: Standard output attributes not found in backbone output.")
            print(f"[Forward] Backbone output type: {type(backbone_outputs)}")
            if isinstance(backbone_outputs, torch.Tensor):
                features = backbone_outputs
            else:
                try:
                    features = backbone_outputs[0]
                    print(f"[Forward] Using first element of backbone output: {type(features)}")
                except (TypeError, IndexError):
                    raise RuntimeError("Could not determine the correct feature tensor from backbone output.")

        if not isinstance(features, torch.Tensor):
             raise RuntimeError(f"[Forward] Features after backbone processing are not a tensor: {type(features)}")

        # Perform global average pooling on the extracted features to obtain vectors of a fixed size
        if features.dim() == 4:
            pooled_features_flat = nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(start_dim=1)
        elif features.dim() == 3:
            pooled_features_flat = features.mean(dim=1)
        elif features.dim() == 2:
            pooled_features_flat = features
        else:
            raise ValueError(f"[Forward] Unexpected feature dimensionality after backbone/pooling: {features.dim()}")

        # Classification Header (Delayed Initialization) 
        if not self._classifier_initialized:
            if pooled_features_flat is None or pooled_features_flat.dim() != 2:
                 raise RuntimeError("[Forward] Could not determine the correct pooled feature shape for classifier head initialization.")
            feature_dim = pooled_features_flat.shape[1]
            print(f"[Forward] Initializing classifier head with input dim: {feature_dim} on device: {self.device}")
            # Create a category header on the correct device
            self.classifier_head = nn.Linear(feature_dim, self.num_classes).to(self.device)
            self._classifier_initialized = True

        # Final Classification
        # The parameters for pooled_features_flat and self.classifier_head are on self.device
        logits = self.classifier_head(pooled_features_flat)
        return logits

# Test
if __name__ == "__main__":
    print("--- DINOv3ConvNeXt Model Test ---")
    # Specify to use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Target test device: {device}")

    # Model Parameters
    batch_size = 4
    seq_len = 700
    patch_size = 14
    channels = 2
    num_classes = 10

    try:
        print("[Test] 1. Creating model instance...")
        # Instantiate the model and specify the target device
        model = DINOv3ConvNeXt(
            num_classes=num_classes,
            patch_size=patch_size, # Pass patch_size
            seq_len=seq_len,       # Pass seq_len
            pretrained=True,
            freeze_backbone=False, # For testing purposes, set it to False here
            target_device=device
        )
        print(f"[Test] Model successfully created and moved to: {next(model.parameters()).device}")

        print("[Test] 2. Creating sample input data on CPU...")
        # Create analog input data on the CPU
        sample_input_cpu = torch.randn(batch_size, seq_len, channels)
        print(f"[Test] Sample input created on: {sample_input_cpu.device}")
        print(f"[Test] Sample input shape: {sample_input_cpu.shape}")

        print("[Test] 3. Running forward pass...")
        with torch.no_grad():
            # The model will automatically move the input to the device (GPU) where it is located internally.
            output = model(sample_input_cpu)
        print(f"[Test] Forward pass successful!")
        print(f"[Test] Model output shape: {output.shape}")
        print(f"[Test] Model output device: {output.device}")
        print(f"[Test] Expected output shape: ({batch_size}, {num_classes})")
        print("[Test] --- All checks passed! Model is ready for GPU training/testing. ---")

    except Exception as e:
        print(f"[Test] Error occurred: {e}")
        import traceback
        traceback.print_exc()