import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: ptflops for FLOPs/params
try:
    from ptflops import get_model_complexity_info
    _HAS_PTFLOPS = True
except ImportError:
    get_model_complexity_info = None
    _HAS_PTFLOPS = False


class ConvBNAct1D(nn.Module):
    """
    Conv1d + BatchNorm1d + Activation + Dropout
    Basic modules for MRF encoder
    """
    def __init__(self, in_ch, out_ch, k, s=1, p=None, dropout=0.0, act_layer=nn.ReLU):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = act_layer()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MRFModule1D(nn.Module):
    """
    1D Material-Related Feature module (MRF-module)
    Corresponding to the roughness removal autoencoder structure in airTac, but here we are only responsible for
    the autoencoder reconstruction of x -> z' is provided for subsequent classification
    """

    def __init__(self, input_channels: int, base_channels: int = 16, output_channels: int = None):
        super().__init__()
        if output_channels is None:
            output_channels = input_channels

        # Encoder: four layers 1D Conv，kernel=4, stride=2, padding=1
        # Channels: in -> B -> 2B -> 4B -> 8B
        enc_c = [input_channels, base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoder = nn.Sequential(
            ConvBNAct1D(enc_c[0], enc_c[1], k=4, s=2, p=1, dropout=0.1, act_layer=nn.ReLU),
            ConvBNAct1D(enc_c[1], enc_c[2], k=4, s=2, p=1, dropout=0.1, act_layer=nn.ReLU),
            ConvBNAct1D(enc_c[2], enc_c[3], k=4, s=2, p=1, dropout=0.1, act_layer=nn.ReLU),
            ConvBNAct1D(enc_c[3], enc_c[4], k=4, s=2, p=1, dropout=0.1, act_layer=nn.ReLU),
        )

        # Decoder: symmetrical ConvTranspose1d，last layer is Tanh
        dec_c = [enc_c[4], enc_c[3], enc_c[2], enc_c[1], output_channels]
        self.deconv1 = nn.ConvTranspose1d(dec_c[0], dec_c[1], kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose1d(dec_c[1], dec_c[2], kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose1d(dec_c[2], dec_c[3], kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose1d(dec_c[3], dec_c[4], kernel_size=4, stride=2, padding=1, bias=True)

        self.bn1 = nn.BatchNorm1d(dec_c[1])
        self.bn2 = nn.BatchNorm1d(dec_c[2])
        self.bn3 = nn.BatchNorm1d(dec_c[3])
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x: (B, L, C_in)
        return: z_prime (B, L, C_out) - Reconstructed material-related FDMF
        """
        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1).contiguous()
        z = self.encoder(x)
        z = self.deconv1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.deconv2(z)
        z = self.bn2(z)
        z = self.relu(z)
        z = self.deconv3(z)
        z = self.bn3(z)
        z = self.relu(z)
        z = self.deconv4(z)
        z = self.tanh(z)
        # (B, C_out, L) -> (B, L, C_out)
        z = z.permute(0, 2, 1).contiguous()
        return z


class MCModule1D(nn.Module):
    """
    MC-module: Two-layer fully connected material classifier.
                Enter z' to flatten and classify.
    """

    def __init__(self, seq_len: int, input_channels: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.fc1 = nn.Linear(seq_len * input_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, z_prime):
        """
        z_prime: (B, L, C) -> flatten and classify
        """
        b, l, c = z_prime.shape
        assert l == self.seq_len, f"Expected seq_len={self.seq_len}, got {l}"
        assert c == self.input_channels, f"Expected channels={self.input_channels}, got {c}"
        x = z_prime.reshape(b, l * c)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class AirTacMNet1D(nn.Module):
    """
    AirTac network with only M-Net
    - MRFModule1D: Feature purification in the form of an autoencoder
    - MCModule1D : material classifier
    """

    def __init__(
        self,
        input_channels: int,
        seq_len: int,
        num_classes: int,
        base_channels_mrf: int = 16,
        hidden_dim_mc: int = 256,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels

        self.mrf = MRFModule1D(
            input_channels=input_channels,
            base_channels=base_channels_mrf,
            output_channels=input_channels,
        )
        self.mc = MCModule1D(
            seq_len=seq_len,
            input_channels=input_channels,
            hidden_dim=hidden_dim_mc,
            num_classes=num_classes,
        )

    def forward(self, x):
        """
        Standard forward: Material classification
        x: (B, L, C)
        return: logits (B, num_classes)
        """
        z_prime = self.mrf(x)          # (B, L, C)
        logits_mat = self.mc(z_prime)  # (B, num_classes)
        return logits_mat


def compute_flops_p1(model, seq_len=2048, channels=2, device="cpu"):
    """
    Calculate the forward FLOPs (P1) and the number of parameters using ptflops
    P1: 1 MAC = 2 FLOPs
    """
    if not _HAS_PTFLOPS:
        print("[FLOPs] ptflops not installed，skip FLOPs calculation")
        print("Install: pip install ptflops")
        return

    model = model.to(device)
    model.eval()

    def input_constructor(_):
        return torch.randn(1, seq_len, channels, device=device)

    try:
        macs, params = get_model_complexity_info(
            model,
            (seq_len, channels),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        flops = macs * 2.0
        print(f"[FLOPs] P1 (M-Net forward): {flops / 1e6:.3f} M")
        print(f"[FLOPs] Params            : {params / 1e6:.3f} M")
    except Exception as e:
        print(f"[FLOPs] Failed to compute FLOPs via ptflops: {e}")


# Test
if __name__ == "__main__":
    print("--- AirTacMNet1D Model Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Target device: {device}")

    batch_size = 4
    seq_len = 2048
    channels = 2
    num_classes = 16

    try:
        print("[Test] 1. Creating model instance...")
        model = AirTacMNet1D(
            input_channels=channels,
            seq_len=seq_len,
            num_classes=num_classes,
            base_channels_mrf=16,
            hidden_dim_mc=256,
        ).to(device)
        print(f"[Test] Model created on: {next(model.parameters()).device}")

        print("[Test] 1.1 Computing FLOPs/Params...")
        compute_flops_p1(model, seq_len=seq_len, channels=channels, device=device)

        print("[Test] 2. Creating dummy input...")
        sample = torch.randn(batch_size, seq_len, channels)
        print(f"[Test] Sample shape: {sample.shape}")

        print("[Test] 3. Running forward...")
        with torch.no_grad():
            out = model(sample.to(device))
        print(f"[Test] Output logits shape: {out.shape}  (expected: [{batch_size}, {num_classes}])")

        print("[Test] --- All checks passed. ---")

    except Exception as e:
        print(f"[Test] Error occurred: {e}")
        import traceback
        traceback.print_exc()