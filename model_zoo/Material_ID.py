"""
Material-ID 1D
- Input: (B, 2048, 2) Complex signal spectrum
- Network: 5 layers 1D CNN feature extractor + 3 layers 1D CNN classification head + FC
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
try:
    from ptflops import get_model_complexity_info
    _HAS_PTFLOPS = True
except ImportError:
    get_model_complexity_info = None
    _HAS_PTFLOPS = False

# ---------------------------------------------------------
# Config / utils
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Model: Feature Extractor + Classifier (S1 only)
class ConvBNReLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=None, dropout=0.2):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class MaterialID1D(nn.Module):
    """
    1D version Material-ID backbone:
    - Feature Extractor: 5-layer 1D CNN
    - Classifier: 3-layer 1D CNN + GlobalAvgPool + FC
    """

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32):
        super().__init__()
        # Feature extractor: 5 conv layers
        self.feat = nn.Sequential(
            ConvBNReLU1D(in_channels, base_channels, k=7, s=2, dropout=0.2),
            ConvBNReLU1D(base_channels, base_channels * 2, k=5, s=2, dropout=0.2),
            ConvBNReLU1D(base_channels * 2, base_channels * 4, k=5, s=2, dropout=0.2),
            ConvBNReLU1D(base_channels * 4, base_channels * 8, k=3, s=2, dropout=0.2),
            ConvBNReLU1D(base_channels * 8, base_channels * 8, k=3, s=2, dropout=0.2),
        )
        feat_channels = base_channels * 8

        # Material classifier: 3 conv + GAP + FC
        self.classifier_cnn = nn.Sequential(
            ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
            ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
            ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
        )
        self.fc = nn.Linear(feat_channels, num_classes)

    def forward(self, x):
        """
        x: (B, L, C_in)  ->  (B, C_in, L)
        """
        x = x.permute(0, 2, 1).contiguous()  # (B, 2, 2048)
        z = self.feat(x)                     # (B, C_f, L_f)
        z = self.classifier_cnn(z)           # (B, C_f, L_f)
        # Global Average Pooling over length
        z = z.mean(dim=-1)                   # (B, C_f)
        logits = self.fc(z)                  # (B, num_classes)
        return logits


# Model (S2/S3)
class MaterialID1DAdv(nn.Module):
    """
    Material-ID 1D + Continuous domain adversarial dual discriminator (distance/angle)
    - Feature extractor: 5 layers 1D CNN
    - Classification head: 3 layers 1D CNN + GAP + FC
    - Distance discriminator: 3 layers 1D CNN + GAP + FC -> [mu_r, logvar_r]
    - Angle discriminator: 3 layers 1D CNN + GAP + FC -> [mu_a, logvar_a]

    Usage Tips：
      - infer / normal classification training：model(x) -> logits
      - S2/S3 cross domain training：logits, mu_r, logvar_r, mu_a, logvar_a = model.forward_with_domains(x)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        use_distance_disc: bool = True,
        use_angle_disc: bool = True,
    ):
        super().__init__()
        self.use_distance_disc = use_distance_disc
        self.use_angle_disc = use_angle_disc

        # ================ feature extractor (5 layers Conv1d) ================
        self.feat = nn.Sequential(
            ConvBNReLU1D(in_channels, base_channels, k=7, s=2, dropout=0.2),
            ConvBNReLU1D(base_channels, base_channels * 2, k=5, s=2, dropout=0.2),
            ConvBNReLU1D(base_channels * 2, base_channels * 4, k=5, s=2, dropout=0.2),
            ConvBNReLU1D(base_channels * 4, base_channels * 8, k=3, s=2, dropout=0.2),
            ConvBNReLU1D(base_channels * 8, base_channels * 8, k=3, s=2, dropout=0.2),
        )
        feat_channels = base_channels * 8

        # ================ classification head (3 layers Conv1d + GAP + FC) ================
        self.classifier_cnn = nn.Sequential(
            ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
            ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
            ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
        )
        self.fc = nn.Linear(feat_channels, num_classes)

        # ================ distance discriminator CNN + FC ================
        if self.use_distance_disc:
            self.dist_disc_cnn = nn.Sequential(
                ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
                ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
                ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
            )
            # 输出 [mu_r, logvar_r]
            self.dist_fc = nn.Linear(feat_channels, 2)
        else:
            self.dist_disc_cnn = None
            self.dist_fc = None

        # ================ angle discriminator CNN + FC ================
        if self.use_angle_disc:
            self.angle_disc_cnn = nn.Sequential(
                ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
                ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
                ConvBNReLU1D(feat_channels, feat_channels, k=3, s=1, dropout=0.2),
            )
            # 输出 [mu_a, logvar_a]
            self.angle_fc = nn.Linear(feat_channels, 2)
        else:
            self.angle_disc_cnn = None
            self.angle_fc = None

    # ----------- Internal tool function: feature extraction + classification -----------

    def _extract_feat(self, x):
        """
        x: (B, L, C_in) = (B, 2048, 2)
        return: Z (B, C_f, L_f)
        """
        x = x.permute(0, 2, 1).contiguous()  # -> (B, C_in, L)
        z = self.feat(x)                     # (B, C_f, L_f)
        return z

    def _classify(self, z):
        """
        z: (B, C_f, L_f)
        return: logits (B, num_classes)
        """
        z_cls = self.classifier_cnn(z)       # (B, C_f, L_f)
        pooled = z_cls.mean(dim=-1)          # GAP -> (B, C_f)
        logits = self.fc(pooled)             # (B, num_classes)
        return logits

    # ----------- forward without discriminator output -----------

    def forward(self, x):
        """
        Standard forward, used for the classification loss part of S2 and S3
        x: (B, L, C)
        return: logits (B, num_classes)
        """
        z = self._extract_feat(x)
        logits = self._classify(z)
        return logits

    # ----------- forward with discriminator output -----------

    def forward_with_domains(self, x):
        """
        Used for S2/S3 spilts training：Return the classification output + discriminator parameters
        x: (B, L, C)
        return:
          logits        : (B, num_classes)
          dist_mu       : (B, 1) or None
          dist_logvar   : (B, 1) or None
          angle_mu      : (B, 1) or None
          angle_logvar  : (B, 1) or None
        """
        z = self._extract_feat(x)
        logits = self._classify(z)

        dist_mu = dist_logvar = None
        angle_mu = angle_logvar = None

        if self.use_distance_disc and self.dist_disc_cnn is not None:
            dr_feat = self.dist_disc_cnn(z)      # (B, C_f, L_f)
            dr_pool = dr_feat.mean(dim=-1)      # GAP -> (B, C_f)
            dr_out = self.dist_fc(dr_pool)      # (B, 2)
            dist_mu = dr_out[:, :1]             # (B, 1)
            dist_logvar = dr_out[:, 1:]         # (B, 1)

        if self.use_angle_disc and self.angle_disc_cnn is not None:
            da_feat = self.angle_disc_cnn(z)
            da_pool = da_feat.mean(dim=-1)
            da_out = self.angle_fc(da_pool)
            angle_mu = da_out[:, :1]
            angle_logvar = da_out[:, 1:]

        return logits, dist_mu, dist_logvar, angle_mu, angle_logvar

    # ----------- Gaussian negative log-likelihood -----------

    @staticmethod
    def gaussian_nll(mu, logvar, target):
        """
        L_d = ( (mu - U)^2 / (2 * sigma^2) ) + 0.5 * log(sigma^2)
        where logvar = log(sigma^2)
        """
        if mu is None or logvar is None or target is None:
            return None

        # Ensure the same shape (B, 1)
        target = target.view_as(mu)
        var = logvar.exp()  # sigma^2

        nll = (mu - target) ** 2 / (2.0 * var + 1e-8) + 0.5 * logvar
        return nll.mean()



# ---------------------------------------------------------
# FLOPs calculation (P1)
# ---------------------------------------------------------
def compute_flops_p1(model, seq_len=2048, channels=2, device="cpu"):
    """
    Calculate the forward FLOPs (P1) and parameter quantities
    - Use ptflops.get_model_complexity_info
    - The input shape according to the model: (B, L, C) = (1, seq_len, channels)
    - Unit: M (1e6)
    """
    if not _HAS_PTFLOPS:
        print("[FLOPs] ptflops not installed，skip FLOPs calculation.")
        print("Please install: pip install ptflops")
        return

    model = model.to(device)
    model.eval()

    def input_constructor(_):
        # dummy input，align with real forward: (1, L, C)
        return torch.randn(1, seq_len, channels, device=device)

    macs, params = get_model_complexity_info(
        model,
        (seq_len, channels),           # Just a shape hint, what's really used is input_constructor
        input_constructor=input_constructor,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    flops = macs * 2.0  # P1: 1 MAC = 2 FLOPs

    print(f"[FLOPs] P1: {flops / 1e6:.3f} M")
    print(f"[FLOPs] Params: {params / 1e6:.3f} M")



# ------------------ Test ------------------
if __name__ == "__main__":
    print("--- MaterialID1D Model Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Target test device: {device}")

    # Model Parameters
    batch_size = 4
    seq_len = 2048 
    channels = 2     # dual_channel
    num_classes = 10

    try:
        print("[Test] 1. Creating model instance...")
        model = MaterialID1D(
            in_channels=channels,
            base_channels=32,
            num_classes=num_classes
        ).to(device)
        print(f"[Test] Model successfully created and moved to: {next(model.parameters()).device}")

        print("[Test] 1.1 Calculating FLOPs/Params ...")
        compute_flops_p1(model, seq_len=seq_len, channels=channels, device=device)

        print("[Test] 2. Creating sample input data on CPU...")
        sample_input_cpu = torch.randn(batch_size, seq_len, channels)
        print(f"[Test] Sample input created on: {sample_input_cpu.device}")
        print(f"[Test] Sample input shape: {sample_input_cpu.shape}")

        print("[Test] 3. Running forward pass...")
        with torch.no_grad():
            sample_input = sample_input_cpu.to(device)
            output = model(sample_input)

        print("[Test] Forward pass successful!")
        print(f"[Test] Model output shape: {output.shape}")
        print(f"[Test] Model output device: {output.device}")
        print(f"[Test] Expected output shape: ({batch_size}, {num_classes})")
        print("[Test] --- All checks passed! MaterialID1D is ready for training/testing. ---")

    except Exception as e:
        print(f"[Test] Error occurred: {e}")
        import traceback
        traceback.print_exc()
