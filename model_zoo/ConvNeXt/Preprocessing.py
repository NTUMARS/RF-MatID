
import torch
import torch.nn as nn
import torch.nn.functional as f
from .Embed import PositionalEmbedding

class Preprocessing(nn.Module):
    """
    Old Method:
        [B, 2048, 2] -split into patches-> [B, 128, 16, 2]
        -flatten each patch-> [B, 128, 32]
        -linear embedding-> [B, 128, 1024]
        - + PositionalEmbedding-> [B, 128, 32, 32]

    New Method:
        if L % patch != 0: 
            pad(0, size = L + (patch - (L % patch))) 

        [B, 2048, 2] -permute-> [B, 2, 2048]
        -Conv1D(kernel=patch_size, stride=patch_size, out_ch=1024)->
        [B, 1024, 128] -permute-> [B, 128, 1024] -+PE-> [B, 128, 32, 32]
    """
    def __init__(self, patch_size=16, num_channels=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels=num_channels,out_channels=1024,kernel_size=patch_size,stride=patch_size)
        self.posembed = PositionalEmbedding(1024)

    def forward(self, x):
        B, L, C = x.shape
        pad_len = (self.patch_size - (L % self.patch_size)) % self.patch_size
        pad_tensor = torch.zeros(B, pad_len, C, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad_tensor], dim=1)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        x += self.posembed(x)
        B, N, _ = x.shape
        x = x.view(B, N, 32, 32)
        return x
    
if __name__ == "__main__":
    pre = Preprocessing()
    x = torch.ones((8,700,2))
    print(pre(x).shape)
