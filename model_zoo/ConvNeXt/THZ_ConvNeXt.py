from .ConvNeXt import ConvNeXt
from .Preprocessing import Preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as f

from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, flop_count, parameter_count

class THZ_ConvNeXt(nn.Module):
    def __init__(self, num_classes, input_dim = 2, patch_size=16, seq_len = 2048):
        """
        input_dim:
            for [2048, 2], input_dim=2
        """
        super().__init__()
        self.prep = Preprocessing(patch_size, input_dim)
        #########################
        #''' BUF TO BE FIXED '''#
        #########################
        if seq_len % patch_size != 0:
            self.convnext = ConvNeXt(in_chans=(seq_len//patch_size + 1), num_classes=num_classes)
        else:
            self.convnext = ConvNeXt(in_chans=int(seq_len/patch_size), num_classes=num_classes)

    def forward(self, x):
        x = self.prep(x)
        out = self.convnext(x)
        
        return out
    

if __name__ == "__main__":
    x = torch.ones((1,2048,2))
    model = THZ_ConvNeXt(num_classes=16, input_dim=2, seq_len=2048)
    print("-----------------thop-------------------")
    macs, params = profile(model=model, inputs=(x,)) # type: ignore
    macs_readable, params_readable = clever_format([macs, params], "%.3f")
    print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")

    print("----------------fvcore-----------------")

    params =  parameter_count(model=model)
    flops = FlopCountAnalysis(model=model,inputs=(x,))

    macs_readable, params_readable = clever_format([flops.total(), params[""]], "%.3f")
    print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
