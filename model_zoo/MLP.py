import torch.nn as nn
import torch.nn.functional as f
import torch

from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, flop_count, parameter_count

class MLP(nn.Module):
    def __init__(self, num_classes, input_dim=4096, expansion=2):
        """ 
        input_dim: 
                for data like [2048, 2], input_dim = 2048*2 = 4096
        """
        super().__init__()
        self.input_dim = input_dim
        hidden_dim = input_dim * expansion
        # TODO: for flexible input_dim: 1. lazy init?   2. padding?   3. masking?
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
            )
        self.classifier = nn.Linear(input_dim, num_classes, bias=False)
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc(x)
        out = self.classifier(x)
        return out

if __name__ == "__main__":
    x = torch.ones((1,2048,2))
    model = MLP(num_classes=16,input_dim=4096)
    print("-----------------thop-------------------")
    macs, params = profile(model=model, inputs=(x,)) # type: ignore
    macs_readable, params_readable = clever_format([macs, params], "%.3f")
    print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")

    print("----------------fvcore-----------------")

    params =  parameter_count(model=model)
    flops = FlopCountAnalysis(model=model,inputs=(x,))

    macs_readable, params_readable = clever_format([flops.total(), params[""]], "%.3f")
    print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")

    
