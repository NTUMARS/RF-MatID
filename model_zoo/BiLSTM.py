import torch
import torch.nn as nn
import torch.nn.functional as f

from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, flop_count, parameter_count

class BiLSTM(nn.Module):
    def __init__(self, num_classes, input_dim=2, hidden_size=128, num_layers=2):
        """
        input_dim:
            for data like [2048, 2], input_dim = 2
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes, bias=False)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x) # hn @(D, B, H)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.classifier(hn)
        return out
    

if __name__ == "__main__":
    x = torch.ones((1,2048,2))
    model = BiLSTM(num_classes=16, input_dim=2)
    print("-----------------thop-------------------")
    macs, params = profile(model=model, inputs=(x,)) # type: ignore
    macs_readable, params_readable = clever_format([macs, params], "%.3f")
    print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")

    print("----------------fvcore-----------------")

    params =  parameter_count(model=model)
    flops = FlopCountAnalysis(model=model,inputs=(x,))

    macs_readable, params_readable = clever_format([flops.total(), params[""]], "%.3f")
    print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
