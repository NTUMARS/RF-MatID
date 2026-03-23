import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block for 1D Data
class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

# ResNet Architecture for 1D Data
class ResNet1D(nn.Module):
    def __init__(self, in_channels=128, num_classes=4):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResNetBlock1D(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

# # Combined LSTM + ResNet Model
# class LSTMResNet(nn.Module):
#     def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, num_classes=4):
#         super(LSTMResNet, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first = True
#         )
#         self.resnet = ResNet1D(in_channels=hidden_dim, num_classes=num_classes)

#     def forward(self, x):
#         x = x.unsqueeze(-1) # (batch_size, seq_len) -> (batch_size, seq_len, input_dim=1)
#         #print(x.shape)
#         x, _ = self.lstm(x)  # Output shape: (batch_size, seq_len, hidden_dim)
#         x = x.permute(0, 2, 1) # x@(batch_size, hidden_dim, seq_len) hidden_dim become input_channels for ResNet
#         x = self.resnet(x)
#         return x

class LSTMResNetDualChannel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, num_classes=6):
        super(LSTMResNetDualChannel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,  
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.resnet = ResNet1D(in_channels=hidden_dim, num_classes=num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # (batch_size, seq_len, input_dim=2)
        x, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    batch_size = 32 
    seq_len = 1024  
    input_dim = 2 

    
    test_input = torch.randn(batch_size, seq_len, input_dim)
    model = LSTMResNetDualChannel(input_dim=input_dim)
    output = model(test_input)

    print(output.shape) # Expected output: torch.Size([batch_size, num_classes])