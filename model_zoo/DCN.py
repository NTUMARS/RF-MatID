import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexLinear, ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu

# 自定义 1D 复数卷积层（因 complexPyTorch 未提供 ComplexConv1d）
class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # 实部和虚部分别定义独立的卷积核
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        real = self.conv_real(x.real) - self.conv_imag(x.imag)
        imag = self.conv_real(x.imag) + self.conv_imag(x.real)
        return torch.complex(real, imag)

class modReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, z):
        magnitude = torch.abs(z)
        phase = z / (magnitude + 1e-8)
        return torch.relu(magnitude + self.b) * phase

# 1D 深度复数网络
class DeepComplexNet1D(nn.Module):
    def __init__(self, input_length=1024, num_classes=2):
        super().__init__()
        self.conv1 = ComplexConv1d(1, 64, kernel_size=7, stride=3)
        self.act1 = modReLU()
        
        self.conv2 = ComplexConv1d(64, 128, kernel_size=7, stride=3)
        self.act2 = modReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        self.fc_input_dim = int(128 * ((((((input_length-7) // 3) + 1) -7) // 3) + 1))
        
        self.fc1 = ComplexLinear(self.fc_input_dim, 256)
        self.act3 = modReLU()
        self.fc2 = ComplexLinear(256, num_classes)
    
    def forward(self, x):
        # 输入形状: [batch, 1024] -> 转换为 [batch, 1, 1024]
        x = x.unsqueeze(1)  # 增加通道维度
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        # batch, 64, 340
        # x = x.permute(0, 2, 1)
        # x = self.bn1(x)
        # x = x.permute(0, 2, 1)
        x = self.act1(x)
        # x = self.pool1(x.abs())
        
        x = self.conv2(x)
        # batch, 128, 112
        # x = x.permute(0, 2, 1)
        # x = self.bn2(x)
        # x = x.permute(0, 2, 1)
        x = self.act2(x)
        # x = self.pool2(x.abs())
        
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        
        # 输出取模（用于分类）
        return torch.abs(x)

# 生成随机复数数据
# def generate_complex_data(batch_size, length=1024):
#     real = torch.randn(batch_size, length)
#     imag = torch.randn(batch_size, length)
#     return torch.complex(real, imag)

# # 训练示例
# if __name__ == "__main__":
#     # 参数
#     batch_size = 32
#     input_length = 1024
#     num_classes = 2
#     num_epochs = 10
    
#     # 模型、损失函数、优化器
#     model = DeepComplexNet1D(input_length, num_classes)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
#     # 生成训练数据
#     x_train = generate_complex_data(batch_size, input_length)
#     y_train = torch.randint(0, num_classes, (batch_size,))
    
#     # 训练循环
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
        
#         outputs = model(x_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()
        
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

#     # 测试
#     model.eval()
#     x_test = generate_complex_data(5, input_length)
#     with torch.no_grad():
#         predictions = model(x_test)
#         print("Predictions:", predictions.argmax(dim=1))