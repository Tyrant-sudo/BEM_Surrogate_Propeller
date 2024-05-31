import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

import pandas as pd

class My_Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
    
class Model_T0(nn.Module):
    def __init__(self):
        super(Model_T0, self).__init__()
        self.fc1 = nn.Linear(26*2, 128)  # 输入层到隐藏层1，将输入向量展平
        self.fc2 = nn.Linear(128, 256)    # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(256, 64)     # 隐藏层2到隐藏层3
        self.fc4 = nn.Linear(64, 2)      # 隐藏层3到输出层
        # 初始化权重
        # self._initialize_weights()

    def forward(self, x):
        # 假设输入 x 的形状为 [batch_size, 26, 2]
        x = x.view(-1, 26*2)  # 将输入展平为 [batch_size, 52]
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        
        x = self.fc4(x)  # 输出层不使用激活函数，直接返回预测值
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier 初始化
                init.xavier_uniform_(m.weight)
                # 或者使用 Kaiming 初始化
                # init.kaiming_uniform_(m.weight, nonlinearity='relu')
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class Model_T1(nn.Module):
    def __init__(self):
        super(Model_T1, self).__init__()
        self.fc1 = nn.Linear(26*2, 256)  # 输入层到隐藏层1，将输入向量展平
        self.fc2 = nn.Linear(256, 1024)    # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(1024, 256)     # 隐藏层2到隐藏层3
        self.fc33 = nn.Linear(256,64)
        self.fc4 = nn.Linear(64, 2)      # 隐藏层3到输出层
        # 初始化权重
        # self._initialize_weights()

    def forward(self, x):
        # 假设输入 x 的形状为 [batch_size, 26, 2]
        x = x.view(-1, 26*2)  # 将输入展平为 [batch_size, 52]
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))

        x = F.relu(self.fc33(x))
        
        x = self.fc4(x)  # 输出层不使用激活函数，直接返回预测值
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier 初始化
                init.xavier_uniform_(m.weight)
                # 或者使用 Kaiming 初始化
                # init.kaiming_uniform_(m.weight, nonlinearity='relu')
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class Model_N0(nn.Module):
    def __init__(self):
        super(Model_N0, self).__init__()
        self.fc1 = nn.Linear(26*2, 128)  # 输入层到隐藏层1，将输入向量展平
        self.fc2 = nn.Linear(128, 256)    # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(256, 128)     # 隐藏层2到隐藏层3
        self.fc4 = nn.Linear(128, 72)      # 隐藏层3到输出层
        # 初始化权重
        # self._initialize_weights()

    def forward(self, x):
        # 假设输入 x 的形状为 [batch_size, 26, 2]
        x = x.view(-1, 26*2)  # 将输入展平为 [batch_size, 52]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        
        x = self.fc4(x)  # 输出层不使用激活函数，直接返回预测值
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier 初始化
                init.xavier_uniform_(m.weight)
                # 或者使用 Kaiming 初始化
                # init.kaiming_uniform_(m.weight, nonlinearity='relu')
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class Model_TN0(nn.Module):
    def __init__(self):
        super(Model_TN0, self).__init__()
        # 编码器部分
        self.enc_conv1 = nn.Conv1d(2, 4, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        # 解码器部分
        self.dec_conv1 = nn.ConvTranspose1d(8, 4, kernel_size=3, stride=1, padding=1)
        # 跳跃连接后的处理
        self.post_cat_conv = nn.Conv1d(8, 2, kernel_size=3, padding=1)
        # 最终的线性层以得到期望的输出维度
        self.final_fc = nn.Linear(26 * 2, 72)

    def forward(self, x):
        # 编码器路径
        enc1 = F.relu(self.enc_conv1(x))
        enc2 = F.relu(self.enc_conv2(enc1))

        # 解码器路径
        dec1 = F.relu(self.dec_conv1(enc2))

        # 跳跃连接
        dec1 = torch.cat((dec1, enc1), 1)  # 合并编码器和解码器层的特征
        dec1 = F.relu(self.post_cat_conv(dec1))

        # 扁平化并通过最终线性层
        dec1 = dec1.view(dec1.size(0), -1)
        out = self.final_fc(dec1)
        return out

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.enc_conv1 = nn.Conv1d(26, 10, kernel_size=2, stride=1, padding=1)
        self.enc_conv2 = nn.Conv1d(10, 5, kernel_size=2, stride=1, padding=1)
        self.dec_conv1 = nn.Conv1d(5, 10, kernel_size=2, stride=1, padding=1)
        self.fc = nn.Linear(10 * 3, 72)

    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x1 = F.relu(self.enc_conv2(x))
        x = F.relu(self.dec_conv1(x1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x1, x

# 实例化模型
model = CustomNetwork()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 假设我们使用MSE损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设有一些训练数据
# 输入数据
input_tensor = torch.randn(10, 26, 2)  # 输入维度为(B, 26, 2)

# 目标数据
target_mid = torch.randn(10, 5)  # 中间层目标
target_final = torch.randn(10, 72)  # 最终输出目标

# 训练循环
model.train()
optimizer.zero_grad()

# 前向传播
mid_output, final_output = model(input_tensor)

# 计算损失
loss_mid = criterion(mid_output, target_mid)
loss_final = criterion(final_output, target_final)
total_loss = loss_mid + loss_final  # 可以根据需要为不同的损失分配权重

# 反向传播和优化
total_loss.backward()
optimizer.step()

print(f"Total Loss: {total_loss.item()}")

"""

if __name__ == "__main__":

    a = torch.zeros((512, 26, 2))
    b = torch.zeros((512, 2))
    
    model = Model_T0()
    c = model(a)
    print(c.shape)
