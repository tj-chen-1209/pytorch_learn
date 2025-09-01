import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 定义网络


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除 batch 外所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化网络和优化器
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 一个小训练循环
for epoch in range(5):  # 跑5个epoch
    # 假数据：batch_size=1，1通道，32x32 图像
    inputs = torch.randn(1, 1, 32, 32)
    targets = torch.randn(1, 10)  # 随机目标，形状 (1,10)

    # ① 梯度清零
    optimizer.zero_grad()

    # ② 前向传播
    outputs = net(inputs)

    # ③ 计算损失
    loss = criterion(outputs, targets)

    # ④ 反向传播
    loss.backward()

    # ⑤ 更新参数
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
