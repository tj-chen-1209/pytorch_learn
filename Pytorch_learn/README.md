# PyTorch 学习笔记

> 记录自己学习 PyTorch 的过程、代码示例和心得体会。

---

## 📌 学习计划

- [ ] 安装与环境配置
- [ ] Tensor 基础
- [ ] Autograd 自动求导
- [ ] 神经网络构建
- [ ] 模型训练与优化
- [ ] 数据加载与预处理
- [ ] 项目实战（MNIST / CIFAR-10）

---

## 🛠 环境配置

```bash
# 去官网下载并安装Anaconda (版本自己去官网下载改动)
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# 安装
bash Anaconda3-2024.02-1-Linux-x86_64.sh

# 刷新环境变量
source ~/.zshrc

# 用 conda 在 base 环境里安装 mamba
conda install -n base -c conda-forge mamba -y

# 创建虚拟环境（mamba 推荐 提速）
mamba create -n test python=3.10 -y

# 激活环境
conda activate test

# 安装 pytorch (根据官网选择 CUDA/CPU 版本)
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

## 📚 相关函数

- Tensor 创建方式

```python
# unintialized data
x = torch.empty(5, 3)

# random data 均匀分布 [0, 1】
x = torch.rand(5, 3) 

# random data 正态随机分布
x = torch.randn(5, 3)

x = torch.zeros(5, 3, dtype=torch.long)

x = torch.tensor([5.5, 3])

# new_* methods take in sizes
x = x.new_ones(5, 3, dtype=torch.double)
# override dtype!
x = torch.randn_like(x, dtype=torch.float)
```

- 张量改变维度

```python
# torch.view( , )
x = torch.randn(4, 4)
y = x.view(16) # 16 * 1
z = x.view(-1, 8) # 2 * 8
```

- 反向传播

```python
# required_grad 是告诉torch要记录x的节点图
x = torch.ones(2, 2, requires_grad=True)

# grad_fn 是某个前向算子对应的反向规则 通俗来讲就是他是用什么计算得到的就会输出什么规则
# 基本算子：
# AddBackward0：加法
# SubBackward0：减法
# MulBackward0：乘法
# DivBackward0：除法
# PowBackward0：幂运算
# 为什么都是 ...Backward0?  
#PyTorch 在 新版 Autograd 引擎（基于 C++ DifferentiableGraph） 里生成的默认后缀。
y = x + 2
print(y.grad_fn)

# 对于最后生成的是标量：
z = y * 3 * 3
out = z.mean()
out.backward()
# 就可以随意输出在这条传播路线上的输出out对任何一个变量的梯度
x.grad # (dout)/dx

# 对于生产是多维向量：
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x * 2 * 2
v = torch.tensor([1.0, 0.5]) 
y.backward(v) # 相当于定义了一个新的loss = v_T * y
x.grad # (dloss)/dx

# stop autograd 切断后面所有的节点图 下面这个k就没有backward
x = torch.tensor([2., 2.], requires_grad=True)
with torch.no_grad():
    y = x * 2
    # y.sum().backward() 这个就会报错 因为required_grad=False

x.sum().backward()
x.grad
```

- Jacobian Matrix:

相当于多维度上的斜率矩阵，可以近似取函数$\delta x$的值，计算方法：

$$
y = f(x) = \begin{bmatrix}
x_1^2 \\
x_2^2
\end{bmatrix}
$$

它的 Jacobian 是：

$$
J = \frac{\partial y}{\partial x} =\begin{bmatrix}
2x_1 & 0 \\
0 & 2x_2
\end{bmatrix}
$$

调用 `y.backward(v)` 时，PyTorch 计算的是：

$$
\nabla_x (v^\top y) = v^\top J
$$

也就是：

$$
\frac{\partial (v_1 y_1 + v_2 y_2)}{\partial x}
= (2v_1 x_1, \; 2v_2 x_2)
$$

- 最小神经网络搭建：

```python

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
```
