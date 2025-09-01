import torch
x = torch.empty(5, 3)  # uninitialized data
x = torch.rand(5, 3)  # random data
x = torch.zeros(5, 3, dtype=torch.long)
x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
# new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)
# override dtype!
print(x)
# result has the same size
print(x.size())

y = torch.rand(5, 3)  # 均匀分布
print(torch.add(x, y))

x = torch.randn(4, 4)  # 正态随机分布
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
