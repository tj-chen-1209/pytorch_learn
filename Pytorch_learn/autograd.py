import torch

# x = torch.ones(2, 2, requires_grad=True)

# y = x + 2
# print(y.grad_fn)  # 这个就是为了区分节点的反向规则

# z = y * y * 3
# out = z.mean()

# out.backward()

# print(x.grad)

# x = torch.randn(3, requires_grad=True)

# y = x * 2
# while y.data.norm() < 1000:
#     y = y * 2

# print(y)

# v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
# y.backward(v)

# print(x.grad)
# print(y)

# print(x.requires_grad)
# print((x ** 2).requires_grad)

# stop autograd 切断后面所有的节点图 下面这个k就没有backward
x = torch.tensor([2., 2.], requires_grad=True)
with torch.no_grad():
    y = x * 2
    # y.sum().backward()

x.sum().backward()
x.grad
